#include "hybrid_localization/semantic_pose_graph.h"
#include "hybrid_localization/i2nav_ranger_odometer.h"
#include "hybrid_localization/visual_rotation_tracker.h"
#include "hybrid_localization/visual_loop_detector.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

using hybrid_localization::DebugPairStage;
using hybrid_localization::GraphDebugPair;
using hybrid_localization::GraphDebugPairVector;
using hybrid_localization::SemanticGraphPoint;
using hybrid_localization::SemanticGraphPointVector;
using hybrid_localization::SemanticLoopDebug;
using hybrid_localization::SemanticPoseGraph;
using hybrid_localization::SemanticPoseGraphOptions;
using hybrid_localization::SemanticPoseGraphStats;
using hybrid_localization::VisualRotationEstimate;
using hybrid_localization::VisualRotationTracker;
using hybrid_localization::VisualRotationTrackerOptions;
using hybrid_localization::VisualLidarPointVector;
using hybrid_localization::VisualLoopDetector;
using hybrid_localization::VisualLoopDetectorOptions;
using hybrid_localization::VisualLoopResult;

struct OdomSample
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stamp = 0.0;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  nav_msgs::Odometry message;
};

using OdomDeque = std::deque<OdomSample, Eigen::aligned_allocator<OdomSample>>;

struct VisualImageSample
{
  double stamp = 0.0;
  cv::Mat image;
};

struct VisualCloudSample
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stamp = 0.0;
  VisualLidarPointVector points;
};

// Image/depth loop verification runs in the registered-cloud callback.  The
// matching graph keyframe can lag that callback by one or more frames, so keep
// only the small, already-verified proposal until both graph keys exist.
struct PendingVisualLoopConstraint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double reference_stamp = 0.0;
  double current_stamp = 0.0;
  Eigen::Isometry3d reference_from_current = Eigen::Isometry3d::Identity();
  double quality = 0.0;
};

using PendingVisualLoopDeque =
    std::deque<PendingVisualLoopConstraint,
               Eigen::aligned_allocator<PendingVisualLoopConstraint>>;

// Semantic point clouds can be delivered before their frontend odometry on a
// separate ROS connection. Keep the source timestamp with the message so the
// graph can associate it deterministically once odometry catches up.
struct PendingSemanticCloud
{
  double stamp = 0.0;
  sensor_msgs::PointCloud2ConstPtr message;
};

Eigen::Isometry3d poseFromMessage(const geometry_msgs::Pose &message)
{
  Eigen::Quaterniond quaternion(message.orientation.w, message.orientation.x,
                                message.orientation.y, message.orientation.z);
  if (!std::isfinite(quaternion.norm()) || quaternion.norm() < 1e-9)
  {
    quaternion = Eigen::Quaterniond::Identity();
  }
  quaternion.normalize();
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = quaternion.toRotationMatrix();
  pose.translation() = Eigen::Vector3d(message.position.x, message.position.y,
                                       message.position.z);
  return pose;
}

geometry_msgs::Pose poseMessage(const Eigen::Isometry3d &pose)
{
  geometry_msgs::Pose message;
  const Eigen::Quaterniond quaternion(pose.rotation());
  message.position.x = pose.translation().x();
  message.position.y = pose.translation().y();
  message.position.z = pose.translation().z();
  message.orientation.x = quaternion.x();
  message.orientation.y = quaternion.y();
  message.orientation.z = quaternion.z();
  message.orientation.w = quaternion.w();
  return message;
}

double rotationDegrees(const Eigen::Matrix3d &rotation)
{
  const double cosine = std::max(-1.0, std::min(1.0,
      0.5 * (rotation.trace() - 1.0)));
  return std::acos(cosine) * 180.0 / std::acos(-1.0);
}

Eigen::Isometry3d projectToSE3(const Eigen::Isometry3d &pose)
{
  Eigen::Quaterniond quaternion(pose.rotation());
  if (!std::isfinite(quaternion.norm()) || quaternion.norm() < 1e-9)
  {
    quaternion = Eigen::Quaterniond::Identity();
  }
  quaternion.normalize();
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = quaternion.toRotationMatrix();
  result.translation() = pose.translation();
  return result;
}

Eigen::Isometry3d rateLimitedCorrection(const Eigen::Isometry3d &current,
                                        const Eigen::Isometry3d &target,
                                        double maximum_xy_step,
                                        double maximum_z_step,
                                        double maximum_rotation_step_rad)
{
  Eigen::Isometry3d result = projectToSE3(current);
  const Eigen::Vector2d xy_delta =
      target.translation().head<2>() - current.translation().head<2>();
  const double xy_distance = xy_delta.norm();
  if (maximum_xy_step > 0.0 && xy_distance > maximum_xy_step && xy_distance > 1e-12)
  {
    result.translation().head<2>() = current.translation().head<2>() +
        xy_delta * (maximum_xy_step / xy_distance);
  }
  else
  {
    result.translation().head<2>() = target.translation().head<2>();
  }
  const double z_delta = target.translation().z() - current.translation().z();
  if (maximum_z_step > 0.0 && std::abs(z_delta) > maximum_z_step)
  {
    result.translation().z() = current.translation().z() +
        (z_delta > 0.0 ? maximum_z_step : -maximum_z_step);
  }
  else
  {
    result.translation().z() = target.translation().z();
  }
  Eigen::Quaterniond current_q(current.rotation());
  Eigen::Quaterniond target_q(target.rotation());
  if (current_q.dot(target_q) < 0.0) target_q.coeffs() *= -1.0;
  const Eigen::Matrix3d rotation_delta = current.rotation().transpose() * target.rotation();
  const double angle = std::acos(std::max(-1.0, std::min(1.0,
      0.5 * (rotation_delta.trace() - 1.0))));
  const double ratio = maximum_rotation_step_rad > 0.0 && angle > maximum_rotation_step_rad &&
      angle > 1e-12 ? maximum_rotation_step_rad / angle : 1.0;
  result.linear() = current_q.slerp(ratio, target_q).normalized().toRotationMatrix();
  return projectToSE3(result);
}

sensor_msgs::Image imageMessage(const cv::Mat &image, double stamp,
                                const std::string &frame_id)
{
  sensor_msgs::Image message;
  message.header.stamp = std::isfinite(stamp) ? ros::Time(stamp) : ros::Time::now();
  message.header.frame_id = frame_id;
  if (image.empty()) return message;
  cv::Mat bgr;
  if (image.type() == CV_8UC3)
  {
    bgr = image;
  }
  else if (image.type() == CV_8UC1)
  {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  }
  else
  {
    image.convertTo(bgr, CV_8U);
    if (bgr.channels() == 1) cv::cvtColor(bgr, bgr, cv::COLOR_GRAY2BGR);
  }
  if (!bgr.isContinuous()) bgr = bgr.clone();
  message.height = static_cast<uint32_t>(bgr.rows);
  message.width = static_cast<uint32_t>(bgr.cols);
  message.encoding = "bgr8";
  message.is_bigendian = false;
  message.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
  const std::size_t bytes = bgr.total() * bgr.elemSize();
  message.data.assign(bgr.data, bgr.data + bytes);
  return message;
}

const sensor_msgs::PointField *findField(const sensor_msgs::PointCloud2 &cloud,
                                         const std::string &name)
{
  for (const sensor_msgs::PointField &field : cloud.fields)
  {
    if (field.name == name) return &field;
  }
  return nullptr;
}

double readNumeric(const uint8_t *data, const sensor_msgs::PointField &field)
{
  switch (field.datatype)
  {
    case sensor_msgs::PointField::INT8:
    {
      int8_t value = 0;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
    case sensor_msgs::PointField::UINT8:
    {
      uint8_t value = 0;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
    case sensor_msgs::PointField::INT16:
    {
      int16_t value = 0;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
    case sensor_msgs::PointField::UINT16:
    {
      uint16_t value = 0;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
    case sensor_msgs::PointField::INT32:
    {
      int32_t value = 0;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
    case sensor_msgs::PointField::UINT32:
    {
      uint32_t value = 0;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
    case sensor_msgs::PointField::FLOAT64:
    {
      double value = 0.0;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
    case sensor_msgs::PointField::FLOAT32:
    default:
    {
      float value = 0.0F;
      std::memcpy(&value, data + field.offset, sizeof(value));
      return value;
    }
  }
}

SemanticGraphPointVector decodeCloud(const sensor_msgs::PointCloud2 &cloud,
                                     int maximum_points, bool read_semantics,
                                     const std::string &label_field,
                                     const std::string &confidence_field)
{
  SemanticGraphPointVector points;
  const sensor_msgs::PointField *x = findField(cloud, "x");
  const sensor_msgs::PointField *y = findField(cloud, "y");
  const sensor_msgs::PointField *z = findField(cloud, "z");
  const sensor_msgs::PointField *label = read_semantics ? findField(cloud, label_field) : nullptr;
  const sensor_msgs::PointField *confidence = read_semantics ?
      findField(cloud, confidence_field) : nullptr;
  if (!x || !y || !z || cloud.point_step == 0U) return points;
  const std::size_t total = static_cast<std::size_t>(cloud.width) * cloud.height;
  const std::size_t stride = std::max<std::size_t>(
      1U, total / static_cast<std::size_t>(std::max(1, maximum_points)));
  points.reserve(std::min(total, static_cast<std::size_t>(std::max(1, maximum_points))));
  for (std::size_t index = 0; index < total; index += stride)
  {
    const std::size_t offset = index * cloud.point_step;
    if (offset + cloud.point_step > cloud.data.size()) break;
    const uint8_t *data = cloud.data.data() + offset;
    SemanticGraphPoint point;
    point.point = Eigen::Vector3d(readNumeric(data, *x), readNumeric(data, *y),
                                  readNumeric(data, *z));
    if (!point.point.allFinite()) continue;
    if (label)
    {
      point.label = static_cast<uint8_t>(std::max(0.0, std::min(255.0,
          readNumeric(data, *label))));
    }
    if (confidence)
    {
      point.confidence = static_cast<float>(std::max(0.0, std::min(1.0,
          readNumeric(data, *confidence))));
    }
    points.push_back(point);
    if (static_cast<int>(points.size()) >= maximum_points) break;
  }
  return points;
}

std_msgs::ColorRGBA stageColor(DebugPairStage stage)
{
  std_msgs::ColorRGBA color;
  color.a = 0.95F;
  switch (stage)
  {
    case DebugPairStage::Candidate:
      color.r = 1.0F; color.g = 0.55F; color.b = 0.05F; break;
    case DebugPairStage::Inlier:
      color.r = 0.05F; color.g = 0.90F; color.b = 0.95F; break;
    case DebugPairStage::Outlier:
      color.r = 0.55F; color.g = 0.55F; color.b = 0.55F; break;
    case DebugPairStage::Applied:
      color.r = 0.10F; color.g = 1.0F; color.b = 0.20F; break;
  }
  return color;
}

std::string stageName(DebugPairStage stage)
{
  switch (stage)
  {
    case DebugPairStage::Candidate: return "candidate";
    case DebugPairStage::Inlier: return "inlier";
    case DebugPairStage::Outlier: return "outlier";
    case DebugPairStage::Applied: return "applied";
  }
  return "unknown";
}

std::uint32_t semanticRgb(std::uint8_t label)
{
  switch (label)
  {
    case 1U: return (70U << 16U) | (190U << 8U) | 90U;
    case 2U: return (40U << 16U) | (200U << 8U) | 220U;
    case 3U: return (230U << 16U) | (75U << 8U) | 65U;
    case 4U: return (40U << 16U) | (150U << 8U) | 45U;
    case 5U: return (235U << 16U) | (65U << 8U) | 210U;
    case 6U: return (245U << 16U) | (190U << 8U) | 45U;
    default: return (150U << 16U) | (150U << 8U) | 150U;
  }
}

class SemanticGtsamPoseGraphNode
{
public:
  SemanticGtsamPoseGraphNode() : nh_(), private_nh_("~")
  {
    loadParameters();
    graph_.reset(new SemanticPoseGraph(options_));
    visual_tracker_.reset(new VisualRotationTracker(visual_options_));
    visual_loop_detector_.reset(new VisualLoopDetector(visual_loop_options_));
    odom_sub_ = nh_.subscribe(frontend_odom_topic_, 100,
                              &SemanticGtsamPoseGraphNode::odomCallback, this);
    if (process_registered_clouds_ && !registered_cloud_topic_.empty())
    {
      cloud_sub_ = nh_.subscribe(registered_cloud_topic_, 100,
                                 &SemanticGtsamPoseGraphNode::cloudCallback, this);
    }
    else
    {
      ROS_INFO("[semantic_gtsam] registered-cloud processing disabled; "
               "publishing frontend odometry without cloud keyframes");
    }
    if (subscribe_semantic_ && !semantic_cloud_topic_.empty())
    {
      semantic_sub_ = nh_.subscribe(semantic_cloud_topic_, 2,
                                    &SemanticGtsamPoseGraphNode::semanticCallback, this);
    }
    if (options_.enable_wheel_factors && !wheel_topic_.empty())
    {
      wheel_sub_ = nh_.subscribe<insprobe_msgs::RangerOdometer>(
          wheel_topic_, 200, &SemanticGtsamPoseGraphNode::wheelCallback, this);
    }
    if (subscribe_camera_ && !camera_topic_.empty())
    {
      camera_sub_ = nh_.subscribe<sensor_msgs::CompressedImage>(
          camera_topic_, 4, &SemanticGtsamPoseGraphNode::cameraCallback, this);
    }
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>(output_odom_topic_, 50);
    path_pub_ = nh_.advertise<nav_msgs::Path>(output_path_topic_, 2, true);
    corrected_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        corrected_cloud_topic_, 2);
    stats_pub_ = nh_.advertise<std_msgs::String>(stats_topic_, 2, true);
    xy_debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(xy_debug_topic_, 2, true);
    z_debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(z_debug_topic_, 2, true);
    semantic_xy_debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        semantic_xy_debug_topic_, 2, true);
    semantic_z_debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        semantic_z_debug_topic_, 2, true);
    semantic_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        semantic_map_topic_, 1, true);
    visual_projection_debug_pub_ = nh_.advertise<sensor_msgs::Image>(
        visual_projection_debug_topic_, 1);
    visual_tracking_debug_pub_ = nh_.advertise<sensor_msgs::Image>(
        visual_tracking_debug_topic_, 1);
    visual_pnp_debug_pub_ = nh_.advertise<sensor_msgs::Image>(
        visual_pnp_debug_topic_, 1);
    visual_time_difference_pub_ = nh_.advertise<std_msgs::Float64>(
        visual_time_difference_topic_, 10);
    visual_loop_debug_pub_ = nh_.advertise<sensor_msgs::Image>(
        visual_loop_debug_topic_, 1);
    publishDebug();
    timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.1, path_publish_rate_)),
                             &SemanticGtsamPoseGraphNode::timerCallback, this);
    ROS_INFO("[semantic_gtsam] ready: frontend_odom=%s cloud=%s semantic=%s camera=%s output=%s",
             frontend_odom_topic_.c_str(), registered_cloud_topic_.c_str(),
             subscribe_semantic_ ? semantic_cloud_topic_.c_str() : "disabled",
             subscribe_camera_ ? camera_topic_.c_str() : "disabled",
             output_odom_topic_.c_str());
    if (subscribe_camera_ && visual_observation_only_)
    {
      ROS_INFO("[semantic_gtsam] visual observation-only mode: graph pose factors disabled");
    }
  }

  ~SemanticGtsamPoseGraphNode()
  {
    saveTrajectory("shutdown");
  }

  void spin()
  {
    ros::spin();
  }

private:
  void loadParameters()
  {
    private_nh_.param<std::string>("frontend_odom_topic", frontend_odom_topic_,
                                   "/hybrid/frontend/odometry");
    private_nh_.param<std::string>("registered_cloud_topic", registered_cloud_topic_,
                                   "/hybrid/cloud_registered");
    private_nh_.param<std::string>("semantic_cloud_topic", semantic_cloud_topic_,
                                   "/hybrid/semantic_cloud");
    private_nh_.param<std::string>("wheel_topic", wheel_topic_,
                                   "/insprobe/ranger/odometer");
    private_nh_.param("wheel_time_offset", wheel_time_offset_, 0.0);
    private_nh_.param<std::string>("camera_topic", camera_topic_,
                                   "/avt_camera/left/image/compressed");
    private_nh_.param<std::string>("output_odom_topic", output_odom_topic_,
                                   "/hybrid/odometry");
    private_nh_.param<std::string>("output_path_topic", output_path_topic_,
                                   "/hybrid/path");
    private_nh_.param<std::string>("corrected_cloud_topic", corrected_cloud_topic_,
                                   "/hybrid/cloud_registered_corrected");
    private_nh_.param<std::string>("stats_topic", stats_topic_, "/hybrid/semantic_graph/stats");
    private_nh_.param<std::string>("xy_debug_topic", xy_debug_topic_,
                                   "/semantic_slam/xy_constraint_debug");
    private_nh_.param<std::string>("z_debug_topic", z_debug_topic_,
                                   "/semantic_slam/z_constraint_debug");
    private_nh_.param<std::string>("semantic_xy_debug_topic", semantic_xy_debug_topic_,
                                   "/semantic_slam/semantic_xy_observation_debug");
    private_nh_.param<std::string>("semantic_z_debug_topic", semantic_z_debug_topic_,
                                   "/semantic_slam/semantic_z_observation_debug");
    private_nh_.param<std::string>("semantic_map_topic", semantic_map_topic_,
                                   "/hybrid/semantic_graph/map");
    private_nh_.param<std::string>("visual_projection_debug_topic",
                                   visual_projection_debug_topic_,
                                   "/hybrid/visual/debug/projection");
    private_nh_.param<std::string>("visual_tracking_debug_topic",
                                   visual_tracking_debug_topic_,
                                   "/hybrid/visual/debug/tracks");
    private_nh_.param<std::string>("visual_pnp_debug_topic",
                                   visual_pnp_debug_topic_,
                                   "/hybrid/visual/debug/pnp_inliers");
    private_nh_.param<std::string>("visual_time_difference_topic",
                                   visual_time_difference_topic_,
                                   "/hybrid/visual/debug/image_cloud_dt");
    private_nh_.param<std::string>("visual_loop_debug_topic",
                                   visual_loop_debug_topic_,
                                   "/hybrid/visual/debug/loop");
    private_nh_.param<std::string>("map_frame", map_frame_, "map");
    private_nh_.param<std::string>("body_frame", body_frame_, "base_link");
    private_nh_.param<std::string>("label_field", label_field_, "label");
    private_nh_.param<std::string>("confidence_field", confidence_field_, "confidence");
    private_nh_.param<std::string>("trajectory_save_path", trajectory_save_path_, "");

    private_nh_.param("subscribe_semantic", subscribe_semantic_, false);
    private_nh_.param("subscribe_camera", subscribe_camera_, false);
    private_nh_.param("process_registered_clouds", process_registered_clouds_, true);
    private_nh_.param("visual_observation_only", visual_observation_only_, true);
    private_nh_.param("publish_visual_debug_images",
                      publish_visual_debug_images_, true);
    private_nh_.param("semantic_cloud_in_map_frame", semantic_cloud_in_map_frame_, true);
    private_nh_.param("broadcast_tf", broadcast_tf_, true);
    private_nh_.param("save_on_shutdown", save_on_shutdown_, true);
    private_nh_.param("max_pose_lookup_dt", max_pose_lookup_dt_, 0.15);
    private_nh_.param("odom_history_sec", odom_history_sec_, 120.0);
    private_nh_.param("max_registered_points", max_registered_points_, 8000);
    private_nh_.param("max_pending_registered_clouds",
                      max_pending_registered_clouds_, 100);
    private_nh_.param("max_pending_semantic_clouds",
                      max_pending_semantic_clouds_, 240);
    private_nh_.param("max_semantic_process_per_tick",
                      max_semantic_process_per_tick_, 2);
    private_nh_.param("max_semantic_points", max_semantic_points_, 6000);
    private_nh_.param("max_debug_pairs", max_debug_pairs_, 500);
    private_nh_.param("path_publish_rate", path_publish_rate_, 0.5);
    private_nh_.param("trajectory_checkpoint_keyframes", trajectory_checkpoint_keyframes_, 25);
    private_nh_.param("camera_time_offset", camera_time_offset_, 0.0);
    private_nh_.param("visual_sync_tolerance", visual_sync_tolerance_, 0.06);
    private_nh_.param("semantic_map_publish_rate", semantic_map_publish_rate_, 0.20);
    private_nh_.param("semantic_map_voxel_size", semantic_map_voxel_size_, 0.30);
    private_nh_.param("semantic_map_max_points", semantic_map_max_points_, 120000);
    // GTSAM remains the source of truth.  This optional output layer only
    // rate-limits map-from-odom changes after an accepted global factor so a
    // live controller or mapper does not receive a metre-scale jump in one
    // odometry callback.
    private_nh_.param("continuous_correction_enabled", continuous_correction_enabled_, false);
    private_nh_.param("continuous_correction_max_xy_rate",
                      continuous_correction_max_xy_rate_, 0.0);
    private_nh_.param("continuous_correction_max_z_rate",
                      continuous_correction_max_z_rate_, 0.0);
    private_nh_.param("continuous_correction_max_rotation_rate_deg",
                      continuous_correction_max_rotation_rate_deg_, 0.0);
    private_nh_.param("continuous_correction_reset_gap_sec",
                      continuous_correction_reset_gap_sec_, 5.0);

    max_pending_semantic_clouds_ = std::max(1, max_pending_semantic_clouds_);
    max_semantic_process_per_tick_ = std::max(1, max_semantic_process_per_tick_);

    private_nh_.param("graph/enabled", options_.enabled, true);
    private_nh_.param("graph/enable_xy_loops", options_.enable_xy_loops, false);
    private_nh_.param("graph/enable_z_loops", options_.enable_z_loops, false);
    private_nh_.param("graph/enable_sequential_ground_z", options_.enable_sequential_ground_z, false);
    private_nh_.param("graph/enable_wheel_factors", options_.enable_wheel_factors, false);
    private_nh_.param("graph/enable_visual_rotation_factors",
                      options_.enable_visual_rotation_factors, false);
    private_nh_.param("graph/enable_visual_translation_factors",
                      options_.enable_visual_translation_factors, false);
    private_nh_.param("graph/enable_visual_loop_factors",
                      options_.enable_visual_loop_factors, false);
    private_nh_.param("graph/enable_semantic_observation_factors",
                      options_.enable_semantic_observation_factors, false);
    private_nh_.param("graph/enable_semantic_observation_xy_factors",
                      options_.enable_semantic_observation_xy_factors, false);
    private_nh_.param("graph/enable_semantic_observation_z_factors",
                      options_.enable_semantic_observation_z_factors, false);
    private_nh_.param("graph/semantic_observation_require_xy_for_z",
                      options_.semantic_observation_require_xy_for_z, true);
    private_nh_.param("graph/use_semantics", options_.use_semantics, true);
    private_nh_.param("graph/keyframe_distance", options_.keyframe_distance, 1.0);
    private_nh_.param("graph/keyframe_yaw_deg", options_.keyframe_yaw_deg, 8.0);
    private_nh_.param("graph/keyframe_interval_sec", options_.keyframe_interval_sec, 1.0);
    private_nh_.param("graph/max_keyframes", options_.max_keyframes, 2500);
    private_nh_.param("graph/submap_frames", options_.submap_frames, 8);
    private_nh_.param("graph/max_points_per_frame", options_.max_points_per_frame, 7000);
    private_nh_.param("graph/max_features_per_keyframe", options_.max_features_per_keyframe, 3500);
    private_nh_.param("graph/feature_resolution", options_.feature_resolution, 0.60);
    private_nh_.param("graph/feature_min_points", options_.feature_min_points, 3);
    private_nh_.param("graph/structural_min_height_span", options_.structural_min_height_span, 0.55);
    private_nh_.param("graph/structural_min_z", options_.structural_min_z, -0.35);
    private_nh_.param("graph/ground_max_height", options_.ground_max_height, -0.20);
    private_nh_.param("graph/ground_max_height_span", options_.ground_max_height_span, 0.30);
    private_nh_.param("graph/ground_min_points", options_.ground_min_points, 4);
    private_nh_.param("graph/descriptor_rings", options_.descriptor_rings, 20);
    private_nh_.param("graph/descriptor_sectors", options_.descriptor_sectors, 60);
    private_nh_.param("graph/descriptor_max_radius", options_.descriptor_max_radius, 55.0);
    private_nh_.param("graph/descriptor_min_similarity", options_.descriptor_min_similarity, 0.22);
    private_nh_.param("graph/descriptor_min_score_gap", options_.descriptor_min_score_gap, 0.015);
    private_nh_.param("graph/semantic_weight", options_.semantic_weight, 0.35);
    private_nh_.param("graph/semantic_submap_observations",
                      options_.semantic_submap_observations, 3);
    private_nh_.param("graph/semantic_observation_min_index_gap",
                      options_.semantic_observation_min_index_gap, 4);
    private_nh_.param("graph/semantic_observation_max_index_gap",
                      options_.semantic_observation_max_index_gap, 0);
    private_nh_.param("graph/semantic_observation_min_time_separation_sec",
                      options_.semantic_observation_min_time_separation_sec, 90.0);
    private_nh_.param("graph/semantic_observation_max_reference_uses",
                      options_.semantic_observation_max_reference_uses, 0);
    private_nh_.param("graph/semantic_observation_interval",
                      options_.semantic_observation_interval, 1);
    private_nh_.param("graph/semantic_observation_minimum_interval_sec",
                      options_.semantic_observation_minimum_interval_sec, 180.0);
    private_nh_.param("graph/semantic_observation_max_factors",
                      options_.semantic_observation_max_factors, 2);
    private_nh_.param("graph/semantic_observation_min_features",
                      options_.semantic_observation_min_features, 80);
    private_nh_.param("graph/semantic_observation_min_inliers",
                      options_.semantic_observation_min_inliers, 45);
    private_nh_.param("graph/semantic_observation_min_z_inliers",
                      options_.semantic_observation_min_z_inliers, 30);
    private_nh_.param("graph/min_semantic_observation_factors_for_xy_output",
                      options_.min_semantic_observation_factors_for_xy_output, 2);
    private_nh_.param("graph/semantic_observation_search_radius",
                      options_.semantic_observation_search_radius, 35.0);
    private_nh_.param("graph/semantic_observation_min_baseline",
                      options_.semantic_observation_min_baseline, 0.0);
    private_nh_.param("graph/semantic_observation_max_time_offset",
                      options_.semantic_observation_max_time_offset, 0.75);
    private_nh_.param("graph/semantic_observation_correspondence_distance",
                      options_.semantic_observation_correspondence_distance, 0.85);
    private_nh_.param("graph/semantic_observation_ransac_inlier_distance",
                      options_.semantic_observation_ransac_inlier_distance, 0.38);
    private_nh_.param("graph/semantic_observation_min_inlier_ratio",
                      options_.semantic_observation_min_inlier_ratio, 0.35);
    private_nh_.param("graph/semantic_observation_min_spread",
                      options_.semantic_observation_min_spread, 5.0);
    private_nh_.param("graph/semantic_observation_min_spread_ratio",
                      options_.semantic_observation_min_spread_ratio, 0.08);
    private_nh_.param("graph/semantic_observation_max_rmse",
                      options_.semantic_observation_max_rmse, 0.32);
    private_nh_.param("graph/semantic_observation_max_xy_correction",
                      options_.semantic_observation_max_xy_correction, 0.50);
    private_nh_.param("graph/semantic_observation_max_yaw_correction_deg",
                      options_.semantic_observation_max_yaw_correction_deg, 1.5);
    private_nh_.param("graph/semantic_observation_max_z_correction",
                      options_.semantic_observation_max_z_correction, 0.35);
    private_nh_.param("graph/semantic_observation_sigma_xy",
                      options_.semantic_observation_sigma_xy, 0.30);
    private_nh_.param("graph/semantic_observation_sigma_yaw_deg",
                      options_.semantic_observation_sigma_yaw_deg, 1.00);
    private_nh_.param("graph/semantic_observation_sigma_z",
                      options_.semantic_observation_sigma_z, 0.25);
    private_nh_.param("graph/semantic_observation_huber_k",
                      options_.semantic_observation_huber_k, 1.345);
    private_nh_.param("graph/loop_min_index_gap", options_.loop_min_index_gap, 30);
    private_nh_.param("graph/loop_min_time_separation_sec",
                      options_.loop_min_time_separation_sec, 90.0);
    private_nh_.param("graph/loop_max_candidates", options_.loop_max_candidates, 6);
    private_nh_.param("graph/loop_min_support", options_.loop_min_support, 3);
    private_nh_.param("graph/loop_support_reference_neighborhood",
                      options_.loop_support_reference_neighborhood, 12);
    private_nh_.param("graph/loop_support_current_max_gap",
                      options_.loop_support_current_max_gap, 5);
    private_nh_.param("graph/loop_minimum_interval_sec",
                      options_.loop_minimum_interval_sec, 120.0);
    private_nh_.param("graph/loop_max_factors", options_.loop_max_factors, 4);
    private_nh_.param("graph/loop_require_xy_for_z", options_.loop_require_xy_for_z, true);
    private_nh_.param("graph/loop_search_radius", options_.loop_search_radius, 18.0);
    private_nh_.param("graph/loop_max_yaw_difference_deg", options_.loop_max_yaw_difference_deg, 35.0);
    private_nh_.param("graph/coarse_xy_radius", options_.coarse_xy_radius, 5.0);
    private_nh_.param("graph/coarse_xy_step", options_.coarse_xy_step, 0.75);
    private_nh_.param("graph/coarse_yaw_radius_deg", options_.coarse_yaw_radius_deg, 8.0);
    private_nh_.param("graph/coarse_yaw_step_deg", options_.coarse_yaw_step_deg, 2.0);
    private_nh_.param("graph/coarse_max_points", options_.coarse_max_points, 900);
    private_nh_.param("graph/coarse_min_inliers", options_.coarse_min_inliers, 70);
    private_nh_.param("graph/correspondence_distance", options_.correspondence_distance, 1.20);
    private_nh_.param("graph/ransac_inlier_distance", options_.ransac_inlier_distance, 0.55);
    private_nh_.param("graph/ransac_iterations", options_.ransac_iterations, 120);
    private_nh_.param("graph/min_xy_inliers", options_.min_xy_inliers, 90);
    private_nh_.param("graph/min_xy_inlier_ratio", options_.min_xy_inlier_ratio, 0.30);
    private_nh_.param("graph/min_xy_spread", options_.min_xy_spread, 7.0);
    private_nh_.param("graph/min_xy_spread_ratio", options_.min_xy_spread_ratio, 0.12);
    private_nh_.param("graph/max_xy_rmse", options_.max_xy_rmse, 0.42);
    private_nh_.param("graph/huber_delta", options_.huber_delta, 0.35);
    private_nh_.param("graph/huber_iterations", options_.huber_iterations, 5);
    private_nh_.param("graph/max_xy_correction", options_.max_xy_correction, 5.0);
    private_nh_.param("graph/max_yaw_correction_deg", options_.max_yaw_correction_deg, 8.0);
    private_nh_.param("graph/z_correspondence_distance", options_.z_correspondence_distance, 0.80);
    private_nh_.param("graph/z_candidate_residual_gate", options_.z_candidate_residual_gate, 1.20);
    private_nh_.param("graph/z_inlier_residual_gate", options_.z_inlier_residual_gate, 0.22);
    private_nh_.param("graph/min_z_inliers", options_.min_z_inliers, 45);
    private_nh_.param("graph/max_z_mad", options_.max_z_mad, 0.12);
    private_nh_.param("graph/max_z_correction", options_.max_z_correction, 1.20);
    private_nh_.param("graph/graph_consistency_max_xy", options_.graph_consistency_max_xy, 5.0);
    private_nh_.param("graph/graph_consistency_max_yaw_deg", options_.graph_consistency_max_yaw_deg, 8.0);
    private_nh_.param("graph/graph_consistency_max_z", options_.graph_consistency_max_z, 1.5);
    private_nh_.param("graph/min_loops_for_xy_output", options_.min_loops_for_xy_output, 2);
    private_nh_.param("graph/odom_sigma_roll_pitch", options_.odom_sigma_roll_pitch, 0.010);
    private_nh_.param("graph/odom_sigma_yaw", options_.odom_sigma_yaw, 0.008);
    private_nh_.param("graph/odom_sigma_xy_base", options_.odom_sigma_xy_base, 0.035);
    private_nh_.param("graph/odom_sigma_xy_per_meter", options_.odom_sigma_xy_per_meter, 0.010);
    private_nh_.param("graph/odom_sigma_z_base", options_.odom_sigma_z_base, 0.050);
    private_nh_.param("graph/loop_sigma_xy", options_.loop_sigma_xy, 0.12);
    private_nh_.param("graph/loop_sigma_yaw_deg", options_.loop_sigma_yaw_deg, 0.45);
    private_nh_.param("graph/loop_sigma_z", options_.loop_sigma_z, 0.08);
    private_nh_.param("graph/sequential_ground_sigma_z", options_.sequential_ground_sigma_z, 0.06);
    private_nh_.param("graph/sequential_ground_interval", options_.sequential_ground_interval, 1);
    private_nh_.param("graph/sequential_ground_min_spread",
                      options_.sequential_ground_min_spread, 0.0);
    private_nh_.param("graph/sequential_ground_min_spread_ratio",
                      options_.sequential_ground_min_spread_ratio, 0.0);
    private_nh_.param("graph/sequential_ground_min_support",
                      options_.sequential_ground_min_support, 1);
    private_nh_.param("graph/sequential_ground_support_max_gap",
                      options_.sequential_ground_support_max_gap, 2);
    private_nh_.param("graph/sequential_ground_support_max_z_disagreement",
                      options_.sequential_ground_support_max_z_disagreement, 0.0);
    private_nh_.param("graph/sequential_ground_min_inlier_ratio",
                      options_.sequential_ground_min_inlier_ratio, 0.0);
    private_nh_.param("graph/sequential_ground_max_step",
                      options_.sequential_ground_max_step, 0.0);
    private_nh_.param("graph/loop_huber_k", options_.loop_huber_k, 1.345);
    private_nh_.param("graph/sequential_ground_huber_k", options_.sequential_ground_huber_k, 1.345);
    private_nh_.param("graph/sequential_ground_use_dcs",
                      options_.sequential_ground_use_dcs, false);
    private_nh_.param("graph/sequential_ground_dcs_k",
                      options_.sequential_ground_dcs_k, 1.0);
    private_nh_.param("graph/wheel_speed_scale", options_.wheel_speed_scale, 0.9865);
    private_nh_.param("graph/wheel_max_gap", options_.wheel_max_gap, 0.08);
    private_nh_.param("graph/wheel_sigma_base", options_.wheel_sigma_base, 0.08);
    private_nh_.param("graph/wheel_sigma_per_meter", options_.wheel_sigma_per_meter, 0.025);
    private_nh_.param("graph/wheel_lateral_sigma", options_.wheel_lateral_sigma, 0.15);
    private_nh_.param("graph/wheel_huber_k", options_.wheel_huber_k, 1.345);
    private_nh_.param("graph/wheel_min_samples", options_.wheel_min_samples, 5);
    private_nh_.param("graph/wheel_min_raw_translation",
                      options_.wheel_min_raw_translation, 0.0);
    private_nh_.param("graph/wheel_max_arc_disagreement",
                      options_.wheel_max_arc_disagreement, 0.0);
    private_nh_.param("graph/wheel_max_relative_arc_disagreement",
                      options_.wheel_max_relative_arc_disagreement, 0.0);
    private_nh_.param("graph/wheel_use_dcs", options_.wheel_use_dcs, false);
    private_nh_.param("graph/wheel_dcs_k", options_.wheel_dcs_k, 1.0);
    private_nh_.param("graph/visual_max_time_offset", options_.visual_max_time_offset, 0.15);
    private_nh_.param("graph/visual_min_quality", options_.visual_min_quality, 0.30);
    private_nh_.param("graph/visual_max_angular_disagreement_deg",
                      options_.visual_max_angular_disagreement_deg, 4.0);
    private_nh_.param("graph/visual_sigma_roll_pitch_deg",
                      options_.visual_sigma_roll_pitch_deg, 0.80);
    private_nh_.param("graph/visual_sigma_yaw_deg", options_.visual_sigma_yaw_deg, 0.50);
    private_nh_.param("graph/visual_quality_sigma_scale",
                      options_.visual_quality_sigma_scale, 1.5);
    private_nh_.param("graph/visual_huber_k", options_.visual_huber_k, 1.345);
    private_nh_.param("graph/visual_max_translation_disagreement",
                      options_.visual_max_translation_disagreement, 1.0);
    private_nh_.param("graph/visual_sigma_xy_base", options_.visual_sigma_xy_base, 0.10);
    private_nh_.param("graph/visual_sigma_z_base", options_.visual_sigma_z_base, 0.18);
    private_nh_.param("graph/visual_sigma_translation_per_meter",
                      options_.visual_sigma_translation_per_meter, 0.03);
    private_nh_.param("graph/visual_loop_max_time_offset",
                      options_.visual_loop_max_time_offset, 0.65);
    private_nh_.param("graph/visual_loop_min_index_gap",
                      options_.visual_loop_min_index_gap, 20);
    private_nh_.param("graph/visual_loop_min_time_separation_sec",
                      options_.visual_loop_min_time_separation_sec, 45.0);
    private_nh_.param("graph/visual_loop_min_quality",
                      options_.visual_loop_min_quality, 0.40);
    private_nh_.param("graph/visual_loop_require_lidar_geometry",
                      options_.visual_loop_require_lidar_geometry, true);
    private_nh_.param("graph/visual_loop_min_quality_with_lidar_geometry",
                      options_.visual_loop_min_quality_with_lidar_geometry, 0.55);
    private_nh_.param("graph/visual_loop_lidar_use_pnp_seed",
                      options_.visual_loop_lidar_use_pnp_seed, false);
    private_nh_.param("graph/visual_loop_lidar_max_pnp_xy_disagreement",
                      options_.visual_loop_lidar_max_pnp_xy_disagreement, 1.20);
    private_nh_.param("graph/visual_loop_lidar_max_pnp_yaw_disagreement_deg",
                      options_.visual_loop_lidar_max_pnp_yaw_disagreement_deg, 3.0);
    private_nh_.param("graph/visual_loop_max_translation_disagreement",
                      options_.visual_loop_max_translation_disagreement, 3.0);
    private_nh_.param("graph/visual_loop_max_rotation_disagreement_deg",
                      options_.visual_loop_max_rotation_disagreement_deg, 10.0);
    private_nh_.param("graph/visual_loop_minimum_interval_sec",
                      options_.visual_loop_minimum_interval_sec, 180.0);
    private_nh_.param("graph/visual_loop_reference_neighborhood",
                      options_.visual_loop_reference_neighborhood, 16);
    private_nh_.param("graph/visual_loop_max_factors",
                      options_.visual_loop_max_factors, 1);
    private_nh_.param("graph/visual_loop_sigma_roll_pitch_deg",
                      options_.visual_loop_sigma_roll_pitch_deg, 2.0);
    private_nh_.param("graph/visual_loop_sigma_yaw_deg",
                      options_.visual_loop_sigma_yaw_deg, 1.0);
    private_nh_.param("graph/visual_loop_sigma_xy",
                      options_.visual_loop_sigma_xy, 0.25);
    private_nh_.param("graph/visual_loop_sigma_z",
                      options_.visual_loop_sigma_z, 0.35);
    private_nh_.param("graph/visual_loop_quality_sigma_scale",
                      options_.visual_loop_quality_sigma_scale, 1.5);
    private_nh_.param("graph/visual_loop_huber_k",
                      options_.visual_loop_huber_k, 1.345);
    private_nh_.param("graph/visual_loop_use_dcs",
                      options_.visual_loop_use_dcs, false);
    private_nh_.param("graph/visual_loop_dcs_k",
                      options_.visual_loop_dcs_k, 1.0);
    private_nh_.param("graph/visual_loop_min_support",
                      options_.visual_loop_min_support, 1);
    private_nh_.param("graph/visual_loop_support_reference_neighborhood",
                      options_.visual_loop_support_reference_neighborhood, 16);
    private_nh_.param("graph/visual_loop_support_current_max_gap",
                      options_.visual_loop_support_current_max_gap, 5);
    private_nh_.param("graph/visual_loop_support_max_correction_xy",
                      options_.visual_loop_support_max_correction_xy, 0.0);
    private_nh_.param("graph/visual_loop_support_max_correction_yaw_deg",
                      options_.visual_loop_support_max_correction_yaw_deg, 0.0);
    private_nh_.param("graph/visual_loop_support_max_correction_z",
                      options_.visual_loop_support_max_correction_z, 0.0);
    private_nh_.param("graph/visual_loop_graph_consistency_max_xy",
                      options_.visual_loop_graph_consistency_max_xy, 0.0);
    private_nh_.param("graph/visual_loop_graph_consistency_max_yaw_deg",
                      options_.visual_loop_graph_consistency_max_yaw_deg, 0.0);
    private_nh_.param("graph/visual_loop_graph_consistency_max_z",
                      options_.visual_loop_graph_consistency_max_z, 0.0);
    private_nh_.param("graph/visual_loop_constrain_roll_pitch",
                      options_.visual_loop_constrain_roll_pitch, false);
    private_nh_.param("graph/visual_loop_refine_z_with_ground",
                      options_.visual_loop_refine_z_with_ground, true);
    private_nh_.param("graph/visual_loop_allow_pnp_z_without_ground",
                      options_.visual_loop_allow_pnp_z_without_ground, false);
    private_nh_.param("graph/visual_loop_ground_z_candidate_residual_gate",
                      options_.visual_loop_ground_z_candidate_residual_gate, 1.50);
    private_nh_.param("graph/visual_loop_ground_z_inlier_residual_gate",
                      options_.visual_loop_ground_z_inlier_residual_gate, 0.16);
    private_nh_.param("graph/visual_loop_ground_z_min_inliers",
                      options_.visual_loop_ground_z_min_inliers, 70);
    private_nh_.param("graph/visual_loop_ground_z_max_mad",
                      options_.visual_loop_ground_z_max_mad, 0.08);
    private_nh_.param("graph/visual_loop_ground_z_max_correction",
                      options_.visual_loop_ground_z_max_correction, 1.50);
    private_nh_.param("graph/visual_loop_ground_z_max_step",
                      options_.visual_loop_ground_z_max_step, 0.0);
    private_nh_.param("graph/visual_loop_ground_z_sigma",
                      options_.visual_loop_ground_z_sigma, 0.15);
    private_nh_.param("graph/visual_loop_ground_z_clipped_sigma",
                      options_.visual_loop_ground_z_clipped_sigma, 0.50);
    private_nh_.param("graph/visual_loop_ground_z_sparse_min_inliers",
                      options_.visual_loop_ground_z_sparse_min_inliers, 0);
    private_nh_.param("graph/visual_loop_ground_z_sparse_min_inlier_ratio",
                      options_.visual_loop_ground_z_sparse_min_inlier_ratio, 0.0);
    private_nh_.param("graph/visual_loop_ground_z_sparse_min_spread",
                      options_.visual_loop_ground_z_sparse_min_spread, 0.0);
    private_nh_.param("graph/visual_loop_ground_z_sparse_min_spread_ratio",
                      options_.visual_loop_ground_z_sparse_min_spread_ratio, 0.0);
    private_nh_.param("graph/visual_loop_ground_z_sparse_max_mad",
                      options_.visual_loop_ground_z_sparse_max_mad, 0.0);
    private_nh_.param("graph/visual_loop_ground_z_sparse_min_lidar_xy_inliers",
                      options_.visual_loop_ground_z_sparse_min_lidar_xy_inliers, 0);
    private_nh_.param("graph/visual_loop_ground_z_sparse_max_lidar_xy_rmse",
                      options_.visual_loop_ground_z_sparse_max_lidar_xy_rmse, 0.0);
    private_nh_.param("graph/visual_loop_ground_z_sparse_min_lidar_xy_spread",
                      options_.visual_loop_ground_z_sparse_min_lidar_xy_spread, 0.0);
    private_nh_.param("graph/isam_relinearize_threshold", options_.isam_relinearize_threshold, 0.05);
    private_nh_.param("graph/isam_relinearize_skip", options_.isam_relinearize_skip, 1);

    private_nh_.param("visual/fx", visual_options_.fx, 1064.8950);
    private_nh_.param("visual/fy", visual_options_.fy, 1065.2546);
    private_nh_.param("visual/cx", visual_options_.cx, 801.4049);
    private_nh_.param("visual/cy", visual_options_.cy, 624.6878);
    private_nh_.param("visual/image_scale", visual_options_.image_scale, 0.5);
    private_nh_.param("visual/minimum_interval", visual_options_.minimum_interval, 0.15);
    private_nh_.param("visual/maximum_reference_gap",
                      visual_options_.maximum_reference_gap, 0.60);
    private_nh_.param("visual/maximum_features", visual_options_.maximum_features, 1400);
    private_nh_.param("visual/minimum_tracks", visual_options_.minimum_tracks, 90);
    private_nh_.param("visual/minimum_inliers", visual_options_.minimum_inliers, 65);
    private_nh_.param("visual/feature_quality", visual_options_.feature_quality, 0.01);
    private_nh_.param("visual/feature_minimum_distance",
                      visual_options_.feature_minimum_distance, 10.0);
    private_nh_.param("visual/forward_backward_error",
                      visual_options_.forward_backward_error, 1.2);
    private_nh_.param("visual/minimum_median_parallax",
                      visual_options_.minimum_median_parallax, 1.2);
    private_nh_.param("visual/ransac_probability",
                      visual_options_.ransac_probability, 0.999);
    private_nh_.param("visual/ransac_threshold_pixels",
                      visual_options_.ransac_threshold_pixels, 1.5);
    private_nh_.param("visual/minimum_inlier_ratio",
                      visual_options_.minimum_inlier_ratio, 0.50);
    private_nh_.param("visual/grid_rows", visual_options_.grid_rows, 4);
    private_nh_.param("visual/grid_cols", visual_options_.grid_cols, 6);
    private_nh_.param("visual/minimum_occupied_cells",
                      visual_options_.minimum_occupied_cells, 8);
    private_nh_.param("visual/maximum_rotation_rate_deg",
                      visual_options_.maximum_rotation_rate_deg, 80.0);
    private_nh_.param("visual/maximum_rotation_step_deg",
                      visual_options_.maximum_rotation_step_deg, 12.0);
    private_nh_.param("visual/equalize_histogram",
                      visual_options_.equalize_histogram, true);
    private_nh_.param("visual/generate_debug_images",
                      visual_options_.generate_debug_images, true);
    visual_options_.generate_debug_images =
        visual_options_.generate_debug_images && publish_visual_debug_images_;
    private_nh_.param("visual/enable_lidar_pnp", visual_options_.enable_lidar_pnp, false);
    private_nh_.param("visual/require_lidar_pnp", visual_options_.require_lidar_pnp, true);
    private_nh_.param("visual/pnp_association_radius_pixels",
                      visual_options_.pnp_association_radius_pixels, 5);
    private_nh_.param("visual/pnp_minimum_correspondences",
                      visual_options_.pnp_minimum_correspondences, 45);
    private_nh_.param("visual/pnp_minimum_inliers",
                      visual_options_.pnp_minimum_inliers, 40);
    private_nh_.param("visual/pnp_minimum_occupied_cells",
                      visual_options_.pnp_minimum_occupied_cells, 6);
    private_nh_.param("visual/pnp_minimum_inlier_ratio",
                      visual_options_.pnp_minimum_inlier_ratio, 0.25);
    private_nh_.param("visual/pnp_ransac_reprojection_error",
                      visual_options_.pnp_ransac_reprojection_error, 2.0);
    private_nh_.param("visual/pnp_ransac_iterations",
                      visual_options_.pnp_ransac_iterations, 600);
    private_nh_.param("visual/pnp_maximum_reprojection_rmse",
                      visual_options_.pnp_maximum_reprojection_rmse, 1.8);
    private_nh_.param("visual/pnp_maximum_local_depth_difference",
                      visual_options_.pnp_maximum_local_depth_difference, 1.25);
    private_nh_.param("visual/pnp_maximum_local_depth_ratio",
                      visual_options_.pnp_maximum_local_depth_ratio, 0.08);
    private_nh_.param("visual/pnp_minimum_depth",
                      visual_options_.pnp_minimum_depth, 1.0);
    private_nh_.param("visual/pnp_maximum_depth",
                      visual_options_.pnp_maximum_depth, 80.0);
    private_nh_.param("visual/pnp_maximum_translation_speed",
                      visual_options_.pnp_maximum_translation_speed, 15.0);
    private_nh_.param("visual/pnp_maximum_translation_step",
                      visual_options_.pnp_maximum_translation_step, 5.0);
    std::vector<double> distortion;
    private_nh_.param<std::vector<double>>("visual/distortion", distortion,
                                          std::vector<double>());
    if (distortion.size() == visual_options_.distortion.size())
    {
      std::copy(distortion.begin(), distortion.end(), visual_options_.distortion.begin());
    }
    std::vector<double> body_from_camera;
    private_nh_.param<std::vector<double>>("visual/body_from_camera", body_from_camera,
                                          std::vector<double>());
    if (body_from_camera.size() == 9U || body_from_camera.size() == 12U ||
        body_from_camera.size() == 16U)
    {
      const int row_stride = body_from_camera.size() == 9U ? 3 : 4;
      for (int row = 0; row < 3; ++row)
      {
        for (int col = 0; col < 3; ++col)
        {
          visual_options_.body_from_camera_rotation(row, col) =
              body_from_camera[static_cast<std::size_t>(row * row_stride + col)];
        }
      }
      if (row_stride == 4)
      {
        for (int row = 0; row < 3; ++row)
        {
          visual_options_.body_from_camera_translation(row) =
              body_from_camera[static_cast<std::size_t>(row * row_stride + 3)];
        }
      }
    }

    visual_loop_options_.enabled = options_.enable_visual_loop_factors;
    visual_loop_options_.fx = visual_options_.fx;
    visual_loop_options_.fy = visual_options_.fy;
    visual_loop_options_.cx = visual_options_.cx;
    visual_loop_options_.cy = visual_options_.cy;
    visual_loop_options_.distortion = visual_options_.distortion;
    visual_loop_options_.body_from_camera_rotation =
        visual_options_.body_from_camera_rotation;
    visual_loop_options_.body_from_camera_translation =
        visual_options_.body_from_camera_translation;
    private_nh_.param("visual_loop/image_scale", visual_loop_options_.image_scale, 0.5);
    private_nh_.param("visual_loop/maximum_features",
                      visual_loop_options_.maximum_features, 1800);
    private_nh_.param("visual_loop/minimum_depth_features",
                      visual_loop_options_.minimum_depth_features, 80);
    private_nh_.param("visual_loop/maximum_database_size",
                      visual_loop_options_.maximum_database_size, 1600);
    private_nh_.param("visual_loop/debug_image_history_size",
                      visual_loop_options_.debug_image_history_size, 96);
    private_nh_.param("visual_loop/keyframe_distance",
                      visual_loop_options_.keyframe_distance, 0.75);
    private_nh_.param("visual_loop/keyframe_interval_sec",
                      visual_loop_options_.keyframe_interval_sec, 1.0);
    private_nh_.param("visual_loop/retrieval_interval_sec",
                      visual_loop_options_.retrieval_interval_sec, 4.0);
    private_nh_.param("visual_loop/minimum_index_gap",
                      visual_loop_options_.minimum_index_gap, 25);
    private_nh_.param("visual_loop/minimum_time_separation_sec",
                      visual_loop_options_.minimum_time_separation_sec, 45.0);
    private_nh_.param("visual_loop/search_radius",
                      visual_loop_options_.search_radius, 25.0);
    private_nh_.param("visual_loop/maximum_yaw_difference_deg",
                      visual_loop_options_.maximum_yaw_difference_deg, 70.0);
    private_nh_.param("visual_loop/maximum_retrieval_candidates",
                      visual_loop_options_.maximum_retrieval_candidates, 40);
    private_nh_.param("visual_loop/maximum_geometric_candidates",
                      visual_loop_options_.maximum_geometric_candidates, 5);
    private_nh_.param("visual_loop/enable_global_retrieval_fallback",
                      visual_loop_options_.enable_global_retrieval_fallback, false);
    private_nh_.param("visual_loop/maximum_global_retrieval_candidates",
                      visual_loop_options_.maximum_global_retrieval_candidates, 0);
    private_nh_.param("visual_loop/global_retrieval_feature_count",
                      visual_loop_options_.global_retrieval_feature_count, 450);
    private_nh_.param("visual_loop/global_retrieval_min_votes",
                      visual_loop_options_.global_retrieval_min_votes, 10);
    private_nh_.param("visual_loop/global_retrieval_min_table_count",
                      visual_loop_options_.global_retrieval_min_table_count, 2);
    private_nh_.param("visual_loop/minimum_global_geometric_candidates",
                      visual_loop_options_.minimum_global_geometric_candidates, 0);
    private_nh_.param("visual_loop/maximum_verified_candidates",
                      visual_loop_options_.maximum_verified_candidates, 1);
    private_nh_.param("visual_loop/candidate_reference_min_separation_sec",
                      visual_loop_options_.candidate_reference_min_separation_sec, 0.0);
    private_nh_.param("visual_loop/descriptor_ratio",
                      visual_loop_options_.descriptor_ratio, 0.75);
    private_nh_.param("visual_loop/minimum_descriptor_matches",
                      visual_loop_options_.minimum_descriptor_matches, 55);
    private_nh_.param("visual_loop/depth_association_radius_pixels",
                      visual_loop_options_.depth_association_radius_pixels, 5);
    private_nh_.param("visual_loop/minimum_depth",
                      visual_loop_options_.minimum_depth, 1.0);
    private_nh_.param("visual_loop/maximum_depth",
                      visual_loop_options_.maximum_depth, 80.0);
    private_nh_.param("visual_loop/pnp_iterations",
                      visual_loop_options_.pnp_iterations, 600);
    private_nh_.param("visual_loop/pnp_reprojection_error",
                      visual_loop_options_.pnp_reprojection_error, 2.5);
    private_nh_.param("visual_loop/minimum_pnp_inliers",
                      visual_loop_options_.minimum_pnp_inliers, 40);
    private_nh_.param("visual_loop/minimum_pnp_inlier_ratio",
                      visual_loop_options_.minimum_pnp_inlier_ratio, 0.35);
    private_nh_.param("visual_loop/grid_rows", visual_loop_options_.grid_rows, 4);
    private_nh_.param("visual_loop/grid_cols", visual_loop_options_.grid_cols, 6);
    private_nh_.param("visual_loop/minimum_occupied_cells",
                      visual_loop_options_.minimum_occupied_cells, 8);
    private_nh_.param("visual_loop/maximum_reprojection_rmse",
                      visual_loop_options_.maximum_reprojection_rmse, 2.0);
    private_nh_.param("visual_loop/maximum_translation_disagreement",
                      visual_loop_options_.maximum_translation_disagreement, 3.0);
    private_nh_.param("visual_loop/maximum_rotation_disagreement_deg",
                      visual_loop_options_.maximum_rotation_disagreement_deg, 10.0);
    private_nh_.param("visual_loop/minimum_quality",
                      visual_loop_options_.minimum_quality, 0.40);
    private_nh_.param("visual_loop/pending_max_size", max_pending_visual_loops_, 24);
    private_nh_.param("visual_loop/pending_max_retries_per_tick",
                      max_pending_visual_loop_retries_per_tick_, 2);
    private_nh_.param("visual_loop/pending_max_keyframe_lag_sec",
                      visual_loop_pending_max_keyframe_lag_sec_, 3.0);
  }

  bool lookupOdom(double stamp, OdomSample *sample) const
  {
    if (odom_history_.empty()) return false;
    double best = std::numeric_limits<double>::infinity();
    const OdomSample *match = nullptr;
    for (auto it = odom_history_.rbegin(); it != odom_history_.rend(); ++it)
    {
      const double difference = std::abs(it->stamp - stamp);
      if (difference < best)
      {
        best = difference;
        match = &*it;
      }
      if (it->stamp < stamp - max_pose_lookup_dt_) break;
    }
    if (!match || best > max_pose_lookup_dt_) return false;
    *sample = *match;
    return true;
  }

  Eigen::Isometry3d outputCorrection(const OdomSample &sample)
  {
    const Eigen::Isometry3d target = projectToSE3(
        graph_->correctedPose(sample.pose) * sample.pose.inverse());
    if (!target.matrix().allFinite()) return Eigen::Isometry3d::Identity();
    target_map_from_odom_ = target;
    if (!continuous_correction_enabled_)
    {
      applied_map_from_odom_ = target;
      have_continuous_correction_ = true;
      last_continuous_correction_stamp_ = sample.stamp;
    }
    else if (!have_continuous_correction_ ||
             !std::isfinite(last_continuous_correction_stamp_) ||
             sample.stamp + 1e-6 < last_continuous_correction_stamp_ ||
             sample.stamp - last_continuous_correction_stamp_ >
                 std::max(0.0, continuous_correction_reset_gap_sec_))
    {
      applied_map_from_odom_ = target;
      have_continuous_correction_ = true;
      last_continuous_correction_stamp_ = sample.stamp;
    }
    else
    {
      const double dt = std::max(0.0, sample.stamp - last_continuous_correction_stamp_);
      if (dt > 0.0)
      {
        applied_map_from_odom_ = rateLimitedCorrection(
            applied_map_from_odom_, target,
            std::max(0.0, continuous_correction_max_xy_rate_) * dt,
            std::max(0.0, continuous_correction_max_z_rate_) * dt,
            std::max(0.0, continuous_correction_max_rotation_rate_deg_) *
                std::acos(-1.0) / 180.0 * dt);
        last_continuous_correction_stamp_ = sample.stamp;
      }
    }
    const Eigen::Isometry3d lag = projectToSE3(
        applied_map_from_odom_.inverse() * target_map_from_odom_);
    correction_target_xy_ = target_map_from_odom_.translation().head<2>().norm();
    correction_target_z_ = target_map_from_odom_.translation().z();
    correction_target_rotation_deg_ = rotationDegrees(target_map_from_odom_.rotation());
    correction_applied_xy_ = applied_map_from_odom_.translation().head<2>().norm();
    correction_applied_z_ = applied_map_from_odom_.translation().z();
    correction_applied_rotation_deg_ = rotationDegrees(applied_map_from_odom_.rotation());
    correction_lag_xy_ = lag.translation().head<2>().norm();
    correction_lag_z_ = std::abs(lag.translation().z());
    correction_lag_rotation_deg_ = rotationDegrees(lag.rotation());
    return applied_map_from_odom_;
  }

  void publishCorrectedOdometry(const OdomSample &sample)
  {
    if (!graph_) return;
    const Eigen::Isometry3d correction = outputCorrection(sample);
    const Eigen::Isometry3d optimized = projectToSE3(correction * sample.pose);
    nav_msgs::Odometry output = sample.message;
    const ros::Time stamp = sample.message.header.stamp.isZero() ?
        ros::Time(sample.stamp) : sample.message.header.stamp;
    output.header.stamp = stamp;
    output.header.frame_id = map_frame_;
    output.child_frame_id = body_frame_;
    output.pose.pose = poseMessage(optimized);
    odom_pub_.publish(output);
    last_corrected_pose_ = optimized;
    last_stamp_ = stamp;
    have_output_ = true;
    if (!broadcast_tf_) return;

    geometry_msgs::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = map_frame_;
    transform.child_frame_id = body_frame_;
    transform.transform.translation.x = optimized.translation().x();
    transform.transform.translation.y = optimized.translation().y();
    transform.transform.translation.z = optimized.translation().z();
    const Eigen::Quaterniond quaternion(optimized.rotation());
    transform.transform.rotation.x = quaternion.x();
    transform.transform.rotation.y = quaternion.y();
    transform.transform.rotation.z = quaternion.z();
    transform.transform.rotation.w = quaternion.w();
    tf_broadcaster_.sendTransform(transform);
  }

  void odomCallback(const nav_msgs::OdometryConstPtr &message)
  {
    const double stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                        : message->header.stamp.toSec();
    OdomSample sample;
    sample.stamp = stamp;
    sample.pose = poseFromMessage(message->pose.pose);
    sample.message = *message;
    odom_history_.push_back(sample);
    while (!odom_history_.empty() && stamp - odom_history_.front().stamp > odom_history_sec_)
    {
      odom_history_.pop_front();
    }
    graph_->addOdometrySample(stamp, sample.pose);
    publishCorrectedOdometry(sample);
    processPendingClouds();
    processPendingSemanticClouds();
  }

  void semanticCallback(const sensor_msgs::PointCloud2ConstPtr &message)
  {
    if (!message) return;
    const double stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                       : message->header.stamp.toSec();
    ++semantic_messages_received_;
    if (!std::isfinite(stamp))
    {
      ++semantic_empty_messages_;
      return;
    }
    const auto insertion = std::upper_bound(
        pending_semantic_clouds_.begin(), pending_semantic_clouds_.end(), stamp,
        [](double value, const PendingSemanticCloud &candidate)
        {
          return value < candidate.stamp;
        });
    pending_semantic_clouds_.insert(insertion, PendingSemanticCloud{stamp, message});
    while (pending_semantic_clouds_.size() >
           static_cast<std::size_t>(max_pending_semantic_clouds_))
    {
      pending_semantic_clouds_.pop_back();
      ++semantic_queue_drops_;
    }
    processPendingSemanticClouds();
  }

  void processPendingSemanticClouds()
  {
    int processed = 0;
    while (processed < max_semantic_process_per_tick_ &&
           !pending_semantic_clouds_.empty() && !odom_history_.empty())
    {
      const PendingSemanticCloud &pending = pending_semantic_clouds_.front();
      // Match the registered-cloud ordering contract: a semantic observation
      // is only consumed after a frontend sample reaches its source time.
      if (odom_history_.back().stamp + 1e-9 < pending.stamp)
      {
        return;
      }
      OdomSample odom;
      if (!lookupOdom(pending.stamp, &odom))
      {
        ++semantic_age_rejections_;
        ROS_WARN_THROTTLE(5.0,
            "[semantic_gtsam] semantic observation has no frontend pose near %.6f",
            pending.stamp);
        pending_semantic_clouds_.pop_front();
        continue;
      }
      const sensor_msgs::PointCloud2ConstPtr message = pending.message;
      const double stamp = pending.stamp;
      pending_semantic_clouds_.pop_front();
      processSemanticCloud(message, stamp, odom);
      ++processed;
    }
  }

  void processSemanticCloud(const sensor_msgs::PointCloud2ConstPtr &message,
                            double stamp, const OdomSample &odom)
  {
    SemanticGraphPointVector semantic_points = decodeCloud(
        *message, max_semantic_points_, true, label_field_, confidence_field_);
    semantic_points_received_ += static_cast<std::uint64_t>(semantic_points.size());
    if (semantic_points.empty())
    {
      ++semantic_empty_messages_;
      return;
    }
    latest_semantic_age_ = std::abs(odom.stamp - stamp);
    if (semantic_cloud_in_map_frame_)
    {
      const Eigen::Isometry3d local_from_map = odom.pose.inverse();
      for (SemanticGraphPoint &point : semantic_points)
      {
        point.point = local_from_map * point.point;
      }
    }
    const SemanticPoseGraphStats before = graph_->stats();
    if (graph_->addSemanticObservation(stamp, odom.pose, semantic_points))
    {
      const SemanticPoseGraphStats after = graph_->stats();
      if (after.semantic_observation_factors > before.semantic_observation_factors)
      {
        const SemanticLoopDebug &debug = graph_->lastSemanticDebug();
        ROS_INFO("[semantic_gtsam] semantic factor applied: total=%d xy=%d z=%d "
                 "ref=%d cur=%d xy_inliers=%d z_inliers=%d xy_rmse=%.3f",
                 after.semantic_observation_factors,
                 after.semantic_observation_xy_factors,
                 after.semantic_observation_z_factors,
                 debug.reference_id, debug.current_id,
                 after.last_semantic_xy_inliers,
                 after.last_semantic_z_inliers,
                 after.last_semantic_xy_rmse);
      }
      ++semantic_clouds_used_;
      semantic_points_used_ += static_cast<std::uint64_t>(semantic_points.size());
      publishDebug();
      publishSemanticMap(false);
      publishStats();
    }
  }

  void wheelCallback(const insprobe_msgs::RangerOdometerConstPtr &message)
  {
    std::array<double, 4> speeds{{message->left_front_speed,
                                  message->right_front_speed,
                                  message->right_back_speed,
                                  message->left_back_speed}};
    if (!std::all_of(speeds.begin(), speeds.end(),
                     [](double value) { return std::isfinite(value); })) return;
    std::sort(speeds.begin(), speeds.end());
    const double stamp = !message->header.stamp.isZero() ? message->header.stamp.toSec()
        : std::isfinite(message->unixtime) && message->unixtime > 0.0
            ? message->unixtime : ros::Time::now().toSec();
    graph_->addWheelSample(stamp + wheel_time_offset_, 0.5 * (speeds[1] + speeds[2]));
  }

  void publishVisualDebug(const VisualRotationEstimate &estimate)
  {
    if (std::isfinite(estimate.image_cloud_time_difference))
    {
      std_msgs::Float64 difference;
      difference.data = estimate.image_cloud_time_difference;
      visual_time_difference_pub_.publish(difference);
      last_visual_time_difference_ = estimate.image_cloud_time_difference;
    }
    if (!publish_visual_debug_images_) return;
    if (!estimate.projection_debug_image.empty())
    {
      visual_projection_debug_pub_.publish(imageMessage(
          estimate.projection_debug_image, estimate.stamp, body_frame_));
    }
    if (!estimate.tracking_debug_image.empty())
    {
      visual_tracking_debug_pub_.publish(imageMessage(
          estimate.tracking_debug_image, estimate.stamp, body_frame_));
    }
    if (!estimate.pnp_debug_image.empty())
    {
      visual_pnp_debug_pub_.publish(imageMessage(
          estimate.pnp_debug_image, estimate.stamp, body_frame_));
    }
  }

  void handleVisualEstimate(const VisualRotationEstimate &estimate)
  {
    publishVisualDebug(estimate);
    last_visual_reason_ = estimate.reason;
    if (estimate.tracks > 0)
    {
      last_visual_tracks_ = estimate.tracks;
      last_visual_inliers_ = estimate.inliers;
    }
    if (estimate.pnp_correspondences > 0)
    {
      last_pnp_correspondences_ = estimate.pnp_correspondences;
      last_pnp_inliers_ = estimate.pnp_inliers;
      last_pnp_occupied_cells_ = estimate.pnp_occupied_cells;
      last_pnp_reprojection_rmse_ = estimate.pnp_reprojection_rmse;
    }
    if (estimate.observation_valid && !visual_observation_only_)
    {
      if (estimate.metric_pose_valid)
      {
        graph_->addVisualPoseSample(estimate.stamp, estimate.visual_from_body_pose,
                                    estimate.segment, estimate.quality);
      }
      else
      {
        graph_->addVisualRotationSample(estimate.stamp,
                                        estimate.visual_from_body_rotation,
                                        estimate.segment, estimate.quality);
      }
    }
    if (estimate.motion_valid)
    {
      ++visual_tracker_accepts_;
      last_visual_quality_ = estimate.quality;
      if (estimate.metric_pose_valid)
      {
        ++visual_pnp_accepts_;
        visual_pnp_quality_sum_ += estimate.quality;
        visual_pnp_inlier_ratio_sum_ += estimate.pnp_inlier_ratio;
        visual_pnp_reprojection_rmse_sum_ += estimate.pnp_reprojection_rmse;
        visual_pnp_translation_sum_ += estimate.relative_body_pose.translation().norm();
      }
    }
    else if (estimate.reason != "minimum_interval" &&
             estimate.reason != "initialized" &&
             estimate.reason != "reference_reset")
    {
      ++visual_tracker_rejections_;
    }
  }

  void handleVisualLoop(const VisualImageSample &image,
                        const VisualCloudSample &cloud)
  {
    if (!visual_loop_detector_ || !visual_loop_options_.enabled) return;
    OdomSample odom;
    if (!lookupOdom(image.stamp, &odom))
    {
      ++visual_loop_pose_drops_;
      last_visual_loop_reason_ = "visual_loop_pose_lookup_failed";
      return;
    }
    const VisualLoopResult result = visual_loop_detector_->process(
        image.stamp, image.image, cloud.points, odom.pose);
    last_visual_loop_reason_ = result.reason;
    if (result.keyframe_created) ++visual_loop_keyframes_;
    visual_loop_database_keyframes_ = static_cast<int>(visual_loop_detector_->keyframeCount());
    if (result.candidate_found) ++visual_loop_candidates_;
    last_visual_loop_matches_ = result.descriptor_matches;
    last_visual_loop_inliers_ = result.pnp_inliers;
    last_visual_loop_quality_ = result.quality;
    last_visual_loop_reprojection_rmse_ = result.reprojection_rmse;
    visual_loop_global_retrieval_candidates_ +=
        static_cast<std::uint64_t>(std::max(0, result.global_retrieval_candidates));
    visual_loop_global_retrieval_descriptor_matches_ += static_cast<std::uint64_t>(
        std::max(0, result.global_retrieval_descriptor_matches));
    last_visual_loop_global_retrieval_ = result.global_retrieval;
    last_visual_loop_global_retrieval_votes_ = result.global_retrieval_votes;
    last_visual_loop_global_retrieval_tables_ = result.global_retrieval_tables;
    if (!result.debug_image.empty() && publish_visual_debug_images_)
    {
      visual_loop_debug_pub_.publish(imageMessage(
          result.debug_image, result.current_stamp, body_frame_));
    }
    const auto &verified_candidates =
        visual_loop_detector_->lastAcceptedCandidates();
    if (verified_candidates.empty()) return;

    // The detector returns only independently verified, reference-diverse
    // PnP proposals. The first remains the primary debug result, while later
    // candidates are a bounded fallback when that reference cannot enter the
    // graph for a non-geometric reason such as cooldown.
    visual_loop_detector_accepts_ += static_cast<int>(verified_candidates.size());
    for (const VisualLoopResult &candidate : verified_candidates)
    {
      if (candidate.global_retrieval) ++visual_loop_global_retrieval_accepts_;
    }
    if (verified_candidates.size() > 1U)
    {
      visual_loop_alternative_candidates_ +=
          static_cast<int>(verified_candidates.size() - 1U);
    }
    const VisualLoopResult &primary = verified_candidates.front();
    last_accepted_visual_loop_reference_id_ = primary.reference_id;
    last_accepted_visual_loop_current_id_ = primary.current_id;
    last_accepted_visual_loop_matches_ = primary.descriptor_matches;
    last_accepted_visual_loop_inliers_ = primary.pnp_inliers;
    last_accepted_visual_loop_quality_ = primary.quality;
    last_accepted_visual_loop_reprojection_rmse_ = primary.reprojection_rmse;
    last_accepted_visual_loop_time_separation_sec_ = primary.temporal_separation_sec;
    last_accepted_visual_loop_xy_separation_ = primary.raw_xy_separation;
    last_accepted_visual_loop_z_separation_ = primary.raw_z_separation;

    for (std::size_t index = 0U; index < verified_candidates.size(); ++index)
    {
      const VisualLoopResult &candidate = verified_candidates[index];
      if (index > 0U) ++visual_loop_alternative_attempts_;
      PendingVisualLoopConstraint constraint;
      constraint.reference_stamp = candidate.reference_stamp;
      constraint.current_stamp = candidate.current_stamp;
      constraint.reference_from_current = candidate.reference_from_current;
      constraint.quality = candidate.quality;
      if (!visualLoopKeyframesReady(constraint))
      {
        // A pending candidate must preserve its future multi-frame support
        // state. Queue only the best one; adjacent frames will regenerate
        // alternatives once this pair reaches the graph keyframe timeline.
        if (index == 0U) enqueueVisualLoop(constraint);
        else ++visual_loop_alternative_pending_skips_;
        publishStats();
        break;
      }
      if (applyVisualLoopConstraint(constraint, &odom))
      {
        if (candidate.global_retrieval) ++visual_loop_global_retrieval_applied_;
        if (index > 0U) ++visual_loop_alternative_applied_;
        break;
      }

      // Once a candidate is collecting temporal support, evaluating another
      // reference at the same current keyframe would reset that hypothesis.
      // A graph-wide factor limit likewise makes all alternatives impossible.
      const bool support_in_progress =
          last_visual_loop_reason_ == "visual_loop_awaiting_multiframe_support" ||
          last_visual_loop_reason_ == "visual_loop_multiframe_disagreement";
      if (support_in_progress ||
          last_visual_loop_reason_ == "visual_loop_factor_limit")
      {
        break;
      }
    }
  }

  bool visualLoopKeyframesReady(const PendingVisualLoopConstraint &constraint) const
  {
    return graph_ && graph_->hasKeyframeNear(constraint.reference_stamp) &&
        graph_->hasKeyframeNear(constraint.current_stamp);
  }

  void enqueueVisualLoop(const PendingVisualLoopConstraint &constraint)
  {
    const auto duplicate = std::find_if(
        pending_visual_loops_.begin(), pending_visual_loops_.end(),
        [&](const PendingVisualLoopConstraint &candidate)
        {
          return std::abs(candidate.reference_stamp - constraint.reference_stamp) < 1e-6 &&
              std::abs(candidate.current_stamp - constraint.current_stamp) < 1e-6;
        });
    if (duplicate != pending_visual_loops_.end()) return;
    const std::size_t maximum = static_cast<std::size_t>(
        std::max(1, max_pending_visual_loops_));
    if (pending_visual_loops_.size() >= maximum)
    {
      ++visual_loop_graph_rejections_;
      ++visual_loop_pending_drops_;
      pending_visual_loops_.pop_front();
    }
    pending_visual_loops_.push_back(constraint);
    ++visual_loop_pending_enqueued_;
    last_visual_loop_reason_ = "visual_loop_waiting_for_keyframe";
  }

  bool applyVisualLoopConstraint(const PendingVisualLoopConstraint &constraint,
                                 const OdomSample *current_odom)
  {
    if (!graph_) return false;
    const Eigen::Isometry3d pose_before_loop = current_odom != nullptr ?
        graph_->correctedPose(current_odom->pose) : Eigen::Isometry3d::Identity();
    if (!graph_->addVisualLoopConstraint(
            constraint.reference_stamp, constraint.current_stamp,
            constraint.reference_from_current, constraint.quality))
    {
      const SemanticPoseGraphStats stats = graph_->stats();
      last_visual_loop_reason_ = stats.last_visual_loop_reason.empty() ?
          "visual_loop_graph_rejected" : stats.last_visual_loop_reason;
      if (last_visual_loop_reason_ == "visual_loop_awaiting_multiframe_support")
      {
        ++visual_loop_graph_consensus_waits_;
      }
      else
      {
        ++visual_loop_graph_rejections_;
      }
      publishStats();
      return false;
    }
    ++visual_loop_graph_applied_;
    last_visual_loop_reason_ = "visual_loop_applied";
    if (current_odom != nullptr)
    {
      const Eigen::Isometry3d pose_after_loop = graph_->correctedPose(current_odom->pose);
      const Eigen::Isometry3d correction = pose_before_loop.inverse() * pose_after_loop;
      last_visual_loop_correction_translation_ = correction.translation().norm();
      last_visual_loop_correction_rotation_deg_ = rotationDegrees(correction.rotation());
      // A graph factor is inserted after this odometry sample can already be
      // published. Re-emit it so live output reflects the iSAM2 update.
      publishCorrectedOdometry(*current_odom);
    }
    publishPath();
    publishSemanticMap(false);
    publishStats();
    return true;
  }

  void processPendingVisualLoops()
  {
    int processed = 0;
    const int maximum = std::max(1, max_pending_visual_loop_retries_per_tick_);
    while (processed < maximum && !pending_visual_loops_.empty())
    {
      const PendingVisualLoopConstraint constraint = pending_visual_loops_.front();
      if (!visualLoopKeyframesReady(constraint))
      {
        const double latest_keyframe_stamp = graph_ ? graph_->latestKeyframeStamp() :
            std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(latest_keyframe_stamp) ||
            latest_keyframe_stamp <= constraint.current_stamp +
                std::max(0.0, visual_loop_pending_max_keyframe_lag_sec_))
        {
          break;
        }
        pending_visual_loops_.pop_front();
        ++visual_loop_graph_rejections_;
        ++visual_loop_pending_expired_;
        last_visual_loop_reason_ = "visual_loop_keyframe_unavailable";
        ++processed;
        continue;
      }
      pending_visual_loops_.pop_front();
      ++visual_loop_pending_retries_;
      OdomSample odom;
      const OdomSample *current_odom = lookupOdom(constraint.current_stamp, &odom) ?
          &odom : nullptr;
      applyVisualLoopConstraint(constraint, current_odom);
      ++processed;
    }
  }

  void processVisualQueues()
  {
    while (!visual_image_queue_.empty() && !visual_cloud_queue_.empty())
    {
      const double difference = visual_image_queue_.front().stamp -
                                visual_cloud_queue_.front().stamp;
      if (std::abs(difference) <= visual_sync_tolerance_)
      {
        const VisualImageSample image = visual_image_queue_.front();
        const VisualCloudSample cloud = visual_cloud_queue_.front();
        visual_image_queue_.pop_front();
        visual_cloud_queue_.pop_front();
        VisualRotationEstimate estimate = visual_tracker_->process(
            image.stamp, image.image, cloud.points);
        estimate.image_cloud_time_difference = difference;
        handleVisualEstimate(estimate);
        handleVisualLoop(image, cloud);
      }
      else if (difference < 0.0)
      {
        visual_image_queue_.pop_front();
        ++visual_sync_drops_;
      }
      else
      {
        visual_cloud_queue_.pop_front();
        ++visual_sync_drops_;
      }
    }
  }

  void cameraCallback(const sensor_msgs::CompressedImageConstPtr &message)
  {
    if (!visual_tracker_ || message->data.empty()) return;
    const cv::Mat encoded(1, static_cast<int>(message->data.size()), CV_8U,
                          const_cast<uint8_t *>(message->data.data()));
    const cv::Mat image = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
      ++visual_decode_failures_;
      return;
    }
    const double raw_stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                            : message->header.stamp.toSec();
    const double stamp = raw_stamp + camera_time_offset_;
    if (visual_options_.enable_lidar_pnp || visual_loop_options_.enabled)
    {
      visual_image_queue_.push_back(VisualImageSample{stamp, image});
      while (visual_image_queue_.size() > 12U) visual_image_queue_.pop_front();
      processVisualQueues();
      return;
    }
    handleVisualEstimate(visual_tracker_->process(stamp, image));
  }

  sensor_msgs::PointCloud2 correctedCloudMessage(const sensor_msgs::PointCloud2 &source,
                                                 const SemanticGraphPointVector &map_points,
                                                 const Eigen::Isometry3d &correction) const
  {
    sensor_msgs::PointCloud2 output;
    output.header = source.header;
    output.header.frame_id = map_frame_;
    output.height = 1;
    output.is_dense = false;
    output.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(map_points.size());
    sensor_msgs::PointCloud2Iterator<float> x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> z(output, "z");
    for (const SemanticGraphPoint &source_point : map_points)
    {
      const Eigen::Vector3d point = correction * source_point.point;
      *x = static_cast<float>(point.x());
      *y = static_cast<float>(point.y());
      *z = static_cast<float>(point.z());
      ++x; ++y; ++z;
    }
    return output;
  }

  sensor_msgs::PointCloud2 semanticMapMessage(
      const SemanticGraphPointVector &points) const
  {
    sensor_msgs::PointCloud2 message;
    message.header.frame_id = map_frame_;
    message.header.stamp = last_stamp_.isZero() ? ros::Time::now() : last_stamp_;
    message.height = 1U;
    message.width = static_cast<std::uint32_t>(points.size());
    message.is_dense = false;
    message.is_bigendian = false;
    message.point_step = 24U;
    message.row_step = message.point_step * message.width;
    message.fields.resize(6U);
    const auto set_field = [&](std::size_t index, const std::string &name,
                               std::uint32_t offset, std::uint8_t datatype)
    {
      message.fields[index].name = name;
      message.fields[index].offset = offset;
      message.fields[index].datatype = datatype;
      message.fields[index].count = 1U;
    };
    set_field(0U, "x", 0U, sensor_msgs::PointField::FLOAT32);
    set_field(1U, "y", 4U, sensor_msgs::PointField::FLOAT32);
    set_field(2U, "z", 8U, sensor_msgs::PointField::FLOAT32);
    set_field(3U, "rgb", 12U, sensor_msgs::PointField::FLOAT32);
    set_field(4U, "label", 16U, sensor_msgs::PointField::UINT32);
    set_field(5U, "confidence", 20U, sensor_msgs::PointField::FLOAT32);
    message.data.resize(static_cast<std::size_t>(message.row_step), 0U);
    for (std::size_t index = 0; index < points.size(); ++index)
    {
      std::uint8_t *data = message.data.data() + index * message.point_step;
      const float x = static_cast<float>(points[index].point.x());
      const float y = static_cast<float>(points[index].point.y());
      const float z = static_cast<float>(points[index].point.z());
      const std::uint32_t rgb = semanticRgb(points[index].label);
      const std::uint32_t label = points[index].label;
      const float confidence = points[index].confidence;
      std::memcpy(data + 0U, &x, sizeof(x));
      std::memcpy(data + 4U, &y, sizeof(y));
      std::memcpy(data + 8U, &z, sizeof(z));
      std::memcpy(data + 12U, &rgb, sizeof(rgb));
      std::memcpy(data + 16U, &label, sizeof(label));
      std::memcpy(data + 20U, &confidence, sizeof(confidence));
    }
    return message;
  }

  void publishSemanticMap(bool force)
  {
    if (!subscribe_semantic_ || semantic_map_publish_rate_ <= 0.0 || !graph_) return;
    const SemanticPoseGraphStats stats = graph_->stats();
    if (stats.semantic_keyframes <= 0 ||
        (!force && stats.semantic_keyframes == last_semantic_map_keyframes_))
    {
      return;
    }
    const ros::WallTime now = ros::WallTime::now();
    const double period = 1.0 / std::max(0.01, semantic_map_publish_rate_);
    if (!force && last_semantic_map_publish_wall_.toSec() > 0.0 &&
        (now - last_semantic_map_publish_wall_).toSec() < period)
    {
      return;
    }
    const SemanticGraphPointVector points = graph_->semanticMap(
        semantic_map_voxel_size_, semantic_map_max_points_);
    if (points.empty()) return;
    semantic_map_pub_.publish(semanticMapMessage(points));
    semantic_map_points_ = static_cast<int>(points.size());
    ++semantic_map_publications_;
    last_semantic_map_keyframes_ = stats.semantic_keyframes;
    last_semantic_map_publish_wall_ = now;
  }

  void processRegisteredCloud(const sensor_msgs::PointCloud2ConstPtr &message,
                              const OdomSample &odom)
  {
    const double stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                        : message->header.stamp.toSec();
    const SemanticGraphPointVector map_points = decodeCloud(*message, max_registered_points_,
                                                             false, label_field_,
                                                             confidence_field_);
    if (map_points.empty()) return;
    SemanticGraphPointVector local_points;
    local_points.reserve(map_points.size());
    const Eigen::Isometry3d local_from_map = odom.pose.inverse();
    for (const SemanticGraphPoint &map_point : map_points)
    {
      SemanticGraphPoint point = map_point;
      point.point = local_from_map * map_point.point;
      local_points.push_back(point);
    }
    const bool keyframe_added = graph_->addFrame(stamp, odom.pose, local_points);
    if (subscribe_camera_ &&
        (visual_options_.enable_lidar_pnp || visual_loop_options_.enabled))
    {
      VisualCloudSample visual_cloud;
      visual_cloud.stamp = stamp;
      visual_cloud.points.reserve(local_points.size());
      for (const SemanticGraphPoint &point : local_points)
      {
        if (point.point.allFinite()) visual_cloud.points.push_back(point.point);
      }
      visual_cloud_queue_.push_back(std::move(visual_cloud));
      while (visual_cloud_queue_.size() > 6U) visual_cloud_queue_.pop_front();
      processVisualQueues();
    }
    if (keyframe_added) processPendingVisualLoops();

    const Eigen::Isometry3d correction = outputCorrection(odom);
    corrected_cloud_pub_.publish(correctedCloudMessage(*message, map_points, correction));
    if (keyframe_added)
    {
      publishDebug();
      publishStats();
      publishSemanticMap(false);
      const SemanticPoseGraphStats stats = graph_->stats();
      if (trajectory_checkpoint_keyframes_ > 0 &&
          stats.keyframes >= last_checkpoint_keyframes_ + trajectory_checkpoint_keyframes_)
      {
        saveTrajectory("checkpoint");
        last_checkpoint_keyframes_ = stats.keyframes;
      }
    }
  }

  void processPendingClouds()
  {
    while (!pending_registered_clouds_.empty() && !odom_history_.empty())
    {
      const sensor_msgs::PointCloud2ConstPtr message = pending_registered_clouds_.front();
      const double stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                          : message->header.stamp.toSec();
      // Wait for at least one odometry sample at or after the cloud stamp. This
      // makes cross-topic callback ordering deterministic during live use and
      // rosbag replay, while lookupOdom still selects the nearest sample.
      if (odom_history_.back().stamp + 1e-9 < stamp) return;
      OdomSample odom;
      if (!lookupOdom(stamp, &odom))
      {
        ++registered_cloud_pose_drops_;
        ROS_WARN_THROTTLE(5.0,
            "[semantic_gtsam] registered cloud has no frontend pose near %.6f", stamp);
        pending_registered_clouds_.pop_front();
        continue;
      }
      pending_registered_clouds_.pop_front();
      processRegisteredCloud(message, odom);
    }
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &message)
  {
    if (!process_registered_clouds_ || !message) return;
    pending_registered_clouds_.push_back(message);
    const std::size_t maximum = static_cast<std::size_t>(
        std::max(1, max_pending_registered_clouds_));
    while (pending_registered_clouds_.size() > maximum)
    {
      pending_registered_clouds_.pop_front();
      ++registered_cloud_queue_drops_;
    }
    processPendingClouds();
  }

  visualization_msgs::MarkerArray makeDebugMarkers(const GraphDebugPairVector &pairs,
                                                    const SemanticLoopDebug &debug,
                                                    const std::string &prefix) const
  {
    visualization_msgs::MarkerArray array;
    visualization_msgs::Marker clear;
    clear.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(clear);
    const std::array<DebugPairStage, 4> stages{{DebugPairStage::Candidate,
                                                DebugPairStage::Inlier,
                                                DebugPairStage::Outlier,
                                                DebugPairStage::Applied}};
    for (std::size_t stage_index = 0; stage_index < stages.size(); ++stage_index)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = map_frame_;
      marker.header.stamp = ros::Time::now();
      marker.ns = prefix + "/" + stageName(stages[stage_index]);
      marker.id = static_cast<int>(stage_index);
      marker.type = visualization_msgs::Marker::LINE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = stages[stage_index] == DebugPairStage::Applied ? 0.055 : 0.035;
      marker.color = stageColor(stages[stage_index]);
      const int stage_count = static_cast<int>(std::count_if(pairs.begin(), pairs.end(),
          [&](const GraphDebugPair &pair) { return pair.stage == stages[stage_index]; }));
      const int stride = std::max(1, stage_count / std::max(1, max_debug_pairs_));
      int index = 0;
      for (const GraphDebugPair &pair : pairs)
      {
        if (pair.stage != stages[stage_index]) continue;
        if (index++ % stride != 0) continue;
        geometry_msgs::Point source;
        source.x = pair.source_world.x(); source.y = pair.source_world.y();
        source.z = pair.source_world.z();
        geometry_msgs::Point target;
        target.x = pair.target_world.x(); target.y = pair.target_world.y();
        target.z = pair.target_world.z();
        marker.points.push_back(source);
        marker.points.push_back(target);
        if (static_cast<int>(marker.points.size()) >= 2 * max_debug_pairs_) break;
      }
      array.markers.push_back(marker);
    }
    visualization_msgs::Marker text;
    text.header.frame_id = map_frame_;
    text.header.stamp = ros::Time::now();
    text.ns = prefix + "/status";
    text.id = 10;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::Marker::ADD;
    text.pose.orientation.w = 1.0;
    text.pose.position = have_output_ ? poseMessage(last_corrected_pose_).position
                                     : geometry_msgs::Point();
    text.pose.position.z += 2.0;
    text.scale.z = 0.45;
    text.color.r = 1.0F; text.color.g = 1.0F; text.color.b = 1.0F; text.color.a = 1.0F;
    std::ostringstream stream;
    stream << "ref=" << debug.reference_id << " cur=" << debug.current_id
           << " " << (!debug.valid ? "WAIT" : debug.accepted ? "ACCEPT" : "REJECT")
           << " reason=" << debug.reason;
    if (prefix.find("xy") != std::string::npos)
      stream << " rmse=" << std::fixed << std::setprecision(3) << debug.xy_rmse;
    else stream << " dz=" << std::fixed << std::setprecision(3) << debug.z_median
                << " mad=" << debug.z_mad;
    text.text = stream.str();
    array.markers.push_back(text);
    return array;
  }

  void publishDebug()
  {
    const SemanticLoopDebug &debug = graph_->lastDebug();
    xy_debug_pub_.publish(makeDebugMarkers(debug.xy_pairs, debug, "xy"));
    z_debug_pub_.publish(makeDebugMarkers(debug.z_pairs, debug, "z"));
    const SemanticLoopDebug &semantic = graph_->lastSemanticDebug();
    semantic_xy_debug_pub_.publish(makeDebugMarkers(
        semantic.xy_pairs, semantic, "semantic_xy"));
    semantic_z_debug_pub_.publish(makeDebugMarkers(
        semantic.z_pairs, semantic, "semantic_z"));
  }

  void publishStats()
  {
    const SemanticPoseGraphStats stats = graph_->stats();
    const SemanticLoopDebug &debug = graph_->lastDebug();
    const SemanticLoopDebug &semantic_debug = graph_->lastSemanticDebug();
    std_msgs::String message;
    std::ostringstream stream;
    const auto metric = [](double value)
    {
      return std::isfinite(value) ? value : -1.0;
    };
    stream << "{\"keyframes\":" << stats.keyframes
           << ",\"odom_factors\":" << stats.odometry_factors
           << ",\"sequential_ground_factors\":" << stats.sequential_ground_factors
           << ",\"sequential_ground_rejections\":" << stats.sequential_ground_rejections
           << ",\"sequential_ground_support_waits\":"
           << stats.sequential_ground_support_waits
           << ",\"sequential_ground_support_confirmations\":"
           << stats.sequential_ground_support_confirmations
           << ",\"wheel_factors\":" << stats.wheel_factors
           << ",\"wheel_rejections\":" << stats.wheel_rejections
           << ",\"visual_rotation_factors\":" << stats.visual_rotation_factors
           << ",\"visual_translation_factors\":" << stats.visual_translation_factors
           << ",\"visual_rotation_rejections\":" << stats.visual_rotation_rejections
           << ",\"visual_loop_attempts\":" << stats.visual_loop_attempts
           << ",\"visual_loop_rejections\":" << stats.visual_loop_rejections
           << ",\"visual_loop_factors\":" << stats.visual_loop_factors
           << ",\"visual_loop_ground_z_refinements\":"
           << stats.visual_loop_ground_z_refinements
           << ",\"visual_loop_ground_z_clipped_refinements\":"
           << stats.visual_loop_ground_z_clipped_refinements
           << ",\"visual_loop_ground_z_sparse_refinements\":"
           << stats.visual_loop_ground_z_sparse_refinements
           << ",\"visual_loop_cooldown_rejections\":"
           << stats.visual_loop_cooldown_rejections
           << ",\"visual_loop_factor_limit_rejections\":"
           << stats.visual_loop_factor_limit_rejections
           << ",\"visual_loop_z_without_ground_suppressed\":"
           << stats.visual_loop_z_without_ground_suppressed
           << ",\"visual_loop_lidar_geometry_validations\":"
           << stats.visual_loop_lidar_geometry_validations
           << ",\"visual_loop_lidar_seeded_validations\":"
           << stats.visual_loop_lidar_seeded_validations
           << ",\"visual_loop_lidar_geometry_rejections\":"
           << stats.visual_loop_lidar_geometry_rejections
           << ",\"visual_loop_support_waits\":" << stats.visual_loop_support_waits
           << ",\"visual_loop_support_confirmations\":"
           << stats.visual_loop_support_confirmations
           << ",\"visual_loop_support_resets\":"
           << stats.visual_loop_support_resets
           << ",\"visual_loop_support_disagreements\":"
           << stats.visual_loop_support_disagreements
           << ",\"visual_loop_last_support\":" << stats.last_visual_loop_support
           << ",\"visual_loop_support_xy_disagreement\":"
           << metric(stats.last_visual_loop_support_xy_disagreement)
           << ",\"visual_loop_support_yaw_disagreement_deg\":"
           << metric(stats.last_visual_loop_support_yaw_disagreement_deg)
           << ",\"visual_loop_support_z_disagreement\":"
           << metric(stats.last_visual_loop_support_z_disagreement)
           << ",\"visual_loop_lidar_candidates\":"
           << stats.last_visual_loop_lidar_candidates
           << ",\"visual_loop_lidar_inliers\":"
           << stats.last_visual_loop_lidar_inliers
           << ",\"visual_loop_lidar_accepted\":"
           << (stats.last_visual_loop_lidar_accepted ? "true" : "false")
           << ",\"visual_loop_lidar_seeded\":"
           << (stats.last_visual_loop_lidar_seeded ? "true" : "false")
           << ",\"visual_loop_lidar_rmse\":"
           << metric(stats.last_visual_loop_lidar_rmse)
           << ",\"visual_loop_lidar_spread\":"
           << metric(stats.last_visual_loop_lidar_spread)
           << ",\"visual_loop_lidar_pnp_xy_disagreement\":"
           << metric(stats.last_visual_loop_lidar_pnp_xy_disagreement)
           << ",\"visual_loop_lidar_pnp_yaw_disagreement_deg\":"
           << metric(stats.last_visual_loop_lidar_pnp_yaw_disagreement_deg)
           << ",\"visual_loop_ground_z_candidates\":"
           << stats.last_visual_loop_ground_z_candidates
           << ",\"visual_loop_ground_z_broad_candidates\":"
           << stats.last_visual_loop_ground_z_broad_candidates
           << ",\"visual_loop_ground_z_inliers\":"
           << stats.last_visual_loop_ground_z_inliers
           << ",\"visual_loop_ground_z_accepted\":"
           << (stats.last_visual_loop_ground_z_accepted ? "true" : "false")
           << ",\"visual_loop_ground_z_sparse_accepted\":"
           << (stats.last_visual_loop_ground_z_sparse_accepted ? "true" : "false")
           << ",\"visual_loop_z_constrained\":"
           << (stats.last_visual_loop_z_constrained ? "true" : "false")
           << ",\"visual_loop_ground_z_correction\":"
           << metric(stats.last_visual_loop_ground_z_correction)
           << ",\"visual_loop_ground_z_applied_correction\":"
           << metric(stats.last_visual_loop_ground_z_applied_correction)
           << ",\"visual_loop_ground_z_mad\":"
           << metric(stats.last_visual_loop_ground_z_mad)
           << ",\"visual_loop_ground_z_inlier_ratio\":"
           << metric(stats.last_visual_loop_ground_z_inlier_ratio)
           << ",\"visual_loop_ground_z_spread\":"
           << metric(stats.last_visual_loop_ground_z_spread)
           << ",\"visual_loop_ground_z_spread_ratio\":"
           << metric(stats.last_visual_loop_ground_z_spread_ratio)
           << ",\"visual_loop_graph_xy_innovation\":"
           << metric(stats.last_visual_loop_graph_xy_innovation)
           << ",\"visual_loop_graph_yaw_innovation_deg\":"
           << metric(stats.last_visual_loop_graph_yaw_innovation_deg)
           << ",\"visual_loop_graph_z_innovation\":"
           << metric(stats.last_visual_loop_graph_z_innovation)
           << ",\"visual_loop_graph_reason\":\""
           << stats.last_visual_loop_reason << "\""
           << ",\"visual_loop_keyframes\":" << visual_loop_keyframes_
           << ",\"visual_loop_database_keyframes\":"
           << visual_loop_database_keyframes_
           << ",\"visual_loop_candidates\":" << visual_loop_candidates_
           << ",\"visual_loop_detector_accepts\":" << visual_loop_detector_accepts_
           << ",\"visual_loop_alternative_candidates\":"
           << visual_loop_alternative_candidates_
           << ",\"visual_loop_alternative_attempts\":"
           << visual_loop_alternative_attempts_
           << ",\"visual_loop_alternative_applied\":"
           << visual_loop_alternative_applied_
           << ",\"visual_loop_alternative_pending_skips\":"
           << visual_loop_alternative_pending_skips_
           << ",\"visual_loop_global_retrieval_candidates\":"
           << visual_loop_global_retrieval_candidates_
           << ",\"visual_loop_global_retrieval_descriptor_matches\":"
           << visual_loop_global_retrieval_descriptor_matches_
           << ",\"visual_loop_global_retrieval_accepts\":"
           << visual_loop_global_retrieval_accepts_
           << ",\"visual_loop_global_retrieval_applied\":"
           << visual_loop_global_retrieval_applied_
           << ",\"visual_loop_last_global_retrieval\":"
           << (last_visual_loop_global_retrieval_ ? "true" : "false")
           << ",\"visual_loop_last_global_votes\":"
           << last_visual_loop_global_retrieval_votes_
           << ",\"visual_loop_last_global_tables\":"
           << last_visual_loop_global_retrieval_tables_
           << ",\"visual_loop_graph_applied\":" << visual_loop_graph_applied_
           << ",\"visual_loop_graph_rejections\":" << visual_loop_graph_rejections_
           << ",\"visual_loop_graph_consensus_waits\":"
           << visual_loop_graph_consensus_waits_
           << ",\"visual_loop_pending\":" << pending_visual_loops_.size()
           << ",\"visual_loop_pending_enqueued\":" << visual_loop_pending_enqueued_
           << ",\"visual_loop_pending_retries\":" << visual_loop_pending_retries_
           << ",\"visual_loop_pending_expired\":" << visual_loop_pending_expired_
           << ",\"visual_loop_pending_drops\":" << visual_loop_pending_drops_
           << ",\"visual_loop_pose_drops\":" << visual_loop_pose_drops_
           << ",\"visual_loop_matches\":" << last_visual_loop_matches_
           << ",\"visual_loop_inliers\":" << last_visual_loop_inliers_
           << ",\"visual_loop_quality\":" << metric(last_visual_loop_quality_)
           << ",\"visual_loop_reprojection_rmse\":"
           << metric(last_visual_loop_reprojection_rmse_)
           << ",\"visual_loop_correction_translation\":"
           << metric(last_visual_loop_correction_translation_)
           << ",\"visual_loop_correction_rotation_deg\":"
           << metric(last_visual_loop_correction_rotation_deg_)
           << ",\"visual_loop_last_accepted_reference_id\":"
           << last_accepted_visual_loop_reference_id_
           << ",\"visual_loop_last_accepted_current_id\":"
           << last_accepted_visual_loop_current_id_
           << ",\"visual_loop_last_accepted_matches\":"
           << last_accepted_visual_loop_matches_
           << ",\"visual_loop_last_accepted_inliers\":"
           << last_accepted_visual_loop_inliers_
           << ",\"visual_loop_last_accepted_quality\":"
           << metric(last_accepted_visual_loop_quality_)
           << ",\"visual_loop_last_accepted_reprojection_rmse\":"
           << metric(last_accepted_visual_loop_reprojection_rmse_)
           << ",\"visual_loop_last_accepted_time_separation_sec\":"
           << metric(last_accepted_visual_loop_time_separation_sec_)
           << ",\"visual_loop_last_accepted_xy_separation\":"
           << metric(last_accepted_visual_loop_xy_separation_)
           << ",\"visual_loop_last_accepted_z_separation\":"
           << metric(last_accepted_visual_loop_z_separation_)
           << ",\"visual_loop_reason\":\"" << last_visual_loop_reason_ << "\""
           << ",\"continuous_correction_enabled\":"
           << (continuous_correction_enabled_ ? "true" : "false")
           << ",\"correction_target_xy\":" << metric(correction_target_xy_)
           << ",\"correction_target_z\":" << metric(correction_target_z_)
           << ",\"correction_target_rotation_deg\":"
           << metric(correction_target_rotation_deg_)
           << ",\"correction_applied_xy\":" << metric(correction_applied_xy_)
           << ",\"correction_applied_z\":" << metric(correction_applied_z_)
           << ",\"correction_applied_rotation_deg\":"
           << metric(correction_applied_rotation_deg_)
           << ",\"correction_lag_xy\":" << metric(correction_lag_xy_)
           << ",\"correction_lag_z\":" << metric(correction_lag_z_)
           << ",\"correction_lag_rotation_deg\":"
           << metric(correction_lag_rotation_deg_)
           << ",\"visual_observation_only\":"
           << (visual_observation_only_ ? "true" : "false")
           << ",\"visual_tracker_accepts\":" << visual_tracker_accepts_
           << ",\"visual_pnp_accepts\":" << visual_pnp_accepts_
           << ",\"visual_tracker_rejections\":" << visual_tracker_rejections_
           << ",\"visual_sync_drops\":" << visual_sync_drops_
           << ",\"visual_decode_failures\":" << visual_decode_failures_
           << ",\"visual_image_cloud_dt\":" << metric(last_visual_time_difference_)
           << ",\"visual_tracks\":" << last_visual_tracks_
           << ",\"visual_inliers\":" << last_visual_inliers_
           << ",\"pnp_correspondences\":" << last_pnp_correspondences_
           << ",\"pnp_inliers\":" << last_pnp_inliers_
           << ",\"pnp_occupied_cells\":" << last_pnp_occupied_cells_
           << ",\"pnp_reprojection_rmse\":" << last_pnp_reprojection_rmse_
           << ",\"visual_quality\":" << last_visual_quality_
           << ",\"pnp_mean_quality\":"
           << (visual_pnp_accepts_ > 0 ?
               visual_pnp_quality_sum_ / visual_pnp_accepts_ : 0.0)
           << ",\"pnp_mean_inlier_ratio\":"
           << (visual_pnp_accepts_ > 0 ?
               visual_pnp_inlier_ratio_sum_ / visual_pnp_accepts_ : 0.0)
           << ",\"pnp_mean_reprojection_rmse\":"
           << (visual_pnp_accepts_ > 0 ?
               visual_pnp_reprojection_rmse_sum_ / visual_pnp_accepts_ : 0.0)
           << ",\"pnp_mean_translation\":"
           << (visual_pnp_accepts_ > 0 ?
               visual_pnp_translation_sum_ / visual_pnp_accepts_ : 0.0)
           << ",\"visual_reason\":\"" << last_visual_reason_ << "\""
           << ",\"loop_attempts\":" << stats.loop_attempts
           << ",\"loop_rejections\":" << stats.loop_rejections
           << ",\"loop_factors\":" << stats.loop_factors
           << ",\"loop_support_confirmations\":" << stats.loop_support_confirmations
           << ",\"loop_cooldown_rejections\":" << stats.loop_cooldown_rejections
           << ",\"loop_factor_limit_rejections\":"
           << stats.loop_factor_limit_rejections
           << ",\"loop_last_support\":" << stats.last_loop_support
           << ",\"loop_last_accepted_reference_id\":" << stats.last_loop_reference_id
           << ",\"loop_last_accepted_current_id\":" << stats.last_loop_current_id
           << ",\"loop_last_accepted_time_separation_sec\":"
           << metric(stats.last_loop_time_separation_sec)
           << ",\"xy_loop_factors\":" << stats.xy_loop_factors
           << ",\"z_loop_factors\":" << stats.z_loop_factors
           << ",\"semantic_keyframes\":" << stats.semantic_keyframes
           << ",\"semantic_observations_received\":"
           << stats.semantic_observations_received
           << ",\"semantic_observations_associated\":"
           << stats.semantic_observations_associated
           << ",\"semantic_subscribed\":" << (subscribe_semantic_ ? "true" : "false")
           << ",\"registered_cloud_pending\":"
           << pending_registered_clouds_.size()
           << ",\"registered_cloud_queue_drops\":"
           << registered_cloud_queue_drops_
           << ",\"registered_cloud_pose_drops\":"
           << registered_cloud_pose_drops_
           << ",\"semantic_pending\":" << pending_semantic_clouds_.size()
           << ",\"semantic_queue_drops\":" << semantic_queue_drops_
           << ",\"semantic_messages\":" << semantic_messages_received_
           << ",\"semantic_points_received\":" << semantic_points_received_
           << ",\"semantic_empty_messages\":" << semantic_empty_messages_
           << ",\"semantic_clouds_used\":" << semantic_clouds_used_
           << ",\"semantic_points_used\":" << semantic_points_used_
           << ",\"semantic_age_rejections\":" << semantic_age_rejections_
           << ",\"semantic_latest_age\":" << latest_semantic_age_
           << ",\"semantic_observation_attempts\":"
           << stats.semantic_observation_attempts
           << ",\"semantic_observation_skips\":"
           << stats.semantic_observation_skips
           << ",\"semantic_observation_rejections\":"
           << stats.semantic_observation_rejections
           << ",\"semantic_observation_reference_rejections\":"
           << stats.semantic_observation_reference_rejections
           << ",\"semantic_observation_factors\":"
           << stats.semantic_observation_factors
           << ",\"semantic_observation_xy_factors\":"
           << stats.semantic_observation_xy_factors
           << ",\"semantic_observation_z_factors\":"
           << stats.semantic_observation_z_factors
           << ",\"semantic_observation_inliers\":"
           << stats.semantic_observation_inliers
           << ",\"semantic_pending_observations\":"
           << stats.semantic_pending_observations
           << ",\"last_semantic_xy_candidates\":"
           << stats.last_semantic_xy_candidates
           << ",\"last_semantic_xy_inliers\":"
           << stats.last_semantic_xy_inliers
           << ",\"last_semantic_xy_inlier_ratio\":"
           << metric(stats.last_semantic_xy_inlier_ratio)
           << ",\"last_semantic_xy_rmse\":"
           << metric(stats.last_semantic_xy_rmse)
           << ",\"last_semantic_xy_spread\":"
           << metric(stats.last_semantic_xy_spread)
           << ",\"last_semantic_xy_spread_ratio\":"
           << metric(stats.last_semantic_xy_spread_ratio)
           << ",\"last_semantic_baseline\":"
           << metric(stats.last_semantic_baseline)
           << ",\"last_semantic_xy_correction\":"
           << metric(stats.last_semantic_xy_correction)
           << ",\"last_semantic_yaw_correction_deg\":"
           << metric(stats.last_semantic_yaw_correction_deg)
           << ",\"last_semantic_z_candidates\":"
           << stats.last_semantic_z_candidates
           << ",\"last_semantic_z_inliers\":"
           << stats.last_semantic_z_inliers
           << ",\"last_semantic_z_median\":"
           << metric(stats.last_semantic_z_median)
           << ",\"last_semantic_z_mad\":"
           << metric(stats.last_semantic_z_mad)
           << ",\"semantic_map_points\":" << semantic_map_points_
           << ",\"semantic_map_publications\":" << semantic_map_publications_
           << ",\"last_semantic_reason\":\"" << semantic_debug.reason << "\""
           << ",\"optimization_ms\":" << stats.last_optimization_ms
           << ",\"last_reason\":\"" << debug.reason << "\"}";
    message.data = stream.str();
    stats_pub_.publish(message);
  }

  void publishPath()
  {
    const auto trajectory = graph_->optimizedTrajectory();
    if (trajectory.empty()) return;
    nav_msgs::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = last_stamp_.isZero() ? ros::Time::now() : last_stamp_;
    const std::size_t maximum = 20000U;
    const std::size_t stride = std::max<std::size_t>(1U, trajectory.size() / maximum);
    path.poses.reserve(std::min(maximum, trajectory.size()));
    for (std::size_t index = 0; index < trajectory.size(); index += stride)
    {
      geometry_msgs::PoseStamped pose;
      pose.header.frame_id = map_frame_;
      pose.header.stamp = ros::Time(trajectory[index].stamp);
      pose.pose = poseMessage(trajectory[index].optimized_pose);
      path.poses.push_back(pose);
    }
    path_pub_.publish(path);
  }

  void timerCallback(const ros::TimerEvent &)
  {
    publishPath();
    publishSemanticMap(false);
    publishStats();
  }

  void saveTrajectory(const std::string &reason)
  {
    if (!save_on_shutdown_ || trajectory_save_path_.empty() || !graph_) return;
    if (graph_->saveOptimizedTrajectory(trajectory_save_path_))
    {
      ROS_INFO("[semantic_gtsam] optimized trajectory saved (%s): %s",
               reason.c_str(), trajectory_save_path_.c_str());
    }
    else
    {
      ROS_WARN("[semantic_gtsam] failed to save optimized trajectory (%s): %s",
               reason.c_str(), trajectory_save_path_.c_str());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber semantic_sub_;
  ros::Subscriber wheel_sub_;
  ros::Subscriber camera_sub_;
  ros::Publisher odom_pub_;
  ros::Publisher path_pub_;
  ros::Publisher corrected_cloud_pub_;
  ros::Publisher stats_pub_;
  ros::Publisher xy_debug_pub_;
  ros::Publisher z_debug_pub_;
  ros::Publisher semantic_xy_debug_pub_;
  ros::Publisher semantic_z_debug_pub_;
  ros::Publisher semantic_map_pub_;
  ros::Publisher visual_projection_debug_pub_;
  ros::Publisher visual_tracking_debug_pub_;
  ros::Publisher visual_pnp_debug_pub_;
  ros::Publisher visual_time_difference_pub_;
  ros::Publisher visual_loop_debug_pub_;
  ros::Timer timer_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  std::string frontend_odom_topic_;
  std::string registered_cloud_topic_;
  std::string semantic_cloud_topic_;
  std::string wheel_topic_;
  double wheel_time_offset_ = 0.0;
  std::string camera_topic_;
  std::string output_odom_topic_;
  std::string output_path_topic_;
  std::string corrected_cloud_topic_;
  std::string stats_topic_;
  std::string xy_debug_topic_;
  std::string z_debug_topic_;
  std::string semantic_xy_debug_topic_;
  std::string semantic_z_debug_topic_;
  std::string semantic_map_topic_;
  std::string visual_projection_debug_topic_;
  std::string visual_tracking_debug_topic_;
  std::string visual_pnp_debug_topic_;
  std::string visual_time_difference_topic_;
  std::string visual_loop_debug_topic_;
  std::string map_frame_;
  std::string body_frame_;
  std::string label_field_;
  std::string confidence_field_;
  std::string trajectory_save_path_;

  bool subscribe_semantic_ = false;
  bool subscribe_camera_ = false;
  bool process_registered_clouds_ = true;
  bool visual_observation_only_ = true;
  bool publish_visual_debug_images_ = true;
  bool semantic_cloud_in_map_frame_ = true;
  bool broadcast_tf_ = true;
  bool save_on_shutdown_ = true;
  bool continuous_correction_enabled_ = false;
  bool have_output_ = false;
  bool have_continuous_correction_ = false;
  double max_pose_lookup_dt_ = 0.15;
  double odom_history_sec_ = 120.0;
  double path_publish_rate_ = 0.5;
  double semantic_map_publish_rate_ = 0.20;
  double semantic_map_voxel_size_ = 0.30;
  double camera_time_offset_ = 0.0;
  double visual_sync_tolerance_ = 0.06;
  double continuous_correction_max_xy_rate_ = 0.0;
  double continuous_correction_max_z_rate_ = 0.0;
  double continuous_correction_max_rotation_rate_deg_ = 0.0;
  double continuous_correction_reset_gap_sec_ = 5.0;
  double last_continuous_correction_stamp_ =
      std::numeric_limits<double>::quiet_NaN();
  double correction_target_xy_ = 0.0;
  double correction_target_z_ = 0.0;
  double correction_target_rotation_deg_ = 0.0;
  double correction_applied_xy_ = 0.0;
  double correction_applied_z_ = 0.0;
  double correction_applied_rotation_deg_ = 0.0;
  double correction_lag_xy_ = 0.0;
  double correction_lag_z_ = 0.0;
  double correction_lag_rotation_deg_ = 0.0;
  double last_visual_time_difference_ = std::numeric_limits<double>::quiet_NaN();
  double latest_semantic_age_ = -1.0;
  int max_registered_points_ = 8000;
  int max_pending_registered_clouds_ = 100;
  int max_pending_semantic_clouds_ = 240;
  int max_semantic_process_per_tick_ = 2;
  int max_semantic_points_ = 6000;
  int semantic_map_max_points_ = 120000;
  int semantic_map_points_ = 0;
  int semantic_map_publications_ = 0;
  int last_semantic_map_keyframes_ = 0;
  int max_debug_pairs_ = 500;
  int trajectory_checkpoint_keyframes_ = 25;
  int last_checkpoint_keyframes_ = 0;
  int visual_tracker_accepts_ = 0;
  int visual_pnp_accepts_ = 0;
  int visual_tracker_rejections_ = 0;
  int visual_sync_drops_ = 0;
  int visual_decode_failures_ = 0;
  int visual_loop_keyframes_ = 0;
  int visual_loop_database_keyframes_ = 0;
  int visual_loop_candidates_ = 0;
  int visual_loop_detector_accepts_ = 0;
  int visual_loop_alternative_candidates_ = 0;
  int visual_loop_alternative_attempts_ = 0;
  int visual_loop_alternative_applied_ = 0;
  int visual_loop_alternative_pending_skips_ = 0;
  std::uint64_t visual_loop_global_retrieval_candidates_ = 0U;
  std::uint64_t visual_loop_global_retrieval_descriptor_matches_ = 0U;
  int visual_loop_global_retrieval_accepts_ = 0;
  int visual_loop_global_retrieval_applied_ = 0;
  int visual_loop_graph_applied_ = 0;
  int visual_loop_graph_rejections_ = 0;
  int visual_loop_graph_consensus_waits_ = 0;
  int visual_loop_pose_drops_ = 0;
  int max_pending_visual_loops_ = 24;
  int max_pending_visual_loop_retries_per_tick_ = 2;
  double visual_loop_pending_max_keyframe_lag_sec_ = 3.0;
  std::uint64_t visual_loop_pending_enqueued_ = 0U;
  std::uint64_t visual_loop_pending_retries_ = 0U;
  std::uint64_t visual_loop_pending_expired_ = 0U;
  std::uint64_t visual_loop_pending_drops_ = 0U;
  int last_visual_loop_matches_ = 0;
  int last_visual_loop_inliers_ = 0;
  int last_accepted_visual_loop_reference_id_ = -1;
  int last_accepted_visual_loop_current_id_ = -1;
  int last_accepted_visual_loop_matches_ = 0;
  int last_accepted_visual_loop_inliers_ = 0;
  int last_visual_tracks_ = 0;
  int last_visual_inliers_ = 0;
  int last_pnp_correspondences_ = 0;
  int last_pnp_inliers_ = 0;
  int last_pnp_occupied_cells_ = 0;
  std::uint64_t semantic_messages_received_ = 0U;
  std::uint64_t registered_cloud_queue_drops_ = 0U;
  std::uint64_t registered_cloud_pose_drops_ = 0U;
  std::uint64_t semantic_points_received_ = 0U;
  std::uint64_t semantic_empty_messages_ = 0U;
  std::uint64_t semantic_clouds_used_ = 0U;
  std::uint64_t semantic_points_used_ = 0U;
  std::uint64_t semantic_age_rejections_ = 0U;
  std::uint64_t semantic_queue_drops_ = 0U;
  double last_visual_quality_ = 0.0;
  double last_pnp_reprojection_rmse_ = 0.0;
  double visual_pnp_quality_sum_ = 0.0;
  double visual_pnp_inlier_ratio_sum_ = 0.0;
  double visual_pnp_reprojection_rmse_sum_ = 0.0;
  double visual_pnp_translation_sum_ = 0.0;
  double last_visual_loop_quality_ = 0.0;
  double last_visual_loop_reprojection_rmse_ =
      std::numeric_limits<double>::quiet_NaN();
  bool last_visual_loop_global_retrieval_ = false;
  int last_visual_loop_global_retrieval_votes_ = 0;
  int last_visual_loop_global_retrieval_tables_ = 0;
  double last_visual_loop_correction_translation_ =
      std::numeric_limits<double>::quiet_NaN();
  double last_visual_loop_correction_rotation_deg_ =
      std::numeric_limits<double>::quiet_NaN();
  double last_accepted_visual_loop_quality_ = 0.0;
  double last_accepted_visual_loop_reprojection_rmse_ =
      std::numeric_limits<double>::quiet_NaN();
  double last_accepted_visual_loop_time_separation_sec_ =
      std::numeric_limits<double>::quiet_NaN();
  double last_accepted_visual_loop_xy_separation_ =
      std::numeric_limits<double>::quiet_NaN();
  double last_accepted_visual_loop_z_separation_ =
      std::numeric_limits<double>::quiet_NaN();
  std::string last_visual_reason_ = "not_received";
  std::string last_visual_loop_reason_ = "not_received";

  SemanticPoseGraphOptions options_;
  VisualRotationTrackerOptions visual_options_;
  VisualLoopDetectorOptions visual_loop_options_;
  std::unique_ptr<SemanticPoseGraph> graph_;
  std::unique_ptr<VisualRotationTracker> visual_tracker_;
  std::unique_ptr<VisualLoopDetector> visual_loop_detector_;
  OdomDeque odom_history_;
  std::deque<sensor_msgs::PointCloud2ConstPtr> pending_registered_clouds_;
  std::deque<PendingSemanticCloud> pending_semantic_clouds_;
  std::deque<VisualImageSample> visual_image_queue_;
  std::deque<VisualCloudSample, Eigen::aligned_allocator<VisualCloudSample>>
      visual_cloud_queue_;
  PendingVisualLoopDeque pending_visual_loops_;
  Eigen::Isometry3d target_map_from_odom_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d applied_map_from_odom_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_corrected_pose_ = Eigen::Isometry3d::Identity();
  ros::Time last_stamp_;
  ros::WallTime last_semantic_map_publish_wall_;
};

}  // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "semantic_gtsam_pose_graph_node");
  try
  {
    SemanticGtsamPoseGraphNode node;
    node.spin();
  }
  catch (const std::exception &exception)
  {
    ROS_FATAL("[semantic_gtsam] fatal exception: %s", exception.what());
    return 1;
  }
  return 0;
}
