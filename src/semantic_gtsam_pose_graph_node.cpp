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
    cloud_sub_ = nh_.subscribe(registered_cloud_topic_, 100,
                               &SemanticGtsamPoseGraphNode::cloudCallback, this);
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
    private_nh_.param("max_semantic_points", max_semantic_points_, 6000);
    private_nh_.param("max_debug_pairs", max_debug_pairs_, 500);
    private_nh_.param("path_publish_rate", path_publish_rate_, 0.5);
    private_nh_.param("trajectory_checkpoint_keyframes", trajectory_checkpoint_keyframes_, 25);
    private_nh_.param("camera_time_offset", camera_time_offset_, 0.0);
    private_nh_.param("visual_sync_tolerance", visual_sync_tolerance_, 0.06);
    private_nh_.param("semantic_map_publish_rate", semantic_map_publish_rate_, 0.20);
    private_nh_.param("semantic_map_voxel_size", semantic_map_voxel_size_, 0.30);
    private_nh_.param("semantic_map_max_points", semantic_map_max_points_, 120000);

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
                      options_.enable_semantic_observation_factors, true);
    private_nh_.param("graph/enable_semantic_observation_xy_factors",
                      options_.enable_semantic_observation_xy_factors, true);
    private_nh_.param("graph/enable_semantic_observation_z_factors",
                      options_.enable_semantic_observation_z_factors, true);
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
                      options_.semantic_observation_max_index_gap, 30);
    private_nh_.param("graph/semantic_observation_max_reference_uses",
                      options_.semantic_observation_max_reference_uses, 0);
    private_nh_.param("graph/semantic_observation_interval",
                      options_.semantic_observation_interval, 1);
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
    private_nh_.param("graph/loop_max_candidates", options_.loop_max_candidates, 6);
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
    private_nh_.param("graph/loop_huber_k", options_.loop_huber_k, 1.345);
    private_nh_.param("graph/sequential_ground_huber_k", options_.sequential_ground_huber_k, 1.345);
    private_nh_.param("graph/wheel_speed_scale", options_.wheel_speed_scale, 0.9865);
    private_nh_.param("graph/wheel_max_gap", options_.wheel_max_gap, 0.08);
    private_nh_.param("graph/wheel_sigma_base", options_.wheel_sigma_base, 0.08);
    private_nh_.param("graph/wheel_sigma_per_meter", options_.wheel_sigma_per_meter, 0.025);
    private_nh_.param("graph/wheel_lateral_sigma", options_.wheel_lateral_sigma, 0.15);
    private_nh_.param("graph/wheel_huber_k", options_.wheel_huber_k, 1.345);
    private_nh_.param("graph/wheel_min_samples", options_.wheel_min_samples, 5);
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
    private_nh_.param("graph/visual_loop_min_quality",
                      options_.visual_loop_min_quality, 0.40);
    private_nh_.param("graph/visual_loop_max_translation_disagreement",
                      options_.visual_loop_max_translation_disagreement, 3.0);
    private_nh_.param("graph/visual_loop_max_rotation_disagreement_deg",
                      options_.visual_loop_max_rotation_disagreement_deg, 10.0);
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
                      visual_loop_options_.maximum_database_size, 1000);
    private_nh_.param("visual_loop/keyframe_distance",
                      visual_loop_options_.keyframe_distance, 0.75);
    private_nh_.param("visual_loop/keyframe_interval_sec",
                      visual_loop_options_.keyframe_interval_sec, 1.0);
    private_nh_.param("visual_loop/minimum_index_gap",
                      visual_loop_options_.minimum_index_gap, 25);
    private_nh_.param("visual_loop/search_radius",
                      visual_loop_options_.search_radius, 25.0);
    private_nh_.param("visual_loop/maximum_yaw_difference_deg",
                      visual_loop_options_.maximum_yaw_difference_deg, 70.0);
    private_nh_.param("visual_loop/maximum_retrieval_candidates",
                      visual_loop_options_.maximum_retrieval_candidates, 40);
    private_nh_.param("visual_loop/maximum_geometric_candidates",
                      visual_loop_options_.maximum_geometric_candidates, 5);
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
    const Eigen::Isometry3d optimized = graph_->correctedPose(sample.pose);
    nav_msgs::Odometry output = *message;
    output.header.frame_id = map_frame_;
    output.child_frame_id = body_frame_;
    output.pose.pose = poseMessage(optimized);
    odom_pub_.publish(output);
    last_corrected_pose_ = optimized;
    last_stamp_ = message->header.stamp.isZero() ? ros::Time(stamp) : message->header.stamp;
    have_output_ = true;
    if (broadcast_tf_)
    {
      geometry_msgs::TransformStamped transform;
      transform.header.stamp = last_stamp_;
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
    processPendingClouds();
  }

  void semanticCallback(const sensor_msgs::PointCloud2ConstPtr &message)
  {
    SemanticGraphPointVector semantic_points = decodeCloud(
        *message, max_semantic_points_, true, label_field_, confidence_field_);
    const double stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                       : message->header.stamp.toSec();
    ++semantic_messages_received_;
    semantic_points_received_ += static_cast<std::uint64_t>(semantic_points.size());
    if (semantic_points.empty())
    {
      ++semantic_empty_messages_;
      return;
    }
    OdomSample odom;
    if (!lookupOdom(stamp, &odom))
    {
      ++semantic_age_rejections_;
      ROS_WARN_THROTTLE(5.0,
          "[semantic_gtsam] semantic observation has no frontend pose near %.6f", stamp);
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
    if (graph_->addSemanticObservation(stamp, odom.pose, semantic_points))
    {
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
    graph_->addWheelSample(stamp, 0.5 * (speeds[1] + speeds[2]));
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
    if (result.candidate_found) ++visual_loop_candidates_;
    last_visual_loop_matches_ = result.descriptor_matches;
    last_visual_loop_inliers_ = result.pnp_inliers;
    last_visual_loop_quality_ = result.quality;
    last_visual_loop_reprojection_rmse_ = result.reprojection_rmse;
    if (!result.debug_image.empty() && publish_visual_debug_images_)
    {
      visual_loop_debug_pub_.publish(imageMessage(
          result.debug_image, result.current_stamp, body_frame_));
    }
    if (!result.accepted) return;
    ++visual_loop_detector_accepts_;
    if (graph_->addVisualLoopConstraint(
            result.reference_stamp, result.current_stamp,
            result.reference_from_current, result.quality))
    {
      ++visual_loop_graph_applied_;
      last_visual_loop_reason_ = "visual_loop_applied";
      publishStats();
    }
    else
    {
      ++visual_loop_graph_rejections_;
      last_visual_loop_reason_ = "visual_loop_graph_rejected";
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

    const Eigen::Isometry3d corrected = graph_->correctedPose(odom.pose);
    const Eigen::Isometry3d correction = corrected * odom.pose.inverse();
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
           << ",\"wheel_factors\":" << stats.wheel_factors
           << ",\"visual_rotation_factors\":" << stats.visual_rotation_factors
           << ",\"visual_translation_factors\":" << stats.visual_translation_factors
           << ",\"visual_rotation_rejections\":" << stats.visual_rotation_rejections
           << ",\"visual_loop_attempts\":" << stats.visual_loop_attempts
           << ",\"visual_loop_rejections\":" << stats.visual_loop_rejections
           << ",\"visual_loop_factors\":" << stats.visual_loop_factors
           << ",\"visual_loop_keyframes\":" << visual_loop_keyframes_
           << ",\"visual_loop_candidates\":" << visual_loop_candidates_
           << ",\"visual_loop_detector_accepts\":" << visual_loop_detector_accepts_
           << ",\"visual_loop_graph_applied\":" << visual_loop_graph_applied_
           << ",\"visual_loop_graph_rejections\":" << visual_loop_graph_rejections_
           << ",\"visual_loop_pose_drops\":" << visual_loop_pose_drops_
           << ",\"visual_loop_matches\":" << last_visual_loop_matches_
           << ",\"visual_loop_inliers\":" << last_visual_loop_inliers_
           << ",\"visual_loop_quality\":" << metric(last_visual_loop_quality_)
           << ",\"visual_loop_reprojection_rmse\":"
           << metric(last_visual_loop_reprojection_rmse_)
           << ",\"visual_loop_reason\":\"" << last_visual_loop_reason_ << "\""
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
  bool visual_observation_only_ = true;
  bool publish_visual_debug_images_ = true;
  bool semantic_cloud_in_map_frame_ = true;
  bool broadcast_tf_ = true;
  bool save_on_shutdown_ = true;
  bool have_output_ = false;
  double max_pose_lookup_dt_ = 0.15;
  double odom_history_sec_ = 120.0;
  double path_publish_rate_ = 0.5;
  double semantic_map_publish_rate_ = 0.20;
  double semantic_map_voxel_size_ = 0.30;
  double camera_time_offset_ = 0.0;
  double visual_sync_tolerance_ = 0.06;
  double last_visual_time_difference_ = std::numeric_limits<double>::quiet_NaN();
  double latest_semantic_age_ = -1.0;
  int max_registered_points_ = 8000;
  int max_pending_registered_clouds_ = 100;
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
  int visual_loop_candidates_ = 0;
  int visual_loop_detector_accepts_ = 0;
  int visual_loop_graph_applied_ = 0;
  int visual_loop_graph_rejections_ = 0;
  int visual_loop_pose_drops_ = 0;
  int last_visual_loop_matches_ = 0;
  int last_visual_loop_inliers_ = 0;
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
  double last_visual_quality_ = 0.0;
  double last_pnp_reprojection_rmse_ = 0.0;
  double visual_pnp_quality_sum_ = 0.0;
  double visual_pnp_inlier_ratio_sum_ = 0.0;
  double visual_pnp_reprojection_rmse_sum_ = 0.0;
  double visual_pnp_translation_sum_ = 0.0;
  double last_visual_loop_quality_ = 0.0;
  double last_visual_loop_reprojection_rmse_ =
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
  std::deque<VisualImageSample> visual_image_queue_;
  std::deque<VisualCloudSample, Eigen::aligned_allocator<VisualCloudSample>>
      visual_cloud_queue_;
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
