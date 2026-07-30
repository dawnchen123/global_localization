#include "hybrid_localization/core.h"
#include "hybrid_localization/i2nav_ranger_odometer.h"
#include "hybrid_localization/lidar_odometry.h"
#include "hybrid_localization/sparse_visual_map.h"

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <tf2_ros/transform_broadcaster.h>

#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
using hybrid_localization::BevPoint;
using hybrid_localization::convertSemanticLabel;
using hybrid_localization::FactorType;
using hybrid_localization::ImuSample;
using hybrid_localization::LidarOdometry;
using hybrid_localization::LidarOdometryOptions;
using hybrid_localization::LidarOdometryResult;
using hybrid_localization::MapMatchResult;
using hybrid_localization::MatchPair;
using hybrid_localization::PoseState;
using hybrid_localization::PriorMap;
using hybrid_localization::PriorMatcher;
using hybrid_localization::PointVector;
using hybrid_localization::SlidingWindowOptimizer;
using hybrid_localization::SparseVisualFrame;
using hybrid_localization::SparseVisualMap;
using hybrid_localization::SparseVisualMapOptions;
using hybrid_localization::SparseVisualMapStats;
using hybrid_localization::TimedPoint;
using hybrid_localization::TimedPointVector;
using hybrid_localization::VisualPoseLinearization;
using hybrid_localization::WheelSample;
using hybrid_localization::VisualUpdateResult;

constexpr double kPi = 3.14159265358979323846;

struct CloudFieldView
{
  const sensor_msgs::PointField *x = nullptr;
  const sensor_msgs::PointField *y = nullptr;
  const sensor_msgs::PointField *z = nullptr;
  const sensor_msgs::PointField *label = nullptr;
  const sensor_msgs::PointField *confidence = nullptr;
  const sensor_msgs::PointField *instance = nullptr;
  const sensor_msgs::PointField *time = nullptr;
};

struct CloudPoint
{
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  uint8_t label = 0U;
  float confidence = 1.0F;
  int instance = 0;
  double point_time = 0.0;
  bool has_point_time = false;
};

struct LocalFrame
{
  double stamp = 0.0;
  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> points;
};

struct ObjectState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = 0;
  uint8_t label = 0U;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  int observations = 0;
  double last_stamp = 0.0;
  double reliability = 0.0;
  bool reachable = false;
  bool dynamic = false;
};

enum class MeasurementEventType
{
  LIDAR = 0,
  IMAGE = 1
};

struct MeasurementEvent
{
  double stamp = 0.0;
  std::uint64_t sequence = 0U;
  MeasurementEventType type = MeasurementEventType::LIDAR;
  sensor_msgs::PointCloud2ConstPtr lidar;
  sensor_msgs::CompressedImageConstPtr image;
};

struct CachedCameraFrame
{
  double stamp = 0.0;
  cv::Mat gray;
};

const sensor_msgs::PointField *findField(const sensor_msgs::PointCloud2 &cloud,
                                         const std::vector<std::string> &names)
{
  for (const std::string &name : names)
  {
    for (const sensor_msgs::PointField &field : cloud.fields)
    {
      if (field.name == name)
      {
        return &field;
      }
    }
  }
  return nullptr;
}

double readField(const uint8_t *data, const sensor_msgs::PointField *field)
{
  if (field == nullptr)
  {
    return 0.0;
  }
  const uint8_t *value = data + field->offset;
  switch (field->datatype)
  {
    case sensor_msgs::PointField::INT8:
      return static_cast<double>(*reinterpret_cast<const int8_t *>(value));
    case sensor_msgs::PointField::UINT8:
      return static_cast<double>(*reinterpret_cast<const uint8_t *>(value));
    case sensor_msgs::PointField::INT16:
    {
      int16_t v = 0;
      std::memcpy(&v, value, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT16:
    {
      uint16_t v = 0;
      std::memcpy(&v, value, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::INT32:
    {
      int32_t v = 0;
      std::memcpy(&v, value, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT32:
    {
      uint32_t v = 0;
      std::memcpy(&v, value, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::FLOAT32:
    {
      float v = 0.0F;
      std::memcpy(&v, value, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::FLOAT64:
    {
      double v = 0.0;
      std::memcpy(&v, value, sizeof(v));
      return v;
    }
    default:
      return 0.0;
  }
}

CloudFieldView makeFieldView(const sensor_msgs::PointCloud2 &cloud, const std::string &label_field)
{
  CloudFieldView view;
  view.x = findField(cloud, {"x"});
  view.y = findField(cloud, {"y"});
  view.z = findField(cloud, {"z"});
  view.label = findField(cloud, {label_field, "label", "semantic", "class_id", "label_id"});
  view.confidence = findField(cloud, {"confidence", "probability", "prob", "score"});
  view.instance = findField(cloud, {"instance_id", "instance", "object_id"});
  view.time = findField(cloud, {"offset_time", "time", "t", "timestamp"});
  return view;
}

std::vector<CloudPoint> decodeCloud(const sensor_msgs::PointCloud2 &cloud, const std::string &label_field,
                                    int max_points)
{
  const CloudFieldView fields = makeFieldView(cloud, label_field);
  std::vector<CloudPoint> points;
  if (fields.x == nullptr || fields.y == nullptr || fields.z == nullptr || cloud.point_step == 0)
  {
    return points;
  }
  const std::size_t total = static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height);
  const std::size_t output_count = std::min<std::size_t>(
      total, static_cast<std::size_t>(std::max(1, max_points)));
  points.reserve(output_count);
  for (std::size_t output_index = 0; output_index < output_count; ++output_index)
  {
    const std::size_t i = std::min(total - 1U,
        static_cast<std::size_t>(std::floor(
            static_cast<double>(output_index) * static_cast<double>(total) /
            static_cast<double>(output_count))));
    const uint8_t *data = cloud.data.data() + i * cloud.point_step;
    CloudPoint point;
    point.point = Eigen::Vector3d(readField(data, fields.x), readField(data, fields.y),
                                  readField(data, fields.z));
    if (!point.point.allFinite())
    {
      continue;
    }
    if (fields.label != nullptr)
    {
      point.label = static_cast<uint8_t>(std::max(0.0, std::min(255.0, readField(data, fields.label))));
    }
    if (fields.confidence != nullptr)
    {
      point.confidence = static_cast<float>(std::max(0.0, std::min(1.0, readField(data, fields.confidence))));
    }
    if (fields.instance != nullptr)
    {
      point.instance = static_cast<int>(readField(data, fields.instance));
    }
    if (fields.time != nullptr)
    {
      point.point_time = readField(data, fields.time);
      point.has_point_time = std::isfinite(point.point_time);
    }
    points.push_back(point);
  }
  return points;
}

geometry_msgs::Pose poseMessage(const Eigen::Isometry3d &pose)
{
  geometry_msgs::Pose result;
  const Eigen::Isometry3d normalized_pose = hybrid_localization::projectToSE3(pose);
  const Eigen::Quaterniond q(normalized_pose.rotation());
  result.position.x = normalized_pose.translation().x();
  result.position.y = normalized_pose.translation().y();
  result.position.z = normalized_pose.translation().z();
  result.orientation.x = q.x();
  result.orientation.y = q.y();
  result.orientation.z = q.z();
  result.orientation.w = q.w();
  return result;
}

Eigen::Isometry3d poseFromMessage(const geometry_msgs::Pose &pose)
{
  Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  if (q.norm() < 1e-8)
  {
    q = Eigen::Quaterniond::Identity();
  }
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = q.normalized().toRotationMatrix();
  result.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return result;
}

sensor_msgs::Image visualDebugImageMessage(const ros::Time &stamp,
                                           const std::string &frame_id,
                                           const cv::Mat &source)
{
  sensor_msgs::Image message;
  message.header.stamp = stamp;
  message.header.frame_id = frame_id;
  if (source.empty()) return message;

  cv::Mat bgr;
  if (source.type() == CV_8UC3)
  {
    bgr = source;
  }
  else if (source.channels() == 1)
  {
    cv::Mat gray;
    if (source.depth() == CV_8U) gray = source;
    else source.convertTo(gray, CV_8U);
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
  }
  else if (source.channels() == 4)
  {
    cv::cvtColor(source, bgr, cv::COLOR_BGRA2BGR);
  }
  else
  {
    return message;
  }
  if (!bgr.isContinuous()) bgr = bgr.clone();
  message.height = static_cast<uint32_t>(bgr.rows);
  message.width = static_cast<uint32_t>(bgr.cols);
  message.encoding = "bgr8";
  message.is_bigendian = false;
  message.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
  message.data.assign(bgr.datastart, bgr.dataend);
  return message;
}

cv::Mat mono8Image(const sensor_msgs::Image &message)
{
  if (message.height == 0U || message.width == 0U ||
      message.step < message.width ||
      message.data.size() < static_cast<std::size_t>(message.step) *
                              static_cast<std::size_t>(message.height))
  {
    return cv::Mat();
  }
  if (message.encoding != "mono8" && message.encoding != "8UC1" &&
      message.encoding != "8uc1")
  {
    return cv::Mat();
  }
  cv::Mat image(static_cast<int>(message.height),
                static_cast<int>(message.width), CV_8U);
  for (uint32_t row = 0U; row < message.height; ++row)
  {
    std::memcpy(image.ptr(static_cast<int>(row)),
                message.data.data() + static_cast<std::size_t>(row) *
                                          message.step,
                static_cast<std::size_t>(message.width));
  }
  return image;
}

Eigen::Matrix<double, 6, 6> poseInformationFromCovariance(const boost::array<double, 36> &covariance,
                                                          double default_scale)
{
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity() * default_scale;
  Eigen::Matrix<double, 6, 6> covariance_matrix = Eigen::Matrix<double, 6, 6>::Zero();
  bool valid = false;
  for (int row = 0; row < 6; ++row)
  {
    for (int col = 0; col < 6; ++col)
    {
      covariance_matrix(row, col) = covariance[static_cast<std::size_t>(row * 6 + col)];
      valid = valid || std::isfinite(covariance_matrix(row, col));
    }
  }
  if (valid && covariance_matrix.diagonal().array().isFinite().all() &&
      covariance_matrix.diagonal().array().abs().maxCoeff() > 1e-12)
  {
    for (int i = 0; i < 6; ++i)
    {
      covariance_matrix(i, i) = std::max(1e-6, std::abs(covariance_matrix(i, i)));
    }
    information = covariance_matrix.inverse();
  }
  return information;
}

Eigen::Vector3d covariancePosition(const boost::array<double, 36> &covariance)
{
  Eigen::Vector3d result;
  result.x() = std::max(1e-4, std::abs(covariance[0]));
  result.y() = std::max(1e-4, std::abs(covariance[7]));
  result.z() = std::max(1e-4, std::abs(covariance[14]));
  return result;
}

std_msgs::ColorRGBA colorForState(const std::string &state)
{
  std_msgs::ColorRGBA color;
  color.a = 0.95F;
  if (state == "candidate")
  {
    color.r = 1.0F; color.g = 0.55F; color.b = 0.05F;
  }
  else if (state == "inlier")
  {
    color.r = 0.05F; color.g = 0.9F; color.b = 0.95F;
  }
  else if (state == "outlier")
  {
    color.r = 0.55F; color.g = 0.55F; color.b = 0.55F;
  }
  else
  {
    color.r = 0.1F; color.g = 1.0F; color.b = 0.2F;
  }
  return color;
}

class HybridLocalizationNode
{
public:
  HybridLocalizationNode()
      : nh_(), private_nh_("~"), imu_nh_(nh_), lidar_nh_(nh_),
        camera_nh_(nh_), optimizer_(12)
  {
    loadParameters();
    if (save_on_shutdown_ && !trajectory_save_path_.empty())
    {
      trajectory_stream_.open(trajectory_save_path_.c_str(), std::ios::out | std::ios::trunc);
      if (trajectory_stream_.is_open())
      {
        trajectory_stream_ << "timestamp,x,y,z,qx,qy,qz,qw\n";
        trajectory_stream_.flush();
      }
      else
      {
        ROS_ERROR("[hybrid_localization] cannot open trajectory output: %s",
                  trajectory_save_path_.c_str());
      }
    }
    configureLidarOdometry();
    configureVisualMap();
    configureMatcher();
    initializeRos();
    loadPriorPointCloud();
    ROS_INFO("[hybrid_localization] standalone C++ LiDAR/IMU localization ready: raw_cloud=%s imu=%s camera=%s semantic=%s",
             lidar_cloud_topic_.c_str(), imu_topic_.c_str(),
             subscribe_camera_frontend_ ? camera_topic_.c_str() : "disabled",
             semantic_cloud_topic_.c_str());
  }

  ~HybridLocalizationNode();

  void spin()
  {
    ros::AsyncSpinner imu_spinner(1, &imu_callback_queue_);
    ros::AsyncSpinner lidar_spinner(1, &lidar_callback_queue_);
    ros::AsyncSpinner camera_spinner(1, &camera_callback_queue_);
    imu_spinner.start();
    if (measurement_scheduler_enabled_ && subscribe_lidar_)
      lidar_spinner.start();
    if (measurement_scheduler_enabled_ && subscribe_camera_frontend_)
      camera_spinner.start();
    ros::spin();
    if (measurement_scheduler_enabled_ && subscribe_camera_frontend_)
      camera_spinner.stop();
    if (measurement_scheduler_enabled_ && subscribe_lidar_)
      lidar_spinner.stop();
    imu_spinner.stop();
  }

private:
  void loadParameters();
  void configureLidarOdometry();
  void configureVisualMap();
  void configureMatcher();
  void initializeRos();
  void loadPriorPointCloud();
  void imuCallback(const sensor_msgs::ImuConstPtr &message);
  void drainPendingImuSamples(
      double up_to_stamp = std::numeric_limits<double>::infinity());
  double latestReceivedImuStamp() const;
  std::size_t pendingImuSampleCount() const;
  std::uint64_t pendingImuQueueDrops() const;
  void wheelCallback(const nav_msgs::OdometryConstPtr &message);
  void rangerWheelCallback(const insprobe_msgs::RangerOdometerConstPtr &message);
  void lidarCallback(const sensor_msgs::PointCloud2ConstPtr &message);
  void cameraCallback(const sensor_msgs::CompressedImageConstPtr &message);
  void sam3CameraLabelCallback(const sensor_msgs::ImageConstPtr &message);
  void drainPendingSam3CameraLabels();
  void processSam3CameraLabel(const sensor_msgs::ImageConstPtr &message);
  void enqueueMeasurement(MeasurementEvent event, bool process_queue = true);
  void drainMeasurementQueue(
      std::size_t max_events = std::numeric_limits<std::size_t>::max());
  void measurementSchedulerTimer(const ros::WallTimerEvent &event);
  cv::Mat semanticFlowGray(const cv::Mat &image) const;
  bool propagateSemanticLabels(const cv::Mat &current_gray,
                               double current_stamp);
  void cacheCameraFrame(const cv::Mat &gray, double stamp);
  void processImageMessage(const sensor_msgs::CompressedImageConstPtr &message,
                           double corrected_stamp);
  void processLidarMessage(const sensor_msgs::PointCloud2ConstPtr &message);
  void semanticCallback(const sensor_msgs::PointCloud2ConstPtr &message);
  void visualFactorCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &message);
  void priorGridCallback(const nav_msgs::OccupancyGridConstPtr &message);
  void priorCloudCallback(const sensor_msgs::PointCloud2ConstPtr &message);
  void mapMatchTimer(const ros::TimerEvent &event);
  void publishTimer(const ros::TimerEvent &event);

  void handleLidarOdometry(const LidarOdometryResult &result);
  void handleVisualUpdate(const VisualUpdateResult &result);
  void processSemanticCloud(const sensor_msgs::PointCloud2ConstPtr &message);
  Eigen::Isometry3d lookupOdomPose(double stamp, bool *ok) const;
  uint8_t normalizeLabel(uint8_t label) const;
  void appendLocalFrame(double stamp,
                        const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points,
                        bool semantic);
  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> collectLocalPoints() const;
  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> collectSemanticPoints() const;
  void addSemanticObjects(double stamp, const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points);
  void buildPriorFromPoints(const std::vector<CloudPoint> &points);
  void buildPriorFromGrid(const nav_msgs::OccupancyGrid &grid);
  void optimizeCurrentWindow();
  void applyMapMatch(const MapMatchResult &result);
  void updatePriorMap(const MapMatchResult &result);

  nav_msgs::Odometry makeOutputOdometry(const ros::Time &stamp, const Eigen::Isometry3d &pose) const;
  void publishCurrentState(const ros::Time &stamp);
  void publishLocalBev(const ros::Time &stamp);
  void publishObjects(const ros::Time &stamp);
  void publishMapDebug(const ros::Time &stamp, const MapMatchResult &result);
  void publishStatus(const ros::Time &stamp);
  sensor_msgs::PointCloud2 makeSemanticCloudMessage(const ros::Time &stamp) const;
  sensor_msgs::PointCloud2 makeSemanticPointMessage(
      const ros::Time &stamp,
      const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points) const;
  sensor_msgs::PointCloud2 makeRegisteredCloudMessage(
      const ros::Time &stamp, const PointVector &body_points,
      const Eigen::Isometry3d &global_pose) const;
  nav_msgs::OccupancyGrid makePriorGridMessage(const ros::Time &stamp) const;
  nav_msgs::OccupancyGrid makeLocalGridMessage(const ros::Time &stamp) const;
  static void appendPoseToPath(nav_msgs::Path &path, const nav_msgs::Odometry &odom, std::size_t max_poses);

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::NodeHandle imu_nh_;
  ros::NodeHandle lidar_nh_;
  ros::NodeHandle camera_nh_;
  ros::CallbackQueue imu_callback_queue_;
  ros::CallbackQueue lidar_callback_queue_;
  ros::CallbackQueue camera_callback_queue_;
  ros::Subscriber imu_sub_;
  ros::Subscriber wheel_sub_;
  ros::Subscriber lidar_sub_;
  ros::Subscriber camera_sub_;
  ros::Subscriber sam3_camera_label_sub_;
  ros::Subscriber semantic_sub_;
  ros::Subscriber visual_factor_sub_;
  ros::Subscriber prior_grid_sub_;
  ros::Subscriber prior_cloud_sub_;
  ros::Publisher output_odom_pub_;
  ros::Publisher output_path_pub_;
  ros::Publisher local_bev_pub_;
  ros::Publisher prior_map_pub_;
  ros::Publisher object_pose_pub_;
  ros::Publisher object_marker_pub_;
  ros::Publisher object_array_pub_;
  ros::Publisher semantic_cloud_pub_;
  ros::Publisher semantic_observation_pub_;
  ros::Publisher registered_cloud_pub_;
  ros::Publisher status_pub_;
  ros::Publisher quality_pub_;
  ros::Publisher debug_marker_pub_;
  ros::Publisher diagnostic_pub_;
  ros::Publisher visual_direct_debug_pub_;
  ros::Timer map_match_timer_;
  ros::Timer publish_timer_;
  ros::WallTimer measurement_scheduler_timer_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  std::string imu_topic_;
  std::string wheel_topic_;
  std::string wheel_message_type_;
  std::string lidar_cloud_topic_;
  std::string camera_topic_;
  std::string sam3_camera_label_topic_;
  std::string semantic_cloud_topic_;
  std::string visual_factor_topic_;
  std::string prior_grid_topic_;
  std::string prior_cloud_topic_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string body_frame_;
  std::string lidar_frame_;
  std::string output_odom_topic_;
  std::string output_path_topic_;
  std::string local_bev_topic_;
  std::string prior_map_topic_;
  std::string semantic_output_topic_;
  std::string semantic_observation_topic_;
  std::string registered_cloud_topic_;
  std::string object_pose_topic_;
  std::string object_marker_topic_;
  std::string object_array_topic_;
  std::string status_topic_;
  std::string quality_topic_;
  std::string debug_marker_topic_;
  std::string diagnostics_topic_;
  std::string visual_direct_debug_topic_;
  std::string label_field_;
  std::string lidar_label_mode_;
  std::string semantic_label_mode_;
  std::string lidar_point_time_mode_;
  std::string prior_pcd_path_;
  std::string trajectory_save_path_;
  std::string object_save_path_;

  bool subscribe_imu_ = true;
  bool subscribe_wheel_ = false;
  bool subscribe_lidar_ = true;
  bool subscribe_camera_frontend_ = false;
  bool subscribe_semantic_ = true;
  bool subscribe_visual_factor_ = false;
  bool subscribe_prior_grid_ = true;
  bool subscribe_prior_cloud_ = false;
  bool semantic_cloud_in_map_frame_ = false;
  bool visual_pose_is_absolute_ = true;
  bool broadcast_tf_ = true;
  bool publish_debug_ = true;
  bool publish_local_bev_ = true;
  bool publish_objects_ = true;
  bool prior_update_enabled_ = false;
  bool save_on_shutdown_ = true;
  bool include_dynamic_objects_ = false;
  bool lidar_deskew_enabled_ = true;
  bool lidar_stamp_is_end_ = true;
  bool measurement_scheduler_enabled_ = true;
  bool visual_frontend_enabled_ = false;
  bool visual_observation_only_ = true;
  bool publish_visual_direct_debug_ = true;
  bool visual_use_sam3_ = false;

  int max_lidar_points_ = 10000;
  int max_semantic_points_ = 10000;
  int max_local_points_ = 18000;
  int max_local_semantic_points_ = 60000;
  int max_semantic_output_points_ = 12000;
  int max_objects_ = 1000;
  int dynamic_label_ = 5;
  int traversable_label_max_ = 2;
  int keyframe_max_states_ = 12;
  int optimizer_iterations_ = 3;
  double optimizer_max_step_norm_ = 0.5;
  double keyframe_distance_ = 0.8;
  double keyframe_angle_deg_ = 8.0;
  double keyframe_interval_sec_ = 0.5;
  double max_odom_lookup_dt_ = 0.15;
  double local_map_window_sec_ = 12.0;
  double local_map_resolution_ = 0.25;
  double local_map_size_m_ = 80.0;
  double local_ground_z_ = -2.0;
  double local_max_z_ = 4.0;
  double local_min_range_ = 0.5;
  double local_max_range_ = 80.0;
  double map_match_rate_ = 1.0;
  double map_match_min_points_ = 100.0;
  double map_factor_huber_delta_ = 1.0;
  double map_factor_information_ = 2.0;
  double map_factor_min_confidence_ = 0.45;
  double prior_resolution_ = 0.25;
  double prior_z_min_ = -3.0;
  double prior_z_max_ = 4.0;
  double prior_update_alpha_ = 0.02;
  double prior_update_min_confidence_ = 0.7;
  double object_assoc_radius_ = 1.5;
  double object_min_confidence_ = 0.45;
  double object_reachable_radius_ = 8.0;
  double publish_rate_ = 10.0;
  double status_rate_ = 1.0;
  double lidar_point_time_scale_ = 1.0;
  double lidar_imu_wait_sec_ = 0.0;
  double camera_time_offset_ = 0.0;
  double camera_imu_wait_sec_ = 0.01;
  double camera_observation_interval_sec_ = 0.20;
  double measurement_reorder_window_sec_ = 0.02;
  double measurement_scheduler_process_rate_hz_ = 200.0;
  int measurement_scheduler_max_events_per_tick_ = 2;
  double last_processed_measurement_stamp_ =
      -std::numeric_limits<double>::infinity();
  double latest_enqueued_lidar_stamp_ =
      -std::numeric_limits<double>::infinity();
  double latest_enqueued_image_stamp_ =
      -std::numeric_limits<double>::infinity();
  double last_camera_stamp_ = -std::numeric_limits<double>::infinity();
  int max_measurement_queue_ = 200;
  int sam3_max_cached_frames_ = 80;
  int sam3_max_replay_frames_ = 40;
  double sam3_flow_scale_ = 0.5;
  double sam3_cache_duration_sec_ = 6.0;
  double sam3_label_sync_tolerance_sec_ = 0.08;
  double sam3_max_source_age_sec_ = 3.0;
  double sam3_max_flow_pixels_ = 80.0;
  double last_wheel_speed_ = 0.0;
  int wheel_samples_ = 0;
  Eigen::Isometry3d body_from_lidar_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d body_from_imu_ = Eigen::Isometry3d::Identity();

  SlidingWindowOptimizer optimizer_;
  LidarOdometry lidar_odometry_;
  SparseVisualMap sparse_visual_map_;
  PriorMatcher matcher_{hybrid_localization::MatcherOptions()};
  PriorMap prior_map_;
  hybrid_localization::BevGrid local_bev_;
  std::deque<LocalFrame> local_frames_;
  std::deque<LocalFrame> semantic_frames_;
  std::deque<std::pair<double, Eigen::Isometry3d>> odom_history_;
  mutable std::mutex measurement_mutex_;
  std::deque<MeasurementEvent> measurement_queue_;
  mutable std::mutex pending_imu_mutex_;
  std::deque<ImuSample, Eigen::aligned_allocator<ImuSample>> pending_imu_samples_;
  mutable std::mutex pending_sam3_label_mutex_;
  std::deque<sensor_msgs::ImageConstPtr> pending_sam3_labels_;
  std::deque<CachedCameraFrame> camera_frame_cache_;
  std::deque<geometry_msgs::PoseWithCovarianceStamped> visual_factor_queue_;
  std::vector<ObjectState, Eigen::aligned_allocator<ObjectState>> objects_;
  nav_msgs::Path output_path_;
  PoseState latest_state_;
  Eigen::Isometry3d latest_odom_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_keyframe_odom_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d global_from_odom_ = Eigen::Isometry3d::Identity();
  Eigen::Matrix3d position_covariance_ = Eigen::Matrix3d::Identity() * 0.1;
  Eigen::Isometry3d latest_wheel_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_keyframe_wheel_pose_ = Eigen::Isometry3d::Identity();
  bool have_odom_ = false;
  bool have_wheel_ = false;
  bool have_keyframe_ = false;
  double last_odom_stamp_ = 0.0;
  double last_imu_stamp_ = 0.0;
  double last_keyframe_stamp_ = 0.0;
  double last_map_match_stamp_ = -std::numeric_limits<double>::infinity();
  double last_semantic_stamp_ = -std::numeric_limits<double>::infinity();
  double map_match_confidence_ = 0.0;
  double map_match_score_ = 0.0;
  double last_score_gap_ = 0.0;
  double last_inlier_ratio_ = 0.0;
  int map_match_attempts_ = 0;
  int map_match_accepts_ = 0;
  int factor_count_ = 0;
  int dropped_clouds_ = 0;
  std::uint64_t next_measurement_sequence_ = 0U;
  std::uint64_t scheduled_lidar_events_ = 0U;
  std::uint64_t scheduled_image_events_ = 0U;
  std::uint64_t processed_image_events_ = 0U;
  std::uint64_t measurement_queue_drops_ = 0U;
  std::uint64_t measurement_image_queue_drops_ = 0U;
  std::uint64_t camera_observation_interval_drops_ = 0U;
  std::uint64_t measurement_stale_drops_ = 0U;
  std::uint64_t camera_decode_failures_ = 0U;
  std::uint64_t pending_imu_queue_drops_ = 0U;
  double latest_received_imu_stamp_ =
      -std::numeric_limits<double>::infinity();
  std::uint64_t visual_update_attempts_ = 0U;
  std::uint64_t visual_update_accepts_ = 0U;
  std::uint64_t sam3_camera_labels_received_ = 0U;
  std::uint64_t sam3_camera_labels_applied_ = 0U;
  std::uint64_t sam3_camera_label_queue_drops_ = 0U;
  std::uint64_t sam3_flow_propagations_ = 0U;
  std::uint64_t sam3_flow_failures_ = 0U;
  std::uint64_t sam3_dynamic_pixels_ = 0U;
  double sam3_source_stamp_ = -std::numeric_limits<double>::infinity();
  double sam3_propagated_stamp_ = -std::numeric_limits<double>::infinity();
  cv::Mat propagated_semantic_labels_;
  cv::Mat semantic_propagation_gray_;
  double visual_update_rmse_ = std::numeric_limits<double>::infinity();
  double visual_update_ncc_ = 0.0;
  int visual_update_landmarks_ = 0;
  int visual_update_residuals_ = 0;
  int visual_update_iterations_ = 0;
  std::string visual_update_reason_ = "not_received";
  std::uint64_t semantic_clouds_received_ = 0U;
  std::uint64_t semantic_points_received_ = 0U;
  int lidar_registration_attempts_ = 0;
  int lidar_registration_accepts_ = 0;
  int lidar_registration_iterations_ = 0;
  double lidar_registration_rmse_ = std::numeric_limits<double>::infinity();
  double lidar_registration_inlier_ratio_ = 0.0;
  double lidar_processing_ms_ = 0.0;
  int lidar_scan_points_ = 0;
  int lidar_correspondence_sectors_ = 0;
  int lidar_point_knn_fallback_queries_ = 0;
  int lidar_point_knn_fallback_matches_ = 0;
  bool lidar_registration_degenerate_ = false;
  bool lidar_registration_used_imu_ = false;
  bool lidar_registration_used_wheel_ = false;
  bool lidar_strong_support_ = false;
  bool lidar_recovery_mode_ = false;
  bool lidar_loss_limited_ = false;
  bool lidar_loss_frozen_ = false;
  int lidar_consecutive_rejections_ = 0;
  bool lidar_imu_initialized_ = false;
  double lidar_imu_init_progress_ = 0.0;
  Eigen::Vector3d lidar_gyro_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d lidar_acceleration_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d lidar_gravity_ = Eigen::Vector3d(0.0, 0.0, -9.81);
  std::string lidar_registration_status_ = "waiting_for_raw_lidar";
  int next_object_id_ = 1;
  std::string last_map_reject_reason_ = "not_attempted";
  MapMatchResult last_map_match_;
  ros::Time last_publish_stamp_;
  std::ofstream trajectory_stream_;
  double last_saved_trajectory_stamp_ = -std::numeric_limits<double>::infinity();
};

}  // namespace

void HybridLocalizationNode::loadParameters()
{
  private_nh_.param<std::string>("imu_topic", imu_topic_, "/adi/adis16465/imu");
  private_nh_.param<std::string>("wheel_topic", wheel_topic_, "");
  private_nh_.param<std::string>("wheel_message_type", wheel_message_type_,
                                 "nav_msgs_odometry");
  private_nh_.param<std::string>("lidar_cloud_topic", lidar_cloud_topic_, "/hesai/at128/points");
  private_nh_.param<std::string>("camera_topic", camera_topic_,
                                 "/avt_camera/left/image/compressed");
  private_nh_.param<std::string>("visual_frontend/sam3_camera_label_topic",
                                 sam3_camera_label_topic_,
                                 "/sam3/camera/label");
  private_nh_.param<std::string>("semantic_cloud_topic", semantic_cloud_topic_, "/rangenet/semantic_points");
  private_nh_.param<std::string>("visual_factor_topic", visual_factor_topic_, "");
  private_nh_.param<std::string>("prior_grid_topic", prior_grid_topic_, "/hybrid/prior_grid");
  private_nh_.param<std::string>("prior_cloud_topic", prior_cloud_topic_, "");
  private_nh_.param<std::string>("map_frame", map_frame_, "hybrid_map");
  private_nh_.param<std::string>("odom_frame", odom_frame_, "odom");
  private_nh_.param<std::string>("body_frame", body_frame_, "base_link");
  private_nh_.param<std::string>("lidar_frame", lidar_frame_, "lidar");
  private_nh_.param<std::string>("output_odom_topic", output_odom_topic_, "/hybrid/odometry");
  private_nh_.param<std::string>("output_path_topic", output_path_topic_, "/hybrid/path");
  private_nh_.param<std::string>("local_bev_topic", local_bev_topic_, "/hybrid/local_bev");
  private_nh_.param<std::string>("prior_map_topic", prior_map_topic_, "/hybrid/prior_map");
  private_nh_.param<std::string>("semantic_output_topic", semantic_output_topic_, "/hybrid/semantic_cloud");
  private_nh_.param<std::string>("semantic_observation_topic", semantic_observation_topic_,
                                 "/hybrid/semantic_observation");
  private_nh_.param<std::string>("registered_cloud_topic", registered_cloud_topic_, "/hybrid/cloud_registered");
  private_nh_.param<std::string>("object_pose_topic", object_pose_topic_, "/hybrid/objects/poses");
  private_nh_.param<std::string>("object_marker_topic", object_marker_topic_, "/hybrid/objects/markers");
  private_nh_.param<std::string>("object_array_topic", object_array_topic_, "/hybrid/objects/array");
  private_nh_.param<std::string>("status_topic", status_topic_, "/hybrid/status");
  private_nh_.param<std::string>("quality_topic", quality_topic_, "/hybrid/quality");
  private_nh_.param<std::string>("debug_marker_topic", debug_marker_topic_, "/hybrid/map_match_debug");
  private_nh_.param<std::string>("diagnostics_topic", diagnostics_topic_, "/hybrid/diagnostics");
  private_nh_.param<std::string>("visual_direct_debug_topic",
                                 visual_direct_debug_topic_,
                                 "/hybrid/visual/direct_debug");
  private_nh_.param<std::string>("label_field", label_field_, "label");
  private_nh_.param<std::string>("lidar_label_mode", lidar_label_mode_, "semantic_kitti");
  private_nh_.param<std::string>("semantic_label_mode", semantic_label_mode_, "semantic_kitti");
  private_nh_.param<std::string>("lidar_point_time_mode", lidar_point_time_mode_, "none");
  private_nh_.param<std::string>("prior_pcd_path", prior_pcd_path_, "");
  private_nh_.param<std::string>("trajectory_save_path", trajectory_save_path_, "");
  private_nh_.param<std::string>("object_save_path", object_save_path_, "");

  private_nh_.param("subscribe_imu", subscribe_imu_, true);
  private_nh_.param("subscribe_wheel", subscribe_wheel_, false);
  private_nh_.param("subscribe_lidar", subscribe_lidar_, true);
  private_nh_.param("subscribe_camera_frontend", subscribe_camera_frontend_, false);
  private_nh_.param("subscribe_semantic", subscribe_semantic_, true);
  private_nh_.param("subscribe_visual_factor", subscribe_visual_factor_, false);
  private_nh_.param("subscribe_prior_grid", subscribe_prior_grid_, true);
  private_nh_.param("subscribe_prior_cloud", subscribe_prior_cloud_, false);
  private_nh_.param("semantic_cloud_in_map_frame", semantic_cloud_in_map_frame_, false);
  private_nh_.param("visual_pose_is_absolute", visual_pose_is_absolute_, true);
  private_nh_.param("broadcast_tf", broadcast_tf_, true);
  private_nh_.param("publish_debug", publish_debug_, true);
  private_nh_.param("publish_local_bev", publish_local_bev_, true);
  private_nh_.param("publish_objects", publish_objects_, true);
  private_nh_.param("prior_update_enabled", prior_update_enabled_, false);
  private_nh_.param("save_on_shutdown", save_on_shutdown_, true);
  private_nh_.param("include_dynamic_objects", include_dynamic_objects_, false);
  private_nh_.param("lidar_deskew_enabled", lidar_deskew_enabled_, true);
  private_nh_.param("lidar_stamp_is_end", lidar_stamp_is_end_, true);
  private_nh_.param("measurement_scheduler/enabled",
                    measurement_scheduler_enabled_, true);
  private_nh_.param("visual_frontend/enabled", visual_frontend_enabled_, false);
  private_nh_.param("visual_frontend/observation_only",
                    visual_observation_only_, true);
  private_nh_.param("visual_frontend/publish_debug",
                    publish_visual_direct_debug_, true);
  private_nh_.param("visual_frontend/use_sam3", visual_use_sam3_, false);
  private_nh_.param("visual_frontend/sam3_flow_scale", sam3_flow_scale_, 0.5);
  private_nh_.param("visual_frontend/sam3_cache_duration_sec",
                    sam3_cache_duration_sec_, 6.0);
  private_nh_.param("visual_frontend/sam3_label_sync_tolerance_sec",
                    sam3_label_sync_tolerance_sec_, 0.08);
  private_nh_.param("visual_frontend/sam3_max_source_age_sec",
                    sam3_max_source_age_sec_, 3.0);
  private_nh_.param("visual_frontend/sam3_max_flow_pixels",
                    sam3_max_flow_pixels_, 80.0);
  private_nh_.param("visual_frontend/sam3_max_cached_frames",
                    sam3_max_cached_frames_, 80);
  private_nh_.param("visual_frontend/sam3_max_replay_frames",
                    sam3_max_replay_frames_, 40);
  sam3_flow_scale_ = std::max(0.1, std::min(1.0, sam3_flow_scale_));
  sam3_cache_duration_sec_ = std::max(0.5, sam3_cache_duration_sec_);
  sam3_label_sync_tolerance_sec_ = std::max(0.005,
                                            sam3_label_sync_tolerance_sec_);
  sam3_max_source_age_sec_ = std::max(0.1, sam3_max_source_age_sec_);
  sam3_max_flow_pixels_ = std::max(1.0, sam3_max_flow_pixels_);
  sam3_max_cached_frames_ = std::max(2, sam3_max_cached_frames_);
  sam3_max_replay_frames_ = std::max(1, sam3_max_replay_frames_);

  private_nh_.param("max_lidar_points", max_lidar_points_, 10000);
  private_nh_.param("max_semantic_points", max_semantic_points_, 10000);
  private_nh_.param("max_local_points", max_local_points_, 18000);
  private_nh_.param("max_local_semantic_points", max_local_semantic_points_, 60000);
  private_nh_.param("max_semantic_output_points", max_semantic_output_points_, 12000);
  private_nh_.param("max_objects", max_objects_, 1000);
  private_nh_.param("dynamic_label", dynamic_label_, 5);
  private_nh_.param("traversable_label_max", traversable_label_max_, 2);
  private_nh_.param("keyframe_max_states", keyframe_max_states_, 12);
  private_nh_.param("optimizer_iterations", optimizer_iterations_, 3);
  private_nh_.param("optimizer_max_step_norm", optimizer_max_step_norm_, 0.5);
  private_nh_.param("keyframe_distance", keyframe_distance_, 0.8);
  private_nh_.param("keyframe_angle_deg", keyframe_angle_deg_, 8.0);
  private_nh_.param("keyframe_interval_sec", keyframe_interval_sec_, 0.5);
  private_nh_.param("max_pose_lookup_dt", max_odom_lookup_dt_, 0.15);
  private_nh_.param("local_map_window_sec", local_map_window_sec_, 12.0);
  private_nh_.param("local_map_resolution", local_map_resolution_, 0.25);
  private_nh_.param("local_map_size_m", local_map_size_m_, 80.0);
  private_nh_.param("local_ground_z", local_ground_z_, -2.0);
  private_nh_.param("local_max_z", local_max_z_, 4.0);
  private_nh_.param("local_min_range", local_min_range_, 0.5);
  private_nh_.param("local_max_range", local_max_range_, 80.0);
  private_nh_.param("map_match_rate", map_match_rate_, 1.0);
  private_nh_.param("map_match_min_points", map_match_min_points_, 100.0);
  private_nh_.param("map_factor_huber_delta", map_factor_huber_delta_, 1.0);
  private_nh_.param("map_factor_information", map_factor_information_, 2.0);
  private_nh_.param("map_factor_min_confidence", map_factor_min_confidence_, 0.45);
  private_nh_.param("prior_resolution", prior_resolution_, 0.25);
  private_nh_.param("prior_z_min", prior_z_min_, -3.0);
  private_nh_.param("prior_z_max", prior_z_max_, 4.0);
  private_nh_.param("prior_update_alpha", prior_update_alpha_, 0.02);
  private_nh_.param("prior_update_min_confidence", prior_update_min_confidence_, 0.7);
  private_nh_.param("object_assoc_radius", object_assoc_radius_, 1.5);
  private_nh_.param("object_min_confidence", object_min_confidence_, 0.45);
  private_nh_.param("object_reachable_radius", object_reachable_radius_, 8.0);
  private_nh_.param("publish_rate", publish_rate_, 10.0);
  private_nh_.param("status_rate", status_rate_, 1.0);
  private_nh_.param("lidar_point_time_scale", lidar_point_time_scale_, 1.0);
  private_nh_.param("lidar_imu_wait_sec", lidar_imu_wait_sec_, 0.0);
  private_nh_.param("camera_time_offset", camera_time_offset_, 0.0);
  private_nh_.param("measurement_scheduler/camera_imu_wait_sec",
                    camera_imu_wait_sec_, 0.01);
  private_nh_.param("measurement_scheduler/camera_observation_interval_sec",
                    camera_observation_interval_sec_, 0.20);
  private_nh_.param("measurement_scheduler/reorder_window_sec",
                    measurement_reorder_window_sec_, 0.02);
  private_nh_.param("measurement_scheduler/process_rate_hz",
                    measurement_scheduler_process_rate_hz_, 200.0);
  private_nh_.param("measurement_scheduler/max_events_per_tick",
                    measurement_scheduler_max_events_per_tick_, 2);
  private_nh_.param("measurement_scheduler/max_queue",
                    max_measurement_queue_, 200);
  camera_imu_wait_sec_ = std::max(0.0, camera_imu_wait_sec_);
  camera_observation_interval_sec_ = std::max(0.0,
                                               camera_observation_interval_sec_);
  measurement_reorder_window_sec_ = std::max(0.0, measurement_reorder_window_sec_);
  measurement_scheduler_process_rate_hz_ = std::max(
      10.0, measurement_scheduler_process_rate_hz_);
  measurement_scheduler_max_events_per_tick_ = std::max(
      1, measurement_scheduler_max_events_per_tick_);
  max_measurement_queue_ = std::max(10, max_measurement_queue_);

  const auto load_extrinsic = [this](const std::string &name, Eigen::Isometry3d &transform)
  {
    std::vector<double> values;
    private_nh_.param<std::vector<double>>(name, values, std::vector<double>());
    if (values.size() != 12 && values.size() != 16)
    {
      return;
    }
    transform.setIdentity();
    for (int row = 0; row < 3; ++row)
    {
      for (int col = 0; col < 4; ++col)
      {
        transform(row, col) = values[static_cast<std::size_t>(row * 4 + col)];
      }
    }
  };
  load_extrinsic("body_from_lidar", body_from_lidar_);
  load_extrinsic("body_from_imu", body_from_imu_);
  optimizer_ = SlidingWindowOptimizer(static_cast<std::size_t>(std::max(2, keyframe_max_states_)));
  local_bev_.reset(static_cast<int>(local_map_size_m_ / local_map_resolution_),
                   static_cast<int>(local_map_size_m_ / local_map_resolution_),
                   local_map_resolution_, 0.0, 0.0);
}

void HybridLocalizationNode::configureLidarOdometry()
{
  LidarOdometryOptions options;
  options.imu_enabled = subscribe_imu_;
  private_nh_.param("lidar_odometry/scan_voxel_size", options.scan_voxel_size, 0.35);
  private_nh_.param("lidar_odometry/map_voxel_size", options.map_voxel_size, 0.45);
  private_nh_.param("lidar_odometry/map_insert_voxel_size",
                    options.map_insert_voxel_size, 0.20);
  private_nh_.param("lidar_odometry/max_correspondence_distance",
                    options.max_correspondence_distance, 1.5);
  private_nh_.param("lidar_odometry/max_plane_distance", options.max_plane_distance, 0.45);
  private_nh_.param("lidar_odometry/plane_max_eigen_ratio",
                    options.plane_max_eigen_ratio, 0.20);
  private_nh_.param("lidar_odometry/lidar_range_noise", options.lidar_range_noise, 0.03);
  private_nh_.param("lidar_odometry/lidar_beam_noise", options.lidar_beam_noise, 0.0015);
  private_nh_.param("lidar_odometry/lidar_measurement_noise",
                    options.lidar_measurement_noise, 0.05);
  private_nh_.param("lidar_odometry/huber_delta", options.huber_delta, 0.20);
  private_nh_.param("lidar_odometry/max_rmse", options.max_rmse, 0.35);
  private_nh_.param("lidar_odometry/min_inlier_ratio", options.min_inlier_ratio, 0.18);
  private_nh_.param("lidar_odometry/convergence_translation",
                    options.convergence_translation, 0.002);
  private_nh_.param("lidar_odometry/convergence_rotation_deg",
                    options.convergence_rotation_deg, 0.05);
  private_nh_.param("lidar_odometry/degeneracy_eigen_ratio",
                    options.degeneracy_eigen_ratio, 1e-4);
  private_nh_.param("lidar_odometry/solver_damping", options.solver_damping, 1e-5);
  private_nh_.param("lidar_odometry/max_translation_per_scan",
                    options.max_translation_per_scan, 3.0);
  private_nh_.param("lidar_odometry/max_rotation_per_scan_deg",
                    options.max_rotation_per_scan_deg, 35.0);
  private_nh_.param("lidar_odometry/max_translation_speed",
                    options.max_translation_speed, 4.0);
  private_nh_.param("lidar_odometry/max_rotation_speed_deg",
                    options.max_rotation_speed_deg, 40.0);
  private_nh_.param("lidar_odometry/max_lidar_correction_translation",
                    options.max_lidar_correction_translation, 0.40);
  private_nh_.param("lidar_odometry/max_lidar_correction_rotation_deg",
                    options.max_lidar_correction_rotation_deg, 4.0);
  private_nh_.param("lidar_odometry/degenerate_min_inlier_ratio",
                    options.degenerate_min_inlier_ratio, 0.28);
  private_nh_.param("lidar_odometry/degenerate_max_rmse",
                    options.degenerate_max_rmse, 0.18);
  private_nh_.param("lidar_odometry/lidar_loss_hold_after_rejections",
                    options.lidar_loss_hold_after_rejections, 3);
  private_nh_.param("lidar_odometry/lidar_loss_freeze_after_rejections",
                    options.lidar_loss_freeze_after_rejections, 12);
  private_nh_.param("lidar_odometry/lidar_loss_max_vertical_offset",
                    options.lidar_loss_max_vertical_offset, 0.35);
  private_nh_.param("lidar_odometry/lidar_loss_max_horizontal_speed",
                    options.lidar_loss_max_horizontal_speed, 3.0);
  private_nh_.param("lidar_odometry/lidar_loss_max_horizontal_step",
                    options.lidar_loss_max_horizontal_step, 0.75);
  private_nh_.param("lidar_odometry/lidar_loss_velocity_decay",
                    options.lidar_loss_velocity_decay, 0.98);
  private_nh_.param("lidar_odometry/local_map_radius", options.local_map_radius, 60.0);
  private_nh_.param("lidar_odometry/imu_max_gap", options.imu_max_gap, 0.10);
  private_nh_.param("lidar_odometry/imu_buffer_duration", options.imu_buffer_duration, 5.0);
  private_nh_.param("lidar_odometry/imu_init_duration", options.imu_init_duration, 1.5);
  private_nh_.param("lidar_odometry/imu_init_samples", options.imu_init_samples, 200);
  private_nh_.param("lidar_odometry/imu_init_require_stationary",
                    options.imu_init_require_stationary, true);
  private_nh_.param("lidar_odometry/imu_init_max_acc_std",
                    options.imu_init_max_acc_std, 0.80);
  private_nh_.param("lidar_odometry/imu_init_max_gyro_std",
                    options.imu_init_max_gyro_std, 0.08);
  private_nh_.param("lidar_odometry/imu_init_max_gyro_bias",
                    options.imu_init_max_gyro_bias, 0.20);
  private_nh_.param("lidar_odometry/gravity_magnitude", options.gravity_magnitude, 9.81);
  private_nh_.param("lidar_odometry/auto_acceleration_scale",
                    options.auto_acceleration_scale, true);
  private_nh_.param("lidar_odometry/acceleration_scale", options.acceleration_scale, 1.0);
  private_nh_.param("lidar_odometry/gyro_noise", options.gyro_noise, 0.015);
  private_nh_.param("lidar_odometry/acceleration_noise", options.acceleration_noise, 0.10);
  private_nh_.param("lidar_odometry/gyro_bias_random_walk",
                    options.gyro_bias_random_walk, 0.00010);
  private_nh_.param("lidar_odometry/acceleration_bias_random_walk",
                    options.acceleration_bias_random_walk, 0.0010);
  private_nh_.param("lidar_odometry/gravity_random_walk",
                    options.gravity_random_walk, 0.00001);
  private_nh_.param("lidar_odometry/max_gyro_bias", options.max_gyro_bias, 0.50);
  private_nh_.param("lidar_odometry/max_acceleration_bias",
                    options.max_acceleration_bias, 3.0);
  options.wheel_enabled = subscribe_wheel_;
  private_nh_.param("lidar_odometry/wheel_speed_scale", options.wheel_speed_scale, 1.0);
  private_nh_.param("lidar_odometry/wheel_max_age", options.wheel_max_age, 0.12);
  private_nh_.param("lidar_odometry/wheel_max_speed", options.wheel_max_speed, 12.0);
  private_nh_.param("lidar_odometry/wheel_forward_noise", options.wheel_forward_noise, 0.18);
  private_nh_.param("lidar_odometry/wheel_lateral_noise", options.wheel_lateral_noise, 0.15);
  private_nh_.param("lidar_odometry/wheel_vertical_noise", options.wheel_vertical_noise, 0.25);
  private_nh_.param("lidar_odometry/wheel_huber_delta", options.wheel_huber_delta, 1.5);
  private_nh_.param("lidar_odometry/wheel_buffer_duration",
                    options.wheel_buffer_duration, 5.0);
  private_nh_.param("lidar_odometry/max_iterations", options.max_iterations, 10);
  private_nh_.param("lidar_odometry/min_scan_points", options.min_scan_points, 150);
  private_nh_.param("lidar_odometry/min_correspondences", options.min_correspondences, 80);
  private_nh_.param("lidar_odometry/max_scan_points", options.max_scan_points, 7000);
  private_nh_.param("lidar_odometry/max_map_points", options.max_map_points, 60000);
  private_nh_.param("lidar_odometry/map_insertion_max_plane_distance",
                    options.map_insertion_max_plane_distance, 0.25);
  private_nh_.param("lidar_odometry/normal_neighbor_voxels",
                    options.normal_neighbor_voxels, 2);
  private_nh_.param("lidar_odometry/min_normal_neighbors", options.min_normal_neighbors, 6);
  private_nh_.param("lidar_odometry/max_plane_neighbors", options.max_plane_neighbors, 20);
  private_nh_.param("lidar_odometry/max_plane_variance", options.max_plane_variance, 0.035);
  private_nh_.param("lidar_odometry/plane_uncertainty_scale",
                    options.plane_uncertainty_scale, 1.0);
  private_nh_.param("lidar_odometry/plane_fit_residual_gate",
                    options.plane_fit_residual_gate, 0.15);
  private_nh_.param("lidar_odometry/use_point_knn_plane",
                    options.use_point_knn_plane, false);
  private_nh_.param("lidar_odometry/use_compatible_voxel_plane",
                    options.use_compatible_voxel_plane, false);
  private_nh_.param("lidar_odometry/point_knn_fallback",
                    options.point_knn_fallback, false);
  private_nh_.param("lidar_odometry/point_knn_fallback_max_queries",
                    options.point_knn_fallback_max_queries, 0);
  private_nh_.param("lidar_odometry/strong_support_min_correspondences",
                    options.strong_support_min_correspondences, 0);
  private_nh_.param("lidar_odometry/strong_support_min_azimuth_sectors",
                    options.strong_support_min_azimuth_sectors, 4);
  private_nh_.param("lidar_odometry/strong_support_max_rmse",
                    options.strong_support_max_rmse, 0.0);
  private_nh_.param("lidar_odometry/recovery_after_rejections",
                    options.recovery_after_rejections, 0);
  private_nh_.param("lidar_odometry/recovery_max_lidar_correction_translation",
                    options.recovery_max_lidar_correction_translation, 0.0);
  private_nh_.param("lidar_odometry/recovery_max_lidar_correction_rotation_deg",
                    options.recovery_max_lidar_correction_rotation_deg, 0.0);
  private_nh_.param("lidar_odometry/min_voxel_plane_points",
                    options.min_voxel_plane_points, 8);
  private_nh_.param("lidar_odometry/max_voxel_points", options.max_voxel_points, 120);
  private_nh_.param("lidar_odometry/max_voxel_samples", options.max_voxel_samples, 12);
  options.visual_enabled = subscribe_camera_frontend_ && visual_frontend_enabled_ &&
      !visual_observation_only_;
  private_nh_.param("visual_frontend/eskf_max_iterations",
                    options.visual_max_iterations, 4);
  private_nh_.param("visual_frontend/min_landmarks",
                    options.visual_min_landmarks, 20);
  private_nh_.param("visual_frontend/min_residuals",
                    options.visual_min_residuals, 240);
  private_nh_.param("visual_frontend/max_rmse", options.visual_max_rmse, 1.20);
  private_nh_.param("visual_frontend/max_translation_step",
                    options.visual_max_translation_step, 0.35);
  private_nh_.param("visual_frontend/max_rotation_step_deg",
                    options.visual_max_rotation_step_deg, 4.0);
  private_nh_.param("visual_frontend/convergence_translation",
                    options.visual_convergence_translation, 0.0005);
  private_nh_.param("visual_frontend/convergence_rotation_deg",
                    options.visual_convergence_rotation_deg, 0.01);
  private_nh_.param("visual_frontend/solver_damping",
                    options.visual_solver_damping, 1e-6);
  lidar_odometry_ = LidarOdometry(options);
}

void HybridLocalizationNode::configureVisualMap()
{
  SparseVisualMapOptions options;
  options.enabled = subscribe_camera_frontend_ && visual_frontend_enabled_;
  private_nh_.param("visual_frontend/fx", options.fx, 1064.8950);
  private_nh_.param("visual_frontend/fy", options.fy, 1065.2546);
  private_nh_.param("visual_frontend/cx", options.cx, 801.4049);
  private_nh_.param("visual_frontend/cy", options.cy, 624.6878);
  private_nh_.param("visual_frontend/image_scale", options.image_scale, 0.5);
  private_nh_.param("visual_frontend/patch_half_size", options.patch_half_size, 3);
  private_nh_.param("visual_frontend/grid_size_pixels", options.grid_size_pixels, 24);
  private_nh_.param("visual_frontend/max_landmarks", options.max_landmarks, 1400);
  private_nh_.param("visual_frontend/max_active_landmarks",
                    options.max_active_landmarks, 450);
  private_nh_.param("visual_frontend/max_lidar_candidates",
                    options.max_lidar_candidates, 9000);
  private_nh_.param("visual_frontend/max_new_landmarks_per_frame",
                    options.max_new_landmarks_per_frame, 180);
  private_nh_.param("visual_frontend/max_missed_frames",
                    options.max_missed_frames, 20);
  private_nh_.param("visual_frontend/reference_refresh_observations",
                    options.reference_refresh_observations, 8);
  private_nh_.param("visual_frontend/minimum_depth", options.minimum_depth, 1.0);
  private_nh_.param("visual_frontend/maximum_depth", options.maximum_depth, 80.0);
  private_nh_.param("visual_frontend/minimum_gradient",
                    options.minimum_gradient, 12.0);
  private_nh_.param("visual_frontend/minimum_patch_stddev",
                    options.minimum_patch_stddev, 6.0);
  private_nh_.param("visual_frontend/minimum_ncc", options.minimum_ncc, 0.72);
  private_nh_.param("visual_frontend/photometric_huber_delta",
                    options.photometric_huber_delta, 1.5);
  private_nh_.param("visual_frontend/photometric_noise",
                    options.photometric_noise, 1.0);
  private_nh_.param("visual_frontend/information_scale",
                    options.information_scale, 0.04);
  private_nh_.param("visual_frontend/landmark_voxel_size",
                    options.landmark_voxel_size, 0.20);
  private_nh_.param("visual_frontend/local_map_radius",
                    options.local_map_radius, 65.0);
  std::vector<double> semantic_class_weights;
  private_nh_.param<std::vector<double>>(
      "visual_frontend/semantic_class_weights", semantic_class_weights,
      std::vector<double>());
  if (semantic_class_weights.size() == options.semantic_class_weights.size())
  {
    for (std::size_t index = 0; index < semantic_class_weights.size(); ++index)
    {
      options.semantic_class_weights[index] =
          static_cast<float>(semantic_class_weights[index]);
    }
  }

  std::vector<double> distortion;
  private_nh_.param<std::vector<double>>(
      "visual_frontend/distortion", distortion, std::vector<double>());
  if (distortion.size() == options.distortion.size())
  {
    std::copy(distortion.begin(), distortion.end(), options.distortion.begin());
  }
  std::vector<double> body_from_camera;
  private_nh_.param<std::vector<double>>(
      "visual_frontend/body_from_camera", body_from_camera,
      std::vector<double>());
  if (body_from_camera.size() == 12U || body_from_camera.size() == 16U)
  {
    for (int row = 0; row < 3; ++row)
    {
      for (int col = 0; col < 3; ++col)
      {
        options.body_from_camera_rotation(row, col) =
            body_from_camera[static_cast<std::size_t>(row * 4 + col)];
      }
      options.body_from_camera_translation(row) =
          body_from_camera[static_cast<std::size_t>(row * 4 + 3)];
    }
  }
  sparse_visual_map_ = SparseVisualMap(options);
}

void HybridLocalizationNode::configureMatcher()
{
  hybrid_localization::MatcherOptions options;
  private_nh_.param("matcher_min_search_radius", options.min_search_radius, 5.0);
  private_nh_.param("matcher_max_search_radius", options.max_search_radius, 60.0);
  private_nh_.param("matcher_confidence_gamma", options.confidence_gamma, 9.21);
  private_nh_.param("matcher_coarse_translation_step", options.coarse_translation_step, 1.0);
  private_nh_.param("matcher_fine_translation_step", options.fine_translation_step, 0.25);
  private_nh_.param("matcher_coarse_yaw_step_deg", options.coarse_yaw_step_deg, 10.0);
  private_nh_.param("matcher_fine_yaw_step_deg", options.fine_yaw_step_deg, 2.0);
  private_nh_.param("matcher_yaw_search_deg", options.yaw_search_deg, 45.0);
  private_nh_.param("matcher_distance", options.match_distance, 1.0);
  private_nh_.param("matcher_inlier_distance", options.inlier_distance, 0.65);
  private_nh_.param("matcher_min_inlier_ratio", options.min_inlier_ratio, 0.25);
  private_nh_.param("matcher_min_confidence", options.min_confidence, 0.45);
  private_nh_.param("matcher_min_score_gap", options.min_score_gap, 0.04);
  private_nh_.param("matcher_occupancy_weight", options.occupancy_weight, 1.0);
  private_nh_.param("matcher_edge_weight", options.edge_weight, 0.35);
  private_nh_.param("matcher_semantic_weight", options.semantic_weight, 0.25);
  private_nh_.param("matcher_dynamic_penalty", options.dynamic_penalty, 0.75);
  private_nh_.param("matcher_max_points", options.max_points, 2500);
  private_nh_.param("matcher_max_candidates", options.max_candidates, 200000);
  matcher_ = PriorMatcher(options);
}

void HybridLocalizationNode::initializeRos()
{
  if (subscribe_imu_ && !imu_topic_.empty())
  {
    imu_nh_.setCallbackQueue(&imu_callback_queue_);
    imu_sub_ = imu_nh_.subscribe(imu_topic_, 20000,
                                &HybridLocalizationNode::imuCallback, this);
  }
  if (subscribe_wheel_ && !wheel_topic_.empty())
  {
    if (wheel_message_type_ == "insprobe_ranger")
    {
      wheel_sub_ = nh_.subscribe<insprobe_msgs::RangerOdometer>(
          wheel_topic_, 200, &HybridLocalizationNode::rangerWheelCallback, this);
    }
    else
    {
      wheel_sub_ = nh_.subscribe<nav_msgs::Odometry>(
          wheel_topic_, 200, &HybridLocalizationNode::wheelCallback, this);
    }
  }
  if (subscribe_lidar_ && !lidar_cloud_topic_.empty())
  {
    if (measurement_scheduler_enabled_)
    {
      lidar_nh_.setCallbackQueue(&lidar_callback_queue_);
      lidar_sub_ = lidar_nh_.subscribe(
          lidar_cloud_topic_, 100, &HybridLocalizationNode::lidarCallback, this);
    }
    else
    {
      lidar_sub_ = nh_.subscribe(
          lidar_cloud_topic_, 20, &HybridLocalizationNode::lidarCallback, this);
    }
  }
  if (subscribe_camera_frontend_ && !camera_topic_.empty())
  {
    if (measurement_scheduler_enabled_)
    {
      camera_nh_.setCallbackQueue(&camera_callback_queue_);
      camera_sub_ = camera_nh_.subscribe<sensor_msgs::CompressedImage>(
          camera_topic_, 100, &HybridLocalizationNode::cameraCallback, this);
    }
    else
    {
      camera_sub_ = nh_.subscribe<sensor_msgs::CompressedImage>(
          camera_topic_, 20, &HybridLocalizationNode::cameraCallback, this);
    }
  }
  if (subscribe_camera_frontend_ && visual_frontend_enabled_ &&
      visual_use_sam3_ && !sam3_camera_label_topic_.empty())
  {
    if (measurement_scheduler_enabled_)
    {
      camera_nh_.setCallbackQueue(&camera_callback_queue_);
      sam3_camera_label_sub_ = camera_nh_.subscribe<sensor_msgs::Image>(
          sam3_camera_label_topic_, 10,
          &HybridLocalizationNode::sam3CameraLabelCallback, this);
    }
    else
    {
      sam3_camera_label_sub_ = nh_.subscribe<sensor_msgs::Image>(
          sam3_camera_label_topic_, 5,
          &HybridLocalizationNode::sam3CameraLabelCallback, this);
    }
  }
  if (subscribe_semantic_ && !semantic_cloud_topic_.empty())
  {
    semantic_sub_ = nh_.subscribe(semantic_cloud_topic_, 20, &HybridLocalizationNode::semanticCallback, this);
  }
  if (subscribe_visual_factor_ && !visual_factor_topic_.empty())
  {
    visual_factor_sub_ = nh_.subscribe(visual_factor_topic_, 50,
                                       &HybridLocalizationNode::visualFactorCallback, this);
  }
  if (subscribe_prior_grid_ && !prior_grid_topic_.empty())
  {
    prior_grid_sub_ = nh_.subscribe(prior_grid_topic_, 1, &HybridLocalizationNode::priorGridCallback, this);
  }
  if (subscribe_prior_cloud_ && !prior_cloud_topic_.empty())
  {
    prior_cloud_sub_ = nh_.subscribe(prior_cloud_topic_, 1, &HybridLocalizationNode::priorCloudCallback, this);
  }

  output_odom_pub_ = nh_.advertise<nav_msgs::Odometry>(output_odom_topic_, 20);
  output_path_pub_ = nh_.advertise<nav_msgs::Path>(output_path_topic_, 2, true);
  local_bev_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(local_bev_topic_, 2);
  prior_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(prior_map_topic_, 1, true);
  object_pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>(object_pose_topic_, 2);
  object_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(object_marker_topic_, 2);
  object_array_pub_ = nh_.advertise<std_msgs::Float32MultiArray>(object_array_topic_, 2);
  semantic_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(semantic_output_topic_, 2);
  semantic_observation_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      semantic_observation_topic_, 4);
  registered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(registered_cloud_topic_, 2);
  status_pub_ = nh_.advertise<std_msgs::String>(status_topic_, 2, true);
  quality_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(quality_topic_, 2);
  debug_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(debug_marker_topic_, 2);
  diagnostic_pub_ = nh_.advertise<diagnostic_msgs::DiagnosticArray>(diagnostics_topic_, 2);
  visual_direct_debug_pub_ = nh_.advertise<sensor_msgs::Image>(
      visual_direct_debug_topic_, 2);

  map_match_timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.1, map_match_rate_)),
                                     &HybridLocalizationNode::mapMatchTimer, this);
  publish_timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.1, publish_rate_)),
                                   &HybridLocalizationNode::publishTimer, this);
  if (measurement_scheduler_enabled_)
  {
    measurement_scheduler_timer_ = nh_.createWallTimer(
        ros::WallDuration(1.0 / measurement_scheduler_process_rate_hz_),
        &HybridLocalizationNode::measurementSchedulerTimer, this);
  }
  output_path_.header.frame_id = map_frame_;
}

HybridLocalizationNode::~HybridLocalizationNode()
{
  if (!save_on_shutdown_)
  {
    return;
  }
  if (trajectory_stream_.is_open())
  {
    trajectory_stream_.flush();
    trajectory_stream_.close();
  }
  else if (!trajectory_save_path_.empty())
  {
    std::ofstream output(trajectory_save_path_.c_str());
    if (output.is_open())
    {
      output << "timestamp,x,y,z,qx,qy,qz,qw\n";
      for (const geometry_msgs::PoseStamped &pose : output_path_.poses)
      {
        output << std::fixed << pose.header.stamp.toSec() << ","
               << pose.pose.position.x << "," << pose.pose.position.y << ","
               << pose.pose.position.z << "," << pose.pose.orientation.x << ","
               << pose.pose.orientation.y << "," << pose.pose.orientation.z << ","
               << pose.pose.orientation.w << "\n";
      }
    }
  }
  if (!object_save_path_.empty())
  {
    std::ofstream output(object_save_path_.c_str());
    if (output.is_open())
    {
      output << "id,label,x,y,z,cov_x,cov_y,cov_z,observations,reliability,reachable\n";
      for (const ObjectState &object : objects_)
      {
        output << object.id << "," << static_cast<int>(object.label) << ","
               << object.mean.x() << "," << object.mean.y() << "," << object.mean.z() << ","
               << object.covariance(0, 0) << "," << object.covariance(1, 1) << ","
               << object.covariance(2, 2) << "," << object.observations << ","
               << object.reliability << "," << (object.reachable ? 1 : 0) << "\n";
      }
    }
  }
}

void HybridLocalizationNode::loadPriorPointCloud()
{
  if (prior_pcd_path_.empty())
  {
    return;
  }
  std::ifstream input(prior_pcd_path_.c_str());
  if (!input.is_open())
  {
    ROS_WARN("[hybrid_localization] unable to open prior_pcd_path: %s", prior_pcd_path_.c_str());
    return;
  }
  std::vector<CloudPoint> points;
  std::string line;
  bool data_section = false;
  bool ascii_data = true;
  while (std::getline(input, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }
    if (line.find("DATA") == 0 || line.find("data") == 0)
    {
      data_section = true;
      ascii_data = line.find("ascii") != std::string::npos || line.find("ASCII") != std::string::npos;
      continue;
    }
    if (!data_section && line.find("VERSION") != 0 && line.find("FIELDS") != 0 &&
        line.find("SIZE") != 0 && line.find("TYPE") != 0 && line.find("COUNT") != 0 &&
        line.find("WIDTH") != 0 && line.find("HEIGHT") != 0 && line.find("VIEWPOINT") != 0 &&
        line.find("POINTS") != 0)
    {
      data_section = true;
    }
    if (!data_section || !ascii_data)
    {
      continue;
    }
    std::stringstream stream(line);
    CloudPoint point;
    if (stream >> point.point.x() >> point.point.y() >> point.point.z())
    {
      if (point.point.allFinite())
      {
        points.push_back(point);
      }
    }
  }
  if (points.empty())
  {
    ROS_WARN("[hybrid_localization] prior point cloud is empty or not ASCII: %s", prior_pcd_path_.c_str());
    return;
  }
  buildPriorFromPoints(points);
  ROS_INFO("[hybrid_localization] loaded %zu prior points from %s", points.size(), prior_pcd_path_.c_str());
}

void HybridLocalizationNode::buildPriorFromPoints(const std::vector<CloudPoint> &points)
{
  if (points.empty())
  {
    return;
  }
  Eigen::Vector3d min_point = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d max_point = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  for (const CloudPoint &point : points)
  {
    if (point.point.z() < prior_z_min_ || point.point.z() > prior_z_max_)
    {
      continue;
    }
    min_point = min_point.cwiseMin(point.point);
    max_point = max_point.cwiseMax(point.point);
  }
  if (!min_point.allFinite() || !max_point.allFinite())
  {
    return;
  }
  const double padding = 2.0 * prior_resolution_;
  int width = std::max(1, static_cast<int>(std::ceil((max_point.x() - min_point.x()) / prior_resolution_)) + 1);
  int height = std::max(1, static_cast<int>(std::ceil((max_point.y() - min_point.y()) / prior_resolution_)) + 1);
  if (static_cast<long long>(width) * static_cast<long long>(height) > 9000000LL)
  {
    const double scale = std::sqrt(static_cast<double>(width) * static_cast<double>(height) / 9000000.0);
    prior_resolution_ *= scale;
    width = std::max(1, static_cast<int>(std::ceil((max_point.x() - min_point.x()) / prior_resolution_)) + 1);
    height = std::max(1, static_cast<int>(std::ceil((max_point.y() - min_point.y()) / prior_resolution_)) + 1);
  }
  prior_map_.width = width;
  prior_map_.height = height;
  prior_map_.resolution = std::max(1e-3, prior_resolution_);
  prior_map_.origin_x = min_point.x() - padding;
  prior_map_.origin_y = min_point.y() - padding;
  prior_map_.occupancy.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0.0F);
  prior_map_.labels.assign(prior_map_.occupancy.size(), 0U);
  for (const CloudPoint &point : points)
  {
    if (point.point.z() < prior_z_min_ || point.point.z() > prior_z_max_)
    {
      continue;
    }
    int ix = 0;
    int iy = 0;
    if (!prior_map_.worldToCell(point.point.x(), point.point.y(), ix, iy))
    {
      continue;
    }
    const std::size_t cell = static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
                             static_cast<std::size_t>(ix);
    prior_map_.occupancy[cell] = std::min(1.0F, prior_map_.occupancy[cell] + 0.12F);
    if (point.label != 0U)
    {
      prior_map_.labels[cell] = normalizeLabel(point.label);
    }
  }
  prior_map_.recomputeEdges();
  prior_map_pub_.publish(makePriorGridMessage(ros::Time::now()));
}

void HybridLocalizationNode::buildPriorFromGrid(const nav_msgs::OccupancyGrid &grid)
{
  prior_map_.width = static_cast<int>(grid.info.width);
  prior_map_.height = static_cast<int>(grid.info.height);
  prior_map_.resolution = grid.info.resolution;
  prior_map_.origin_x = grid.info.origin.position.x;
  prior_map_.origin_y = grid.info.origin.position.y;
  prior_map_.occupancy.assign(static_cast<std::size_t>(prior_map_.width) *
                                  static_cast<std::size_t>(prior_map_.height), 0.0F);
  prior_map_.labels.assign(prior_map_.occupancy.size(), 0U);
  for (std::size_t index = 0; index < prior_map_.occupancy.size() && index < grid.data.size(); ++index)
  {
    prior_map_.occupancy[index] = grid.data[index] < 0 ? 0.0F :
        static_cast<float>(grid.data[index]) / 100.0F;
  }
  prior_map_.recomputeEdges();
  prior_map_pub_.publish(makePriorGridMessage(grid.header.stamp.isZero() ? ros::Time::now() : grid.header.stamp));
  ROS_INFO("[hybrid_localization] received prior occupancy grid %dx%d at %.3fm/cell",
           prior_map_.width, prior_map_.height, prior_map_.resolution);
}

void HybridLocalizationNode::priorGridCallback(const nav_msgs::OccupancyGridConstPtr &message)
{
  buildPriorFromGrid(*message);
}

void HybridLocalizationNode::priorCloudCallback(const sensor_msgs::PointCloud2ConstPtr &message)
{
  const std::vector<CloudPoint> points = decodeCloud(*message, label_field_, max_lidar_points_);
  buildPriorFromPoints(points);
}

Eigen::Isometry3d HybridLocalizationNode::lookupOdomPose(double stamp, bool *ok) const
{
  if (ok != nullptr) *ok = false;
  if (odom_history_.empty())
  {
    return latest_odom_pose_;
  }
  double best_dt = std::numeric_limits<double>::infinity();
  Eigen::Isometry3d best_pose = odom_history_.back().second;
  for (const auto &entry : odom_history_)
  {
    const double dt = std::abs(entry.first - stamp);
    if (dt < best_dt)
    {
      best_dt = dt;
      best_pose = entry.second;
    }
  }
  if (ok != nullptr) *ok = best_dt <= max_odom_lookup_dt_;
  return best_pose;
}

uint8_t HybridLocalizationNode::normalizeLabel(uint8_t label) const
{
  return convertSemanticLabel(static_cast<int>(label), semantic_label_mode_);
}

void HybridLocalizationNode::imuCallback(const sensor_msgs::ImuConstPtr &message)
{
  if (!message) return;
  const double stamp = message->header.stamp.toSec();
  const Eigen::Matrix3d body_from_imu_rotation = body_from_imu_.rotation();
  const Eigen::Vector3d acceleration = body_from_imu_rotation * Eigen::Vector3d(
      message->linear_acceleration.x, message->linear_acceleration.y,
      message->linear_acceleration.z);
  const Eigen::Vector3d angular_velocity = body_from_imu_rotation * Eigen::Vector3d(
      message->angular_velocity.x, message->angular_velocity.y,
      message->angular_velocity.z);
  ImuSample sample;
  sample.stamp = stamp;
  sample.acceleration = acceleration;
  sample.angular_velocity = angular_velocity;
  if (!std::isfinite(sample.stamp) || !sample.acceleration.allFinite() ||
      !sample.angular_velocity.allFinite()) return;
  std::lock_guard<std::mutex> lock(pending_imu_mutex_);
  if (sample.stamp <= latest_received_imu_stamp_) return;
  latest_received_imu_stamp_ = sample.stamp;
  pending_imu_samples_.push_back(sample);
  while (pending_imu_samples_.size() > 20000U)
  {
    pending_imu_samples_.pop_front();
    ++pending_imu_queue_drops_;
  }
}

void HybridLocalizationNode::drainPendingImuSamples(double up_to_stamp)
{
  std::deque<ImuSample, Eigen::aligned_allocator<ImuSample>> samples;
  {
    std::lock_guard<std::mutex> lock(pending_imu_mutex_);
    while (!pending_imu_samples_.empty() &&
           pending_imu_samples_.front().stamp <= up_to_stamp + 1e-9)
    {
      samples.push_back(pending_imu_samples_.front());
      pending_imu_samples_.pop_front();
    }
  }
  for (const ImuSample &sample : samples)
  {
    if (sample.stamp <= last_imu_stamp_) continue;
    lidar_odometry_.addImuSample(sample);
    last_imu_stamp_ = sample.stamp;
  }
}

double HybridLocalizationNode::latestReceivedImuStamp() const
{
  std::lock_guard<std::mutex> lock(pending_imu_mutex_);
  return latest_received_imu_stamp_;
}

std::size_t HybridLocalizationNode::pendingImuSampleCount() const
{
  std::lock_guard<std::mutex> lock(pending_imu_mutex_);
  return pending_imu_samples_.size();
}

std::uint64_t HybridLocalizationNode::pendingImuQueueDrops() const
{
  std::lock_guard<std::mutex> lock(pending_imu_mutex_);
  return pending_imu_queue_drops_;
}

void HybridLocalizationNode::wheelCallback(const nav_msgs::OdometryConstPtr &message)
{
  latest_wheel_pose_ = poseFromMessage(message->pose.pose);
  have_wheel_ = true;
  WheelSample sample;
  sample.stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                : message->header.stamp.toSec();
  sample.forward_speed = message->twist.twist.linear.x;
  lidar_odometry_.addWheelSample(sample);
  last_wheel_speed_ = sample.forward_speed;
  ++wheel_samples_;
}

void HybridLocalizationNode::rangerWheelCallback(
    const insprobe_msgs::RangerOdometerConstPtr &message)
{
  std::array<double, 4> speeds{{message->left_front_speed,
                                message->right_front_speed,
                                message->right_back_speed,
                                message->left_back_speed}};
  if (!std::all_of(speeds.begin(), speeds.end(),
                   [](double value) { return std::isfinite(value); }))
  {
    return;
  }
  std::sort(speeds.begin(), speeds.end());
  WheelSample sample;
  sample.stamp = !message->header.stamp.isZero() ? message->header.stamp.toSec()
      : std::isfinite(message->unixtime) && message->unixtime > 0.0
          ? message->unixtime : ros::Time::now().toSec();
  sample.forward_speed = 0.5 * (speeds[1] + speeds[2]);
  lidar_odometry_.addWheelSample(sample);
  last_wheel_speed_ = sample.forward_speed;
  ++wheel_samples_;
}

void HybridLocalizationNode::visualFactorCallback(
    const geometry_msgs::PoseWithCovarianceStampedConstPtr &message)
{
  if (!visual_pose_is_absolute_)
  {
    ROS_WARN_THROTTLE(5.0, "[hybrid_localization] visual_pose_is_absolute=false is reserved for a future relative adapter");
    return;
  }
  visual_factor_queue_.push_back(*message);
  while (visual_factor_queue_.size() > 50)
  {
    visual_factor_queue_.pop_front();
  }
}

void HybridLocalizationNode::handleLidarOdometry(const LidarOdometryResult &result)
{
  const double stamp = result.stamp;
  const Eigen::Isometry3d odom_pose = hybrid_localization::projectToSE3(result.pose);
  odom_history_.push_back(std::make_pair(stamp, odom_pose));
  while (odom_history_.size() > 1200)
  {
    odom_history_.pop_front();
  }

  for (int index = 0; index < 3; ++index)
  {
    position_covariance_(index, index) = std::max(
        1e-4, std::abs(result.covariance(index + 3, index + 3)));
  }
  if (!have_odom_)
  {
    latest_odom_pose_ = odom_pose;
    last_keyframe_odom_pose_ = odom_pose;
    global_from_odom_ = Eigen::Isometry3d::Identity();
    latest_state_.stamp = stamp;
    latest_state_.pose = odom_pose;
    latest_state_.velocity = result.velocity;
    latest_state_.covariance.setIdentity();
    latest_state_.covariance.topLeftCorner<3, 3>() = result.covariance.topLeftCorner<3, 3>();
    latest_state_.covariance.block<3, 3>(3, 3) = position_covariance_;
    optimizer_.reset();
    optimizer_.addState(latest_state_);
    have_odom_ = true;
    have_keyframe_ = true;
    last_odom_stamp_ = stamp;
    last_keyframe_stamp_ = stamp;
    last_publish_stamp_ = ros::Time(stamp);
    publishCurrentState(ros::Time(stamp));
    return;
  }

  latest_odom_pose_ = odom_pose;
  latest_state_.stamp = stamp;
  latest_state_.velocity = result.velocity;
  latest_state_.covariance.topLeftCorner<3, 3>() =
      result.covariance.topLeftCorner<3, 3>();
  latest_state_.covariance.block<3, 3>(3, 3) =
      result.covariance.block<3, 3>(3, 3);
  const Eigen::Isometry3d keyframe_motion = last_keyframe_odom_pose_.inverse() * odom_pose;
  const double translation = keyframe_motion.translation().norm();
  const double rotation = hybrid_localization::logSE3(keyframe_motion).head<3>().norm() * 180.0 / kPi;
  const bool keyframe = result.accepted && (translation >= keyframe_distance_ ||
      rotation >= keyframe_angle_deg_ || stamp - last_keyframe_stamp_ >= keyframe_interval_sec_);
  if (keyframe)
  {
    const Eigen::Isometry3d relative_lidar = last_keyframe_odom_pose_.inverse() * odom_pose;
    PoseState state = latest_state_;
    state.stamp = stamp;
    state.pose = global_from_odom_ * odom_pose;
    state.velocity = latest_state_.velocity;
    state.covariance.block<3, 3>(3, 3) = position_covariance_;
    const int current_index = optimizer_.addState(state);
    if (current_index > 0)
    {
      const int previous_index = current_index - 1;
      Eigen::Matrix<double, 6, 6> lidar_covariance = result.covariance;
      for (int index = 0; index < 6; ++index)
      {
        lidar_covariance(index, index) = std::max(1e-4,
            std::min(1.0, std::abs(lidar_covariance(index, index))));
      }
      const double registration_quality = std::max(0.1, std::min(1.0,
          result.inlier_ratio * std::exp(-result.rmse)));
      const Eigen::Matrix<double, 6, 6> lidar_information =
          registration_quality * lidar_covariance.inverse();
      optimizer_.addRelativeFactor(previous_index, current_index, relative_lidar,
                                   lidar_information, FactorType::LidarRegistration, 0.8);
      ++factor_count_;
      bool independent_factor_added = false;
      if (have_wheel_)
      {
        const Eigen::Isometry3d wheel_relative = last_keyframe_wheel_pose_.inverse() * latest_wheel_pose_;
        Eigen::Vector3d wheel_measurement(wheel_relative.translation().x(), wheel_relative.translation().y(),
                                          hybrid_localization::yawOf(wheel_relative));
        optimizer_.addWheelFactor(previous_index, current_index, wheel_measurement,
                                  Eigen::Matrix3d::Identity() * 3.0, 1.0);
        ++factor_count_;
        independent_factor_added = true;
      }
      last_keyframe_odom_pose_ = odom_pose;
      while (!visual_factor_queue_.empty())
      {
        const geometry_msgs::PoseWithCovarianceStamped &visual = visual_factor_queue_.front();
        if (visual.header.stamp.toSec() > stamp + max_odom_lookup_dt_)
        {
          break;
        }
        if (std::abs(visual.header.stamp.toSec() - stamp) <= max_odom_lookup_dt_)
        {
          const Eigen::Matrix<double, 6, 6> information =
              poseInformationFromCovariance(visual.pose.covariance, 1.0);
          optimizer_.addAbsoluteFactor(current_index, poseFromMessage(visual.pose.pose), information,
                                       FactorType::Visual, 1.0, 1.5);
          ++factor_count_;
          independent_factor_added = true;
        }
        visual_factor_queue_.pop_front();
      }
      if (independent_factor_added)
      {
        optimizeCurrentWindow();
      }
    }
    last_keyframe_odom_pose_ = odom_pose;
    if (have_wheel_)
    {
      last_keyframe_wheel_pose_ = latest_wheel_pose_;
    }
    last_keyframe_stamp_ = stamp;
    have_keyframe_ = true;
  }
  last_odom_stamp_ = stamp;
  publishCurrentState(ros::Time(stamp));
}

void HybridLocalizationNode::handleVisualUpdate(const VisualUpdateResult &result)
{
  if (!result.propagated || !have_odom_) return;

  const double stamp = result.stamp;
  const Eigen::Isometry3d odom_pose =
      hybrid_localization::projectToSE3(result.pose);
  latest_odom_pose_ = odom_pose;
  latest_state_.stamp = stamp;
  latest_state_.pose = hybrid_localization::projectToSE3(
      global_from_odom_ * odom_pose);
  latest_state_.velocity = result.velocity;
  latest_state_.covariance.topLeftCorner<3, 3>() =
      result.covariance.topLeftCorner<3, 3>();
  latest_state_.covariance.block<3, 3>(3, 3) =
      result.covariance.block<3, 3>(3, 3);
  for (int index = 0; index < 3; ++index)
  {
    position_covariance_(index, index) = std::max(
        1e-4, std::abs(result.covariance(index + 3, index + 3)));
  }
  odom_history_.push_back(std::make_pair(stamp, odom_pose));
  while (odom_history_.size() > 1200) odom_history_.pop_front();
  last_odom_stamp_ = stamp;
  publishCurrentState(ros::Time(stamp));
}

void HybridLocalizationNode::optimizeCurrentWindow()
{
  if (optimizer_.empty())
  {
    return;
  }
  optimizer_.optimize(optimizer_iterations_, std::max(1e-3, optimizer_max_step_norm_));
  latest_state_ = optimizer_.latestState();
  latest_state_.pose = hybrid_localization::projectToSE3(latest_state_.pose);
  if (have_odom_)
  {
    const Eigen::Isometry3d corrected = latest_state_.pose * last_keyframe_odom_pose_.inverse();
    global_from_odom_ = hybrid_localization::projectToSE3(corrected);
  }
}

void HybridLocalizationNode::lidarCallback(const sensor_msgs::PointCloud2ConstPtr &message)
{
  MeasurementEvent event;
  event.stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                : message->header.stamp.toSec();
  event.type = MeasurementEventType::LIDAR;
  event.lidar = message;
  enqueueMeasurement(std::move(event), !measurement_scheduler_enabled_);
}

void HybridLocalizationNode::cameraCallback(
    const sensor_msgs::CompressedImageConstPtr &message)
{
  MeasurementEvent event;
  event.stamp = (message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                : message->header.stamp.toSec()) +
      camera_time_offset_;
  event.type = MeasurementEventType::IMAGE;
  event.image = message;
  enqueueMeasurement(std::move(event), !measurement_scheduler_enabled_);
}

cv::Mat HybridLocalizationNode::semanticFlowGray(const cv::Mat &image) const
{
  if (image.empty()) return cv::Mat();
  cv::Mat gray;
  if (image.channels() == 1) gray = image;
  else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  else return cv::Mat();
  if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U);
  if (sam3_flow_scale_ < 0.999)
  {
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(), sam3_flow_scale_, sam3_flow_scale_,
               cv::INTER_AREA);
    gray = resized;
  }
  return gray.clone();
}

void HybridLocalizationNode::cacheCameraFrame(const cv::Mat &gray, double stamp)
{
  if (gray.empty() || !std::isfinite(stamp)) return;
  CachedCameraFrame frame;
  frame.stamp = stamp;
  frame.gray = gray.clone();
  camera_frame_cache_.push_back(std::move(frame));
  while (!camera_frame_cache_.empty() &&
         (camera_frame_cache_.size() >
              static_cast<std::size_t>(sam3_max_cached_frames_) ||
          stamp - camera_frame_cache_.front().stamp >
              sam3_cache_duration_sec_))
  {
    camera_frame_cache_.pop_front();
  }
}

bool HybridLocalizationNode::propagateSemanticLabels(
    const cv::Mat &current_gray, double current_stamp)
{
  if (current_gray.empty() || propagated_semantic_labels_.empty() ||
      semantic_propagation_gray_.empty() || !std::isfinite(current_stamp) ||
      current_stamp <= sam3_propagated_stamp_ + 1e-8)
  {
    return false;
  }
  try
  {
    cv::Mat previous_gray = semantic_propagation_gray_;
    cv::Mat previous_labels = propagated_semantic_labels_;
    if (previous_gray.size() != current_gray.size())
    {
      cv::resize(previous_gray, previous_gray, current_gray.size(), 0.0, 0.0,
                 cv::INTER_AREA);
    }
    if (previous_labels.size() != current_gray.size())
    {
      cv::resize(previous_labels, previous_labels, current_gray.size(), 0.0,
                 0.0, cv::INTER_NEAREST);
    }

    // Backward flow maps each current pixel to its source in the previous
    // semantic image, which makes nearest-neighbor label remapping direct.
    cv::Mat backward_flow;
    cv::calcOpticalFlowFarneback(current_gray, previous_gray, backward_flow,
                                 0.5, 3, 21, 3, 5, 1.2, 0);
    cv::Mat map_x(current_gray.size(), CV_32F);
    cv::Mat map_y(current_gray.size(), CV_32F);
    cv::Mat invalid(current_gray.size(), CV_8U, cv::Scalar(0));
    for (int row = 0; row < current_gray.rows; ++row)
    {
      const cv::Vec2f *flow_row = backward_flow.ptr<cv::Vec2f>(row);
      float *map_x_row = map_x.ptr<float>(row);
      float *map_y_row = map_y.ptr<float>(row);
      uint8_t *invalid_row = invalid.ptr<uint8_t>(row);
      for (int col = 0; col < current_gray.cols; ++col)
      {
        const cv::Vec2f flow = flow_row[col];
        map_x_row[col] = static_cast<float>(col) + flow[0];
        map_y_row[col] = static_cast<float>(row) + flow[1];
        const double magnitude = std::hypot(flow[0], flow[1]);
        if (!std::isfinite(magnitude) || magnitude > sam3_max_flow_pixels_)
        {
          invalid_row[col] = 255U;
        }
      }
    }
    cv::Mat warped;
    cv::remap(previous_labels, warped, map_x, map_y, cv::INTER_NEAREST,
              cv::BORDER_CONSTANT, cv::Scalar(0));
    warped.setTo(cv::Scalar(0), invalid);
    propagated_semantic_labels_ = warped;
    semantic_propagation_gray_ = current_gray.clone();
    sam3_propagated_stamp_ = current_stamp;
    ++sam3_flow_propagations_;
    return true;
  }
  catch (const cv::Exception &exception)
  {
    ++sam3_flow_failures_;
    ROS_WARN_THROTTLE(2.0, "[hybrid_localization] SAM3 label flow failed: %s",
                      exception.what());
    return false;
  }
}

void HybridLocalizationNode::sam3CameraLabelCallback(
    const sensor_msgs::ImageConstPtr &message)
{
  if (!measurement_scheduler_enabled_)
  {
    processSam3CameraLabel(message);
    return;
  }
  std::lock_guard<std::mutex> lock(pending_sam3_label_mutex_);
  while (pending_sam3_labels_.size() >= 8U)
  {
    pending_sam3_labels_.pop_front();
    ++sam3_camera_label_queue_drops_;
  }
  pending_sam3_labels_.push_back(message);
}

void HybridLocalizationNode::drainPendingSam3CameraLabels()
{
  const auto stamp = [](const sensor_msgs::ImageConstPtr &message)
  {
    return message && !message->header.stamp.isZero() ?
        message->header.stamp.toSec() :
        -std::numeric_limits<double>::infinity();
  };
  sensor_msgs::ImageConstPtr latest;
  std::size_t received = 0U;
  {
    std::lock_guard<std::mutex> lock(pending_sam3_label_mutex_);
    if (pending_sam3_labels_.empty()) return;
    const auto latest_iterator = std::max_element(
        pending_sam3_labels_.begin(), pending_sam3_labels_.end(),
        [&stamp](const sensor_msgs::ImageConstPtr &left,
                 const sensor_msgs::ImageConstPtr &right)
        {
          return stamp(left) < stamp(right);
        });
    const double source_stamp = stamp(*latest_iterator) + camera_time_offset_;
    if (std::isfinite(source_stamp) &&
        last_camera_stamp_ + sam3_label_sync_tolerance_sec_ < source_stamp)
    {
      return;
    }
    latest = *latest_iterator;
    received = pending_sam3_labels_.size();
    pending_sam3_labels_.clear();
  }
  sam3_camera_labels_received_ += received - 1U;
  processSam3CameraLabel(latest);
}

void HybridLocalizationNode::processSam3CameraLabel(
    const sensor_msgs::ImageConstPtr &message)
{
  ++sam3_camera_labels_received_;
  if (!message) return;
  const cv::Mat labels = mono8Image(*message);
  if (labels.empty() || camera_frame_cache_.empty())
  {
    ++sam3_flow_failures_;
    return;
  }
  const double source_stamp =
      (message->header.stamp.isZero() ? ros::Time::now().toSec()
                                      : message->header.stamp.toSec()) +
      camera_time_offset_;
  if (source_stamp <= sam3_source_stamp_ + 1e-8) return;

  auto best = camera_frame_cache_.begin();
  double best_difference = std::numeric_limits<double>::infinity();
  for (auto iterator = camera_frame_cache_.begin();
       iterator != camera_frame_cache_.end(); ++iterator)
  {
    const double difference = std::abs(iterator->stamp - source_stamp);
    if (difference < best_difference)
    {
      best = iterator;
      best_difference = difference;
    }
  }
  if (best_difference > sam3_label_sync_tolerance_sec_)
  {
    ++sam3_flow_failures_;
    ROS_WARN_THROTTLE(2.0,
        "[hybrid_localization] SAM3 camera label has no cached image: dt=%.3f cache=%zu",
        best_difference, camera_frame_cache_.size());
    return;
  }

  cv::resize(labels, propagated_semantic_labels_, best->gray.size(), 0.0, 0.0,
             cv::INTER_NEAREST);
  semantic_propagation_gray_ = best->gray.clone();
  sam3_source_stamp_ = source_stamp;
  sam3_propagated_stamp_ = best->stamp;

  std::vector<const CachedCameraFrame *> replay;
  for (auto iterator = std::next(best); iterator != camera_frame_cache_.end();
       ++iterator)
  {
    replay.push_back(&(*iterator));
  }
  const std::size_t stride = std::max<std::size_t>(
      1U, (replay.size() + static_cast<std::size_t>(sam3_max_replay_frames_) - 1U) /
              static_cast<std::size_t>(sam3_max_replay_frames_));
  for (std::size_t index = 0U; index < replay.size(); index += stride)
  {
    propagateSemanticLabels(replay[index]->gray, replay[index]->stamp);
  }
  if (!replay.empty() &&
      sam3_propagated_stamp_ + 1e-8 < replay.back()->stamp)
  {
    propagateSemanticLabels(replay.back()->gray, replay.back()->stamp);
  }
}

void HybridLocalizationNode::enqueueMeasurement(MeasurementEvent event,
                                                bool process_queue)
{
  if (process_queue)
  {
    // Dedicated ingress callbacks only buffer data. Applying IMU samples here
    // keeps every ESKF/map mutation serialized with LiDAR and image events.
    drainPendingImuSamples();
  }
  if (!std::isfinite(event.stamp))
  {
    std::lock_guard<std::mutex> lock(measurement_mutex_);
    ++measurement_stale_drops_;
    return;
  }
  if (!measurement_scheduler_enabled_)
  {
    if (event.type == MeasurementEventType::LIDAR && event.lidar)
    {
      processLidarMessage(event.lidar);
    }
    else if (event.type == MeasurementEventType::IMAGE && event.image)
    {
      processImageMessage(event.image, event.stamp);
    }
    return;
  }
  {
    std::lock_guard<std::mutex> lock(measurement_mutex_);
    event.sequence = next_measurement_sequence_++;
    if (event.type == MeasurementEventType::LIDAR)
    {
      latest_enqueued_lidar_stamp_ = std::max(
          latest_enqueued_lidar_stamp_, event.stamp);
      ++scheduled_lidar_events_;
    }
    else
    {
      if (camera_observation_interval_sec_ > 0.0 &&
          std::isfinite(latest_enqueued_image_stamp_) &&
          event.stamp < latest_enqueued_image_stamp_ +
              camera_observation_interval_sec_ - 1e-8)
      {
        ++camera_observation_interval_drops_;
        return;
      }
      latest_enqueued_image_stamp_ = std::max(
          latest_enqueued_image_stamp_, event.stamp);
      ++scheduled_image_events_;
    }
    const auto before = [](const MeasurementEvent &left,
                           const MeasurementEvent &right)
    {
      if (std::abs(left.stamp - right.stamp) > 1e-9)
        return left.stamp < right.stamp;
      if (left.type != right.type)
      {
        return left.type == MeasurementEventType::LIDAR;
      }
      return left.sequence < right.sequence;
    };
    const auto insertion = std::upper_bound(
        measurement_queue_.begin(), measurement_queue_.end(), event,
        [&before](const MeasurementEvent &value,
                  const MeasurementEvent &element)
        {
          return before(value, element);
        });
    measurement_queue_.insert(insertion, std::move(event));
    while (measurement_queue_.size() >
           static_cast<std::size_t>(max_measurement_queue_))
    {
      const auto image = std::find_if(
          measurement_queue_.begin(), measurement_queue_.end(),
          [](const MeasurementEvent &queued)
          {
            return queued.type == MeasurementEventType::IMAGE;
          });
      if (image != measurement_queue_.end())
      {
        measurement_queue_.erase(image);
        ++measurement_image_queue_drops_;
      }
      else
      {
        measurement_queue_.pop_front();
      }
      ++measurement_queue_drops_;
    }
  }
  if (process_queue) drainMeasurementQueue();
}

void HybridLocalizationNode::drainMeasurementQueue(std::size_t max_events)
{
  std::size_t popped_events = 0U;
  while (popped_events < max_events)
  {
    drainPendingSam3CameraLabels();
    MeasurementEvent event;
    bool stale = false;
    double required_imu_stamp = -std::numeric_limits<double>::infinity();
    {
      std::lock_guard<std::mutex> lock(measurement_mutex_);
      if (measurement_queue_.empty()) break;
      event = measurement_queue_.front();
      const double sensor_wait = event.type == MeasurementEventType::LIDAR ?
          lidar_imu_wait_sec_ : camera_imu_wait_sec_;
      required_imu_stamp = event.stamp +
          std::max(sensor_wait, measurement_reorder_window_sec_);
      double sensor_watermark = std::numeric_limits<double>::infinity();
      if (subscribe_lidar_)
      {
        sensor_watermark = std::min(sensor_watermark,
                                    latest_enqueued_lidar_stamp_);
      }
      // A delayed image must never hold up the LiDAR/IMU state update. Image
      // events still wait for both streams to preserve their own time order.
      if (event.type == MeasurementEventType::IMAGE &&
          subscribe_camera_frontend_)
      {
        sensor_watermark = std::min(sensor_watermark,
                                    latest_enqueued_image_stamp_);
      }
      const bool sensor_reorder_ready =
          std::isfinite(sensor_watermark) && sensor_watermark + 1e-6 >=
          event.stamp + measurement_reorder_window_sec_;
      if (!sensor_reorder_ready ||
          (subscribe_imu_ &&
           latestReceivedImuStamp() + 1e-6 < required_imu_stamp))
      {
        break;
      }
      measurement_queue_.pop_front();
      ++popped_events;
      stale = event.stamp + 1e-8 < last_processed_measurement_stamp_;
      if (stale)
      {
        ++measurement_stale_drops_;
        if (event.type == MeasurementEventType::LIDAR) ++dropped_clouds_;
      }
    }
    if (stale) continue;
    if (subscribe_imu_)
    {
      drainPendingImuSamples(required_imu_stamp);
    }
    if (event.type == MeasurementEventType::LIDAR && event.lidar)
    {
      processLidarMessage(event.lidar);
    }
    else if (event.type == MeasurementEventType::IMAGE && event.image)
    {
      processImageMessage(event.image, event.stamp);
    }
    last_processed_measurement_stamp_ = std::max(last_processed_measurement_stamp_,
                                                 event.stamp);
  }
}

void HybridLocalizationNode::measurementSchedulerTimer(
    const ros::WallTimerEvent &)
{
  drainMeasurementQueue(static_cast<std::size_t>(
      measurement_scheduler_max_events_per_tick_));
}

void HybridLocalizationNode::processImageMessage(
    const sensor_msgs::CompressedImageConstPtr &message, double corrected_stamp)
{
  if (!message || message->data.empty())
  {
    ++camera_decode_failures_;
    return;
  }
  const cv::Mat encoded(1, static_cast<int>(message->data.size()), CV_8U,
                        const_cast<uint8_t *>(message->data.data()));
  const cv::Mat image = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
  if (image.empty())
  {
    ++camera_decode_failures_;
    return;
  }
  last_camera_stamp_ = corrected_stamp;
  ++processed_image_events_;

  if (!visual_frontend_enabled_) return;
  cv::Mat semantic_labels;
  cv::Mat dynamic_mask;
  if (visual_use_sam3_)
  {
    const cv::Mat flow_gray = semanticFlowGray(image);
    cacheCameraFrame(flow_gray, corrected_stamp);
    if (!propagated_semantic_labels_.empty() &&
        corrected_stamp > sam3_propagated_stamp_ + 1e-8)
    {
      propagateSemanticLabels(flow_gray, corrected_stamp);
    }
    const bool synchronized = !propagated_semantic_labels_.empty() &&
        std::abs(sam3_propagated_stamp_ - corrected_stamp) <=
            sam3_label_sync_tolerance_sec_;
    const bool fresh = std::isfinite(sam3_source_stamp_) &&
        corrected_stamp - sam3_source_stamp_ <= sam3_max_source_age_sec_;
    if (synchronized && fresh)
    {
      semantic_labels = propagated_semantic_labels_.clone();
      cv::compare(semantic_labels, cv::Scalar(dynamic_label_), dynamic_mask,
                  cv::CMP_EQ);
      sam3_dynamic_pixels_ = static_cast<std::uint64_t>(
          cv::countNonZero(dynamic_mask));
      ++sam3_camera_labels_applied_;
    }
  }
  const SparseVisualFrame frame = sparse_visual_map_.prepareFrame(
      corrected_stamp, image, dynamic_mask, semantic_labels);
  if (!frame.valid())
  {
    visual_update_reason_ = "invalid_visual_frame";
    return;
  }

  ++visual_update_attempts_;
  VisualUpdateResult update;
  if (visual_observation_only_)
  {
    const VisualPoseLinearization observation = sparse_visual_map_.linearize(
        frame, lidar_odometry_.pose());
    update.propagated = lidar_odometry_.initialized();
    update.stamp = corrected_stamp;
    update.pose = lidar_odometry_.pose();
    update.velocity = lidar_odometry_.velocity();
    update.landmarks = observation.landmarks;
    update.residuals = observation.residuals;
    update.rmse = observation.rmse;
    update.mean_ncc = observation.mean_ncc;
    update.reason = observation.valid ? "visual_observation_only" :
        observation.reason;
  }
  else
  {
    update = lidar_odometry_.processVisual(
        corrected_stamp,
        [this, &frame](const Eigen::Isometry3d &pose)
        {
          return sparse_visual_map_.linearize(frame, pose);
        });
  }
  visual_update_rmse_ = update.rmse;
  visual_update_ncc_ = update.mean_ncc;
  visual_update_landmarks_ = update.landmarks;
  visual_update_residuals_ = update.residuals;
  visual_update_iterations_ = update.iterations;
  visual_update_reason_ = update.reason;
  if (update.accepted) ++visual_update_accepts_;

  if (lidar_odometry_.initialized())
  {
    sparse_visual_map_.commitFrame(frame, update.pose, update.accepted);
  }
  if (publish_visual_direct_debug_ &&
      visual_direct_debug_pub_.getNumSubscribers() > 0U &&
      !sparse_visual_map_.debugImage().empty())
  {
    visual_direct_debug_pub_.publish(visualDebugImageMessage(
        ros::Time(corrected_stamp), message->header.frame_id,
        sparse_visual_map_.debugImage()));
  }
  if (!visual_observation_only_)
  {
    handleVisualUpdate(update);
  }
}

void HybridLocalizationNode::processLidarMessage(
    const sensor_msgs::PointCloud2ConstPtr &message)
{
  const double header_stamp = message->header.stamp.isZero() ? ros::Time::now().toSec()
                                                              : message->header.stamp.toSec();
  const std::vector<CloudPoint> input = decodeCloud(*message, label_field_, max_lidar_points_);
  TimedPointVector timed_points;
  timed_points.reserve(input.size());
  double maximum_point_time = -std::numeric_limits<double>::infinity();
  for (const CloudPoint &source : input)
  {
    const double sensor_range = source.point.norm();
    if (!source.point.allFinite() || sensor_range < local_min_range_ || sensor_range > local_max_range_)
    {
      continue;
    }
    const Eigen::Vector3d body_point = body_from_lidar_ * source.point;
    if (body_point.allFinite())
    {
      TimedPoint timed_point;
      timed_point.point = body_point;
      if (source.has_point_time && lidar_point_time_mode_ != "none")
      {
        timed_point.time_from_scan_end = source.point_time * lidar_point_time_scale_;
        maximum_point_time = std::max(maximum_point_time, timed_point.time_from_scan_end);
      }
      timed_points.push_back(timed_point);
    }
  }

  double stamp = header_stamp;
  if (std::isfinite(maximum_point_time))
  {
    bool use_absolute_point_time = lidar_point_time_mode_ == "absolute";
    const bool point_time_looks_absolute =
        maximum_point_time > 1.0e8 && std::abs(maximum_point_time - header_stamp) < 10.0;
    if (!use_absolute_point_time && point_time_looks_absolute)
    {
      use_absolute_point_time = true;
      ROS_WARN_ONCE("[hybrid_localization] LiDAR point time looks like an absolute timestamp; "
                    "overriding lidar_point_time_mode=relative. Set lidar_point_time_mode=absolute.");
    }
    if (use_absolute_point_time)
    {
      stamp = maximum_point_time;
      for (TimedPoint &point : timed_points)
      {
        point.time_from_scan_end -= stamp;
      }
    }
    else
    {
      stamp = lidar_stamp_is_end_ ? header_stamp : header_stamp + maximum_point_time;
      for (TimedPoint &point : timed_points)
      {
        point.time_from_scan_end -= maximum_point_time;
      }
    }
  }
  if (!lidar_deskew_enabled_)
  {
    for (TimedPoint &point : timed_points)
    {
      point.time_from_scan_end = 0.0;
    }
  }

  ++lidar_registration_attempts_;
  const ros::WallTime registration_start = ros::WallTime::now();
  const LidarOdometryResult result = lidar_odometry_.processScan(timed_points, stamp);
  lidar_processing_ms_ = (ros::WallTime::now() - registration_start).toSec() * 1000.0;
  const PointVector &body_points = result.deskewed_points;
  lidar_registration_rmse_ = result.rmse;
  lidar_registration_inlier_ratio_ = result.inlier_ratio;
  lidar_registration_iterations_ = result.iterations;
  lidar_scan_points_ = result.scan_points;
  lidar_correspondence_sectors_ = result.correspondence_azimuth_sectors;
  lidar_point_knn_fallback_queries_ = result.point_knn_fallback_queries;
  lidar_point_knn_fallback_matches_ = result.point_knn_fallback_matches;
  lidar_registration_degenerate_ = result.degenerate;
  lidar_registration_used_imu_ = result.used_imu;
  lidar_registration_used_wheel_ = result.used_wheel;
  lidar_strong_support_ = result.strong_support;
  lidar_recovery_mode_ = result.recovery_mode;
  lidar_loss_limited_ = result.loss_limited;
  lidar_loss_frozen_ = result.loss_frozen;
  lidar_consecutive_rejections_ = result.consecutive_rejections;
  lidar_imu_initialized_ = result.imu_initialized;
  lidar_imu_init_progress_ = result.imu_init_progress;
  lidar_gyro_bias_ = result.gyro_bias;
  lidar_acceleration_bias_ = result.acceleration_bias;
  lidar_gravity_ = result.gravity;
  lidar_registration_status_ = result.reject_reason;
  if (!lidar_odometry_.initialized())
  {
    ++dropped_clouds_;
    ROS_WARN_THROTTLE(2.0,
                      "[hybrid_localization] raw LiDAR frame rejected before initialization: %s "
                      "raw_points=%zu filtered_points=%zu deskewed_points=%zu",
                      result.reject_reason.c_str(), input.size(), timed_points.size(), body_points.size());
    return;
  }
  if (result.accepted)
  {
    ++lidar_registration_accepts_;
    if (visual_frontend_enabled_ && !body_points.empty())
    {
      sparse_visual_map_.addLidarFrame(stamp, result.pose, body_points);
    }
  }
  else
  {
    ROS_WARN_THROTTLE(2.0,
                      "[hybrid_localization] LiDAR registration degraded: %s scan=%d corr=%d sectors=%d "
                      "knn=%d/%d rmse=%.3f strong=%d recovery=%d rejections=%d loss_limited=%d loss_frozen=%d",
                      result.reject_reason.c_str(), result.scan_points, result.correspondences,
                      result.correspondence_azimuth_sectors,
                      result.point_knn_fallback_matches, result.point_knn_fallback_queries, result.rmse,
                      result.strong_support ? 1 : 0, result.recovery_mode ? 1 : 0,
                      result.consecutive_rejections, result.loss_limited ? 1 : 0,
                      result.loss_frozen ? 1 : 0);
  }

  handleLidarOdometry(result);
  if (!result.accepted)
  {
    return;
  }

  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> points;
  points.reserve(body_points.size());
  for (const Eigen::Vector3d &body_point : body_points)
  {
    const Eigen::Vector3d odom_point = result.pose * body_point;
    const double relative_z = odom_point.z() - result.pose.translation().z();
    if (!odom_point.allFinite() || relative_z < local_ground_z_ || relative_z > local_max_z_)
    {
      continue;
    }
    BevPoint point;
    point.point = odom_point;
    point.label = 0U;
    point.confidence = 1.0F;
    point.dynamic = false;
    points.push_back(point);
  }
  if (!points.empty())
  {
    appendLocalFrame(stamp, points, false);
  }
  registered_cloud_pub_.publish(makeRegisteredCloudMessage(
      ros::Time(stamp), body_points, global_from_odom_ * result.pose));
}

void HybridLocalizationNode::semanticCallback(const sensor_msgs::PointCloud2ConstPtr &message)
{
  processSemanticCloud(message);
}

void HybridLocalizationNode::processSemanticCloud(const sensor_msgs::PointCloud2ConstPtr &message)
{
  if (!have_odom_)
  {
    ++dropped_clouds_;
    return;
  }
  const double stamp = message->header.stamp.isZero() ? last_odom_stamp_ : message->header.stamp.toSec();
  bool pose_ok = false;
  const Eigen::Isometry3d odom_pose = lookupOdomPose(stamp, &pose_ok);
  if (!pose_ok)
  {
    ++dropped_clouds_;
    ROS_WARN_THROTTLE(5.0, "[hybrid_localization] dropped semantic cloud: no internal LiDAR pose near %.3f",
                      stamp);
    return;
  }
  const std::vector<CloudPoint> input = decodeCloud(*message, label_field_, max_semantic_points_);
  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> points;
  points.reserve(input.size());
  for (const CloudPoint &source : input)
  {
    Eigen::Vector3d body_point;
    Eigen::Vector3d odom_point;
    if (semantic_cloud_in_map_frame_)
    {
      odom_point = global_from_odom_.inverse() * source.point;
      body_point = odom_pose.inverse() * odom_point;
    }
    else
    {
      body_point = body_from_lidar_ * source.point;
      odom_point = odom_pose * body_point;
    }
    const double sensor_range = body_point.norm();
    if (sensor_range < local_min_range_ || sensor_range > local_max_range_)
    {
      continue;
    }
    const double relative_z = odom_point.z() - odom_pose.translation().z();
    if (!odom_point.allFinite() || relative_z < local_ground_z_ || relative_z > local_max_z_)
    {
      continue;
    }
    BevPoint point;
    point.point = odom_point;
    point.label = normalizeLabel(source.label);
    point.confidence = std::max(0.05F, std::min(1.0F, source.confidence));
    point.dynamic = static_cast<int>(point.label) == dynamic_label_;
    points.push_back(point);
  }
  if (points.empty())
  {
    return;
  }
  appendLocalFrame(stamp, points, true);
  ++semantic_clouds_received_;
  semantic_points_received_ += static_cast<std::uint64_t>(points.size());
  semantic_observation_pub_.publish(makeSemanticPointMessage(ros::Time(stamp), points));
  last_semantic_stamp_ = stamp;
  addSemanticObjects(stamp, points);
}

void HybridLocalizationNode::appendLocalFrame(
    double stamp, const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points,
    bool semantic)
{
  LocalFrame frame;
  frame.stamp = stamp;
  frame.points = points;
  std::deque<LocalFrame> &frames = semantic ? semantic_frames_ : local_frames_;
  const int maximum_points = semantic ? max_local_semantic_points_ : max_local_points_;
  frames.push_back(std::move(frame));
  while (!frames.empty() && stamp - frames.front().stamp > local_map_window_sec_)
  {
    frames.pop_front();
  }
  std::size_t total = 0;
  for (const LocalFrame &item : frames)
  {
    total += item.points.size();
  }
  while (total > static_cast<std::size_t>(std::max(1, maximum_points)) && !frames.empty())
  {
    total -= frames.front().points.size();
    frames.pop_front();
  }
}

void HybridLocalizationNode::addSemanticObjects(
    double stamp, const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points)
{
  struct Observation
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    uint8_t label = 0U;
    bool dynamic = false;
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_squared = Eigen::Vector3d::Zero();
    double weight = 0.0;
  };
  std::map<long long, Observation> observations;
  const Eigen::Isometry3d global_from_odom = global_from_odom_;
  for (const BevPoint &point : points)
  {
    if (point.label == 0U || point.confidence < object_min_confidence_)
    {
      continue;
    }
    const bool dynamic = static_cast<int>(point.label) == dynamic_label_;
    if (dynamic && !include_dynamic_objects_)
    {
      continue;
    }
    const Eigen::Vector3d global_point = global_from_odom * point.point;
    const int cell_x = static_cast<int>(std::floor(global_point.x() / std::max(0.25, object_assoc_radius_)));
    const int cell_y = static_cast<int>(std::floor(global_point.y() / std::max(0.25, object_assoc_radius_)));
    long long key = static_cast<long long>(point.label) << 48;
    key ^= (static_cast<long long>(cell_x) & 0xFFFFFFLL) << 24;
    key ^= (static_cast<long long>(cell_y) & 0xFFFFFFLL);
    Observation &observation = observations[key];
    observation.label = point.label;
    observation.dynamic = dynamic;
    observation.sum += point.confidence * global_point;
    observation.sum_squared += point.confidence * global_point.cwiseProduct(global_point);
    observation.weight += point.confidence;
  }
  for (const auto &entry : observations)
  {
    const Observation &observation = entry.second;
    if (observation.weight < 1.0)
    {
      continue;
    }
    const Eigen::Vector3d mean = observation.sum / observation.weight;
    Eigen::Vector3d variance = observation.sum_squared / observation.weight - mean.cwiseProduct(mean);
    variance = variance.cwiseMax(Eigen::Vector3d::Constant(0.01));
    int associated = -1;
    double best_distance = object_assoc_radius_;
    for (std::size_t i = 0; i < objects_.size(); ++i)
    {
      if (objects_[i].label != observation.label || objects_[i].dynamic != observation.dynamic)
      {
        continue;
      }
      const double distance = (objects_[i].mean - mean).norm();
      if (distance < best_distance)
      {
        best_distance = distance;
        associated = static_cast<int>(i);
      }
    }
    if (associated < 0)
    {
      if (static_cast<int>(objects_.size()) >= max_objects_)
      {
        continue;
      }
      ObjectState object;
      object.id = next_object_id_++;
      object.label = observation.label;
      object.mean = mean;
      object.covariance = variance.asDiagonal();
      object.observations = 1;
      object.last_stamp = stamp;
      object.reliability = std::min(1.0, 0.2 * observation.weight);
      object.dynamic = observation.dynamic;
      object.reachable = !object.dynamic && object.mean.head<2>().norm() <= object_reachable_radius_;
      objects_.push_back(object);
    }
    else
    {
      ObjectState &object = objects_[static_cast<std::size_t>(associated)];
      const double alpha = 1.0 / static_cast<double>(std::min(20, object.observations + 1));
      const Eigen::Vector3d innovation = mean - object.mean;
      object.mean += alpha * innovation;
      object.covariance *= (1.0 - alpha);
      object.covariance.diagonal() += alpha * variance;
      object.observations += 1;
      object.last_stamp = stamp;
      object.reliability = std::min(1.0, object.reliability + 0.1 * alpha * observation.weight);
      object.reachable = !object.dynamic && object.mean.head<2>().norm() <= object_reachable_radius_;
    }
  }
}

std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> HybridLocalizationNode::collectLocalPoints() const
{
  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> result;
  std::size_t total = 0;
  for (const LocalFrame &frame : local_frames_)
  {
    if (last_odom_stamp_ - frame.stamp <= local_map_window_sec_) total += frame.points.size();
  }
  const std::size_t geometry_limit = static_cast<std::size_t>(std::max(1, max_local_points_));
  const std::size_t stride = std::max<std::size_t>(
      1U, (total + geometry_limit - 1U) / geometry_limit);
  result.reserve(std::min(total, geometry_limit) +
                 static_cast<std::size_t>(std::max(1, max_semantic_output_points_)));
  std::size_t index = 0;
  for (const LocalFrame &frame : local_frames_)
  {
    if (last_odom_stamp_ - frame.stamp > local_map_window_sec_) continue;
    for (const BevPoint &point : frame.points)
    {
      if (index++ % stride == 0 && result.size() < geometry_limit)
      {
        result.push_back(point);
      }
    }
  }
  const auto semantic = collectSemanticPoints();
  result.insert(result.end(), semantic.begin(), semantic.end());
  return result;
}

std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>>
HybridLocalizationNode::collectSemanticPoints() const
{
  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> result;
  std::size_t total = 0U;
  for (const LocalFrame &frame : semantic_frames_)
  {
    if (last_odom_stamp_ - frame.stamp <= local_map_window_sec_) total += frame.points.size();
  }
  if (total == 0U) return result;
  const std::size_t limit = static_cast<std::size_t>(
      std::max(1, max_semantic_output_points_));
  const std::size_t stride = std::max<std::size_t>(
      1U, (total + limit - 1U) / limit);
  result.reserve(std::min(total, limit));
  std::size_t index = 0U;
  for (const LocalFrame &frame : semantic_frames_)
  {
    if (last_odom_stamp_ - frame.stamp > local_map_window_sec_) continue;
    for (const BevPoint &point : frame.points)
    {
      if (point.label == 0U) continue;
      if (index++ % stride == 0U && result.size() < limit) result.push_back(point);
    }
  }
  return result;
}

void HybridLocalizationNode::mapMatchTimer(const ros::TimerEvent &)
{
  if (!have_odom_)
  {
    return;
  }
  const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> local_points = collectLocalPoints();
  if (local_points.size() < static_cast<std::size_t>(std::max(1.0, map_match_min_points_)))
  {
    last_map_reject_reason_ = "insufficient_local_bev_points";
    if (publish_debug_)
    {
      MapMatchResult empty;
      empty.reject_reason = last_map_reject_reason_;
      empty.global_from_odom = global_from_odom_;
      publishMapDebug(ros::Time::now(), empty);
    }
    return;
  }
  if (!prior_map_.valid())
  {
    last_map_reject_reason_ = "prior_map_unavailable";
    return;
  }
  ++map_match_attempts_;
  const MapMatchResult result = matcher_.match(local_points, prior_map_, global_from_odom_, position_covariance_);
  last_map_match_ = result;
  map_match_confidence_ = result.confidence;
  map_match_score_ = result.best_score;
  last_score_gap_ = std::isfinite(result.second_score) ? result.best_score - result.second_score : 0.0;
  last_inlier_ratio_ = result.inlier_ratio;
  last_map_reject_reason_ = result.gate_passed ? "accepted" : result.reject_reason;
  if (publish_debug_)
  {
    publishMapDebug(ros::Time::now(), result);
  }
  if (!result.gate_passed || result.confidence < map_factor_min_confidence_)
  {
    return;
  }
  applyMapMatch(result);
  if (publish_debug_)
  {
    publishMapDebug(ros::Time::now(), last_map_match_);
  }
  last_map_match_stamp_ = last_odom_stamp_;
}

void HybridLocalizationNode::applyMapMatch(const MapMatchResult &result)
{
  const Eigen::Isometry3d map_pose = result.global_from_odom * last_keyframe_odom_pose_;
  if (optimizer_.empty())
  {
    global_from_odom_ = hybrid_localization::projectToSE3(result.global_from_odom);
    ++map_match_accepts_;
    return;
  }
  const int latest_index = static_cast<int>(optimizer_.size() - 1);
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
  information.topLeftCorner<3, 3>() *= map_factor_information_ * result.confidence;
  information.bottomRightCorner<3, 3>() *= map_factor_information_ * result.confidence;
  optimizer_.addAbsoluteFactor(latest_index, map_pose, information, FactorType::MapMatch,
                               result.confidence, map_factor_huber_delta_);
  ++factor_count_;
  optimizeCurrentWindow();
  ++map_match_accepts_;
  MapMatchResult applied = result;
  for (MatchPair &pair : applied.pairs)
  {
    pair.applied = pair.inlier;
  }
  last_map_match_ = applied;
  updatePriorMap(applied);
}

void HybridLocalizationNode::updatePriorMap(const MapMatchResult &result)
{
  if (!prior_update_enabled_ || !prior_map_.valid() || result.confidence < prior_update_min_confidence_)
  {
    return;
  }
  const float alpha = static_cast<float>(std::max(0.0, std::min(1.0, prior_update_alpha_ * result.confidence)));
  for (const MatchPair &pair : result.pairs)
  {
    if (!pair.inlier)
    {
      continue;
    }
    const Eigen::Vector3d global_point = result.global_from_odom * pair.source;
    int ix = 0;
    int iy = 0;
    if (!prior_map_.worldToCell(global_point.x(), global_point.y(), ix, iy))
    {
      continue;
    }
    const std::size_t cell = static_cast<std::size_t>(iy) * static_cast<std::size_t>(prior_map_.width) +
                             static_cast<std::size_t>(ix);
    prior_map_.occupancy[cell] = (1.0F - alpha) * prior_map_.occupancy[cell] + alpha;
    if (pair.source_label != 0U)
    {
      prior_map_.labels[cell] = pair.source_label;
    }
  }
  prior_map_.recomputeEdges();
  prior_map_pub_.publish(makePriorGridMessage(ros::Time::now()));
}

void HybridLocalizationNode::publishMapDebug(const ros::Time &stamp, const MapMatchResult &result)
{
  visualization_msgs::MarkerArray markers;
  visualization_msgs::Marker clear;
  clear.header.stamp = stamp;
  clear.header.frame_id = map_frame_;
  clear.ns = "hybrid_map_match";
  clear.id = 0;
  clear.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(clear);

  const std::vector<std::string> states = {"candidate", "inlier", "outlier", "applied"};
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index)
  {
    visualization_msgs::Marker line;
    line.header.stamp = stamp;
    line.header.frame_id = map_frame_;
    line.ns = "hybrid_map_match/" + states[state_index];
    line.id = static_cast<int>(state_index) + 1;
    line.type = visualization_msgs::Marker::LINE_LIST;
    line.action = visualization_msgs::Marker::ADD;
    line.pose.orientation.w = 1.0;
    line.scale.x = state_index == 0 ? 0.015 : 0.035;
    line.color = colorForState(states[state_index]);
    for (const MatchPair &pair : result.pairs)
    {
      const bool show = state_index == 0 ? pair.candidate :
          state_index == 1 ? pair.inlier : state_index == 2 ? pair.outlier : pair.applied;
      if (!show)
      {
        continue;
      }
      const Eigen::Vector3d source = result.global_from_odom * pair.source;
      geometry_msgs::Point source_point;
      source_point.x = source.x(); source_point.y = source.y(); source_point.z = source.z();
      geometry_msgs::Point target_point;
      target_point.x = pair.target.x(); target_point.y = pair.target.y(); target_point.z = pair.target.z();
      line.points.push_back(source_point);
      line.points.push_back(target_point);
    }
    markers.markers.push_back(line);
  }
  visualization_msgs::Marker text;
  text.header.stamp = stamp;
  text.header.frame_id = map_frame_;
  text.ns = "hybrid_map_match/status";
  text.id = 20;
  text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text.action = visualization_msgs::Marker::ADD;
  text.pose.position.x = latest_state_.pose.translation().x();
  text.pose.position.y = latest_state_.pose.translation().y();
  text.pose.position.z = latest_state_.pose.translation().z() + 1.5;
  text.pose.orientation.w = 1.0;
  text.scale.z = 0.35;
  text.color.r = 1.0F;
  text.color.g = 1.0F;
  text.color.b = 1.0F;
  text.color.a = 1.0F;
  std::ostringstream description;
  description << (result.gate_passed ? "ACCEPT" : "REJECT")
              << " score=" << result.best_score
              << " gap=" << (std::isfinite(result.second_score) ? result.best_score - result.second_score : 0.0)
              << " conf=" << result.confidence
              << " inlier=" << result.inlier_ratio
              << " reason=" << (result.reject_reason.empty() ? "none" : result.reject_reason);
  text.text = description.str();
  markers.markers.push_back(text);
  debug_marker_pub_.publish(markers);
}

nav_msgs::Odometry HybridLocalizationNode::makeOutputOdometry(const ros::Time &stamp,
                                                              const Eigen::Isometry3d &pose) const
{
  nav_msgs::Odometry output;
  output.header.stamp = stamp;
  output.header.frame_id = map_frame_;
  output.child_frame_id = body_frame_;
  output.pose.pose = poseMessage(pose);
  output.twist.twist.linear.x = latest_state_.velocity.x();
  output.twist.twist.linear.y = latest_state_.velocity.y();
  output.twist.twist.linear.z = latest_state_.velocity.z();
  for (int i = 0; i < 3; ++i)
  {
    output.pose.covariance[static_cast<std::size_t>(i * 6 + i)] = latest_state_.covariance(3 + i, 3 + i);
    output.pose.covariance[static_cast<std::size_t>((i + 3) * 6 + i + 3)] = latest_state_.covariance(i, i);
    output.twist.covariance[static_cast<std::size_t>(i * 6 + i)] = latest_state_.covariance(6 + i, 6 + i);
  }
  return output;
}

void HybridLocalizationNode::appendPoseToPath(nav_msgs::Path &path, const nav_msgs::Odometry &odom,
                                               std::size_t max_poses)
{
  if (!path.poses.empty() &&
      std::abs(path.poses.back().header.stamp.toSec() - odom.header.stamp.toSec()) < 1e-9)
  {
    path.poses.back().pose = odom.pose.pose;
    return;
  }
  geometry_msgs::PoseStamped pose;
  pose.header = odom.header;
  pose.pose = odom.pose.pose;
  path.poses.push_back(pose);
  while (path.poses.size() > max_poses)
  {
    path.poses.erase(path.poses.begin());
  }
}

void HybridLocalizationNode::publishCurrentState(const ros::Time &stamp)
{
  if (!have_odom_)
  {
    return;
  }
  const Eigen::Isometry3d global_pose = hybrid_localization::projectToSE3(
      global_from_odom_ * latest_odom_pose_);
  latest_state_.pose = global_pose;
  const nav_msgs::Odometry output = makeOutputOdometry(stamp, global_pose);
  output_odom_pub_.publish(output);
  appendPoseToPath(output_path_, output, 20000);
  output_path_.header.stamp = stamp;
  output_path_.header.frame_id = map_frame_;
  output_path_pub_.publish(output_path_);
  if (trajectory_stream_.is_open() &&
      stamp.toSec() > last_saved_trajectory_stamp_ + 1e-6)
  {
    const Eigen::Quaterniond saved_q(global_pose.rotation());
    trajectory_stream_ << std::fixed << stamp.toSec() << ","
                       << global_pose.translation().x() << ","
                       << global_pose.translation().y() << ","
                       << global_pose.translation().z() << ","
                       << saved_q.x() << "," << saved_q.y() << ","
                       << saved_q.z() << "," << saved_q.w() << "\n";
    trajectory_stream_.flush();
    last_saved_trajectory_stamp_ = stamp.toSec();
  }
  if (broadcast_tf_)
  {
    geometry_msgs::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = map_frame_;
    transform.child_frame_id = odom_frame_;
    const Eigen::Quaterniond q(hybrid_localization::projectToSE3(global_from_odom_).rotation());
    transform.transform.translation.x = global_from_odom_.translation().x();
    transform.transform.translation.y = global_from_odom_.translation().y();
    transform.transform.translation.z = global_from_odom_.translation().z();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();
    tf_broadcaster_.sendTransform(transform);

    geometry_msgs::TransformStamped body_transform;
    body_transform.header.stamp = stamp;
    body_transform.header.frame_id = odom_frame_;
    body_transform.child_frame_id = body_frame_;
    const Eigen::Quaterniond body_q(
        hybrid_localization::projectToSE3(latest_odom_pose_).rotation());
    body_transform.transform.translation.x = latest_odom_pose_.translation().x();
    body_transform.transform.translation.y = latest_odom_pose_.translation().y();
    body_transform.transform.translation.z = latest_odom_pose_.translation().z();
    body_transform.transform.rotation.x = body_q.x();
    body_transform.transform.rotation.y = body_q.y();
    body_transform.transform.rotation.z = body_q.z();
    body_transform.transform.rotation.w = body_q.w();
    tf_broadcaster_.sendTransform(body_transform);
  }
  last_publish_stamp_ = stamp;
}

nav_msgs::OccupancyGrid HybridLocalizationNode::makeLocalGridMessage(const ros::Time &stamp) const
{
  nav_msgs::OccupancyGrid grid;
  grid.header.stamp = stamp;
  grid.header.frame_id = map_frame_;
  grid.info.resolution = static_cast<float>(local_bev_.resolution);
  grid.info.width = static_cast<uint32_t>(local_bev_.width);
  grid.info.height = static_cast<uint32_t>(local_bev_.height);
  grid.info.origin.position.x = local_bev_.origin_x;
  grid.info.origin.position.y = local_bev_.origin_y;
  grid.info.origin.orientation.w = 1.0;
  grid.data.resize(local_bev_.occupancy.size(), -1);
  for (std::size_t index = 0; index < local_bev_.occupancy.size(); ++index)
  {
    if (local_bev_.occupancy[index] > 0.35F)
    {
      grid.data[index] = 100;
    }
    else if (local_bev_.quality[index] > 0.0F)
    {
      grid.data[index] = 0;
    }
  }
  return grid;
}

nav_msgs::OccupancyGrid HybridLocalizationNode::makePriorGridMessage(const ros::Time &stamp) const
{
  nav_msgs::OccupancyGrid grid;
  grid.header.stamp = stamp;
  grid.header.frame_id = map_frame_;
  grid.info.resolution = static_cast<float>(prior_map_.resolution);
  grid.info.width = static_cast<uint32_t>(std::max(0, prior_map_.width));
  grid.info.height = static_cast<uint32_t>(std::max(0, prior_map_.height));
  grid.info.origin.position.x = prior_map_.origin_x;
  grid.info.origin.position.y = prior_map_.origin_y;
  grid.info.origin.orientation.w = 1.0;
  grid.data.resize(prior_map_.occupancy.size(), -1);
  for (std::size_t index = 0; index < prior_map_.occupancy.size(); ++index)
  {
    grid.data[index] = static_cast<int8_t>(std::max(0.0, std::min(100.0,
        100.0 * static_cast<double>(prior_map_.occupancy[index]))));
  }
  return grid;
}

sensor_msgs::PointCloud2 HybridLocalizationNode::makeSemanticCloudMessage(const ros::Time &stamp) const
{
  return makeSemanticPointMessage(stamp, collectSemanticPoints());
}

sensor_msgs::PointCloud2 HybridLocalizationNode::makeSemanticPointMessage(
    const ros::Time &stamp,
    const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points) const
{
  sensor_msgs::PointCloud2 message;
  message.header.stamp = stamp;
  message.header.frame_id = map_frame_;
  message.height = 1;
  message.width = static_cast<uint32_t>(points.size());
  message.is_dense = false;
  message.is_bigendian = false;
  message.point_step = 20;
  message.row_step = message.point_step * message.width;
  message.fields.resize(5);
  message.fields[0].name = "x"; message.fields[0].offset = 0; message.fields[0].datatype = sensor_msgs::PointField::FLOAT32; message.fields[0].count = 1;
  message.fields[1].name = "y"; message.fields[1].offset = 4; message.fields[1].datatype = sensor_msgs::PointField::FLOAT32; message.fields[1].count = 1;
  message.fields[2].name = "z"; message.fields[2].offset = 8; message.fields[2].datatype = sensor_msgs::PointField::FLOAT32; message.fields[2].count = 1;
  message.fields[3].name = "label"; message.fields[3].offset = 12; message.fields[3].datatype = sensor_msgs::PointField::UINT32; message.fields[3].count = 1;
  message.fields[4].name = "confidence"; message.fields[4].offset = 16; message.fields[4].datatype = sensor_msgs::PointField::FLOAT32; message.fields[4].count = 1;
  message.data.resize(static_cast<std::size_t>(message.row_step) * message.height, 0U);
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    uint8_t *data = message.data.data() + i * message.point_step;
    const Eigen::Vector3d point = global_from_odom_ * points[i].point;
    const float x = static_cast<float>(point.x());
    const float y = static_cast<float>(point.y());
    const float z = static_cast<float>(point.z());
    const uint32_t label = points[i].label;
    const float confidence = points[i].confidence;
    std::memcpy(data + 0, &x, sizeof(x));
    std::memcpy(data + 4, &y, sizeof(y));
    std::memcpy(data + 8, &z, sizeof(z));
    std::memcpy(data + 12, &label, sizeof(label));
    std::memcpy(data + 16, &confidence, sizeof(confidence));
  }
  return message;
}

sensor_msgs::PointCloud2 HybridLocalizationNode::makeRegisteredCloudMessage(
    const ros::Time &stamp, const PointVector &body_points,
    const Eigen::Isometry3d &global_pose) const
{
  sensor_msgs::PointCloud2 message;
  message.header.stamp = stamp;
  message.header.frame_id = map_frame_;
  message.height = 1;
  message.is_bigendian = false;
  message.is_dense = false;
  sensor_msgs::PointCloud2Modifier modifier(message);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(body_points.size());
  sensor_msgs::PointCloud2Iterator<float> x(message, "x");
  sensor_msgs::PointCloud2Iterator<float> y(message, "y");
  sensor_msgs::PointCloud2Iterator<float> z(message, "z");
  for (const Eigen::Vector3d &body_point : body_points)
  {
    const Eigen::Vector3d point = global_pose * body_point;
    *x = static_cast<float>(point.x());
    *y = static_cast<float>(point.y());
    *z = static_cast<float>(point.z());
    ++x;
    ++y;
    ++z;
  }
  return message;
}

void HybridLocalizationNode::publishLocalBev(const ros::Time &stamp)
{
  if (!publish_local_bev_)
  {
    return;
  }
  const Eigen::Isometry3d current_pose = global_from_odom_ * latest_odom_pose_;
  local_bev_.reset(static_cast<int>(local_map_size_m_ / local_map_resolution_),
                   static_cast<int>(local_map_size_m_ / local_map_resolution_),
                   local_map_resolution_, current_pose.translation().x(), current_pose.translation().y());
  const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> points = collectLocalPoints();
  for (const BevPoint &source : points)
  {
    BevPoint point = source;
    point.point = global_from_odom_ * source.point;
    local_bev_.insert(point, local_ground_z_, local_max_z_ + current_pose.translation().z());
  }
  local_bev_pub_.publish(makeLocalGridMessage(stamp));
  if (prior_map_.valid())
  {
    prior_map_pub_.publish(makePriorGridMessage(stamp));
  }
  semantic_cloud_pub_.publish(makeSemanticCloudMessage(stamp));
}

void HybridLocalizationNode::publishObjects(const ros::Time &stamp)
{
  if (!publish_objects_)
  {
    return;
  }
  geometry_msgs::PoseArray poses;
  poses.header.stamp = stamp;
  poses.header.frame_id = map_frame_;
  visualization_msgs::MarkerArray markers;
  visualization_msgs::Marker clear;
  clear.header = poses.header;
  clear.ns = "hybrid_objects";
  clear.id = 0;
  clear.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(clear);
  std_msgs::Float32MultiArray array;
  for (const ObjectState &object : objects_)
  {
    geometry_msgs::Pose pose;
    pose.position.x = object.mean.x();
    pose.position.y = object.mean.y();
    pose.position.z = object.mean.z();
    pose.orientation.w = 1.0;
    poses.poses.push_back(pose);
    visualization_msgs::Marker sphere;
    sphere.header = poses.header;
    sphere.ns = "hybrid_objects/point";
    sphere.id = object.id;
    sphere.type = visualization_msgs::Marker::SPHERE;
    sphere.action = visualization_msgs::Marker::ADD;
    sphere.pose = pose;
    sphere.scale.x = 0.35; sphere.scale.y = 0.35; sphere.scale.z = 0.35;
    sphere.color.a = static_cast<float>(std::max(0.25, object.reliability));
    sphere.color.r = object.label == 3U ? 0.95F : object.label == 4U ? 0.2F : 0.2F;
    sphere.color.g = object.label == 1U || object.label == 2U ? 0.9F : 0.5F;
    sphere.color.b = object.label == 3U ? 0.2F : 0.9F;
    markers.markers.push_back(sphere);
    visualization_msgs::Marker text;
    text.header = poses.header;
    text.ns = "hybrid_objects/label";
    text.id = 100000 + object.id;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::Marker::ADD;
    text.pose = pose;
    text.pose.position.z += 0.45;
    text.scale.z = 0.25;
    text.color.a = 1.0F; text.color.r = 1.0F; text.color.g = 1.0F; text.color.b = 1.0F;
    std::ostringstream label;
    label << "id=" << object.id << " l=" << static_cast<int>(object.label)
          << " q=" << object.reliability << (object.reachable ? " reachable" : "");
    text.text = label.str();
    markers.markers.push_back(text);
    array.data.push_back(static_cast<float>(object.id));
    array.data.push_back(static_cast<float>(object.label));
    array.data.push_back(static_cast<float>(object.mean.x()));
    array.data.push_back(static_cast<float>(object.mean.y()));
    array.data.push_back(static_cast<float>(object.mean.z()));
    array.data.push_back(static_cast<float>(object.covariance(0, 0)));
    array.data.push_back(static_cast<float>(object.covariance(1, 1)));
    array.data.push_back(static_cast<float>(object.covariance(2, 2)));
    array.data.push_back(static_cast<float>(object.observations));
    array.data.push_back(static_cast<float>(object.reliability));
    array.data.push_back(object.reachable ? 1.0F : 0.0F);
  }
  object_pose_pub_.publish(poses);
  object_marker_pub_.publish(markers);
  object_array_pub_.publish(array);
}

void HybridLocalizationNode::publishStatus(const ros::Time &stamp)
{
  const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> local_points = collectLocalPoints();
  const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> semantic_points =
      collectSemanticPoints();
  const SparseVisualMapStats visual_stats = sparse_visual_map_.stats();
  std::size_t measurement_queue_size = 0U;
  std::uint64_t scheduled_lidar = 0U;
  std::uint64_t scheduled_images = 0U;
  std::uint64_t scheduler_queue_drops = 0U;
  std::uint64_t scheduler_image_queue_drops = 0U;
  std::uint64_t camera_interval_drops = 0U;
  std::uint64_t scheduler_stale_drops = 0U;
  std::size_t pending_sam3_labels = 0U;
  std::uint64_t sam3_label_queue_drops = 0U;
  {
    std::lock_guard<std::mutex> lock(measurement_mutex_);
    measurement_queue_size = measurement_queue_.size();
    scheduled_lidar = scheduled_lidar_events_;
    scheduled_images = scheduled_image_events_;
    scheduler_queue_drops = measurement_queue_drops_;
    scheduler_image_queue_drops = measurement_image_queue_drops_;
    camera_interval_drops = camera_observation_interval_drops_;
    scheduler_stale_drops = measurement_stale_drops_;
  }
  {
    std::lock_guard<std::mutex> lock(pending_sam3_label_mutex_);
    pending_sam3_labels = pending_sam3_labels_.size();
    sam3_label_queue_drops = sam3_camera_label_queue_drops_;
  }
  std::ostringstream status;
  const bool lidar_healthy = have_odom_ &&
      (lidar_registration_status_ == "accepted" || lidar_registration_status_ == "initialized");
  status << "state=" << (have_odom_ ? (lidar_healthy ? "running" : "degraded")
                                      : "waiting_raw_lidar")
         << ";prior=" << (prior_map_.valid() ? "ready" : "unavailable")
         << ";lio_status=" << lidar_registration_status_
         << ";lio_attempts=" << lidar_registration_attempts_
         << ";lio_accepts=" << lidar_registration_accepts_
         << ";lio_rmse=" << lidar_registration_rmse_
         << ";lio_inlier_ratio=" << lidar_registration_inlier_ratio_
         << ";lio_iterations=" << lidar_registration_iterations_
         << ";lio_processing_ms=" << lidar_processing_ms_
         << ";lio_scan_points=" << lidar_scan_points_
         << ";lio_correspondence_sectors=" << lidar_correspondence_sectors_
         << ";lio_knn_fallback_queries=" << lidar_point_knn_fallback_queries_
         << ";lio_knn_fallback_matches=" << lidar_point_knn_fallback_matches_
         << ";lio_map_points=" << lidar_odometry_.mapPointCount()
         << ";lio_degenerate=" << (lidar_registration_degenerate_ ? 1 : 0)
         << ";lio_strong_support=" << (lidar_strong_support_ ? 1 : 0)
         << ";lio_recovery_mode=" << (lidar_recovery_mode_ ? 1 : 0)
         << ";lio_used_imu=" << (lidar_registration_used_imu_ ? 1 : 0)
         << ";lio_used_wheel=" << (lidar_registration_used_wheel_ ? 1 : 0)
         << ";lio_consecutive_rejections=" << lidar_consecutive_rejections_
         << ";lio_loss_limited=" << (lidar_loss_limited_ ? 1 : 0)
         << ";lio_loss_frozen=" << (lidar_loss_frozen_ ? 1 : 0)
         << ";wheel_samples=" << wheel_samples_
         << ";wheel_speed=" << last_wheel_speed_
         << ";imu_initialized=" << (lidar_imu_initialized_ ? 1 : 0)
         << ";imu_init_progress=" << lidar_imu_init_progress_
         << ";gyro_bias=" << lidar_gyro_bias_.transpose()
         << ";acceleration_bias=" << lidar_acceleration_bias_.transpose()
         << ";gravity=" << lidar_gravity_.transpose()
         << ";measurement_queue=" << measurement_queue_size
         << ";pending_imu_queue=" << pendingImuSampleCount()
         << ";pending_imu_queue_drops=" << pendingImuQueueDrops()
         << ";scheduled_lidar=" << scheduled_lidar
         << ";scheduled_images=" << scheduled_images
         << ";processed_images=" << processed_image_events_
         << ";scheduler_queue_drops=" << scheduler_queue_drops
         << ";scheduler_stale_drops=" << scheduler_stale_drops
         << ";camera_decode_failures=" << camera_decode_failures_
         << ";camera_interval_drops=" << camera_interval_drops
         << ";last_camera_stamp=" << last_camera_stamp_
         << ";visual_enabled=" << (visual_frontend_enabled_ ? 1 : 0)
         << ";visual_observation_only=" << (visual_observation_only_ ? 1 : 0)
         << ";visual_attempts=" << visual_update_attempts_
         << ";visual_accepts=" << visual_update_accepts_
         << ";visual_landmarks=" << visual_stats.landmarks
         << ";visual_active_landmarks=" << visual_update_landmarks_
         << ";visual_residuals=" << visual_update_residuals_
         << ";visual_rmse=" << visual_update_rmse_
         << ";visual_mean_ncc=" << visual_update_ncc_
         << ";visual_iterations=" << visual_update_iterations_
         << ";visual_reason=" << visual_update_reason_
         << ";visual_dynamic_rejections=" << visual_stats.dynamic_rejections
         << ";sam3_camera_labels_received=" << sam3_camera_labels_received_
         << ";sam3_camera_labels_applied=" << sam3_camera_labels_applied_
         << ";pending_sam3_labels=" << pending_sam3_labels
         << ";sam3_label_queue_drops=" << sam3_label_queue_drops
         << ";sam3_flow_propagations=" << sam3_flow_propagations_
         << ";sam3_flow_failures=" << sam3_flow_failures_
         << ";sam3_dynamic_pixels=" << sam3_dynamic_pixels_
         << ";sam3_source_age=" << (std::isfinite(sam3_source_stamp_) ?
              stamp.toSec() - sam3_source_stamp_ :
              std::numeric_limits<double>::infinity())
         << ";local_points=" << local_points.size()
         << ";semantic_points=" << semantic_points.size()
         << ";semantic_clouds=" << semantic_clouds_received_
         << ";semantic_points_received=" << semantic_points_received_
         << ";objects=" << objects_.size()
         << ";factors=" << factor_count_
         << ";map_attempts=" << map_match_attempts_
         << ";map_accepts=" << map_match_accepts_
         << ";map_confidence=" << map_match_confidence_
         << ";inlier_ratio=" << last_inlier_ratio_
         << ";cov_trace=" << position_covariance_.trace()
         << ";reject_reason=" << last_map_reject_reason_;
  std_msgs::String message;
  message.data = status.str();
  status_pub_.publish(message);

  std_msgs::Float64MultiArray quality;
  quality.data = {map_match_confidence_, last_inlier_ratio_, position_covariance_.trace(),
                  static_cast<double>(local_points.size()), static_cast<double>(objects_.size()),
                  static_cast<double>(factor_count_), static_cast<double>(map_match_attempts_),
                  static_cast<double>(map_match_accepts_), lidar_registration_rmse_,
                  lidar_registration_inlier_ratio_, static_cast<double>(lidar_odometry_.mapPointCount()),
                  static_cast<double>(lidar_registration_accepts_), lidar_processing_ms_,
                  static_cast<double>(visual_update_accepts_), visual_update_rmse_,
                  visual_update_ncc_, static_cast<double>(visual_stats.landmarks)};
  quality_pub_.publish(quality);

  diagnostic_msgs::DiagnosticArray diagnostics;
  diagnostics.header.stamp = stamp;
  diagnostic_msgs::DiagnosticStatus state;
  state.name = "hybrid_localization/state";
  state.hardware_id = "hybrid_localization";
  state.level = lidar_healthy ? diagnostic_msgs::DiagnosticStatus::OK
                              : diagnostic_msgs::DiagnosticStatus::WARN;
  state.message = have_odom_ ? (lidar_healthy ? "running" : "LiDAR odometry degraded")
                             : "waiting for raw LiDAR";
  state.values.resize(26);
  state.values[0].key = "prior_map"; state.values[0].value = prior_map_.valid() ? "ready" : "unavailable";
  state.values[1].key = "local_points"; state.values[1].value = std::to_string(local_points.size());
  state.values[2].key = "objects"; state.values[2].value = std::to_string(objects_.size());
  state.values[3].key = "map_confidence"; state.values[3].value = std::to_string(map_match_confidence_);
  state.values[4].key = "reject_reason"; state.values[4].value = last_map_reject_reason_;
  state.values[5].key = "lidar_odometry"; state.values[5].value = lidar_registration_status_;
  state.values[6].key = "lidar_rmse"; state.values[6].value = std::to_string(lidar_registration_rmse_);
  state.values[7].key = "lidar_inlier_ratio"; state.values[7].value = std::to_string(lidar_registration_inlier_ratio_);
  state.values[8].key = "lidar_map_points"; state.values[8].value = std::to_string(lidar_odometry_.mapPointCount());
  state.values[9].key = "lidar_degenerate"; state.values[9].value = lidar_registration_degenerate_ ? "true" : "false";
  state.values[10].key = "imu_initialized"; state.values[10].value = lidar_imu_initialized_ ? "true" : "false";
  state.values[11].key = "imu_init_progress"; state.values[11].value = std::to_string(lidar_imu_init_progress_);
  state.values[12].key = "gyro_bias"; state.values[12].value = std::to_string(lidar_gyro_bias_.norm());
  state.values[13].key = "acceleration_bias"; state.values[13].value = std::to_string(lidar_acceleration_bias_.norm());
  state.values[14].key = "gravity_norm"; state.values[14].value = std::to_string(lidar_gravity_.norm());
  state.values[15].key = "semantic_points"; state.values[15].value = std::to_string(semantic_points.size());
  state.values[16].key = "semantic_clouds"; state.values[16].value = std::to_string(semantic_clouds_received_);
  state.values[17].key = "lidar_consecutive_rejections";
  state.values[17].value = std::to_string(lidar_consecutive_rejections_);
  state.values[18].key = "lidar_loss_limited";
  state.values[18].value = lidar_loss_limited_ ? "true" : "false";
  state.values[19].key = "lidar_loss_frozen";
  state.values[19].value = lidar_loss_frozen_ ? "true" : "false";
  state.values[20].key = "lidar_scan_points";
  state.values[20].value = std::to_string(lidar_scan_points_);
  state.values[21].key = "lidar_correspondence_sectors";
  state.values[21].value = std::to_string(lidar_correspondence_sectors_);
  state.values[22].key = "lidar_strong_support";
  state.values[22].value = lidar_strong_support_ ? "true" : "false";
  state.values[23].key = "lidar_recovery_mode";
  state.values[23].value = lidar_recovery_mode_ ? "true" : "false";
  state.values[24].key = "lidar_knn_fallback_queries";
  state.values[24].value = std::to_string(lidar_point_knn_fallback_queries_);
  state.values[25].key = "lidar_knn_fallback_matches";
  state.values[25].value = std::to_string(lidar_point_knn_fallback_matches_);
  diagnostics.status.push_back(state);
  diagnostic_msgs::DiagnosticStatus scheduler;
  scheduler.name = "hybrid_localization/measurement_scheduler";
  scheduler.hardware_id = "hybrid_localization";
  scheduler.level = scheduler_queue_drops == 0U && scheduler_stale_drops == 0U ?
      diagnostic_msgs::DiagnosticStatus::OK : diagnostic_msgs::DiagnosticStatus::WARN;
  scheduler.message = measurement_scheduler_enabled_ ? "timestamp ordered" : "disabled";
  scheduler.values.resize(12);
  scheduler.values[0].key = "queue";
  scheduler.values[0].value = std::to_string(measurement_queue_size);
  scheduler.values[1].key = "scheduled_lidar";
  scheduler.values[1].value = std::to_string(scheduled_lidar);
  scheduler.values[2].key = "scheduled_images";
  scheduler.values[2].value = std::to_string(scheduled_images);
  scheduler.values[3].key = "processed_images";
  scheduler.values[3].value = std::to_string(processed_image_events_);
  scheduler.values[4].key = "queue_drops";
  scheduler.values[4].value = std::to_string(scheduler_queue_drops);
  scheduler.values[5].key = "stale_drops";
  scheduler.values[5].value = std::to_string(scheduler_stale_drops);
  scheduler.values[6].key = "camera_decode_failures";
  scheduler.values[6].value = std::to_string(camera_decode_failures_);
  scheduler.values[7].key = "last_processed_stamp";
  scheduler.values[7].value = std::to_string(last_processed_measurement_stamp_);
  scheduler.values[8].key = "pending_imu_samples";
  scheduler.values[8].value = std::to_string(pendingImuSampleCount());
  scheduler.values[9].key = "pending_imu_queue_drops";
  scheduler.values[9].value = std::to_string(pendingImuQueueDrops());
  scheduler.values[10].key = "image_queue_drops";
  scheduler.values[10].value = std::to_string(scheduler_image_queue_drops);
  scheduler.values[11].key = "camera_interval_drops";
  scheduler.values[11].value = std::to_string(camera_interval_drops);
  diagnostics.status.push_back(scheduler);
  diagnostic_msgs::DiagnosticStatus visual;
  visual.name = "hybrid_localization/direct_visual_eskf";
  visual.hardware_id = "hybrid_localization";
  visual.level = !visual_frontend_enabled_ || visual_update_attempts_ == 0U ||
      visual_update_accepts_ > 0U ? diagnostic_msgs::DiagnosticStatus::OK
                                  : diagnostic_msgs::DiagnosticStatus::WARN;
  visual.message = visual_frontend_enabled_ ? visual_update_reason_ : "disabled";
  visual.values.resize(19);
  visual.values[0].key = "attempts";
  visual.values[0].value = std::to_string(visual_update_attempts_);
  visual.values[1].key = "accepts";
  visual.values[1].value = std::to_string(visual_update_accepts_);
  visual.values[2].key = "map_landmarks";
  visual.values[2].value = std::to_string(visual_stats.landmarks);
  visual.values[3].key = "active_landmarks";
  visual.values[3].value = std::to_string(visual_update_landmarks_);
  visual.values[4].key = "residuals";
  visual.values[4].value = std::to_string(visual_update_residuals_);
  visual.values[5].key = "rmse";
  visual.values[5].value = std::to_string(visual_update_rmse_);
  visual.values[6].key = "mean_ncc";
  visual.values[6].value = std::to_string(visual_update_ncc_);
  visual.values[7].key = "iterations";
  visual.values[7].value = std::to_string(visual_update_iterations_);
  visual.values[8].key = "dynamic_rejections";
  visual.values[8].value = std::to_string(visual_stats.dynamic_rejections);
  visual.values[9].key = "reason";
  visual.values[9].value = visual_update_reason_;
  visual.values[10].key = "sam3_labels_received";
  visual.values[10].value = std::to_string(sam3_camera_labels_received_);
  visual.values[11].key = "sam3_labels_applied";
  visual.values[11].value = std::to_string(sam3_camera_labels_applied_);
  visual.values[12].key = "sam3_flow_propagations";
  visual.values[12].value = std::to_string(sam3_flow_propagations_);
  visual.values[13].key = "sam3_flow_failures";
  visual.values[13].value = std::to_string(sam3_flow_failures_);
  visual.values[14].key = "sam3_dynamic_pixels";
  visual.values[14].value = std::to_string(sam3_dynamic_pixels_);
  visual.values[15].key = "sam3_source_stamp";
  visual.values[15].value = std::to_string(sam3_source_stamp_);
  visual.values[16].key = "sam3_propagated_stamp";
  visual.values[16].value = std::to_string(sam3_propagated_stamp_);
  visual.values[17].key = "pending_sam3_labels";
  visual.values[17].value = std::to_string(pending_sam3_labels);
  visual.values[18].key = "sam3_label_queue_drops";
  visual.values[18].value = std::to_string(sam3_label_queue_drops);
  diagnostics.status.push_back(visual);
  diagnostic_pub_.publish(diagnostics);
}

void HybridLocalizationNode::publishTimer(const ros::TimerEvent &)
{
  const ros::Time stamp = ros::Time(last_odom_stamp_ > 0.0 ? last_odom_stamp_ : ros::Time::now().toSec());
  if (have_odom_)
  {
    publishLocalBev(stamp);
    publishObjects(stamp);
  }
  publishStatus(stamp);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hybrid_localization");
  HybridLocalizationNode node;
  node.spin();
  return 0;
}
