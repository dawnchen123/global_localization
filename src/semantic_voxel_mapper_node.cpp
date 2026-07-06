#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/String.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Cholesky>

namespace
{

enum SemanticLabel : uint32_t
{
  LABEL_UNKNOWN = 0,
  LABEL_ROAD = 1,
  LABEL_SIDEWALK = 2,
  LABEL_BUILDING = 3,
  LABEL_VEGETATION = 4,
  LABEL_DYNAMIC = 5,
  LABEL_OTHER = 6,
  LABEL_COUNT = 7
};

struct VoxelKey
{
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const VoxelKey& rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey& k) const
  {
    const std::size_t hx = std::hash<int>()(k.x);
    const std::size_t hy = std::hash<int>()(k.y);
    const std::size_t hz = std::hash<int>()(k.z);
    return hx ^ (hy + 0x9e3779b97f4a7c15ULL + (hx << 6) + (hx >> 2)) ^
           (hz + 0x9e3779b97f4a7c15ULL + (hy << 6) + (hy >> 2));
  }
};

struct VoxelState
{
  Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;
  std::array<float, LABEL_COUNT> votes;
  ros::Time last_update;
  uint32_t observations = 0;

  VoxelState()
  {
    votes.fill(0.0f);
  }
};

struct OutputPoint
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint32_t label = 0;
  float confidence = 0.0f;
  float votes = 0.0f;
};

struct FrameSemanticPoint
{
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  uint32_t label = LABEL_UNKNOWN;
  uint32_t instance_id = 0;
  double confidence = 0.0;
  double weight = 0.0;
};

struct InstanceClusterStats
{
  std::vector<int> indices;
  std::array<int, LABEL_COUNT> label_counts;
  Eigen::Vector3d min_pt = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d max_pt = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

  InstanceClusterStats()
  {
    label_counts.fill(0);
  }

  void add(int index, const FrameSemanticPoint& p)
  {
    indices.push_back(index);
    if (p.label < LABEL_COUNT)
    {
      label_counts[p.label] += 1;
    }
    min_pt = min_pt.cwiseMin(p.p);
    max_pt = max_pt.cwiseMax(p.p);
  }
};

struct GroundFrameFilter
{
  struct Cell
  {
    double min_z = std::numeric_limits<double>::infinity();
    int count = 0;
  };

  std::unordered_map<VoxelKey, Cell, VoxelKeyHash> cells;
  double frame_ground_z = 0.0;
  bool valid = false;
};

struct SemanticAnchor
{
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  double z = 0.0;
  double weight = 0.0;
  uint32_t observations = 0;
  ros::Time last_update;
};

struct AnchorAccum
{
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  double z_sum = 0.0;
  double weight_sum = 0.0;
  uint32_t count = 0;
};

struct ZResidualSample
{
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();
  double residual = 0.0;
  double weight = 1.0;
};

struct SE2MatchSample
{
  Eigen::Vector2d source = Eigen::Vector2d::Zero();
  Eigen::Vector2d target = Eigen::Vector2d::Zero();
  double weight = 1.0;
};

struct ConstraintDebugPair
{
  Eigen::Vector3d source = Eigen::Vector3d::Zero();
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  uint32_t label = LABEL_UNKNOWN;
  double residual = 0.0;
  double weight = 1.0;
};

struct LocalSemanticFrame
{
  ros::Time stamp;
  std::vector<FrameSemanticPoint> points;
};

struct RebuiltSemanticPoint
{
  Eigen::Vector3d local = Eigen::Vector3d::Zero();
  uint32_t label = LABEL_UNKNOWN;
  float confidence = 0.0f;
  float weight = 1.0f;
};

struct RebuiltSemanticKeyframe
{
  ros::Time stamp;
  Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d corrected_pose = Eigen::Isometry3d::Identity();
  std::vector<RebuiltSemanticPoint> points;
};

const sensor_msgs::PointField* findField(const sensor_msgs::PointCloud2& msg, const std::string& name)
{
  for (const auto& field : msg.fields)
  {
    if (field.name == name)
    {
      return &field;
    }
  }
  return nullptr;
}

const sensor_msgs::PointField* findFirstField(const sensor_msgs::PointCloud2& msg,
                                              const std::vector<std::string>& names)
{
  for (const auto& name : names)
  {
    const sensor_msgs::PointField* field = findField(msg, name);
    if (field)
    {
      return field;
    }
  }
  return nullptr;
}

double readFieldAsDouble(const sensor_msgs::PointCloud2& msg,
                         const sensor_msgs::PointField& field,
                         const std::size_t point_index)
{
  const uint8_t* ptr = &msg.data[point_index * msg.point_step + field.offset];
  switch (field.datatype)
  {
    case sensor_msgs::PointField::INT8:
    {
      int8_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT8:
    {
      uint8_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::INT16:
    {
      int16_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT16:
    {
      uint16_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::INT32:
    {
      int32_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT32:
    {
      uint32_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::FLOAT32:
    {
      float v = 0.0f;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::FLOAT64:
    {
      double v = 0.0;
      std::memcpy(&v, ptr, sizeof(v));
      return v;
    }
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

float rgbFloat(uint8_t r, uint8_t g, uint8_t b)
{
  const uint32_t rgb = (static_cast<uint32_t>(r) << 16) |
                       (static_cast<uint32_t>(g) << 8) |
                       static_cast<uint32_t>(b);
  float out = 0.0f;
  std::memcpy(&out, &rgb, sizeof(out));
  return out;
}

float labelRgbFloat(uint32_t label)
{
  switch (label)
  {
    case LABEL_ROAD:
      return rgbFloat(70, 70, 70);
    case LABEL_SIDEWALK:
      return rgbFloat(160, 120, 80);
    case LABEL_BUILDING:
      return rgbFloat(220, 40, 40);
    case LABEL_VEGETATION:
      return rgbFloat(20, 170, 40);
    case LABEL_DYNAMIC:
      return rgbFloat(255, 180, 0);
    case LABEL_OTHER:
      return rgbFloat(120, 120, 220);
    default:
      return rgbFloat(180, 180, 180);
  }
}

cv::Vec3b labelColorBgr(uint32_t label)
{
  switch (label)
  {
    case LABEL_ROAD:
      return cv::Vec3b(70, 70, 70);
    case LABEL_SIDEWALK:
      return cv::Vec3b(80, 120, 160);
    case LABEL_BUILDING:
      return cv::Vec3b(40, 40, 220);
    case LABEL_VEGETATION:
      return cv::Vec3b(40, 170, 20);
    case LABEL_DYNAMIC:
      return cv::Vec3b(0, 180, 255);
    case LABEL_OTHER:
      return cv::Vec3b(220, 120, 120);
    default:
      return cv::Vec3b(180, 180, 180);
  }
}

Eigen::Isometry3d odomToIsometry(const nav_msgs::Odometry& odom)
{
  const auto& qmsg = odom.pose.pose.orientation;
  const auto& tmsg = odom.pose.pose.position;
  Eigen::Quaterniond q(qmsg.w, qmsg.x, qmsg.y, qmsg.z);
  if (q.norm() < 1e-6)
  {
    q = Eigen::Quaterniond::Identity();
  }
  q.normalize();

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = Eigen::Vector3d(tmsg.x, tmsg.y, tmsg.z);
  return T;
}

bool loadMatrixParam(const ros::NodeHandle& pnh, const std::string& name, Eigen::Isometry3d& T)
{
  std::vector<double> values;
  if (!pnh.getParam(name, values))
  {
    return false;
  }
  if (values.size() != 16)
  {
    ROS_WARN("Parameter ~%s must contain 16 numbers; got %zu. Keeping identity.", name.c_str(), values.size());
    return false;
  }

  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 4; ++r)
  {
    for (int c = 0; c < 4; ++c)
    {
      M(r, c) = values[static_cast<std::size_t>(r * 4 + c)];
    }
  }
  T.matrix() = M;
  return true;
}

}  // namespace

class SemanticVoxelMapper
{
 public:
  SemanticVoxelMapper() : nh_(), pnh_("~")
  {
    pnh_.param<std::string>("odom_topic", odom_topic_, "/aft_mapped_to_init");
    pnh_.param<std::string>("lidar_semantic_cloud_topic", lidar_cloud_topic_, "/rangenet/semantic_points");
    pnh_.param<std::string>("segformer_semantic_cloud_topic", image_cloud_topic_, "/segformer/projected_semantic_points");
    pnh_.param<bool>("subscribe_lidar_semantic_cloud", subscribe_lidar_cloud_, true);
    pnh_.param<bool>("subscribe_segformer_semantic_cloud", subscribe_image_cloud_, false);
    pnh_.param<bool>("lidar_cloud_in_map_frame", lidar_cloud_in_map_frame_, false);
    pnh_.param<bool>("image_cloud_in_map_frame", image_cloud_in_map_frame_, false);
    pnh_.param<std::string>("lidar_label_mode", lidar_label_mode_, "semantic_kitti");
    pnh_.param<std::string>("image_label_mode", image_label_mode_, "internal");
    pnh_.param<std::string>("map_frame", map_frame_, "camera_init");
    pnh_.param<std::string>("label_field", label_field_name_, "label");
    pnh_.param<double>("voxel_size", voxel_size_, 0.20);
    pnh_.param<double>("min_range_m", min_range_m_, 0.5);
    pnh_.param<double>("max_range_m", max_range_m_, 120.0);
    pnh_.param<double>("z_min_map", z_min_map_, -5.0);
    pnh_.param<double>("z_max_map", z_max_map_, 25.0);
    pnh_.param<double>("lidar_source_weight", lidar_source_weight_, 1.0);
    pnh_.param<double>("image_source_weight", image_source_weight_, 0.65);
    pnh_.param<double>("default_confidence", default_confidence_, 0.75);
    pnh_.param<double>("min_votes", min_votes_, 2.0);
    pnh_.param<double>("min_confidence", min_confidence_, 0.55);
    pnh_.param<double>("confidence_vote_scale", confidence_vote_scale_, 4.0);
    pnh_.param<bool>("fuse_dynamic_objects", fuse_dynamic_objects_, false);
    pnh_.param<bool>("enable_instance_dynamic_filter", enable_instance_dynamic_filter_, true);
    pnh_.param<double>("instance_cluster_resolution_m", instance_cluster_resolution_m_, 0.55);
    pnh_.param<int>("instance_min_points", instance_min_points_, 8);
    pnh_.param<double>("instance_dynamic_label_ratio", instance_dynamic_label_ratio_, 0.15);
    pnh_.param<double>("instance_dynamic_min_height_m", instance_dynamic_min_height_m_, 0.35);
    pnh_.param<double>("instance_dynamic_max_height_m", instance_dynamic_max_height_m_, 2.80);
    pnh_.param<double>("instance_dynamic_max_length_m", instance_dynamic_max_length_m_, 6.50);
    pnh_.param<double>("instance_dynamic_max_width_m", instance_dynamic_max_width_m_, 3.20);
    pnh_.param<bool>("publish_static_labeled_cloud", publish_static_labeled_cloud_, true);
    pnh_.param<int>("max_publish_points", max_publish_points_, 800000);
    pnh_.param<int>("max_voxels", max_voxels_, 1800000);
    pnh_.param<double>("voxel_ttl_sec", voxel_ttl_sec_, 0.0);
    pnh_.param<double>("publish_rate", publish_rate_, 2.0);

    pnh_.param<bool>("enable_semantic_z_correction", enable_semantic_z_correction_, true);
    pnh_.param<double>("z_anchor_resolution", z_anchor_resolution_, 1.0);
    pnh_.param<double>("object_anchor_resolution", object_anchor_resolution_, 2.5);
    pnh_.param<int>("min_z_constraint_matches", min_z_constraint_matches_, 80);
    pnh_.param<int>("min_ground_anchor_observations", min_ground_anchor_observations_, 3);
    pnh_.param<int>("min_object_anchor_observations", min_object_anchor_observations_, 3);
    pnh_.param<int>("max_z_constraint_samples", max_z_constraint_samples_, 2000);
    pnh_.param<double>("z_correction_alpha", z_correction_alpha_, 0.08);
    pnh_.param<double>("max_z_correction_step", max_z_correction_step_, 0.04);
    pnh_.param<double>("max_abs_z_correction", max_abs_z_correction_, 3.0);
    pnh_.param<double>("max_z_residual_abs", max_z_residual_abs_, 0.8);
    pnh_.param<double>("max_z_residual_mad", max_z_residual_mad_, 0.35);
    pnh_.param<int>("min_z_constraint_accept_streak", min_z_constraint_accept_streak_, 3);
    pnh_.param<double>("max_z_residual_jump", max_z_residual_jump_, 0.25);
    pnh_.param<double>("z_correction_deadband", z_correction_deadband_, 0.03);
    pnh_.param<bool>("enable_ground_geometry_filter", enable_ground_geometry_filter_, true);
    pnh_.param<double>("ground_filter_resolution_m", ground_filter_resolution_m_, 0.8);
    pnh_.param<double>("ground_filter_frame_percentile", ground_filter_frame_percentile_, 0.10);
    pnh_.param<int>("ground_filter_min_cell_points", ground_filter_min_cell_points_, 4);
    pnh_.param<int>("ground_filter_neighbor_radius_cells", ground_filter_neighbor_radius_cells_, 1);
    pnh_.param<double>("max_ground_height_above_cell_min", max_ground_height_above_cell_min_, 0.35);
    pnh_.param<double>("max_ground_height_above_frame_ground", max_ground_height_above_frame_ground_, 1.60);
    pnh_.param<int>("ground_anchor_search_radius_cells", ground_anchor_search_radius_cells_, 1);
    pnh_.param<double>("max_ground_anchor_match_xy_m", max_ground_anchor_match_xy_m_, 1.25);
    pnh_.param<double>("min_ground_anchor_age_sec", min_ground_anchor_age_sec_, 6.0);
    pnh_.param<double>("max_ground_anchor_update_z_residual", max_ground_anchor_update_z_residual_, 0.35);
    pnh_.param<double>("min_semantic_z_correction_trust", min_semantic_z_correction_trust_, 0.05);
    pnh_.param<int>("max_z_constraint_stale_frames", max_z_constraint_stale_frames_, 20);
    pnh_.param<bool>("enable_ground_robot_z_prior", enable_ground_robot_z_prior_, true);
    pnh_.param<double>("ground_robot_z_prior_alpha", ground_robot_z_prior_alpha_, 0.12);
    pnh_.param<double>("ground_robot_z_prior_max_step", ground_robot_z_prior_max_step_, 0.035);
    pnh_.param<double>("ground_robot_z_prior_deadband", ground_robot_z_prior_deadband_, 0.05);
    pnh_.param<double>("ground_robot_z_prior_max_grade", ground_robot_z_prior_max_grade_, 0.01);
    pnh_.param<double>("ground_robot_z_prior_vertical_margin", ground_robot_z_prior_vertical_margin_, 0.01);
    pnh_.param<double>("ground_robot_z_prior_max_abs_correction", ground_robot_z_prior_max_abs_correction_, 8.0);
    pnh_.param<double>("ground_robot_z_prior_max_drift", ground_robot_z_prior_max_drift_, 1.2);
    pnh_.param<bool>("use_keyframe_rebuilt_map", use_keyframe_rebuilt_map_, true);
    pnh_.param<bool>("enable_ground_robot_roll_pitch_prior", enable_ground_robot_roll_pitch_prior_, true);
    pnh_.param<double>("ground_robot_roll_pitch_scale", ground_robot_roll_pitch_scale_, 0.0);
    pnh_.param<double>("rebuilt_keyframe_min_distance", rebuilt_keyframe_min_distance_, 1.0);
    pnh_.param<double>("rebuilt_keyframe_min_interval_sec", rebuilt_keyframe_min_interval_sec_, 0.5);
    pnh_.param<int>("rebuilt_keyframe_max_frames", rebuilt_keyframe_max_frames_, 700);
    pnh_.param<int>("rebuilt_keyframe_max_points_per_frame", rebuilt_keyframe_max_points_per_frame_, 2500);
    pnh_.param<bool>("rebuilt_keyframe_lidar_only", rebuilt_keyframe_lidar_only_, true);
    pnh_.param<bool>("publish_constraint_debug", publish_constraint_debug_, true);
    pnh_.param<int>("constraint_debug_max_pairs", constraint_debug_max_pairs_, 300);
    pnh_.param<std::string>("z_constraint_debug_topic", z_constraint_debug_topic_, "/semantic_slam/z_constraint_debug");
    pnh_.param<std::string>("xy_constraint_debug_topic", xy_constraint_debug_topic_, "/semantic_slam/xy_constraint_debug");
    pnh_.param<std::string>("z_constraint_points_topic", z_constraint_points_topic_, "/semantic_slam/z_constraint_points");
    pnh_.param<std::string>("xy_constraint_points_topic", xy_constraint_points_topic_, "/semantic_slam/xy_constraint_points");
    pnh_.param<bool>("enable_semantic_z_slope_correction", enable_semantic_z_slope_correction_, true);
    pnh_.param<int>("min_z_slope_constraint_matches", min_z_slope_constraint_matches_, 120);
    pnh_.param<double>("min_z_slope_spread_m", min_z_slope_spread_m_, 8.0);
    pnh_.param<double>("z_slope_correction_alpha", z_slope_correction_alpha_, 0.08);
    pnh_.param<double>("max_z_slope_correction_step", max_z_slope_correction_step_, 0.002);
    pnh_.param<double>("max_abs_z_slope", max_abs_z_slope_, 0.05);
    pnh_.param<double>("anchor_update_alpha", anchor_update_alpha_, 0.06);
    pnh_.param<double>("object_anchor_weight", object_anchor_weight_, 0.35);
    pnh_.param<bool>("use_object_z_anchors", use_object_z_anchors_, false);
    pnh_.param<bool>("enable_semantic_xy_correction", enable_semantic_xy_correction_, true);
    pnh_.param<bool>("use_vegetation_xy_anchors", use_vegetation_xy_anchors_, false);
    pnh_.param<int>("min_xy_constraint_matches", min_xy_constraint_matches_, 12);
    pnh_.param<int>("max_xy_constraint_samples", max_xy_constraint_samples_, 500);
    pnh_.param<int>("xy_anchor_search_radius_cells", xy_anchor_search_radius_cells_, 1);
    pnh_.param<double>("xy_correction_alpha", xy_correction_alpha_, 0.05);
    pnh_.param<double>("max_xy_correction_step", max_xy_correction_step_, 0.03);
    pnh_.param<double>("max_abs_xy_correction", max_abs_xy_correction_, 5.0);
    pnh_.param<double>("max_xy_match_residual", max_xy_match_residual_, 3.0);
    pnh_.param<bool>("enable_semantic_yaw_correction", enable_semantic_yaw_correction_, true);
    pnh_.param<int>("min_se2_constraint_matches", min_se2_constraint_matches_, 30);
    pnh_.param<double>("min_se2_spread_m", min_se2_spread_m_, 8.0);
    pnh_.param<double>("max_se2_residual_rms", max_se2_residual_rms_, 1.5);
    pnh_.param<double>("yaw_correction_alpha", yaw_correction_alpha_, 0.05);
    pnh_.param<double>("max_yaw_correction_step_deg", max_yaw_correction_step_deg_, 0.15);
    pnh_.param<double>("max_abs_yaw_correction_deg", max_abs_yaw_correction_deg_, 8.0);
    pnh_.param<double>("pose_constraint_interval_sec", pose_constraint_interval_sec_, 1.0);
    pnh_.param<int>("max_pose_constraint_stale_frames", max_pose_constraint_stale_frames_, 30);
    pnh_.param<double>("min_semantic_correction_trust", min_semantic_correction_trust_, 0.35);
    pnh_.param<bool>("enable_local_semantic_map_constraint", enable_local_semantic_map_constraint_, true);
    pnh_.param<bool>("local_semantic_lidar_only", local_semantic_lidar_only_, true);
    pnh_.param<double>("local_semantic_window_sec", local_semantic_window_sec_, 3.0);
    pnh_.param<int>("local_semantic_min_frames", local_semantic_min_frames_, 20);
    pnh_.param<int>("local_semantic_max_frames", local_semantic_max_frames_, 80);
    pnh_.param<int>("local_semantic_max_points_per_frame", local_semantic_max_points_per_frame_, 1200);
    pnh_.param<std::string>("corrected_odom_topic", corrected_odom_topic_, "/semantic_corrected_odom");
    pnh_.param<std::string>("corrected_path_topic", corrected_path_topic_, "/semantic_corrected_path");
    pnh_.param<int>("corrected_path_max_poses", corrected_path_max_poses_, 12000);
    pnh_.param<double>("corrected_path_publish_rate", corrected_path_publish_rate_, 2.0);
    pnh_.param<std::string>("z_constraint_topic", z_constraint_topic_, "/semantic_slam/z_constraint");

    pnh_.param<std::string>("semantic_cloud_topic", semantic_cloud_topic_, "/semantic_cloud_map");
    pnh_.param<std::string>("semantic_cloud_stats_topic", stats_topic_, "/semantic_cloud_map/stats");
    pnh_.param<std::string>("static_labeled_cloud_topic", static_labeled_cloud_topic_, "/semantic_cloud/static_labeled_points");
    pnh_.param<std::string>("bev_label_topic", bev_label_topic_, "/semantic_bev/label");
    pnh_.param<std::string>("bev_color_topic", bev_color_topic_, "/semantic_bev/color");
    pnh_.param<std::string>("bev_confidence_topic", bev_confidence_topic_, "/semantic_bev/confidence");
    pnh_.param<std::string>("bev_traversable_topic", bev_traversable_topic_, "/semantic_bev/traversable");
    pnh_.param<bool>("publish_bev", publish_bev_, true);
    pnh_.param<double>("bev_resolution", bev_resolution_, 0.20);
    pnh_.param<double>("bev_size_m", bev_size_m_, 100.0);
    pnh_.param<bool>("bev_center_on_latest_pose", bev_center_on_latest_pose_, true);

    loadMatrixParam(pnh_, "T_body_lidar", T_body_lidar_);

    latest_T_map_body_.setIdentity();
    latest_odom_time_ = ros::Time(0);

    odom_sub_ = nh_.subscribe(odom_topic_, 100, &SemanticVoxelMapper::odomCb, this);
    if (subscribe_lidar_cloud_)
    {
      lidar_cloud_sub_ = nh_.subscribe(lidar_cloud_topic_, 5, &SemanticVoxelMapper::lidarCloudCb, this);
    }
    if (subscribe_image_cloud_)
    {
      image_cloud_sub_ = nh_.subscribe(image_cloud_topic_, 5, &SemanticVoxelMapper::imageCloudCb, this);
    }

    semantic_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(semantic_cloud_topic_, 2);
    static_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(static_labeled_cloud_topic_, 2);
    stats_pub_ = nh_.advertise<std_msgs::String>(stats_topic_, 2);
    corrected_odom_pub_ = nh_.advertise<nav_msgs::Odometry>(corrected_odom_topic_, 20);
    corrected_path_pub_ = nh_.advertise<nav_msgs::Path>(corrected_path_topic_, 1, true);
    z_constraint_pub_ = nh_.advertise<std_msgs::String>(z_constraint_topic_, 10);
    z_constraint_debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(z_constraint_debug_topic_, 1);
    xy_constraint_debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(xy_constraint_debug_topic_, 1);
    z_constraint_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(z_constraint_points_topic_, 1);
    xy_constraint_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(xy_constraint_points_topic_, 1);
    bev_label_pub_ = nh_.advertise<sensor_msgs::Image>(bev_label_topic_, 1);
    bev_color_pub_ = nh_.advertise<sensor_msgs::Image>(bev_color_topic_, 1);
    bev_conf_pub_ = nh_.advertise<sensor_msgs::Image>(bev_confidence_topic_, 1);
    bev_traversable_pub_ = nh_.advertise<sensor_msgs::Image>(bev_traversable_topic_, 1);

    publish_timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.1, publish_rate_)),
                                    &SemanticVoxelMapper::publishTimerCb, this);
    maintenance_timer_ = nh_.createTimer(ros::Duration(2.0), &SemanticVoxelMapper::maintenanceTimerCb, this);
    diagnostic_timer_ = nh_.createWallTimer(ros::WallDuration(3.0), &SemanticVoxelMapper::diagnosticTimerCb, this);

    ROS_INFO("semantic_voxel_mapper started lidar_topic=%s image_topic=%s odom=%s voxel=%.2f map_frame=%s",
             lidar_cloud_topic_.c_str(), image_cloud_topic_.c_str(), odom_topic_.c_str(),
             voxel_size_, map_frame_.c_str());
    ROS_INFO("label modes: lidar=%s image=%s cloud_in_map lidar=%s image=%s",
             lidar_label_mode_.c_str(), image_label_mode_.c_str(),
             lidar_cloud_in_map_frame_ ? "true" : "false",
             image_cloud_in_map_frame_ ? "true" : "false");
  }

 private:
  void odomCb(const nav_msgs::OdometryConstPtr& msg)
  {
    latest_T_map_body_ = odomToIsometry(*msg);
    latest_odom_time_ = msg->header.stamp;
    if (!has_z_correction_origin_)
    {
      z_correction_origin_xy_ = latest_T_map_body_.translation().head<2>();
      has_z_correction_origin_ = true;
    }
    latest_odom_msg_ = *msg;
    has_latest_odom_msg_ = true;
    raw_odom_history_.push_back(*msg);
    const int max_history = std::max(1, corrected_path_max_poses_);
    while (static_cast<int>(raw_odom_history_.size()) > max_history)
    {
      raw_odom_history_.pop_front();
    }
    updateGroundRobotZPrior(msg->header.stamp);
    publishCorrectedOdom(msg->header.stamp);
  }

  uint32_t mapLabel(uint32_t raw_label, const std::string& mode) const
  {
    if (mode == "internal")
    {
      return raw_label < LABEL_COUNT ? raw_label : LABEL_UNKNOWN;
    }

    // SemanticKITTI/RangeNet++ ids. Moving classes are folded into DYNAMIC.
    switch (raw_label)
    {
      case 40:  // road
      case 44:  // parking
      case 49:  // other-ground
        return LABEL_ROAD;
      case 48:  // sidewalk
      case 72:  // terrain
        return LABEL_SIDEWALK;
      case 50:  // building
      case 51:  // fence
      case 52:  // other-structure
        return LABEL_BUILDING;
      case 70:  // vegetation
      case 71:  // trunk
      case 80:  // pole
      case 81:  // traffic-sign
        return LABEL_VEGETATION;
      case 10:   // car
      case 11:   // bicycle
      case 13:   // bus
      case 15:   // motorcycle
      case 16:   // on-rails
      case 18:   // truck
      case 20:   // other-vehicle
      case 30:   // person
      case 31:   // bicyclist
      case 32:   // motorcyclist
      case 252:  // moving-car
      case 253:  // moving-bicyclist
      case 254:  // moving-person
      case 255:  // moving-motorcyclist
      case 256:  // moving-on-rails
      case 257:  // moving-bus
      case 258:  // moving-truck
      case 259:  // moving-other-vehicle
        return LABEL_DYNAMIC;
      case 99:  // other-object
        return LABEL_OTHER;
      default:
        return LABEL_UNKNOWN;
    }
  }

  double labelWeight(uint32_t label) const
  {
    switch (label)
    {
      case LABEL_ROAD:
        return 1.25;
      case LABEL_SIDEWALK:
        return 1.10;
      case LABEL_BUILDING:
        return 1.35;
      case LABEL_VEGETATION:
        return 0.85;
      case LABEL_DYNAMIC:
        return 0.50;
      case LABEL_OTHER:
        return 0.75;
      default:
        return 0.0;
    }
  }

  VoxelKey voxelKey(const Eigen::Vector3d& p) const
  {
    const double inv = 1.0 / std::max(0.02, voxel_size_);
    return VoxelKey{
      static_cast<int>(std::floor(p.x() * inv)),
      static_cast<int>(std::floor(p.y() * inv)),
      static_cast<int>(std::floor(p.z() * inv))
    };
  }

  VoxelKey anchorKey(const Eigen::Vector3d& p, double resolution, int z_id = 0) const
  {
    const double inv = 1.0 / std::max(0.05, resolution);
    return VoxelKey{
      static_cast<int>(std::floor(p.x() * inv)),
      static_cast<int>(std::floor(p.y() * inv)),
      z_id
    };
  }

  static double normalizeAngle(double a)
  {
    while (a > M_PI)
    {
      a -= 2.0 * M_PI;
    }
    while (a < -M_PI)
    {
      a += 2.0 * M_PI;
    }
    return a;
  }

  Eigen::Matrix2d planarCorrectionRotation() const
  {
    const double trust = semanticCorrectionTrust();
    const double c = std::cos(semantic_yaw_correction_ * trust);
    const double s = std::sin(semantic_yaw_correction_ * trust);
    Eigen::Matrix2d R;
    R << c, -s, s, c;
    return R;
  }

  double semanticCorrectionTrust() const
  {
    const int max_stale = std::max(1, max_pose_constraint_stale_frames_);
    const double stale = static_cast<double>(std::max(0, pose_constraint_stale_frames_));
    const double t = 1.0 - std::min(1.0, stale / static_cast<double>(max_stale));
    const double min_trust = std::max(0.0, std::min(1.0, min_semantic_correction_trust_));
    return min_trust + (1.0 - min_trust) * t;
  }

  double semanticZCorrectionTrust() const
  {
    const int max_stale = std::max(1, max_z_constraint_stale_frames_);
    const double stale = static_cast<double>(std::max(0, z_constraint_stale_frames_));
    const double t = 1.0 - std::min(1.0, stale / static_cast<double>(max_stale));
    const double min_trust = std::max(0.0, std::min(1.0, min_semantic_z_correction_trust_));
    return min_trust + (1.0 - min_trust) * t;
  }

  Eigen::Vector2d applyPlanarCorrection(const Eigen::Vector2d& p) const
  {
    const double trust = semanticCorrectionTrust();
    if (!has_z_correction_origin_)
    {
      return p + semantic_xy_correction_ * trust;
    }
    return planarCorrectionRotation() * (p - z_correction_origin_xy_) +
           z_correction_origin_xy_ + semantic_xy_correction_ * trust;
  }

  double zCorrectionAt(const Eigen::Vector3d& p) const
  {
    double dz = semantic_z_correction_;
    if (enable_semantic_z_slope_correction_ && has_z_correction_origin_)
    {
      dz += semantic_z_slope_.dot(p.head<2>() - z_correction_origin_xy_);
    }
    return ground_robot_z_correction_ + dz * semanticZCorrectionTrust();
  }

  void updateGroundRobotZPrior(const ros::Time& /*stamp*/)
  {
    last_ground_robot_z_step_ = 0.0;
    last_ground_robot_z_residual_ = 0.0;
    if (!enable_ground_robot_z_prior_ || !has_latest_odom_msg_)
    {
      return;
    }

    const Eigen::Vector3d raw_p(latest_odom_msg_.pose.pose.position.x,
                                latest_odom_msg_.pose.pose.position.y,
                                latest_odom_msg_.pose.pose.position.z);
    if (!std::isfinite(raw_p.x()) || !std::isfinite(raw_p.y()) || !std::isfinite(raw_p.z()))
    {
      return;
    }
    const Eigen::Vector2d xy = raw_p.head<2>();
    if (!has_ground_robot_z_prior_)
    {
      has_ground_robot_z_prior_ = true;
      ground_robot_reference_z_ = raw_p.z();
      ground_robot_prior_z_ = raw_p.z();
      ground_robot_last_raw_z_ = raw_p.z();
      ground_robot_last_xy_ = xy;
      return;
    }

    const double dxy = (xy - ground_robot_last_xy_).norm();
    const double raw_dz = raw_p.z() - ground_robot_last_raw_z_;
    const double max_grade = std::max(0.0, ground_robot_z_prior_max_grade_);
    const double margin = std::max(0.0, ground_robot_z_prior_vertical_margin_);
    const double allowed_dz = max_grade * dxy + margin;
    const double limited_dz = std::max(-allowed_dz, std::min(allowed_dz, raw_dz));
    ground_robot_prior_z_ += limited_dz;

    const double max_drift = std::max(0.0, ground_robot_z_prior_max_drift_);
    if (max_drift > 0.0)
    {
      ground_robot_prior_z_ = std::max(ground_robot_reference_z_ - max_drift,
                                       std::min(ground_robot_reference_z_ + max_drift,
                                                ground_robot_prior_z_));
    }

    const double corrected_z = raw_p.z() + zCorrectionAt(raw_p);
    const double residual = corrected_z - ground_robot_prior_z_;
    last_ground_robot_z_residual_ = residual;
    const double deadband = std::max(0.0, ground_robot_z_prior_deadband_);
    const double abs_residual = std::fabs(residual);
    if (abs_residual > deadband)
    {
      const double signed_residual = residual > 0.0 ? (abs_residual - deadband) : -(abs_residual - deadband);
      const double alpha = std::max(0.0, std::min(1.0, ground_robot_z_prior_alpha_));
      const double max_step = std::max(0.001, ground_robot_z_prior_max_step_);
      const double step = std::max(-max_step, std::min(max_step, -alpha * signed_residual));
      ground_robot_z_correction_ += step;
      const double max_abs = std::max(0.0, ground_robot_z_prior_max_abs_correction_);
      if (max_abs > 0.0)
      {
        ground_robot_z_correction_ = std::max(-max_abs, std::min(max_abs, ground_robot_z_correction_));
      }
      last_ground_robot_z_step_ = step;
      pose_z_variance_ = std::max(min_pose_z_variance_, std::min(pose_z_variance_, abs_residual * abs_residual));
    }

    ground_robot_last_raw_z_ = raw_p.z();
    ground_robot_last_xy_ = xy;
  }

  Eigen::Isometry3d groundRobotCorrectedPose(const Eigen::Isometry3d& raw_pose) const
  {
    Eigen::Isometry3d corrected = Eigen::Isometry3d::Identity();
    const Eigen::Vector3d raw_t = raw_pose.translation();
    corrected.translation() = raw_t;
    corrected.translation().z() += zCorrectionAt(raw_t);

    const Eigen::Matrix3d R = raw_pose.rotation();
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_raw(R);
    if (q_raw.norm() < 1e-6)
    {
      q_raw = Eigen::Quaterniond::Identity();
    }
    q_raw.normalize();
    q_yaw.normalize();

    if (enable_ground_robot_roll_pitch_prior_)
    {
      const double scale = std::max(0.0, std::min(1.0, ground_robot_roll_pitch_scale_));
      corrected.linear() = q_yaw.slerp(scale, q_raw).normalized().toRotationMatrix();
    }
    else
    {
      corrected.linear() = q_raw.toRotationMatrix();
    }
    return corrected;
  }

  std::vector<OutputPoint> rebuildFramePoints(const std::vector<FrameSemanticPoint>& frame_points,
                                              const Eigen::Isometry3d& raw_pose,
                                              const Eigen::Isometry3d& corrected_pose,
                                              int max_points) const
  {
    std::vector<OutputPoint> out;
    if (frame_points.empty())
    {
      return out;
    }
    const std::size_t limit = max_points > 0 ? static_cast<std::size_t>(max_points) : frame_points.size();
    const std::size_t stride = std::max<std::size_t>(1, frame_points.size() / std::max<std::size_t>(1, limit));
    out.reserve(std::min<std::size_t>(frame_points.size(), limit));
    const Eigen::Isometry3d T_body_raw_map = raw_pose.inverse();
    for (std::size_t i = 0; i < frame_points.size() && out.size() < limit; i += stride)
    {
      const FrameSemanticPoint& fp = frame_points[i];
      if (fp.label == LABEL_DYNAMIC && !fuse_dynamic_objects_)
      {
        continue;
      }
      const Eigen::Vector3d local = T_body_raw_map * fp.p;
      const Eigen::Vector3d p = corrected_pose * local;
      if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()) ||
          p.z() < z_min_map_ || p.z() > z_max_map_)
      {
        continue;
      }
      OutputPoint pt;
      pt.x = static_cast<float>(p.x());
      pt.y = static_cast<float>(p.y());
      pt.z = static_cast<float>(p.z());
      pt.label = fp.label;
      pt.confidence = static_cast<float>(std::max(0.0, std::min(1.0, fp.confidence)));
      pt.votes = static_cast<float>(std::max(0.05, fp.weight));
      out.push_back(pt);
    }
    return out;
  }

  void maybeAddRebuiltKeyframe(const std::vector<FrameSemanticPoint>& frame_points,
                               const ros::Time& stamp,
                               bool lidar_source)
  {
    if (!use_keyframe_rebuilt_map_ || frame_points.empty() || !has_latest_odom_msg_)
    {
      return;
    }
    if (rebuilt_keyframe_lidar_only_ && !lidar_source)
    {
      return;
    }

    const Eigen::Vector3d raw_t = latest_T_map_body_.translation();
    if (has_last_rebuilt_keyframe_)
    {
      const double dt = stamp.isZero() || last_rebuilt_keyframe_time_.isZero() ?
                            rebuilt_keyframe_min_interval_sec_ :
                            (stamp - last_rebuilt_keyframe_time_).toSec();
      const double dist = (raw_t - last_rebuilt_keyframe_position_).norm();
      if (dt < std::max(0.0, rebuilt_keyframe_min_interval_sec_) ||
          dist < std::max(0.0, rebuilt_keyframe_min_distance_))
      {
        return;
      }
    }

    RebuiltSemanticKeyframe kf;
    kf.stamp = stamp;
    kf.raw_pose = latest_T_map_body_;
    kf.corrected_pose = groundRobotCorrectedPose(latest_T_map_body_);
    const Eigen::Isometry3d T_body_raw_map = latest_T_map_body_.inverse();
    const std::size_t limit =
        static_cast<std::size_t>(std::max(1, rebuilt_keyframe_max_points_per_frame_));
    const std::size_t stride = std::max<std::size_t>(1, frame_points.size() / limit);
    kf.points.reserve(std::min<std::size_t>(frame_points.size(), limit));
    for (std::size_t i = 0; i < frame_points.size() && kf.points.size() < limit; i += stride)
    {
      const FrameSemanticPoint& fp = frame_points[i];
      if (fp.label == LABEL_DYNAMIC && !fuse_dynamic_objects_)
      {
        continue;
      }
      RebuiltSemanticPoint pt;
      pt.local = T_body_raw_map * fp.p;
      pt.label = fp.label;
      pt.confidence = static_cast<float>(std::max(0.0, std::min(1.0, fp.confidence)));
      pt.weight = static_cast<float>(std::max(0.05, fp.weight));
      kf.points.push_back(pt);
    }
    if (kf.points.empty())
    {
      return;
    }

    rebuilt_keyframes_.push_back(std::move(kf));
    const int max_frames = std::max(1, rebuilt_keyframe_max_frames_);
    while (static_cast<int>(rebuilt_keyframes_.size()) > max_frames)
    {
      rebuilt_keyframes_.pop_front();
    }
    has_last_rebuilt_keyframe_ = true;
    last_rebuilt_keyframe_time_ = stamp;
    last_rebuilt_keyframe_position_ = raw_t;
  }

  Eigen::Vector3d correctedPointForZAnchors(const Eigen::Vector3d& p) const
  {
    Eigen::Vector3d out = p;
    out.z() += zCorrectionAt(p);
    return out;
  }

  Eigen::Vector3d applySemanticCorrection(const Eigen::Vector3d& p) const
  {
    Eigen::Vector3d out = p;
    const Eigen::Vector2d xy = applyPlanarCorrection(p.head<2>());
    out.x() = xy.x();
    out.y() = xy.y();
    out.z() += zCorrectionAt(p);
    return out;
  }

  void publishCorrectedOdom(const ros::Time& stamp)
  {
    if (!has_latest_odom_msg_)
    {
      return;
    }
    nav_msgs::Odometry odom = latest_odom_msg_;
    odom.header.stamp = stamp.isZero() ? latest_odom_msg_.header.stamp : stamp;
    const Eigen::Vector3d raw_p(latest_odom_msg_.pose.pose.position.x,
                                latest_odom_msg_.pose.pose.position.y,
                                latest_odom_msg_.pose.pose.position.z);
    const Eigen::Vector2d corrected_xy = applyPlanarCorrection(raw_p.head<2>());
    odom.pose.pose.position.x = corrected_xy.x();
    odom.pose.pose.position.y = corrected_xy.y();
    odom.pose.pose.position.z += zCorrectionAt(raw_p);
    const Eigen::Quaterniond q_ground(groundRobotCorrectedPose(latest_T_map_body_).rotation());
    const Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(semantic_yaw_correction_ * semanticCorrectionTrust(),
                                                     Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond q_out = (q_yaw * q_ground).normalized();
    odom.pose.pose.orientation.w = q_out.w();
    odom.pose.pose.orientation.x = q_out.x();
    odom.pose.pose.orientation.y = q_out.y();
    odom.pose.pose.orientation.z = q_out.z();
    odom.pose.covariance[0] = pose_xy_variance_;
    odom.pose.covariance[7] = pose_xy_variance_;
    odom.pose.covariance[14] = pose_z_variance_;
    odom.pose.covariance[35] = pose_yaw_variance_;
    corrected_odom_pub_.publish(odom);
    publishCorrectedPath(stamp);
  }

  geometry_msgs::PoseStamped correctedPoseStamped(const nav_msgs::Odometry& raw_odom,
                                                  const ros::Time& fallback_stamp) const
  {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = raw_odom.header.stamp.isZero() ? fallback_stamp : raw_odom.header.stamp;
    pose.header.frame_id = map_frame_;

    const Eigen::Vector3d raw_p(raw_odom.pose.pose.position.x,
                                raw_odom.pose.pose.position.y,
                                raw_odom.pose.pose.position.z);
    const Eigen::Vector2d corrected_xy = applyPlanarCorrection(raw_p.head<2>());
    pose.pose.position.x = corrected_xy.x();
    pose.pose.position.y = corrected_xy.y();
    pose.pose.position.z = raw_p.z() + zCorrectionAt(raw_p);

    const Eigen::Isometry3d raw_T = odomToIsometry(raw_odom);
    const Eigen::Quaterniond q_ground(groundRobotCorrectedPose(raw_T).rotation());
    const Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(semantic_yaw_correction_ * semanticCorrectionTrust(),
                                                     Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond q_out = (q_yaw * q_ground).normalized();
    pose.pose.orientation.w = q_out.w();
    pose.pose.orientation.x = q_out.x();
    pose.pose.orientation.y = q_out.y();
    pose.pose.orientation.z = q_out.z();
    return pose;
  }

  void publishCorrectedPath(const ros::Time& stamp)
  {
    if (!corrected_path_pub_ || raw_odom_history_.empty())
    {
      return;
    }
    const double rate = std::max(0.0, corrected_path_publish_rate_);
    const ros::WallTime now = ros::WallTime::now();
    if (rate > 0.0 && !last_corrected_path_pub_wall_.isZero() &&
        (now - last_corrected_path_pub_wall_).toSec() < 1.0 / rate)
    {
      return;
    }
    last_corrected_path_pub_wall_ = now;

    nav_msgs::Path path;
    path.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
    path.header.frame_id = map_frame_;
    path.poses.reserve(raw_odom_history_.size());
    for (const auto& raw_odom : raw_odom_history_)
    {
      path.poses.push_back(correctedPoseStamped(raw_odom, path.header.stamp));
    }
    corrected_path_pub_.publish(path);
  }

  static geometry_msgs::Point markerPoint(const Eigen::Vector3d& p)
  {
    geometry_msgs::Point out;
    out.x = p.x();
    out.y = p.y();
    out.z = p.z();
    return out;
  }

  static std_msgs::ColorRGBA markerColor(double r, double g, double b, double a)
  {
    std_msgs::ColorRGBA c;
    c.r = static_cast<float>(r);
    c.g = static_cast<float>(g);
    c.b = static_cast<float>(b);
    c.a = static_cast<float>(a);
    return c;
  }

  void publishConstraintDebug(const ros::Time& stamp,
                              const std::string& ns,
                              const std::vector<ConstraintDebugPair>& pairs,
                              bool accepted,
                              const std::string& reason,
                              ros::Publisher& marker_pub,
                              ros::Publisher& points_pub) const
  {
    if (!publish_constraint_debug_)
    {
      return;
    }
    if (!marker_pub)
    {
      return;
    }

    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = map_frame_;
    clear.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear);

    visualization_msgs::Marker lines;
    lines.header.stamp = stamp;
    lines.header.frame_id = map_frame_;
    lines.ns = ns + "_links";
    lines.id = 1;
    lines.type = visualization_msgs::Marker::LINE_LIST;
    lines.action = visualization_msgs::Marker::ADD;
    lines.pose.orientation.w = 1.0;
    lines.scale.x = accepted ? 0.05 : 0.035;
    lines.color = accepted ? markerColor(0.0, 1.0, 0.15, 0.95) : markerColor(1.0, 0.1, 0.0, 0.85);

    visualization_msgs::Marker src_points;
    src_points.header = lines.header;
    src_points.ns = ns + "_source";
    src_points.id = 2;
    src_points.type = visualization_msgs::Marker::SPHERE_LIST;
    src_points.action = visualization_msgs::Marker::ADD;
    src_points.pose.orientation.w = 1.0;
    src_points.scale.x = src_points.scale.y = src_points.scale.z = 0.25;
    src_points.color = markerColor(0.1, 0.45, 1.0, 0.95);

    visualization_msgs::Marker target_points;
    target_points.header = lines.header;
    target_points.ns = ns + "_target";
    target_points.id = 3;
    target_points.type = visualization_msgs::Marker::SPHERE_LIST;
    target_points.action = visualization_msgs::Marker::ADD;
    target_points.pose.orientation.w = 1.0;
    target_points.scale.x = target_points.scale.y = target_points.scale.z = 0.32;
    target_points.color = markerColor(1.0, 0.85, 0.0, 0.95);

    std::vector<OutputPoint> debug_points;
    const std::size_t limit = static_cast<std::size_t>(std::max(1, constraint_debug_max_pairs_));
    const std::size_t stride = pairs.size() > limit ? std::max<std::size_t>(1, pairs.size() / limit) : 1;
    double abs_res_sum = 0.0;
    double abs_res_max = 0.0;
    std::size_t used = 0;
    for (std::size_t i = 0; i < pairs.size() && used < limit; i += stride)
    {
      const ConstraintDebugPair& pair = pairs[i];
      if (!std::isfinite(pair.source.x()) || !std::isfinite(pair.target.x()))
      {
        continue;
      }
      lines.points.push_back(markerPoint(pair.source));
      lines.points.push_back(markerPoint(pair.target));
      src_points.points.push_back(markerPoint(pair.source));
      target_points.points.push_back(markerPoint(pair.target));

      OutputPoint src;
      src.x = static_cast<float>(pair.source.x());
      src.y = static_cast<float>(pair.source.y());
      src.z = static_cast<float>(pair.source.z());
      src.label = pair.label;
      src.confidence = 1.0f;
      src.votes = static_cast<float>(pair.weight);
      debug_points.push_back(src);

      OutputPoint tgt;
      tgt.x = static_cast<float>(pair.target.x());
      tgt.y = static_cast<float>(pair.target.y());
      tgt.z = static_cast<float>(pair.target.z());
      tgt.label = LABEL_OTHER;
      tgt.confidence = 1.0f;
      tgt.votes = static_cast<float>(pair.weight);
      debug_points.push_back(tgt);

      const double ar = std::fabs(pair.residual);
      abs_res_sum += ar;
      abs_res_max = std::max(abs_res_max, ar);
      ++used;
    }

    visualization_msgs::Marker text;
    text.header = lines.header;
    text.ns = ns + "_status";
    text.id = 4;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::Marker::ADD;
    text.pose.orientation.w = 1.0;
    Eigen::Vector3d label_pos = Eigen::Vector3d::Zero();
    if (pairs.empty())
    {
      label_pos = latest_T_map_body_.translation();
      label_pos.z() += 2.0;
    }
    else
    {
      label_pos = pairs.front().source;
      label_pos.z() += 1.5;
    }
    text.pose.position = markerPoint(label_pos);
    text.scale.z = 0.8;
    text.color = markerColor(1.0, 1.0, 1.0, 0.95);
    std::ostringstream label;
    label << ns << " " << (accepted ? "ACCEPT" : "REJECT")
          << " reason=" << reason
          << " pairs=" << pairs.size()
          << " shown=" << used;
    if (used > 0)
    {
      label << " mean_abs_res=" << (abs_res_sum / static_cast<double>(used))
            << " max_abs_res=" << abs_res_max;
    }
    text.text = label.str();

    markers.markers.push_back(lines);
    markers.markers.push_back(src_points);
    markers.markers.push_back(target_points);
    markers.markers.push_back(text);
    marker_pub.publish(markers);
    publishPointCloud(debug_points, ns + "_debug_points", points_pub, stamp);
  }

  void lidarCloudCb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    ++lidar_msg_count_;
    last_lidar_input_wall_ = ros::WallTime::now();
    processCloud(msg, lidar_source_weight_, lidar_cloud_in_map_frame_, lidar_label_mode_, true);
  }

  void imageCloudCb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    ++image_msg_count_;
    last_image_input_wall_ = ros::WallTime::now();
    processCloud(msg, image_source_weight_, image_cloud_in_map_frame_, image_label_mode_, false);
  }

  static bool isGroundLabel(uint32_t label)
  {
    return label == LABEL_ROAD || label == LABEL_SIDEWALK;
  }

  static bool isObjectAnchorLabel(uint32_t label)
  {
    return label == LABEL_BUILDING || label == LABEL_VEGETATION || label == LABEL_OTHER;
  }

  bool isXYObjectAnchorLabel(uint32_t label) const
  {
    if (label == LABEL_BUILDING || label == LABEL_OTHER)
    {
      return true;
    }
    return use_vegetation_xy_anchors_ && label == LABEL_VEGETATION;
  }

  static double robustMedian(std::vector<double>& values)
  {
    if (values.empty())
    {
      return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    double med = values[mid];
    if ((values.size() % 2) == 0 && mid > 0)
    {
      std::nth_element(values.begin(), values.begin() + static_cast<long>(mid - 1), values.end());
      med = 0.5 * (med + values[mid - 1]);
    }
    return med;
  }

  static double robustMad(std::vector<double> values, double center)
  {
    for (double& v : values)
    {
      v = std::fabs(v - center);
    }
    return robustMedian(values);
  }

  void addResidualSample(std::vector<double>& residuals, double residual, double weight) const
  {
    if (!std::isfinite(residual) || weight <= 0.0 ||
        residuals.size() >= static_cast<std::size_t>(std::max(1, max_z_constraint_samples_)))
    {
      return;
    }
    const int copies = std::max(1, static_cast<int>(std::round(std::max(0.1, std::min(1.0, weight)) * 3.0)));
    for (int i = 0; i < copies && residuals.size() < static_cast<std::size_t>(max_z_constraint_samples_); ++i)
    {
      residuals.push_back(residual);
    }
  }

  void addZResidualSample(std::vector<double>& residuals,
                          std::vector<ZResidualSample>& fit_samples,
                          const Eigen::Vector3d& p,
                          double residual,
                          double weight,
                          bool use_for_slope) const
  {
    if (max_z_residual_abs_ > 0.0 && std::fabs(residual) > max_z_residual_abs_)
    {
      return;
    }
    addResidualSample(residuals, residual, weight);
    if (!use_for_slope || !std::isfinite(residual) || !std::isfinite(p.x()) || !std::isfinite(p.y()))
    {
      return;
    }
    if (fit_samples.size() >= static_cast<std::size_t>(std::max(1, max_z_constraint_samples_)))
    {
      return;
    }
    ZResidualSample sample;
    sample.xy = p.head<2>();
    sample.residual = residual;
    sample.weight = std::max(0.05, std::min(1.0, weight));
    fit_samples.push_back(sample);
  }

  void addXYResidualSample(std::vector<double>& residual_x,
                           std::vector<double>& residual_y,
                           const Eigen::Vector2d& residual,
                           double weight) const
  {
    const std::size_t limit = static_cast<std::size_t>(std::max(1, max_xy_constraint_samples_));
    if (!std::isfinite(residual.x()) || !std::isfinite(residual.y()) ||
        residual_x.size() >= limit || residual_y.size() >= limit)
    {
      return;
    }
    if (residual.norm() > std::max(0.05, max_xy_match_residual_))
    {
      return;
    }

    const int copies = std::max(1, static_cast<int>(std::round(std::max(0.1, std::min(1.0, weight)) * 3.0)));
    for (int i = 0; i < copies && residual_x.size() < limit && residual_y.size() < limit; ++i)
    {
      residual_x.push_back(residual.x());
      residual_y.push_back(residual.y());
    }
  }

  const SemanticAnchor*
  findNearestObjectAnchorIn(const std::unordered_map<VoxelKey, SemanticAnchor, VoxelKeyHash>& anchors,
                            const VoxelKey& key,
                            const Eigen::Vector3d& mean,
                            uint32_t label,
                            int search_radius_cells,
                            double* best_distance = nullptr) const
  {
    const int radius = std::max(0, search_radius_cells);
    const SemanticAnchor* best = nullptr;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int dx = -radius; dx <= radius; ++dx)
    {
      for (int dy = -radius; dy <= radius; ++dy)
      {
        const VoxelKey candidate{key.x + dx, key.y + dy, static_cast<int>(label)};
        auto it = anchors.find(candidate);
        if (it == anchors.end() ||
            it->second.observations < static_cast<uint32_t>(std::max(1, min_object_anchor_observations_)))
        {
          continue;
        }
        const Eigen::Vector2d delta(mean.x() - it->second.mean.x(), mean.y() - it->second.mean.y());
        const double dist = delta.norm();
        if (dist < best_dist)
        {
          best_dist = dist;
          best = &it->second;
        }
      }
    }
    if (best_distance)
    {
      *best_distance = best_dist;
    }
    return best;
  }

  const SemanticAnchor* findNearestObjectAnchor(const VoxelKey& key,
                                                const Eigen::Vector3d& mean,
                                                uint32_t label,
                                                double* best_distance = nullptr) const
  {
    return findNearestObjectAnchorIn(object_anchors_, key, mean, label, xy_anchor_search_radius_cells_, best_distance);
  }

  const SemanticAnchor* findZObjectAnchor(const VoxelKey& key,
                                          const Eigen::Vector3d& mean,
                                          uint32_t label) const
  {
    return findNearestObjectAnchorIn(z_object_anchors_, key, mean, label, 0, nullptr);
  }

  VoxelKey groundFilterKey(const Eigen::Vector3d& p) const
  {
    const double res = std::max(0.10, ground_filter_resolution_m_);
    return VoxelKey{static_cast<int>(std::floor(p.x() / res)),
                    static_cast<int>(std::floor(p.y() / res)),
                    0};
  }

  GroundFrameFilter buildGroundFrameFilter(const std::vector<FrameSemanticPoint>& points) const
  {
    GroundFrameFilter filter;
    if (!enable_ground_geometry_filter_ || points.empty())
    {
      filter.valid = !enable_ground_geometry_filter_;
      return filter;
    }

    std::vector<double> z_values;
    z_values.reserve(points.size());
    for (const auto& fp : points)
    {
      if (fp.label == LABEL_DYNAMIC)
      {
        continue;
      }
      const Eigen::Vector3d p = correctedPointForZAnchors(fp.p);
      if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
      {
        continue;
      }
      z_values.push_back(p.z());
      const VoxelKey key = groundFilterKey(p);
      GroundFrameFilter::Cell& cell = filter.cells[key];
      cell.min_z = std::min(cell.min_z, p.z());
      ++cell.count;
    }

    if (z_values.empty() || filter.cells.empty())
    {
      filter.valid = false;
      return filter;
    }
    const double pct = std::max(0.01, std::min(0.50, ground_filter_frame_percentile_));
    std::size_t idx = static_cast<std::size_t>(pct * static_cast<double>(z_values.size()));
    idx = std::min(idx, z_values.size() - 1);
    std::nth_element(z_values.begin(), z_values.begin() + static_cast<std::ptrdiff_t>(idx), z_values.end());
    filter.frame_ground_z = z_values[idx];
    filter.valid = true;
    return filter;
  }

  bool passesGroundGeometryFilter(const Eigen::Vector3d& p,
                                  const GroundFrameFilter& filter) const
  {
    if (!enable_ground_geometry_filter_)
    {
      return true;
    }
    if (!filter.valid)
    {
      return false;
    }
    const VoxelKey key = groundFilterKey(p);
    const int radius = std::max(0, ground_filter_neighbor_radius_cells_);
    const int min_points = std::max(1, ground_filter_min_cell_points_);
    double neighborhood_min_z = std::numeric_limits<double>::infinity();
    int neighborhood_points = 0;
    for (int dx = -radius; dx <= radius; ++dx)
    {
      for (int dy = -radius; dy <= radius; ++dy)
      {
        const VoxelKey candidate{key.x + dx, key.y + dy, 0};
        const auto it = filter.cells.find(candidate);
        if (it == filter.cells.end())
        {
          continue;
        }
        neighborhood_min_z = std::min(neighborhood_min_z, it->second.min_z);
        neighborhood_points += it->second.count;
      }
    }
    if (!std::isfinite(neighborhood_min_z) || neighborhood_points < min_points)
    {
      return false;
    }
    const double max_above_cell = std::max(0.0, max_ground_height_above_cell_min_);
    if (p.z() > neighborhood_min_z + max_above_cell)
    {
      return false;
    }
    const double max_above_frame = std::max(0.0, max_ground_height_above_frame_ground_);
    if (max_above_frame > 0.0 && p.z() > filter.frame_ground_z + max_above_frame)
    {
      return false;
    }
    return true;
  }

  const SemanticAnchor* findGroundAnchor(const VoxelKey& key,
                                         const Eigen::Vector3d& p,
                                         const ros::Time& stamp,
                                         double* best_distance = nullptr) const
  {
    const int radius = std::max(0, ground_anchor_search_radius_cells_);
    const double max_dist = std::max(0.0, max_ground_anchor_match_xy_m_);
    const double min_age = std::max(0.0, min_ground_anchor_age_sec_);
    const SemanticAnchor* best = nullptr;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int dx = -radius; dx <= radius; ++dx)
    {
      for (int dy = -radius; dy <= radius; ++dy)
      {
        const VoxelKey candidate{key.x + dx, key.y + dy, 0};
        auto it = ground_anchors_.find(candidate);
        if (it == ground_anchors_.end() ||
            it->second.observations < static_cast<uint32_t>(std::max(1, min_ground_anchor_observations_)))
        {
          continue;
        }
        if (min_age > 0.0 && !stamp.isZero() && !it->second.last_update.isZero())
        {
          const double age = (stamp - it->second.last_update).toSec();
          if (std::isfinite(age) && age < min_age)
          {
            continue;
          }
        }
        const double dist = (p.head<2>() - it->second.mean.head<2>()).norm();
        if (max_dist > 0.0 && dist > max_dist)
        {
          continue;
        }
        if (dist < best_dist)
        {
          best_dist = dist;
          best = &it->second;
        }
      }
    }
    if (best_distance)
    {
      *best_distance = best_dist;
    }
    return best;
  }

  std::unordered_map<VoxelKey, AnchorAccum, VoxelKeyHash>
  buildObjectAccum(const std::vector<FrameSemanticPoint>& points, bool full_correction) const
  {
    std::unordered_map<VoxelKey, AnchorAccum, VoxelKeyHash> accum;
    for (const auto& fp : points)
    {
      if (!isObjectAnchorLabel(fp.label))
      {
        continue;
      }
      const Eigen::Vector3d p = full_correction ? applySemanticCorrection(fp.p) : correctedPointForZAnchors(fp.p);
      const VoxelKey key = anchorKey(p, object_anchor_resolution_, static_cast<int>(fp.label));
      AnchorAccum& a = accum[key];
      const double w = std::max(0.05, fp.weight);
      a.sum += p * w;
      a.z_sum += p.z() * w;
      a.weight_sum += w;
      a.count += 1;
    }
    return accum;
  }

  void updateGroundAnchors(const std::vector<FrameSemanticPoint>& points,
                           const ros::Time& stamp,
                           const GroundFrameFilter& ground_filter)
  {
    last_ground_anchor_update_skipped_ = 0;
    for (const auto& fp : points)
    {
      if (!isGroundLabel(fp.label))
      {
        continue;
      }
      const Eigen::Vector3d p = correctedPointForZAnchors(fp.p);
      if (!passesGroundGeometryFilter(p, ground_filter))
      {
        ++last_ground_geometry_rejects_;
        continue;
      }
      const VoxelKey key = anchorKey(p, z_anchor_resolution_, 0);
      SemanticAnchor& anchor = ground_anchors_[key];
      const double w = std::max(0.05, fp.weight);
      if (anchor.observations == 0)
      {
        anchor.mean = p;
        anchor.z = p.z();
        anchor.weight = w;
      }
      else
      {
        const double update_gate = std::max(0.0, max_ground_anchor_update_z_residual_);
        const bool mature_anchor =
            anchor.observations >= static_cast<uint32_t>(std::max(1, min_ground_anchor_observations_));
        if (update_gate > 0.0 && mature_anchor && std::fabs(p.z() - anchor.z) > update_gate)
        {
          ++last_ground_anchor_update_skipped_;
          continue;
        }
        const double a = std::max(0.001, std::min(0.5, anchor_update_alpha_));
        anchor.mean = (1.0 - a) * anchor.mean + a * p;
        anchor.z = (1.0 - a) * anchor.z + a * p.z();
        anchor.weight = (1.0 - a) * anchor.weight + a * w;
      }
      anchor.observations += 1;
      anchor.last_update = stamp;
    }
  }

  void updateObjectAnchors(const std::unordered_map<VoxelKey, AnchorAccum, VoxelKeyHash>& accum,
                           const ros::Time& stamp,
                           std::unordered_map<VoxelKey, SemanticAnchor, VoxelKeyHash>& anchors)
  {
    for (const auto& kv : accum)
    {
      const AnchorAccum& a = kv.second;
      if (a.count < 8 || a.weight_sum <= 1e-6)
      {
        continue;
      }
      const Eigen::Vector3d mean = a.sum / a.weight_sum;
      const double z_mean = a.z_sum / a.weight_sum;
      SemanticAnchor& anchor = anchors[kv.first];
      if (anchor.observations == 0)
      {
        anchor.mean = mean;
        anchor.z = z_mean;
        anchor.weight = a.weight_sum;
      }
      else
      {
        const double alpha = std::max(0.001, std::min(0.5, anchor_update_alpha_ * 0.5));
        anchor.mean = (1.0 - alpha) * anchor.mean + alpha * mean;
        anchor.z = (1.0 - alpha) * anchor.z + alpha * z_mean;
        anchor.weight = (1.0 - alpha) * anchor.weight + alpha * a.weight_sum;
      }
      anchor.observations += 1;
      anchor.last_update = stamp;
    }
  }

  void updateSemanticXYCorrection(const std::vector<FrameSemanticPoint>& points, const ros::Time& stamp)
  {
    const double interval = std::max(0.0, pose_constraint_interval_sec_);
    if (interval > 0.0 && !last_pose_constraint_time_.isZero() &&
        (stamp - last_pose_constraint_time_).toSec() < interval)
    {
      return;
    }
    last_pose_constraint_time_ = stamp;

    last_xy_matches_ = 0;
    last_xy_residual_.setZero();
    last_xy_step_.setZero();
    last_yaw_residual_ = 0.0;
    last_yaw_step_ = 0.0;
    last_se2_residual_rms_ = 0.0;
    last_se2_spread_ = 0.0;
    last_pose_constraint_accepted_ = false;

    if (points.empty() || object_anchors_.empty())
    {
      pose_constraint_stale_frames_ += 1;
      publishConstraintDebug(stamp, "xy_constraint", {}, false, "empty_or_no_anchors",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }

    std::vector<double> residual_x;
    std::vector<double> residual_y;
    std::vector<SE2MatchSample> samples;
    std::vector<ConstraintDebugPair> debug_pairs;
    residual_x.reserve(static_cast<std::size_t>(std::max(1, max_xy_constraint_samples_)));
    residual_y.reserve(static_cast<std::size_t>(std::max(1, max_xy_constraint_samples_)));
    samples.reserve(static_cast<std::size_t>(std::max(1, max_xy_constraint_samples_)));
    debug_pairs.reserve(static_cast<std::size_t>(std::max(1, max_xy_constraint_samples_)));

    const auto object_accum_current = buildObjectAccum(points, true);
    for (const auto& kv : object_accum_current)
    {
      const uint32_t label = static_cast<uint32_t>(std::max(0, kv.first.z));
      if (!isXYObjectAnchorLabel(label))
      {
        continue;
      }
      const AnchorAccum& a = kv.second;
      if (a.count < 8 || a.weight_sum <= 1e-6)
      {
        continue;
      }
      const Eigen::Vector3d mean = a.sum / a.weight_sum;
      double anchor_distance = 0.0;
      const SemanticAnchor* anchor = findNearestObjectAnchor(kv.first, mean, label, &anchor_distance);
      if (!anchor || anchor_distance > std::max(0.05, max_xy_match_residual_))
      {
        continue;
      }
      const Eigen::Vector2d residual(mean.x() - anchor->mean.x(), mean.y() - anchor->mean.y());
      const double sample_weight = std::min(1.0, static_cast<double>(a.count) / 30.0);
      addXYResidualSample(residual_x, residual_y, residual, sample_weight);
      if (samples.size() < static_cast<std::size_t>(std::max(1, max_xy_constraint_samples_)))
      {
        SE2MatchSample sample;
        sample.source = mean.head<2>();
        sample.target = anchor->mean.head<2>();
        sample.weight = std::max(0.05, sample_weight);
        samples.push_back(sample);
        ConstraintDebugPair dbg;
        dbg.source = mean;
        dbg.target = anchor->mean;
        dbg.label = label;
        dbg.residual = residual.norm();
        dbg.weight = sample.weight;
        debug_pairs.push_back(dbg);
      }
    }

    last_xy_matches_ = static_cast<int>(std::min(residual_x.size(), residual_y.size()));
    if (last_xy_matches_ < std::max(1, min_xy_constraint_matches_))
    {
      pose_constraint_stale_frames_ += 1;
      pose_xy_variance_ = std::min(max_pose_xy_variance_, pose_xy_variance_ * 1.05 + 0.02);
      pose_yaw_variance_ = std::min(max_pose_yaw_variance_, pose_yaw_variance_ * 1.05 + 1e-5);
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, false, "too_few_xy_matches",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }

    const double rx = robustMedian(residual_x);
    const double ry = robustMedian(residual_y);
    last_xy_residual_ = Eigen::Vector2d(rx, ry);

    const int min_se2 = std::max(std::max(1, min_xy_constraint_matches_), min_se2_constraint_matches_);
    if (static_cast<int>(samples.size()) < min_se2)
    {
      pose_constraint_stale_frames_ += 1;
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, false, "too_few_se2_samples",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }

    double wsum = 0.0;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    for (const auto& sample : samples)
    {
      const double w = std::max(0.05, sample.weight);
      center += sample.source * w;
      wsum += w;
    }
    if (wsum <= 1e-6)
    {
      pose_constraint_stale_frames_ += 1;
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, false, "zero_weight",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }
    center /= wsum;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    double spread_sum = 0.0;
    for (const auto& sample : samples)
    {
      const double w = std::max(0.05, sample.weight);
      const Eigen::Vector2d d = sample.source - center;
      const Eigen::Vector2d rhs = sample.target - sample.source;
      Eigen::Matrix<double, 2, 3> A;
      A << 1.0, 0.0, -d.y(),
           0.0, 1.0,  d.x();
      H += w * (A.transpose() * A);
      g += w * (A.transpose() * rhs);
      spread_sum += w * d.squaredNorm();
    }
    last_se2_spread_ = std::sqrt(spread_sum / wsum);
    if (last_se2_spread_ < std::max(0.5, min_se2_spread_m_))
    {
      pose_constraint_stale_frames_ += 1;
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, false, "low_spread",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }
    if (std::fabs(H.determinant()) < 1e-9)
    {
      pose_constraint_stale_frames_ += 1;
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, false, "singular_se2",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }

    const Eigen::Vector3d delta = H.ldlt().solve(g);
    if (!std::isfinite(delta.x()) || !std::isfinite(delta.y()) || !std::isfinite(delta.z()))
    {
      pose_constraint_stale_frames_ += 1;
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, false, "nonfinite_solution",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }

    double err_sum = 0.0;
    for (const auto& sample : samples)
    {
      const double w = std::max(0.05, sample.weight);
      const Eigen::Vector2d d = sample.source - center;
      const Eigen::Vector2d pred(delta.x() - delta.z() * d.y(),
                                 delta.y() + delta.z() * d.x());
      const Eigen::Vector2d rhs = sample.target - sample.source;
      err_sum += w * (pred - rhs).squaredNorm();
    }
    last_se2_residual_rms_ = std::sqrt(err_sum / std::max(1e-6, wsum));
    if (last_se2_residual_rms_ > std::max(0.05, max_se2_residual_rms_))
    {
      pose_constraint_stale_frames_ += 1;
      pose_xy_variance_ = std::min(max_pose_xy_variance_, pose_xy_variance_ * 1.10 + 0.03);
      pose_yaw_variance_ = std::min(max_pose_yaw_variance_, pose_yaw_variance_ * 1.10 + 2e-5);
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, false, "high_se2_rms",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }

    if (!enable_semantic_xy_correction_)
    {
      pose_constraint_stale_frames_ += 1;
      publishConstraintDebug(stamp, "xy_constraint", debug_pairs, true, "debug_only_xy_correction_disabled",
                             xy_constraint_debug_pub_, xy_constraint_points_pub_);
      return;
    }

    Eigen::Vector2d step_xy(delta.x(), delta.y());
    step_xy *= std::max(0.0, std::min(1.0, xy_correction_alpha_));
    const double max_step = std::max(0.001, max_xy_correction_step_);
    const double step_norm = step_xy.norm();
    if (step_norm > max_step)
    {
      step_xy *= max_step / step_norm;
    }

    double yaw_step = 0.0;
    if (enable_semantic_yaw_correction_)
    {
      yaw_step = std::max(0.0, std::min(1.0, yaw_correction_alpha_)) * delta.z();
      const double max_yaw_step = std::max(1e-6, max_yaw_correction_step_deg_ * M_PI / 180.0);
      yaw_step = std::max(-max_yaw_step, std::min(max_yaw_step, yaw_step));
    }

    const double c = std::cos(yaw_step);
    const double s = std::sin(yaw_step);
    Eigen::Matrix2d R_step;
    R_step << c, -s, s, c;
    if (!has_z_correction_origin_)
    {
      z_correction_origin_xy_ = center;
      has_z_correction_origin_ = true;
    }
    semantic_xy_correction_ =
        R_step * (z_correction_origin_xy_ + semantic_xy_correction_ - center) +
        center + step_xy - z_correction_origin_xy_;
    semantic_yaw_correction_ = normalizeAngle(semantic_yaw_correction_ + yaw_step);
    const double max_abs_yaw = std::max(0.0, max_abs_yaw_correction_deg_ * M_PI / 180.0);
    semantic_yaw_correction_ = std::max(-max_abs_yaw, std::min(max_abs_yaw, semantic_yaw_correction_));
    const double max_abs = std::max(0.0, max_abs_xy_correction_);
    semantic_xy_correction_.x() = std::max(-max_abs, std::min(max_abs, semantic_xy_correction_.x()));
    semantic_xy_correction_.y() = std::max(-max_abs, std::min(max_abs, semantic_xy_correction_.y()));
    last_xy_step_ = step_xy;
    last_yaw_residual_ = delta.z();
    last_yaw_step_ = yaw_step;
    last_pose_constraint_accepted_ = true;
    pose_constraint_stale_frames_ = 0;
    pose_xy_variance_ = std::max(min_pose_xy_variance_,
                                 (last_se2_residual_rms_ * last_se2_residual_rms_) /
                                     std::max(1.0, static_cast<double>(samples.size())));
    pose_yaw_variance_ = std::max(min_pose_yaw_variance_,
                                  std::fabs(delta.z()) * std::fabs(delta.z()) /
                                      std::max(1.0, static_cast<double>(samples.size())));
    publishConstraintDebug(stamp, "xy_constraint", debug_pairs, true, "accepted",
                           xy_constraint_debug_pub_, xy_constraint_points_pub_);
  }

  void publishSemanticConstraintStatus(const ros::Time& stamp)
  {
    std::ostringstream ss;
    ss << "{";
    ss << "\"x_correction\":" << semantic_xy_correction_.x() << ",";
    ss << "\"y_correction\":" << semantic_xy_correction_.y() << ",";
    ss << "\"yaw_correction_deg\":" << semantic_yaw_correction_ * 180.0 / M_PI << ",";
    ss << "\"z_correction\":" << semantic_z_correction_ << ",";
    ss << "\"z_slope_x\":" << semantic_z_slope_.x() << ",";
    ss << "\"z_slope_y\":" << semantic_z_slope_.y() << ",";
    ss << "\"xy_last_residual_x\":" << last_xy_residual_.x() << ",";
    ss << "\"xy_last_residual_y\":" << last_xy_residual_.y() << ",";
    ss << "\"xy_last_step_x\":" << last_xy_step_.x() << ",";
    ss << "\"xy_last_step_y\":" << last_xy_step_.y() << ",";
    ss << "\"xy_matches\":" << last_xy_matches_ << ",";
    ss << "\"yaw_last_residual_deg\":" << last_yaw_residual_ * 180.0 / M_PI << ",";
    ss << "\"yaw_last_step_deg\":" << last_yaw_step_ * 180.0 / M_PI << ",";
    ss << "\"se2_residual_rms\":" << last_se2_residual_rms_ << ",";
    ss << "\"se2_spread\":" << last_se2_spread_ << ",";
    ss << "\"pose_constraint_accepted\":" << (last_pose_constraint_accepted_ ? "true" : "false") << ",";
    ss << "\"pose_constraint_stale_frames\":" << pose_constraint_stale_frames_ << ",";
    ss << "\"semantic_correction_trust\":" << semanticCorrectionTrust() << ",";
    ss << "\"z_constraint_stale_frames\":" << z_constraint_stale_frames_ << ",";
    ss << "\"semantic_z_correction_trust\":" << semanticZCorrectionTrust() << ",";
    ss << "\"ground_robot_z_correction\":" << ground_robot_z_correction_ << ",";
    ss << "\"ground_robot_prior_z\":" << ground_robot_prior_z_ << ",";
    ss << "\"ground_robot_reference_z\":" << ground_robot_reference_z_ << ",";
    ss << "\"ground_robot_z_residual\":" << last_ground_robot_z_residual_ << ",";
    ss << "\"ground_robot_z_step\":" << last_ground_robot_z_step_ << ",";
    ss << "\"pose_xy_variance\":" << pose_xy_variance_ << ",";
    ss << "\"pose_yaw_variance\":" << pose_yaw_variance_ << ",";
    ss << "\"z_last_residual\":" << last_z_residual_ << ",";
    ss << "\"z_last_mad\":" << last_z_mad_ << ",";
    ss << "\"ground_anchor_update_skipped\":" << last_ground_anchor_update_skipped_ << ",";
    ss << "\"ground_geometry_rejects\":" << last_ground_geometry_rejects_ << ",";
    ss << "\"z_accept_streak\":" << z_constraint_accept_streak_ << ",";
    ss << "\"z_candidate_residual\":" << z_candidate_residual_ << ",";
    ss << "\"z_last_step\":" << last_z_step_ << ",";
    ss << "\"z_matches\":" << last_z_matches_ << ",";
    ss << "\"local_semantic_frames\":" << last_local_semantic_frames_ << ",";
    ss << "\"local_semantic_points\":" << last_local_semantic_points_ << ",";
    ss << "\"local_semantic_constraints\":" << local_semantic_constraints_ << ",";
    ss << "\"z_slope_samples\":" << last_z_slope_samples_ << ",";
    ss << "\"z_slope_spread\":" << last_z_slope_spread_ << ",";
    ss << "\"z_slope_fit_x\":" << last_z_slope_fit_.x() << ",";
    ss << "\"z_slope_fit_y\":" << last_z_slope_fit_.y() << ",";
    ss << "\"z_slope_step_x\":" << last_z_slope_step_.x() << ",";
    ss << "\"z_slope_step_y\":" << last_z_slope_step_.y() << ",";
    ss << "\"ground_anchors\":" << ground_anchors_.size() << ",";
    ss << "\"object_anchors\":" << object_anchors_.size() << ",";
    ss << "\"z_object_anchors\":" << z_object_anchors_.size();
    ss << "}";
    std_msgs::String msg;
    msg.data = ss.str();
    z_constraint_pub_.publish(msg);
    publishCorrectedOdom(stamp);
  }

  void updateSemanticZSlopeCorrection(const std::vector<ZResidualSample>& samples, double scalar_residual)
  {
    last_z_slope_step_.setZero();
    last_z_slope_fit_.setZero();
    last_z_slope_samples_ = static_cast<int>(samples.size());
    last_z_slope_spread_ = 0.0;
    if (!enable_semantic_z_slope_correction_ ||
        last_z_slope_samples_ < std::max(1, min_z_slope_constraint_matches_))
    {
      return;
    }

    double weight_sum = 0.0;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    for (const auto& sample : samples)
    {
      const double w = std::max(0.05, sample.weight);
      center += sample.xy * w;
      weight_sum += w;
    }
    if (weight_sum <= 1e-6)
    {
      return;
    }
    center /= weight_sum;
    if (!has_z_correction_origin_)
    {
      z_correction_origin_xy_ = center;
      has_z_correction_origin_ = true;
    }

    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b = Eigen::Vector2d::Zero();
    double spread_sum = 0.0;
    for (const auto& sample : samples)
    {
      const double w = std::max(0.05, sample.weight);
      const Eigen::Vector2d d = sample.xy - center;
      const double r = sample.residual - scalar_residual;
      H += w * (d * d.transpose());
      b += w * d * r;
      spread_sum += w * d.squaredNorm();
    }
    last_z_slope_spread_ = std::sqrt(spread_sum / weight_sum);
    if (last_z_slope_spread_ < std::max(0.5, min_z_slope_spread_m_))
    {
      return;
    }
    const double det = H.determinant();
    if (std::fabs(det) < 1e-9)
    {
      return;
    }

    const Eigen::Vector2d fit = H.ldlt().solve(b);
    if (!std::isfinite(fit.x()) || !std::isfinite(fit.y()))
    {
      return;
    }
    last_z_slope_fit_ = fit;

    const double alpha = std::max(0.0, std::min(1.0, z_slope_correction_alpha_));
    Eigen::Vector2d step = -alpha * fit;
    const double max_step = std::max(1e-6, max_z_slope_correction_step_);
    const double step_norm = step.norm();
    if (step_norm > max_step)
    {
      step *= max_step / step_norm;
    }

    semantic_z_slope_ += step;
    const double max_abs = std::max(1e-6, max_abs_z_slope_);
    const double slope_norm = semantic_z_slope_.norm();
    if (slope_norm > max_abs)
    {
      semantic_z_slope_ *= max_abs / slope_norm;
    }

    // Rotate the correction plane around the current local observation center so
    // the slope update does not introduce a sudden Z jump at the vehicle.
    semantic_z_correction_ -= step.dot(center - z_correction_origin_xy_);
    semantic_z_correction_ = std::max(-max_abs_z_correction_, std::min(max_abs_z_correction_, semantic_z_correction_));
    last_z_slope_step_ = step;
  }

  void updateSemanticZCorrection(const std::vector<FrameSemanticPoint>& points, const ros::Time& stamp)
  {
    last_z_matches_ = 0;
    last_z_residual_ = 0.0;
    last_z_mad_ = 0.0;
    last_z_step_ = 0.0;
    last_ground_geometry_rejects_ = 0;
    if (points.empty())
    {
      z_constraint_stale_frames_ += 1;
      z_constraint_accept_streak_ = 0;
      has_z_candidate_residual_ = false;
      publishConstraintDebug(stamp, "z_constraint", {}, false, "empty_points",
                             z_constraint_debug_pub_, z_constraint_points_pub_);
      return;
    }

    const GroundFrameFilter ground_filter = buildGroundFrameFilter(points);
    std::vector<ConstraintDebugPair> debug_pairs;
    debug_pairs.reserve(static_cast<std::size_t>(std::max(1, constraint_debug_max_pairs_)));
    if (enable_semantic_z_correction_)
    {
      std::vector<double> residuals;
      std::vector<ZResidualSample> z_fit_samples;
      residuals.reserve(static_cast<std::size_t>(std::max(1, max_z_constraint_samples_)));
      z_fit_samples.reserve(static_cast<std::size_t>(std::max(1, max_z_constraint_samples_)));

      for (const auto& fp : points)
      {
        if (!isGroundLabel(fp.label))
        {
          continue;
        }
        const Eigen::Vector3d p = correctedPointForZAnchors(fp.p);
        if (!passesGroundGeometryFilter(p, ground_filter))
        {
          ++last_ground_geometry_rejects_;
          continue;
        }
        const VoxelKey key = anchorKey(p, z_anchor_resolution_, 0);
        double anchor_distance = 0.0;
        const SemanticAnchor* anchor = findGroundAnchor(key, p, stamp, &anchor_distance);
        if (!anchor)
        {
          continue;
        }
        const double residual = p.z() - anchor->z;
        const double max_match_dist = std::max(0.05, max_ground_anchor_match_xy_m_);
        const double distance_weight =
            std::max(0.20, 1.0 - std::min(1.0, anchor_distance / max_match_dist));
        const double sample_weight = std::max(0.05, fp.weight) * distance_weight;
        if (debug_pairs.size() < static_cast<std::size_t>(std::max(1, constraint_debug_max_pairs_)))
        {
          ConstraintDebugPair dbg;
          dbg.source = p;
          dbg.target = Eigen::Vector3d(anchor->mean.x(), anchor->mean.y(), anchor->z);
          dbg.label = fp.label;
          dbg.residual = residual;
          dbg.weight = sample_weight;
          debug_pairs.push_back(dbg);
        }
        addZResidualSample(residuals, z_fit_samples, p, residual, sample_weight, true);
      }

      if (use_object_z_anchors_ && object_anchor_weight_ > 0.0)
      {
        const auto object_accum_current = buildObjectAccum(points, false);
        for (const auto& kv : object_accum_current)
        {
          const AnchorAccum& a = kv.second;
          if (a.count < 8 || a.weight_sum <= 1e-6)
          {
            continue;
          }
          const Eigen::Vector3d mean = a.sum / a.weight_sum;
          const uint32_t label = static_cast<uint32_t>(std::max(0, kv.first.z));
          const SemanticAnchor* anchor = findZObjectAnchor(kv.first, mean, label);
          if (!anchor)
          {
            continue;
          }
          const double residual = mean.z() - anchor->z;
          if (debug_pairs.size() < static_cast<std::size_t>(std::max(1, constraint_debug_max_pairs_)))
          {
            ConstraintDebugPair dbg;
            dbg.source = mean;
            dbg.target = Eigen::Vector3d(anchor->mean.x(), anchor->mean.y(), anchor->z);
            dbg.label = label;
            dbg.residual = residual;
            dbg.weight = object_anchor_weight_;
            debug_pairs.push_back(dbg);
          }
          addZResidualSample(residuals, z_fit_samples, mean, residual, object_anchor_weight_, false);
        }
      }

      last_z_matches_ = static_cast<int>(residuals.size());
      if (last_z_matches_ >= std::max(1, min_z_constraint_matches_))
      {
        const double residual = robustMedian(residuals);
        last_z_mad_ = robustMad(residuals, residual);
        const bool residual_ok = max_z_residual_abs_ <= 0.0 || std::fabs(residual) <= max_z_residual_abs_;
        const bool mad_ok = max_z_residual_mad_ <= 0.0 || last_z_mad_ <= max_z_residual_mad_;
        if (!residual_ok || !mad_ok)
        {
          z_constraint_stale_frames_ += 1;
          z_constraint_accept_streak_ = 0;
          has_z_candidate_residual_ = false;
          pose_z_variance_ = std::min(max_pose_z_variance_, pose_z_variance_ * 1.10 + 0.05);
          last_z_residual_ = residual;
          updateGroundAnchors(points, stamp, ground_filter);
          updateObjectAnchors(buildObjectAccum(points, false), stamp, z_object_anchors_);
          updateObjectAnchors(buildObjectAccum(points, true), stamp, object_anchors_);
          publishConstraintDebug(stamp, "z_constraint", debug_pairs, false,
                                 !residual_ok ? "median_residual_too_large" : "mad_too_large",
                                 z_constraint_debug_pub_, z_constraint_points_pub_);
          publishSemanticConstraintStatus(stamp);
          return;
        }

        const double jump_gate = std::max(0.0, max_z_residual_jump_);
        if (!has_z_candidate_residual_ ||
            (jump_gate > 0.0 && std::fabs(residual - z_candidate_residual_) > jump_gate))
        {
          z_candidate_residual_ = residual;
          has_z_candidate_residual_ = true;
          z_constraint_accept_streak_ = 1;
        }
        else
        {
          z_candidate_residual_ = 0.75 * z_candidate_residual_ + 0.25 * residual;
          z_constraint_accept_streak_ += 1;
        }

        last_z_residual_ = residual;
        const int min_streak = std::max(1, min_z_constraint_accept_streak_);
        if (z_constraint_accept_streak_ < min_streak)
        {
          z_constraint_stale_frames_ += 1;
          pose_z_variance_ = std::min(max_pose_z_variance_, pose_z_variance_ * 1.02 + 0.01);
          updateGroundAnchors(points, stamp, ground_filter);
          updateObjectAnchors(buildObjectAccum(points, false), stamp, z_object_anchors_);
          updateObjectAnchors(buildObjectAccum(points, true), stamp, object_anchors_);
          publishConstraintDebug(stamp, "z_constraint", debug_pairs, false, "wait_stable_z_residual",
                                 z_constraint_debug_pub_, z_constraint_points_pub_);
          publishSemanticConstraintStatus(stamp);
          return;
        }

        const double deadband = std::max(0.0, z_correction_deadband_);
        const double gated_residual = std::fabs(z_candidate_residual_) < deadband ? 0.0 : z_candidate_residual_;
        const double desired_step = -std::max(0.0, std::min(1.0, z_correction_alpha_)) * gated_residual;
        const double max_step = std::max(0.001, max_z_correction_step_);
        const double step = std::max(-max_step, std::min(max_step, desired_step));
        semantic_z_correction_ += step;
        semantic_z_correction_ = std::max(-max_abs_z_correction_, std::min(max_abs_z_correction_, semantic_z_correction_));
        last_z_step_ = step;
        z_constraint_stale_frames_ = 0;
        pose_z_variance_ = std::max(min_pose_z_variance_,
                                    (residual * residual) / std::max(1.0, static_cast<double>(last_z_matches_)));
        updateSemanticZSlopeCorrection(z_fit_samples, residual);
        publishConstraintDebug(stamp, "z_constraint", debug_pairs, true, "accepted",
                               z_constraint_debug_pub_, z_constraint_points_pub_);
      }
      else
      {
        z_constraint_stale_frames_ += 1;
        z_constraint_accept_streak_ = 0;
        has_z_candidate_residual_ = false;
        pose_z_variance_ = std::min(max_pose_z_variance_, pose_z_variance_ * 1.05 + 0.02);
        publishConstraintDebug(stamp, "z_constraint", debug_pairs, false, "too_few_z_matches",
                               z_constraint_debug_pub_, z_constraint_points_pub_);
      }
    }
    else
    {
      z_constraint_accept_streak_ = 0;
      has_z_candidate_residual_ = false;
      publishConstraintDebug(stamp, "z_constraint", debug_pairs, false, "disabled",
                             z_constraint_debug_pub_, z_constraint_points_pub_);
    }

    updateGroundAnchors(points, stamp, ground_filter);
    updateObjectAnchors(buildObjectAccum(points, false), stamp, z_object_anchors_);
    updateObjectAnchors(buildObjectAccum(points, true), stamp, object_anchors_);
    publishSemanticConstraintStatus(stamp);
  }

  bool localSemanticCorrectionLabel(uint32_t label) const
  {
    return isGroundLabel(label) || isXYObjectAnchorLabel(label);
  }

  LocalSemanticFrame makeLocalSemanticFrame(const std::vector<FrameSemanticPoint>& points,
                                            const ros::Time& stamp) const
  {
    LocalSemanticFrame frame;
    frame.stamp = stamp;
    const std::size_t limit =
        static_cast<std::size_t>(std::max(1, local_semantic_max_points_per_frame_));
    const std::size_t usable = points.size();
    const std::size_t stride = std::max<std::size_t>(1, usable / limit);
    frame.points.reserve(std::min<std::size_t>(usable, limit));
    for (std::size_t i = 0; i < usable && frame.points.size() < limit; i += stride)
    {
      const FrameSemanticPoint& fp = points[i];
      if (!localSemanticCorrectionLabel(fp.label))
      {
        continue;
      }
      frame.points.push_back(fp);
    }
    return frame;
  }

  std::vector<FrameSemanticPoint> combinedLocalSemanticPoints() const
  {
    std::vector<FrameSemanticPoint> combined;
    std::size_t total = 0;
    for (const auto& frame : local_semantic_frames_)
    {
      total += frame.points.size();
    }
    combined.reserve(total);
    for (const auto& frame : local_semantic_frames_)
    {
      combined.insert(combined.end(), frame.points.begin(), frame.points.end());
    }
    return combined;
  }

  void updateLocalSemanticMapCorrection(const std::vector<FrameSemanticPoint>& points,
                                        const ros::Time& stamp)
  {
    LocalSemanticFrame frame = makeLocalSemanticFrame(points, stamp);
    if (!frame.points.empty())
    {
      local_semantic_frames_.push_back(std::move(frame));
    }

    const int max_frames = std::max(1, local_semantic_max_frames_);
    while (static_cast<int>(local_semantic_frames_.size()) > max_frames)
    {
      local_semantic_frames_.pop_front();
    }

    last_local_semantic_frames_ = static_cast<int>(local_semantic_frames_.size());
    last_local_semantic_points_ = 0;
    for (const auto& f : local_semantic_frames_)
    {
      last_local_semantic_points_ += static_cast<int>(f.points.size());
    }

    const bool enough_frames =
        static_cast<int>(local_semantic_frames_.size()) >= std::max(1, local_semantic_min_frames_);
    const double window_sec =
        local_semantic_frames_.empty() ? 0.0 :
        (local_semantic_frames_.back().stamp - local_semantic_frames_.front().stamp).toSec();
    if (!enough_frames || window_sec < std::max(0.0, local_semantic_window_sec_))
    {
      publishSemanticConstraintStatus(stamp);
      return;
    }

    std::vector<FrameSemanticPoint> local_points = combinedLocalSemanticPoints();
    if (local_points.empty())
    {
      local_semantic_frames_.clear();
      publishSemanticConstraintStatus(stamp);
      return;
    }

    ++local_semantic_constraints_;
    updateSemanticXYCorrection(local_points, stamp);
    updateSemanticZCorrection(local_points, stamp);
    local_semantic_frames_.clear();
    last_local_semantic_frames_ = 0;
    last_local_semantic_points_ = static_cast<int>(local_points.size());
  }

  VoxelKey instanceCellKey(const Eigen::Vector3d& p) const
  {
    const double res = std::max(0.10, instance_cluster_resolution_m_);
    return VoxelKey{static_cast<int>(std::floor(p.x() / res)),
                    static_cast<int>(std::floor(p.y() / res)),
                    0};
  }

  bool instanceLooksDynamic(const InstanceClusterStats& c, bool allow_geometry_guess) const
  {
    const int count = static_cast<int>(c.indices.size());
    if (count <= 0)
    {
      return false;
    }
    const double dynamic_ratio =
        static_cast<double>(c.label_counts[LABEL_DYNAMIC]) / std::max(1, count);
    if (dynamic_ratio >= std::max(0.0, instance_dynamic_label_ratio_))
    {
      return true;
    }
    if (!allow_geometry_guess)
    {
      return false;
    }

    const Eigen::Vector3d span = c.max_pt - c.min_pt;
    const double length = std::max(span.x(), span.y());
    const double width = std::min(span.x(), span.y());
    const double height = span.z();
    const int building_like = c.label_counts[LABEL_BUILDING] + c.label_counts[LABEL_OTHER];
    const double building_ratio = static_cast<double>(building_like) / std::max(1, count);
    const bool compact_object =
        length <= std::max(0.5, instance_dynamic_max_length_m_) &&
        width <= std::max(0.3, instance_dynamic_max_width_m_) &&
        height >= std::max(0.05, instance_dynamic_min_height_m_) &&
        height <= std::max(instance_dynamic_min_height_m_, instance_dynamic_max_height_m_);
    return compact_object && building_ratio < 0.65;
  }

  void filterDynamicInstances(std::vector<FrameSemanticPoint>& points,
                              int& instance_dynamic_skipped,
                              int& instance_noise_skipped) const
  {
    instance_dynamic_skipped = 0;
    instance_noise_skipped = 0;
    if (!enable_instance_dynamic_filter_ || points.empty())
    {
      return;
    }

    std::vector<uint8_t> remove(points.size(), 0);

    std::unordered_map<uint32_t, InstanceClusterStats> semantic_instances;
    semantic_instances.reserve(points.size() / 8 + 1);
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
    {
      const FrameSemanticPoint& p = points[static_cast<std::size_t>(i)];
      if (isGroundLabel(p.label) || p.instance_id == 0)
      {
        continue;
      }
      semantic_instances[p.instance_id].add(i, p);
    }
    for (const auto& kv : semantic_instances)
    {
      const InstanceClusterStats& cluster = kv.second;
      const int count = static_cast<int>(cluster.indices.size());
      if (count <= 0 || !instanceLooksDynamic(cluster, false))
      {
        continue;
      }
      for (const int idx : cluster.indices)
      {
        remove[static_cast<std::size_t>(idx)] = 1;
      }
      instance_dynamic_skipped += count;
    }

    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> cells;
    cells.reserve(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
    {
      const FrameSemanticPoint& p = points[static_cast<std::size_t>(i)];
      if (remove[static_cast<std::size_t>(i)] || isGroundLabel(p.label) || p.instance_id != 0)
      {
        continue;
      }
      cells[instanceCellKey(p.p)].push_back(i);
    }
    if (cells.empty() && instance_dynamic_skipped == 0)
    {
      return;
    }

    std::unordered_map<VoxelKey, bool, VoxelKeyHash> visited;
    visited.reserve(cells.size());
    std::vector<VoxelKey> stack;
    stack.reserve(256);
    for (const auto& seed : cells)
    {
      if (visited[seed.first])
      {
        continue;
      }
      InstanceClusterStats cluster;
      stack.clear();
      stack.push_back(seed.first);
      visited[seed.first] = true;
      while (!stack.empty())
      {
        const VoxelKey key = stack.back();
        stack.pop_back();
        const auto cell_it = cells.find(key);
        if (cell_it != cells.end())
        {
          for (const int idx : cell_it->second)
          {
            cluster.add(idx, points[static_cast<std::size_t>(idx)]);
          }
        }
        for (int dx = -1; dx <= 1; ++dx)
        {
          for (int dy = -1; dy <= 1; ++dy)
          {
            if (dx == 0 && dy == 0)
            {
              continue;
            }
            const VoxelKey next{key.x + dx, key.y + dy, 0};
            if (visited[next] || cells.find(next) == cells.end())
            {
              continue;
            }
            visited[next] = true;
            stack.push_back(next);
          }
        }
      }

      const int count = static_cast<int>(cluster.indices.size());
      const bool noise = count < std::max(1, instance_min_points_);
      const bool dynamic_instance = !noise && instanceLooksDynamic(cluster, true);
      if (!noise && !dynamic_instance)
      {
        continue;
      }
      for (const int idx : cluster.indices)
      {
        remove[static_cast<std::size_t>(idx)] = 1;
      }
      if (noise)
      {
        instance_noise_skipped += count;
      }
      else
      {
        instance_dynamic_skipped += count;
      }
    }

    if (instance_dynamic_skipped == 0 && instance_noise_skipped == 0)
    {
      return;
    }
    std::vector<FrameSemanticPoint> filtered;
    filtered.reserve(points.size() - static_cast<std::size_t>(instance_dynamic_skipped + instance_noise_skipped));
    for (std::size_t i = 0; i < points.size(); ++i)
    {
      if (!remove[i])
      {
        filtered.push_back(points[i]);
      }
    }
    points.swap(filtered);
  }

  void processCloud(const sensor_msgs::PointCloud2ConstPtr& msg,
                    double source_weight,
                    bool cloud_in_map_frame,
                    const std::string& label_mode,
                    bool publish_static_current_cloud)
  {
    if (!cloud_in_map_frame && latest_odom_time_.isZero())
    {
      ROS_WARN_THROTTLE(2.0, "Waiting for FAST-LIVO2 odometry on %s before fusing semantic cloud", odom_topic_.c_str());
      return;
    }

    const sensor_msgs::PointField* fx = findField(*msg, "x");
    const sensor_msgs::PointField* fy = findField(*msg, "y");
    const sensor_msgs::PointField* fz = findField(*msg, "z");
    const sensor_msgs::PointField* flabel = findFirstField(*msg, {label_field_name_, "semantic", "class_id", "label_id"});
    const sensor_msgs::PointField* fconf = findFirstField(*msg, {"confidence", "probability", "prob", "score"});
    const sensor_msgs::PointField* finstance = findFirstField(*msg, {"instance_id", "instance", "object_id"});
    if (!fx || !fy || !fz || !flabel)
    {
      ROS_WARN_THROTTLE(3.0,
                        "Semantic cloud needs x/y/z and label fields. topic frame=%s fields=%zu label_field=%s",
                        msg->header.frame_id.c_str(), msg->fields.size(), label_field_name_.c_str());
      return;
    }

    const Eigen::Isometry3d T_map_lidar = latest_T_map_body_ * T_body_lidar_;
    const std::size_t n = static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
    std::vector<OutputPoint> static_points;
    if (publish_static_labeled_cloud_ && publish_static_current_cloud)
    {
      static_points.reserve(std::min<std::size_t>(n, 200000));
    }
    std::vector<FrameSemanticPoint> frame_points;
    frame_points.reserve(std::min<std::size_t>(n, 250000));

    int dynamic_skipped = 0;
    int unknown_skipped = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const double x = readFieldAsDouble(*msg, *fx, i);
      const double y = readFieldAsDouble(*msg, *fy, i);
      const double z = readFieldAsDouble(*msg, *fz, i);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      {
        continue;
      }
      const double r = std::sqrt(x * x + y * y + z * z);
      if (r < min_range_m_ || r > max_range_m_)
      {
        continue;
      }

      const uint32_t raw_label = static_cast<uint32_t>(std::max(0.0, std::round(readFieldAsDouble(*msg, *flabel, i))));
      const uint32_t label = mapLabel(raw_label, label_mode);
      if (label == LABEL_UNKNOWN)
      {
        ++unknown_skipped;
        continue;
      }

      const double confidence = fconf ? readFieldAsDouble(*msg, *fconf, i) : default_confidence_;
      if (!std::isfinite(confidence) || confidence <= 0.0)
      {
        continue;
      }
      const uint32_t instance_id = finstance ?
          static_cast<uint32_t>(std::max(0.0, std::round(readFieldAsDouble(*msg, *finstance, i)))) : 0U;

      Eigen::Vector3d p_map(x, y, z);
      if (!cloud_in_map_frame)
      {
        p_map = T_map_lidar * p_map;
      }
      if (!std::isfinite(p_map.x()) || !std::isfinite(p_map.y()) || !std::isfinite(p_map.z()))
      {
        continue;
      }

      if (label == LABEL_DYNAMIC && !fuse_dynamic_objects_)
      {
        ++dynamic_skipped;
        continue;
      }

      const double w = source_weight * labelWeight(label) * std::max(0.05, std::min(1.0, confidence));
      if (w <= 0.0)
      {
        continue;
      }

      FrameSemanticPoint fp;
      fp.p = p_map;
      fp.label = label;
      fp.instance_id = instance_id;
      fp.confidence = confidence;
      fp.weight = w;
      frame_points.push_back(fp);
    }

    int instance_dynamic_skipped = 0;
    int instance_noise_skipped = 0;
    filterDynamicInstances(frame_points, instance_dynamic_skipped, instance_noise_skipped);

    const ros::Time stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    if (enable_local_semantic_map_constraint_)
    {
      if (!local_semantic_lidar_only_ || publish_static_current_cloud)
      {
        updateLocalSemanticMapCorrection(frame_points, stamp);
      }
      else
      {
        publishSemanticConstraintStatus(stamp);
      }
    }
    else
    {
      updateSemanticXYCorrection(frame_points, stamp);
      updateSemanticZCorrection(frame_points, stamp);
    }

    int used = 0;
    const bool use_rebuilt_for_this_cloud = use_keyframe_rebuilt_map_ && !cloud_in_map_frame;
    if (use_rebuilt_for_this_cloud)
    {
      const Eigen::Isometry3d corrected_pose = groundRobotCorrectedPose(latest_T_map_body_);
      std::vector<OutputPoint> frame_map_points =
          rebuildFramePoints(frame_points, latest_T_map_body_, corrected_pose, 0);
      used = static_cast<int>(frame_map_points.size());
      maybeAddRebuiltKeyframe(frame_points, stamp, publish_static_current_cloud);
      if (publish_static_labeled_cloud_ && publish_static_current_cloud)
      {
        static_points = std::move(frame_map_points);
      }
    }
    else
    {
      for (const auto& fp : frame_points)
      {
        const Eigen::Vector3d p_map = applySemanticCorrection(fp.p);
        if (p_map.z() < z_min_map_ || p_map.z() > z_max_map_)
        {
          continue;
        }

        if (publish_static_labeled_cloud_ && publish_static_current_cloud && fp.label != LABEL_DYNAMIC)
        {
          OutputPoint pt;
          pt.x = static_cast<float>(p_map.x());
          pt.y = static_cast<float>(p_map.y());
          pt.z = static_cast<float>(p_map.z());
          pt.label = fp.label;
          pt.confidence = static_cast<float>(std::max(0.0, std::min(1.0, fp.confidence)));
          pt.votes = 1.0f;
          static_points.push_back(pt);
        }

        VoxelState& state = voxels_[voxelKey(p_map)];
        state.weighted_sum += p_map * fp.weight;
        state.weight_sum += fp.weight;
        state.votes[fp.label] += static_cast<float>(fp.weight);
        state.last_update = stamp;
        state.observations += 1;
        ++used;
      }
    }

    if (publish_static_labeled_cloud_ && publish_static_current_cloud)
    {
      publishPointCloud(static_points, static_labeled_cloud_topic_, static_cloud_pub_, stamp);
    }

    if (publish_static_current_cloud)
    {
      last_lidar_points_ = used;
    }
    else
    {
      last_image_points_ = used;
    }
    ROS_INFO_THROTTLE(3.0,
                      "semantic fusion input used=%d voxels=%zu rebuilt_keyframes=%zu unknown_skip=%d dynamic_skip=%d instance_dynamic_skip=%d instance_noise_skip=%d source=%s",
                      used, voxels_.size(), rebuilt_keyframes_.size(), unknown_skipped, dynamic_skipped,
                      instance_dynamic_skipped, instance_noise_skipped,
                      publish_static_current_cloud ? "rangenet" : "segformer");
  }

  void bestLabelAndConfidence(const VoxelState& state, uint32_t& best_label, float& confidence, float& total_votes) const
  {
    best_label = LABEL_UNKNOWN;
    float best = 0.0f;
    float second = 0.0f;
    total_votes = 0.0f;
    for (uint32_t label = 1; label < LABEL_COUNT; ++label)
    {
      const float v = state.votes[label];
      total_votes += v;
      if (v > best)
      {
        second = best;
        best = v;
        best_label = label;
      }
      else if (v > second)
      {
        second = v;
      }
    }

    if (total_votes <= 1e-5f || best_label == LABEL_UNKNOWN)
    {
      confidence = 0.0f;
      return;
    }
    const float purity = best / total_votes;
    const float margin = (best - second) / total_votes;
    const float vote_mag = 1.0f - std::exp(-total_votes / static_cast<float>(std::max(0.1, confidence_vote_scale_)));
    confidence = std::max(0.0f, std::min(1.0f, (0.65f * purity + 0.35f * margin) * (0.45f + 0.55f * vote_mag)));
  }

  std::vector<OutputPoint> collectMapPoints(std::array<int, LABEL_COUNT>* counts = nullptr) const
  {
    if (use_keyframe_rebuilt_map_ && !rebuilt_keyframes_.empty())
    {
      return collectRebuiltMapPoints(counts);
    }
    if (counts)
    {
      counts->fill(0);
    }
    std::vector<OutputPoint> points;
    points.reserve(std::min<std::size_t>(voxels_.size(), static_cast<std::size_t>(std::max(1, max_publish_points_))));
    int stride = 1;
    if (max_publish_points_ > 0 && static_cast<int>(voxels_.size()) > max_publish_points_)
    {
      stride = static_cast<int>(std::ceil(static_cast<double>(voxels_.size()) / static_cast<double>(max_publish_points_)));
    }

    int index = 0;
    for (const auto& kv : voxels_)
    {
      ++index;
      if (stride > 1 && (index % stride) != 0)
      {
        continue;
      }
      const VoxelState& state = kv.second;
      if (state.weight_sum <= 1e-6)
      {
        continue;
      }
      uint32_t label = LABEL_UNKNOWN;
      float conf = 0.0f;
      float votes = 0.0f;
      bestLabelAndConfidence(state, label, conf, votes);
      if (label == LABEL_UNKNOWN || votes < min_votes_ || conf < min_confidence_)
      {
        continue;
      }
      const Eigen::Vector3d p = state.weighted_sum / state.weight_sum;
      OutputPoint out;
      out.x = static_cast<float>(p.x());
      out.y = static_cast<float>(p.y());
      out.z = static_cast<float>(p.z());
      out.label = label;
      out.confidence = conf;
      out.votes = votes;
      points.push_back(out);
      if (counts && label < LABEL_COUNT)
      {
        (*counts)[label] += 1;
      }
    }
    return points;
  }

  std::vector<OutputPoint> collectRebuiltMapPoints(std::array<int, LABEL_COUNT>* counts = nullptr) const
  {
    if (counts)
    {
      counts->fill(0);
    }

    std::size_t total = 0;
    for (const auto& kf : rebuilt_keyframes_)
    {
      total += kf.points.size();
    }
    std::vector<OutputPoint> points;
    if (total == 0)
    {
      return points;
    }
    points.reserve(std::min<std::size_t>(total, static_cast<std::size_t>(std::max(1, max_publish_points_))));

    int stride = 1;
    if (max_publish_points_ > 0 && total > static_cast<std::size_t>(max_publish_points_))
    {
      stride = static_cast<int>(std::ceil(static_cast<double>(total) / static_cast<double>(max_publish_points_)));
    }

    int index = 0;
    for (const auto& kf : rebuilt_keyframes_)
    {
      for (const auto& local_pt : kf.points)
      {
        ++index;
        if (stride > 1 && (index % stride) != 0)
        {
          continue;
        }
        const Eigen::Isometry3d corrected_pose = groundRobotCorrectedPose(kf.raw_pose);
        const Eigen::Vector3d p = corrected_pose * local_pt.local;
        if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()) ||
            p.z() < z_min_map_ || p.z() > z_max_map_)
        {
          continue;
        }
        OutputPoint out;
        out.x = static_cast<float>(p.x());
        out.y = static_cast<float>(p.y());
        out.z = static_cast<float>(p.z());
        out.label = local_pt.label;
        out.confidence = local_pt.confidence;
        out.votes = local_pt.weight;
        if (out.label == LABEL_UNKNOWN || out.confidence < min_confidence_)
        {
          continue;
        }
        points.push_back(out);
        if (counts && out.label < LABEL_COUNT)
        {
          (*counts)[out.label] += 1;
        }
      }
    }
    return points;
  }

  void publishPointCloud(const std::vector<OutputPoint>& points,
                         const std::string& topic,
                         ros::Publisher& publisher,
                         const ros::Time& stamp) const
  {
    if (!publisher)
    {
      return;
    }
    sensor_msgs::PointCloud2 msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;
    msg.height = 1;
    msg.width = static_cast<uint32_t>(points.size());
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2Fields(
      6,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::PointField::FLOAT32,
      "label", 1, sensor_msgs::PointField::UINT32,
      "confidence", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(msg, "rgb");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_label(msg, "label");
    sensor_msgs::PointCloud2Iterator<float> iter_conf(msg, "confidence");
    for (const auto& pt : points)
    {
      *iter_x = pt.x;
      *iter_y = pt.y;
      *iter_z = pt.z;
      *iter_rgb = labelRgbFloat(pt.label);
      *iter_label = pt.label;
      *iter_conf = pt.confidence;
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_rgb;
      ++iter_label;
      ++iter_conf;
    }
    publisher.publish(msg);
    (void)topic;
  }

  void publishBev(const std::vector<OutputPoint>& points, const ros::Time& stamp)
  {
    if (!publish_bev_)
    {
      return;
    }
    const int size_px = std::max(32, static_cast<int>(std::round(bev_size_m_ / std::max(0.02, bev_resolution_))));
    const double half = bev_size_m_ * 0.5;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    if (bev_center_on_latest_pose_)
    {
      center = latest_T_map_body_.translation().head<2>();
    }

    cv::Mat label_img(size_px, size_px, CV_8UC1, cv::Scalar(0));
    cv::Mat conf_img(size_px, size_px, CV_8UC1, cv::Scalar(0));
    cv::Mat color_img(size_px, size_px, CV_8UC3, cv::Scalar(180, 180, 180));
    cv::Mat traversable_img(size_px, size_px, CV_8UC1, cv::Scalar(0));

    for (const auto& pt : points)
    {
      const double dx = static_cast<double>(pt.x) - center.x();
      const double dy = static_cast<double>(pt.y) - center.y();
      if (dx < -half || dx >= half || dy < -half || dy >= half)
      {
        continue;
      }
      const int ix = static_cast<int>(std::floor((dx + half) / bev_resolution_));
      const int iy = static_cast<int>(std::floor((half - dy) / bev_resolution_));
      if (ix < 0 || ix >= size_px || iy < 0 || iy >= size_px)
      {
        continue;
      }
      const uint8_t conf = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, pt.confidence)) * 255.0f);
      if (conf < conf_img.at<uint8_t>(iy, ix))
      {
        continue;
      }
      label_img.at<uint8_t>(iy, ix) = static_cast<uint8_t>(pt.label);
      conf_img.at<uint8_t>(iy, ix) = conf;
      color_img.at<cv::Vec3b>(iy, ix) = labelColorBgr(pt.label);
      if (pt.label == LABEL_ROAD || pt.label == LABEL_SIDEWALK)
      {
        traversable_img.at<uint8_t>(iy, ix) = 255;
      }
    }

    std_msgs::Header header;
    header.stamp = stamp;
    header.frame_id = map_frame_;
    bev_label_pub_.publish(cv_bridge::CvImage(header, "mono8", label_img).toImageMsg());
    bev_color_pub_.publish(cv_bridge::CvImage(header, "bgr8", color_img).toImageMsg());
    bev_conf_pub_.publish(cv_bridge::CvImage(header, "mono8", conf_img).toImageMsg());
    bev_traversable_pub_.publish(cv_bridge::CvImage(header, "mono8", traversable_img).toImageMsg());
  }

  void publishStats(const std::array<int, LABEL_COUNT>& counts, const std::size_t published_points)
  {
    std::ostringstream ss;
    ss << "{";
    ss << "\"voxels\":" << voxels_.size() << ",";
    ss << "\"use_keyframe_rebuilt_map\":" << (use_keyframe_rebuilt_map_ ? "true" : "false") << ",";
    ss << "\"rebuilt_keyframes\":" << rebuilt_keyframes_.size() << ",";
    ss << "\"published_points\":" << published_points << ",";
    ss << "\"last_lidar_points\":" << last_lidar_points_ << ",";
    ss << "\"last_image_points\":" << last_image_points_ << ",";
    ss << "\"voxel_size\":" << voxel_size_ << ",";
    ss << "\"min_votes\":" << min_votes_ << ",";
    ss << "\"min_confidence\":" << min_confidence_ << ",";
    ss << "\"x_correction\":" << semantic_xy_correction_.x() << ",";
    ss << "\"y_correction\":" << semantic_xy_correction_.y() << ",";
    ss << "\"yaw_correction_deg\":" << semantic_yaw_correction_ * 180.0 / M_PI << ",";
    ss << "\"xy_matches\":" << last_xy_matches_ << ",";
    ss << "\"xy_last_residual_x\":" << last_xy_residual_.x() << ",";
    ss << "\"xy_last_residual_y\":" << last_xy_residual_.y() << ",";
    ss << "\"yaw_last_residual_deg\":" << last_yaw_residual_ * 180.0 / M_PI << ",";
    ss << "\"yaw_last_step_deg\":" << last_yaw_step_ * 180.0 / M_PI << ",";
    ss << "\"se2_residual_rms\":" << last_se2_residual_rms_ << ",";
    ss << "\"se2_spread\":" << last_se2_spread_ << ",";
    ss << "\"pose_constraint_accepted\":" << (last_pose_constraint_accepted_ ? "true" : "false") << ",";
    ss << "\"semantic_correction_trust\":" << semanticCorrectionTrust() << ",";
    ss << "\"z_constraint_stale_frames\":" << z_constraint_stale_frames_ << ",";
    ss << "\"semantic_z_correction_trust\":" << semanticZCorrectionTrust() << ",";
    ss << "\"z_correction\":" << semantic_z_correction_ << ",";
    ss << "\"z_slope_x\":" << semantic_z_slope_.x() << ",";
    ss << "\"z_slope_y\":" << semantic_z_slope_.y() << ",";
    ss << "\"z_matches\":" << last_z_matches_ << ",";
    ss << "\"z_last_residual\":" << last_z_residual_ << ",";
    ss << "\"z_last_mad\":" << last_z_mad_ << ",";
    ss << "\"ground_anchor_update_skipped\":" << last_ground_anchor_update_skipped_ << ",";
    ss << "\"ground_geometry_rejects\":" << last_ground_geometry_rejects_ << ",";
    ss << "\"z_accept_streak\":" << z_constraint_accept_streak_ << ",";
    ss << "\"z_candidate_residual\":" << z_candidate_residual_ << ",";
    ss << "\"local_semantic_frames\":" << last_local_semantic_frames_ << ",";
    ss << "\"local_semantic_points\":" << last_local_semantic_points_ << ",";
    ss << "\"local_semantic_constraints\":" << local_semantic_constraints_ << ",";
    ss << "\"z_slope_samples\":" << last_z_slope_samples_ << ",";
    ss << "\"z_slope_spread\":" << last_z_slope_spread_ << ",";
    ss << "\"ground_anchors\":" << ground_anchors_.size() << ",";
    ss << "\"object_anchors\":" << object_anchors_.size() << ",";
    ss << "\"z_object_anchors\":" << z_object_anchors_.size() << ",";
    ss << "\"counts\":{";
    ss << "\"road\":" << counts[LABEL_ROAD] << ",";
    ss << "\"sidewalk\":" << counts[LABEL_SIDEWALK] << ",";
    ss << "\"building\":" << counts[LABEL_BUILDING] << ",";
    ss << "\"vegetation\":" << counts[LABEL_VEGETATION] << ",";
    ss << "\"dynamic\":" << counts[LABEL_DYNAMIC] << ",";
    ss << "\"other\":" << counts[LABEL_OTHER];
    ss << "}}";

    std_msgs::String msg;
    msg.data = ss.str();
    stats_pub_.publish(msg);
  }

  void publishTimerCb(const ros::TimerEvent&)
  {
    std::array<int, LABEL_COUNT> counts;
    const std::vector<OutputPoint> points = collectMapPoints(&counts);
    const ros::Time stamp = ros::Time::now();
    publishPointCloud(points, semantic_cloud_topic_, semantic_cloud_pub_, stamp);
    publishBev(points, stamp);
    publishStats(counts, points.size());
    ROS_INFO_THROTTLE(5.0,
                      "semantic voxel map published points=%zu voxels=%zu road=%d sidewalk=%d building=%d vegetation=%d dynamic=%d",
                      points.size(), voxels_.size(), counts[LABEL_ROAD], counts[LABEL_SIDEWALK],
                      counts[LABEL_BUILDING], counts[LABEL_VEGETATION], counts[LABEL_DYNAMIC]);
  }

  void maintenanceTimerCb(const ros::TimerEvent&)
  {
    const ros::Time now = ros::Time::now();
    if (voxel_ttl_sec_ > 0.0)
    {
      for (auto it = voxels_.begin(); it != voxels_.end();)
      {
        if (!it->second.last_update.isZero() && (now - it->second.last_update).toSec() > voxel_ttl_sec_)
        {
          it = voxels_.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }

    if (max_voxels_ > 0 && static_cast<int>(voxels_.size()) > max_voxels_)
    {
      const std::size_t target = static_cast<std::size_t>(static_cast<double>(max_voxels_) * 0.90);
      for (auto it = voxels_.begin(); it != voxels_.end() && voxels_.size() > target;)
      {
        it = voxels_.erase(it);
      }
      ROS_WARN("semantic voxel map exceeded max_voxels; pruned to %zu", voxels_.size());
    }
  }

  void diagnosticTimerCb(const ros::WallTimerEvent&)
  {
    const ros::WallTime now = ros::WallTime::now();
    if (subscribe_lidar_cloud_ && lidar_msg_count_ == 0)
    {
      ROS_WARN("semantic_voxel_mapper has not received RangeNet++ semantic cloud on %s. "
               "Expected PointCloud2 fields: x y z %s [confidence] [instance_id]. "
               "For real RangeNet++ use rangenetpp_semantic_slam.launch, or start "
               "livox_custom_to_pointcloud2_node + rangenetpp_ros_node before this mapper.",
               lidar_cloud_topic_.c_str(), label_field_name_.c_str());
    }
    else if (subscribe_lidar_cloud_ && (now - last_lidar_input_wall_).toSec() > 3.0)
    {
      ROS_WARN("semantic_voxel_mapper RangeNet++ input stalled: topic=%s last_age=%.1fs",
               lidar_cloud_topic_.c_str(), (now - last_lidar_input_wall_).toSec());
    }

    if (subscribe_image_cloud_)
    {
      if (image_msg_count_ == 0)
      {
        ROS_WARN("semantic_voxel_mapper has not received SegFormer projected cloud on %s",
                 image_cloud_topic_.c_str());
      }
      else if ((now - last_image_input_wall_).toSec() > 3.0)
      {
        ROS_WARN("semantic_voxel_mapper SegFormer projected input stalled: topic=%s last_age=%.1fs",
                 image_cloud_topic_.c_str(), (now - last_image_input_wall_).toSec());
      }
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber lidar_cloud_sub_;
  ros::Subscriber image_cloud_sub_;
  ros::Publisher semantic_cloud_pub_;
  ros::Publisher static_cloud_pub_;
  ros::Publisher stats_pub_;
  ros::Publisher corrected_odom_pub_;
  ros::Publisher corrected_path_pub_;
  ros::Publisher z_constraint_pub_;
  ros::Publisher z_constraint_debug_pub_;
  ros::Publisher xy_constraint_debug_pub_;
  ros::Publisher z_constraint_points_pub_;
  ros::Publisher xy_constraint_points_pub_;
  ros::Publisher bev_label_pub_;
  ros::Publisher bev_color_pub_;
  ros::Publisher bev_conf_pub_;
  ros::Publisher bev_traversable_pub_;
  ros::Timer publish_timer_;
  ros::Timer maintenance_timer_;
  ros::WallTimer diagnostic_timer_;

  std::unordered_map<VoxelKey, VoxelState, VoxelKeyHash> voxels_;
  std::unordered_map<VoxelKey, SemanticAnchor, VoxelKeyHash> ground_anchors_;
  std::unordered_map<VoxelKey, SemanticAnchor, VoxelKeyHash> object_anchors_;
  std::unordered_map<VoxelKey, SemanticAnchor, VoxelKeyHash> z_object_anchors_;
  Eigen::Isometry3d latest_T_map_body_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_body_lidar_ = Eigen::Isometry3d::Identity();
  ros::Time latest_odom_time_;
  nav_msgs::Odometry latest_odom_msg_;
  bool has_latest_odom_msg_ = false;

  std::string odom_topic_;
  std::string lidar_cloud_topic_;
  std::string image_cloud_topic_;
  std::string semantic_cloud_topic_;
  std::string stats_topic_;
  std::string static_labeled_cloud_topic_;
  std::string corrected_odom_topic_;
  std::string corrected_path_topic_;
  std::string z_constraint_topic_;
  std::string z_constraint_debug_topic_;
  std::string xy_constraint_debug_topic_;
  std::string z_constraint_points_topic_;
  std::string xy_constraint_points_topic_;
  std::string bev_label_topic_;
  std::string bev_color_topic_;
  std::string bev_confidence_topic_;
  std::string bev_traversable_topic_;
  std::string lidar_label_mode_;
  std::string image_label_mode_;
  std::string map_frame_;
  std::string label_field_name_;

  bool subscribe_lidar_cloud_ = true;
  bool subscribe_image_cloud_ = false;
  bool lidar_cloud_in_map_frame_ = false;
  bool image_cloud_in_map_frame_ = false;
  bool fuse_dynamic_objects_ = false;
  bool enable_instance_dynamic_filter_ = true;
  bool publish_static_labeled_cloud_ = true;
  bool publish_bev_ = true;
  bool bev_center_on_latest_pose_ = true;
  bool enable_semantic_z_correction_ = true;
  bool enable_semantic_z_slope_correction_ = true;
  bool enable_ground_geometry_filter_ = true;
  bool enable_ground_robot_z_prior_ = true;
  bool use_keyframe_rebuilt_map_ = true;
  bool enable_ground_robot_roll_pitch_prior_ = true;
  bool rebuilt_keyframe_lidar_only_ = true;
  bool publish_constraint_debug_ = true;
  bool enable_semantic_xy_correction_ = true;
  bool enable_semantic_yaw_correction_ = true;
  bool use_vegetation_xy_anchors_ = false;
  bool use_object_z_anchors_ = false;
  bool enable_local_semantic_map_constraint_ = true;
  bool local_semantic_lidar_only_ = true;
  bool has_z_correction_origin_ = false;
  bool has_ground_robot_z_prior_ = false;
  bool has_last_rebuilt_keyframe_ = false;
  bool last_pose_constraint_accepted_ = false;
  bool has_z_candidate_residual_ = false;

  double voxel_size_ = 0.20;
  double min_range_m_ = 0.5;
  double max_range_m_ = 120.0;
  double z_min_map_ = -5.0;
  double z_max_map_ = 25.0;
  double lidar_source_weight_ = 1.0;
  double image_source_weight_ = 0.65;
  double default_confidence_ = 0.75;
  double min_votes_ = 2.0;
  double min_confidence_ = 0.55;
  double confidence_vote_scale_ = 4.0;
  double instance_cluster_resolution_m_ = 0.55;
  double instance_dynamic_label_ratio_ = 0.15;
  double instance_dynamic_min_height_m_ = 0.35;
  double instance_dynamic_max_height_m_ = 2.80;
  double instance_dynamic_max_length_m_ = 6.50;
  double instance_dynamic_max_width_m_ = 3.20;
  Eigen::Vector2d semantic_xy_correction_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_xy_residual_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_xy_step_ = Eigen::Vector2d::Zero();
  double semantic_yaw_correction_ = 0.0;
  double last_yaw_residual_ = 0.0;
  double last_yaw_step_ = 0.0;
  double last_se2_residual_rms_ = 0.0;
  double last_se2_spread_ = 0.0;
  Eigen::Vector2d semantic_z_slope_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d z_correction_origin_xy_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_z_slope_fit_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_z_slope_step_ = Eigen::Vector2d::Zero();
  double semantic_z_correction_ = 0.0;
  double ground_robot_z_correction_ = 0.0;
  double ground_robot_reference_z_ = 0.0;
  double ground_robot_prior_z_ = 0.0;
  double ground_robot_last_raw_z_ = 0.0;
  double last_ground_robot_z_residual_ = 0.0;
  double last_ground_robot_z_step_ = 0.0;
  double z_anchor_resolution_ = 1.0;
  double object_anchor_resolution_ = 2.5;
  double xy_correction_alpha_ = 0.05;
  double max_xy_correction_step_ = 0.03;
  double max_abs_xy_correction_ = 5.0;
  double max_xy_match_residual_ = 3.0;
  double yaw_correction_alpha_ = 0.05;
  double max_yaw_correction_step_deg_ = 0.15;
  double max_abs_yaw_correction_deg_ = 8.0;
  double min_se2_spread_m_ = 8.0;
  double max_se2_residual_rms_ = 1.5;
  double pose_constraint_interval_sec_ = 1.0;
  double pose_xy_variance_ = 1.0;
  double pose_z_variance_ = 1.0;
  double pose_yaw_variance_ = 0.05;
  double min_pose_xy_variance_ = 0.01;
  double min_pose_z_variance_ = 0.01;
  double min_pose_yaw_variance_ = 1e-4;
  double max_pose_xy_variance_ = 25.0;
  double max_pose_z_variance_ = 9.0;
  double max_pose_yaw_variance_ = 0.5;
  double min_semantic_correction_trust_ = 0.35;
  double min_semantic_z_correction_trust_ = 0.05;
  double z_correction_alpha_ = 0.08;
  double max_z_correction_step_ = 0.04;
  double max_abs_z_correction_ = 3.0;
  double max_z_residual_abs_ = 0.8;
  double max_z_residual_mad_ = 0.35;
  double max_z_residual_jump_ = 0.25;
  double z_correction_deadband_ = 0.03;
  double z_candidate_residual_ = 0.0;
  double ground_filter_resolution_m_ = 0.8;
  double ground_filter_frame_percentile_ = 0.10;
  int ground_filter_min_cell_points_ = 4;
  int ground_filter_neighbor_radius_cells_ = 1;
  double max_ground_height_above_cell_min_ = 0.35;
  double max_ground_height_above_frame_ground_ = 1.60;
  double max_ground_anchor_match_xy_m_ = 1.25;
  double min_ground_anchor_age_sec_ = 6.0;
  double max_ground_anchor_update_z_residual_ = 0.35;
  double corrected_path_publish_rate_ = 2.0;
  double ground_robot_z_prior_alpha_ = 0.12;
  double ground_robot_z_prior_max_step_ = 0.035;
  double ground_robot_z_prior_deadband_ = 0.05;
  double ground_robot_z_prior_max_grade_ = 0.01;
  double ground_robot_z_prior_vertical_margin_ = 0.01;
  double ground_robot_z_prior_max_abs_correction_ = 8.0;
  double ground_robot_z_prior_max_drift_ = 1.2;
  double ground_robot_roll_pitch_scale_ = 0.0;
  double rebuilt_keyframe_min_distance_ = 1.0;
  double rebuilt_keyframe_min_interval_sec_ = 0.5;
  double z_slope_correction_alpha_ = 0.08;
  double max_z_slope_correction_step_ = 0.002;
  double max_abs_z_slope_ = 0.05;
  double min_z_slope_spread_m_ = 8.0;
  double last_z_slope_spread_ = 0.0;
  double anchor_update_alpha_ = 0.06;
  double object_anchor_weight_ = 0.35;
  double last_z_residual_ = 0.0;
  double last_z_mad_ = 0.0;
  double last_z_step_ = 0.0;
  double voxel_ttl_sec_ = 0.0;
  double publish_rate_ = 2.0;
  double bev_resolution_ = 0.20;
  double bev_size_m_ = 100.0;
  double local_semantic_window_sec_ = 3.0;
  Eigen::Vector2d ground_robot_last_xy_ = Eigen::Vector2d::Zero();
  Eigen::Vector3d last_rebuilt_keyframe_position_ = Eigen::Vector3d::Zero();
  int max_publish_points_ = 800000;
  int max_voxels_ = 1800000;
  int instance_min_points_ = 8;
  int last_lidar_points_ = 0;
  int last_image_points_ = 0;
  int min_z_constraint_matches_ = 80;
  int min_ground_anchor_observations_ = 3;
  int min_object_anchor_observations_ = 3;
  int ground_anchor_search_radius_cells_ = 1;
  int min_z_constraint_accept_streak_ = 3;
  int max_z_constraint_samples_ = 2000;
  int min_z_slope_constraint_matches_ = 120;
  int last_z_slope_samples_ = 0;
  int min_xy_constraint_matches_ = 12;
  int max_xy_constraint_samples_ = 500;
  int xy_anchor_search_radius_cells_ = 1;
  int min_se2_constraint_matches_ = 30;
  int max_pose_constraint_stale_frames_ = 30;
  int max_z_constraint_stale_frames_ = 20;
  int rebuilt_keyframe_max_frames_ = 700;
  int rebuilt_keyframe_max_points_per_frame_ = 2500;
  int constraint_debug_max_pairs_ = 300;
  int local_semantic_min_frames_ = 20;
  int local_semantic_max_frames_ = 80;
  int local_semantic_max_points_per_frame_ = 1200;
  int corrected_path_max_poses_ = 12000;
  int pose_constraint_stale_frames_ = 0;
  int z_constraint_stale_frames_ = 0;
  int last_xy_matches_ = 0;
  int last_z_matches_ = 0;
  int last_ground_anchor_update_skipped_ = 0;
  int last_ground_geometry_rejects_ = 0;
  int z_constraint_accept_streak_ = 0;
  int last_local_semantic_frames_ = 0;
  int last_local_semantic_points_ = 0;
  int local_semantic_constraints_ = 0;
  int lidar_msg_count_ = 0;
  int image_msg_count_ = 0;
  ros::WallTime last_lidar_input_wall_;
  ros::WallTime last_image_input_wall_;
  ros::WallTime last_corrected_path_pub_wall_;
  ros::Time last_pose_constraint_time_;
  ros::Time last_rebuilt_keyframe_time_;
  std::deque<LocalSemanticFrame> local_semantic_frames_;
  std::deque<RebuiltSemanticKeyframe> rebuilt_keyframes_;
  std::deque<nav_msgs::Odometry> raw_odom_history_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "semantic_voxel_mapper");
  SemanticVoxelMapper node;
  ros::spin();
  return 0;
}
