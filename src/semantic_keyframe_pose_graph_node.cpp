#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

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

struct Pose4
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double yaw = 0.0;
};

struct SemanticPoint
{
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  uint32_t label = LABEL_UNKNOWN;
  float confidence = 1.0f;
};

struct Keyframe
{
  int id = 0;
  ros::Time stamp;
  Pose4 raw;
  Pose4 opt;
  std::vector<SemanticPoint> local_points;
};

struct Edge
{
  int i = 0;
  int j = 0;
  Pose4 relative;
  double weight_xy = 1.0;
  double weight_z = 1.0;
  double weight_yaw = 1.0;
  bool loop = false;
  int inliers = 0;
  double rms = 0.0;
};

struct CellKey
{
  int x = 0;
  int y = 0;
  uint32_t label = 0;

  bool operator==(const CellKey& rhs) const
  {
    return x == rhs.x && y == rhs.y && label == rhs.label;
  }
};

struct CellKeyHash
{
  std::size_t operator()(const CellKey& k) const
  {
    const std::size_t hx = std::hash<int>()(k.x);
    const std::size_t hy = std::hash<int>()(k.y);
    const std::size_t hl = std::hash<uint32_t>()(k.label);
    return hx ^ (hy + 0x9e3779b97f4a7c15ULL + (hx << 6) + (hx >> 2)) ^
           (hl + 0x9e3779b97f4a7c15ULL + (hy << 6) + (hy >> 2));
  }
};

struct CellStats
{
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  double weight = 0.0;
  int count = 0;
  uint32_t label = LABEL_UNKNOWN;
  double min_z = std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();

  Eigen::Vector3d mean() const
  {
    if (weight > 1e-9)
    {
      return sum / weight;
    }
    return Eigen::Vector3d::Zero();
  }
};

struct MatchResult
{
  bool accepted = false;
  bool xy_accepted = false;
  bool z_accepted = false;
  Pose4 relative;
  int inliers = 0;
  int ground_inliers = 0;
  int building_inliers = 0;
  double rms = 0.0;
  double spread = 0.0;
  double spread_ratio = 0.0;
  double z_median = 0.0;
  double z_mad = 0.0;
  double yaw_delta = 0.0;
  double xy_delta = 0.0;
  double score = -std::numeric_limits<double>::infinity();
  double score_ratio = 0.0;
  double score_gap = 0.0;
};

struct CandidateScore
{
  int inliers = 0;
  int ground_inliers = 0;
  double rms = std::numeric_limits<double>::infinity();
  double score = -std::numeric_limits<double>::infinity();
};

struct PoseDelta
{
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  double dyaw = 0.0;
  double wxy = 0.0;
  double wz = 0.0;
  double wyaw = 0.0;
};

double normalizeAngle(double a)
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

double clampValue(double v, double lo, double hi)
{
  return std::max(lo, std::min(hi, v));
}

double yawFromQuat(const geometry_msgs::Quaternion& qmsg)
{
  tf2::Quaternion q;
  tf2::fromMsg(qmsg, q);
  if (q.length2() < 1e-12)
  {
    return 0.0;
  }
  q.normalize();
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return yaw;
}

geometry_msgs::Quaternion quatFromYaw(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  q.normalize();
  return tf2::toMsg(q);
}

geometry_msgs::Quaternion quatWithRawRollPitchAndYaw(const geometry_msgs::Quaternion& raw_msg,
                                                     double yaw)
{
  tf2::Quaternion raw_q;
  tf2::fromMsg(raw_msg, raw_q);
  if (raw_q.length2() < 1e-12)
  {
    return quatFromYaw(yaw);
  }
  raw_q.normalize();
  double roll = 0.0;
  double pitch = 0.0;
  double raw_yaw = 0.0;
  tf2::Matrix3x3(raw_q).getRPY(roll, pitch, raw_yaw);
  (void)raw_yaw;

  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  q.normalize();
  return tf2::toMsg(q);
}

Eigen::Vector2d rotate2(double yaw, const Eigen::Vector2d& p)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return Eigen::Vector2d(c * p.x() - s * p.y(), s * p.x() + c * p.y());
}

Pose4 odomToPose4(const nav_msgs::Odometry& odom)
{
  Pose4 p;
  p.x = odom.pose.pose.position.x;
  p.y = odom.pose.pose.position.y;
  p.z = odom.pose.pose.position.z;
  p.yaw = yawFromQuat(odom.pose.pose.orientation);
  return p;
}

Eigen::Vector3d transformLocalToMap(const Pose4& pose, const Eigen::Vector3d& p)
{
  const Eigen::Vector2d xy = rotate2(pose.yaw, p.head<2>()) + Eigen::Vector2d(pose.x, pose.y);
  return Eigen::Vector3d(xy.x(), xy.y(), p.z() + pose.z);
}

Eigen::Vector3d transformMapToLocal(const Pose4& pose, const Eigen::Vector3d& p)
{
  const Eigen::Vector2d d(p.x() - pose.x, p.y() - pose.y);
  const Eigen::Vector2d xy = rotate2(-pose.yaw, d);
  return Eigen::Vector3d(xy.x(), xy.y(), p.z() - pose.z);
}

Pose4 relativePose(const Pose4& a, const Pose4& b)
{
  Pose4 r;
  const Eigen::Vector2d d(b.x - a.x, b.y - a.y);
  const Eigen::Vector2d local = rotate2(-a.yaw, d);
  r.x = local.x();
  r.y = local.y();
  r.z = b.z - a.z;
  r.yaw = normalizeAngle(b.yaw - a.yaw);
  return r;
}

Pose4 composePose(const Pose4& a, const Pose4& rel)
{
  Pose4 out;
  const Eigen::Vector2d xy = rotate2(a.yaw, Eigen::Vector2d(rel.x, rel.y)) + Eigen::Vector2d(a.x, a.y);
  out.x = xy.x();
  out.y = xy.y();
  out.z = a.z + rel.z;
  out.yaw = normalizeAngle(a.yaw + rel.yaw);
  return out;
}

double poseDistance2D(const Pose4& a, const Pose4& b)
{
  return std::hypot(a.x - b.x, a.y - b.y);
}

bool isGroundLabel(uint32_t label)
{
  return label == LABEL_ROAD || label == LABEL_SIDEWALK;
}

bool isGraphLabel(uint32_t label)
{
  return label == LABEL_ROAD || label == LABEL_SIDEWALK ||
         label == LABEL_BUILDING || label == LABEL_OTHER;
}

bool isXYMatchLabel(uint32_t label)
{
  return label == LABEL_ROAD || label == LABEL_SIDEWALK ||
         label == LABEL_BUILDING || label == LABEL_OTHER;
}

bool isStructuralXYLabel(uint32_t label)
{
  return label == LABEL_BUILDING || label == LABEL_OTHER;
}

bool isZMatchLabel(uint32_t label)
{
  return label == LABEL_ROAD || label == LABEL_SIDEWALK;
}

uint32_t labelColor(uint32_t label)
{
  switch (label)
  {
    case LABEL_ROAD:
      return (80u << 16) | (80u << 8) | 80u;
    case LABEL_SIDEWALK:
      return (80u << 16) | (180u << 8) | 80u;
    case LABEL_BUILDING:
      return (230u << 16) | (40u << 8) | 40u;
    case LABEL_VEGETATION:
      return (30u << 16) | (180u << 8) | 30u;
    case LABEL_DYNAMIC:
      return (40u << 16) | (80u << 8) | 230u;
    case LABEL_OTHER:
      return (180u << 16) | (80u << 8) | 200u;
    default:
      return (120u << 16) | (120u << 8) | 120u;
  }
}

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

const sensor_msgs::PointField*
findFirstField(const sensor_msgs::PointCloud2& msg, const std::vector<std::string>& names)
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
                         std::size_t index)
{
  const std::size_t offset = index * msg.point_step + field.offset;
  if (offset + field.count > msg.data.size())
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const uint8_t* ptr = &msg.data[offset];
  switch (field.datatype)
  {
    case sensor_msgs::PointField::FLOAT32:
    {
      float v = 0.0f;
      std::memcpy(&v, ptr, sizeof(float));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::FLOAT64:
    {
      double v = 0.0;
      std::memcpy(&v, ptr, sizeof(double));
      return v;
    }
    case sensor_msgs::PointField::UINT32:
    {
      uint32_t v = 0;
      std::memcpy(&v, ptr, sizeof(uint32_t));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::INT32:
    {
      int32_t v = 0;
      std::memcpy(&v, ptr, sizeof(int32_t));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT16:
    {
      uint16_t v = 0;
      std::memcpy(&v, ptr, sizeof(uint16_t));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::INT16:
    {
      int16_t v = 0;
      std::memcpy(&v, ptr, sizeof(int16_t));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT8:
      return static_cast<double>(*ptr);
    case sensor_msgs::PointField::INT8:
      return static_cast<double>(*reinterpret_cast<const int8_t*>(ptr));
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

double robustMedian(std::vector<double> values)
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

double robustMad(std::vector<double> values, double center)
{
  for (double& v : values)
  {
    v = std::fabs(v - center);
  }
  return robustMedian(values);
}

}  // namespace

class SemanticKeyframePoseGraph
{
 public:
  SemanticKeyframePoseGraph() : nh_(), pnh_("~")
  {
    pnh_.param<std::string>("odom_topic", odom_topic_, "/aft_mapped_to_init");
    pnh_.param<std::string>("semantic_cloud_topic", cloud_topic_, "/rangenet/semantic_points");
    pnh_.param<std::string>("output_odom_topic", output_odom_topic_, "/semantic_graph_corrected_odom");
    pnh_.param<std::string>("path_topic", path_topic_, "/semantic_pose_graph/path");
    pnh_.param<std::string>("map_topic", map_topic_, "/semantic_pose_graph/map");
    pnh_.param<std::string>("stats_topic", stats_topic_, "/semantic_pose_graph/stats");
    pnh_.param<std::string>("map_frame", map_frame_, "camera_init");
    pnh_.param<std::string>("label_field", label_field_, "label");
    pnh_.param<std::string>("label_mode", label_mode_, "internal");
    pnh_.param<std::string>("output_mode", output_mode_, "z_only");
    pnh_.param<std::string>("output_orientation_mode", output_orientation_mode_, "preserve_roll_pitch");

    pnh_.param<bool>("cloud_in_map_frame", cloud_in_map_frame_, true);
    pnh_.param<bool>("enable_loop_constraints", enable_loop_constraints_, true);
    pnh_.param<bool>("enable_xy_loop_constraints", enable_xy_loop_constraints_, false);
    pnh_.param<bool>("enable_z_loop_constraints", enable_z_loop_constraints_, true);
    pnh_.param<bool>("publish_optimized_map", publish_optimized_map_, true);
    pnh_.param<bool>("allow_unsafe_full_output", allow_unsafe_full_output_, false);

    pnh_.param<double>("keyframe_min_distance", keyframe_min_distance_, 2.0);
    pnh_.param<double>("keyframe_min_yaw_deg", keyframe_min_yaw_deg_, 8.0);
    pnh_.param<double>("keyframe_min_interval_sec", keyframe_min_interval_sec_, 0.5);
    pnh_.param<int>("max_keyframes", max_keyframes_, 2000);
    pnh_.param<int>("max_points_per_keyframe", max_points_per_keyframe_, 1800);
    pnh_.param<double>("min_point_range", min_point_range_, 0.5);
    pnh_.param<double>("max_point_range", max_point_range_, 120.0);

    pnh_.param<int>("loop_min_index_gap", loop_min_index_gap_, 30);
    pnh_.param<double>("loop_search_radius", loop_search_radius_, 18.0);
    pnh_.param<int>("loop_max_candidates", loop_max_candidates_, 8);
    pnh_.param<int>("loop_min_inliers", loop_min_inliers_, 80);
    pnh_.param<int>("loop_min_ground_inliers", loop_min_ground_inliers_, 30);
    pnh_.param<double>("loop_max_rms", loop_max_rms_, 0.85);
    pnh_.param<double>("loop_min_spread", loop_min_spread_, 6.0);
    pnh_.param<double>("loop_max_xy_correction", loop_max_xy_correction_, 4.0);
    pnh_.param<double>("loop_max_yaw_correction_deg", loop_max_yaw_correction_deg_, 6.0);
    pnh_.param<double>("loop_max_z_median", loop_max_z_median_, 1.5);
    pnh_.param<double>("loop_max_z_mad", loop_max_z_mad_, 0.35);
    pnh_.param<int>("loop_min_building_inliers", loop_min_building_inliers_, 50);
    pnh_.param<double>("loop_min_spread_ratio", loop_min_spread_ratio_, 0.18);
    pnh_.param<double>("loop_min_score_ratio", loop_min_score_ratio_, 1.15);
    pnh_.param<double>("loop_min_score_gap", loop_min_score_gap_, 40.0);
    pnh_.param<double>("loop_max_xy_correction_for_xy", loop_max_xy_correction_for_xy_, 3.0);
    pnh_.param<double>("loop_max_yaw_correction_for_xy_deg", loop_max_yaw_correction_for_xy_deg_, 5.0);
    pnh_.param<double>("loop_max_xy_correction_for_z", loop_max_xy_correction_for_z_, 6.0);
    pnh_.param<double>("icp_grid_resolution", icp_grid_resolution_, 0.8);
    pnh_.param<double>("icp_max_correspondence", icp_max_correspondence_, 1.2);
    pnh_.param<int>("feature_min_cell_points", feature_min_cell_points_, 3);
    pnh_.param<int>("ground_min_cell_points", ground_min_cell_points_, 4);
    pnh_.param<double>("max_ground_cell_z_span", max_ground_cell_z_span_, 0.35);
    pnh_.param<double>("ground_cell_local_z_min", ground_cell_local_z_min_, -4.0);
    pnh_.param<double>("ground_cell_local_z_max", ground_cell_local_z_max_, 1.2);
    pnh_.param<int>("icp_iterations", icp_iterations_, 5);
    pnh_.param<bool>("enable_coarse_loop_search", enable_coarse_loop_search_, true);
    pnh_.param<double>("coarse_xy_search_radius", coarse_xy_search_radius_, 10.0);
    pnh_.param<double>("coarse_xy_search_step", coarse_xy_search_step_, 2.0);
    pnh_.param<double>("coarse_yaw_search_deg", coarse_yaw_search_deg_, 8.0);
    pnh_.param<double>("coarse_yaw_search_step_deg", coarse_yaw_search_step_deg_, 2.0);
    pnh_.param<int>("coarse_sample_points", coarse_sample_points_, 700);
    pnh_.param<int>("coarse_min_inliers", coarse_min_inliers_, 60);

    pnh_.param<double>("odom_edge_weight", odom_edge_weight_, 1.0);
    pnh_.param<double>("loop_edge_weight", loop_edge_weight_, 0.35);
    pnh_.param<double>("loop_edge_min_weight", loop_edge_min_weight_, 0.25);
    pnh_.param<double>("loop_edge_max_weight", loop_edge_max_weight_, 2.5);
    pnh_.param<double>("loop_weight_inlier_reference", loop_weight_inlier_reference_, 500.0);
    pnh_.param<double>("loop_weight_rms_reference", loop_weight_rms_reference_, 0.55);
    pnh_.param<double>("loop_weight_z_mad_reference", loop_weight_z_mad_reference_, 0.20);
    pnh_.param<double>("loop_z_weight_scale", loop_z_weight_scale_, 1.0);
    pnh_.param<double>("loop_yaw_weight_scale", loop_yaw_weight_scale_, 1.3);
    pnh_.param<double>("loop_robust_kernel", loop_robust_kernel_, 4.0);
    pnh_.param<int>("optimization_iterations", optimization_iterations_, 8);
    pnh_.param<double>("optimization_gain", optimization_gain_, 0.25);
    pnh_.param<double>("max_opt_translation_step", max_opt_translation_step_, 0.25);
    pnh_.param<double>("max_opt_z_step", max_opt_z_step_, 0.08);
    pnh_.param<double>("max_opt_yaw_step_deg", max_opt_yaw_step_deg_, 0.25);
    pnh_.param<double>("output_xy_scale", output_xy_scale_, 0.0);
    pnh_.param<double>("output_yaw_scale", output_yaw_scale_, 0.0);
    pnh_.param<double>("output_z_scale", output_z_scale_, 1.0);
    pnh_.param<double>("max_output_xy_correction", max_output_xy_correction_, 1.0);
    pnh_.param<double>("max_output_z_correction", max_output_z_correction_, 3.0);
    pnh_.param<double>("max_output_yaw_correction_deg", max_output_yaw_correction_deg_, 1.0);
    pnh_.param<int>("min_loop_edges_for_xy_output", min_loop_edges_for_xy_output_, 3);
    pnh_.param<double>("max_output_opt_residual", max_output_opt_residual_, 0.8);

    pnh_.param<double>("publish_rate", publish_rate_, 1.0);
    pnh_.param<int>("max_publish_map_points", max_publish_map_points_, 600000);

    odom_sub_ = nh_.subscribe(odom_topic_, 200, &SemanticKeyframePoseGraph::odomCb, this);
    cloud_sub_ = nh_.subscribe(cloud_topic_, 10, &SemanticKeyframePoseGraph::cloudCb, this);
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>(output_odom_topic_, 20);
    path_pub_ = nh_.advertise<nav_msgs::Path>(path_topic_, 1, true);
    map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(map_topic_, 1, true);
    stats_pub_ = nh_.advertise<std_msgs::String>(stats_topic_, 10);
    timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.1, publish_rate_)),
                            &SemanticKeyframePoseGraph::timerCb, this);

    ROS_INFO("semantic_keyframe_pose_graph started odom=%s cloud=%s output=%s mode=%s orientation=%s unsafe_full=%d",
             odom_topic_.c_str(), cloud_topic_.c_str(), output_odom_topic_.c_str(),
             output_mode_.c_str(), output_orientation_mode_.c_str(),
             allow_unsafe_full_output_ ? 1 : 0);
  }

 private:
  uint32_t mapLabel(uint32_t raw) const
  {
    if (label_mode_ == "internal")
    {
      return raw < LABEL_COUNT ? raw : LABEL_UNKNOWN;
    }
    if (raw == 40 || raw == 44)
    {
      return LABEL_ROAD;
    }
    if (raw == 48 || raw == 49)
    {
      return LABEL_SIDEWALK;
    }
    if (raw == 50 || raw == 51 || raw == 52)
    {
      return LABEL_BUILDING;
    }
    if (raw == 70 || raw == 71 || raw == 72 || raw == 80 || raw == 81)
    {
      return LABEL_VEGETATION;
    }
    if ((raw >= 10 && raw <= 18) || raw == 252 || raw == 253 || raw == 254 ||
        raw == 255 || raw == 256 || raw == 257 || raw == 258 || raw == 259)
    {
      return LABEL_DYNAMIC;
    }
    if (raw > 0)
    {
      return LABEL_OTHER;
    }
    return LABEL_UNKNOWN;
  }

  void odomCb(const nav_msgs::OdometryConstPtr& msg)
  {
    latest_odom_ = *msg;
    latest_pose_ = odomToPose4(*msg);
    latest_stamp_ = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    has_odom_ = true;
    publishCorrectedOdom(latest_stamp_);
  }

  void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    if (!has_odom_)
    {
      ROS_WARN_THROTTLE(2.0, "semantic pose graph waiting for odometry on %s", odom_topic_.c_str());
      return;
    }
    const ros::Time stamp = msg->header.stamp.isZero() ? latest_stamp_ : msg->header.stamp;
    if (!shouldCreateKeyframe(stamp, latest_pose_))
    {
      return;
    }

    std::vector<SemanticPoint> points = extractKeyframePoints(*msg, latest_pose_);
    if (points.size() < static_cast<std::size_t>(std::max(30, loop_min_inliers_ / 2)))
    {
      ROS_WARN_THROTTLE(3.0, "semantic pose graph skip sparse keyframe points=%zu", points.size());
      return;
    }
    addKeyframe(stamp, latest_pose_, std::move(points));
  }

  bool shouldCreateKeyframe(const ros::Time& stamp, const Pose4& pose) const
  {
    if (keyframes_.empty())
    {
      return true;
    }
    const Keyframe& last = keyframes_.back();
    const double dt = (stamp - last.stamp).toSec();
    if (dt < std::max(0.0, keyframe_min_interval_sec_))
    {
      return false;
    }
    const double dist = poseDistance2D(last.raw, pose);
    const double dyaw = std::fabs(normalizeAngle(pose.yaw - last.raw.yaw)) * 180.0 / M_PI;
    return dist >= keyframe_min_distance_ || dyaw >= keyframe_min_yaw_deg_;
  }

  std::vector<SemanticPoint> extractKeyframePoints(const sensor_msgs::PointCloud2& msg,
                                                   const Pose4& key_pose) const
  {
    std::vector<SemanticPoint> points;
    const sensor_msgs::PointField* fx = findField(msg, "x");
    const sensor_msgs::PointField* fy = findField(msg, "y");
    const sensor_msgs::PointField* fz = findField(msg, "z");
    const sensor_msgs::PointField* flabel = findFirstField(msg, {label_field_, "semantic", "class_id", "label_id"});
    const sensor_msgs::PointField* fconf = findFirstField(msg, {"confidence", "probability", "prob", "score"});
    if (!fx || !fy || !fz || !flabel)
    {
      ROS_WARN_THROTTLE(3.0, "semantic pose graph cloud missing x/y/z/label fields");
      return points;
    }

    const std::size_t n = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
    const std::size_t limit = static_cast<std::size_t>(std::max(1, max_points_per_keyframe_));
    const std::size_t stride = std::max<std::size_t>(1, n / std::max<std::size_t>(limit * 4, 1));
    points.reserve(limit);
    for (std::size_t i = 0; i < n && points.size() < limit; i += stride)
    {
      const double x = readFieldAsDouble(msg, *fx, i);
      const double y = readFieldAsDouble(msg, *fy, i);
      const double z = readFieldAsDouble(msg, *fz, i);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      {
        continue;
      }
      const double range = std::sqrt(x * x + y * y + z * z);
      if (range < min_point_range_ || range > max_point_range_)
      {
        continue;
      }
      const uint32_t raw_label = static_cast<uint32_t>(
          std::max(0.0, std::round(readFieldAsDouble(msg, *flabel, i))));
      const uint32_t label = mapLabel(raw_label);
      if (!isGraphLabel(label))
      {
        continue;
      }
      const double confidence = fconf ? readFieldAsDouble(msg, *fconf, i) : 1.0;
      if (!std::isfinite(confidence) || confidence <= 0.0)
      {
        continue;
      }

      Eigen::Vector3d p(x, y, z);
      if (!cloud_in_map_frame_)
      {
        p = transformLocalToMap(latest_pose_, p);
      }
      SemanticPoint out;
      out.p = transformMapToLocal(key_pose, p);
      out.label = label;
      out.confidence = static_cast<float>(std::max(0.05, std::min(1.0, confidence)));
      points.push_back(out);
    }
    return points;
  }

  void addKeyframe(const ros::Time& stamp, const Pose4& raw_pose, std::vector<SemanticPoint>&& points)
  {
    Keyframe kf;
    kf.id = static_cast<int>(keyframes_.size());
    kf.stamp = stamp;
    kf.raw = raw_pose;
    kf.opt = raw_pose;
    kf.local_points = std::move(points);
    if (!keyframes_.empty())
    {
      const Keyframe& prev = keyframes_.back();
      Edge odom_edge;
      odom_edge.i = prev.id;
      odom_edge.j = kf.id;
      odom_edge.relative = relativePose(prev.raw, kf.raw);
      odom_edge.weight_xy = odom_edge_weight_;
      odom_edge.weight_z = odom_edge_weight_;
      odom_edge.weight_yaw = odom_edge_weight_;
      odom_edge.loop = false;
      edges_.push_back(odom_edge);
      kf.opt = composePose(prev.opt, odom_edge.relative);
    }
    keyframes_.push_back(std::move(kf));

    if (enable_loop_constraints_)
    {
      tryAddLoopForLatest();
    }
    optimizeGraph();
    trimOldKeyframesIfNeeded();
    const ros::Time output_stamp = latest_odom_.header.stamp.isZero() ? latest_stamp_ : latest_odom_.header.stamp;
    publishCorrectedOdom(output_stamp);
  }

  void trimOldKeyframesIfNeeded()
  {
    if (max_keyframes_ <= 0 || static_cast<int>(keyframes_.size()) <= max_keyframes_)
    {
      return;
    }
    ROS_WARN_THROTTLE(5.0, "semantic pose graph reached max_keyframes=%d; keeping all nodes to preserve edge ids", max_keyframes_);
  }

  void tryAddLoopForLatest()
  {
    if (static_cast<int>(keyframes_.size()) <= loop_min_index_gap_ + 1)
    {
      return;
    }
    const int current_id = static_cast<int>(keyframes_.size()) - 1;
    const Keyframe& current = keyframes_.back();

    std::vector<std::pair<double, int>> candidates;
    const int max_old = current_id - std::max(1, loop_min_index_gap_);
    for (int i = 0; i <= max_old; ++i)
    {
      const double d = std::min(poseDistance2D(keyframes_[i].raw, current.raw),
                                poseDistance2D(keyframes_[i].opt, current.opt));
      if (d <= loop_search_radius_)
      {
        candidates.emplace_back(d, i);
      }
    }
    std::sort(candidates.begin(), candidates.end());
    if (static_cast<int>(candidates.size()) > loop_max_candidates_)
    {
      candidates.resize(static_cast<std::size_t>(loop_max_candidates_));
    }

    MatchResult best;
    MatchResult second;
    int best_id = -1;
    for (const auto& c : candidates)
    {
      MatchResult result = matchKeyframes(keyframes_[c.second], current);
      if (!enable_xy_loop_constraints_)
      {
        result.xy_accepted = false;
      }
      if (!enable_z_loop_constraints_)
      {
        result.z_accepted = false;
      }
      result.accepted = result.xy_accepted || result.z_accepted;
      if (!result.accepted)
      {
        continue;
      }
      if (result.z_accepted && !result.xy_accepted)
      {
        result.score = static_cast<double>(result.ground_inliers) /
                       (1.0 + std::max(0.03, result.z_mad));
      }
      if (!best.accepted || result.score > best.score)
      {
        second = best;
        best = result;
        best_id = c.second;
      }
      else if (!second.accepted || result.score > second.score)
      {
        second = result;
      }
    }

    last_loop_attempts_ += static_cast<int>(candidates.size());
    if (!best.accepted)
    {
      return;
    }

    if (second.accepted && std::isfinite(second.score) && second.score > 0.0)
    {
      best.score_ratio = best.score / std::max(1e-6, second.score);
      best.score_gap = best.score - second.score;
      if (best.xy_accepted &&
          (best.score_ratio < loop_min_score_ratio_ || best.score_gap < loop_min_score_gap_))
      {
        best.xy_accepted = false;
        ++loop_xy_rejected_ambiguous_;
      }
    }
    else
    {
      best.score_ratio = std::numeric_limits<double>::infinity();
      best.score_gap = std::numeric_limits<double>::infinity();
    }

    if (!best.xy_accepted && !best.z_accepted)
    {
      return;
    }

    Edge loop;
    loop.i = best_id;
    loop.j = current_id;
    loop.relative = best.relative;
    const double loop_weight = computeLoopWeight(best);
    loop.weight_xy = (enable_xy_loop_constraints_ && best.xy_accepted) ? loop_weight : 0.0;
    loop.weight_z = (enable_z_loop_constraints_ && best.z_accepted) ? loop_weight * loop_z_weight_scale_ : 0.0;
    loop.weight_yaw = (enable_xy_loop_constraints_ && best.xy_accepted) ? loop_weight * loop_yaw_weight_scale_ : 0.0;
    if (loop.weight_xy <= 0.0 && loop.weight_z <= 0.0 && loop.weight_yaw <= 0.0)
    {
      return;
    }
    loop.loop = true;
    loop.inliers = best.inliers;
    loop.rms = best.rms;
    edges_.push_back(loop);
    ++loop_edges_;
    if (best.xy_accepted && enable_xy_loop_constraints_)
    {
      ++xy_loop_edges_;
    }
    if (best.z_accepted && enable_z_loop_constraints_)
    {
      ++z_loop_edges_;
    }
    last_loop_inliers_ = best.inliers;
    last_loop_ground_inliers_ = best.ground_inliers;
    last_loop_rms_ = best.rms;
    last_loop_z_median_ = best.z_median;
    last_loop_z_mad_ = best.z_mad;
    last_loop_yaw_delta_deg_ = best.yaw_delta * 180.0 / M_PI;
    last_loop_xy_delta_ = best.xy_delta;
    last_loop_weight_ = loop_weight;
    last_loop_building_inliers_ = best.building_inliers;
    last_loop_spread_ratio_ = best.spread_ratio;
    last_loop_score_ = best.score;
    last_loop_score_ratio_ = best.score_ratio;
    last_loop_score_gap_ = best.score_gap;
    ROS_INFO("semantic loop accepted %d -> %d xy=%d z=%d inliers=%d building=%d ground=%d rms=%.3f spread_ratio=%.2f z=%.3f mad=%.3f score=%.1f ratio=%.2f weight=%.2f xy_delta=%.2f yaw_delta=%.2fdeg",
             best_id, current_id, best.xy_accepted ? 1 : 0, best.z_accepted ? 1 : 0,
             best.inliers, best.building_inliers, best.ground_inliers, best.rms,
             best.spread_ratio, best.z_median, best.z_mad, best.score, best.score_ratio,
             loop_weight, best.xy_delta, best.yaw_delta * 180.0 / M_PI);
  }

  double computeLoopWeight(const MatchResult& match) const
  {
    const double inlier_factor =
        clampValue(static_cast<double>(match.inliers) / std::max(1.0, loop_weight_inlier_reference_),
                   0.5, 2.0);
    const double ground_factor =
        clampValue(static_cast<double>(match.ground_inliers) /
                       std::max(1.0, static_cast<double>(loop_min_ground_inliers_)),
                   0.7, 1.5);
    const double rms_factor =
        clampValue(loop_weight_rms_reference_ / std::max(0.05, match.rms), 0.5, 2.0);
    const double z_factor =
        clampValue(loop_weight_z_mad_reference_ / std::max(0.03, match.z_mad), 0.5, 1.8);
    return clampValue(loop_edge_weight_ * inlier_factor * ground_factor * rms_factor * z_factor,
                      loop_edge_min_weight_, loop_edge_max_weight_);
  }

  using FeatureGrid = std::unordered_map<CellKey, CellStats, CellKeyHash>;

  CellKey cellKeyForPoint(const Eigen::Vector3d& p, uint32_t label) const
  {
    const double res = std::max(0.1, icp_grid_resolution_);
    CellKey key;
    key.x = static_cast<int>(std::floor(p.x() / res));
    key.y = static_cast<int>(std::floor(p.y() / res));
    key.label = label;
    return key;
  }

  void addPointToGrid(FeatureGrid& grid, const Eigen::Vector3d& p, uint32_t label, double confidence) const
  {
    if (!isGraphLabel(label) || !std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
    {
      return;
    }
    const double w = clampValue(confidence, 0.05, 1.0);
    CellStats& cell = grid[cellKeyForPoint(p, label)];
    cell.sum += p * w;
    cell.weight += w;
    cell.count += 1;
    cell.label = label;
    cell.min_z = std::min(cell.min_z, p.z());
    cell.max_z = std::max(cell.max_z, p.z());
  }

  bool cellUsableForMatch(const CellStats& cell, bool xy_match) const
  {
    if (cell.count <= 0 || cell.weight <= 1e-9)
    {
      return false;
    }
    if (xy_match)
    {
      return cell.count >= std::max(1, feature_min_cell_points_);
    }
    if (!isZMatchLabel(cell.label))
    {
      return false;
    }
    if (cell.count < std::max(1, ground_min_cell_points_))
    {
      return false;
    }
    const double z_span = cell.max_z - cell.min_z;
    if (max_ground_cell_z_span_ > 0.0 && z_span > max_ground_cell_z_span_)
    {
      return false;
    }
    const double z = cell.mean().z();
    if (ground_cell_local_z_min_ < ground_cell_local_z_max_ &&
        (z < ground_cell_local_z_min_ || z > ground_cell_local_z_max_))
    {
      return false;
    }
    return true;
  }

  int cellMatchWeight(const CellStats& a, const CellStats& b) const
  {
    return std::max(1, std::min(5, std::min(a.count, b.count)));
  }

  const CellStats* findNearestCell(const FeatureGrid& grid,
                                   const Eigen::Vector3d& p,
                                   uint32_t label,
                                   bool xy_match,
                                   double* best_d2 = nullptr) const
  {
    const CellKey base = cellKeyForPoint(p, label);
    const double max_d2 = icp_max_correspondence_ * icp_max_correspondence_;
    double best = max_d2;
    const CellStats* best_cell = nullptr;
    for (int dx = -1; dx <= 1; ++dx)
    {
      for (int dy = -1; dy <= 1; ++dy)
      {
        const CellKey key{base.x + dx, base.y + dy, label};
        const auto it = grid.find(key);
        if (it == grid.end() || !cellUsableForMatch(it->second, xy_match))
        {
          continue;
        }
        const double d2 = (it->second.mean().head<2>() - p.head<2>()).squaredNorm();
        if (d2 < best)
        {
          best = d2;
          best_cell = &it->second;
        }
      }
    }
    if (best_d2)
    {
      *best_d2 = best;
    }
    return best_cell;
  }

  FeatureGrid buildGrid(const Keyframe& ref) const
  {
    FeatureGrid grid;
    for (const auto& p : ref.local_points)
    {
      if (!isStructuralXYLabel(p.label) && !isZMatchLabel(p.label))
      {
        continue;
      }
      addPointToGrid(grid, p.p, p.label, p.confidence);
    }
    return grid;
  }

  FeatureGrid buildTransformedGrid(const Keyframe& cur, const Pose4& rel) const
  {
    FeatureGrid grid;
    for (const auto& p : cur.local_points)
    {
      if (!isStructuralXYLabel(p.label) && !isZMatchLabel(p.label))
      {
        continue;
      }
      addPointToGrid(grid, transformLocalToMap(rel, p.p), p.label, p.confidence);
    }
    return grid;
  }

  CandidateScore evaluateCandidate(
      const Keyframe& ref,
      const Keyframe& cur,
      const FeatureGrid& grid,
      const Pose4& rel) const
  {
    (void)ref;
    CandidateScore out;
    const FeatureGrid cur_grid = buildTransformedGrid(cur, rel);
    double err_sum = 0.0;
    for (const auto& kv : cur_grid)
    {
      const CellStats& cell = kv.second;
      if (!isStructuralXYLabel(cell.label) || !cellUsableForMatch(cell, true))
      {
        continue;
      }
      const Eigen::Vector3d p_ref = cell.mean();
      double best_d2 = 0.0;
      const CellStats* ref_cell = findNearestCell(grid, p_ref, cell.label, true, &best_d2);
      if (!ref_cell)
      {
        continue;
      }
      const int w = cellMatchWeight(cell, *ref_cell);
      out.inliers += w;
      if (isGroundLabel(cell.label))
      {
        out.ground_inliers += w;
      }
      err_sum += best_d2 * static_cast<double>(w);
    }

    if (out.inliers > 0)
    {
      out.rms = std::sqrt(err_sum / static_cast<double>(out.inliers));
      out.score = static_cast<double>(out.inliers) / (1.0 + out.rms) +
                  0.25 * static_cast<double>(out.ground_inliers);
    }
    return out;
  }

  Pose4 coarseSearchRelative(
      const Keyframe& ref,
      const Keyframe& cur,
      const FeatureGrid& grid,
      const Pose4& initial) const
  {
    if (!enable_coarse_loop_search_)
    {
      return initial;
    }

    const double xy_radius = std::max(0.0, coarse_xy_search_radius_);
    const double xy_step = std::max(0.25, coarse_xy_search_step_);
    const double yaw_radius = std::max(0.0, coarse_yaw_search_deg_) * M_PI / 180.0;
    const double yaw_step = std::max(0.25, coarse_yaw_search_step_deg_) * M_PI / 180.0;
    const int min_inliers = std::max(10, coarse_min_inliers_);

    Pose4 best_rel = initial;
    CandidateScore best = evaluateCandidate(ref, cur, grid, initial);
    if (best.inliers < min_inliers)
    {
      best.score = -std::numeric_limits<double>::infinity();
    }

    for (double dyaw = -yaw_radius; dyaw <= yaw_radius + 1e-9; dyaw += yaw_step)
    {
      for (double dx = -xy_radius; dx <= xy_radius + 1e-9; dx += xy_step)
      {
        for (double dy = -xy_radius; dy <= xy_radius + 1e-9; dy += xy_step)
        {
          if (std::fabs(dx) < 1e-9 && std::fabs(dy) < 1e-9 && std::fabs(dyaw) < 1e-9)
          {
            continue;
          }
          Pose4 candidate = initial;
          candidate.x += dx;
          candidate.y += dy;
          candidate.yaw = normalizeAngle(candidate.yaw + dyaw);
          const CandidateScore score = evaluateCandidate(ref, cur, grid, candidate);
          if (score.inliers >= min_inliers && score.score > best.score)
          {
            best = score;
            best_rel = candidate;
          }
        }
      }
    }

    return best_rel;
  }

  MatchResult matchKeyframes(const Keyframe& ref, const Keyframe& cur) const
  {
    MatchResult out;
    Pose4 rel = relativePose(ref.raw, cur.raw);
    const Pose4 raw_rel = rel;
    const auto grid = buildGrid(ref);
    if (grid.empty())
    {
      return out;
    }
    rel = coarseSearchRelative(ref, cur, grid, rel);

    bool have_xy_structure = true;
    for (int iter = 0; iter < std::max(1, icp_iterations_); ++iter)
    {
      std::vector<Eigen::Vector2d> src;
      std::vector<Eigen::Vector2d> tgt;
      std::vector<double> z_residuals;
      collectCorrespondences(ref, cur, grid, rel, src, tgt, z_residuals);
      if (static_cast<int>(src.size()) < std::max(3, loop_min_building_inliers_))
      {
        have_xy_structure = false;
        break;
      }

      Eigen::Vector2d center = Eigen::Vector2d::Zero();
      for (const auto& p : src)
      {
        center += p;
      }
      center /= static_cast<double>(src.size());

      Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
      Eigen::Vector3d g = Eigen::Vector3d::Zero();
      double spread = 0.0;
      for (std::size_t i = 0; i < src.size(); ++i)
      {
        const Eigen::Vector2d d = src[i] - center;
        const Eigen::Vector2d rhs = tgt[i] - src[i];
        Eigen::Matrix<double, 2, 3> A;
        A << 1.0, 0.0, -d.y(),
             0.0, 1.0,  d.x();
        H += A.transpose() * A;
        g += A.transpose() * rhs;
        spread += d.squaredNorm();
      }
      spread = std::sqrt(spread / std::max<std::size_t>(1, src.size()));
      if (spread < loop_min_spread_ || std::fabs(H.determinant()) < 1e-9)
      {
        have_xy_structure = false;
        break;
      }
      const Eigen::Vector3d delta = H.ldlt().solve(g);
      if (!std::isfinite(delta.x()) || !std::isfinite(delta.y()) || !std::isfinite(delta.z()))
      {
        have_xy_structure = false;
        break;
      }

      const double max_xy_step = 0.6;
      Eigen::Vector2d dxy(delta.x(), delta.y());
      if (dxy.norm() > max_xy_step)
      {
        dxy *= max_xy_step / dxy.norm();
      }
      const double max_yaw_step = 2.0 * M_PI / 180.0;
      const double dyaw = std::max(-max_yaw_step, std::min(max_yaw_step, delta.z()));
      rel.x += dxy.x();
      rel.y += dxy.y();
      rel.yaw = normalizeAngle(rel.yaw + dyaw);
    }

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> tgt;
    std::vector<double> unused_z_residuals;
    std::vector<uint32_t> labels;
    collectCorrespondences(ref, cur, grid, rel, src, tgt, unused_z_residuals, &labels);

    std::vector<double> z_residuals;
    const int ground = collectZResiduals(ref, cur, grid, rel, z_residuals);
    if (!z_residuals.empty())
    {
      rel.z += clampValue(robustMedian(z_residuals), -0.2, 0.2);
      z_residuals.clear();
      collectZResiduals(ref, cur, grid, rel, z_residuals);
    }

    double err_sum = 0.0;
    Eigen::Vector2d mean_src = Eigen::Vector2d::Zero();
    int building = 0;
    for (std::size_t i = 0; i < src.size(); ++i)
    {
      err_sum += (src[i] - tgt[i]).squaredNorm();
      mean_src += src[i];
      if (i < labels.size() && labels[i] == LABEL_BUILDING)
      {
        ++building;
      }
    }
    if (!src.empty())
    {
      mean_src /= static_cast<double>(src.size());
    }
    double spread = 0.0;
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& p : src)
    {
      const Eigen::Vector2d d = p - mean_src;
      spread += d.squaredNorm();
      cov += d * d.transpose();
    }
    double spread_ratio = 0.0;
    if (!src.empty())
    {
      spread = std::sqrt(spread / static_cast<double>(src.size()));
      cov /= static_cast<double>(src.size());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
      if (solver.info() == Eigen::Success)
      {
        const double l0 = std::max(0.0, solver.eigenvalues().x());
        const double l1 = std::max(0.0, solver.eigenvalues().y());
        spread_ratio = std::sqrt(std::max(1e-9, std::min(l0, l1)) /
                                 std::max(1e-9, std::max(l0, l1)));
      }
    }
    const double rms = src.empty() ? std::numeric_limits<double>::infinity() :
                       std::sqrt(err_sum / static_cast<double>(src.size()));
    const double z_med = z_residuals.empty() ? 0.0 : robustMedian(z_residuals);
    const double z_mad = z_residuals.empty() ? std::numeric_limits<double>::infinity() :
                         robustMad(z_residuals, z_med);
    const double xy_delta = std::hypot(rel.x - raw_rel.x, rel.y - raw_rel.y);
    const double yaw_delta = std::fabs(normalizeAngle(rel.yaw - raw_rel.yaw));

    out.relative = rel;
    out.inliers = static_cast<int>(src.size());
    out.ground_inliers = ground;
    out.building_inliers = building;
    out.rms = rms;
    out.spread = spread;
    out.spread_ratio = spread_ratio;
    out.z_median = z_med;
    out.z_mad = z_mad;
    out.xy_delta = xy_delta;
    out.yaw_delta = yaw_delta;
    const bool xy_ok =
        have_xy_structure &&
        out.inliers >= loop_min_inliers_ &&
        building >= loop_min_building_inliers_ &&
        rms <= loop_max_rms_ &&
        spread >= loop_min_spread_ &&
        spread_ratio >= loop_min_spread_ratio_ &&
        xy_delta <= std::min(loop_max_xy_correction_, loop_max_xy_correction_for_xy_) &&
        yaw_delta <= std::min(loop_max_yaw_correction_deg_, loop_max_yaw_correction_for_xy_deg_) * M_PI / 180.0;
    const bool z_ok =
        ground >= loop_min_ground_inliers_ &&
        std::fabs(z_med) <= loop_max_z_median_ &&
        z_mad <= loop_max_z_mad_ &&
        xy_delta <= loop_max_xy_correction_for_z_;
    out.xy_accepted = xy_ok;
    out.z_accepted = z_ok;
    out.accepted = out.xy_accepted || out.z_accepted;
    const double xy_score = xy_ok ? (static_cast<double>(out.inliers) / (1.0 + rms) +
                                    2.0 * static_cast<double>(building) +
                                    100.0 * spread_ratio) : 0.0;
    const double z_score = z_ok ? (static_cast<double>(ground) / (1.0 + z_mad)) : 0.0;
    out.score = std::max(xy_score, z_score);
    return out;
  }

  void collectCorrespondences(const Keyframe& ref,
                              const Keyframe& cur,
                              const FeatureGrid& grid,
                              const Pose4& rel,
                              std::vector<Eigen::Vector2d>& src,
                              std::vector<Eigen::Vector2d>& tgt,
                              std::vector<double>& z_residuals,
                              std::vector<uint32_t>* labels = nullptr) const
  {
    (void)ref;
    (void)z_residuals;
    const FeatureGrid cur_grid = buildTransformedGrid(cur, rel);
    for (const auto& kv : cur_grid)
    {
      const CellStats& cell = kv.second;
      if (!isStructuralXYLabel(cell.label) || !cellUsableForMatch(cell, true))
      {
        continue;
      }
      const Eigen::Vector3d p_ref = cell.mean();
      const CellStats* best = findNearestCell(grid, p_ref, cell.label, true);
      if (!best)
      {
        continue;
      }
      src.push_back(p_ref.head<2>());
      tgt.push_back(best->mean().head<2>());
      if (labels)
      {
        labels->push_back(cell.label);
      }
    }
  }

  int collectZResiduals(const Keyframe& ref,
                        const Keyframe& cur,
                        const FeatureGrid& grid,
                        const Pose4& rel,
                        std::vector<double>& z_residuals) const
  {
    (void)ref;
    const FeatureGrid cur_grid = buildTransformedGrid(cur, rel);
    int ground = 0;
    for (const auto& kv : cur_grid)
    {
      const CellStats& cell = kv.second;
      if (!isZMatchLabel(cell.label) || !cellUsableForMatch(cell, false))
      {
        continue;
      }
      const Eigen::Vector3d p_ref = cell.mean();
      const CellStats* best = findNearestCell(grid, p_ref, cell.label, false);
      if (!best)
      {
        continue;
      }
      z_residuals.push_back(best->mean().z() - p_ref.z());
      ++ground;
    }
    return ground;
  }

  void optimizeGraph()
  {
    if (keyframes_.size() < 2 || edges_.empty())
    {
      return;
    }
    const double max_t = std::max(0.01, max_opt_translation_step_);
    const double max_z = std::max(0.005, max_opt_z_step_);
    const double max_yaw = std::max(1e-5, max_opt_yaw_step_deg_ * M_PI / 180.0);
    const double gain = std::max(0.001, std::min(1.0, optimization_gain_));
    last_opt_mean_residual_ = 0.0;
    last_opt_max_step_ = 0.0;
    last_opt_iterations_ = 0;

    for (int iter = 0; iter < std::max(1, optimization_iterations_); ++iter)
    {
      std::vector<PoseDelta> deltas(keyframes_.size());
      double residual_sum = 0.0;
      int residual_count = 0;

      for (const Edge& edge : edges_)
      {
        if (edge.i < 0 || edge.j < 0 ||
            edge.i >= static_cast<int>(keyframes_.size()) ||
            edge.j >= static_cast<int>(keyframes_.size()))
        {
          continue;
        }
        const Pose4 pred = composePose(keyframes_[static_cast<std::size_t>(edge.i)].opt, edge.relative);
        const Pose4& pj = keyframes_[static_cast<std::size_t>(edge.j)].opt;
        Eigen::Vector2d e_xy(pred.x - pj.x, pred.y - pj.y);
        const double xy_norm = e_xy.norm();
        const double robust =
            edge.loop ? std::min(1.0, loop_robust_kernel_ / std::max(loop_robust_kernel_, xy_norm)) : 1.0;
        const double wxy = robust * std::max(0.0, edge.weight_xy);
        const double wz = robust * std::max(0.0, edge.weight_z);
        const double wyaw = robust * std::max(0.0, edge.weight_yaw);
        const double ez = pred.z - pj.z;
        const double eyaw = normalizeAngle(pred.yaw - pj.yaw);

        PoseDelta& di = deltas[static_cast<std::size_t>(edge.i)];
        PoseDelta& dj = deltas[static_cast<std::size_t>(edge.j)];
        di.dx -= wxy * e_xy.x();
        di.dy -= wxy * e_xy.y();
        di.dz -= wz * ez;
        di.dyaw -= wyaw * eyaw;
        di.wxy += wxy;
        di.wz += wz;
        di.wyaw += wyaw;

        dj.dx += wxy * e_xy.x();
        dj.dy += wxy * e_xy.y();
        dj.dz += wz * ez;
        dj.dyaw += wyaw * eyaw;
        dj.wxy += wxy;
        dj.wz += wz;
        dj.wyaw += wyaw;

        residual_sum += std::sqrt(e_xy.squaredNorm() + ez * ez);
        ++residual_count;
      }

      double max_step = 0.0;
      for (std::size_t i = 1; i < keyframes_.size(); ++i)
      {
        PoseDelta& d = deltas[i];
        Pose4& pose = keyframes_[i].opt;
        Eigen::Vector2d step_xy(0.0, 0.0);
        if (d.wxy > 1e-9)
        {
          step_xy = gain * Eigen::Vector2d(d.dx / d.wxy, d.dy / d.wxy);
          const double n = step_xy.norm();
          if (n > max_t)
          {
            step_xy *= max_t / n;
          }
          pose.x += step_xy.x();
          pose.y += step_xy.y();
        }

        double step_z = 0.0;
        if (d.wz > 1e-9)
        {
          step_z = gain * d.dz / d.wz;
          step_z = clampValue(step_z, -max_z, max_z);
          pose.z += step_z;
        }

        double step_yaw = 0.0;
        if (d.wyaw > 1e-9)
        {
          step_yaw = gain * d.dyaw / d.wyaw;
          step_yaw = clampValue(step_yaw, -max_yaw, max_yaw);
          pose.yaw = normalizeAngle(pose.yaw + step_yaw);
        }
        max_step = std::max(max_step, std::max(step_xy.norm(), std::fabs(step_z)));
        max_step = std::max(max_step, std::fabs(step_yaw));
      }
      keyframes_.front().opt = keyframes_.front().raw;
      last_opt_mean_residual_ =
          residual_count > 0 ? residual_sum / static_cast<double>(residual_count) : 0.0;
      last_opt_max_step_ = max_step;
      last_opt_iterations_ = iter + 1;
      if (max_step < 1e-4)
      {
        break;
      }
    }
  }

  Pose4 correctedLatestPose() const
  {
    if (keyframes_.empty())
    {
      return latest_pose_;
    }
    const Keyframe& ref = keyframes_.back();
    return composePose(ref.opt, relativePose(ref.raw, latest_pose_));
  }

  Pose4 outputPoseFromGraph(const Pose4& raw, const Pose4& graph) const
  {
    if (output_mode_ == "full" && allow_unsafe_full_output_)
    {
      return graph;
    }
    const std::string mode = (output_mode_ == "full") ? std::string("safe") : output_mode_;

    const Eigen::Vector2d dxy_graph(graph.x - raw.x, graph.y - raw.y);
    const double dyaw_graph = normalizeAngle(graph.yaw - raw.yaw);
    const double dz_graph = graph.z - raw.z;
    const double max_xy = std::max(0.0, max_output_xy_correction_);
    const double max_z = std::max(0.0, max_output_z_correction_);
    const double max_yaw = std::max(0.0, max_output_yaw_correction_deg_) * M_PI / 180.0;

    Pose4 out = raw;
    Eigen::Vector2d dxy(0.0, 0.0);
    double dyaw = 0.0;
    double z_scale = clampValue(output_z_scale_, 0.0, 1.0);
    double xy_scale = clampValue(output_xy_scale_, 0.0, 1.0);
    double yaw_scale = clampValue(output_yaw_scale_, 0.0, 1.0);

    const bool safe_xy =
        loop_edges_ >= std::max(0, min_loop_edges_for_xy_output_) &&
        (max_output_opt_residual_ <= 0.0 || last_opt_mean_residual_ <= max_output_opt_residual_) &&
        dxy_graph.norm() <= std::max(1e-6, max_xy) &&
        std::fabs(dyaw_graph) <= std::max(1e-6, max_yaw);

    if (mode == "blend" || (mode == "safe" && safe_xy))
    {
      dxy = xy_scale * dxy_graph;
      dyaw = yaw_scale * dyaw_graph;
    }
    else if (mode == "z_only" || mode == "safe")
    {
      dxy.setZero();
      dyaw = 0.0;
    }
    else
    {
      ROS_WARN_THROTTLE(5.0, "unknown semantic pose graph output_mode=%s; using z_only",
                        output_mode_.c_str());
      dxy.setZero();
      dyaw = 0.0;
    }

    if (dxy.norm() > max_xy && dxy.norm() > 1e-9)
    {
      dxy *= max_xy / dxy.norm();
    }
    dyaw = clampValue(dyaw, -max_yaw, max_yaw);
    const double dz = clampValue(z_scale * dz_graph, -max_z, max_z);

    out.x += dxy.x();
    out.y += dxy.y();
    out.z += dz;
    out.yaw = normalizeAngle(out.yaw + dyaw);
    return out;
  }

  void publishCorrectedOdom(const ros::Time& stamp)
  {
    if (!has_odom_)
    {
      return;
    }
    nav_msgs::Odometry out = latest_odom_;
    out.header.stamp = stamp.isZero() ? latest_stamp_ : stamp;
    out.header.frame_id = map_frame_;
    const Pose4 graph = correctedLatestPose();
    const Pose4 p = outputPoseFromGraph(latest_pose_, graph);
    last_output_dx_ = p.x - latest_pose_.x;
    last_output_dy_ = p.y - latest_pose_.y;
    last_output_dz_ = p.z - latest_pose_.z;
    last_output_dyaw_deg_ = normalizeAngle(p.yaw - latest_pose_.yaw) * 180.0 / M_PI;
    out.pose.pose.position.x = p.x;
    out.pose.pose.position.y = p.y;
    out.pose.pose.position.z = p.z;
    if (output_orientation_mode_ == "raw")
    {
      out.pose.pose.orientation = latest_odom_.pose.pose.orientation;
    }
    else if (output_orientation_mode_ == "yaw_only")
    {
      out.pose.pose.orientation = quatFromYaw(p.yaw);
    }
    else
    {
      out.pose.pose.orientation =
          quatWithRawRollPitchAndYaw(latest_odom_.pose.pose.orientation, p.yaw);
    }
    const bool xy_passthrough = std::fabs(last_output_dx_) < 1e-6 &&
                                std::fabs(last_output_dy_) < 1e-6 &&
                                std::fabs(last_output_dyaw_deg_) < 1e-6;
    out.pose.covariance[0] = xy_passthrough ? latest_odom_.pose.covariance[0] : 1.0;
    out.pose.covariance[7] = xy_passthrough ? latest_odom_.pose.covariance[7] : out.pose.covariance[0];
    out.pose.covariance[14] = keyframes_.empty() ? latest_odom_.pose.covariance[14] : 0.25;
    out.pose.covariance[35] = xy_passthrough ? latest_odom_.pose.covariance[35] : 0.05;
    odom_pub_.publish(out);
  }

  void timerCb(const ros::TimerEvent&)
  {
    publishPath();
    if (publish_optimized_map_)
    {
      publishMap();
    }
    publishStats();
  }

  void publishPath()
  {
    nav_msgs::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = ros::Time::now();
    path.poses.reserve(keyframes_.size());
    for (const auto& kf : keyframes_)
    {
      const Pose4 pose = outputPoseFromGraph(kf.raw, kf.opt);
      geometry_msgs::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = pose.x;
      ps.pose.position.y = pose.y;
      ps.pose.position.z = pose.z;
      ps.pose.orientation = quatFromYaw(pose.yaw);
      path.poses.push_back(ps);
    }
    path_pub_.publish(path);
  }

  void publishMap()
  {
    std::size_t total = 0;
    for (const auto& kf : keyframes_)
    {
      total += kf.local_points.size();
    }
    if (total == 0)
    {
      return;
    }
    const std::size_t limit = static_cast<std::size_t>(std::max(1, max_publish_map_points_));
    const std::size_t stride = std::max<std::size_t>(1, total / limit);

    sensor_msgs::PointCloud2 msg;
    msg.header.frame_id = map_frame_;
    msg.header.stamp = ros::Time::now();
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2Fields(6,
                                  "x", 1, sensor_msgs::PointField::FLOAT32,
                                  "y", 1, sensor_msgs::PointField::FLOAT32,
                                  "z", 1, sensor_msgs::PointField::FLOAT32,
                                  "rgb", 1, sensor_msgs::PointField::UINT32,
                                  "label", 1, sensor_msgs::PointField::UINT32,
                                  "confidence", 1, sensor_msgs::PointField::FLOAT32);
    const std::size_t out_count = std::min(limit, (total + stride - 1) / stride);
    modifier.resize(out_count);
    sensor_msgs::PointCloud2Iterator<float> ix(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iy(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iz(msg, "z");
    sensor_msgs::PointCloud2Iterator<uint32_t> irgb(msg, "rgb");
    sensor_msgs::PointCloud2Iterator<uint32_t> ilabel(msg, "label");
    sensor_msgs::PointCloud2Iterator<float> iconf(msg, "confidence");

    std::size_t seen = 0;
    std::size_t written = 0;
    for (const auto& kf : keyframes_)
    {
      const Pose4 pose = outputPoseFromGraph(kf.raw, kf.opt);
      for (const auto& p : kf.local_points)
      {
        if ((seen++ % stride) != 0 || written >= out_count)
        {
          continue;
        }
        const Eigen::Vector3d pm = transformLocalToMap(pose, p.p);
        *ix = static_cast<float>(pm.x());
        *iy = static_cast<float>(pm.y());
        *iz = static_cast<float>(pm.z());
        *irgb = labelColor(p.label);
        *ilabel = p.label;
        *iconf = p.confidence;
        ++ix; ++iy; ++iz; ++irgb; ++ilabel; ++iconf;
        ++written;
      }
    }
    if (written < out_count)
    {
      modifier.resize(written);
    }
    map_pub_.publish(msg);
  }

  void publishStats()
  {
    std::ostringstream ss;
    ss << "{";
    ss << "\"keyframes\":" << keyframes_.size() << ",";
    ss << "\"edges\":" << edges_.size() << ",";
    ss << "\"loop_edges\":" << loop_edges_ << ",";
    ss << "\"xy_loop_edges\":" << xy_loop_edges_ << ",";
    ss << "\"z_loop_edges\":" << z_loop_edges_ << ",";
    ss << "\"loop_xy_rejected_ambiguous\":" << loop_xy_rejected_ambiguous_ << ",";
    ss << "\"loop_attempts\":" << last_loop_attempts_ << ",";
    ss << "\"last_loop_inliers\":" << last_loop_inliers_ << ",";
    ss << "\"last_loop_building_inliers\":" << last_loop_building_inliers_ << ",";
    ss << "\"last_loop_ground_inliers\":" << last_loop_ground_inliers_ << ",";
    ss << "\"last_loop_rms\":" << last_loop_rms_ << ",";
    ss << "\"last_loop_z_median\":" << last_loop_z_median_ << ",";
    ss << "\"last_loop_z_mad\":" << last_loop_z_mad_ << ",";
    ss << "\"last_loop_spread_ratio\":" << last_loop_spread_ratio_ << ",";
    ss << "\"last_loop_score\":" << last_loop_score_ << ",";
    ss << "\"last_loop_score_ratio\":" << last_loop_score_ratio_ << ",";
    ss << "\"last_loop_score_gap\":" << last_loop_score_gap_ << ",";
    ss << "\"last_loop_xy_delta\":" << last_loop_xy_delta_ << ",";
    ss << "\"last_loop_yaw_delta_deg\":" << last_loop_yaw_delta_deg_ << ",";
    ss << "\"last_loop_weight\":" << last_loop_weight_ << ",";
    ss << "\"last_opt_mean_residual\":" << last_opt_mean_residual_ << ",";
    ss << "\"last_opt_max_step\":" << last_opt_max_step_ << ",";
    ss << "\"last_opt_iterations\":" << last_opt_iterations_ << ",";
    ss << "\"output_mode\":\"" << output_mode_ << "\",";
    ss << "\"output_orientation_mode\":\"" << output_orientation_mode_ << "\",";
    ss << "\"allow_unsafe_full_output\":" << (allow_unsafe_full_output_ ? 1 : 0) << ",";
    ss << "\"output_dx\":" << last_output_dx_ << ",";
    ss << "\"output_dy\":" << last_output_dy_ << ",";
    ss << "\"output_dz\":" << last_output_dz_ << ",";
    ss << "\"output_dyaw_deg\":" << last_output_dyaw_deg_ << ",";
    if (!keyframes_.empty())
    {
      const Keyframe& k = keyframes_.back();
      ss << "\"latest_raw_x\":" << k.raw.x << ",";
      ss << "\"latest_raw_y\":" << k.raw.y << ",";
      ss << "\"latest_opt_x\":" << k.opt.x << ",";
      ss << "\"latest_opt_y\":" << k.opt.y << ",";
      ss << "\"latest_opt_z\":" << k.opt.z;
    }
    else
    {
      ss << "\"latest_raw_x\":0,\"latest_raw_y\":0,\"latest_opt_x\":0,\"latest_opt_y\":0,\"latest_opt_z\":0";
    }
    ss << "}";
    std_msgs::String msg;
    msg.data = ss.str();
    stats_pub_.publish(msg);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber cloud_sub_;
  ros::Publisher odom_pub_;
  ros::Publisher path_pub_;
  ros::Publisher map_pub_;
  ros::Publisher stats_pub_;
  ros::Timer timer_;

  std::string odom_topic_;
  std::string cloud_topic_;
  std::string output_odom_topic_;
  std::string path_topic_;
  std::string map_topic_;
  std::string stats_topic_;
  std::string map_frame_;
  std::string label_field_;
  std::string label_mode_;
  std::string output_mode_ = "z_only";
  std::string output_orientation_mode_ = "preserve_roll_pitch";

  bool cloud_in_map_frame_ = true;
  bool enable_loop_constraints_ = true;
  bool enable_coarse_loop_search_ = true;
  bool enable_xy_loop_constraints_ = false;
  bool enable_z_loop_constraints_ = true;
  bool publish_optimized_map_ = true;
  bool allow_unsafe_full_output_ = false;
  bool has_odom_ = false;

  double keyframe_min_distance_ = 2.0;
  double keyframe_min_yaw_deg_ = 8.0;
  double keyframe_min_interval_sec_ = 0.5;
  double min_point_range_ = 0.5;
  double max_point_range_ = 120.0;
  double loop_search_radius_ = 18.0;
  double loop_max_rms_ = 0.85;
  double loop_min_spread_ = 6.0;
  double loop_max_xy_correction_ = 4.0;
  double loop_max_yaw_correction_deg_ = 6.0;
  double loop_max_z_median_ = 1.5;
  double loop_max_z_mad_ = 0.35;
  double loop_min_spread_ratio_ = 0.18;
  double loop_min_score_ratio_ = 1.15;
  double loop_min_score_gap_ = 40.0;
  double loop_max_xy_correction_for_xy_ = 3.0;
  double loop_max_yaw_correction_for_xy_deg_ = 5.0;
  double loop_max_xy_correction_for_z_ = 6.0;
  double icp_grid_resolution_ = 0.8;
  double icp_max_correspondence_ = 1.2;
  double max_ground_cell_z_span_ = 0.35;
  double ground_cell_local_z_min_ = -4.0;
  double ground_cell_local_z_max_ = 1.2;
  double coarse_xy_search_radius_ = 10.0;
  double coarse_xy_search_step_ = 2.0;
  double coarse_yaw_search_deg_ = 8.0;
  double coarse_yaw_search_step_deg_ = 2.0;
  double odom_edge_weight_ = 1.0;
  double loop_edge_weight_ = 0.35;
  double loop_edge_min_weight_ = 0.25;
  double loop_edge_max_weight_ = 2.5;
  double loop_weight_inlier_reference_ = 500.0;
  double loop_weight_rms_reference_ = 0.55;
  double loop_weight_z_mad_reference_ = 0.20;
  double loop_z_weight_scale_ = 1.0;
  double loop_yaw_weight_scale_ = 1.3;
  double loop_robust_kernel_ = 4.0;
  double optimization_gain_ = 0.25;
  double max_opt_translation_step_ = 0.25;
  double max_opt_z_step_ = 0.08;
  double max_opt_yaw_step_deg_ = 0.25;
  double output_xy_scale_ = 0.0;
  double output_yaw_scale_ = 0.0;
  double output_z_scale_ = 1.0;
  double max_output_xy_correction_ = 1.0;
  double max_output_z_correction_ = 3.0;
  double max_output_yaw_correction_deg_ = 1.0;
  double max_output_opt_residual_ = 0.8;
  double publish_rate_ = 1.0;

  int max_keyframes_ = 2000;
  int max_points_per_keyframe_ = 1800;
  int loop_min_index_gap_ = 30;
  int loop_max_candidates_ = 8;
  int loop_min_inliers_ = 80;
  int loop_min_ground_inliers_ = 30;
  int loop_min_building_inliers_ = 50;
  int feature_min_cell_points_ = 3;
  int ground_min_cell_points_ = 4;
  int icp_iterations_ = 5;
  int coarse_sample_points_ = 700;
  int coarse_min_inliers_ = 60;
  int optimization_iterations_ = 8;
  int min_loop_edges_for_xy_output_ = 3;
  int max_publish_map_points_ = 600000;

  nav_msgs::Odometry latest_odom_;
  Pose4 latest_pose_;
  ros::Time latest_stamp_;
  std::vector<Keyframe> keyframes_;
  std::vector<Edge> edges_;

  int loop_edges_ = 0;
  int xy_loop_edges_ = 0;
  int z_loop_edges_ = 0;
  int loop_xy_rejected_ambiguous_ = 0;
  int last_loop_attempts_ = 0;
  int last_loop_inliers_ = 0;
  int last_loop_ground_inliers_ = 0;
  int last_loop_building_inliers_ = 0;
  double last_loop_rms_ = 0.0;
  double last_loop_z_median_ = 0.0;
  double last_loop_z_mad_ = 0.0;
  double last_loop_spread_ratio_ = 0.0;
  double last_loop_score_ = 0.0;
  double last_loop_score_ratio_ = 0.0;
  double last_loop_score_gap_ = 0.0;
  double last_loop_xy_delta_ = 0.0;
  double last_loop_yaw_delta_deg_ = 0.0;
  double last_loop_weight_ = 0.0;
  double last_opt_mean_residual_ = 0.0;
  double last_opt_max_step_ = 0.0;
  double last_output_dx_ = 0.0;
  double last_output_dy_ = 0.0;
  double last_output_dz_ = 0.0;
  double last_output_dyaw_deg_ = 0.0;
  int last_opt_iterations_ = 0;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "semantic_keyframe_pose_graph");
  SemanticKeyframePoseGraph node;
  ros::spin();
  return 0;
}
