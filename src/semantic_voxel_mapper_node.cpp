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
#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>

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
    pnh_.param<bool>("publish_static_labeled_cloud", publish_static_labeled_cloud_, true);
    pnh_.param<int>("max_publish_points", max_publish_points_, 800000);
    pnh_.param<int>("max_voxels", max_voxels_, 1800000);
    pnh_.param<double>("voxel_ttl_sec", voxel_ttl_sec_, 0.0);
    pnh_.param<double>("publish_rate", publish_rate_, 2.0);

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

    int used = 0;
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

      Eigen::Vector3d p_map(x, y, z);
      if (!cloud_in_map_frame)
      {
        p_map = T_map_lidar * p_map;
      }
      if (!std::isfinite(p_map.x()) || !std::isfinite(p_map.y()) || !std::isfinite(p_map.z()))
      {
        continue;
      }
      if (p_map.z() < z_min_map_ || p_map.z() > z_max_map_)
      {
        continue;
      }

      if (publish_static_labeled_cloud_ && publish_static_current_cloud && label != LABEL_DYNAMIC)
      {
        OutputPoint pt;
        pt.x = static_cast<float>(p_map.x());
        pt.y = static_cast<float>(p_map.y());
        pt.z = static_cast<float>(p_map.z());
        pt.label = label;
        pt.confidence = static_cast<float>(std::max(0.0, std::min(1.0, confidence)));
        pt.votes = 1.0f;
        static_points.push_back(pt);
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

      VoxelState& state = voxels_[voxelKey(p_map)];
      state.weighted_sum += p_map * w;
      state.weight_sum += w;
      state.votes[label] += static_cast<float>(w);
      state.last_update = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
      state.observations += 1;
      ++used;
    }

    if (publish_static_labeled_cloud_ && publish_static_current_cloud)
    {
      publishPointCloud(static_points, static_labeled_cloud_topic_, static_cloud_pub_, ros::Time::now());
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
                      "semantic fusion input used=%d voxels=%zu unknown_skip=%d dynamic_skip=%d source=%s",
                      used, voxels_.size(), unknown_skipped, dynamic_skipped,
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
    ss << "\"published_points\":" << published_points << ",";
    ss << "\"last_lidar_points\":" << last_lidar_points_ << ",";
    ss << "\"last_image_points\":" << last_image_points_ << ",";
    ss << "\"voxel_size\":" << voxel_size_ << ",";
    ss << "\"min_votes\":" << min_votes_ << ",";
    ss << "\"min_confidence\":" << min_confidence_ << ",";
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
               "Expected PointCloud2 fields: x y z %s [confidence].",
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
  ros::Publisher bev_label_pub_;
  ros::Publisher bev_color_pub_;
  ros::Publisher bev_conf_pub_;
  ros::Publisher bev_traversable_pub_;
  ros::Timer publish_timer_;
  ros::Timer maintenance_timer_;
  ros::WallTimer diagnostic_timer_;

  std::unordered_map<VoxelKey, VoxelState, VoxelKeyHash> voxels_;
  Eigen::Isometry3d latest_T_map_body_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_body_lidar_ = Eigen::Isometry3d::Identity();
  ros::Time latest_odom_time_;

  std::string odom_topic_;
  std::string lidar_cloud_topic_;
  std::string image_cloud_topic_;
  std::string semantic_cloud_topic_;
  std::string stats_topic_;
  std::string static_labeled_cloud_topic_;
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
  bool publish_static_labeled_cloud_ = true;
  bool publish_bev_ = true;
  bool bev_center_on_latest_pose_ = true;

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
  double voxel_ttl_sec_ = 0.0;
  double publish_rate_ = 2.0;
  double bev_resolution_ = 0.20;
  double bev_size_m_ = 100.0;
  int max_publish_points_ = 800000;
  int max_voxels_ = 1800000;
  int last_lidar_points_ = 0;
  int last_image_points_ = 0;
  int lidar_msg_count_ = 0;
  int image_msg_count_ = 0;
  ros::WallTime last_lidar_input_wall_;
  ros::WallTime last_image_input_wall_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "semantic_voxel_mapper");
  SemanticVoxelMapper node;
  ros::spin();
  return 0;
}
