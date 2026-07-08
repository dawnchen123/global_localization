#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <selector.hpp>

namespace cl = rangenet::segmentation;

namespace
{

const sensor_msgs::PointField* findField(const sensor_msgs::PointCloud2& msg,
                                         const std::string& name)
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

uint32_t semanticKittiToInternal(uint32_t label)
{
  switch (label)
  {
    case 40:
    case 44:
    case 49:
      return 1;  // road/ground
    case 48:
    case 72:
      return 2;  // sidewalk/terrain
    case 50:
    case 51:
    case 52:
      return 3;  // building/structure
    case 70:
    case 71:
    case 80:
    case 81:
      return 4;  // vegetation/static vertical small objects
    case 10:
    case 11:
    case 13:
    case 15:
    case 16:
    case 18:
    case 20:
    case 30:
    case 31:
    case 32:
    case 252:
    case 253:
    case 254:
    case 255:
    case 256:
    case 257:
    case 258:
    case 259:
      return 5;  // dynamic
    case 99:
      return 6;
    default:
      return 0;
  }
}

float internalLabelRgb(uint32_t internal_label)
{
  switch (internal_label)
  {
    case 1:
      return rgbFloat(70, 70, 70);
    case 2:
      return rgbFloat(160, 120, 80);
    case 3:
      return rgbFloat(220, 40, 40);
    case 4:
      return rgbFloat(20, 170, 40);
    case 5:
      return rgbFloat(255, 180, 0);
    case 6:
      return rgbFloat(120, 120, 220);
    default:
      return rgbFloat(180, 180, 180);
  }
}

float robustConfidence(const std::vector<float>& scores, int best_idx, float best_score)
{
  if (best_idx < 0 || scores.empty() || !std::isfinite(best_score))
  {
    return 0.0f;
  }

  double sum = 0.0;
  bool all_prob_like = true;
  for (float s : scores)
  {
    if (!std::isfinite(s) || s < -1e-3f || s > 1.0f + 1e-3f)
    {
      all_prob_like = false;
      break;
    }
    sum += static_cast<double>(s);
  }
  if (all_prob_like && sum > 0.5 && sum < 1.5)
  {
    return static_cast<float>(std::max(0.0, std::min(1.0, static_cast<double>(best_score))));
  }

  const float max_logit = best_score;
  double exp_sum = 0.0;
  for (float s : scores)
  {
    if (std::isfinite(s))
    {
      exp_sum += std::exp(static_cast<double>(s - max_logit));
    }
  }
  if (exp_sum <= 1e-9)
  {
    return 0.0f;
  }
  return static_cast<float>(std::max(0.0, std::min(1.0, 1.0 / exp_sum)));
}

uint32_t internalToSemanticKitti(uint32_t internal_label, uint32_t fallback_raw)
{
  switch (internal_label)
  {
    case 1:
      return 40;
    case 2:
      return 48;
    case 3:
      return 50;
    case 4:
      return 70;
    case 6:
      return 99;
    default:
      return fallback_raw;
  }
}

struct CandidatePoint
{
  // x/y/z are always LiDAR-frame coordinates used by RangeNet++ and geometry refinement.
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  // out_* keep the coordinate frame to publish. With /cloud_registered input they remain map-frame.
  float out_x = 0.0f;
  float out_y = 0.0f;
  float out_z = 0.0f;
  float intensity = 0.0f;
  float bx = 0.0f;
  float by = 0.0f;
  float bz = 0.0f;
  bool in_model_fov = true;
};

struct CellStats
{
  int count = 0;
  float min_z = std::numeric_limits<float>::infinity();
  float max_z = -std::numeric_limits<float>::infinity();

  void add(float z)
  {
    ++count;
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }
};

int64_t gridKey(int ix, int iy)
{
  return (static_cast<int64_t>(ix) << 32) ^ static_cast<uint32_t>(iy);
}

}  // namespace

class RangeNetPPRosNode
{
 public:
  RangeNetPPRosNode() : nh_(), pnh_("~")
  {
    pnh_.param<std::string>("input_cloud_topic", input_topic_, "/hesai/at128/points");
    pnh_.param<std::string>("output_cloud_topic", output_topic_, "/rangenet/semantic_points");
    pnh_.param<std::string>("odom_topic", odom_topic_, "/aft_mapped_to_init");
    pnh_.param<std::string>("map_frame", map_frame_, "camera_init");
    pnh_.param<std::string>("model_dir", model_dir_, "/home/dawn/models/rangenetpp");
    pnh_.param<std::string>("backend", backend_, "tensorrt");
    pnh_.param<bool>("input_cloud_in_map_frame", input_cloud_in_map_frame_, false);
    pnh_.param<bool>("output_cloud_in_map_frame", output_cloud_in_map_frame_, false);
    pnh_.param<bool>("restamp_output_with_receipt_time", restamp_output_with_receipt_time_, true);
    pnh_.param<bool>("verbose", verbose_, false);
    pnh_.param<int>("subscriber_queue_size", subscriber_queue_size_, 1);
    pnh_.param<int>("point_stride", point_stride_, 1);
    pnh_.param<int>("max_points", max_points_, 180000);
    pnh_.param<int>("publish_every_n", publish_every_n_, 1);
    pnh_.param<double>("min_inference_interval_sec", min_inference_interval_sec_, 0.0);
    pnh_.param<double>("min_range_m", min_range_m_, 0.3);
    pnh_.param<double>("max_range_m", max_range_m_, 220.0);
    pnh_.param<bool>("enable_pitch_filter", enable_pitch_filter_, true);
    pnh_.param<double>("min_pitch_deg", min_pitch_deg_, -25.0);
    pnh_.param<double>("max_pitch_deg", max_pitch_deg_, 15.0);
    pnh_.param<bool>("preserve_out_of_fov_points", preserve_out_of_fov_points_, true);
    pnh_.param<bool>("require_intensity", require_intensity_, false);
    pnh_.param<double>("default_intensity", default_intensity_, 0.0);
    pnh_.param<double>("intensity_scale", intensity_scale_, 255.0);
    pnh_.param<bool>("clamp_intensity", clamp_intensity_, true);
    pnh_.param<bool>("enable_geometry_refinement", enable_geometry_refinement_, true);
    pnh_.param<double>("geometry_grid_resolution", geometry_grid_resolution_, 0.6);
    pnh_.param<double>("ground_height_threshold", ground_height_threshold_, 0.28);
    pnh_.param<double>("ground_reference_percentile", ground_reference_percentile_, 0.12);
    pnh_.param<double>("ground_reference_max_above", ground_reference_max_above_, 0.75);
    pnh_.param<double>("structure_min_height", structure_min_height_, 0.80);
    pnh_.param<double>("structure_min_vertical_span", structure_min_vertical_span_, 1.10);
    pnh_.param<int>("structure_min_points", structure_min_points_, 5);
    pnh_.param<double>("keep_dynamic_confidence", keep_dynamic_confidence_, 0.80);
    pnh_.param<double>("geometry_confidence", geometry_confidence_, 0.85);
    loadBodyLidarParam();

    if (point_stride_ < 1)
    {
      point_stride_ = 1;
    }
    if (publish_every_n_ < 1)
    {
      publish_every_n_ = 1;
    }
    if (geometry_grid_resolution_ < 0.1)
    {
      geometry_grid_resolution_ = 0.1;
    }
    min_range2_ = min_range_m_ * min_range_m_;
    max_range2_ = max_range_m_ * max_range_m_;
    min_pitch_rad_ = min_pitch_deg_ * M_PI / 180.0;
    max_pitch_rad_ = max_pitch_deg_ * M_PI / 180.0;

    std::vector<std::string> intensity_candidates;
    if (pnh_.getParam("intensity_field_candidates", intensity_candidates) &&
        !intensity_candidates.empty())
    {
      intensity_field_candidates_ = intensity_candidates;
    }

    ROS_INFO("Loading real RangeNet++ model_dir=%s backend=%s", model_dir_.c_str(), backend_.c_str());
    net_ = cl::make_net(model_dir_, backend_);
    net_->verbosity(verbose_);
    label_map_ = net_->getLabelMap();
    ROS_INFO("RangeNet++ loaded labels=%zu input=%s output=%s input_in_map=%s output_in_map=%s restamp=%s map_frame=%s odom=%s stride=%d max_points=%d pitch_filter=%s[%.1f,%.1f]deg preserve_oof=%s intensity_scale=%.3f geometry_refine=%s",
             label_map_.size(), input_topic_.c_str(), output_topic_.c_str(),
             input_cloud_in_map_frame_ ? "true" : "false",
             output_cloud_in_map_frame_ ? "true" : "false",
             restamp_output_with_receipt_time_ ? "true" : "false",
             map_frame_.c_str(), odom_topic_.c_str(),
             point_stride_, max_points_,
             enable_pitch_filter_ ? "true" : "false", min_pitch_deg_, max_pitch_deg_,
             preserve_out_of_fov_points_ ? "true" : "false",
             intensity_scale_, enable_geometry_refinement_ ? "true" : "false");

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
    sub_ = nh_.subscribe(input_topic_, subscriber_queue_size_, &RangeNetPPRosNode::cloudCb, this);
    if (input_cloud_in_map_frame_ || output_cloud_in_map_frame_)
    {
      odom_sub_ = nh_.subscribe(odom_topic_, 200, &RangeNetPPRosNode::odomCb, this);
    }
    diagnostic_timer_ = nh_.createWallTimer(
        ros::WallDuration(3.0), &RangeNetPPRosNode::diagnosticTimerCb, this);
  }

 private:
  using Mat4 = std::array<double, 16>;

  static Mat4 identityMatrix()
  {
    Mat4 T{};
    T[0] = 1.0;
    T[5] = 1.0;
    T[10] = 1.0;
    T[15] = 1.0;
    return T;
  }

  static Mat4 multiplyMatrix(const Mat4& A, const Mat4& B)
  {
    Mat4 C{};
    for (int r = 0; r < 4; ++r)
    {
      for (int c = 0; c < 4; ++c)
      {
        double v = 0.0;
        for (int k = 0; k < 4; ++k)
        {
          v += A[static_cast<std::size_t>(4 * r + k)] * B[static_cast<std::size_t>(4 * k + c)];
        }
        C[static_cast<std::size_t>(4 * r + c)] = v;
      }
    }
    return C;
  }

  static Mat4 invertRigid(const Mat4& T)
  {
    Mat4 I = identityMatrix();
    I[0] = T[0];
    I[1] = T[4];
    I[2] = T[8];
    I[4] = T[1];
    I[5] = T[5];
    I[6] = T[9];
    I[8] = T[2];
    I[9] = T[6];
    I[10] = T[10];

    const double tx = T[3];
    const double ty = T[7];
    const double tz = T[11];
    I[3] = -(I[0] * tx + I[1] * ty + I[2] * tz);
    I[7] = -(I[4] * tx + I[5] * ty + I[6] * tz);
    I[11] = -(I[8] * tx + I[9] * ty + I[10] * tz);
    return I;
  }

  static void transformPoint(const Mat4& T,
                             double x, double y, double z,
                             double& ox, double& oy, double& oz)
  {
    ox = T[0] * x + T[1] * y + T[2] * z + T[3];
    oy = T[4] * x + T[5] * y + T[6] * z + T[7];
    oz = T[8] * x + T[9] * y + T[10] * z + T[11];
  }

  Mat4 bodyLidarMatrix() const
  {
    Mat4 T{};
    for (std::size_t i = 0; i < 16; ++i)
    {
      T[i] = T_body_lidar_[i];
    }
    return T;
  }

  void odomCb(const nav_msgs::Odometry::ConstPtr& msg)
  {
    if (!msg)
    {
      return;
    }

    const auto& q = msg->pose.pose.orientation;
    const auto& t = msg->pose.pose.position;
    double qx = q.x;
    double qy = q.y;
    double qz = q.z;
    double qw = q.w;
    const double qn = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (qn < 1e-12)
    {
      qx = 0.0;
      qy = 0.0;
      qz = 0.0;
      qw = 1.0;
    }
    else
    {
      qx /= qn;
      qy /= qn;
      qz /= qn;
      qw /= qn;
    }

    const double xx = qx * qx;
    const double yy = qy * qy;
    const double zz = qz * qz;
    const double xy = qx * qy;
    const double xz = qx * qz;
    const double yz = qy * qz;
    const double wx = qw * qx;
    const double wy = qw * qy;
    const double wz = qw * qz;

    Mat4 T = identityMatrix();
    T[0] = 1.0 - 2.0 * (yy + zz);
    T[1] = 2.0 * (xy - wz);
    T[2] = 2.0 * (xz + wy);
    T[4] = 2.0 * (xy + wz);
    T[5] = 1.0 - 2.0 * (xx + zz);
    T[6] = 2.0 * (yz - wx);
    T[8] = 2.0 * (xz - wy);
    T[9] = 2.0 * (yz + wx);
    T[10] = 1.0 - 2.0 * (xx + yy);
    T[3] = t.x;
    T[7] = t.y;
    T[11] = t.z;

    std::lock_guard<std::mutex> lock(odom_mutex_);
    latest_T_map_body_ = T;
    latest_odom_stamp_ = msg->header.stamp;
    have_odom_ = true;
  }

  bool latestLidarMapTransforms(Mat4& T_map_lidar, Mat4& T_lidar_map)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!have_odom_)
    {
      return false;
    }
    T_map_lidar = multiplyMatrix(latest_T_map_body_, bodyLidarMatrix());
    T_lidar_map = invertRigid(T_map_lidar);
    return true;
  }

  void loadBodyLidarParam()
  {
    T_body_lidar_[0] = 1.0;
    T_body_lidar_[5] = 1.0;
    T_body_lidar_[10] = 1.0;
    T_body_lidar_[15] = 1.0;
    std::vector<double> values;
    if (!pnh_.getParam("T_body_lidar", values))
    {
      return;
    }
    if (values.size() != 16)
    {
      ROS_WARN("Parameter ~T_body_lidar must contain 16 numbers; got %zu. Keeping identity.", values.size());
      return;
    }
    for (std::size_t i = 0; i < 16; ++i)
    {
      T_body_lidar_[i] = values[i];
    }
  }

  CandidatePoint makeCandidate(float x, float y, float z, float intensity,
                               float out_x, float out_y, float out_z) const
  {
    CandidatePoint p;
    p.x = x;
    p.y = y;
    p.z = z;
    p.out_x = out_x;
    p.out_y = out_y;
    p.out_z = out_z;
    p.intensity = intensity;
    p.bx = static_cast<float>(T_body_lidar_[0] * x + T_body_lidar_[1] * y + T_body_lidar_[2] * z + T_body_lidar_[3]);
    p.by = static_cast<float>(T_body_lidar_[4] * x + T_body_lidar_[5] * y + T_body_lidar_[6] * z + T_body_lidar_[7]);
    p.bz = static_cast<float>(T_body_lidar_[8] * x + T_body_lidar_[9] * y + T_body_lidar_[10] * z + T_body_lidar_[11]);
    return p;
  }

  float normalizeIntensity(double intensity) const
  {
    if (!std::isfinite(intensity))
    {
      intensity = default_intensity_;
    }
    if (intensity_scale_ > 1e-9)
    {
      intensity /= intensity_scale_;
    }
    if (clamp_intensity_)
    {
      intensity = std::max(0.0, std::min(1.0, intensity));
    }
    return static_cast<float>(intensity);
  }

  void buildGeometryStats(const std::vector<CandidatePoint>& points,
                          std::unordered_map<int64_t, CellStats>& cells,
                          float& ground_reference) const
  {
    std::vector<float> zs;
    zs.reserve(points.size());
    for (const CandidatePoint& p : points)
    {
      const int ix = static_cast<int>(std::floor(p.bx / geometry_grid_resolution_));
      const int iy = static_cast<int>(std::floor(p.by / geometry_grid_resolution_));
      cells[gridKey(ix, iy)].add(p.bz);
      zs.push_back(p.bz);
    }

    if (zs.empty())
    {
      ground_reference = 0.0f;
      return;
    }
    const double pct = std::max(0.01, std::min(0.50, ground_reference_percentile_));
    std::size_t idx = static_cast<std::size_t>(pct * static_cast<double>(zs.size()));
    idx = std::min(idx, zs.size() - 1);
    std::nth_element(zs.begin(), zs.begin() + static_cast<std::ptrdiff_t>(idx), zs.end());
    ground_reference = zs[idx];
  }

  CellStats neighborhoodStats(const std::unordered_map<int64_t, CellStats>& cells,
                              const CandidatePoint& p) const
  {
    const int ix0 = static_cast<int>(std::floor(p.bx / geometry_grid_resolution_));
    const int iy0 = static_cast<int>(std::floor(p.by / geometry_grid_resolution_));
    CellStats out;
    for (int dx = -1; dx <= 1; ++dx)
    {
      for (int dy = -1; dy <= 1; ++dy)
      {
        const auto it = cells.find(gridKey(ix0 + dx, iy0 + dy));
        if (it == cells.end())
        {
          continue;
        }
        out.count += it->second.count;
        out.min_z = std::min(out.min_z, it->second.min_z);
        out.max_z = std::max(out.max_z, it->second.max_z);
      }
    }
    if (!std::isfinite(out.min_z) || !std::isfinite(out.max_z))
    {
      out.count = 0;
      out.min_z = p.bz;
      out.max_z = p.bz;
    }
    return out;
  }

  uint32_t refineLabelWithGeometry(uint32_t raw_label,
                                   float confidence,
                                   const CandidatePoint& p,
                                   const CellStats& stats,
                                   float ground_reference,
                                   bool& ground_override,
                                   bool& structure_override) const
  {
    if (!enable_geometry_refinement_)
    {
      return raw_label;
    }

    const uint32_t internal = semanticKittiToInternal(raw_label);
    if (internal == 5 && confidence >= keep_dynamic_confidence_)
    {
      return raw_label;
    }

    const float local_ground = stats.min_z;
    const float height = p.bz - local_ground;
    const float vertical_span = stats.max_z - stats.min_z;
    const bool near_frame_ground = p.bz <= ground_reference + static_cast<float>(ground_reference_max_above_);
    const bool is_ground = height >= -0.08f &&
                           height <= static_cast<float>(ground_height_threshold_) &&
                           near_frame_ground;
    if (is_ground)
    {
      ground_override = raw_label != 40 && raw_label != 44 && raw_label != 48 && raw_label != 49 && raw_label != 72;
      return 40;
    }

    const bool has_vertical_support =
        stats.count >= structure_min_points_ &&
        height >= static_cast<float>(structure_min_height_) &&
        vertical_span >= static_cast<float>(structure_min_vertical_span_);
    if (has_vertical_support &&
        (internal == 0 || internal == 6 || confidence < 0.45f))
    {
      structure_override = raw_label != 50 && raw_label != 51 && raw_label != 52;
      return 50;
    }

    return raw_label;
  }

  void cloudCb(const sensor_msgs::PointCloud2::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    ++input_msg_count_;
    last_input_wall_ = ros::WallTime::now();
    const ros::Time receipt_stamp = ros::Time::now();
    if (!msg)
    {
      return;
    }
    ++frame_count_;
    if ((frame_count_ - 1) % static_cast<uint64_t>(publish_every_n_) != 0)
    {
      return;
    }
    if (min_inference_interval_sec_ > 0.0 && !last_inference_wall_.isZero() &&
        (last_input_wall_ - last_inference_wall_).toSec() < min_inference_interval_sec_)
    {
      return;
    }
    last_inference_wall_ = last_input_wall_;

    const sensor_msgs::PointField* fx = findField(*msg, "x");
    const sensor_msgs::PointField* fy = findField(*msg, "y");
    const sensor_msgs::PointField* fz = findField(*msg, "z");
    if (!fx || !fy || !fz)
    {
      ROS_WARN_THROTTLE(2.0, "RangeNet++ input cloud lacks x/y/z fields");
      return;
    }
    const sensor_msgs::PointField* fint = findFirstField(*msg, intensity_field_candidates_);
    if (!fint && require_intensity_)
    {
      ROS_WARN_THROTTLE(2.0, "RangeNet++ input cloud lacks intensity/reflectivity field");
      return;
    }

    const std::size_t total_points = static_cast<std::size_t>(msg->width) *
                                     static_cast<std::size_t>(msg->height);
    if (total_points == 0 || msg->point_step == 0 || msg->data.empty())
    {
      ROS_WARN_THROTTLE(2.0, "RangeNet++ received empty cloud");
      return;
    }

    std::vector<CandidatePoint> candidates;
    candidates.reserve(std::min<std::size_t>(total_points, static_cast<std::size_t>(max_points_)));
    std::vector<float> scan;
    scan.reserve(4 * std::min<std::size_t>(total_points, static_cast<std::size_t>(max_points_)));
    std::vector<uint32_t> inference_candidate_indices;
    inference_candidate_indices.reserve(std::min<std::size_t>(total_points, static_cast<std::size_t>(max_points_)));
    uint32_t pitch_out_of_fov = 0;
    Mat4 T_map_lidar = identityMatrix();
    Mat4 T_lidar_map = identityMatrix();
    if ((input_cloud_in_map_frame_ || output_cloud_in_map_frame_) &&
        !latestLidarMapTransforms(T_map_lidar, T_lidar_map))
    {
      ROS_WARN_THROTTLE(2.0,
                        "RangeNet++ needs odometry on %s for input/output map-frame conversion "
                        "(input_cloud_in_map_frame=%s output_cloud_in_map_frame=%s).",
                        odom_topic_.c_str(),
                        input_cloud_in_map_frame_ ? "true" : "false",
                        output_cloud_in_map_frame_ ? "true" : "false");
      return;
    }
    for (std::size_t i = 0; i < total_points; i += static_cast<std::size_t>(point_stride_))
    {
      if (max_points_ > 0 && candidates.size() >= static_cast<std::size_t>(max_points_))
      {
        break;
      }

      const double in_x = readFieldAsDouble(*msg, *fx, i);
      const double in_y = readFieldAsDouble(*msg, *fy, i);
      const double in_z = readFieldAsDouble(*msg, *fz, i);
      if (!std::isfinite(in_x) || !std::isfinite(in_y) || !std::isfinite(in_z))
      {
        continue;
      }
      double x = in_x;
      double y = in_y;
      double z = in_z;
      if (input_cloud_in_map_frame_)
      {
        transformPoint(T_lidar_map, in_x, in_y, in_z, x, y, z);
      }
      const double r2 = x * x + y * y + z * z;
      if (r2 < min_range2_ || r2 > max_range2_)
      {
        continue;
      }
      bool in_model_fov = true;
      if (enable_pitch_filter_)
      {
        const double range = std::sqrt(r2);
        if (range > 1e-6)
        {
          const double pitch = std::asin(std::max(-1.0, std::min(1.0, z / range)));
          if (pitch < min_pitch_rad_ || pitch > max_pitch_rad_)
          {
            in_model_fov = false;
            ++pitch_out_of_fov;
            if (!preserve_out_of_fov_points_)
            {
              continue;
            }
          }
        }
      }

      double intensity = default_intensity_;
      if (fint)
      {
        intensity = readFieldAsDouble(*msg, *fint, i);
      }
      double out_x = in_x;
      double out_y = in_y;
      double out_z = in_z;
      if (output_cloud_in_map_frame_)
      {
        if (input_cloud_in_map_frame_)
        {
          out_x = in_x;
          out_y = in_y;
          out_z = in_z;
        }
        else
        {
          transformPoint(T_map_lidar, x, y, z, out_x, out_y, out_z);
        }
      }
      else if (input_cloud_in_map_frame_)
      {
        out_x = x;
        out_y = y;
        out_z = z;
      }

      const CandidatePoint candidate = makeCandidate(static_cast<float>(x),
                                                     static_cast<float>(y),
                                                     static_cast<float>(z),
                                                     normalizeIntensity(intensity),
                                                     static_cast<float>(out_x),
                                                     static_cast<float>(out_y),
                                                     static_cast<float>(out_z));
      const uint32_t candidate_index = static_cast<uint32_t>(candidates.size());
      candidates.push_back(candidate);
      candidates.back().in_model_fov = in_model_fov;
      if (in_model_fov)
      {
        scan.push_back(candidate.x);
        scan.push_back(candidate.y);
        scan.push_back(candidate.z);
        scan.push_back(candidate.intensity);
        inference_candidate_indices.push_back(candidate_index);
      }
    }

    const uint32_t inference_n = static_cast<uint32_t>(scan.size() / 4);
    const uint32_t output_n = static_cast<uint32_t>(candidates.size());
    if (output_n == 0)
    {
      ROS_WARN_THROTTLE(2.0, "RangeNet++ cloud became empty after filtering");
      return;
    }
    std::unordered_map<int64_t, CellStats> geometry_cells;
    float ground_reference = 0.0f;
    if (enable_geometry_refinement_)
    {
      buildGeometryStats(candidates, geometry_cells, ground_reference);
    }

    std::vector<std::vector<float>> scores;
    if (inference_n > 0)
    {
      try
      {
        scores = net_->infer(scan, inference_n);
      }
      catch (const std::exception& ex)
      {
        ROS_ERROR_THROTTLE(1.0, "RangeNet++ inference failed: %s", ex.what());
        return;
      }
      if (scores.size() != inference_n)
      {
        ROS_ERROR_THROTTLE(1.0, "RangeNet++ output size mismatch scores=%zu points=%u", scores.size(), inference_n);
        return;
      }
    }

    std::vector<int32_t> candidate_score_index(candidates.size(), -1);
    for (std::size_t i = 0; i < inference_candidate_indices.size(); ++i)
    {
      const uint32_t candidate_index = inference_candidate_indices[i];
      if (candidate_index < candidate_score_index.size())
      {
        candidate_score_index[candidate_index] = static_cast<int32_t>(i);
      }
    }

    sensor_msgs::PointCloud2 out;
    out.header = msg->header;
    if (restamp_output_with_receipt_time_)
    {
      out.header.stamp = receipt_stamp;
    }
    if (output_cloud_in_map_frame_)
    {
      out.header.frame_id = map_frame_;
    }
    out.height = 1;
    out.is_bigendian = false;
    out.is_dense = false;

    sensor_msgs::PointCloud2Modifier modifier(out);
    modifier.setPointCloud2Fields(
        7,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::PointField::FLOAT32,
        "label", 1, sensor_msgs::PointField::UINT32,
        "confidence", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(output_n);

    sensor_msgs::PointCloud2Iterator<float> iter_x(out, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(out, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(out, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_i(out, "intensity");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(out, "rgb");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_label(out, "label");
    sensor_msgs::PointCloud2Iterator<float> iter_conf(out, "confidence");

    uint32_t road = 0;
    uint32_t sidewalk = 0;
    uint32_t building = 0;
    uint32_t vegetation = 0;
    uint32_t dynamic = 0;
    uint32_t other = 0;
    uint32_t unknown = 0;
    uint32_t ground_overrides = 0;
    uint32_t structure_overrides = 0;
    uint32_t geometry_only = 0;
    for (uint32_t i = 0; i < output_n; ++i)
    {
      uint32_t raw_label = 0;
      float confidence = 0.0f;
      const int32_t score_index = candidate_score_index[i];
      if (score_index >= 0)
      {
        const std::vector<float>& s = scores[static_cast<std::size_t>(score_index)];
        int best_idx = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < s.size(); ++j)
        {
          if (std::isfinite(s[j]) && s[j] > best_score)
          {
            best_score = s[j];
            best_idx = static_cast<int>(j);
          }
        }

        if (best_idx >= 0 && static_cast<std::size_t>(best_idx) < label_map_.size())
        {
          raw_label = static_cast<uint32_t>(std::max(0, label_map_[static_cast<std::size_t>(best_idx)]));
        }
        confidence = robustConfidence(s, best_idx, best_score);
      }
      else
      {
        ++geometry_only;
      }

      bool ground_override = false;
      bool structure_override = false;
      if (enable_geometry_refinement_)
      {
        const CellStats stats = neighborhoodStats(geometry_cells, candidates[i]);
        raw_label = refineLabelWithGeometry(raw_label, confidence, candidates[i], stats, ground_reference,
                                            ground_override, structure_override);
        if (ground_override || structure_override)
        {
          confidence = std::max(confidence, static_cast<float>(geometry_confidence_));
        }
      }
      if (ground_override)
      {
        ++ground_overrides;
      }
      if (structure_override)
      {
        ++structure_overrides;
      }
      const uint32_t internal = semanticKittiToInternal(raw_label);
      switch (internal)
      {
        case 1:
          ++road;
          break;
        case 2:
          ++sidewalk;
          break;
        case 3:
          ++building;
          break;
        case 4:
          ++vegetation;
          break;
        case 5:
          ++dynamic;
          break;
        case 6:
          ++other;
          break;
        default:
          ++unknown;
          break;
      }

      *iter_x = candidates[i].out_x;
      *iter_y = candidates[i].out_y;
      *iter_z = candidates[i].out_z;
      *iter_i = candidates[i].intensity;
      *iter_rgb = internalLabelRgb(internal);
      *iter_label = raw_label;
      *iter_conf = confidence;

      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_i;
      ++iter_rgb;
      ++iter_label;
      ++iter_conf;
    }

    pub_.publish(out);
    ++output_msg_count_;
    last_output_wall_ = ros::WallTime::now();
    ROS_INFO_THROTTLE(2.0,
                      "real RangeNet++ output=%u infer=%u geom_only=%u road=%u sidewalk=%u building=%u vegetation=%u dynamic=%u other=%u unknown=%u pitch_oof=%u geom_ground=%u geom_structure=%u frame=%s input_in_map=%s output_in_map=%s",
                      output_n, inference_n, geometry_only,
                      road, sidewalk, building, vegetation, dynamic, other, unknown,
                      pitch_out_of_fov, ground_overrides, structure_overrides,
                      out.header.frame_id.c_str(),
                      input_cloud_in_map_frame_ ? "true" : "false",
                      output_cloud_in_map_frame_ ? "true" : "false");
  }

  void diagnosticTimerCb(const ros::WallTimerEvent&)
  {
    const ros::WallTime now = ros::WallTime::now();
    if (input_msg_count_ == 0)
    {
      ROS_WARN("rangenetpp_ros_node has not received input PointCloud2 on %s. "
               "Current i2Nav AT128 setup expects raw /hesai/at128/points with input_cloud_in_map_frame=false. "
               "Do not use /cloud_registered for RangeNet++ inference unless you intentionally convert map-frame points back to LiDAR frame. "
               "For legacy Livox CustomMsg, start livox_custom_to_pointcloud2_node first.",
               input_topic_.c_str());
      return;
    }
    if ((now - last_input_wall_).toSec() > 3.0)
    {
      ROS_WARN("rangenetpp_ros_node input stalled: topic=%s last_age=%.1fs",
               input_topic_.c_str(), (now - last_input_wall_).toSec());
    }
    if (output_msg_count_ == 0)
    {
      ROS_WARN("rangenetpp_ros_node received %llu input clouds but has not published %s. "
               "Check x/y/z fields, range filters, and TensorRT inference errors above.",
               static_cast<unsigned long long>(input_msg_count_), output_topic_.c_str());
    }
    else if ((now - last_output_wall_).toSec() > 3.0)
    {
      ROS_WARN("rangenetpp_ros_node output stalled: topic=%s last_age=%.1fs",
               output_topic_.c_str(), (now - last_output_wall_).toSec());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher pub_;
  ros::WallTimer diagnostic_timer_;
  std::unique_ptr<cl::Net> net_;
  std::vector<int> label_map_;
  std::vector<std::string> intensity_field_candidates_{"intensity", "reflectivity", "remission"};
  std::string input_topic_;
  std::string output_topic_;
  std::string odom_topic_;
  std::string map_frame_ = "camera_init";
  std::string model_dir_;
  std::string backend_;
  bool input_cloud_in_map_frame_ = false;
  bool output_cloud_in_map_frame_ = false;
  bool restamp_output_with_receipt_time_ = true;
  bool verbose_ = false;
  bool require_intensity_ = false;
  int subscriber_queue_size_ = 1;
  int point_stride_ = 1;
  int max_points_ = 180000;
  int publish_every_n_ = 1;
  double min_range_m_ = 0.3;
  double max_range_m_ = 220.0;
  double min_pitch_deg_ = -25.0;
  double max_pitch_deg_ = 15.0;
  double min_inference_interval_sec_ = 0.0;
  double intensity_scale_ = 255.0;
  double geometry_grid_resolution_ = 0.6;
  double ground_height_threshold_ = 0.28;
  double ground_reference_percentile_ = 0.12;
  double ground_reference_max_above_ = 0.75;
  double structure_min_height_ = 0.80;
  double structure_min_vertical_span_ = 1.10;
  double keep_dynamic_confidence_ = 0.80;
  double geometry_confidence_ = 0.85;
  double min_range2_ = 0.09;
  double max_range2_ = 14400.0;
  double min_pitch_rad_ = -25.0 * M_PI / 180.0;
  double max_pitch_rad_ = 15.0 * M_PI / 180.0;
  double default_intensity_ = 0.0;
  bool enable_pitch_filter_ = true;
  bool preserve_out_of_fov_points_ = true;
  bool clamp_intensity_ = true;
  bool enable_geometry_refinement_ = true;
  int structure_min_points_ = 5;
  double T_body_lidar_[16] = {0.0};
  Mat4 latest_T_map_body_ = identityMatrix();
  ros::Time latest_odom_stamp_;
  bool have_odom_ = false;
  uint64_t frame_count_ = 0;
  uint64_t input_msg_count_ = 0;
  uint64_t output_msg_count_ = 0;
  ros::WallTime last_input_wall_;
  ros::WallTime last_output_wall_;
  ros::WallTime last_inference_wall_;
  std::mutex inference_mutex_;
  std::mutex odom_mutex_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rangenetpp_ros_node");
  try
  {
    RangeNetPPRosNode node;
    ros::spin();
  }
  catch (const std::exception& ex)
  {
    ROS_FATAL("Failed to start real RangeNet++ ROS node: %s", ex.what());
    return 1;
  }
  return 0;
}
