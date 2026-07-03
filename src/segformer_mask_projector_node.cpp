#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>

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
  LABEL_OTHER = 6
};

struct ProjectedPoint
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint32_t label = 0;
  float confidence = 0.0f;
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

bool loadMatrixParam(const ros::NodeHandle& pnh, const std::string& name, Eigen::Matrix4d& T)
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
  for (int r = 0; r < 4; ++r)
  {
    for (int c = 0; c < 4; ++c)
    {
      T(r, c) = values[static_cast<std::size_t>(r * 4 + c)];
    }
  }
  return true;
}

}  // namespace

class SegFormerMaskProjector
{
 public:
  SegFormerMaskProjector() : nh_(), pnh_("~")
  {
    pnh_.param<std::string>("cloud_topic", cloud_topic_, "/rangenet/semantic_points");
    pnh_.param<std::string>("mask_topic", mask_topic_, "/segformer/label_mask");
    pnh_.param<std::string>("camera_info_topic", camera_info_topic_, "/camera/camera_info");
    pnh_.param<std::string>("output_topic", output_topic_, "/segformer/projected_semantic_points");
    pnh_.param<bool>("use_camera_info", use_camera_info_, false);
    pnh_.param<std::string>("mask_label_mode", mask_label_mode_, "cityscapes");
    pnh_.param<double>("fx", fx_, 0.0);
    pnh_.param<double>("fy", fy_, 0.0);
    pnh_.param<double>("cx", cx_, 0.0);
    pnh_.param<double>("cy", cy_, 0.0);
    pnh_.param<double>("d0", d0_, 0.0);
    pnh_.param<double>("d1", d1_, 0.0);
    pnh_.param<double>("d2", d2_, 0.0);
    pnh_.param<double>("d3", d3_, 0.0);
    pnh_.param<bool>("use_distortion", use_distortion_, true);
    pnh_.param<double>("min_project_depth_m", min_project_depth_m_, 0.2);
    pnh_.param<double>("max_mask_age_sec", max_mask_age_sec_, 0.25);
    pnh_.param<int>("point_stride", point_stride_, 1);
    pnh_.param<double>("min_mask_support_ratio", min_mask_support_ratio_, 0.30);
    pnh_.param<int>("support_kernel_px", support_kernel_px_, 5);
    pnh_.param<double>("default_confidence", default_confidence_, 0.65);
    pnh_.param<bool>("drop_unknown", drop_unknown_, true);

    T_cam_lidar_.setIdentity();
    loadMatrixParam(pnh_, "T_cam_lidar", T_cam_lidar_);

    mask_sub_ = nh_.subscribe(mask_topic_, 2, &SegFormerMaskProjector::maskCb, this);
    cloud_sub_ = nh_.subscribe(cloud_topic_, 5, &SegFormerMaskProjector::cloudCb, this);
    if (use_camera_info_)
    {
      camera_info_sub_ = nh_.subscribe(camera_info_topic_, 2, &SegFormerMaskProjector::cameraInfoCb, this);
    }
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 2);
    diagnostic_timer_ = nh_.createWallTimer(ros::WallDuration(3.0), &SegFormerMaskProjector::diagnosticTimerCb, this);

    ROS_INFO("segformer_mask_projector started cloud=%s mask=%s output=%s mode=%s",
             cloud_topic_.c_str(), mask_topic_.c_str(), output_topic_.c_str(), mask_label_mode_.c_str());
  }

 private:
  uint32_t mapMaskLabel(uint8_t raw) const
  {
    if (mask_label_mode_ == "internal")
    {
      return raw <= LABEL_OTHER ? raw : LABEL_UNKNOWN;
    }

    // Cityscapes-style SegFormer output:
    // 0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 8 vegetation, 9 terrain,
    // 11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle.
    switch (raw)
    {
      case 0:
        return LABEL_ROAD;
      case 1:
      case 9:
        return LABEL_SIDEWALK;
      case 2:
      case 3:
      case 4:
        return LABEL_BUILDING;
      case 8:
        return LABEL_VEGETATION;
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
        return LABEL_DYNAMIC;
      default:
        return LABEL_UNKNOWN;
    }
  }

  void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr& msg)
  {
    fx_ = msg->K[0];
    fy_ = msg->K[4];
    cx_ = msg->K[2];
    cy_ = msg->K[5];
    if (msg->D.size() >= 4)
    {
      d0_ = msg->D[0];
      d1_ = msg->D[1];
      d2_ = msg->D[2];
      d3_ = msg->D[3];
    }
  }

  void maskCb(const sensor_msgs::ImageConstPtr& msg)
  {
    try
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
      if (cv_ptr->image.channels() == 1)
      {
        latest_mask_ = cv_ptr->image.clone();
      }
      else
      {
        std::vector<cv::Mat> channels;
        cv::split(cv_ptr->image, channels);
        latest_mask_ = channels.front().clone();
      }
      latest_mask_stamp_ = msg->header.stamp;
      ++mask_msg_count_;
      last_mask_wall_ = ros::WallTime::now();
    }
    catch (const std::exception& e)
    {
      ROS_WARN_THROTTLE(2.0, "Failed reading SegFormer mask: %s", e.what());
    }
  }

  double labelSupport(uint32_t internal_label, int u, int v) const
  {
    if (latest_mask_.empty())
    {
      return 0.0;
    }
    int k = std::max(1, support_kernel_px_);
    if (k % 2 == 0)
    {
      ++k;
    }
    const int r = k / 2;
    int same = 0;
    int total = 0;
    for (int yy = std::max(0, v - r); yy <= std::min(latest_mask_.rows - 1, v + r); ++yy)
    {
      for (int xx = std::max(0, u - r); xx <= std::min(latest_mask_.cols - 1, u + r); ++xx)
      {
        const uint32_t lab = mapMaskLabel(latest_mask_.at<uint8_t>(yy, xx));
        if (lab == internal_label)
        {
          ++same;
        }
        ++total;
      }
    }
    return total > 0 ? static_cast<double>(same) / static_cast<double>(total) : 0.0;
  }

  bool projectPoint(const Eigen::Vector3d& p_lidar, int& u, int& v) const
  {
    const Eigen::Vector4d p_h(p_lidar.x(), p_lidar.y(), p_lidar.z(), 1.0);
    const Eigen::Vector4d p_cam_h = T_cam_lidar_ * p_h;
    const double x = p_cam_h.x();
    const double y = p_cam_h.y();
    const double z = p_cam_h.z();
    if (z <= min_project_depth_m_)
    {
      return false;
    }
    double xn = x / z;
    double yn = y / z;
    if (use_distortion_)
    {
      const double r2 = xn * xn + yn * yn;
      const double r4 = r2 * r2;
      const double radial = 1.0 + d0_ * r2 + d1_ * r4;
      const double xdist = xn * radial + 2.0 * d2_ * xn * yn + d3_ * (r2 + 2.0 * xn * xn);
      const double ydist = yn * radial + d2_ * (r2 + 2.0 * yn * yn) + 2.0 * d3_ * xn * yn;
      xn = xdist;
      yn = ydist;
    }
    u = static_cast<int>(std::lround(fx_ * xn + cx_));
    v = static_cast<int>(std::lround(fy_ * yn + cy_));
    return true;
  }

  void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    ++cloud_msg_count_;
    last_cloud_wall_ = ros::WallTime::now();
    if (latest_mask_.empty())
    {
      ROS_WARN_THROTTLE(2.0, "Waiting for SegFormer mono8 label mask on %s", mask_topic_.c_str());
      return;
    }
    if (fx_ <= 0.0 || fy_ <= 0.0)
    {
      ROS_WARN_THROTTLE(2.0, "Camera intrinsics are not set. Use ~use_camera_info or fx/fy/cx/cy params.");
      return;
    }
    if (!latest_mask_stamp_.isZero() && !msg->header.stamp.isZero())
    {
      const double age = std::fabs((msg->header.stamp - latest_mask_stamp_).toSec());
      if (max_mask_age_sec_ > 0.0 && age > max_mask_age_sec_)
      {
        ROS_WARN_THROTTLE(2.0, "Latest SegFormer mask too old for cloud: age=%.3fs > %.3fs", age, max_mask_age_sec_);
        return;
      }
    }

    const sensor_msgs::PointField* fx_field = findField(*msg, "x");
    const sensor_msgs::PointField* fy_field = findField(*msg, "y");
    const sensor_msgs::PointField* fz_field = findField(*msg, "z");
    if (!fx_field || !fy_field || !fz_field)
    {
      ROS_WARN_THROTTLE(2.0, "Input cloud lacks x/y/z fields");
      return;
    }

    const std::size_t n = static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
    std::vector<ProjectedPoint> out;
    out.reserve(std::min<std::size_t>(n, 200000));
    int projected = 0;
    int labeled = 0;
    const int stride = std::max(1, point_stride_);
    for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(stride))
    {
      const double x = readFieldAsDouble(*msg, *fx_field, i);
      const double y = readFieldAsDouble(*msg, *fy_field, i);
      const double z = readFieldAsDouble(*msg, *fz_field, i);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      {
        continue;
      }
      int u = 0;
      int v = 0;
      if (!projectPoint(Eigen::Vector3d(x, y, z), u, v))
      {
        continue;
      }
      if (u < 0 || u >= latest_mask_.cols || v < 0 || v >= latest_mask_.rows)
      {
        continue;
      }
      ++projected;
      const uint32_t label = mapMaskLabel(latest_mask_.at<uint8_t>(v, u));
      if (label == LABEL_UNKNOWN && drop_unknown_)
      {
        continue;
      }
      const double support = labelSupport(label, u, v);
      if (support < min_mask_support_ratio_)
      {
        continue;
      }
      ProjectedPoint pt;
      pt.x = static_cast<float>(x);
      pt.y = static_cast<float>(y);
      pt.z = static_cast<float>(z);
      pt.label = label;
      pt.confidence = static_cast<float>(std::max(0.05, std::min(1.0, default_confidence_ * support)));
      out.push_back(pt);
      ++labeled;
    }

    sensor_msgs::PointCloud2 out_msg;
    out_msg.header = msg->header;
    out_msg.height = 1;
    out_msg.width = static_cast<uint32_t>(out.size());
    sensor_msgs::PointCloud2Modifier modifier(out_msg);
    modifier.setPointCloud2Fields(
      6,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::PointField::FLOAT32,
      "label", 1, sensor_msgs::PointField::UINT32,
      "confidence", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(out.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(out_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(out_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(out_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(out_msg, "rgb");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_label(out_msg, "label");
    sensor_msgs::PointCloud2Iterator<float> iter_conf(out_msg, "confidence");
    for (const auto& pt : out)
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
    pub_.publish(out_msg);
    ROS_INFO_THROTTLE(3.0, "segformer mask projected points=%d labeled=%d output=%zu", projected, labeled, out.size());
  }

  void diagnosticTimerCb(const ros::WallTimerEvent&)
  {
    const ros::WallTime now = ros::WallTime::now();
    if (cloud_msg_count_ == 0)
    {
      ROS_WARN("segformer_mask_projector has not received input cloud on %s. "
               "Use the same PointCloud2 geometry topic as RangeNet++ output if no separate input cloud exists.",
               cloud_topic_.c_str());
    }
    else if ((now - last_cloud_wall_).toSec() > 3.0)
    {
      ROS_WARN("segformer_mask_projector input cloud stalled: topic=%s last_age=%.1fs",
               cloud_topic_.c_str(), (now - last_cloud_wall_).toSec());
    }

    if (mask_msg_count_ == 0)
    {
      ROS_WARN("segformer_mask_projector has not received SegFormer mono8 mask on %s", mask_topic_.c_str());
    }
    else if ((now - last_mask_wall_).toSec() > 3.0)
    {
      ROS_WARN("segformer_mask_projector mask stalled: topic=%s last_age=%.1fs",
               mask_topic_.c_str(), (now - last_mask_wall_).toSec());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber mask_sub_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber camera_info_sub_;
  ros::Publisher pub_;
  ros::WallTimer diagnostic_timer_;

  std::string cloud_topic_;
  std::string mask_topic_;
  std::string camera_info_topic_;
  std::string output_topic_;
  std::string mask_label_mode_;
  bool use_camera_info_ = false;
  bool use_distortion_ = true;
  bool drop_unknown_ = true;
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  double d0_ = 0.0;
  double d1_ = 0.0;
  double d2_ = 0.0;
  double d3_ = 0.0;
  double min_project_depth_m_ = 0.2;
  double max_mask_age_sec_ = 0.25;
  double min_mask_support_ratio_ = 0.30;
  double default_confidence_ = 0.65;
  int point_stride_ = 1;
  int support_kernel_px_ = 5;
  Eigen::Matrix4d T_cam_lidar_ = Eigen::Matrix4d::Identity();
  cv::Mat latest_mask_;
  ros::Time latest_mask_stamp_;
  int cloud_msg_count_ = 0;
  int mask_msg_count_ = 0;
  ros::WallTime last_cloud_wall_;
  ros::WallTime last_mask_wall_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "segformer_mask_projector");
  SegFormerMaskProjector node;
  ros::spin();
  return 0;
}
