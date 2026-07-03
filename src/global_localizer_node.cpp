#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <cstring>

namespace {
constexpr double kEarthRadius = 6378137.0;

double wrapAngle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

double yawFromQuat(const geometry_msgs::Quaternion &q_msg)
{
  const double n2 = q_msg.x*q_msg.x + q_msg.y*q_msg.y + q_msg.z*q_msg.z + q_msg.w*q_msg.w;
  if (!std::isfinite(n2) || n2 < 1e-12) {
    return 0.0;
  }
  tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
  q.normalize();
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  if (!std::isfinite(yaw)) return 0.0;
  return yaw;
}

bool isFiniteState(const Eigen::Vector3d &x)
{
  return std::isfinite(x(0)) && std::isfinite(x(1)) && std::isfinite(x(2));
}

std::string listPointCloudFields(const sensor_msgs::PointCloud2 &msg)
{
  std::ostringstream oss;
  for (size_t i = 0; i < msg.fields.size(); ++i) {
    oss << msg.fields[i].name << "(off=" << msg.fields[i].offset << ",type=" << int(msg.fields[i].datatype) << ")";
    if (i + 1 < msg.fields.size()) oss << ", ";
  }
  return oss.str();
}

std::string pointCloudBasicInfo(const sensor_msgs::PointCloud2 &msg)
{
  std::ostringstream oss;
  oss << "width=" << msg.width
      << ", height=" << msg.height
      << ", point_step=" << msg.point_step
      << ", row_step=" << msg.row_step
      << ", data_size=" << msg.data.size()
      << ", frame=" << msg.header.frame_id;
  return oss.str();
}

const sensor_msgs::PointField* findField(const sensor_msgs::PointCloud2 &msg, const std::string &name)
{
  for (const auto &f : msg.fields) {
    if (f.name == name) return &f;
  }
  return nullptr;
}


bool hasValidXYZCloud(const sensor_msgs::PointCloud2 &msg)
{
  return msg.width > 0 && msg.height > 0 && msg.point_step > 0 && !msg.data.empty() &&
         findField(msg, "x") && findField(msg, "y") && findField(msg, "z");
}

double readPointFieldAsDouble(const uint8_t *base, const sensor_msgs::PointField &f)
{
  const uint8_t *ptr = base + f.offset;
  switch (f.datatype) {
    case sensor_msgs::PointField::FLOAT32: { float v; std::memcpy(&v, ptr, sizeof(float)); return static_cast<double>(v); }
    case sensor_msgs::PointField::FLOAT64: { double v; std::memcpy(&v, ptr, sizeof(double)); return v; }
    case sensor_msgs::PointField::UINT8:   { uint8_t v; std::memcpy(&v, ptr, sizeof(uint8_t)); return static_cast<double>(v); }
    case sensor_msgs::PointField::INT8:    { int8_t v; std::memcpy(&v, ptr, sizeof(int8_t)); return static_cast<double>(v); }
    case sensor_msgs::PointField::UINT16:  { uint16_t v; std::memcpy(&v, ptr, sizeof(uint16_t)); return static_cast<double>(v); }
    case sensor_msgs::PointField::INT16:   { int16_t v; std::memcpy(&v, ptr, sizeof(int16_t)); return static_cast<double>(v); }
    case sensor_msgs::PointField::UINT32:  { uint32_t v; std::memcpy(&v, ptr, sizeof(uint32_t)); return static_cast<double>(v); }
    case sensor_msgs::PointField::INT32:   { int32_t v; std::memcpy(&v, ptr, sizeof(int32_t)); return static_cast<double>(v); }
    default: return std::numeric_limits<double>::quiet_NaN();
  }
}

bool pointCloud2ToXYZI(const sensor_msgs::PointCloud2 &msg, pcl::PointCloud<pcl::PointXYZI> &cloud)
{
  cloud.clear();
  const sensor_msgs::PointField *fx = findField(msg, "x");
  const sensor_msgs::PointField *fy = findField(msg, "y");
  const sensor_msgs::PointField *fz = findField(msg, "z");
  const sensor_msgs::PointField *fi = findField(msg, "intensity");
  if (msg.width == 0 || msg.height == 0 || msg.point_step == 0 || msg.data.empty() || msg.fields.empty()) {
    ROS_WARN_THROTTLE(5.0, "Input cloud is an empty PointCloud2 or has no fields; waiting for a valid FAST-LIVO2 output. basic=[%s], fields=[%s]", pointCloudBasicInfo(msg).c_str(), listPointCloudFields(msg).c_str());
    return false;
  }
  if (!fx || !fy || !fz) {
    ROS_ERROR_THROTTLE(5.0, "Input cloud has no x/y/z fields. You are probably subscribing to the wrong topic. basic=[%s], fields=[%s]", pointCloudBasicInfo(msg).c_str(), listPointCloudFields(msg).c_str());
    return false;
  }
  const size_t n = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
  cloud.reserve(n);
  for (size_t idx = 0; idx < n; ++idx) {
    const uint8_t *base = &msg.data[0] + idx * msg.point_step;
    double x = readPointFieldAsDouble(base, *fx);
    double y = readPointFieldAsDouble(base, *fy);
    double z = readPointFieldAsDouble(base, *fz);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
    pcl::PointXYZI p;
    p.x = static_cast<float>(x);
    p.y = static_cast<float>(y);
    p.z = static_cast<float>(z);
    p.intensity = 1.0f;
    if (fi) {
      double inten = readPointFieldAsDouble(base, *fi);
      if (std::isfinite(inten)) p.intensity = static_cast<float>(inten);
    }
    cloud.push_back(p);
  }
  if (cloud.empty()) {
    ROS_WARN_THROTTLE(2.0, "Converted point cloud is empty after finite checks. fields=[%s]", listPointCloudFields(msg).c_str());
    return false;
  }
  return true;
}

geometry_msgs::Quaternion quatFromYaw(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(q);
}

bool readSimpleYamlValue(const std::string &path, const std::string &key, std::string &out)
{
  std::ifstream fin(path.c_str());
  if (!fin.is_open()) return false;
  std::string line;
  while (std::getline(fin, line)) {
    std::string k = key + ":";
    auto pos = line.find(k);
    if (pos == std::string::npos) continue;
    std::string value = line.substr(pos + k.size());
    value.erase(0, value.find_first_not_of(" \t\"'"));
    value.erase(value.find_last_not_of(" \t\"'\r") + 1);
    out = value;
    return true;
  }
  return false;
}

bool readSimpleYamlDouble(const std::string &path, const std::string &key, double &out)
{
  std::string s;
  if (!readSimpleYamlValue(path, key, s)) return false;
  try { out = std::stod(s); return true; } catch (...) { return false; }
}

bool readSimpleYamlInt(const std::string &path, const std::string &key, int &out)
{
  std::string s;
  if (!readSimpleYamlValue(path, key, s)) return false;
  try { out = std::stoi(s); return true; } catch (...) { return false; }
}

uint64_t gridKey(int ix, int iy)
{
  return (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) | static_cast<uint32_t>(iy);
}


cv::Mat normalizeFloatTo8U(const cv::Mat &src, double out_min = 0.0, double out_max = 255.0)
{
  if (src.empty()) return cv::Mat();
  double mn = 0.0, mx = 0.0;
  cv::minMaxLoc(src, &mn, &mx);
  cv::Mat dst(src.size(), CV_8UC1, cv::Scalar(static_cast<uchar>(out_min)));
  if (mx - mn < 1e-9) return dst;
  cv::Mat tmp;
  src.convertTo(tmp, CV_32F);
  tmp = (tmp - mn) / (mx - mn);
  tmp = tmp * (out_max - out_min) + out_min;
  tmp.convertTo(dst, CV_8UC1);
  return dst;
}

cv::Mat applyClahe8U(const cv::Mat &gray, double clip_limit = 2.0, int grid_size = 8)
{
  cv::Mat g;
  if (gray.channels() == 3) cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
  else g = gray.clone();
  if (g.type() != CV_8UC1) g.convertTo(g, CV_8UC1);
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(grid_size, grid_size));
  cv::Mat out;
  clahe->apply(g, out);
  return out;
}

cv::Mat makeSatelliteStructureImage(const cv::Mat &bgr, const cv::Size &target_size)
{
  cv::Mat gray, eq, blur, edges, grad_x, grad_y, mag, mag8, structure;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  eq = applyClahe8U(gray, 2.0, 8);
  cv::GaussianBlur(eq, blur, cv::Size(3,3), 0.0);
  cv::Canny(blur, edges, 45, 130);
  cv::dilate(edges, edges, cv::Mat(), cv::Point(-1,-1), 1);
  cv::Sobel(blur, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(blur, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, mag);
  mag8 = normalizeFloatTo8U(mag, 0.0, 255.0);
  // Pseudo structure image: retains low-frequency satellite context and strengthens roof/road/vegetation boundaries.
  cv::addWeighted(eq, 0.55, mag8, 0.25, 0.0, structure);
  cv::addWeighted(structure, 0.80, edges, 0.20, 0.0, structure);
  if (target_size.width > 0 && target_size.height > 0 && structure.size() != target_size) {
    cv::resize(structure, structure, target_size, 0, 0, cv::INTER_AREA);
  }
  return structure;
}
}

struct BevResult
{
  cv::Mat vis;          // BGR pseudo image for visualization
  cv::Mat match;        // grayscale multi-layer BEV pseudo image used for matching
  cv::Mat occ_layer;    // obstacle/non-ground layer
  cv::Mat density_layer;// density layer
  cv::Mat height_layer; // height-difference layer
  cv::Mat ground_layer; // near-ground/traversable observation layer
  bool ok = false;
  double min_e = 0.0;   // local east offset relative to robot, meters, north-up
  double max_e = 0.0;
  double min_n = 0.0;   // local north offset relative to robot, meters, north-up
  double max_n = 0.0;
  int obstacle_points = 0;
};

struct SatCropResult
{
  cv::Mat crop_bgr;     // original satellite crop
  cv::Mat crop_match;   // satellite structure pseudo image resized to BEV match image size
  bool ok = false;
  double abs_min_e = 0.0;
  double abs_max_e = 0.0;
  double abs_min_n = 0.0;
  double abs_max_n = 0.0;
};

struct CloudFrame
{
  ros::Time stamp;
  pcl::PointCloud<pcl::PointXYZI> cloud_in_livo_map;
  std::string frame_id;
};

class GlobalLocalizerNode
{
public:
  GlobalLocalizerNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh), it_(nh_)
  {
    pnh_.param<std::string>("fast_livo_odom_topic", fast_livo_odom_topic_, "/Odometry");
    pnh_.param<std::string>("fast_livo_cloud_topic", fast_livo_cloud_topic_, "/cloud_registered");
    pnh_.param<std::string>("gps_topic", gps_topic_, "/fix");
    pnh_.param<std::string>("imu_topic", imu_topic_, "/imu");
    pnh_.param<std::string>("satellite_yaml", satellite_yaml_, std::string("/tmp/fast_livo2_global_localization/satellite_mosaic.yml"));
    pnh_.param<std::string>("earth_frame", earth_frame_, "earth");
    pnh_.param<std::string>("base_frame", base_frame_, "base_link");
    // body: cloud points are in robot/body frame. livo_map: cloud points are in FAST-LIVO2 odometry map frame.
    pnh_.param<std::string>("cloud_frame_mode", cloud_frame_mode_, "livo_map");
    pnh_.param<double>("cloud_accumulation_window", cloud_accumulation_window_, 20.0);
    pnh_.param<int>("cloud_accumulation_max_frames", cloud_accumulation_max_frames_, 200);
    pnh_.param<int>("cloud_accumulation_stride", cloud_accumulation_stride_, 1);
    pnh_.param<double>("cloud_accumulation_voxel_size", cloud_accumulation_voxel_size_, 0.20);
    pnh_.param<std::string>("matcher_backend", matcher_backend_, "external");
    pnh_.param<std::string>("external_match_pose_topic", external_match_pose_topic_, "/map_match_pose_external");
    pnh_.param<double>("bev_resolution", bev_resolution_, 0.20);
    pnh_.param<double>("bev_min_size_m", bev_min_size_m_, 80.0);
    pnh_.param<double>("bev_max_size_m", bev_max_size_m_, 220.0);
    pnh_.param<double>("bev_padding_m", bev_padding_m_, 30.0);
    pnh_.param<double>("sat_crop_padding_m", sat_crop_padding_m_, 20.0);
    pnh_.param<double>("max_point_range", max_point_range_, 120.0);
    pnh_.param<double>("z_min", z_min_, -3.0);
    pnh_.param<double>("z_max", z_max_, 5.0);
    pnh_.param<double>("ground_grid_size", ground_grid_size_, 1.0);
    pnh_.param<double>("ground_remove_height", ground_remove_height_, 0.28);
    pnh_.param<double>("bev_height_clip", bev_height_clip_, 4.0);
    pnh_.param<double>("bev_density_weight", bev_density_weight_, 0.30);
    pnh_.param<double>("bev_height_weight", bev_height_weight_, 0.30);
    pnh_.param<double>("bev_obstacle_weight", bev_obstacle_weight_, 0.25);
    pnh_.param<double>("bev_ground_weight", bev_ground_weight_, 0.15);
    pnh_.param<bool>("bev_use_multilayer", bev_use_multilayer_, true);
    pnh_.param<int>("bev_dilate_iter", bev_dilate_iter_, 2);
    pnh_.param<double>("ekf_process_xy", q_xy_, 0.08);
    pnh_.param<double>("ekf_process_yaw", q_yaw_, 0.02);
    pnh_.param<double>("gps_noise_std", gps_noise_std_, 2.5);
    pnh_.param<double>("map_match_noise_std", map_match_noise_std_, 1.2);
    pnh_.param<double>("map_match_min_inliers", map_match_min_inliers_, 18.0);
    pnh_.param<double>("map_match_min_confidence", map_match_min_confidence_, 0.45);
    pnh_.param<double>("map_match_min_inlier_ratio", map_match_min_inlier_ratio_, 0.25);
    pnh_.param<double>("map_match_max_correction", map_match_max_correction_, 25.0);
    pnh_.param<double>("map_match_max_gps_disagreement", map_match_max_gps_disagreement_, 20.0);
    pnh_.param<bool>("use_gps_consistency_gate", use_gps_consistency_gate_, true);
    pnh_.param<bool>("publish_rejected_match_debug", publish_rejected_match_debug_, true);
    pnh_.param<double>("match_rate", match_rate_, 1.0);
    pnh_.param<bool>("use_gps_update", use_gps_update_, true);
    pnh_.param<bool>("publish_tf", publish_tf_, false);
    pnh_.param<bool>("init_yaw_from_imu", init_yaw_from_imu_, true);

    state_.setZero();
    P_.setIdentity();
    P_ *= 100.0;

    sub_odom_ = nh_.subscribe(fast_livo_odom_topic_, 200, &GlobalLocalizerNode::odomCallback, this);
    sub_cloud_ = nh_.subscribe(fast_livo_cloud_topic_, 20, &GlobalLocalizerNode::cloudCallback, this);
    sub_gps_ = nh_.subscribe(gps_topic_, 50, &GlobalLocalizerNode::gpsCallback, this);
    sub_imu_ = nh_.subscribe(imu_topic_, 100, &GlobalLocalizerNode::imuCallback, this);
    sub_external_match_ = nh_.subscribe(external_match_pose_topic_, 20, &GlobalLocalizerNode::externalMapMatchCallback, this);

    pub_fused_odom_ = nh_.advertise<nav_msgs::Odometry>("/global_fused_odom", 50);
    pub_map_match_pose_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/map_match_pose", 10);
    pub_global_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_cloud", 5);
    pub_sat_image_ = it_.advertise("/satellite_map/image", 1, true);
    pub_sat_crop_ = it_.advertise("/satellite_map/aligned_crop", 5);
    pub_sat_match_ = it_.advertise("/satellite_map/match_image", 5);
    pub_bev_image_ = it_.advertise("/local_bev/image", 5);
    pub_bev_match_ = it_.advertise("/local_bev/match_image", 5);
    pub_match_debug_ = it_.advertise("/map_match/debug_image", 5);
    pub_match_input_meta_ = nh_.advertise<std_msgs::Float64MultiArray>("/map_match/input_meta", 5);

    timer_match_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.1, match_rate_)), &GlobalLocalizerNode::mapMatchTimer, this);
    timer_sat_reload_ = nh_.createTimer(ros::Duration(2.0), &GlobalLocalizerNode::satelliteReloadTimer, this);

    ROS_INFO("global_localizer_node structured-prior started. FAST-LIVO2 odom=%s cloud=%s", fast_livo_odom_topic_.c_str(), fast_livo_cloud_topic_.c_str());
  }

private:
  void odomCallback(const nav_msgs::OdometryConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    double ox = msg->pose.pose.position.x;
    double oy = msg->pose.pose.position.y;
    double oyaw = yawFromQuat(msg->pose.pose.orientation);
    if (!std::isfinite(ox) || !std::isfinite(oy) || !std::isfinite(oyaw)) {
      ROS_WARN_THROTTLE(2.0, "Skip FAST-LIVO2 odom with non-finite pose/yaw.");
      return;
    }
    if (has_state_ && !isFiniteState(state_)) {
      ROS_ERROR("EKF state became non-finite before odom prediction. Resetting state to zero translation and current yaw.");
      state_ << 0.0, 0.0, oyaw;
      P_.setIdentity();
      P_ *= 100.0;
    }

    if (!has_last_odom_) {
      last_odom_x_ = ox;
      last_odom_y_ = oy;
      last_odom_yaw_ = oyaw;
      has_last_odom_ = true;
      if (!has_state_) {
        double init_yaw = (init_yaw_from_imu_ && has_imu_) ? latest_imu_yaw_ : oyaw;
        state_ << 0.0, 0.0, init_yaw;
        has_state_ = true;
      }
      publishFusedOdom(msg->header.stamp);
      return;
    }

    double dx_o = ox - last_odom_x_;
    double dy_o = oy - last_odom_y_;
    double dyaw = wrapAngle(oyaw - last_odom_yaw_);
    last_odom_x_ = ox;
    last_odom_y_ = oy;
    last_odom_yaw_ = oyaw;

    double c = std::cos(state_(2));
    double s = std::sin(state_(2));
    state_(0) += c * dx_o - s * dy_o;
    state_(1) += s * dx_o + c * dy_o;
    state_(2) = wrapAngle(state_(2) + dyaw);

    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    double step = std::sqrt(dx_o * dx_o + dy_o * dy_o);
    Q(0,0) = q_xy_ * q_xy_ * (1.0 + step);
    Q(1,1) = q_xy_ * q_xy_ * (1.0 + step);
    Q(2,2) = q_yaw_ * q_yaw_ * (1.0 + std::abs(dyaw));
    P_ += Q;
    publishFusedOdom(msg->header.stamp);
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!hasValidXYZCloud(*msg)) {
      ROS_WARN_THROTTLE(5.0,
        "Received invalid/empty cloud on fast_livo_cloud_topic. Keep previous valid accumulated cloud and skip this frame. basic=[%s], fields=[%s]",
        pointCloudBasicInfo(*msg).c_str(), listPointCloudFields(*msg).c_str());
      return;
    }
    pcl::PointCloud<pcl::PointXYZI> cloud;
    if (!pointCloud2ToXYZI(*msg, cloud)) return;
    if (cloud.empty()) return;

    ++cloud_frame_count_;
    if (cloud_accumulation_stride_ > 1 && (cloud_frame_count_ % cloud_accumulation_stride_) != 0) {
      return;
    }

    CloudFrame frame;
    frame.stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    frame.frame_id = msg->header.frame_id;
    frame.cloud_in_livo_map.reserve(cloud.size());

    // Store all accumulated points in FAST-LIVO2 camera_init/map coordinates.
    // cloud_registered is usually already in camera_init. If a body-frame cloud is used,
    // it is transformed into the LIVO map frame by the current FAST-LIVO2 odometry.
    const double c = std::cos(last_odom_yaw_);
    const double ss = std::sin(last_odom_yaw_);
    for (const auto &pt : cloud.points) {
      pcl::PointXYZI q = pt;
      if (cloud_frame_mode_ == "body") {
        q.x = static_cast<float>(last_odom_x_ + c * pt.x - ss * pt.y);
        q.y = static_cast<float>(last_odom_y_ + ss * pt.x + c * pt.y);
        q.z = pt.z;
      }
      frame.cloud_in_livo_map.push_back(q);
    }
    accumulated_clouds_.push_back(frame);
    pruneAccumulatedClouds(frame.stamp);
    has_cloud_ = !accumulated_clouds_.empty();
    publishGlobalCloud(frame.stamp);
    ROS_INFO_THROTTLE(3.0, "Accumulated cloud window: frames=%zu, window=%.1fs, latest points=%zu, mode=%s",
                      accumulated_clouds_.size(), cloud_accumulation_window_, cloud.size(), cloud_frame_mode_.c_str());
  }

  void pruneAccumulatedClouds(const ros::Time &now)
  {
    while (!accumulated_clouds_.empty()) {
      const double age = (now - accumulated_clouds_.front().stamp).toSec();
      if (age <= cloud_accumulation_window_) break;
      accumulated_clouds_.pop_front();
    }
    while (cloud_accumulation_max_frames_ > 0 &&
           accumulated_clouds_.size() > static_cast<size_t>(cloud_accumulation_max_frames_)) {
      accumulated_clouds_.pop_front();
    }
  }

  void imuCallback(const sensor_msgs::ImuConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    latest_imu_ = *msg;
    latest_imu_yaw_ = yawFromQuat(msg->orientation);
    has_imu_ = true;
    if (!has_state_ && init_yaw_from_imu_) {
      state_ << 0.0, 0.0, latest_imu_yaw_;
      has_state_ = true;
    }
  }

  void gpsCallback(const sensor_msgs::NavSatFixConstPtr &msg)
  {
    if (msg->status.status < sensor_msgs::NavSatStatus::STATUS_FIX) return;
    std::lock_guard<std::mutex> lock(mtx_);
    latest_gps_ = *msg;
    has_gps_ = true;
    if (!has_geo_origin_) {
      origin_lat_ = msg->latitude;
      origin_lon_ = msg->longitude;
      has_geo_origin_ = true;
      ROS_INFO("Set ENU origin from first GPS: %.9f %.9f", origin_lat_, origin_lon_);
    }
    Eigen::Vector2d z = latLonToENU(msg->latitude, msg->longitude);
    if (!has_state_ || !isFiniteState(state_)) {
      double init_yaw = (init_yaw_from_imu_ && has_imu_) ? latest_imu_yaw_ : 0.0;
      if (!std::isfinite(init_yaw)) init_yaw = 0.0;
      state_ << z.x(), z.y(), init_yaw;
      P_.setIdentity();
      P_(0,0) = gps_noise_std_ * gps_noise_std_;
      P_(1,1) = gps_noise_std_ * gps_noise_std_;
      P_(2,2) = 0.25;
      has_state_ = true;
      ROS_WARN("EKF state initialized/reset by GPS: state=(%.2f %.2f %.2f)", state_(0), state_(1), state_(2));
      publishFusedOdom(msg->header.stamp);
      return;
    }
    if (use_gps_update_ && has_state_) {
      double std_xy = gps_noise_std_;
      if (msg->position_covariance_type != sensor_msgs::NavSatFix::COVARIANCE_TYPE_UNKNOWN) {
        double cov = 0.5 * (std::max(0.0, msg->position_covariance[0]) + std::max(0.0, msg->position_covariance[4]));
        if (cov > 1e-6) std_xy = std::sqrt(cov);
      }
      updateXY(z, std_xy * std_xy, msg->header.stamp, "gps");
    }
  }

  Eigen::Vector2d latLonToENU(double lat, double lon) const
  {
    double lat0 = origin_lat_ * M_PI / 180.0;
    double dlat = (lat - origin_lat_) * M_PI / 180.0;
    double dlon = (lon - origin_lon_) * M_PI / 180.0;
    return Eigen::Vector2d(kEarthRadius * std::cos(lat0) * dlon, kEarthRadius * dlat);
  }

  void satelliteReloadTimer(const ros::TimerEvent&)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    std::string image_path;
    double center_lat = 0, center_lon = 0, mpp = 0;
    int width = 0, height = 0;
    if (!readSimpleYamlValue(satellite_yaml_, "image_path", image_path)) return;
    if (!readSimpleYamlDouble(satellite_yaml_, "center_lat", center_lat)) return;
    if (!readSimpleYamlDouble(satellite_yaml_, "center_lon", center_lon)) return;
    if (!readSimpleYamlDouble(satellite_yaml_, "meters_per_pixel", mpp)) return;
    readSimpleYamlInt(satellite_yaml_, "width", width);
    readSimpleYamlInt(satellite_yaml_, "height", height);
    if (image_path == satellite_image_path_ && !sat_img_.empty()) return;
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
      ROS_WARN_THROTTLE(10.0, "Cannot read satellite mosaic: %s", image_path.c_str());
      return;
    }
    sat_img_ = img;
    satellite_image_path_ = image_path;
    sat_center_lat_ = center_lat;
    sat_center_lon_ = center_lon;
    sat_mpp_ = mpp;
    sat_width_ = width > 0 ? width : img.cols;
    sat_height_ = height > 0 ? height : img.rows;
    if (!has_geo_origin_) {
      origin_lat_ = center_lat;
      origin_lon_ = center_lon;
      has_geo_origin_ = true;
      ROS_INFO("Set ENU origin from satellite center: %.9f %.9f", origin_lat_, origin_lon_);
    }
    publishSatelliteImage(ros::Time::now());
    ROS_INFO("Loaded high-res satellite mosaic: %s size=%dx%d mpp=%.3f", image_path.c_str(), sat_img_.cols, sat_img_.rows, sat_mpp_);
  }

  bool mapPointToCurrentBodyXY(const pcl::PointXYZI &pt, double &bx, double &by, double &bz) const
  {
    // Points in accumulated_clouds_ are stored in FAST-LIVO2 camera_init/map coordinates.
    // Convert them to the current robot-local frame using the latest FAST-LIVO2 odometry.
    bz = pt.z;
    double dx = static_cast<double>(pt.x) - last_odom_x_;
    double dy = static_cast<double>(pt.y) - last_odom_y_;
    double c = std::cos(last_odom_yaw_);
    double s = std::sin(last_odom_yaw_);
    bx =  c * dx + s * dy;
    by = -s * dx + c * dy;
    return std::isfinite(bx) && std::isfinite(by) && std::isfinite(bz);
  }

  BevResult buildGroundRemovedBEV()
  {
    BevResult res;
    if (!has_cloud_ || accumulated_clouds_.empty()) return res;

    struct P { double e, n, z, dz; bool obs; };
    std::vector<P> pts;
    size_t total_raw_points = 0;
    for (const auto &frame : accumulated_clouds_) total_raw_points += frame.cloud_in_livo_map.size();
    pts.reserve(total_raw_points);

    // First pass: transform accumulated points to a north-up local frame and estimate a coarse ground surface.
    std::unordered_map<uint64_t, double> min_z;
    double cyaw = std::cos(state_(2));
    double syaw = std::sin(state_(2));
    std::vector<P> raw_pts;
    raw_pts.reserve(total_raw_points);
    for (const auto &frame : accumulated_clouds_) {
      for (const auto &pt : frame.cloud_in_livo_map.points) {
        double bx = 0.0, by = 0.0, bz = 0.0;
        if (!mapPointToCurrentBodyXY(pt, bx, by, bz)) continue;
        if (bz < z_min_ || bz > z_max_) continue;
        double rr = std::sqrt(bx * bx + by * by);
        if (rr > max_point_range_) continue;
        double e = cyaw * bx - syaw * by;
        double n = syaw * bx + cyaw * by;
        raw_pts.push_back({e, n, bz, 0.0, false});
        int ix = static_cast<int>(std::floor(e / ground_grid_size_));
        int iy = static_cast<int>(std::floor(n / ground_grid_size_));
        uint64_t key = gridKey(ix, iy);
        auto it = min_z.find(key);
        if (it == min_z.end() || bz < it->second) min_z[key] = bz;
      }
    }
    if (raw_pts.size() < 100) {
      ROS_WARN_THROTTLE(2.0, "Accumulated BEV source has too few valid points after filtering: %zu / raw=%zu. Check cloud topic/mode or max_point_range.", raw_pts.size(), total_raw_points);
      return res;
    }

    pts.reserve(raw_pts.size());
    int obs_count = 0;
    for (const auto &p0 : raw_pts) {
      int ix = static_cast<int>(std::floor(p0.e / ground_grid_size_));
      int iy = static_cast<int>(std::floor(p0.n / ground_grid_size_));
      double ground_z = min_z[gridKey(ix, iy)];
      double dz = std::max(0.0, p0.z - ground_z);
      bool obs = dz > ground_remove_height_;
      if (obs) ++obs_count;
      pts.push_back({p0.e, p0.n, p0.z, dz, obs});
    }
    res.obstacle_points = obs_count;
    if (pts.size() < 100) return res;

    // Use all observed points to define the physical extent. This preserves road/ground coverage and avoids
    // cropping only around sparse non-ground points.
    double min_e = std::numeric_limits<double>::max(), min_n = std::numeric_limits<double>::max();
    double max_e = -std::numeric_limits<double>::max(), max_n = -std::numeric_limits<double>::max();
    for (const auto &p : pts) {
      min_e = std::min(min_e, p.e); max_e = std::max(max_e, p.e);
      min_n = std::min(min_n, p.n); max_n = std::max(max_n, p.n);
    }
    min_e -= bev_padding_m_; max_e += bev_padding_m_;
    min_n -= bev_padding_m_; max_n += bev_padding_m_;
    double width_m = std::max(max_e - min_e, bev_min_size_m_);
    double height_m = std::max(max_n - min_n, bev_min_size_m_);
    width_m = std::min(width_m, bev_max_size_m_);
    height_m = std::min(height_m, bev_max_size_m_);
    double ce = 0.5 * (min_e + max_e), cn = 0.5 * (min_n + max_n);
    min_e = ce - 0.5 * width_m; max_e = ce + 0.5 * width_m;
    min_n = cn - 0.5 * height_m; max_n = cn + 0.5 * height_m;

    int cols = std::max(128, static_cast<int>(std::ceil(width_m / bev_resolution_)));
    int rows = std::max(128, static_cast<int>(std::ceil(height_m / bev_resolution_)));
    cv::Mat density32(rows, cols, CV_32FC1, cv::Scalar(0));
    cv::Mat obs32(rows, cols, CV_32FC1, cv::Scalar(0));
    cv::Mat ground32(rows, cols, CV_32FC1, cv::Scalar(0));
    cv::Mat height32(rows, cols, CV_32FC1, cv::Scalar(0));

    for (const auto &p : pts) {
      int u = static_cast<int>((p.e - min_e) / std::max(1e-6, (max_e - min_e)) * cols);
      int v = static_cast<int>((max_n - p.n) / std::max(1e-6, (max_n - min_n)) * rows);
      if (u < 0 || u >= cols || v < 0 || v >= rows) continue;
      density32.at<float>(v,u) += 1.0f;
      if (p.obs) obs32.at<float>(v,u) += 1.0f;
      else ground32.at<float>(v,u) += 1.0f;
      height32.at<float>(v,u) = std::max(height32.at<float>(v,u), static_cast<float>(std::min(bev_height_clip_, p.dz)));
    }

    // Log-compress density-like layers to reduce dependence on local scan density.
    cv::log(density32 + 1.0, density32);
    cv::log(obs32 + 1.0, obs32);
    cv::log(ground32 + 1.0, ground32);
    cv::Mat density8 = normalizeFloatTo8U(density32);
    cv::Mat obs8 = normalizeFloatTo8U(obs32);
    cv::Mat ground8 = normalizeFloatTo8U(ground32);
    cv::Mat height8 = normalizeFloatTo8U(height32);

    if (bev_dilate_iter_ > 0) {
      cv::dilate(obs8, obs8, cv::Mat(), cv::Point(-1,-1), bev_dilate_iter_);
      cv::dilate(ground8, ground8, cv::Mat(), cv::Point(-1,-1), 1);
    }
    cv::morphologyEx(obs8, obs8, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1,-1), 1);
    cv::morphologyEx(ground8, ground8, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1,-1), 1);

    // Multi-layer pseudo image: contains density, vertical height, obstacles and observed ground.
    cv::Mat densityF, obsF, groundF, heightF;
    density8.convertTo(densityF, CV_32F);
    obs8.convertTo(obsF, CV_32F);
    ground8.convertTo(groundF, CV_32F);
    height8.convertTo(heightF, CV_32F);
    cv::Mat pseudo32(rows, cols, CV_32FC1, cv::Scalar(0));
    if (bev_use_multilayer_) {
      pseudo32 = bev_density_weight_  * densityF
               + bev_height_weight_   * heightF
               + bev_obstacle_weight_ * obsF
               + bev_ground_weight_   * groundF;
    } else {
      pseudo32 = obsF.clone();
    }
    cv::Mat pseudo8 = normalizeFloatTo8U(pseudo32);
    pseudo8 = applyClahe8U(pseudo8, 2.0, 8);
    cv::GaussianBlur(pseudo8, pseudo8, cv::Size(3,3), 0.0);

    // Visualization: color-coded multilayer map, easier to inspect than sparse binary edges.
    cv::Mat color(rows, cols, CV_8UC3, cv::Scalar(245,245,245));
    // ground observations: light gray-blue; obstacles/high structures: dark; high dz: reddish.
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        uchar g = ground8.at<uchar>(r,c);
        uchar o = obs8.at<uchar>(r,c);
        uchar h = height8.at<uchar>(r,c);
        uchar d = density8.at<uchar>(r,c);
        if (g > 10) color.at<cv::Vec3b>(r,c) = cv::Vec3b(215, 225, 225);
        if (d > 20 && o <= 10) color.at<cv::Vec3b>(r,c) = cv::Vec3b(195, 205, 205);
        if (o > 10) color.at<cv::Vec3b>(r,c) = cv::Vec3b(static_cast<uchar>(std::max(30, 180 - int(h/2))), 60, 60);
        if (h > 80) color.at<cv::Vec3b>(r,c) = cv::Vec3b(45, 45, static_cast<uchar>(std::min(220, 80 + int(h))));
      }
    }
    res.vis = color;
    res.match = pseudo8;
    res.occ_layer = obs8;
    res.density_layer = density8;
    res.height_layer = height8;
    res.ground_layer = ground8;
    res.min_e = min_e; res.max_e = max_e; res.min_n = min_n; res.max_n = max_n;
    res.ok = true;
    ROS_INFO_THROTTLE(5.0, "Multi-layer BEV: pts=%zu obs=%d size=%dx%d extent=(%.1fm x %.1fm)",
                      pts.size(), obs_count, cols, rows, max_e-min_e, max_n-min_n);
    return res;
  }

  bool pixelFromENU(double x, double y, double &u, double &v) const
  {
    if (sat_img_.empty() || sat_mpp_ <= 1e-6) return false;
    Eigen::Vector2d sat_center_enu = latLonToENU(sat_center_lat_, sat_center_lon_);
    double dx = x - sat_center_enu.x();
    double dy = y - sat_center_enu.y();
    u = sat_img_.cols / 2.0 + dx / sat_mpp_;
    v = sat_img_.rows / 2.0 - dy / sat_mpp_;
    return true;
  }

  SatCropResult cropSatelliteAlignedToBEV(const BevResult &bev)
  {
    SatCropResult out;
    if (sat_img_.empty() || !bev.ok || sat_mpp_ <= 1e-6) return out;
    double abs_min_e = state_(0) + bev.min_e - sat_crop_padding_m_;
    double abs_max_e = state_(0) + bev.max_e + sat_crop_padding_m_;
    double abs_min_n = state_(1) + bev.min_n - sat_crop_padding_m_;
    double abs_max_n = state_(1) + bev.max_n + sat_crop_padding_m_;

    double u1,v1,u2,v2;
    if (!pixelFromENU(abs_min_e, abs_max_n, u1, v1)) return out;
    if (!pixelFromENU(abs_max_e, abs_min_n, u2, v2)) return out;
    int x0 = std::max(0, static_cast<int>(std::floor(std::min(u1, u2))));
    int y0 = std::max(0, static_cast<int>(std::floor(std::min(v1, v2))));
    int x1 = std::min(sat_img_.cols - 1, static_cast<int>(std::ceil(std::max(u1, u2))));
    int y1 = std::min(sat_img_.rows - 1, static_cast<int>(std::ceil(std::max(v1, v2))));
    if (x1 - x0 < 64 || y1 - y0 < 64) {
      ROS_WARN_THROTTLE(2.0, "Satellite crop too small or outside mosaic: roi=(%d,%d,%d,%d), sat=%dx%d", x0, y0, x1-x0, y1-y0, sat_img_.cols, sat_img_.rows);
      return out;
    }
    cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
    out.crop_bgr = sat_img_(roi).clone();
    out.crop_match = makeSatelliteStructureImage(out.crop_bgr, cv::Size());
    out.abs_min_e = abs_min_e; out.abs_max_e = abs_max_e;
    out.abs_min_n = abs_min_n; out.abs_max_n = abs_max_n;
    out.ok = true;
    return out;
  }

  void publishMatchInputMeta(const BevResult &bev, const SatCropResult &sat)
  {
    std_msgs::Float64MultiArray msg;
    // Layout values:
    // 0 abs_min_e, 1 abs_max_e, 2 abs_min_n, 3 abs_max_n,
    // 4 bev_min_e, 5 bev_max_e, 6 bev_min_n, 7 bev_max_n,
    // 8 robot_u_in_bev_px, 9 robot_v_in_bev_px,
    // 10 bev_cols, 11 bev_rows, 12 sat_crop_cols, 13 sat_crop_rows,
    // 14 current_state_e, 15 current_state_n, 16 current_yaw, 17 sat_mpp.
    const double robot_u = (0.0 - bev.min_e) / std::max(1e-6, (bev.max_e - bev.min_e)) * bev.match.cols;
    const double robot_v = (bev.max_n - 0.0) / std::max(1e-6, (bev.max_n - bev.min_n)) * bev.match.rows;
    msg.data = {sat.abs_min_e, sat.abs_max_e, sat.abs_min_n, sat.abs_max_n,
                bev.min_e, bev.max_e, bev.min_n, bev.max_n,
                robot_u, robot_v,
                static_cast<double>(bev.match.cols), static_cast<double>(bev.match.rows),
                static_cast<double>(sat.crop_bgr.cols), static_cast<double>(sat.crop_bgr.rows),
                state_(0), state_(1), state_(2), sat_mpp_};
    pub_match_input_meta_.publish(msg);
  }

  bool gateMapMatchMeasurement(double meas_e, double meas_n, double confidence, double inlier_ratio, int inliers, const std::string &src)
  {
    const double correction = std::hypot(meas_e - state_(0), meas_n - state_(1));
    if (confidence < map_match_min_confidence_ || inlier_ratio < map_match_min_inlier_ratio_ || inliers < static_cast<int>(map_match_min_inliers_)) {
      ROS_WARN_THROTTLE(1.0,
        "Reject %s map match: weak geometry. meas=(%.2f %.2f), state=(%.2f %.2f), conf=%.2f<%.2f, ratio=%.2f<%.2f, inliers=%d<%.0f",
        src.c_str(), meas_e, meas_n, state_(0), state_(1), confidence, map_match_min_confidence_,
        inlier_ratio, map_match_min_inlier_ratio_, inliers, map_match_min_inliers_);
      return false;
    }
    if (correction > map_match_max_correction_) {
      ROS_WARN_THROTTLE(1.0,
        "Reject %s map match: correction %.2fm exceeds %.2fm. meas=(%.2f %.2f), state=(%.2f %.2f), conf=%.2f, inliers=%d",
        src.c_str(), correction, map_match_max_correction_, meas_e, meas_n, state_(0), state_(1), confidence, inliers);
      return false;
    }
    if (use_gps_consistency_gate_ && has_gps_) {
      Eigen::Vector2d gps_enu = latLonToENU(latest_gps_.latitude, latest_gps_.longitude);
      double dgps = std::hypot(meas_e - gps_enu.x(), meas_n - gps_enu.y());
      if (dgps > map_match_max_gps_disagreement_) {
        ROS_WARN_THROTTLE(1.0,
          "Reject %s map match: GPS disagreement %.2fm exceeds %.2fm. match=(%.2f %.2f), gps=(%.2f %.2f), conf=%.2f, inliers=%d",
          src.c_str(), dgps, map_match_max_gps_disagreement_, meas_e, meas_n, gps_enu.x(), gps_enu.y(), confidence, inliers);
        return false;
      }
    }
    return true;
  }

  void externalMapMatchCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!has_state_) return;
    const double meas_e = msg->pose.pose.position.x;
    const double meas_n = msg->pose.pose.position.y;
    if (!std::isfinite(meas_e) || !std::isfinite(meas_n)) return;
    double confidence = msg->pose.covariance[30];
    double inliers_d = msg->pose.covariance[31];
    double inlier_ratio = msg->pose.covariance[32];
    double var = msg->pose.covariance[0];
    if (!std::isfinite(confidence) || confidence <= 0.0) confidence = 0.7;
    if (!std::isfinite(inliers_d) || inliers_d < 0.0) inliers_d = 0.0;
    if (!std::isfinite(inlier_ratio) || inlier_ratio <= 0.0) inlier_ratio = 0.5;
    if (!std::isfinite(var) || var <= 0.0) var = map_match_noise_std_ * map_match_noise_std_;
    const int inliers = static_cast<int>(std::round(inliers_d));
    if (!gateMapMatchMeasurement(meas_e, meas_n, confidence, inlier_ratio, inliers, "external constrained-prior")) return;
    updateXY(Eigen::Vector2d(meas_e, meas_n), var, msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp, "external_map_match");
    publishMapMatchPose(meas_e, meas_n, state_(2), var, confidence, inliers);
    ROS_INFO("External constrained-prior match accepted: x=%.2f y=%.2f conf=%.2f ratio=%.2f inliers=%d",
             meas_e, meas_n, confidence, inlier_ratio, inliers);
  }

  void mapMatchTimer(const ros::TimerEvent&)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!has_state_ || !has_cloud_ || sat_img_.empty()) return;
    BevResult bev = buildGroundRemovedBEV();
    if (!bev.ok) return;
    publishImage(pub_bev_image_, bev.vis, "bgr8", earth_frame_);
    publishImage(pub_bev_match_, bev.match, "mono8", earth_frame_);

    SatCropResult sat = cropSatelliteAlignedToBEV(bev);
    if (!sat.ok) return;
    publishImage(pub_sat_crop_, sat.crop_bgr, "bgr8", earth_frame_);
    publishImage(pub_sat_match_, sat.crop_match, "mono8", earth_frame_);
    publishMatchInputMeta(bev, sat);

    if (matcher_backend_ == "external" || matcher_backend_ == "eloftr" || matcher_backend_ == "efficient_loftr") {
      ROS_INFO_THROTTLE(5.0, "Published accumulated BEV and aligned geographic crop for external constrained-prior matcher. frames=%zu obs=%d",
                         accumulated_clouds_.size(), bev.obstacle_points);
      return;
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(1500);
    std::vector<cv::KeyPoint> kp_bev, kp_sat;
    cv::Mat desc_bev, desc_sat;
    orb->detectAndCompute(bev.match, cv::noArray(), kp_bev, desc_bev);
    orb->detectAndCompute(sat.crop_match, cv::noArray(), kp_sat, desc_sat);
    if (desc_bev.empty() || desc_sat.empty() || kp_bev.size() < 10 || kp_sat.size() < 10) return;

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc_bev, desc_sat, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto &m : knn) {
      if (m.size() >= 2 && m[0].distance < 0.78 * m[1].distance) good.push_back(m[0]);
    }
    if (good.size() < static_cast<size_t>(map_match_min_inliers_)) return;

    std::vector<cv::Point2f> p_bev, p_sat;
    for (const auto &m : good) {
      p_bev.push_back(kp_bev[m.queryIdx].pt);
      p_sat.push_back(kp_sat[m.trainIdx].pt);
    }
    std::vector<uchar> mask;
    cv::Mat H = cv::findHomography(p_bev, p_sat, cv::RANSAC, 4.0, mask);
    if (H.empty()) return;
    int inliers = std::count(mask.begin(), mask.end(), static_cast<uchar>(1));
    if (inliers < static_cast<int>(map_match_min_inliers_)) return;

    // Robot pixel in the BEV image. This is more correct than simply using image center, because
    // the BEV crop is based on the actual point-cloud physical extent plus padding.
    double robot_u = (0.0 - bev.min_e) / (bev.max_e - bev.min_e) * bev.match.cols;
    double robot_v = (bev.max_n - 0.0) / (bev.max_n - bev.min_n) * bev.match.rows;
    std::vector<cv::Point2f> src(1), dst;
    src[0] = cv::Point2f(static_cast<float>(robot_u), static_cast<float>(robot_v));
    cv::perspectiveTransform(src, dst, H);
    if (dst.empty()) return;
    double sat_u = std::min(std::max(0.0, static_cast<double>(dst[0].x)), static_cast<double>(bev.match.cols-1));
    double sat_v = std::min(std::max(0.0, static_cast<double>(dst[0].y)), static_cast<double>(bev.match.rows-1));
    double meter_per_px_e = (sat.abs_max_e - sat.abs_min_e) / static_cast<double>(bev.match.cols);
    double meter_per_px_n = (sat.abs_max_n - sat.abs_min_n) / static_cast<double>(bev.match.rows);
    double meas_e = sat.abs_min_e + sat_u * meter_per_px_e;
    double meas_n = sat.abs_max_n - sat_v * meter_per_px_n;

    double confidence = std::min(1.0, static_cast<double>(inliers) / 80.0);
    double inlier_ratio = static_cast<double>(inliers) / std::max<size_t>(1, good.size());
    const double correction = std::hypot(meas_e - state_(0), meas_n - state_(1));
    if (confidence < map_match_min_confidence_ || inlier_ratio < map_match_min_inlier_ratio_) {
      ROS_WARN_THROTTLE(1.0,
        "Reject map match: weak geometry. meas=(%.2f %.2f), state=(%.2f %.2f), conf=%.2f<%.2f, inlier_ratio=%.2f<%.2f, inliers=%d, good=%zu",
        meas_e, meas_n, state_(0), state_(1), confidence, map_match_min_confidence_,
        inlier_ratio, map_match_min_inlier_ratio_, inliers, good.size());
      if (publish_rejected_match_debug_) publishMatchDebug(bev.match, sat.crop_match, kp_bev, kp_sat, good, mask);
      return;
    }
    if (correction > map_match_max_correction_) {
      ROS_WARN_THROTTLE(1.0,
        "Reject map match: correction %.2fm exceeds %.2fm. meas=(%.2f %.2f), state=(%.2f %.2f), conf=%.2f, inliers=%d",
        correction, map_match_max_correction_, meas_e, meas_n, state_(0), state_(1), confidence, inliers);
      if (publish_rejected_match_debug_) publishMatchDebug(bev.match, sat.crop_match, kp_bev, kp_sat, good, mask);
      return;
    }
    if (use_gps_consistency_gate_ && has_gps_) {
      Eigen::Vector2d gps_enu = latLonToENU(latest_gps_.latitude, latest_gps_.longitude);
      double dgps = std::hypot(meas_e - gps_enu.x(), meas_n - gps_enu.y());
      if (dgps > map_match_max_gps_disagreement_) {
        ROS_WARN_THROTTLE(1.0,
          "Reject map match: GPS disagreement %.2fm exceeds %.2fm. match=(%.2f %.2f), gps=(%.2f %.2f), conf=%.2f, inliers=%d",
          dgps, map_match_max_gps_disagreement_, meas_e, meas_n, gps_enu.x(), gps_enu.y(), confidence, inliers);
        if (publish_rejected_match_debug_) publishMatchDebug(bev.match, sat.crop_match, kp_bev, kp_sat, good, mask);
        return;
      }
    }
    double meas_var = std::pow(map_match_noise_std_ / std::max(0.15, confidence), 2.0);
    updateXY(Eigen::Vector2d(meas_e, meas_n), meas_var, ros::Time::now(), "map_match");
    publishMapMatchPose(meas_e, meas_n, state_(2), meas_var, confidence, inliers);
    ROS_INFO("Map match accepted after gates: correction=%.2fm conf=%.2f ratio=%.2f inliers=%d good=%zu", correction, confidence, inlier_ratio, inliers, good.size());
    publishMatchDebug(bev.match, sat.crop_match, kp_bev, kp_sat, good, mask);
  }

  void updateXY(const Eigen::Vector2d &z, double var, const ros::Time &stamp, const std::string &src)
  {
    if (!std::isfinite(z.x()) || !std::isfinite(z.y())) {
      ROS_WARN_THROTTLE(2.0, "Skip %s update because measurement is non-finite.", src.c_str());
      return;
    }
    if (!std::isfinite(var) || var < 1e-4) var = 1.0;
    if (!has_state_ || !isFiniteState(state_) || !P_.allFinite()) {
      double yaw = (has_imu_ && std::isfinite(latest_imu_yaw_)) ? latest_imu_yaw_ : 0.0;
      state_ << z.x(), z.y(), yaw;
      P_.setIdentity();
      P_(0,0) = var;
      P_(1,1) = var;
      P_(2,2) = 0.25;
      has_state_ = true;
      ROS_ERROR("EKF state/covariance was invalid. Reset by %s update to state=(%.2f %.2f %.2f).", src.c_str(), state_(0), state_(1), state_(2));
      publishFusedOdom(stamp);
      return;
    }

    Eigen::Matrix<double,2,3> H;
    H.setZero();
    H(0,0) = 1.0;
    H(1,1) = 1.0;
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * var;
    Eigen::Vector2d innov = z - H * state_;
    Eigen::Matrix2d S = H * P_ * H.transpose() + R;
    if (!S.allFinite() || std::abs(S.determinant()) < 1e-12) {
      ROS_WARN_THROTTLE(2.0, "Skip %s update because innovation covariance is singular or non-finite.", src.c_str());
      return;
    }
    Eigen::Matrix<double,3,2> K = P_ * H.transpose() * S.inverse();
    Eigen::Vector3d new_state = state_ + K * innov;
    new_state(2) = wrapAngle(new_state(2));
    Eigen::Matrix3d new_P = (Eigen::Matrix3d::Identity() - K * H) * P_ * (Eigen::Matrix3d::Identity() - K * H).transpose() + K * R * K.transpose();
    if (!new_state.allFinite() || !new_P.allFinite()) {
      ROS_ERROR("Reject %s update because EKF would become non-finite. z=(%.2f %.2f), var=%.3f", src.c_str(), z.x(), z.y(), var);
      return;
    }
    state_ = new_state;
    P_ = 0.5 * (new_P + new_P.transpose());
    P_(0,0) = std::max(P_(0,0), 1e-6);
    P_(1,1) = std::max(P_(1,1), 1e-6);
    P_(2,2) = std::max(P_(2,2), 1e-6);
    publishFusedOdom(stamp);
    ROS_INFO_THROTTLE(2.0, "EKF update from %s: z=(%.2f %.2f), state=(%.2f %.2f %.2f)", src.c_str(), z.x(), z.y(), state_(0), state_(1), state_(2));
  }

  void publishFusedOdom(const ros::Time &stamp)
  {
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = earth_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = state_(0);
    odom.pose.pose.position.y = state_(1);
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = quatFromYaw(state_(2));
    odom.pose.covariance[0] = P_(0,0);
    odom.pose.covariance[7] = P_(1,1);
    odom.pose.covariance[35] = P_(2,2);
    pub_fused_odom_.publish(odom);
    if (publish_tf_) {
      geometry_msgs::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = state_(0);
      tf.transform.translation.y = state_(1);
      tf.transform.translation.z = 0.0;
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_.sendTransform(tf);
    }
  }

  void publishMapMatchPose(double x, double y, double yaw, double var, double confidence, int inliers)
  {
    geometry_msgs::PoseWithCovarianceStamped p;
    p.header.stamp = ros::Time::now();
    p.header.frame_id = earth_frame_;
    p.pose.pose.position.x = x;
    p.pose.pose.position.y = y;
    p.pose.pose.orientation = quatFromYaw(yaw);
    p.pose.covariance[0] = var;
    p.pose.covariance[7] = var;
    p.pose.covariance[35] = 0.2;
    pub_map_match_pose_.publish(p);
    ROS_INFO("Map match accepted: x=%.2f y=%.2f conf=%.2f inliers=%d", x, y, confidence, inliers);
  }

  void publishImage(const image_transport::Publisher &pub, const cv::Mat &img, const std::string &encoding, const std::string &frame)
  {
    if (img.empty()) return;
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), encoding, img).toImageMsg();
    msg->header.stamp = ros::Time::now();
    msg->header.frame_id = frame;
    pub.publish(msg);
  }

  void publishSatelliteImage(const ros::Time &stamp)
  {
    if (sat_img_.empty()) return;
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", sat_img_).toImageMsg();
    img_msg->header.stamp = stamp;
    img_msg->header.frame_id = earth_frame_;
    pub_sat_image_.publish(img_msg);
  }

  void publishMatchDebug(const cv::Mat &bev, const cv::Mat &sat, const std::vector<cv::KeyPoint> &kp1,
                         const std::vector<cv::KeyPoint> &kp2, const std::vector<cv::DMatch> &matches,
                         const std::vector<uchar> &mask)
  {
    std::vector<cv::DMatch> inlier_matches;
    for (size_t i = 0; i < matches.size() && i < mask.size(); ++i) if (mask[i]) inlier_matches.push_back(matches[i]);
    cv::Mat dbg;
    cv::drawMatches(bev, kp1, sat, kp2, inlier_matches, dbg);
    publishImage(pub_match_debug_, dbg, "bgr8", earth_frame_);
  }

  void publishGlobalCloud(const ros::Time &stamp)
  {
    if (!has_cloud_ || accumulated_clouds_.empty()) return;
    pcl::PointCloud<pcl::PointXYZI> out;
    size_t total = 0;
    for (const auto &frame : accumulated_clouds_) total += frame.cloud_in_livo_map.size();
    out.reserve(total);
    double c = std::cos(state_(2));
    double ss = std::sin(state_(2));
    for (const auto &frame : accumulated_clouds_) {
      for (const auto &p : frame.cloud_in_livo_map.points) {
        double bx = 0.0, by = 0.0, bz = 0.0;
        if (!mapPointToCurrentBodyXY(p, bx, by, bz)) continue;
        pcl::PointXYZI q = p;
        q.x = static_cast<float>(state_(0) + c * bx - ss * by);
        q.y = static_cast<float>(state_(1) + ss * bx + c * by);
        q.z = static_cast<float>(bz);
        out.push_back(q);
      }
    }
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(out, msg);
    msg.header.stamp = stamp;
    msg.header.frame_id = earth_frame_;
    pub_global_cloud_.publish(msg);
  }

private:
  ros::NodeHandle nh_, pnh_;
  image_transport::ImageTransport it_;
  ros::Subscriber sub_odom_, sub_cloud_, sub_gps_, sub_imu_, sub_external_match_;
  ros::Publisher pub_fused_odom_, pub_map_match_pose_, pub_global_cloud_, pub_match_input_meta_;
  image_transport::Publisher pub_sat_image_, pub_sat_crop_, pub_sat_match_, pub_bev_image_, pub_bev_match_, pub_match_debug_;
  ros::Timer timer_match_, timer_sat_reload_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  std::mutex mtx_;

  std::string fast_livo_odom_topic_, fast_livo_cloud_topic_, gps_topic_, imu_topic_, satellite_yaml_, earth_frame_, base_frame_, satellite_image_path_, cloud_frame_mode_, matcher_backend_, external_match_pose_topic_;
  double bev_resolution_, bev_min_size_m_, bev_max_size_m_, bev_padding_m_, sat_crop_padding_m_, max_point_range_, z_min_, z_max_, ground_grid_size_, ground_remove_height_, bev_height_clip_, bev_density_weight_, bev_height_weight_, bev_obstacle_weight_, bev_ground_weight_, cloud_accumulation_window_, cloud_accumulation_voxel_size_;
  int bev_dilate_iter_, cloud_accumulation_max_frames_, cloud_accumulation_stride_, cloud_frame_count_ = 0;
  double q_xy_, q_yaw_, gps_noise_std_, map_match_noise_std_, map_match_min_inliers_, match_rate_;
  double map_match_min_confidence_, map_match_min_inlier_ratio_, map_match_max_correction_, map_match_max_gps_disagreement_;
  bool use_gps_update_, publish_tf_, init_yaw_from_imu_, use_gps_consistency_gate_, publish_rejected_match_debug_, bev_use_multilayer_;

  bool has_last_odom_ = false, has_state_ = false, has_cloud_ = false, has_gps_ = false, has_imu_ = false, has_geo_origin_ = false;
  double last_odom_x_ = 0, last_odom_y_ = 0, last_odom_yaw_ = 0, latest_imu_yaw_ = 0;
  double origin_lat_ = 0, origin_lon_ = 0, sat_center_lat_ = 0, sat_center_lon_ = 0, sat_mpp_ = 0;
  int sat_width_ = 0, sat_height_ = 0;
  Eigen::Vector3d state_;
  Eigen::Matrix3d P_;
  sensor_msgs::PointCloud2 latest_cloud_msg_;
  std::deque<CloudFrame> accumulated_clouds_;
  sensor_msgs::NavSatFix latest_gps_;
  sensor_msgs::Imu latest_imu_;
  cv::Mat sat_img_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "global_localizer_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  GlobalLocalizerNode node(nh, pnh);
  ros::spin();
  return 0;
}
