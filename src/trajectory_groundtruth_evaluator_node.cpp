#include <algorithm>
#include <cmath>
#include <deque>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/String.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace
{

constexpr double kWgs84A = 6378137.0;
constexpr double kWgs84F = 1.0 / 298.257223563;
constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);

double clampAngle(double a)
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

Eigen::Vector3d llaToEcef(double lat_deg, double lon_deg, double alt)
{
  const double lat = lat_deg * M_PI / 180.0;
  const double lon = lon_deg * M_PI / 180.0;
  const double sin_lat = std::sin(lat);
  const double cos_lat = std::cos(lat);
  const double sin_lon = std::sin(lon);
  const double cos_lon = std::cos(lon);
  const double N = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sin_lat * sin_lat);
  return Eigen::Vector3d((N + alt) * cos_lat * cos_lon,
                         (N + alt) * cos_lat * sin_lon,
                         (N * (1.0 - kWgs84E2) + alt) * sin_lat);
}

Eigen::Matrix3d ecefToEnuRotation(double origin_lat_deg, double origin_lon_deg)
{
  const double lat = origin_lat_deg * M_PI / 180.0;
  const double lon = origin_lon_deg * M_PI / 180.0;
  const double sin_lat = std::sin(lat);
  const double cos_lat = std::cos(lat);
  const double sin_lon = std::sin(lon);
  const double cos_lon = std::cos(lon);
  Eigen::Matrix3d R;
  R << -sin_lon,             cos_lon,            0.0,
       -sin_lat * cos_lon,  -sin_lat * sin_lon, cos_lat,
        cos_lat * cos_lon,   cos_lat * sin_lon, sin_lat;
  return R;
}

bool validQuat(const geometry_msgs::Quaternion& q)
{
  const double n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  return std::isfinite(n2) && n2 > 1e-8;
}

struct GroundTruthSample
{
  ros::Time stamp;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double pos_var_xy = 1.0;
  double pos_var_z = 1.0;
  double yaw_var = 0.05;
};

struct PairSample
{
  Eigen::Vector3d odom = Eigen::Vector3d::Zero();
  Eigen::Vector3d gt = Eigen::Vector3d::Zero();
};

struct Align2D
{
  bool ready = false;
  double yaw = 0.0;
  Eigen::Vector2d t = Eigen::Vector2d::Zero();
  double z_offset = 0.0;
};

Eigen::Vector2d rotate2(double yaw, const Eigen::Vector2d& p)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return Eigen::Vector2d(c * p.x() - s * p.y(), s * p.x() + c * p.y());
}

}  // namespace

class TrajectoryGroundTruthEvaluator
{
 public:
  TrajectoryGroundTruthEvaluator() : nh_(), pnh_("~")
  {
    pnh_.param<std::string>("rtk_fix_topic", rtk_fix_topic_, "/novatel/oem7/fix");
    pnh_.param<std::string>("imu_topic", imu_topic_, "/adi/adis16465/imu");
    pnh_.param<std::string>("raw_odom_topic", raw_odom_topic_, "/aft_mapped_to_init");
    pnh_.param<std::string>("semantic_odom_topic", semantic_odom_topic_, "/semantic_corrected_odom");
    pnh_.param<std::string>("gt_odom_topic", gt_odom_topic_, "/ground_truth/odom");
    pnh_.param<std::string>("stats_topic", stats_topic_, "/trajectory_eval/rmse");
    pnh_.param<std::string>("map_frame", map_frame_, "rtk_enu");
    pnh_.param<std::string>("alignment_mode", alignment_mode_, "se2_window");
    pnh_.param<std::string>("gt_match_mode", gt_match_mode_, "stamp_or_latest");
    pnh_.param<std::string>("record_dir", record_dir_, "/tmp/trajectory_eval");

    pnh_.param<bool>("use_fixed_origin", use_fixed_origin_, false);
    pnh_.param<double>("origin_latitude", origin_latitude_, 0.0);
    pnh_.param<double>("origin_longitude", origin_longitude_, 0.0);
    pnh_.param<double>("origin_altitude", origin_altitude_, 0.0);
    pnh_.param<int>("min_navsat_status", min_navsat_status_, static_cast<int>(sensor_msgs::NavSatStatus::STATUS_FIX));
    pnh_.param<bool>("use_imu_orientation", use_imu_orientation_, true);
    pnh_.param<bool>("integrate_imu_yaw_when_orientation_invalid", integrate_imu_yaw_when_orientation_invalid_, true);
    pnh_.param<double>("max_imu_age_sec", max_imu_age_sec_, 0.20);
    pnh_.param<double>("max_imu_integration_dt", max_imu_integration_dt_, 0.05);
    pnh_.param<double>("imu_yaw_offset_deg", imu_yaw_offset_deg_, 0.0);
    pnh_.param<double>("imu_gyro_z_sign", imu_gyro_z_sign_, 1.0);
    pnh_.param<double>("imu_gyro_z_bias", imu_gyro_z_bias_, 0.0);
    pnh_.param<double>("default_rtk_xy_std", default_rtk_xy_std_, 0.03);
    pnh_.param<double>("default_rtk_z_std", default_rtk_z_std_, 0.06);
    pnh_.param<double>("default_imu_yaw_std_deg", default_imu_yaw_std_deg_, 1.0);

    pnh_.param<double>("max_gt_match_dt", max_gt_match_dt_, 0.05);
    pnh_.param<double>("max_latest_gt_age_sec", max_latest_gt_age_sec_, 0.50);
    pnh_.param<double>("gt_buffer_sec", gt_buffer_sec_, 30.0);
    pnh_.param<int>("alignment_min_samples", alignment_min_samples_, 30);
    pnh_.param<double>("alignment_min_distance_m", alignment_min_distance_m_, 5.0);
    pnh_.param<bool>("alignment_fallback_to_first_pose", alignment_fallback_to_first_pose_, true);
    pnh_.param<int>("alignment_fallback_samples", alignment_fallback_samples_, 30);
    pnh_.param<int>("max_alignment_samples", max_alignment_samples_, 300);
    pnh_.param<double>("publish_rate", publish_rate_, 1.0);
    pnh_.param<bool>("record_csv", record_csv_, true);
    pnh_.param<bool>("publish_paths", publish_paths_, true);
    pnh_.param<int>("max_path_points", max_path_points_, 20000);

    if (gt_match_mode_ != "stamp" && gt_match_mode_ != "latest" && gt_match_mode_ != "stamp_or_latest")
    {
      ROS_WARN("Unknown gt_match_mode=%s, using stamp_or_latest", gt_match_mode_.c_str());
      gt_match_mode_ = "stamp_or_latest";
    }

    if (use_fixed_origin_)
    {
      setOrigin(origin_latitude_, origin_longitude_, origin_altitude_);
    }

    raw_.name = "raw_fast_livo2";
    raw_.topic = raw_odom_topic_;
    semantic_.name = "semantic_corrected";
    semantic_.topic = semantic_odom_topic_;

    rtk_sub_ = nh_.subscribe(rtk_fix_topic_, 100, &TrajectoryGroundTruthEvaluator::rtkCb, this);
    imu_sub_ = nh_.subscribe(imu_topic_, 200, &TrajectoryGroundTruthEvaluator::imuCb, this);
    raw_sub_ = nh_.subscribe(raw_odom_topic_, 100, &TrajectoryGroundTruthEvaluator::rawOdomCb, this);
    semantic_sub_ = nh_.subscribe(semantic_odom_topic_, 100, &TrajectoryGroundTruthEvaluator::semanticOdomCb, this);

    gt_pub_ = nh_.advertise<nav_msgs::Odometry>(gt_odom_topic_, 20);
    stats_pub_ = nh_.advertise<std_msgs::String>(stats_topic_, 10);
    gt_path_pub_ = nh_.advertise<nav_msgs::Path>("/trajectory_eval/ground_truth_path", 1, true);
    raw_path_pub_ = nh_.advertise<nav_msgs::Path>("/trajectory_eval/raw_aligned_path", 1, true);
    semantic_path_pub_ = nh_.advertise<nav_msgs::Path>("/trajectory_eval/semantic_aligned_path", 1, true);

    stats_timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(0.1, publish_rate_)),
                                  &TrajectoryGroundTruthEvaluator::statsTimerCb, this);

    initCsv();

    ROS_INFO("trajectory_groundtruth_evaluator started: rtk=%s imu=%s raw=%s semantic=%s alignment=%s gt_match=%s max_stamp_dt=%.3f latest_age=%.3f",
             rtk_fix_topic_.c_str(), imu_topic_.c_str(), raw_odom_topic_.c_str(),
             semantic_odom_topic_.c_str(), alignment_mode_.c_str(), gt_match_mode_.c_str(),
             max_gt_match_dt_, max_latest_gt_age_sec_);
  }

 private:
  struct TrajStats
  {
    std::string name;
    std::string topic;
    Align2D align;
    std::vector<PairSample> alignment_pairs;
    uint64_t received = 0;
    uint64_t no_gt_match = 0;
    uint64_t latest_gt_fallback = 0;
    uint64_t waiting_alignment = 0;
    uint64_t samples = 0;
    double sum_sq_xy = 0.0;
    double sum_sq_xyz = 0.0;
    double sum_abs_z = 0.0;
    double max_2d = 0.0;
    double max_3d = 0.0;
    double last_alignment_spread = 0.0;
    double last_alignment_gt_spread = 0.0;
    double last_gt_dt = 0.0;
    double last_reject_dt = 0.0;
    double last_err_2d = 0.0;
    double last_err_3d = 0.0;
    ros::Time first_stamp;
    ros::Time last_stamp;
    std::ofstream csv;
  };

  void setOrigin(double lat, double lon, double alt)
  {
    origin_latitude_ = lat;
    origin_longitude_ = lon;
    origin_altitude_ = alt;
    origin_ecef_ = llaToEcef(lat, lon, alt);
    R_ecef_enu_ = ecefToEnuRotation(lat, lon);
    has_origin_ = true;
    ROS_INFO("GroundTruth ENU origin set: lat=%.9f lon=%.9f alt=%.3f",
             lat, lon, alt);
  }

  Eigen::Vector3d llaToEnu(double lat, double lon, double alt) const
  {
    return R_ecef_enu_ * (llaToEcef(lat, lon, alt) - origin_ecef_);
  }

  void imuCb(const sensor_msgs::ImuConstPtr& msg)
  {
    const ros::Time stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    const bool orientation_cov_valid = msg->orientation_covariance[0] >= 0.0;
    if (use_imu_orientation_ && orientation_cov_valid && validQuat(msg->orientation))
    {
      latest_imu_ = *msg;
      latest_imu_yaw_ = clampAngle(yawFromQuat(msg->orientation) + imu_yaw_offset_deg_ * M_PI / 180.0);
      latest_imu_stamp_ = stamp;
      has_imu_ = true;
      return;
    }

    if (!integrate_imu_yaw_when_orientation_invalid_ || !std::isfinite(msg->angular_velocity.z))
    {
      return;
    }

    latest_imu_ = *msg;
    if (!has_integrated_imu_yaw_)
    {
      latest_imu_yaw_ = imu_yaw_offset_deg_ * M_PI / 180.0;
      has_integrated_imu_yaw_ = true;
    }
    else
    {
      const double dt = (stamp - latest_imu_stamp_).toSec();
      if (dt > 0.0 && dt <= max_imu_integration_dt_)
      {
        latest_imu_yaw_ = clampAngle(latest_imu_yaw_ +
                                     (imu_gyro_z_sign_ * msg->angular_velocity.z - imu_gyro_z_bias_) * dt);
      }
    }
    latest_imu_stamp_ = stamp;
    has_imu_ = true;
  }

  void rtkCb(const sensor_msgs::NavSatFixConstPtr& msg)
  {
    if (msg->status.status < min_navsat_status_)
    {
      return;
    }
    if (!std::isfinite(msg->latitude) || !std::isfinite(msg->longitude) || !std::isfinite(msg->altitude))
    {
      return;
    }

    if (!has_origin_)
    {
      setOrigin(msg->latitude, msg->longitude, msg->altitude);
    }

    GroundTruthSample gt;
    gt.stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    gt.p = llaToEnu(msg->latitude, msg->longitude, msg->altitude);

    bool imu_ok = false;
    if (use_imu_orientation_ && has_imu_)
    {
      const double age = std::fabs((gt.stamp - latest_imu_stamp_).toSec());
      imu_ok = age <= max_imu_age_sec_;
    }
    gt.yaw = imu_ok ? latest_imu_yaw_ : 0.0;
    gt.yaw_var = std::pow(default_imu_yaw_std_deg_ * M_PI / 180.0, 2.0);
    if (imu_ok && latest_imu_.orientation_covariance[8] >= 0.0)
    {
      gt.yaw_var = std::max(1e-6, latest_imu_.orientation_covariance[8]);
    }

    gt.pos_var_xy = default_rtk_xy_std_ * default_rtk_xy_std_;
    gt.pos_var_z = default_rtk_z_std_ * default_rtk_z_std_;
    if (msg->position_covariance_type != sensor_msgs::NavSatFix::COVARIANCE_TYPE_UNKNOWN)
    {
      const double cov_e = std::max(0.0, msg->position_covariance[0]);
      const double cov_n = std::max(0.0, msg->position_covariance[4]);
      const double cov_u = std::max(0.0, msg->position_covariance[8]);
      if (cov_e + cov_n > 1e-10)
      {
        gt.pos_var_xy = 0.5 * (cov_e + cov_n);
      }
      if (cov_u > 1e-10)
      {
        gt.pos_var_z = cov_u;
      }
    }

    gt_buffer_.push_back(gt);
    while (!gt_buffer_.empty() && (gt.stamp - gt_buffer_.front().stamp).toSec() > gt_buffer_sec_)
    {
      gt_buffer_.pop_front();
    }
    latest_gt_ = gt;
    latest_gt_wall_ = ros::WallTime::now();
    has_gt_ = true;
    publishGt(gt);
    recordGt(gt, *msg, imu_ok);
  }

  void publishGt(const GroundTruthSample& gt)
  {
    nav_msgs::Odometry odom;
    odom.header.stamp = gt.stamp;
    odom.header.frame_id = map_frame_;
    odom.child_frame_id = "rtk_imu_ground_truth";
    odom.pose.pose.position.x = gt.p.x();
    odom.pose.pose.position.y = gt.p.y();
    odom.pose.pose.position.z = gt.p.z();
    odom.pose.pose.orientation = quatFromYaw(gt.yaw);
    odom.pose.covariance[0] = gt.pos_var_xy;
    odom.pose.covariance[7] = gt.pos_var_xy;
    odom.pose.covariance[14] = gt.pos_var_z;
    odom.pose.covariance[35] = gt.yaw_var;
    gt_pub_.publish(odom);

    if (publish_paths_)
    {
      appendPath(gt_path_, gt.stamp, gt.p, gt.yaw);
      gt_path_.header.frame_id = map_frame_;
      gt_path_.header.stamp = gt.stamp;
      gt_path_pub_.publish(gt_path_);
    }
  }

  bool nearestGt(const ros::Time& stamp,
                 GroundTruthSample& out,
                 double& dt,
                 bool& used_latest,
                 double& stamp_dt) const
  {
    used_latest = false;
    dt = std::numeric_limits<double>::infinity();
    stamp_dt = std::numeric_limits<double>::infinity();
    if (gt_buffer_.empty())
    {
      return false;
    }
    double best = std::numeric_limits<double>::infinity();
    const GroundTruthSample* best_sample = nullptr;
    for (const auto& gt : gt_buffer_)
    {
      const double d = std::fabs((stamp - gt.stamp).toSec());
      if (d < best)
      {
        best = d;
        best_sample = &gt;
      }
    }
    stamp_dt = best;
    const bool allow_stamp = gt_match_mode_ == "stamp" || gt_match_mode_ == "stamp_or_latest";
    if (allow_stamp && best_sample && best <= max_gt_match_dt_)
    {
      out = *best_sample;
      dt = best;
      return true;
    }

    const bool allow_latest = gt_match_mode_ == "latest" || gt_match_mode_ == "stamp_or_latest";
    if (allow_latest && has_gt_)
    {
      const double age = (ros::WallTime::now() - latest_gt_wall_).toSec();
      if (std::isfinite(age) && age <= max_latest_gt_age_sec_)
      {
        out = latest_gt_;
        dt = age;
        used_latest = true;
        return true;
      }
    }

    return false;
  }

  static Eigen::Vector3d odomPosition(const nav_msgs::Odometry& odom)
  {
    return Eigen::Vector3d(odom.pose.pose.position.x,
                           odom.pose.pose.position.y,
                           odom.pose.pose.position.z);
  }

  bool updateAlignment(TrajStats& stats,
                       const nav_msgs::Odometry& odom,
                       const GroundTruthSample& gt)
  {
    if (alignment_mode_ == "none")
    {
      stats.align.ready = true;
      stats.align.yaw = 0.0;
      stats.align.t.setZero();
      stats.align.z_offset = 0.0;
      return true;
    }

    const Eigen::Vector3d po = odomPosition(odom);
    if (alignment_mode_ == "first_pose")
    {
      if (!stats.align.ready)
      {
        const double odom_yaw = yawFromQuat(odom.pose.pose.orientation);
        stats.align.yaw = clampAngle(gt.yaw - odom_yaw);
        stats.align.t = gt.p.head<2>() - rotate2(stats.align.yaw, po.head<2>());
        stats.align.z_offset = gt.p.z() - po.z();
        stats.align.ready = true;
        ROS_INFO("%s alignment initialized by first_pose: yaw=%.3f deg t=(%.2f %.2f) z=%.2f",
                 stats.name.c_str(), stats.align.yaw * 180.0 / M_PI,
                 stats.align.t.x(), stats.align.t.y(), stats.align.z_offset);
      }
      return true;
    }

    if (alignment_mode_ != "se2_window")
    {
      ROS_WARN_THROTTLE(5.0, "Unknown alignment_mode=%s, using se2_window", alignment_mode_.c_str());
    }
    if (stats.align.ready)
    {
      return true;
    }

    PairSample pair;
    pair.odom = po;
    pair.gt = gt.p;
    stats.alignment_pairs.push_back(pair);
    if (static_cast<int>(stats.alignment_pairs.size()) > std::max(2, max_alignment_samples_))
    {
      stats.alignment_pairs.erase(stats.alignment_pairs.begin());
    }
    if (static_cast<int>(stats.alignment_pairs.size()) < std::max(2, alignment_min_samples_))
    {
      return false;
    }

    Eigen::Vector2d mean_o = Eigen::Vector2d::Zero();
    Eigen::Vector2d mean_g = Eigen::Vector2d::Zero();
    double mean_oz = 0.0;
    double mean_gz = 0.0;
    for (const auto& s : stats.alignment_pairs)
    {
      mean_o += s.odom.head<2>();
      mean_g += s.gt.head<2>();
      mean_oz += s.odom.z();
      mean_gz += s.gt.z();
    }
    const double inv = 1.0 / static_cast<double>(stats.alignment_pairs.size());
    mean_o *= inv;
    mean_g *= inv;
    mean_oz *= inv;
    mean_gz *= inv;

    double spread = 0.0;
    double gt_spread = 0.0;
    double cross = 0.0;
    double dot = 0.0;
    for (const auto& s : stats.alignment_pairs)
    {
      const Eigen::Vector2d so = s.odom.head<2>() - mean_o;
      const Eigen::Vector2d sg = s.gt.head<2>() - mean_g;
      spread += so.squaredNorm();
      gt_spread += sg.squaredNorm();
      cross += so.x() * sg.y() - so.y() * sg.x();
      dot += so.x() * sg.x() + so.y() * sg.y();
    }
    spread = std::sqrt(spread * inv);
    gt_spread = std::sqrt(gt_spread * inv);
    stats.last_alignment_spread = spread;
    stats.last_alignment_gt_spread = gt_spread;
    if (spread < alignment_min_distance_m_)
    {
      if (alignment_fallback_to_first_pose_ &&
          static_cast<int>(stats.alignment_pairs.size()) >= std::max(2, alignment_fallback_samples_))
      {
        const double odom_yaw = yawFromQuat(odom.pose.pose.orientation);
        stats.align.yaw = clampAngle(gt.yaw - odom_yaw);
        stats.align.t = gt.p.head<2>() - rotate2(stats.align.yaw, po.head<2>());
        stats.align.z_offset = gt.p.z() - po.z();
        stats.align.ready = true;
        ROS_WARN("%s alignment fallback to first_pose after %zu samples: odom_spread=%.3f gt_spread=%.3f yaw=%.3f deg t=(%.2f %.2f) z=%.2f",
                 stats.name.c_str(), stats.alignment_pairs.size(), spread, gt_spread,
                 stats.align.yaw * 180.0 / M_PI,
                 stats.align.t.x(), stats.align.t.y(), stats.align.z_offset);
        return true;
      }
      return false;
    }

    stats.align.yaw = std::atan2(cross, dot);
    stats.align.t = mean_g - rotate2(stats.align.yaw, mean_o);
    stats.align.z_offset = mean_gz - mean_oz;
    stats.align.ready = true;
    ROS_INFO("%s alignment ready from %zu samples: yaw=%.3f deg t=(%.2f %.2f) z=%.2f spread=%.2f",
             stats.name.c_str(), stats.alignment_pairs.size(),
             stats.align.yaw * 180.0 / M_PI, stats.align.t.x(), stats.align.t.y(),
             stats.align.z_offset, spread);
    return true;
  }

  Eigen::Vector3d alignPosition(const TrajStats& stats, const Eigen::Vector3d& p) const
  {
    Eigen::Vector3d out;
    const Eigen::Vector2d xy = rotate2(stats.align.yaw, p.head<2>()) + stats.align.t;
    out.x() = xy.x();
    out.y() = xy.y();
    out.z() = p.z() + stats.align.z_offset;
    return out;
  }

  void rawOdomCb(const nav_msgs::OdometryConstPtr& msg)
  {
    processOdom(*msg, raw_, raw_path_, raw_path_pub_);
  }

  void semanticOdomCb(const nav_msgs::OdometryConstPtr& msg)
  {
    processOdom(*msg, semantic_, semantic_path_, semantic_path_pub_);
  }

  void processOdom(const nav_msgs::Odometry& odom,
                   TrajStats& stats,
                   nav_msgs::Path& path,
                   ros::Publisher& path_pub)
  {
    const ros::Time stamp = odom.header.stamp.isZero() ? ros::Time::now() : odom.header.stamp;
    stats.received += 1;
    GroundTruthSample gt;
    double dt = 0.0;
    double stamp_dt = 0.0;
    bool used_latest = false;
    if (!nearestGt(stamp, gt, dt, used_latest, stamp_dt))
    {
      stats.no_gt_match += 1;
      stats.last_reject_dt = stamp_dt;
      ROS_WARN_THROTTLE(5.0,
                        "%s has odom but no matched GT: received=%lu no_gt=%lu gt_samples=%zu stamp_dt=%.3f max_stamp_dt=%.3f mode=%s",
                        stats.name.c_str(),
                        static_cast<unsigned long>(stats.received),
                        static_cast<unsigned long>(stats.no_gt_match),
                        gt_buffer_.size(),
                        std::isfinite(stamp_dt) ? stamp_dt : -1.0,
                        max_gt_match_dt_,
                        gt_match_mode_.c_str());
      return;
    }
    stats.last_gt_dt = dt;
    if (used_latest)
    {
      stats.latest_gt_fallback += 1;
    }
    if (!updateAlignment(stats, odom, gt))
    {
      stats.waiting_alignment += 1;
      return;
    }

    const Eigen::Vector3d raw_p = odomPosition(odom);
    const Eigen::Vector3d aligned = alignPosition(stats, raw_p);
    const Eigen::Vector3d e = aligned - gt.p;
    const double err2d = e.head<2>().norm();
    const double err3d = e.norm();
    stats.samples += 1;
    stats.sum_sq_xy += err2d * err2d;
    stats.sum_sq_xyz += err3d * err3d;
    stats.sum_abs_z += std::fabs(e.z());
    stats.max_2d = std::max(stats.max_2d, err2d);
    stats.max_3d = std::max(stats.max_3d, err3d);
    stats.last_err_2d = err2d;
    stats.last_err_3d = err3d;
    if (stats.samples == 1)
    {
      stats.first_stamp = stamp;
    }
    stats.last_stamp = stamp;

    if (record_csv_ && stats.csv.is_open())
    {
      stats.csv << std::fixed << std::setprecision(9)
                << stamp.toSec() << ","
                << gt.p.x() << "," << gt.p.y() << "," << gt.p.z() << ","
                << raw_p.x() << "," << raw_p.y() << "," << raw_p.z() << ","
                << aligned.x() << "," << aligned.y() << "," << aligned.z() << ","
                << e.x() << "," << e.y() << "," << e.z() << ","
                << err2d << "," << err3d << ","
                << rmse2d(stats) << "," << rmse3d(stats) << ","
                << dt << "\n";
    }

    if (publish_paths_)
    {
      appendPath(path, stamp, aligned, yawFromQuat(odom.pose.pose.orientation) + stats.align.yaw);
      path.header.frame_id = map_frame_;
      path.header.stamp = stamp;
      path_pub.publish(path);
    }
  }

  static double rmse2d(const TrajStats& stats)
  {
    return stats.samples > 0 ? std::sqrt(stats.sum_sq_xy / static_cast<double>(stats.samples)) : 0.0;
  }

  static double rmse3d(const TrajStats& stats)
  {
    return stats.samples > 0 ? std::sqrt(stats.sum_sq_xyz / static_cast<double>(stats.samples)) : 0.0;
  }

  static double meanAbsZ(const TrajStats& stats)
  {
    return stats.samples > 0 ? stats.sum_abs_z / static_cast<double>(stats.samples) : 0.0;
  }

  void appendPath(nav_msgs::Path& path, const ros::Time& stamp, const Eigen::Vector3d& p, double yaw)
  {
    geometry_msgs::PoseStamped ps;
    ps.header.stamp = stamp;
    ps.header.frame_id = map_frame_;
    ps.pose.position.x = p.x();
    ps.pose.position.y = p.y();
    ps.pose.position.z = p.z();
    ps.pose.orientation = quatFromYaw(yaw);
    path.poses.push_back(ps);
    if (max_path_points_ > 0 && static_cast<int>(path.poses.size()) > max_path_points_)
    {
      path.poses.erase(path.poses.begin(), path.poses.begin() + (path.poses.size() - max_path_points_));
    }
  }

  void statsTimerCb(const ros::TimerEvent&)
  {
    std_msgs::String msg;
    std::ostringstream ss;
    ss << "{";
    appendStatsJson(ss, raw_);
    ss << ",";
    appendStatsJson(ss, semantic_);
    ss << ",\"gt_samples\":" << gt_buffer_.size();
    ss << ",\"origin_set\":" << (has_origin_ ? "true" : "false");
    ss << "}";
    msg.data = ss.str();
    stats_pub_.publish(msg);

    ROS_INFO_THROTTLE(2.0,
                      "RMSE raw: 2D=%.3f 3D=%.3f n=%lu recv=%lu no_gt=%lu wait_align=%lu spread=%.2f/%.2f | semantic: 2D=%.3f 3D=%.3f n=%lu recv=%lu no_gt=%lu wait_align=%lu spread=%.2f/%.2f",
                      rmse2d(raw_), rmse3d(raw_), static_cast<unsigned long>(raw_.samples),
                      static_cast<unsigned long>(raw_.received),
                      static_cast<unsigned long>(raw_.no_gt_match),
                      static_cast<unsigned long>(raw_.waiting_alignment),
                      raw_.last_alignment_spread, raw_.last_alignment_gt_spread,
                      rmse2d(semantic_), rmse3d(semantic_), static_cast<unsigned long>(semantic_.samples),
                      static_cast<unsigned long>(semantic_.received),
                      static_cast<unsigned long>(semantic_.no_gt_match),
                      static_cast<unsigned long>(semantic_.waiting_alignment),
                      semantic_.last_alignment_spread, semantic_.last_alignment_gt_spread);
  }

  void appendStatsJson(std::ostringstream& ss, const TrajStats& stats) const
  {
    ss << "\"" << stats.name << "\":{";
    ss << "\"topic\":\"" << stats.topic << "\",";
    ss << "\"aligned\":" << (stats.align.ready ? "true" : "false") << ",";
    ss << "\"received\":" << stats.received << ",";
    ss << "\"no_gt_match\":" << stats.no_gt_match << ",";
    ss << "\"latest_gt_fallback\":" << stats.latest_gt_fallback << ",";
    ss << "\"waiting_alignment\":" << stats.waiting_alignment << ",";
    ss << "\"alignment_pairs\":" << stats.alignment_pairs.size() << ",";
    ss << "\"samples\":" << stats.samples << ",";
    ss << "\"rmse_2d\":" << rmse2d(stats) << ",";
    ss << "\"rmse_3d\":" << rmse3d(stats) << ",";
    ss << "\"mean_abs_z\":" << meanAbsZ(stats) << ",";
    ss << "\"max_2d\":" << stats.max_2d << ",";
    ss << "\"max_3d\":" << stats.max_3d << ",";
    ss << "\"alignment_spread\":" << stats.last_alignment_spread << ",";
    ss << "\"alignment_gt_spread\":" << stats.last_alignment_gt_spread << ",";
    ss << "\"last_gt_dt\":" << stats.last_gt_dt << ",";
    ss << "\"last_reject_dt\":" << stats.last_reject_dt << ",";
    ss << "\"last_err_2d\":" << stats.last_err_2d << ",";
    ss << "\"last_err_3d\":" << stats.last_err_3d << ",";
    ss << "\"align_yaw_deg\":" << stats.align.yaw * 180.0 / M_PI << ",";
    ss << "\"align_tx\":" << stats.align.t.x() << ",";
    ss << "\"align_ty\":" << stats.align.t.y() << ",";
    ss << "\"align_tz\":" << stats.align.z_offset;
    ss << "}";
  }

  void initCsv()
  {
    if (!record_csv_)
    {
      return;
    }
    if (::mkdir(record_dir_.c_str(), 0755) != 0 && errno != EEXIST)
    {
      ROS_WARN("Failed creating record_dir=%s; CSV recording disabled", record_dir_.c_str());
      record_csv_ = false;
      return;
    }

    gt_csv_.open((record_dir_ + "/ground_truth.csv").c_str());
    raw_.csv.open((record_dir_ + "/raw_fast_livo2_eval.csv").c_str());
    semantic_.csv.open((record_dir_ + "/semantic_corrected_eval.csv").c_str());

    if (gt_csv_.is_open())
    {
      gt_csv_ << "stamp,gt_x,gt_y,gt_z,gt_yaw,lat,lon,alt,rtk_cov_xy,rtk_cov_z,imu_ok\n";
    }
    const std::string header =
        "stamp,gt_x,gt_y,gt_z,odom_x,odom_y,odom_z,aligned_x,aligned_y,aligned_z,"
        "err_x,err_y,err_z,err_2d,err_3d,rmse_2d,rmse_3d,gt_dt\n";
    if (raw_.csv.is_open())
    {
      raw_.csv << header;
    }
    if (semantic_.csv.is_open())
    {
      semantic_.csv << header;
    }
  }

  void recordGt(const GroundTruthSample& gt, const sensor_msgs::NavSatFix& fix, bool imu_ok)
  {
    if (!record_csv_ || !gt_csv_.is_open())
    {
      return;
    }
    gt_csv_ << std::fixed << std::setprecision(9)
            << gt.stamp.toSec() << ","
            << gt.p.x() << "," << gt.p.y() << "," << gt.p.z() << ","
            << gt.yaw << ","
            << fix.latitude << "," << fix.longitude << "," << fix.altitude << ","
            << gt.pos_var_xy << "," << gt.pos_var_z << ","
            << (imu_ok ? 1 : 0) << "\n";
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber rtk_sub_;
  ros::Subscriber imu_sub_;
  ros::Subscriber raw_sub_;
  ros::Subscriber semantic_sub_;
  ros::Publisher gt_pub_;
  ros::Publisher stats_pub_;
  ros::Publisher gt_path_pub_;
  ros::Publisher raw_path_pub_;
  ros::Publisher semantic_path_pub_;
  ros::Timer stats_timer_;

  std::string rtk_fix_topic_;
  std::string imu_topic_;
  std::string raw_odom_topic_;
  std::string semantic_odom_topic_;
  std::string gt_odom_topic_;
  std::string stats_topic_;
  std::string map_frame_;
  std::string alignment_mode_;
  std::string gt_match_mode_;
  std::string record_dir_;

  bool use_fixed_origin_ = false;
  bool use_imu_orientation_ = true;
  bool has_origin_ = false;
  bool has_imu_ = false;
  bool has_integrated_imu_yaw_ = false;
  bool has_gt_ = false;
  bool record_csv_ = true;
  bool publish_paths_ = true;
  bool alignment_fallback_to_first_pose_ = true;
  int min_navsat_status_ = sensor_msgs::NavSatStatus::STATUS_FIX;
  int alignment_min_samples_ = 30;
  int alignment_fallback_samples_ = 30;
  int max_alignment_samples_ = 300;
  int max_path_points_ = 20000;
  double origin_latitude_ = 0.0;
  double origin_longitude_ = 0.0;
  double origin_altitude_ = 0.0;
  double max_imu_age_sec_ = 0.20;
  double max_imu_integration_dt_ = 0.05;
  double imu_yaw_offset_deg_ = 0.0;
  double imu_gyro_z_sign_ = 1.0;
  double imu_gyro_z_bias_ = 0.0;
  double default_rtk_xy_std_ = 0.03;
  double default_rtk_z_std_ = 0.06;
  double default_imu_yaw_std_deg_ = 1.0;
  double max_gt_match_dt_ = 0.05;
  double max_latest_gt_age_sec_ = 0.50;
  double gt_buffer_sec_ = 30.0;
  double alignment_min_distance_m_ = 5.0;
  double publish_rate_ = 1.0;
  double latest_imu_yaw_ = 0.0;
  bool integrate_imu_yaw_when_orientation_invalid_ = true;
  ros::Time latest_imu_stamp_;
  ros::WallTime latest_gt_wall_;
  sensor_msgs::Imu latest_imu_;
  GroundTruthSample latest_gt_;
  std::deque<GroundTruthSample> gt_buffer_;

  Eigen::Vector3d origin_ecef_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_ecef_enu_ = Eigen::Matrix3d::Identity();

  TrajStats raw_;
  TrajStats semantic_;
  nav_msgs::Path gt_path_;
  nav_msgs::Path raw_path_;
  nav_msgs::Path semantic_path_;
  std::ofstream gt_csv_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trajectory_groundtruth_evaluator");
  TrajectoryGroundTruthEvaluator node;
  ros::spin();
  return 0;
}
