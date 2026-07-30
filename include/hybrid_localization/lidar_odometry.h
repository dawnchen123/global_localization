#ifndef HYBRID_LOCALIZATION_LIDAR_ODOMETRY_H
#define HYBRID_LOCALIZATION_LIDAR_ODOMETRY_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hybrid_localization
{

using PointVector = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using Vector18d = Eigen::Matrix<double, 18, 1>;
using Matrix18d = Eigen::Matrix<double, 18, 18>;

struct TimedPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  double time_from_scan_end = 0.0;
};

using TimedPointVector = std::vector<TimedPoint, Eigen::aligned_allocator<TimedPoint>>;

struct ImuSample
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stamp = 0.0;
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

struct WheelSample
{
  double stamp = 0.0;
  double forward_speed = 0.0;
};

struct LidarOdometryOptions
{
  // LiDAR measurement update and incremental local map.
  double scan_voxel_size = 0.30;
  double map_voxel_size = 0.45;
  double map_insert_voxel_size = 0.20;
  double max_correspondence_distance = 1.5;
  double max_plane_distance = 0.35;
  double plane_max_eigen_ratio = 0.20;
  double lidar_range_noise = 0.03;
  double lidar_beam_noise = 0.0015;
  double lidar_measurement_noise = 0.05;
  double huber_delta = 0.15;
  double max_rmse = 0.30;
  double min_inlier_ratio = 0.15;
  double convergence_translation = 0.0015;
  double convergence_rotation_deg = 0.03;
  double degeneracy_eigen_ratio = 1e-4;
  double solver_damping = 1e-7;
  double max_translation_per_scan = 2.0;
  double max_rotation_per_scan_deg = 20.0;
  // The measurement scheduler can process a scan later than its acquisition
  // period. Keep the inter-scan gate physical instead of rejecting valid
  // motion solely because several scan periods elapsed before registration.
  double max_translation_speed = 4.0;
  double max_rotation_speed_deg = 40.0;
  // Registration must remain close to the IMU-propagated state.  This is a
  // separate gate from physical inter-scan motion and prevents one bad plane
  // alignment from poisoning both the ESKF state and the local map.
  double max_lidar_correction_translation = 0.40;
  double max_lidar_correction_rotation_deg = 4.0;
  // After a run of rejected LiDAR updates, bound the otherwise unobservable
  // IMU-only motion. This prevents an extended registration outage from
  // turning into an unbounded vertical or horizontal trajectory excursion.
  // Set lidar_loss_hold_after_rejections to zero to disable this protection.
  int lidar_loss_hold_after_rejections = 3;
  // Once loss persists beyond this many rejected scans, freeze the pose at
  // the last trusted LiDAR update rather than publishing a plausible-looking
  // but unconstrained trajectory. Set to zero to keep bounded propagation.
  int lidar_loss_freeze_after_rejections = 12;
  double lidar_loss_max_vertical_offset = 0.35;
  double lidar_loss_max_horizontal_speed = 3.0;
  double lidar_loss_max_horizontal_step = 0.75;
  double lidar_loss_velocity_decay = 0.98;
  // Degenerate geometry is usable only when its residual support is notably
  // stronger than the normal acceptance threshold.
  double degenerate_min_inlier_ratio = 0.28;
  double degenerate_max_rmse = 0.18;
  double map_insertion_max_plane_distance = 0.25;
  double local_map_radius = 70.0;
  double max_plane_variance = 0.035;
  double plane_uncertainty_scale = 1.0;
  double plane_fit_residual_gate = 0.15;
  bool use_point_knn_plane = false;
  bool use_compatible_voxel_plane = false;
  // Smooth voxel planes are cheap and remain the primary association model.
  // When a sparse or recently entered part of the local submap cannot form a
  // stable voxel plane, recover the correspondence from the retained point
  // samples rather than discarding the observation outright.
  bool point_knn_fallback = false;
  // Limit expensive KNN fallback queries per registration iteration. Zero
  // keeps the legacy unbounded behavior; production sensor configs set a
  // budget and retain spatially broad support through the normal voxel path.
  int point_knn_fallback_max_queries = 0;
  // A wide-FOV LiDAR naturally contains many points outside the current local
  // submap.  These thresholds permit a low raw inlier fraction only when the
  // absolute support is strong, spatially distributed, and has a low residual.
  // Set strong_support_min_correspondences to zero to disable this path.
  int strong_support_min_correspondences = 0;
  int strong_support_min_azimuth_sectors = 4;
  double strong_support_max_rmse = 0.0;
  // A short outage can leave the IMU prediction just outside the normal
  // correction gate.  A recovery update is allowed only with strong support;
  // zero values keep the normal correction gates unchanged.
  int recovery_after_rejections = 0;
  double recovery_max_lidar_correction_translation = 0.0;
  double recovery_max_lidar_correction_rotation_deg = 0.0;
  int max_iterations = 5;
  int min_scan_points = 200;
  int min_correspondences = 100;
  int max_scan_points = 6000;
  int max_map_points = 80000;
  int normal_neighbor_voxels = 2;
  int min_normal_neighbors = 6;
  int max_plane_neighbors = 20;
  int min_voxel_plane_points = 8;
  int max_voxel_points = 120;
  int max_voxel_samples = 12;

  // FAST-LIO style error-state propagation: [R, p, v, bg, ba, g].
  bool imu_enabled = true;
  bool imu_init_require_stationary = true;
  bool auto_acceleration_scale = true;
  double gravity_magnitude = 9.81;
  double acceleration_scale = 1.0;
  double imu_init_duration = 1.5;
  int imu_init_samples = 200;
  double imu_init_max_acc_std = 0.80;
  double imu_init_max_gyro_std = 0.08;
  double imu_init_max_gyro_bias = 0.20;
  double imu_max_gap = 0.03;
  double imu_buffer_duration = 5.0;
  double gyro_noise = 0.015;
  double acceleration_noise = 0.10;
  double gyro_bias_random_walk = 0.00010;
  double acceleration_bias_random_walk = 0.0010;
  double gravity_random_walk = 0.00001;
  double max_gyro_bias = 0.50;
  double max_acceleration_bias = 3.0;

  // Ground-vehicle velocity update. The forward wheel speed is measured in
  // the body frame; lateral/vertical components encode non-holonomic motion.
  bool wheel_enabled = false;
  double wheel_speed_scale = 1.0;
  double wheel_max_age = 0.12;
  double wheel_max_speed = 12.0;
  double wheel_forward_noise = 0.18;
  double wheel_lateral_noise = 0.15;
  double wheel_vertical_noise = 0.25;
  double wheel_huber_delta = 1.5;
  double wheel_buffer_duration = 5.0;

  // Asynchronous image update. The visual frontend supplies a robust
  // photometric normal equation over the current body pose.
  bool visual_enabled = false;
  int visual_max_iterations = 4;
  int visual_min_landmarks = 20;
  int visual_min_residuals = 240;
  double visual_max_rmse = 1.20;
  double visual_max_translation_step = 0.35;
  double visual_max_rotation_step_deg = 4.0;
  double visual_convergence_translation = 0.0005;
  double visual_convergence_rotation_deg = 0.01;
  double visual_solver_damping = 1e-6;
};

struct LidarOdometryResult
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool initialized = false;
  bool imu_initialized = false;
  bool accepted = false;
  bool converged = false;
  bool degenerate = false;
  bool used_imu = false;
  bool used_wheel = false;
  bool map_updated = false;
  bool loss_limited = false;
  bool loss_frozen = false;
  bool strong_support = false;
  bool recovery_mode = false;
  double stamp = 0.0;
  double rmse = std::numeric_limits<double>::infinity();
  double inlier_ratio = 0.0;
  double imu_init_progress = 0.0;
  double acceleration_scale = 1.0;
  double wheel_speed = 0.0;
  double wheel_velocity_residual = 0.0;
  int correspondences = 0;
  int scan_points = 0;
  int correspondence_azimuth_sectors = 0;
  int point_knn_fallback_queries = 0;
  int point_knn_fallback_matches = 0;
  int iterations = 0;
  int imu_samples = 0;
  int consecutive_rejections = 0;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d relative_pose = Eigen::Isometry3d::Identity();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acceleration_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  Eigen::Matrix<double, 6, 6> covariance =
      Eigen::Matrix<double, 6, 6>::Identity();
  PointVector deskewed_points;
  std::string reject_reason;
};

struct VisualPoseLinearization
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  int landmarks = 0;
  int residuals = 0;
  double rmse = std::numeric_limits<double>::infinity();
  double mean_ncc = 0.0;
  Eigen::Matrix<double, 6, 6> hessian =
      Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> gradient =
      Eigen::Matrix<double, 6, 1>::Zero();
  std::string reason = "not_linearized";
};

using VisualPoseLinearizer =
    std::function<VisualPoseLinearization(const Eigen::Isometry3d &)>;

struct VisualUpdateResult
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool propagated = false;
  bool accepted = false;
  bool converged = false;
  double stamp = 0.0;
  double rmse = std::numeric_limits<double>::infinity();
  double mean_ncc = 0.0;
  int landmarks = 0;
  int residuals = 0;
  int iterations = 0;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> covariance =
      Eigen::Matrix<double, 6, 6>::Identity();
  std::string reason = "not_processed";
};

class LidarOdometry
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit LidarOdometry(const LidarOdometryOptions &options = LidarOdometryOptions());

  void reset();
  void addImuSample(const ImuSample &sample);
  void addWheelSample(const WheelSample &sample);
  LidarOdometryResult processScan(const TimedPointVector &points, double scan_end_stamp);
  LidarOdometryResult processScan(const PointVector &body_points, double scan_end_stamp);
  VisualUpdateResult processVisual(double stamp,
                                   const VisualPoseLinearizer &linearizer);

  bool initialized() const { return map_initialized_; }
  bool imuInitialized() const { return imu_initialized_; }
  const Eigen::Isometry3d &pose() const { return pose_cache_; }
  const Eigen::Vector3d &velocity() const { return state_.velocity; }
  const Eigen::Vector3d &gyroBias() const { return state_.gyro_bias; }
  const Eigen::Vector3d &accelerationBias() const { return state_.acceleration_bias; }
  const Eigen::Vector3d &gravity() const { return state_.gravity; }
  std::size_t mapPointCount() const { return map_voxels_.size(); }
  std::size_t keyframeCount() const { return accepted_scan_count_; }
  double stateStamp() const { return state_stamp_; }

private:
  struct VoxelKey
  {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey &other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey &key) const;
  };

  struct State
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
    Matrix18d covariance = Matrix18d::Identity();
  };

  struct ImuPose
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double stamp = 0.0;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  };

  struct MapVoxel
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int count = 0;
    int last_seen_scan = 0;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double plane_variance = std::numeric_limits<double>::infinity();
    double plane_ratio = std::numeric_limits<double>::infinity();
    bool plane_valid = false;
    int sample_cursor = 0;
    PointVector samples;
  };

  struct PlaneMatch
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double variance = 0.0;
    double nearest_squared_distance = std::numeric_limits<double>::infinity();
  };

  static Eigen::Matrix3d skew(const Eigen::Vector3d &vector);
  static Eigen::Matrix3d expSO3(const Eigen::Vector3d &rotation_vector);
  static Eigen::Vector3d logSO3(const Eigen::Matrix3d &rotation);
  static double rotationDegrees(const Eigen::Matrix3d &rotation);
  static double huberWeight(double residual, double delta);

  VoxelKey voxelKey(const Eigen::Vector3d &point, double voxel_size) const;
  PointVector voxelDownsample(const PointVector &points, double voxel_size,
                              int max_points) const;
  Eigen::Isometry3d statePose(const State &state) const;
  Vector18d stateDifference(const State &state, const State &reference) const;
  void applyError(State &state, const Vector18d &error) const;

  bool initializeImuIfReady();
  bool propagateTo(double end_stamp, std::vector<ImuPose,
                   Eigen::aligned_allocator<ImuPose>> &trajectory);
  void propagateSegment(State &state, const Eigen::Vector3d &angular_velocity,
                        const Eigen::Vector3d &acceleration, double dt) const;
  void predictWithoutImu(double end_stamp,
                         std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> &trajectory);
  ImuPose interpolatePose(double stamp, const std::vector<ImuPose,
                          Eigen::aligned_allocator<ImuPose>> &trajectory) const;
  PointVector deskewScan(const TimedPointVector &points, double scan_end_stamp,
                         const std::vector<ImuPose,
                         Eigen::aligned_allocator<ImuPose>> &trajectory) const;

  bool findLocalPlane(const Eigen::Vector3d &world_point, PlaneMatch &match) const;
  bool findSmoothVoxelPlane(const Eigen::Vector3d &world_point, PlaneMatch &match) const;
  bool findCompatibleVoxelPlane(const Eigen::Vector3d &world_point,
                                PlaneMatch &match) const;
  bool findPointKnnPlane(const Eigen::Vector3d &world_point, PlaneMatch &match) const;
  void insertMapPoints(const PointVector &body_points, const State &state,
                       bool filter_existing);
  void updateVoxel(MapVoxel &voxel, const Eigen::Vector3d &point);
  void updateVoxelPlane(MapVoxel &voxel);
  bool wheelMeasurement(double stamp, double *forward_speed) const;
  void pruneMap();
  void pruneImu(double stamp);
  bool applyLidarLossProtection(const State &state_before_scan,
                                double scan_end_stamp);
  void fillResultState(LidarOdometryResult &result) const;
  void recordPropagationTrajectory(const std::vector<ImuPose,
                                   Eigen::aligned_allocator<ImuPose>> &trajectory);
  void recordCurrentPose();
  std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> deskewTrajectory(
      double end_stamp) const;

  LidarOdometryOptions options_;
  bool imu_initialized_ = false;
  bool map_initialized_ = false;
  double state_stamp_ = 0.0;
  double acceleration_scale_ = 1.0;
  double imu_init_progress_ = 0.0;
  int consecutive_rejections_ = 0;
  int accepted_scan_count_ = 0;
  bool lidar_loss_limited_ = false;
  bool lidar_loss_frozen_ = false;
  State state_;
  State last_accepted_state_;
  Eigen::Isometry3d pose_cache_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_scan_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d previous_scan_pose_ = Eigen::Isometry3d::Identity();
  double previous_scan_stamp_ = 0.0;

  std::deque<ImuSample, Eigen::aligned_allocator<ImuSample>> imu_buffer_;
  std::deque<ImuPose, Eigen::aligned_allocator<ImuPose>> propagation_history_;
  std::deque<WheelSample> wheel_buffer_;
  using MapVoxelPair = std::pair<const VoxelKey, MapVoxel>;
  std::unordered_map<VoxelKey, MapVoxel, VoxelKeyHash, std::equal_to<VoxelKey>,
                     Eigen::aligned_allocator<MapVoxelPair>> map_voxels_;
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_LIDAR_ODOMETRY_H
