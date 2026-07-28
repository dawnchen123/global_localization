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
  // Model the LiDAR return in its beam frame: range uncertainty is radial,
  // while the angular beam uncertainty grows tangentially with range. This
  // mirrors the directional point covariance used by FAST-LIVO-style updates
  // without changing the local-map or ESKF architecture.
  bool use_directional_lidar_covariance = false;
  Eigen::Vector3d lidar_origin_in_body = Eigen::Vector3d::Zero();
  double huber_delta = 0.15;
  // When positive, robustify the point-to-plane residual after whitening it
  // by the directional measurement covariance. The legacy metric Huber delta
  // remains active when this value is zero.
  double lidar_normalized_huber_delta = 0.0;
  // Maximum scalar NIS for an individual point-to-plane observation.
  // Non-positive values preserve the legacy residual-only association gate.
  double lidar_innovation_gate = 0.0;
  double max_rmse = 0.30;
  double min_inlier_ratio = 0.15;
  double convergence_translation = 0.0015;
  double convergence_rotation_deg = 0.03;
  double degeneracy_eigen_ratio = 1e-4;
  // The scan-to-map Hessian is rank deficient in corridors, planar roads, and
  // sparse views. Project each iterative pose correction onto its observable
  // eigenspace instead of turning an ill-conditioned least-squares step into
  // a spurious yaw, height, or lateral correction.
  double observability_eigen_ratio = 1e-4;
  int min_observable_directions = 3;
  // Mean point-to-plane normalized squared residual. A non-positive value
  // disables the corresponding gate.
  double max_mean_normalized_residual = 10.0;
  int map_insertion_min_observable_directions = 3;
  double map_insertion_max_mean_normalized_residual = 8.0;
  bool preserve_unobservable_covariance = true;
  // Project LiDAR information and residual gradients before the full-state
  // ESKF solve. This keeps scan-unobservable pose modes from conditioning
  // velocity, biases, or gravity through an inconsistent unprojected Hessian.
  bool project_lidar_information_to_observable_subspace = false;
  // The iterated full-state solve can otherwise absorb a scan-unobservable
  // yaw/roll/pitch correction into gyro bias through prior cross-covariance.
  // Reuse the pose observability projection for the gyro-bias mean increment.
  bool project_gyro_bias_update_to_observable_rotation = false;
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
  // Reject or down-weight a query that extrapolates too far beyond the
  // tangential support of its fitted plane. Zero disables each term and keeps
  // the legacy association model.
  double plane_support_radius_scale = 0.0;
  double plane_extrapolation_uncertainty_scale = 0.0;
  // Propagate finite-sample center and normal uncertainty into each query's
  // point-to-plane variance. The normal term grows with tangential distance
  // and with a small PCA eigenvalue gap. Zero preserves the legacy model.
  double plane_parameter_uncertainty_scale = 0.0;
  double plane_fit_residual_gate = 0.15;
  // Refit a smooth neighborhood after rejecting voxel means that do not
  // support its initial plane. This prevents wall/ground and curb/road
  // mixtures from becoming a numerically plausible correspondence plane.
  bool smooth_voxel_robust_refit = false;
  // Preserve the stable root-voxel model in planar regions, but retain a
  // bounded finer map for intersections and mixed root voxels. The fine map
  // is queried only when the root voxel cannot form a valid plane.
  bool use_adaptive_subvoxel_plane = false;
  double adaptive_subvoxel_scale = 0.5;
  int adaptive_subvoxel_search_radius = 1;
  // Non-positive derives a limit of twice max_map_points.
  int max_adaptive_subvoxels = 0;
  bool use_point_knn_plane = false;
  bool use_compatible_voxel_plane = false;
  // Zero derives the compatible-plane search from the full correspondence
  // distance. A positive value bounds hash lookups to that many voxel layers
  // before the smooth-plane fallback is attempted.
  int compatible_voxel_search_radius = 0;
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
  // Do not let an isolated post-outage registration write into the local map.
  // A positive value requires this many consecutive strong-support scans after
  // any rejected frame before map insertion resumes. Zero keeps legacy behavior.
  int recovery_map_insert_min_consecutive_strong_support = 0;
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
  // A mature voxel can otherwise absorb the estimator's own slowly drifting
  // poses forever. Freezing it makes the local map an independent reference.
  bool freeze_mature_voxels = false;
  // Insert accepted scans only after sufficient motion, while the maximum
  // interval still refreshes a stationary or slowly moving local map.
  double map_insertion_min_translation = 0.0;
  double map_insertion_min_rotation_deg = 0.0;
  double map_insertion_max_interval = 0.0;

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
  // For a stationary initialization the bias/gravity estimate is formed from
  // the sample mean. Its covariance is therefore sample_variance / N, not the
  // variance of one IMU sample. Keeping this configurable preserves a
  // conservative fallback for datasets that cannot start while stationary.
  bool imu_init_use_mean_covariance = false;
  double imu_init_gyro_bias_covariance_floor = 1e-8;
  double imu_init_acceleration_bias_covariance = 1e-4;
  double imu_init_gravity_covariance_floor = 1e-6;
  double imu_init_nonstationary_gravity_covariance = 1e-3;
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
  // Maximum scalar NIS for the measured forward wheel velocity. Lateral and
  // vertical non-holonomic constraints remain available when it is rejected.
  double wheel_forward_innovation_gate = 0.0;
  double wheel_buffer_duration = 5.0;
  // Position of the odometer reference point relative to the IMU/body origin,
  // expressed in the ROS body frame. The velocity measurement is compensated
  // by omega x lever_arm when synchronized IMU angular velocity is available.
  Eigen::Vector3d wheel_lever_arm = Eigen::Vector3d::Zero();
  bool wheel_compensate_angular_velocity = true;

  // Apply the error-coordinate reset Jacobian after injecting an iterated
  // LiDAR or visual correction into the nominal ESKF state.
  bool covariance_reset_enabled = true;

  // Asynchronous image update. The visual frontend supplies a robust
  // photometric normal equation over the current body pose.
  bool visual_enabled = false;
  int visual_max_iterations = 4;
  int visual_min_landmarks = 20;
  int visual_min_residuals = 240;
  double visual_max_rmse = 1.20;
  double visual_min_mean_ncc = 0.72;
  double visual_max_translation_step = 0.35;
  double visual_max_rotation_step_deg = 4.0;
  double visual_convergence_translation = 0.0005;
  double visual_convergence_rotation_deg = 0.01;
  double visual_solver_damping = 1e-6;
  // A pose measurement updates every ESKF state correlated with pose through
  // the prior covariance. Disabling this preserves the legacy pose-only mean
  // injection, but that mode is statistically inconsistent because the same
  // update still conditions the full covariance.
  bool visual_fuse_correlated_states = false;
  // Per-image bounds apply to the total correction relative to the propagated
  // state, not once per iterated solve step. Zero disables the corresponding
  // bound while retaining the estimator-wide state limits.
  double visual_max_velocity_step = 0.05;
  double visual_max_gyro_bias_step = 0.0005;
  double visual_max_acceleration_bias_step = 0.01;
  double visual_max_gravity_step = 0.005;
  // Direct monocular alignment has weak depth, roll, and pitch observability
  // on forward-looking road scenes. Restrict visual information to the axes
  // that are deliberately enabled, then keep only well-conditioned modes.
  bool visual_fuse_roll_pitch = true;
  bool visual_fuse_yaw = true;
  bool visual_fuse_translation_xy = true;
  bool visual_fuse_translation_z = true;
  // Keep the historical explicit Z path available, but allow the runtime
  // profile to request Z only when the LiDAR-depth photometric Hessian has
  // independent vertical information. This is intentionally opt-in because
  // a forward-facing monocular image is often depth-degenerate on roads.
  bool visual_fuse_translation_z_when_observable = false;
  // Squared projection of the world-Z axis onto the retained photometric
  // eigenspace, and its Schur-complement information ratio after yaw/XY.
  // Both are dimensionless values in [0, 1].
  double visual_z_min_projection = 0.92;
  double visual_z_min_conditional_information_ratio = 0.20;
  // Zero preserves the legacy explicit-Z behavior. A positive value bounds
  // each automatic Z update after it has passed the observability gate.
  double visual_max_z_step = 0.0;
  double visual_observability_eigen_ratio = 1e-4;
  int visual_min_observable_directions = 3;
  // Optional stricter gate for a direct visual update with exactly two
  // observable pose directions. Zero-valued thresholds disable the
  // corresponding check, preserving the historical three-direction profile.
  int visual_two_mode_min_landmarks = 0;
  int visual_two_mode_min_residuals = 0;
  double visual_two_mode_max_rmse = 0.0;
  double visual_two_mode_min_mean_ncc = 0.0;
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
  bool map_update_deferred = false;
  bool map_keyframe_selected = false;
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
  bool wheel_forward_rejected = false;
  double mean_normalized_residual = std::numeric_limits<double>::infinity();
  double mean_robust_weight = 0.0;
  double measurement_condition = std::numeric_limits<double>::infinity();
  int correspondences = 0;
  int innovation_rejections = 0;
  int observable_directions = 0;
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
  int observable_directions = 0;
  bool z_observable = false;
  bool z_fused = false;
  double z_projection = 0.0;
  double z_conditional_information_ratio = 0.0;
  double z_correction = 0.0;
  Eigen::Vector3d velocity_correction = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias_correction = Eigen::Vector3d::Zero();
  Eigen::Vector3d acceleration_bias_correction = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_correction = Eigen::Vector3d::Zero();
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
  const Matrix18d &stateCovariance() const { return state_.covariance; }
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
    Eigen::Vector3d plane_eigenvalues =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Matrix3d plane_eigenvectors = Eigen::Matrix3d::Identity();
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
  void resetCovarianceAfterInjection(State &state,
                                     const Vector18d &injected_error) const;

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
  bool findAdaptiveSubvoxelPlane(const Eigen::Vector3d &world_point,
                                 PlaneMatch &match) const;
  bool findSmoothVoxelPlane(const Eigen::Vector3d &world_point, PlaneMatch &match) const;
  bool findCompatibleVoxelPlane(const Eigen::Vector3d &world_point,
                                PlaneMatch &match) const;
  bool findPointKnnPlane(const Eigen::Vector3d &world_point, PlaneMatch &match) const;
  bool planeSupportUncertainty(const Eigen::Vector3d &world_point,
                               const Eigen::Vector3d &center,
                               const Eigen::Vector3d &normal,
                               const Eigen::Vector3d &eigenvalues,
                               const Eigen::Matrix3d &eigenvectors,
                               double effective_support,
                               double *additional_variance) const;
  void insertMapPoints(const PointVector &body_points, const State &state,
                       bool filter_existing);
  void updateVoxel(MapVoxel &voxel, const Eigen::Vector3d &point);
  void updateVoxelPlane(MapVoxel &voxel);
  bool shouldInsertMap(const State &state, double stamp) const;
  bool wheelMeasurement(double stamp, double *forward_speed) const;
  bool angularVelocityMeasurement(double stamp,
                                  Eigen::Vector3d *angular_velocity) const;
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
  bool recovery_map_guard_active_ = false;
  int recovery_map_trusted_scan_count_ = 0;
  State state_;
  State last_accepted_state_;
  Eigen::Isometry3d pose_cache_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_scan_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d previous_scan_pose_ = Eigen::Isometry3d::Identity();
  double previous_scan_stamp_ = 0.0;
  bool have_last_map_insert_pose_ = false;
  Eigen::Isometry3d last_map_insert_pose_ = Eigen::Isometry3d::Identity();
  double last_map_insert_stamp_ = 0.0;

  std::deque<ImuSample, Eigen::aligned_allocator<ImuSample>> imu_buffer_;
  std::deque<ImuPose, Eigen::aligned_allocator<ImuPose>> propagation_history_;
  std::deque<WheelSample> wheel_buffer_;
  using MapVoxelPair = std::pair<const VoxelKey, MapVoxel>;
  using MapVoxelMap =
      std::unordered_map<VoxelKey, MapVoxel, VoxelKeyHash,
                         std::equal_to<VoxelKey>,
                         Eigen::aligned_allocator<MapVoxelPair>>;
  MapVoxelMap map_voxels_;
  MapVoxelMap adaptive_subvoxels_;
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_LIDAR_ODOMETRY_H
