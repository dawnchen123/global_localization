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
  // Ranger MINI exposes four wheel speeds. The front/rear averaged
  // right-minus-left speed is retained as an optional independent yaw-rate
  // observation; generic wheel messages leave these values non-finite.
  double differential_speed = std::numeric_limits<double>::quiet_NaN();
  double differential_disagreement =
      std::numeric_limits<double>::infinity();
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
  // FAST-LIVO2-style iterated updates rematch after the first small increment
  // instead of accepting correspondences from a single linearization.  Two
  // consecutive confirmations make the state used for validation and map
  // insertion coincide with the state at which residuals were evaluated.
  int convergence_confirmation_iterations = 2;
  bool require_convergence_for_acceptance = false;
  double max_iteration_translation = 0.40;
  double max_iteration_rotation_deg = 3.0;
  double degeneracy_eigen_ratio = 1e-4;
  // The scan-to-map Hessian is rank deficient in corridors, planar roads, and
  // sparse views. This ratio identifies weak directions for diagnostics,
  // covariance preservation, map-write gating, and optional hard projection.
  double observability_eigen_ratio = 1e-4;
  int min_observable_directions = 3;
  // Marginalize translation from the pose Hessian before judging rotational
  // support. This avoids treating a well-observed position direction as
  // evidence that yaw/roll/pitch are also constrained. When enabled, weak
  // rotational LiDAR modes retain only weak_rotation_information_scale of
  // their information and therefore fall back smoothly to IMU propagation.
  double rotation_observability_eigen_ratio = 1e-4;
  bool project_lidar_information_to_observable_rotation = false;
  double weak_rotation_information_scale = 0.0;
  // Mean point-to-plane normalized squared residual. A non-positive value
  // disables the corresponding gate.
  double max_mean_normalized_residual = 10.0;
  int map_insertion_min_observable_directions = 3;
  // Independent rotational map-write gates. Zero disables each gate and
  // preserves the historical six-dimensional observability policy.
  int map_insertion_min_observable_rotation_directions = 0;
  double map_insertion_min_yaw_observability = 0.0;
  double map_insertion_max_mean_normalized_residual = 8.0;
  bool map_insertion_require_convergence = false;
  // Independent map-write gates.  Zero disables the corresponding gate.
  // They allow a bounded state correction to remain usable without baking a
  // marginal turn registration into the persistent local map.
  double map_insertion_max_lidar_correction_translation = 0.0;
  double map_insertion_max_lidar_correction_rotation_deg = 0.0;
  bool preserve_unobservable_covariance = true;
  // Optional hard projection of LiDAR information and residual gradients
  // before the full-state ESKF solve. The normal FAST-LIVO2-style path leaves
  // this false: the full Hessian and ESKF prior naturally give weak directions
  // small gain, while covariance and map-write guards still use observability.
  bool project_lidar_information_to_observable_subspace = false;
  // The iterated full-state solve can otherwise absorb a scan-unobservable
  // yaw/roll/pitch correction into gyro bias through prior cross-covariance.
  // Reuse the pose observability projection for the gyro-bias mean increment.
  bool project_gyro_bias_update_to_observable_rotation = false;
  // Bound hidden-state corrections induced by one iterated LiDAR/wheel scan.
  // The defaults preserve the previous hard-coded limits. A ground-vehicle
  // profile with a stationary initialization may keep gravity fixed in its
  // initial world frame while still estimating the accelerometer bias.
  bool lidar_update_acceleration_bias = true;
  // Acceleration bias is only indirectly observable through state
  // cross-covariance. Ground vehicles may require a robust low-curvature
  // wheel/gyro window before allowing a scan to modify this hidden state.
  bool lidar_acceleration_bias_require_stable_wheel_motion = false;
  bool lidar_update_gravity = true;
  double max_lidar_velocity_step = 1.0;
  double max_lidar_gyro_bias_step = 0.02;
  double max_lidar_acceleration_bias_step = 0.10;
  double max_lidar_gravity_step = 0.05;
  double solver_damping = 1e-7;
  double max_translation_per_scan = 2.0;
  double max_rotation_per_scan_deg = 20.0;
  // The measurement scheduler can process a scan later than its acquisition
  // period. Keep the inter-scan gate physical instead of rejecting valid
  // motion solely because several scan periods elapsed before registration.
  double max_translation_speed = 4.0;
  double max_rotation_speed_deg = 40.0;
  // A fixed per-scan rotation gate rejects legitimate sharp turns when CPU
  // load stretches the interval between processed scans. When enabled and IMU
  // propagation covered the interval, admit the propagated rotation plus a
  // bounded margin. The independent LiDAR-correction gate remains active.
  bool turn_aware_motion_gate_enabled = false;
  double turn_aware_rotation_margin_deg = 3.0;
  double turn_aware_max_rotation_deg = 90.0;
  double turn_aware_max_scan_dt = 1.5;
  // A modest correction-gate expansion is allowed only when synchronized
  // wheel differential and IMU yaw rates agree.
  double turn_aware_min_yaw_rate = 0.20;
  double turn_aware_lidar_correction_rotation_deg = 0.0;
  double turn_aware_wheel_imu_max_yaw_rate_difference = 0.20;
  // Registration must remain close to the IMU-propagated state.  This is a
  // separate gate from physical inter-scan motion and prevents one bad plane
  // alignment from poisoning both the ESKF state and the local map.
  double max_lidar_correction_translation = 0.40;
  double max_lidar_correction_rotation_deg = 4.0;
  // Mahalanobis gate for the aggregate scan rotation correction relative to
  // the IMU prediction. The angular floor prevents an overconfident filter
  // from rejecting every small calibration/noise correction. Zero NIS gate
  // disables this check.
  double lidar_rotation_correction_nis_gate = 0.0;
  double lidar_rotation_correction_std_floor_deg = 0.5;
  // A sequence of individually small but same-sign scan corrections can evade
  // the per-frame NIS gate and slowly rotate both the state and its map. Keep
  // a time-windowed sum about the gravity axis; zero disables this guard.
  double lidar_yaw_correction_window_sec = 0.0;
  double max_cumulative_lidar_yaw_correction_deg = 0.0;
  // Instead of rejecting the complete scan after the cumulative yaw limit is
  // crossed, clamp only the gravity-axis component of the iterated LiDAR
  // correction. Translation and roll/pitch remain available to the ESKF.
  // The legacy whole-scan rejection remains the default when this is false.
  bool limit_cumulative_lidar_yaw_correction = false;
  // Information retained along the limited yaw axis. Zero falls back to the
  // propagated IMU yaw covariance; a small positive value keeps a weak LiDAR
  // contribution without allowing repeated map corrections to dominate.
  double limited_lidar_yaw_information_scale = 0.0;
  // A limited pose is useful for state correction, but its yaw disagreement
  // must not be written back into the persistent map.
  bool defer_map_when_lidar_yaw_limited = true;
  // After a run of rejected LiDAR updates, bound the otherwise unobservable
  // IMU-only motion. This prevents an extended registration outage from
  // turning into an unbounded vertical or horizontal trajectory excursion.
  // Set lidar_loss_hold_after_rejections to zero to disable this protection.
  int lidar_loss_hold_after_rejections = 3;
  // During a LiDAR outage, use a synchronized wheel speed with the propagated
  // IMU attitude to keep the prediction inside the local-map capture range.
  // This mirrors FAST-LIVO2's continuous propagation through weak scans while
  // map insertion remains disabled. If no wheel sample is available, the
  // bounded inertial-velocity fallback below is used.
  bool lidar_loss_use_wheel_dead_reckoning = true;
  // Once loss persists beyond this many rejected scans, freeze the pose only
  // when wheel dead reckoning is unavailable. Set to zero to always keep the
  // bounded propagation alive so registration can reacquire a moving sensor.
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
  // Preserve voxels outside the current local radius so a later revisit can
  // register against the original geometry. The radius still bounds points
  // admitted from each scan, and max_map_points remains a hard memory limit.
  bool retain_global_map = false;
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
  // Point-to-plane association is read-only with respect to the local map and
  // can be distributed like FAST-LIVO2's BuildResidualListOMP. Keep the
  // production limit deliberately small so semantics and ROS callbacks retain
  // CPU headroom during long bag replays.
  int registration_threads = 1;
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
  // Relative EMA gain after a valid planar voxel reaches max_voxel_points.
  // One preserves the historical update rate and zero holds its plane
  // statistics fixed. Non-planar voxels always keep the full gain so mixed
  // cells can recover when later views provide cleaner support.
  double mature_voxel_update_gain = 1.0;
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
  // Convert corrected right-minus-left wheel speed to body yaw rate. A zero
  // scale disables this observation. The forward leakage term compensates a
  // small left/right wheel-scale mismatch before the yaw conversion.
  double wheel_yaw_rate_scale = 0.0;
  // Keep wheel-yaw consistency available for turn gating without necessarily
  // writing wheel-radius/track-width error into the IMU gyro bias state.
  bool wheel_yaw_rate_fuse_gyro_bias = true;
  // Relative uncertainty of the differential-wheel yaw scale. Its variance
  // contribution grows with squared turn rate, retaining gyro-bias
  // observability while smoothly reducing wheel authority in sharp turns.
  double wheel_yaw_rate_relative_scale_uncertainty = 0.0;
  // A positive window replaces the per-frame wheel/gyro bias factor with a
  // robust estimate formed only from low-curvature samples. This anchors gyro
  // bias on straight motion without writing turn-radius-dependent wheel-scale
  // error into the inertial state. Zero preserves the legacy direct factor.
  double wheel_yaw_bias_window_sec = 0.0;
  int wheel_yaw_bias_min_samples = 8;
  double wheel_yaw_bias_max_abs_rate = 0.0;
  double wheel_yaw_bias_max_mad = 0.0;
  double wheel_yaw_bias_noise_floor = 0.01;
  // Remove a fixed wheel-yaw offset using the initialized IMU gyro bias when
  // the first valid low-curvature window becomes available. This is useful
  // when unequal wheel radii create an apparent constant yaw rate.
  bool wheel_yaw_bias_calibrate_offset = false;
  double wheel_differential_forward_leakage = 0.0;
  double wheel_yaw_rate_noise = 0.10;
  double wheel_yaw_rate_huber_delta = 1.5;
  double wheel_yaw_rate_innovation_gate = 16.0;
  double wheel_yaw_rate_min_speed = 0.30;
  double wheel_yaw_rate_max_abs = 2.0;
  double wheel_yaw_rate_max_imu_difference = 0.0;
  double wheel_differential_max_disagreement = 0.08;
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
  bool visual_require_convergence = false;
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
  bool final_linearization_valid = false;
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
  bool used_wheel_yaw_rate = false;
  bool wheel_yaw_rate_rejected = false;
  double wheel_yaw_rate_effective_noise = 0.0;
  double wheel_yaw_rate = 0.0;
  double imu_yaw_rate = 0.0;
  double wheel_yaw_rate_residual = 0.0;
  int wheel_yaw_bias_window_samples = 0;
  double wheel_yaw_bias_observation = 0.0;
  double wheel_yaw_bias_raw_observation = 0.0;
  double wheel_yaw_bias_offset = 0.0;
  bool wheel_yaw_bias_offset_calibrated = false;
  double wheel_yaw_bias_mad = std::numeric_limits<double>::infinity();
  bool acceleration_bias_update_allowed = false;
  Eigen::Vector3d acceleration_bias_correction = Eigen::Vector3d::Zero();
  bool turn_aware_gate_active = false;
  double expected_rotation_deg = 0.0;
  double rotation_motion_gate_deg = 0.0;
  double lidar_correction_rotation_gate_deg = 0.0;
  double lidar_rotation_correction_nis =
      std::numeric_limits<double>::infinity();
  double lidar_yaw_correction_deg = 0.0;
  double cumulative_lidar_yaw_correction_deg = 0.0;
  bool lidar_yaw_correction_limited = false;
  double lidar_yaw_correction_limit_deg = 0.0;
  double lidar_yaw_information_scale = 1.0;
  double mean_normalized_residual = std::numeric_limits<double>::infinity();
  double mean_robust_weight = 0.0;
  double measurement_condition = std::numeric_limits<double>::infinity();
  double rotation_measurement_condition =
      std::numeric_limits<double>::infinity();
  double yaw_observability = 0.0;
  int correspondences = 0;
  int innovation_rejections = 0;
  int observable_directions = 0;
  int observable_rotation_directions = 0;
  int scan_points = 0;
  int correspondence_azimuth_sectors = 0;
  int point_knn_fallback_queries = 0;
  int point_knn_fallback_matches = 0;
  int iterations = 0;
  int convergence_confirmations = 0;
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
  std::string map_update_reason;
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

  struct YawCorrectionSample
  {
    double stamp = 0.0;
    double correction = 0.0;
  };

  struct WheelYawBiasSample
  {
    double stamp = 0.0;
    double bias = 0.0;
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
  bool wheelMeasurement(double stamp, WheelSample *measurement) const;
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
  std::deque<WheelYawBiasSample> wheel_yaw_bias_history_;
  bool wheel_yaw_bias_offset_initialized_ = false;
  double wheel_yaw_bias_offset_ = 0.0;
  std::deque<YawCorrectionSample> lidar_yaw_correction_history_;
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
