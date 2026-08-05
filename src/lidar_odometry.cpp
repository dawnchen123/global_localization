#include "hybrid_localization/lidar_odometry.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace hybrid_localization
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kSmall = 1e-12;

struct VoxelAccumulator
{
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  int count = 0;
};

struct NeighborPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double squared_distance = 0.0;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

using NeighborVector =
    std::vector<NeighborPoint, Eigen::aligned_allocator<NeighborPoint>>;

double clampNorm(Eigen::Vector3d &vector, double maximum)
{
  const double norm = vector.norm();
  if (maximum > 0.0 && norm > maximum)
  {
    vector *= maximum / norm;
  }
  return norm;
}

double median(std::vector<double> values)
{
  if (values.empty())
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t middle = values.size() / 2U;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if ((values.size() & 1U) != 0U)
  {
    return upper;
  }
  const double lower = *std::max_element(values.begin(), values.begin() + middle);
  return 0.5 * (lower + upper);
}

struct VisualMeasurementProjection
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  int observable_directions = 0;
  bool z_requested = false;
  bool z_observable = false;
  bool z_fused = false;
  double z_projection = 0.0;
  double z_conditional_information_ratio = 0.0;
  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> projection = Eigen::Matrix<double, 6, 6>::Zero();
};

VisualMeasurementProjection projectVisualMeasurement(
    const VisualPoseLinearization &linearization,
    const LidarOdometryOptions &options)
{
  if (!linearization.hessian.allFinite() || !linearization.gradient.allFinite())
  {
    return VisualMeasurementProjection();
  }

  const Eigen::Matrix<double, 6, 6> symmetric_hessian = 0.5 *
      (linearization.hessian + linearization.hessian.transpose());
  const bool z_requested = options.visual_fuse_translation_z ||
      options.visual_fuse_translation_z_when_observable;
  const auto project_axes = [&](bool include_z)
  {
    VisualMeasurementProjection result;
    result.z_requested = z_requested;
    Eigen::Matrix<double, 6, 6> axis_mask =
        Eigen::Matrix<double, 6, 6>::Zero();
    if (options.visual_fuse_roll_pitch)
    {
      axis_mask(0, 0) = 1.0;
      axis_mask(1, 1) = 1.0;
    }
    if (options.visual_fuse_yaw) axis_mask(2, 2) = 1.0;
    if (options.visual_fuse_translation_xy)
    {
      axis_mask(3, 3) = 1.0;
      axis_mask(4, 4) = 1.0;
    }
    if (include_z) axis_mask(5, 5) = 1.0;

    const Eigen::Matrix<double, 6, 6> masked_hessian =
        axis_mask * symmetric_hessian * axis_mask;
    const Eigen::Matrix<double, 6, 1> masked_gradient =
        axis_mask * linearization.gradient;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
        eigen_solver(masked_hessian);
    if (eigen_solver.info() != Eigen::Success ||
        !eigen_solver.eigenvalues().allFinite() ||
        !eigen_solver.eigenvectors().allFinite())
    {
      return result;
    }
    const double maximum_eigenvalue = eigen_solver.eigenvalues().maxCoeff();
    if (!std::isfinite(maximum_eigenvalue) || maximum_eigenvalue <= kSmall)
    {
      return result;
    }
    const double threshold = std::max(
        kSmall, options.visual_observability_eigen_ratio * maximum_eigenvalue);
    for (int index = 0; index < 6; ++index)
    {
      if (eigen_solver.eigenvalues()(index) < threshold) continue;
      const Eigen::Matrix<double, 6, 1> direction =
          eigen_solver.eigenvectors().col(index);
      result.projection.noalias() += direction * direction.transpose();
      ++result.observable_directions;
    }
    if (result.observable_directions == 0) return result;
    result.projection = 0.5 * (result.projection + result.projection.transpose());
    result.hessian = result.projection * masked_hessian * result.projection;
    result.hessian = 0.5 * (result.hessian + result.hessian.transpose());
    result.gradient = result.projection * masked_gradient;
    result.valid = result.hessian.allFinite() && result.gradient.allFinite();

    if (!include_z || !result.valid) return result;
    result.z_projection = std::max(0.0, std::min(1.0, result.projection(5, 5)));
    std::array<int, 5> conditioning_axes{{0, 0, 0, 0, 0}};
    int conditioning_count = 0;
    for (int index = 0; index < 5; ++index)
    {
      if (axis_mask(index, index) > 0.5)
      {
        conditioning_axes[static_cast<std::size_t>(conditioning_count++)] = index;
      }
    }
    if (conditioning_count == 0)
    {
      result.z_conditional_information_ratio = 1.0;
    }
    else
    {
      Eigen::MatrixXd conditioning_hessian(conditioning_count, conditioning_count);
      Eigen::VectorXd cross(conditioning_count);
      for (int row = 0; row < conditioning_count; ++row)
      {
        cross(row) = masked_hessian(
            conditioning_axes[static_cast<std::size_t>(row)], 5);
        for (int col = 0; col < conditioning_count; ++col)
        {
          conditioning_hessian(row, col) = masked_hessian(
              conditioning_axes[static_cast<std::size_t>(row)],
              conditioning_axes[static_cast<std::size_t>(col)]);
        }
      }
      const double regularizer = std::max(
          kSmall, conditioning_hessian.diagonal().cwiseAbs().maxCoeff() * 1e-9);
      conditioning_hessian.diagonal().array() += regularizer;
      const Eigen::LDLT<Eigen::MatrixXd> conditioning_solver(
          conditioning_hessian);
      if (conditioning_solver.info() == Eigen::Success)
      {
        const Eigen::VectorXd solved_cross = conditioning_solver.solve(cross);
        if (solved_cross.allFinite())
        {
          const double conditional_information = std::max(
              0.0, masked_hessian(5, 5) - cross.dot(solved_cross));
          result.z_conditional_information_ratio = std::max(0.0, std::min(1.0,
              conditional_information / std::max(kSmall, masked_hessian(5, 5))));
        }
      }
    }
    result.z_observable = result.z_projection >= options.visual_z_min_projection &&
        result.z_conditional_information_ratio >=
            options.visual_z_min_conditional_information_ratio;
    result.z_fused = options.visual_fuse_translation_z ||
        (options.visual_fuse_translation_z_when_observable && result.z_observable);
    return result;
  };

  VisualMeasurementProjection result = project_axes(z_requested);
  // In the automatic mode, do not retain a coupled Z component from an
  // otherwise observable yaw/XY eigendirection. Re-project the measurement
  // without Z so the ESKF cannot manufacture an altitude correction.
  if (z_requested && !result.z_fused)
  {
    const double z_projection = result.z_projection;
    const double z_conditional_information_ratio =
        result.z_conditional_information_ratio;
    const bool z_observable = result.z_observable;
    result = project_axes(false);
    result.z_requested = true;
    result.z_projection = z_projection;
    result.z_conditional_information_ratio = z_conditional_information_ratio;
    result.z_observable = z_observable;
  }
  return result;
}

bool passesTwoModeVisualQuality(const VisualPoseLinearization &linearization,
                                int observable_directions,
                                const LidarOdometryOptions &options)
{
  if (observable_directions != 2) return true;
  if (options.visual_two_mode_min_landmarks > 0 &&
      linearization.landmarks < options.visual_two_mode_min_landmarks)
  {
    return false;
  }
  if (options.visual_two_mode_min_residuals > 0 &&
      linearization.residuals < options.visual_two_mode_min_residuals)
  {
    return false;
  }
  if (options.visual_two_mode_max_rmse > 0.0 &&
      (!std::isfinite(linearization.rmse) ||
       linearization.rmse > options.visual_two_mode_max_rmse))
  {
    return false;
  }
  if (options.visual_two_mode_min_mean_ncc > 0.0 &&
      (!std::isfinite(linearization.mean_ncc) ||
       linearization.mean_ncc < options.visual_two_mode_min_mean_ncc))
  {
    return false;
  }
  return true;
}
}  // namespace

std::size_t LidarOdometry::VoxelKeyHash::operator()(const VoxelKey &key) const
{
  std::size_t seed = std::hash<int>()(key.x);
  seed ^= std::hash<int>()(key.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  seed ^= std::hash<int>()(key.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  return seed;
}

Eigen::Matrix3d LidarOdometry::skew(const Eigen::Vector3d &vector)
{
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(),
      vector.z(), 0.0, -vector.x(),
      -vector.y(), vector.x(), 0.0;
  return matrix;
}

Eigen::Matrix3d LidarOdometry::expSO3(const Eigen::Vector3d &rotation_vector)
{
  const double angle = rotation_vector.norm();
  if (angle < 1e-10)
  {
    return Eigen::Matrix3d::Identity() + skew(rotation_vector);
  }
  return Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix();
}

Eigen::Vector3d LidarOdometry::logSO3(const Eigen::Matrix3d &rotation)
{
  const Eigen::AngleAxisd angle_axis(rotation);
  if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() < 1e-10)
  {
    return Eigen::Vector3d::Zero();
  }
  return angle_axis.angle() * angle_axis.axis();
}

double LidarOdometry::rotationDegrees(const Eigen::Matrix3d &rotation)
{
  return logSO3(rotation).norm() * 180.0 / kPi;
}

double LidarOdometry::huberWeight(double residual, double delta)
{
  const double magnitude = std::abs(residual);
  return magnitude <= delta || magnitude < kSmall ? 1.0 : delta / magnitude;
}

LidarOdometry::LidarOdometry(const LidarOdometryOptions &options) : options_(options)
{
  options_.scan_voxel_size = std::max(0.05, options_.scan_voxel_size);
  options_.map_voxel_size = std::max(0.10, options_.map_voxel_size);
  options_.map_insert_voxel_size = std::max(0.05, options_.map_insert_voxel_size);
  options_.max_correspondence_distance = std::max(
      options_.map_voxel_size, options_.max_correspondence_distance);
  options_.max_plane_distance = std::max(0.05, options_.max_plane_distance);
  options_.plane_max_eigen_ratio = std::max(0.01, options_.plane_max_eigen_ratio);
  options_.max_plane_variance = std::max(1e-6, options_.max_plane_variance);
  options_.plane_uncertainty_scale = std::max(0.0, options_.plane_uncertainty_scale);
  options_.plane_support_radius_scale = std::max(
      0.0, options_.plane_support_radius_scale);
  options_.plane_extrapolation_uncertainty_scale = std::max(
      0.0, options_.plane_extrapolation_uncertainty_scale);
  options_.plane_parameter_uncertainty_scale = std::max(
      0.0, options_.plane_parameter_uncertainty_scale);
  options_.plane_fit_residual_gate = std::max(0.02, options_.plane_fit_residual_gate);
  options_.adaptive_subvoxel_scale = std::max(
      0.20, std::min(0.80, options_.adaptive_subvoxel_scale));
  options_.adaptive_subvoxel_search_radius = std::max(
      0, std::min(2, options_.adaptive_subvoxel_search_radius));
  options_.strong_support_min_correspondences = std::max(
      0, options_.strong_support_min_correspondences);
  options_.point_knn_fallback_max_queries = std::max(
      0, options_.point_knn_fallback_max_queries);
  options_.compatible_voxel_search_radius = std::max(
      0, options_.compatible_voxel_search_radius);
  options_.strong_support_min_azimuth_sectors = std::max(
      1, options_.strong_support_min_azimuth_sectors);
  options_.strong_support_max_rmse = std::max(0.0, options_.strong_support_max_rmse);
  options_.observability_eigen_ratio = std::max(
      1e-8, std::min(0.25, options_.observability_eigen_ratio));
  options_.rotation_observability_eigen_ratio = std::max(
      1e-8, std::min(0.25, options_.rotation_observability_eigen_ratio));
  options_.weak_rotation_information_scale = std::max(
      0.0, std::min(1.0, options_.weak_rotation_information_scale));
  options_.min_observable_directions = std::max(
      1, std::min(6, options_.min_observable_directions));
  options_.max_mean_normalized_residual = std::max(
      0.0, options_.max_mean_normalized_residual);
  options_.convergence_confirmation_iterations = std::max(
      1, options_.convergence_confirmation_iterations);
  options_.max_iteration_translation = std::max(
      1e-4, options_.max_iteration_translation);
  options_.max_iteration_rotation_deg = std::max(
      1e-3, options_.max_iteration_rotation_deg);
  options_.max_lidar_velocity_step = std::max(
      0.0, options_.max_lidar_velocity_step);
  options_.max_lidar_gyro_bias_step = std::max(
      0.0, options_.max_lidar_gyro_bias_step);
  options_.max_lidar_acceleration_bias_step = std::max(
      0.0, options_.max_lidar_acceleration_bias_step);
  options_.max_lidar_gravity_step = std::max(
      0.0, options_.max_lidar_gravity_step);
  options_.map_insertion_min_observable_directions = std::max(
      1, std::min(6, options_.map_insertion_min_observable_directions));
  options_.map_insertion_min_observable_rotation_directions = std::max(
      0, std::min(3,
                  options_.map_insertion_min_observable_rotation_directions));
  options_.map_insertion_min_yaw_observability = std::max(
      0.0, std::min(1.0, options_.map_insertion_min_yaw_observability));
  options_.map_insertion_max_mean_normalized_residual = std::max(
      0.0, options_.map_insertion_max_mean_normalized_residual);
  options_.map_insertion_max_lidar_correction_translation = std::max(
      0.0, options_.map_insertion_max_lidar_correction_translation);
  options_.map_insertion_max_lidar_correction_rotation_deg = std::max(
      0.0, options_.map_insertion_max_lidar_correction_rotation_deg);
  options_.recovery_after_rejections = std::max(0, options_.recovery_after_rejections);
  options_.recovery_max_lidar_correction_translation = std::max(
      0.0, options_.recovery_max_lidar_correction_translation);
  options_.recovery_max_lidar_correction_rotation_deg = std::max(
      0.0, options_.recovery_max_lidar_correction_rotation_deg);
  options_.lidar_measurement_noise = std::max(0.005, options_.lidar_measurement_noise);
  options_.lidar_normalized_huber_delta = std::max(
      0.0, options_.lidar_normalized_huber_delta);
  options_.lidar_innovation_gate = std::max(0.0, options_.lidar_innovation_gate);
  options_.visual_observability_eigen_ratio = std::max(
      1e-8, std::min(0.25, options_.visual_observability_eigen_ratio));
  options_.visual_min_observable_directions = std::max(
      1, std::min(6, options_.visual_min_observable_directions));
  options_.visual_two_mode_min_landmarks = std::max(
      0, options_.visual_two_mode_min_landmarks);
  options_.visual_two_mode_min_residuals = std::max(
      0, options_.visual_two_mode_min_residuals);
  options_.visual_two_mode_max_rmse = std::max(
      0.0, options_.visual_two_mode_max_rmse);
  options_.visual_two_mode_min_mean_ncc = std::max(
      0.0, std::min(1.0, options_.visual_two_mode_min_mean_ncc));
  options_.visual_z_min_projection = std::max(
      0.0, std::min(1.0, options_.visual_z_min_projection));
  options_.visual_z_min_conditional_information_ratio = std::max(
      0.0, std::min(1.0, options_.visual_z_min_conditional_information_ratio));
  options_.visual_max_z_step = std::max(0.0, options_.visual_max_z_step);
  options_.visual_max_velocity_step = std::max(
      0.0, options_.visual_max_velocity_step);
  options_.visual_max_gyro_bias_step = std::max(
      0.0, options_.visual_max_gyro_bias_step);
  options_.visual_max_acceleration_bias_step = std::max(
      0.0, options_.visual_max_acceleration_bias_step);
  options_.visual_max_gravity_step = std::max(
      0.0, options_.visual_max_gravity_step);
  options_.huber_delta = std::max(0.01, options_.huber_delta);
  options_.max_translation_per_scan = std::max(0.05, options_.max_translation_per_scan);
  options_.max_rotation_per_scan_deg = std::max(0.1, options_.max_rotation_per_scan_deg);
  options_.max_translation_speed = std::max(0.0, options_.max_translation_speed);
  options_.max_rotation_speed_deg = std::max(0.0, options_.max_rotation_speed_deg);
  options_.turn_aware_rotation_margin_deg = std::max(
      0.0, options_.turn_aware_rotation_margin_deg);
  options_.turn_aware_max_rotation_deg = std::max(
      options_.max_rotation_per_scan_deg,
      options_.turn_aware_max_rotation_deg);
  options_.turn_aware_max_scan_dt = std::max(
      0.0, options_.turn_aware_max_scan_dt);
  options_.turn_aware_min_yaw_rate = std::max(
      0.0, options_.turn_aware_min_yaw_rate);
  options_.turn_aware_lidar_correction_rotation_deg = std::max(
      0.0, options_.turn_aware_lidar_correction_rotation_deg);
  options_.turn_aware_wheel_imu_max_yaw_rate_difference = std::max(
      0.0, options_.turn_aware_wheel_imu_max_yaw_rate_difference);
  options_.lidar_rotation_correction_nis_gate = std::max(
      0.0, options_.lidar_rotation_correction_nis_gate);
  options_.lidar_rotation_correction_std_floor_deg = std::max(
      1e-3, options_.lidar_rotation_correction_std_floor_deg);
  options_.lidar_yaw_correction_window_sec = std::max(
      0.0, options_.lidar_yaw_correction_window_sec);
  options_.max_cumulative_lidar_yaw_correction_deg = std::max(
      0.0, options_.max_cumulative_lidar_yaw_correction_deg);
  options_.limited_lidar_yaw_information_scale = std::max(
      0.0, std::min(1.0,
                    options_.limited_lidar_yaw_information_scale));
  options_.lidar_loss_hold_after_rejections = std::max(
      0, options_.lidar_loss_hold_after_rejections);
  options_.lidar_loss_freeze_after_rejections = std::max(
      0, options_.lidar_loss_freeze_after_rejections);
  options_.lidar_loss_max_vertical_offset = std::max(
      0.01, options_.lidar_loss_max_vertical_offset);
  options_.lidar_loss_max_horizontal_speed = std::max(
      0.0, options_.lidar_loss_max_horizontal_speed);
  options_.lidar_loss_max_horizontal_step = std::max(
      0.0, options_.lidar_loss_max_horizontal_step);
  options_.lidar_loss_velocity_decay = std::max(
      0.0, std::min(1.0, options_.lidar_loss_velocity_decay));
  options_.registration_threads = std::max(
      1, std::min(2, options_.registration_threads));
  options_.max_iterations = std::max(1, options_.max_iterations);
  options_.min_scan_points = std::max(20, options_.min_scan_points);
  options_.min_correspondences = std::max(20, options_.min_correspondences);
  options_.max_scan_points = std::max(options_.min_scan_points, options_.max_scan_points);
  options_.max_map_points = std::max(options_.min_correspondences, options_.max_map_points);
  if (options_.retain_global_map)
  {
    // A global profile knows its memory budget up front. Reserving the hash
    // table avoids multi-million-entry rehash pauses that can otherwise make
    // the scheduler discard a different LiDAR frame on each replay.
    map_voxels_.reserve(static_cast<std::size_t>(options_.max_map_points));
  }
  if (options_.max_adaptive_subvoxels <= 0)
  {
    options_.max_adaptive_subvoxels = 2 * options_.max_map_points;
  }
  options_.max_adaptive_subvoxels = std::max(
      options_.min_correspondences, options_.max_adaptive_subvoxels);
  if (options_.retain_global_map && options_.use_adaptive_subvoxel_plane)
  {
    adaptive_subvoxels_.reserve(
        static_cast<std::size_t>(options_.max_adaptive_subvoxels));
  }
  options_.normal_neighbor_voxels = std::max(1, options_.normal_neighbor_voxels);
  options_.min_normal_neighbors = std::max(4, options_.min_normal_neighbors);
  options_.max_plane_neighbors = std::max(options_.min_normal_neighbors,
                                           options_.max_plane_neighbors);
  options_.min_voxel_plane_points = std::max(4, options_.min_voxel_plane_points);
  options_.max_voxel_points = std::max(options_.min_voxel_plane_points,
                                        options_.max_voxel_points);
  options_.max_voxel_samples = std::max(4, options_.max_voxel_samples);
  options_.mature_voxel_update_gain = std::max(
      0.0, std::min(1.0, options_.mature_voxel_update_gain));
  options_.map_insertion_min_translation = std::max(
      0.0, options_.map_insertion_min_translation);
  options_.map_insertion_min_rotation_deg = std::max(
      0.0, options_.map_insertion_min_rotation_deg);
  options_.map_insertion_max_interval = std::max(
      0.0, options_.map_insertion_max_interval);
  options_.imu_init_samples = std::max(20, options_.imu_init_samples);
  options_.imu_init_duration = std::max(0.1, options_.imu_init_duration);
  options_.imu_init_gyro_bias_covariance_floor = std::max(
      1e-12, options_.imu_init_gyro_bias_covariance_floor);
  options_.imu_init_acceleration_bias_covariance = std::max(
      1e-12, options_.imu_init_acceleration_bias_covariance);
  options_.imu_init_gravity_covariance_floor = std::max(
      1e-12, options_.imu_init_gravity_covariance_floor);
  options_.imu_init_nonstationary_gravity_covariance = std::max(
      options_.imu_init_gravity_covariance_floor,
      options_.imu_init_nonstationary_gravity_covariance);
  options_.imu_max_gap = std::max(0.005, options_.imu_max_gap);
  options_.imu_buffer_duration = std::max(options_.imu_init_duration + 1.0,
                                           options_.imu_buffer_duration);
  options_.gravity_magnitude = std::max(1.0, options_.gravity_magnitude);
  options_.wheel_max_age = std::max(0.01, options_.wheel_max_age);
  options_.wheel_max_speed = std::max(0.1, options_.wheel_max_speed);
  options_.wheel_forward_noise = std::max(0.01, options_.wheel_forward_noise);
  options_.wheel_lateral_noise = std::max(0.01, options_.wheel_lateral_noise);
  options_.wheel_vertical_noise = std::max(0.01, options_.wheel_vertical_noise);
  options_.wheel_huber_delta = std::max(0.1, options_.wheel_huber_delta);
  options_.wheel_forward_innovation_gate = std::max(
      0.0, options_.wheel_forward_innovation_gate);
  options_.wheel_yaw_rate_relative_scale_uncertainty = std::max(
      0.0, std::min(1.0,
                    options_.wheel_yaw_rate_relative_scale_uncertainty));
  options_.wheel_yaw_bias_window_sec = std::max(
      0.0, options_.wheel_yaw_bias_window_sec);
  options_.wheel_yaw_bias_min_samples = std::max(
      3, options_.wheel_yaw_bias_min_samples);
  options_.wheel_yaw_bias_max_abs_rate = std::max(
      0.0, options_.wheel_yaw_bias_max_abs_rate);
  options_.wheel_yaw_bias_max_mad = std::max(
      0.0, options_.wheel_yaw_bias_max_mad);
  options_.wheel_yaw_bias_noise_floor = std::max(
      1e-4, options_.wheel_yaw_bias_noise_floor);
  options_.wheel_yaw_rate_noise = std::max(
      0.01, options_.wheel_yaw_rate_noise);
  options_.wheel_yaw_rate_huber_delta = std::max(
      0.1, options_.wheel_yaw_rate_huber_delta);
  options_.wheel_yaw_rate_innovation_gate = std::max(
      0.0, options_.wheel_yaw_rate_innovation_gate);
  options_.wheel_yaw_rate_min_speed = std::max(
      0.0, options_.wheel_yaw_rate_min_speed);
  options_.wheel_yaw_rate_max_abs = std::max(
      0.01, options_.wheel_yaw_rate_max_abs);
  options_.wheel_yaw_rate_max_imu_difference = std::max(
      0.0, options_.wheel_yaw_rate_max_imu_difference);
  options_.wheel_differential_max_disagreement = std::max(
      0.0, options_.wheel_differential_max_disagreement);
  options_.wheel_buffer_duration = std::max(options_.wheel_max_age + 0.1,
                                             options_.wheel_buffer_duration);
  options_.visual_max_iterations = std::max(1, options_.visual_max_iterations);
  options_.visual_min_landmarks = std::max(6, options_.visual_min_landmarks);
  options_.visual_min_residuals = std::max(24, options_.visual_min_residuals);
  options_.visual_max_rmse = std::max(0.05, options_.visual_max_rmse);
  options_.visual_max_translation_step = std::max(
      0.01, options_.visual_max_translation_step);
  options_.visual_max_rotation_step_deg = std::max(
      0.1, options_.visual_max_rotation_step_deg);
  options_.visual_convergence_translation = std::max(
      1e-6, options_.visual_convergence_translation);
  options_.visual_convergence_rotation_deg = std::max(
      1e-5, options_.visual_convergence_rotation_deg);
  options_.visual_solver_damping = std::max(1e-12, options_.visual_solver_damping);
  reset();
}

void LidarOdometry::reset()
{
  imu_initialized_ = !options_.imu_enabled;
  map_initialized_ = false;
  state_stamp_ = 0.0;
  acceleration_scale_ = options_.acceleration_scale;
  imu_init_progress_ = options_.imu_enabled ? 0.0 : 1.0;
  consecutive_rejections_ = 0;
  accepted_scan_count_ = 0;
  lidar_loss_limited_ = false;
  lidar_loss_frozen_ = false;
  recovery_map_guard_active_ = false;
  recovery_map_trusted_scan_count_ = 0;
  state_ = State();
  state_.gravity = Eigen::Vector3d(0.0, 0.0, -options_.gravity_magnitude);
  state_.covariance.setZero();
  state_.covariance.block<3, 3>(0, 0).diagonal().setConstant(0.02);
  state_.covariance.block<3, 3>(3, 3).diagonal().setConstant(0.01);
  state_.covariance.block<3, 3>(6, 6).diagonal().setConstant(0.10);
  state_.covariance.block<3, 3>(9, 9).diagonal().setConstant(1e-4);
  state_.covariance.block<3, 3>(12, 12).diagonal().setConstant(1e-2);
  state_.covariance.block<3, 3>(15, 15).diagonal().setConstant(1e-3);
  last_accepted_state_ = state_;
  pose_cache_.setIdentity();
  last_scan_pose_.setIdentity();
  previous_scan_pose_.setIdentity();
  previous_scan_stamp_ = 0.0;
  have_last_map_insert_pose_ = false;
  last_map_insert_pose_.setIdentity();
  last_map_insert_stamp_ = 0.0;
  imu_buffer_.clear();
  propagation_history_.clear();
  wheel_buffer_.clear();
  wheel_yaw_bias_history_.clear();
  wheel_yaw_bias_offset_initialized_ = false;
  wheel_yaw_bias_offset_ = 0.0;
  lidar_yaw_correction_history_.clear();
  map_voxels_.clear();
  adaptive_subvoxels_.clear();
}

void LidarOdometry::addImuSample(const ImuSample &sample)
{
  if (!options_.imu_enabled || !std::isfinite(sample.stamp) ||
      !sample.acceleration.allFinite() || !sample.angular_velocity.allFinite())
  {
    return;
  }
  if (!imu_buffer_.empty() && sample.stamp <= imu_buffer_.back().stamp)
  {
    return;
  }
  imu_buffer_.push_back(sample);
  const double oldest = sample.stamp - options_.imu_buffer_duration;
  while (imu_buffer_.size() > 2U && imu_buffer_[1].stamp < oldest)
  {
    imu_buffer_.pop_front();
  }
  if (!imu_initialized_)
  {
    initializeImuIfReady();
  }
}

void LidarOdometry::addWheelSample(const WheelSample &sample)
{
  if (!options_.wheel_enabled || !std::isfinite(sample.stamp) ||
      !std::isfinite(sample.forward_speed))
  {
    return;
  }
  WheelSample scaled = sample;
  scaled.forward_speed *= options_.wheel_speed_scale;
  if (std::isfinite(scaled.differential_speed))
  {
    scaled.differential_speed *= options_.wheel_speed_scale;
  }
  if (std::isfinite(scaled.differential_disagreement))
  {
    scaled.differential_disagreement *=
        std::abs(options_.wheel_speed_scale);
  }
  if (std::abs(scaled.forward_speed) > options_.wheel_max_speed ||
      (!wheel_buffer_.empty() && scaled.stamp <= wheel_buffer_.back().stamp))
  {
    return;
  }
  wheel_buffer_.push_back(scaled);
  const double oldest = scaled.stamp - options_.wheel_buffer_duration;
  while (wheel_buffer_.size() > 2U && wheel_buffer_[1].stamp < oldest)
  {
    wheel_buffer_.pop_front();
  }
}

bool LidarOdometry::wheelMeasurement(
    double stamp, WheelSample *measurement) const
{
  if (!options_.wheel_enabled || wheel_buffer_.empty() ||
      measurement == nullptr)
  {
    return false;
  }
  const WheelSample *best = nullptr;
  double best_age = std::numeric_limits<double>::infinity();
  for (auto iterator = wheel_buffer_.rbegin(); iterator != wheel_buffer_.rend(); ++iterator)
  {
    const double age = std::abs(iterator->stamp - stamp);
    if (age < best_age)
    {
      best_age = age;
      best = &*iterator;
    }
    if (iterator->stamp < stamp - options_.wheel_max_age)
    {
      break;
    }
  }
  if (best == nullptr || best_age > options_.wheel_max_age)
  {
    return false;
  }
  *measurement = *best;
  return true;
}

bool LidarOdometry::angularVelocityMeasurement(
    double stamp, Eigen::Vector3d *angular_velocity) const
{
  if (angular_velocity == nullptr || !std::isfinite(stamp) ||
      imu_buffer_.empty())
  {
    return false;
  }
  const auto upper = std::lower_bound(
      imu_buffer_.begin(), imu_buffer_.end(), stamp,
      [](const ImuSample &sample, double query_stamp)
      { return sample.stamp < query_stamp; });
  if (upper == imu_buffer_.begin())
  {
    if (std::abs(upper->stamp - stamp) > options_.imu_max_gap) return false;
    *angular_velocity = upper->angular_velocity;
    return angular_velocity->allFinite();
  }
  if (upper == imu_buffer_.end())
  {
    const ImuSample &last = imu_buffer_.back();
    if (std::abs(last.stamp - stamp) > options_.imu_max_gap) return false;
    *angular_velocity = last.angular_velocity;
    return angular_velocity->allFinite();
  }
  const ImuSample &second = *upper;
  const ImuSample &first = *(upper - 1);
  const double interval = second.stamp - first.stamp;
  if (interval <= 0.0 || interval > options_.imu_max_gap ||
      stamp < first.stamp - options_.imu_max_gap ||
      stamp > second.stamp + options_.imu_max_gap)
  {
    return false;
  }
  const double ratio = std::max(0.0, std::min(1.0,
      (stamp - first.stamp) / interval));
  *angular_velocity = (1.0 - ratio) * first.angular_velocity +
      ratio * second.angular_velocity;
  return angular_velocity->allFinite();
}

bool LidarOdometry::initializeImuIfReady()
{
  if (!options_.imu_enabled)
  {
    imu_initialized_ = true;
    imu_init_progress_ = 1.0;
    return true;
  }
  if (imu_buffer_.size() < 2U)
  {
    return false;
  }

  const double end_stamp = imu_buffer_.back().stamp;
  std::size_t start = imu_buffer_.size() - 1U;
  while (start > 0U && (end_stamp - imu_buffer_[start].stamp < options_.imu_init_duration ||
                        imu_buffer_.size() - start < static_cast<std::size_t>(options_.imu_init_samples)))
  {
    --start;
  }
  const std::size_t count = imu_buffer_.size() - start;
  const double duration = end_stamp - imu_buffer_[start].stamp;
  imu_init_progress_ = std::min(1.0, std::min(
      static_cast<double>(count) / static_cast<double>(options_.imu_init_samples),
      duration / options_.imu_init_duration));
  if (count < static_cast<std::size_t>(options_.imu_init_samples) ||
      duration < options_.imu_init_duration)
  {
    return false;
  }

  Eigen::Vector3d mean_acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_angular_velocity = Eigen::Vector3d::Zero();
  for (std::size_t index = start; index < imu_buffer_.size(); ++index)
  {
    mean_acceleration += imu_buffer_[index].acceleration;
    mean_angular_velocity += imu_buffer_[index].angular_velocity;
  }
  mean_acceleration /= static_cast<double>(count);
  mean_angular_velocity /= static_cast<double>(count);

  Eigen::Vector3d acceleration_variance = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity_variance = Eigen::Vector3d::Zero();
  for (std::size_t index = start; index < imu_buffer_.size(); ++index)
  {
    const Eigen::Vector3d acc_delta = imu_buffer_[index].acceleration - mean_acceleration;
    const Eigen::Vector3d gyro_delta = imu_buffer_[index].angular_velocity - mean_angular_velocity;
    acceleration_variance += acc_delta.cwiseProduct(acc_delta);
    angular_velocity_variance += gyro_delta.cwiseProduct(gyro_delta);
  }
  const double denominator = static_cast<double>(std::max<std::size_t>(1U, count - 1U));
  acceleration_variance /= denominator;
  angular_velocity_variance /= denominator;
  const double acceleration_std = std::sqrt(acceleration_variance.maxCoeff());
  const double angular_velocity_std = std::sqrt(angular_velocity_variance.maxCoeff());
  const bool stationary = acceleration_std <= options_.imu_init_max_acc_std &&
      angular_velocity_std <= options_.imu_init_max_gyro_std &&
      mean_angular_velocity.norm() <= options_.imu_init_max_gyro_bias;
  if (options_.imu_init_require_stationary && !stationary)
  {
    imu_init_progress_ = 0.99;
    return false;
  }

  const double acceleration_norm = mean_acceleration.norm();
  if (!std::isfinite(acceleration_norm) || acceleration_norm < 0.1)
  {
    imu_init_progress_ = 0.0;
    return false;
  }
  acceleration_scale_ = options_.acceleration_scale;
  if (options_.auto_acceleration_scale)
  {
    acceleration_scale_ *= options_.gravity_magnitude / acceleration_norm;
  }

  state_.rotation.setIdentity();
  state_.position.setZero();
  state_.velocity.setZero();
  state_.gyro_bias = stationary ? mean_angular_velocity : Eigen::Vector3d::Zero();
  state_.acceleration_bias.setZero();
  state_.gravity = -mean_acceleration.normalized() * options_.gravity_magnitude;
  state_.covariance.setZero();
  state_.covariance.block<3, 3>(0, 0).diagonal().setConstant(0.01);
  state_.covariance(2, 2) = 0.10;
  state_.covariance.block<3, 3>(3, 3).diagonal().setConstant(0.01);
  state_.covariance.block<3, 3>(6, 6).diagonal().setConstant(0.05);
  if (stationary && options_.imu_init_use_mean_covariance)
  {
    const double inverse_count = 1.0 / static_cast<double>(count);
    const Eigen::Vector3d raw_acceleration_mean_variance =
        acceleration_variance * inverse_count;
    const Eigen::Vector3d acceleration_mean_variance =
        acceleration_scale_ * acceleration_scale_ *
        raw_acceleration_mean_variance;
    const Eigen::Vector3d angular_velocity_mean_variance =
        angular_velocity_variance * inverse_count;
    state_.covariance.block<3, 3>(9, 9).diagonal() =
        angular_velocity_mean_variance.cwiseMax(Eigen::Vector3d::Constant(
            options_.imu_init_gyro_bias_covariance_floor));
    state_.covariance.block<3, 3>(12, 12).diagonal() =
        acceleration_mean_variance.cwiseMax(Eigen::Vector3d::Constant(
            options_.imu_init_acceleration_bias_covariance));

    // g = -g0 * a / ||a||. Propagate the covariance of the stationary
    // acceleration mean through the normalization Jacobian. Gravity has no
    // radial degree of freedom, but a tiny radial variance keeps the
    // information-form update nonsingular before the first covariance reset.
    const Eigen::Vector3d acceleration_direction =
        mean_acceleration / acceleration_norm;
    const Eigen::Matrix3d tangent_projection =
        Eigen::Matrix3d::Identity() -
        acceleration_direction * acceleration_direction.transpose();
    const Eigen::Matrix3d gravity_jacobian =
        -options_.gravity_magnitude / acceleration_norm *
        tangent_projection;
    const Eigen::Matrix3d acceleration_mean_covariance =
        raw_acceleration_mean_variance.asDiagonal();
    Eigen::Matrix3d gravity_covariance =
        gravity_jacobian * acceleration_mean_covariance *
        gravity_jacobian.transpose();
    gravity_covariance.noalias() +=
        options_.imu_init_gravity_covariance_floor * tangent_projection;
    gravity_covariance.noalias() +=
        1e-10 * acceleration_direction * acceleration_direction.transpose();
    state_.covariance.block<3, 3>(15, 15) =
        0.5 * (gravity_covariance + gravity_covariance.transpose());
  }
  else
  {
    state_.covariance.block<3, 3>(9, 9).diagonal() =
        angular_velocity_variance.cwiseMax(Eigen::Vector3d::Constant(
            options_.imu_init_gyro_bias_covariance_floor));
    state_.covariance.block<3, 3>(12, 12).diagonal() =
        (acceleration_scale_ * acceleration_scale_ * acceleration_variance)
            .cwiseMax(Eigen::Vector3d::Constant(
                options_.imu_init_acceleration_bias_covariance));
    state_.covariance.block<3, 3>(15, 15).diagonal().setConstant(
        options_.imu_init_nonstationary_gravity_covariance);
  }
  state_stamp_ = end_stamp;
  pose_cache_ = statePose(state_);
  last_accepted_state_ = state_;
  imu_initialized_ = true;
  imu_init_progress_ = 1.0;
  recordCurrentPose();
  return true;
}

LidarOdometry::VoxelKey LidarOdometry::voxelKey(const Eigen::Vector3d &point,
                                                double voxel_size) const
{
  return VoxelKey{static_cast<int>(std::floor(point.x() / voxel_size)),
                  static_cast<int>(std::floor(point.y() / voxel_size)),
                  static_cast<int>(std::floor(point.z() / voxel_size))};
}

PointVector LidarOdometry::voxelDownsample(const PointVector &points, double voxel_size,
                                           int max_points) const
{
  std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxels;
  voxels.reserve(points.size());
  for (const Eigen::Vector3d &point : points)
  {
    if (!point.allFinite())
    {
      continue;
    }
    VoxelAccumulator &accumulator = voxels[voxelKey(point, voxel_size)];
    accumulator.sum += point;
    ++accumulator.count;
  }

  PointVector result;
  result.reserve(voxels.size());
  for (const auto &entry : voxels)
  {
    if (entry.second.count > 0)
    {
      result.push_back(entry.second.sum / static_cast<double>(entry.second.count));
    }
  }
  if (max_points > 0 && result.size() > static_cast<std::size_t>(max_points))
  {
    PointVector limited;
    limited.reserve(static_cast<std::size_t>(max_points));
    const double stride = static_cast<double>(result.size()) / static_cast<double>(max_points);
    for (int index = 0; index < max_points; ++index)
    {
      limited.push_back(result[std::min(result.size() - 1U,
          static_cast<std::size_t>(std::floor(index * stride)))]);
    }
    return limited;
  }
  return result;
}

Eigen::Isometry3d LidarOdometry::statePose(const State &state) const
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = state.rotation;
  pose.translation() = state.position;
  return pose;
}

Vector18d LidarOdometry::stateDifference(const State &state,
                                         const State &reference) const
{
  Vector18d difference = Vector18d::Zero();
  difference.segment<3>(0) = logSO3(reference.rotation.transpose() * state.rotation);
  difference.segment<3>(3) = state.position - reference.position;
  difference.segment<3>(6) = state.velocity - reference.velocity;
  difference.segment<3>(9) = state.gyro_bias - reference.gyro_bias;
  difference.segment<3>(12) = state.acceleration_bias - reference.acceleration_bias;
  difference.segment<3>(15) = state.gravity - reference.gravity;
  return difference;
}

void LidarOdometry::applyError(State &state, const Vector18d &error) const
{
  state.rotation = state.rotation * expSO3(error.segment<3>(0));
  state.position += error.segment<3>(3);
  state.velocity += error.segment<3>(6);
  state.gyro_bias += error.segment<3>(9);
  state.acceleration_bias += error.segment<3>(12);
  state.gravity += error.segment<3>(15);
  clampNorm(state.gyro_bias, options_.max_gyro_bias);
  clampNorm(state.acceleration_bias, options_.max_acceleration_bias);
  const double gravity_norm = state.gravity.norm();
  if (std::isfinite(gravity_norm) && gravity_norm > 1.0)
  {
    state.gravity *= options_.gravity_magnitude / gravity_norm;
  }
}

void LidarOdometry::resetCovarianceAfterInjection(
    State &state, const Vector18d &injected_error) const
{
  if (!options_.covariance_reset_enabled)
  {
    return;
  }

  Matrix18d reset_jacobian = Matrix18d::Identity();
  // The attitude error is right-multiplicative:
  // R_true = R_nominal Exp(delta_theta).
  reset_jacobian.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() -
      0.5 * skew(injected_error.segment<3>(0));

  // applyError() constrains gravity to the known sphere. Reflect that
  // normalization in the covariance instead of retaining a fictitious radial
  // gravity degree of freedom.
  const double gravity_norm = state.gravity.norm();
  Eigen::Vector3d gravity_direction = Eigen::Vector3d::UnitZ();
  if (std::isfinite(gravity_norm) && gravity_norm > kSmall)
  {
    gravity_direction = state.gravity / gravity_norm;
    reset_jacobian.block<3, 3>(15, 15) =
        Eigen::Matrix3d::Identity() -
        gravity_direction * gravity_direction.transpose();
  }

  state.covariance = reset_jacobian * state.covariance *
      reset_jacobian.transpose();
  // Keep a tiny radial gravity variance so the information-form prior remains
  // numerically invertible on the next asynchronous update.
  state.covariance.block<3, 3>(15, 15).noalias() +=
      1e-10 * gravity_direction * gravity_direction.transpose();
  state.covariance = 0.5 *
      (state.covariance + state.covariance.transpose());
  state.covariance.diagonal() =
      state.covariance.diagonal().cwiseMax(1e-12);
}

void LidarOdometry::propagateSegment(State &state,
                                     const Eigen::Vector3d &angular_velocity,
                                     const Eigen::Vector3d &acceleration,
                                     double dt) const
{
  if (dt <= 0.0)
  {
    return;
  }
  const Eigen::Vector3d unbiased_gyro = angular_velocity - state.gyro_bias;
  const Eigen::Vector3d unbiased_acceleration =
      acceleration_scale_ * acceleration - state.acceleration_bias;
  const Eigen::Matrix3d rotation_before = state.rotation;
  const Eigen::Matrix3d rotation_mid = rotation_before * expSO3(0.5 * dt * unbiased_gyro);
  const Eigen::Vector3d world_acceleration =
      rotation_mid * unbiased_acceleration + state.gravity;

  state.position += state.velocity * dt + 0.5 * world_acceleration * dt * dt;
  state.velocity += world_acceleration * dt;
  state.rotation = rotation_before * expSO3(dt * unbiased_gyro);

  Matrix18d continuous = Matrix18d::Zero();
  continuous.block<3, 3>(0, 0) = -skew(unbiased_gyro);
  continuous.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
  continuous.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
  continuous.block<3, 3>(6, 0) = -rotation_mid * skew(unbiased_acceleration);
  continuous.block<3, 3>(6, 12) = -rotation_mid;
  continuous.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity();
  const Matrix18d first_order = continuous * dt;
  const Matrix18d transition = Matrix18d::Identity() + first_order +
      0.5 * first_order * first_order;

  Matrix18d process_noise = Matrix18d::Zero();
  const double gyro_variance = options_.gyro_noise * options_.gyro_noise;
  const double acceleration_variance = options_.acceleration_noise *
      options_.acceleration_noise;
  process_noise.block<3, 3>(0, 0).diagonal().setConstant(gyro_variance * dt);
  process_noise.block<3, 3>(6, 6).diagonal().setConstant(acceleration_variance * dt);
  process_noise.block<3, 3>(3, 3).diagonal().setConstant(
      0.25 * acceleration_variance * dt * dt * dt);
  process_noise.block<3, 3>(3, 6).diagonal().setConstant(
      0.5 * acceleration_variance * dt * dt);
  process_noise.block<3, 3>(6, 3) = process_noise.block<3, 3>(3, 6);
  process_noise.block<3, 3>(9, 9).diagonal().setConstant(
      options_.gyro_bias_random_walk * options_.gyro_bias_random_walk * dt);
  process_noise.block<3, 3>(12, 12).diagonal().setConstant(
      options_.acceleration_bias_random_walk * options_.acceleration_bias_random_walk * dt);
  process_noise.block<3, 3>(15, 15).diagonal().setConstant(
      options_.gravity_random_walk * options_.gravity_random_walk * dt);

  state.covariance = transition * state.covariance * transition.transpose() + process_noise;
  state.covariance = 0.5 * (state.covariance + state.covariance.transpose());
  state.covariance.diagonal() = state.covariance.diagonal().cwiseMax(1e-12);
}

bool LidarOdometry::propagateTo(
    double end_stamp,
    std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> &trajectory)
{
  trajectory.clear();
  if (!imu_initialized_ || imu_buffer_.size() < 2U ||
      end_stamp + 1e-8 < state_stamp_)
  {
    return false;
  }

  ImuPose initial;
  initial.stamp = state_stamp_;
  initial.rotation = state_.rotation;
  initial.position = state_.position;
  initial.velocity = state_.velocity;
  trajectory.push_back(initial);
  if (end_stamp <= state_stamp_ + 1e-8)
  {
    return true;
  }
  if (imu_buffer_.front().stamp > state_stamp_ + 1e-6 ||
      imu_buffer_.back().stamp < end_stamp - 1e-6)
  {
    return false;
  }

  std::size_t index = 0U;
  while (index + 1U < imu_buffer_.size() &&
         imu_buffer_[index + 1U].stamp <= state_stamp_)
  {
    ++index;
  }
  if (index + 1U >= imu_buffer_.size())
  {
    return false;
  }

  State propagated = state_;
  double current_stamp = state_stamp_;
  while (current_stamp < end_stamp - 1e-9)
  {
    if (index + 1U >= imu_buffer_.size())
    {
      return false;
    }
    const ImuSample &first = imu_buffer_[index];
    const ImuSample &second = imu_buffer_[index + 1U];
    const double sample_dt = second.stamp - first.stamp;
    if (sample_dt <= 0.0 || sample_dt > options_.imu_max_gap)
    {
      return false;
    }
    if (current_stamp < first.stamp - 1e-6)
    {
      return false;
    }
    const double segment_end = std::min(end_stamp, second.stamp);
    const double alpha_begin = std::max(0.0, std::min(1.0,
        (current_stamp - first.stamp) / sample_dt));
    const double alpha_end = std::max(0.0, std::min(1.0,
        (segment_end - first.stamp) / sample_dt));
    const Eigen::Vector3d gyro_begin =
        (1.0 - alpha_begin) * first.angular_velocity + alpha_begin * second.angular_velocity;
    const Eigen::Vector3d gyro_end =
        (1.0 - alpha_end) * first.angular_velocity + alpha_end * second.angular_velocity;
    const Eigen::Vector3d acceleration_begin =
        (1.0 - alpha_begin) * first.acceleration + alpha_begin * second.acceleration;
    const Eigen::Vector3d acceleration_end =
        (1.0 - alpha_end) * first.acceleration + alpha_end * second.acceleration;
    propagateSegment(propagated, 0.5 * (gyro_begin + gyro_end),
                     0.5 * (acceleration_begin + acceleration_end),
                     segment_end - current_stamp);
    current_stamp = segment_end;

    ImuPose pose;
    pose.stamp = current_stamp;
    pose.rotation = propagated.rotation;
    pose.position = propagated.position;
    pose.velocity = propagated.velocity;
    trajectory.push_back(pose);
    if (current_stamp >= second.stamp - 1e-9)
    {
      ++index;
    }
  }

  state_ = propagated;
  state_stamp_ = end_stamp;
  pose_cache_ = statePose(state_);
  recordPropagationTrajectory(trajectory);
  return true;
}

void LidarOdometry::predictWithoutImu(
    double end_stamp,
    std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> &trajectory)
{
  trajectory.clear();
  ImuPose begin;
  begin.stamp = state_stamp_;
  begin.rotation = state_.rotation;
  begin.position = state_.position;
  begin.velocity = state_.velocity;
  trajectory.push_back(begin);
  const double dt = state_stamp_ > 0.0 ? std::max(0.0, end_stamp - state_stamp_) : 0.0;
  state_.position += state_.velocity * dt;
  Matrix18d transition = Matrix18d::Identity();
  transition.block<3, 3>(3, 6) =
      Eigen::Matrix3d::Identity() * dt;
  state_.covariance = transition * state_.covariance *
      transition.transpose();
  if (previous_scan_stamp_ > 0.0 && state_stamp_ > previous_scan_stamp_)
  {
    const double previous_dt = state_stamp_ - previous_scan_stamp_;
    const Eigen::Matrix3d relative_rotation =
        previous_scan_pose_.rotation().transpose() * last_scan_pose_.rotation();
    state_.rotation *= expSO3(logSO3(relative_rotation) * dt / previous_dt);
  }
  state_.covariance.block<3, 3>(0, 0).diagonal().array() += 1e-3 * dt;
  state_.covariance.block<3, 3>(3, 3).diagonal().array() += 1e-2 * dt;
  state_.covariance = 0.5 *
      (state_.covariance + state_.covariance.transpose());
  state_stamp_ = end_stamp;
  pose_cache_ = statePose(state_);
  ImuPose end = begin;
  end.stamp = end_stamp;
  end.rotation = state_.rotation;
  end.position = state_.position;
  end.velocity = state_.velocity;
  trajectory.push_back(end);
  recordPropagationTrajectory(trajectory);
}

LidarOdometry::ImuPose LidarOdometry::interpolatePose(
    double stamp,
    const std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> &trajectory) const
{
  if (trajectory.empty())
  {
    ImuPose pose;
    pose.stamp = stamp;
    pose.rotation = state_.rotation;
    pose.position = state_.position;
    pose.velocity = state_.velocity;
    return pose;
  }
  if (stamp <= trajectory.front().stamp)
  {
    return trajectory.front();
  }
  if (stamp >= trajectory.back().stamp)
  {
    return trajectory.back();
  }
  const auto upper = std::upper_bound(trajectory.begin(), trajectory.end(), stamp,
      [](double value, const ImuPose &pose) { return value < pose.stamp; });
  const ImuPose &second = *upper;
  const ImuPose &first = *(upper - 1);
  const double interval = std::max(1e-9, second.stamp - first.stamp);
  const double alpha = std::max(0.0, std::min(1.0, (stamp - first.stamp) / interval));
  ImuPose result;
  result.stamp = stamp;
  Eigen::Quaterniond first_q(first.rotation);
  Eigen::Quaterniond second_q(second.rotation);
  result.rotation = first_q.slerp(alpha, second_q).normalized().toRotationMatrix();
  result.position = (1.0 - alpha) * first.position + alpha * second.position;
  result.velocity = (1.0 - alpha) * first.velocity + alpha * second.velocity;
  return result;
}

PointVector LidarOdometry::deskewScan(
    const TimedPointVector &points, double scan_end_stamp,
    const std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> &trajectory) const
{
  PointVector deskewed;
  deskewed.reserve(points.size());
  if (trajectory.empty())
  {
    for (const TimedPoint &point : points)
    {
      deskewed.push_back(point.point);
    }
    return deskewed;
  }
  const ImuPose end_pose = interpolatePose(scan_end_stamp, trajectory);
  for (const TimedPoint &timed_point : points)
  {
    if (!timed_point.point.allFinite())
    {
      continue;
    }
    const double point_stamp = std::max(trajectory.front().stamp,
        std::min(scan_end_stamp, scan_end_stamp + timed_point.time_from_scan_end));
    const ImuPose point_pose = interpolatePose(point_stamp, trajectory);
    const Eigen::Vector3d world_point =
        point_pose.rotation * timed_point.point + point_pose.position;
    deskewed.push_back(end_pose.rotation.transpose() * (world_point - end_pose.position));
  }
  return deskewed;
}

bool LidarOdometry::planeSupportUncertainty(
    const Eigen::Vector3d &world_point,
    const Eigen::Vector3d &center,
    const Eigen::Vector3d &normal,
    const Eigen::Vector3d &eigenvalues,
    const Eigen::Matrix3d &eigenvectors,
    double effective_support,
    double *additional_variance) const
{
  if (additional_variance == nullptr || !world_point.allFinite() ||
      !center.allFinite() || !normal.allFinite() || !eigenvalues.allFinite() ||
      !eigenvectors.allFinite())
  {
    return false;
  }
  *additional_variance = 0.0;
  const Eigen::Vector3d delta = world_point - center;
  const double normal_distance = normal.dot(delta);
  const double tangent_squared = std::max(
      0.0, delta.squaredNorm() - normal_distance * normal_distance);
  const double tangent_radius = std::sqrt(std::max(
      1e-8, eigenvalues.maxCoeff()));
  if (options_.plane_support_radius_scale > 0.0 &&
      tangent_squared > std::pow(
          options_.plane_support_radius_scale * tangent_radius, 2.0))
  {
    return false;
  }
  if (options_.plane_extrapolation_uncertainty_scale > 0.0)
  {
    const double plane_ratio = std::max(
        0.0, eigenvalues.minCoeff()) /
        std::max(1e-9, eigenvalues(1));
    *additional_variance +=
        options_.plane_extrapolation_uncertainty_scale *
        plane_ratio * tangent_squared /
        std::max(1.0, effective_support);
  }
  if (options_.plane_parameter_uncertainty_scale > 0.0)
  {
    const double support = std::max(1.0, effective_support);
    const double normal_eigenvalue = std::max(0.0, eigenvalues(0));
    double parameter_variance = normal_eigenvalue / support;
    for (int tangent_axis = 1; tangent_axis < 3; ++tangent_axis)
    {
      const double tangent_eigenvalue =
          std::max(normal_eigenvalue, eigenvalues(tangent_axis));
      const double eigenvalue_gap =
          std::max(1e-9, tangent_eigenvalue - normal_eigenvalue);
      const double normal_variance =
          normal_eigenvalue * tangent_eigenvalue /
          (support * eigenvalue_gap * eigenvalue_gap);
      const double tangent_offset =
          eigenvectors.col(tangent_axis).dot(delta);
      parameter_variance +=
          normal_variance * tangent_offset * tangent_offset;
    }
    *additional_variance +=
        options_.plane_parameter_uncertainty_scale * parameter_variance;
  }
  return std::isfinite(*additional_variance);
}

bool LidarOdometry::findLocalPlane(const Eigen::Vector3d &world_point,
                                    PlaneMatch &match) const
{
  if (options_.use_adaptive_subvoxel_plane &&
      findAdaptiveSubvoxelPlane(world_point, match))
  {
    return true;
  }
  if (options_.use_point_knn_plane) return findPointKnnPlane(world_point, match);
  if (options_.use_compatible_voxel_plane)
  {
    if (findCompatibleVoxelPlane(world_point, match)) return true;
  }
  return findSmoothVoxelPlane(world_point, match);
}

bool LidarOdometry::findAdaptiveSubvoxelPlane(
    const Eigen::Vector3d &world_point, PlaneMatch &match) const
{
  if (adaptive_subvoxels_.empty()) return false;

  // A validated root plane is the stable map representation. Query the finer
  // level only for a genuinely non-planar root cell; allowing both models to
  // compete made correspondence selection drift as the map became denser.
  const VoxelKey root_key = voxelKey(world_point, options_.map_voxel_size);
  const auto root_iterator = map_voxels_.find(root_key);
  if (root_iterator != map_voxels_.end() &&
      root_iterator->second.plane_valid)
  {
    return false;
  }

  const double subvoxel_size =
      options_.map_voxel_size * options_.adaptive_subvoxel_scale;
  const VoxelKey center = voxelKey(world_point, subvoxel_size);
  const int radius = options_.adaptive_subvoxel_search_radius;
  const double maximum_distance_squared =
      options_.max_correspondence_distance *
      options_.max_correspondence_distance;
  const double measurement_variance =
      options_.lidar_measurement_noise *
      options_.lidar_measurement_noise;
  bool found = false;
  double best_score = std::numeric_limits<double>::infinity();
  for (int dx = -radius; dx <= radius; ++dx)
  {
    for (int dy = -radius; dy <= radius; ++dy)
    {
      for (int dz = -radius; dz <= radius; ++dz)
      {
        const auto iterator = adaptive_subvoxels_.find(
            VoxelKey{center.x + dx, center.y + dy, center.z + dz});
        if (iterator == adaptive_subvoxels_.end() ||
            !iterator->second.plane_valid)
        {
          continue;
        }
        const MapVoxel &voxel = iterator->second;
        const Eigen::Vector3d delta = world_point - voxel.mean;
        const double squared_distance = delta.squaredNorm();
        if (squared_distance > maximum_distance_squared) continue;
        const double residual = std::abs(voxel.normal.dot(delta));
        if (residual > options_.max_plane_distance) continue;
        double support_variance = 0.0;
        if (!planeSupportUncertainty(
                world_point, voxel.mean, voxel.normal,
                voxel.plane_eigenvalues, voxel.plane_eigenvectors,
                static_cast<double>(voxel.count), &support_variance))
        {
          continue;
        }
        const double variance =
            std::max(1e-6, voxel.plane_variance) +
            options_.plane_uncertainty_scale * voxel.plane_ratio /
                static_cast<double>(std::max(1, voxel.count)) +
            support_variance;
        const double score =
            residual / std::sqrt(std::max(1e-8,
                measurement_variance + variance)) +
            0.05 * std::sqrt(squared_distance) / subvoxel_size;
        if (!std::isfinite(score) || score >= best_score) continue;
        best_score = score;
        match.center = voxel.mean;
        match.normal = voxel.normal;
        match.variance = variance;
        match.nearest_squared_distance = squared_distance;
        found = true;
      }
    }
  }
  return found;
}

bool LidarOdometry::findCompatibleVoxelPlane(const Eigen::Vector3d &world_point,
                                              PlaneMatch &match) const
{
  if (map_voxels_.empty()) return false;
  const VoxelKey center = voxelKey(world_point, options_.map_voxel_size);
  const int radius = options_.compatible_voxel_search_radius > 0
      ? options_.compatible_voxel_search_radius
      : std::max(1, static_cast<int>(std::ceil(
          options_.max_correspondence_distance /
          options_.map_voxel_size)));
  const double maximum_distance_squared = options_.max_correspondence_distance *
      options_.max_correspondence_distance;
  struct Candidate
  {
    const MapVoxel *voxel = nullptr;
    double squared_distance = 0.0;
    double residual = 0.0;
    double score = std::numeric_limits<double>::infinity();
  };
  std::vector<Candidate> candidates;
  candidates.reserve(static_cast<std::size_t>((2 * radius + 1) *
                                               (2 * radius + 1) *
                                               (2 * radius + 1)));
  for (int dx = -radius; dx <= radius; ++dx)
  {
    for (int dy = -radius; dy <= radius; ++dy)
    {
      for (int dz = -radius; dz <= radius; ++dz)
      {
        const auto iterator = map_voxels_.find(
            VoxelKey{center.x + dx, center.y + dy, center.z + dz});
        if (iterator == map_voxels_.end() || !iterator->second.plane_valid) continue;
        const MapVoxel &voxel = iterator->second;
        const Eigen::Vector3d delta = world_point - voxel.mean;
        const double squared_distance = delta.squaredNorm();
        if (squared_distance > maximum_distance_squared) continue;
        const double residual = std::abs(voxel.normal.dot(delta));
        if (residual > options_.max_plane_distance) continue;
        const double normalized_variance = std::sqrt(std::max(
            1e-6, voxel.plane_variance +
                options_.lidar_measurement_noise * options_.lidar_measurement_noise));
        const double score = residual / normalized_variance +
            0.05 * std::sqrt(squared_distance) / options_.map_voxel_size;
        candidates.push_back(Candidate{&voxel, squared_distance, residual, score});
      }
    }
  }
  if (candidates.empty()) return false;
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &first, const Candidate &second)
            { return first.score < second.score; });
  if (candidates.size() > static_cast<std::size_t>(options_.max_plane_neighbors))
  {
    candidates.resize(static_cast<std::size_t>(options_.max_plane_neighbors));
  }

  const MapVoxel &reference = *candidates.front().voxel;
  const double minimum_normal_cosine = std::cos(20.0 * kPi / 180.0);
  const double offset_gate = std::max(options_.plane_fit_residual_gate,
                                      3.0 * std::sqrt(reference.plane_variance));
  const double kernel_sigma = std::max(options_.map_voxel_size,
                                       0.5 * options_.max_correspondence_distance);
  const double inverse_two_sigma_squared = 0.5 / (kernel_sigma * kernel_sigma);
  Eigen::Vector3d weighted_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d weighted_normal = Eigen::Vector3d::Zero();
  double weighted_variance = 0.0;
  double weighted_ratio = 0.0;
  double total_weight = 0.0;
  int compatible_count = 0;
  for (const Candidate &candidate : candidates)
  {
    const MapVoxel &voxel = *candidate.voxel;
    double alignment = reference.normal.dot(voxel.normal);
    if (std::abs(alignment) < minimum_normal_cosine) continue;
    Eigen::Vector3d normal = voxel.normal;
    if (alignment < 0.0) normal = -normal;
    if (std::abs(reference.normal.dot(voxel.mean - reference.mean)) > offset_gate)
    {
      continue;
    }
    const double variance = std::max(
        1e-5, voxel.plane_variance +
            options_.lidar_measurement_noise * options_.lidar_measurement_noise);
    const double weight = std::exp(
        -candidate.squared_distance * inverse_two_sigma_squared) / variance;
    weighted_center += weight * voxel.mean;
    weighted_normal += weight * normal;
    weighted_variance += weight * voxel.plane_variance;
    weighted_ratio += weight * voxel.plane_ratio;
    total_weight += weight;
    ++compatible_count;
  }
  if (compatible_count == 0 || total_weight < 1e-6 ||
      weighted_normal.norm() < 1e-6)
  {
    return false;
  }
  match.center = weighted_center / total_weight;
  match.normal = weighted_normal.normalized();
  const double residual = std::abs(match.normal.dot(world_point - match.center));
  if (!match.center.allFinite() || !match.normal.allFinite() ||
      residual > options_.max_plane_distance)
  {
    return false;
  }
  match.variance = std::max(1e-6, weighted_variance / total_weight) +
      options_.plane_uncertainty_scale * (weighted_ratio / total_weight) /
          static_cast<double>(std::max(1, compatible_count));
  match.nearest_squared_distance = candidates.front().squared_distance;
  return true;
}

bool LidarOdometry::findPointKnnPlane(const Eigen::Vector3d &world_point,
                                      PlaneMatch &match) const
{
  if (map_voxels_.empty()) return false;
  const VoxelKey center = voxelKey(world_point, options_.map_voxel_size);
  // The configured voxel radius bounds hash lookups per residual. Euclidean
  // distance gating below still rejects samples outside the association range.
  const int radius = options_.normal_neighbor_voxels;
  const double maximum_distance_squared = options_.max_correspondence_distance *
      options_.max_correspondence_distance;
  struct PointNeighbor
  {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    double squared_distance = 0.0;
  };
  std::vector<PointNeighbor, Eigen::aligned_allocator<PointNeighbor>> neighbors;
  neighbors.reserve(static_cast<std::size_t>(options_.max_plane_neighbors * 4));
  for (int dx = -radius; dx <= radius; ++dx)
  {
    for (int dy = -radius; dy <= radius; ++dy)
    {
      for (int dz = -radius; dz <= radius; ++dz)
      {
        const auto iterator = map_voxels_.find(
            VoxelKey{center.x + dx, center.y + dy, center.z + dz});
        if (iterator == map_voxels_.end())
        {
          continue;
        }
        for (const Eigen::Vector3d &sample : iterator->second.samples)
        {
          const double squared_distance = (sample - world_point).squaredNorm();
          if (squared_distance <= maximum_distance_squared)
          {
            neighbors.push_back(PointNeighbor{sample, squared_distance});
          }
        }
      }
    }
  }
  if (neighbors.size() < static_cast<std::size_t>(options_.min_normal_neighbors)) return false;
  if (neighbors.size() > static_cast<std::size_t>(options_.max_plane_neighbors))
  {
    std::nth_element(neighbors.begin(), neighbors.begin() + options_.max_plane_neighbors,
                     neighbors.end(),
                     [](const PointNeighbor &first, const PointNeighbor &second)
                     { return first.squared_distance < second.squared_distance; });
    neighbors.resize(static_cast<std::size_t>(options_.max_plane_neighbors));
  }
  std::sort(neighbors.begin(), neighbors.end(),
            [](const PointNeighbor &first, const PointNeighbor &second)
            { return first.squared_distance < second.squared_distance; });

  const double kernel_sigma = std::max(options_.map_voxel_size,
                                        0.35 * options_.max_correspondence_distance);
  const double inverse_two_sigma_squared = 0.5 / (kernel_sigma * kernel_sigma);
  const auto fit_plane = [&](const std::vector<int> &indices, Eigen::Vector3d *mean,
                             Eigen::Vector3d *normal, Eigen::Vector3d *eigenvalues,
                             Eigen::Matrix3d *eigenvectors,
                             double *total_weight) -> bool
  {
    if (indices.size() < static_cast<std::size_t>(options_.min_normal_neighbors)) return false;
    mean->setZero();
    *total_weight = 0.0;
    for (int index : indices)
    {
      const PointNeighbor &neighbor = neighbors[static_cast<std::size_t>(index)];
      const double weight = std::exp(-neighbor.squared_distance * inverse_two_sigma_squared);
      *mean += weight * neighbor.point;
      *total_weight += weight;
    }
    if (*total_weight < 1e-3) return false;
    *mean /= *total_weight;
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int index : indices)
    {
      const PointNeighbor &neighbor = neighbors[static_cast<std::size_t>(index)];
      const double weight = std::exp(-neighbor.squared_distance * inverse_two_sigma_squared);
      const Eigen::Vector3d centered = neighbor.point - *mean;
      covariance.noalias() += weight * centered * centered.transpose();
    }
    covariance /= *total_weight;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) return false;
    *eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    *eigenvectors = solver.eigenvectors();
    *normal = eigenvectors->col(0).normalized();
    return normal->allFinite() && eigenvectors->allFinite();
  };

  std::vector<int> inliers(neighbors.size());
  std::iota(inliers.begin(), inliers.end(), 0);
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Matrix3d eigenvectors = Eigen::Matrix3d::Identity();
  double total_weight = 0.0;
  if (!fit_plane(inliers, &mean, &normal, &eigenvalues, &eigenvectors,
                 &total_weight))
  {
    return false;
  }

  std::vector<int> robust_inliers;
  robust_inliers.reserve(inliers.size());
  for (int index : inliers)
  {
    if (std::abs(normal.dot(neighbors[static_cast<std::size_t>(index)].point - mean)) <=
        options_.plane_fit_residual_gate)
    {
      robust_inliers.push_back(index);
    }
  }
  if (robust_inliers.size() < static_cast<std::size_t>(options_.min_normal_neighbors) ||
      !fit_plane(robust_inliers, &mean, &normal, &eigenvalues, &eigenvectors,
                 &total_weight))
  {
    return false;
  }
  if (eigenvalues.y() < 1e-6 || eigenvalues.x() > options_.max_plane_variance ||
      eigenvalues.x() > options_.plane_max_eigen_ratio * eigenvalues.y())
  {
    return false;
  }
  double extrapolation_variance = 0.0;
  if (!planeSupportUncertainty(world_point, mean, normal, eigenvalues, eigenvectors,
                               total_weight, &extrapolation_variance))
  {
    return false;
  }
  match.center = mean;
  match.normal = normal;
  const double ratio = eigenvalues.x() / std::max(1e-9, eigenvalues.y());
  match.variance = std::max(1e-6, eigenvalues.x()) +
      options_.plane_uncertainty_scale * ratio / total_weight +
      extrapolation_variance;
  match.nearest_squared_distance = neighbors.front().squared_distance;
  return true;
}

bool LidarOdometry::findSmoothVoxelPlane(const Eigen::Vector3d &world_point,
                                          PlaneMatch &match) const
{
  if (map_voxels_.empty()) return false;
  const VoxelKey center = voxelKey(world_point, options_.map_voxel_size);
  const int radius = std::max(options_.normal_neighbor_voxels,
      static_cast<int>(std::ceil(options_.max_correspondence_distance /
                                 options_.map_voxel_size)) - 1);
  const double maximum_distance_squared = options_.max_correspondence_distance *
      options_.max_correspondence_distance;
  struct VoxelNeighbor
  {
    const MapVoxel *voxel = nullptr;
    double squared_distance = 0.0;
  };
  std::vector<VoxelNeighbor> neighbors;
  neighbors.reserve(static_cast<std::size_t>((2 * radius + 1) *
                                              (2 * radius + 1) * (2 * radius + 1)));
  for (int dx = -radius; dx <= radius; ++dx)
  {
    for (int dy = -radius; dy <= radius; ++dy)
    {
      for (int dz = -radius; dz <= radius; ++dz)
      {
        const auto iterator = map_voxels_.find(
            VoxelKey{center.x + dx, center.y + dy, center.z + dz});
        if (iterator == map_voxels_.end()) continue;
        const double squared_distance =
            (iterator->second.mean - world_point).squaredNorm();
        if (squared_distance <= maximum_distance_squared)
        {
          neighbors.push_back(VoxelNeighbor{&iterator->second, squared_distance});
        }
      }
    }
  }
  if (neighbors.size() < static_cast<std::size_t>(options_.min_normal_neighbors)) return false;
  if (neighbors.size() > static_cast<std::size_t>(options_.max_plane_neighbors))
  {
    std::nth_element(neighbors.begin(), neighbors.begin() + options_.max_plane_neighbors,
                     neighbors.end(),
                     [](const VoxelNeighbor &first, const VoxelNeighbor &second)
                     { return first.squared_distance < second.squared_distance; });
    neighbors.resize(static_cast<std::size_t>(options_.max_plane_neighbors));
  }

  const double kernel_sigma = std::max(options_.map_voxel_size,
                                        0.5 * options_.max_correspondence_distance);
  const double inverse_two_sigma_squared = 0.5 / (kernel_sigma * kernel_sigma);
  const auto neighbor_weight = [&](const VoxelNeighbor &neighbor)
  {
    const double maturity = std::min(
        1.0, static_cast<double>(neighbor.voxel->count) /
                 static_cast<double>(options_.min_voxel_plane_points));
    return maturity * std::exp(
        -neighbor.squared_distance * inverse_two_sigma_squared);
  };
  const auto fit_plane = [&](const std::vector<int> &indices,
                             Eigen::Vector3d *mean,
                             Eigen::Vector3d *normal,
                             Eigen::Vector3d *eigenvalues,
                             Eigen::Matrix3d *eigenvectors,
                             double *total_weight) -> bool
  {
    if (indices.size() <
        static_cast<std::size_t>(options_.min_normal_neighbors))
    {
      return false;
    }
    mean->setZero();
    *total_weight = 0.0;
    for (int index : indices)
    {
      const VoxelNeighbor &neighbor =
          neighbors[static_cast<std::size_t>(index)];
      const double weight = neighbor_weight(neighbor);
      *mean += weight * neighbor.voxel->mean;
      *total_weight += weight;
    }
    if (*total_weight < 1e-3) return false;
    *mean /= *total_weight;

    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (int index : indices)
    {
      const VoxelNeighbor &neighbor =
          neighbors[static_cast<std::size_t>(index)];
      const MapVoxel &voxel = *neighbor.voxel;
      const double weight = neighbor_weight(neighbor);
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
      if (voxel.count > 1)
      {
        covariance = voxel.scatter /
            static_cast<double>(voxel.count - 1);
      }
      const Eigen::Vector3d centered = voxel.mean - *mean;
      scatter.noalias() +=
          weight * (covariance + centered * centered.transpose());
    }
    scatter /= *total_weight;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
    if (solver.info() != Eigen::Success) return false;
    *eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    *eigenvectors = solver.eigenvectors();
    *normal = eigenvectors->col(0).normalized();
    return normal->allFinite() && eigenvalues->allFinite() &&
        eigenvectors->allFinite();
  };

  std::vector<int> fit_indices(neighbors.size());
  std::iota(fit_indices.begin(), fit_indices.end(), 0);
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Matrix3d eigenvectors = Eigen::Matrix3d::Identity();
  double total_weight = 0.0;
  if (!fit_plane(fit_indices, &mean, &normal, &eigenvalues, &eigenvectors,
                 &total_weight))
  {
    return false;
  }

  if (options_.smooth_voxel_robust_refit)
  {
    std::vector<int> robust_indices;
    robust_indices.reserve(fit_indices.size());
    for (int index : fit_indices)
    {
      const MapVoxel &voxel =
          *neighbors[static_cast<std::size_t>(index)].voxel;
      double mean_variance = 0.0;
      if (voxel.count > 1)
      {
        const Eigen::Matrix3d covariance =
            voxel.scatter / static_cast<double>(voxel.count - 1);
        mean_variance = std::max(
            0.0, normal.dot(covariance * normal) /
                     static_cast<double>(voxel.count));
      }
      const double gate = options_.plane_fit_residual_gate +
          2.5 * std::sqrt(mean_variance);
      if (std::abs(normal.dot(voxel.mean - mean)) <= gate)
      {
        robust_indices.push_back(index);
      }
    }
    if (robust_indices.size() <
        static_cast<std::size_t>(options_.min_normal_neighbors))
    {
      return false;
    }
    if (robust_indices.size() != fit_indices.size() &&
        !fit_plane(robust_indices, &mean, &normal, &eigenvalues, &eigenvectors,
                   &total_weight))
    {
      return false;
    }
    fit_indices.swap(robust_indices);
  }

  if (eigenvalues.y() < 1e-6 || eigenvalues.x() > options_.max_plane_variance ||
      eigenvalues.x() > options_.plane_max_eigen_ratio * eigenvalues.y())
  {
    return false;
  }
  double extrapolation_variance = 0.0;
  if (!planeSupportUncertainty(world_point, mean, normal, eigenvalues, eigenvectors,
                               total_weight, &extrapolation_variance))
  {
    return false;
  }
  match.center = mean;
  match.normal = normal;
  const double ratio = eigenvalues.x() / std::max(1e-9, eigenvalues.y());
  match.variance = std::max(1e-6, eigenvalues.x()) +
      options_.plane_uncertainty_scale * ratio / total_weight +
      extrapolation_variance;
  match.nearest_squared_distance = std::numeric_limits<double>::infinity();
  for (int index : fit_indices)
  {
    match.nearest_squared_distance = std::min(
        match.nearest_squared_distance,
        neighbors[static_cast<std::size_t>(index)].squared_distance);
  }
  return true;
}

void LidarOdometry::updateVoxel(MapVoxel &voxel, const Eigen::Vector3d &point)
{
  // Match FAST-LIVO2's octree behavior: only a validated planar cell becomes
  // immutable.  Freezing a full but non-planar cell prevents later views from
  // separating an edge/mixed surface and permanently removes useful map
  // support.
  if (options_.freeze_mature_voxels && voxel.plane_valid &&
      voxel.count >= options_.max_voxel_points)
  {
    voxel.last_seen_scan = accepted_scan_count_;
    return;
  }
  const bool mature_valid_plane = voxel.plane_valid &&
      voxel.count >= options_.max_voxel_points;
  voxel.plane_valid = false;
  if (voxel.samples.size() < static_cast<std::size_t>(options_.max_voxel_samples))
  {
    voxel.samples.push_back(point);
  }
  else
  {
    voxel.sample_cursor = (voxel.sample_cursor + 1) % options_.max_voxel_samples;
    voxel.samples[static_cast<std::size_t>(voxel.sample_cursor)] = point;
  }
  if (voxel.count < options_.max_voxel_points)
  {
    ++voxel.count;
    const Eigen::Vector3d delta = point - voxel.mean;
    voxel.mean += delta / static_cast<double>(voxel.count);
    voxel.scatter += delta * (point - voxel.mean).transpose();
  }
  else
  {
    const double effective_count = static_cast<double>(voxel.count);
    Eigen::Matrix3d covariance = voxel.scatter / std::max(1.0, effective_count - 1.0);
    const Eigen::Vector3d delta = point - voxel.mean;
    const double gain = mature_valid_plane
        ? options_.mature_voxel_update_gain : 1.0;
    const double alpha = gain / effective_count;
    voxel.mean += alpha * delta;
    const Eigen::Vector3d centered = point - voxel.mean;
    covariance = (1.0 - alpha) * covariance + alpha * centered * centered.transpose();
    voxel.scatter = (effective_count - 1.0) *
                    0.5 * (covariance + covariance.transpose());
  }
  voxel.last_seen_scan = accepted_scan_count_;
}

bool LidarOdometry::shouldInsertMap(const State &state, double stamp) const
{
  if (!have_last_map_insert_pose_)
  {
    return true;
  }
  const bool motion_gate_enabled =
      options_.map_insertion_min_translation > 0.0 ||
      options_.map_insertion_min_rotation_deg > 0.0;
  const bool interval_gate_enabled = options_.map_insertion_max_interval > 0.0;
  if (!motion_gate_enabled && !interval_gate_enabled)
  {
    return true;
  }

  const Eigen::Isometry3d relative =
      last_map_insert_pose_.inverse() * statePose(state);
  if (options_.map_insertion_min_translation > 0.0 &&
      relative.translation().norm() >=
          options_.map_insertion_min_translation)
  {
    return true;
  }
  if (options_.map_insertion_min_rotation_deg > 0.0 &&
      rotationDegrees(relative.rotation()) >=
          options_.map_insertion_min_rotation_deg)
  {
    return true;
  }
  return interval_gate_enabled && std::isfinite(stamp) &&
      stamp - last_map_insert_stamp_ >= options_.map_insertion_max_interval;
}

void LidarOdometry::updateVoxelPlane(MapVoxel &voxel)
{
  voxel.plane_valid = false;
  voxel.plane_eigenvalues =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  voxel.plane_eigenvectors.setIdentity();
  voxel.plane_variance = std::numeric_limits<double>::infinity();
  voxel.plane_ratio = std::numeric_limits<double>::infinity();
  if (voxel.count < options_.min_voxel_plane_points) return;
  const Eigen::Matrix3d covariance =
      0.5 * (voxel.scatter + voxel.scatter.transpose()) /
      static_cast<double>(std::max(1, voxel.count - 1));
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) return;
  const Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseMax(0.0);
  if (eigenvalues.y() < 1e-8) return;
  voxel.plane_eigenvalues = eigenvalues;
  voxel.plane_eigenvectors = solver.eigenvectors();
  voxel.plane_variance = std::max(1e-6, eigenvalues.x());
  voxel.plane_ratio = eigenvalues.x() / std::max(1e-9, eigenvalues.y());
  voxel.normal = solver.eigenvectors().col(0).normalized();
  voxel.plane_valid = voxel.normal.allFinite() &&
      voxel.plane_variance <= options_.max_plane_variance &&
      voxel.plane_ratio <= options_.plane_max_eigen_ratio;
}

void LidarOdometry::insertMapPoints(const PointVector &body_points, const State &state,
                                    bool filter_existing)
{
  const PointVector insertion_points = voxelDownsample(
      body_points, options_.map_insert_voxel_size, options_.max_scan_points * 2);
  std::unordered_set<VoxelKey, VoxelKeyHash> touched;
  touched.reserve(insertion_points.size());
  std::unordered_set<VoxelKey, VoxelKeyHash> touched_subvoxels;
  if (options_.use_adaptive_subvoxel_plane)
  {
    touched_subvoxels.reserve(insertion_points.size());
  }
  const double subvoxel_size =
      options_.map_voxel_size * options_.adaptive_subvoxel_scale;
  for (const Eigen::Vector3d &body_point : insertion_points)
  {
    const Eigen::Vector3d world_point = state.rotation * body_point + state.position;
    if (!world_point.allFinite() ||
        (world_point - state.position).norm() > options_.local_map_radius)
    {
      continue;
    }
    const VoxelKey key = voxelKey(world_point, options_.map_voxel_size);
    const auto existing = map_voxels_.find(key);
    bool insert_root = true;
    if (filter_existing && existing != map_voxels_.end() && existing->second.plane_valid)
    {
      const MapVoxel &voxel = existing->second;
      if (std::abs(voxel.normal.dot(world_point - voxel.mean)) >
          options_.map_insertion_max_plane_distance)
      {
        insert_root = false;
      }
    }
    if (insert_root)
    {
      updateVoxel(map_voxels_[key], world_point);
      touched.insert(key);
    }
    // Do not let points rejected by a mature root plane seed the fine map.
    // Those points are commonly moving objects or pose-error outliers and
    // were the source of a progressively self-consistent secondary surface.
    if (options_.use_adaptive_subvoxel_plane && insert_root)
    {
      const VoxelKey subvoxel_key = voxelKey(world_point, subvoxel_size);
      updateVoxel(adaptive_subvoxels_[subvoxel_key], world_point);
      touched_subvoxels.insert(subvoxel_key);
    }
  }
  for (const VoxelKey &key : touched)
  {
    const auto iterator = map_voxels_.find(key);
    if (iterator != map_voxels_.end()) updateVoxelPlane(iterator->second);
  }
  for (const VoxelKey &key : touched_subvoxels)
  {
    const auto iterator = adaptive_subvoxels_.find(key);
    if (iterator != adaptive_subvoxels_.end())
    {
      updateVoxelPlane(iterator->second);
    }
  }
  const bool periodic_local_prune =
      !options_.retain_global_map && accepted_scan_count_ % 10 == 0;
  if (periodic_local_prune ||
      map_voxels_.size() >
          static_cast<std::size_t>(1.10 * options_.max_map_points) ||
      adaptive_subvoxels_.size() >
          static_cast<std::size_t>(
              1.10 * options_.max_adaptive_subvoxels))
  {
    pruneMap();
  }
}

void LidarOdometry::pruneMap()
{
  const double radius_squared = options_.local_map_radius * options_.local_map_radius;
  struct RankedKey
  {
    VoxelKey key;
    double squared_distance = 0.0;
  };
  const auto prune_voxels =
      [&](MapVoxelMap *voxels, int maximum_count)
  {
    if (!options_.retain_global_map)
    {
      for (auto iterator = voxels->begin(); iterator != voxels->end();)
      {
        if ((iterator->second.mean - state_.position).squaredNorm() >
            radius_squared)
        {
          iterator = voxels->erase(iterator);
        }
        else
        {
          ++iterator;
        }
      }
    }
    if (voxels->size() <= static_cast<std::size_t>(maximum_count)) return;
    std::vector<RankedKey> ranked;
    ranked.reserve(voxels->size());
    for (const auto &entry : *voxels)
    {
      ranked.push_back(RankedKey{
          entry.first,
          (entry.second.mean - state_.position).squaredNorm()});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedKey &first, const RankedKey &second)
              { return first.squared_distance < second.squared_distance; });
    for (std::size_t index = static_cast<std::size_t>(maximum_count);
         index < ranked.size(); ++index)
    {
      voxels->erase(ranked[index].key);
    }
  };
  prune_voxels(&map_voxels_, options_.max_map_points);
  prune_voxels(&adaptive_subvoxels_, options_.max_adaptive_subvoxels);
}

void LidarOdometry::pruneImu(double stamp)
{
  const double oldest = stamp - std::min(1.0, options_.imu_buffer_duration);
  while (imu_buffer_.size() > 2U && imu_buffer_[1].stamp < oldest)
  {
    imu_buffer_.pop_front();
  }
}

void LidarOdometry::recordPropagationTrajectory(
    const std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> &trajectory)
{
  for (const ImuPose &pose : trajectory)
  {
    if (!std::isfinite(pose.stamp) || pose.stamp <= 0.0) continue;
    if (propagation_history_.empty() ||
        pose.stamp > propagation_history_.back().stamp + 1e-9)
    {
      propagation_history_.push_back(pose);
    }
    else if (std::abs(pose.stamp - propagation_history_.back().stamp) <= 1e-9)
    {
      propagation_history_.back() = pose;
    }
  }
  if (propagation_history_.empty()) return;
  const double keep_after = propagation_history_.back().stamp - 1.5;
  while (propagation_history_.size() > 2U &&
         propagation_history_[1].stamp < keep_after)
  {
    propagation_history_.pop_front();
  }
}

void LidarOdometry::recordCurrentPose()
{
  if (!std::isfinite(state_stamp_) || state_stamp_ <= 0.0) return;
  ImuPose pose;
  pose.stamp = state_stamp_;
  pose.rotation = state_.rotation;
  pose.position = state_.position;
  pose.velocity = state_.velocity;
  std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> trajectory;
  trajectory.push_back(pose);
  recordPropagationTrajectory(trajectory);
}

bool LidarOdometry::applyLidarLossProtection(const State &state_before_scan,
                                             double scan_end_stamp)
{
  ++consecutive_rejections_;
  if (options_.recovery_map_insert_min_consecutive_strong_support > 0)
  {
    recovery_map_guard_active_ = true;
    recovery_map_trusted_scan_count_ = 0;
  }
  lidar_loss_limited_ = false;
  lidar_loss_frozen_ = false;
  if (!map_initialized_ || options_.lidar_loss_hold_after_rejections <= 0 ||
      consecutive_rejections_ < options_.lidar_loss_hold_after_rejections)
  {
    return false;
  }

  WheelSample wheel_measurement;
  const bool use_wheel_dead_reckoning =
      options_.lidar_loss_use_wheel_dead_reckoning &&
      wheelMeasurement(scan_end_stamp, &wheel_measurement) &&
      std::isfinite(wheel_measurement.forward_speed);

  if (!use_wheel_dead_reckoning &&
      options_.lidar_loss_freeze_after_rejections > 0 &&
      consecutive_rejections_ >= options_.lidar_loss_freeze_after_rejections)
  {
    // Without an independent speed observation, a long inertial-only outage
    // is not trustworthy. Keep the legacy freeze for that case. With wheel
    // coverage, freezing a moving platform makes scan-to-map reacquisition
    // impossible, so the bounded dead-reckoning path below remains active.
    state_.rotation = last_accepted_state_.rotation;
    state_.position = last_accepted_state_.position;
    state_.velocity.setZero();
    lidar_loss_limited_ = true;
    lidar_loss_frozen_ = true;
    return true;
  }

  // Preserve the propagated attitude when it is numerically sound, but make
  // position and velocity conservative. The latest accepted LiDAR state is
  // the only reliable vertical reference during an observation outage.
  if (!state_.position.allFinite() || !state_.velocity.allFinite() ||
      !state_.rotation.allFinite())
  {
    state_ = last_accepted_state_;
  }
  const double dt = previous_scan_stamp_ > 0.0
      ? std::max(0.0, scan_end_stamp - previous_scan_stamp_) : 0.0;
  State reference_state = state_before_scan;
  if (!reference_state.position.allFinite() || !reference_state.velocity.allFinite())
  {
    reference_state = last_accepted_state_;
  }
  Eigen::Vector2d horizontal_velocity = reference_state.velocity.head<2>();
  Eigen::Vector2d current_horizontal_velocity = state_.velocity.head<2>();
  if (use_wheel_dead_reckoning)
  {
    // Integrate the wheel speed at the midpoint attitude. This preserves the
    // IMU turn while removing the accelerometer double-integration that caused
    // the prediction to leave the local map after several rejected scans.
    Eigen::Quaterniond reference_orientation(reference_state.rotation);
    Eigen::Quaterniond current_orientation(state_.rotation);
    reference_orientation.normalize();
    current_orientation.normalize();
    const Eigen::Matrix3d midpoint_rotation =
        reference_orientation.slerp(0.5, current_orientation).toRotationMatrix();
    horizontal_velocity =
        (midpoint_rotation *
         Eigen::Vector3d(wheel_measurement.forward_speed, 0.0, 0.0)).head<2>();
    current_horizontal_velocity =
        (state_.rotation *
         Eigen::Vector3d(wheel_measurement.forward_speed, 0.0, 0.0)).head<2>();
  }
  if (!horizontal_velocity.allFinite())
  {
    horizontal_velocity.setZero();
  }
  const double horizontal_speed = horizontal_velocity.norm();
  if (options_.lidar_loss_max_horizontal_speed > 0.0 &&
      horizontal_speed > options_.lidar_loss_max_horizontal_speed)
  {
    horizontal_velocity *= options_.lidar_loss_max_horizontal_speed / horizontal_speed;
  }
  Eigen::Vector2d horizontal_step = horizontal_velocity * dt;
  const double step_norm = horizontal_step.norm();
  if (options_.lidar_loss_max_horizontal_step > 0.0 &&
      step_norm > options_.lidar_loss_max_horizontal_step)
  {
    horizontal_step *= options_.lidar_loss_max_horizontal_step / step_norm;
  }
  state_.position.head<2>() = reference_state.position.head<2>() + horizontal_step;
  state_.velocity.head<2>() = use_wheel_dead_reckoning
      ? current_horizontal_velocity
      : horizontal_velocity * options_.lidar_loss_velocity_decay;

  const double vertical_offset = state_.position.z() - last_accepted_state_.position.z();
  const double vertical_limit = options_.lidar_loss_max_vertical_offset;
  state_.position.z() = last_accepted_state_.position.z() + std::max(
      -vertical_limit, std::min(vertical_limit,
                                std::isfinite(vertical_offset) ? vertical_offset : 0.0));
  state_.velocity.z() = 0.0;
  lidar_loss_limited_ = true;
  return true;
}

std::vector<LidarOdometry::ImuPose, Eigen::aligned_allocator<LidarOdometry::ImuPose>>
LidarOdometry::deskewTrajectory(double end_stamp) const
{
  std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> trajectory;
  const double keep_after = end_stamp - 1.0;
  for (const ImuPose &pose : propagation_history_)
  {
    if (pose.stamp + 1e-9 < keep_after || pose.stamp > end_stamp + 1e-8) continue;
    trajectory.push_back(pose);
  }
  if (trajectory.empty() && state_stamp_ > 0.0)
  {
    ImuPose pose;
    pose.stamp = state_stamp_;
    pose.rotation = state_.rotation;
    pose.position = state_.position;
    pose.velocity = state_.velocity;
    trajectory.push_back(pose);
  }
  return trajectory;
}

void LidarOdometry::fillResultState(LidarOdometryResult &result) const
{
  result.initialized = map_initialized_;
  result.imu_initialized = imu_initialized_;
  result.imu_init_progress = imu_init_progress_;
  result.imu_samples = static_cast<int>(imu_buffer_.size());
  result.acceleration_scale = acceleration_scale_;
  result.loss_limited = lidar_loss_limited_;
  result.loss_frozen = lidar_loss_frozen_;
  result.consecutive_rejections = consecutive_rejections_;
  result.pose = pose_cache_;
  result.velocity = state_.velocity;
  result.gyro_bias = state_.gyro_bias;
  result.acceleration_bias = state_.acceleration_bias;
  result.gravity = state_.gravity;
  result.covariance.setZero();
  result.covariance.block<3, 3>(0, 0) = state_.covariance.block<3, 3>(0, 0);
  result.covariance.block<3, 3>(0, 3) = state_.covariance.block<3, 3>(0, 3);
  result.covariance.block<3, 3>(3, 0) = state_.covariance.block<3, 3>(3, 0);
  result.covariance.block<3, 3>(3, 3) = state_.covariance.block<3, 3>(3, 3);
}

LidarOdometryResult LidarOdometry::processScan(const PointVector &body_points,
                                               double scan_end_stamp)
{
  TimedPointVector timed_points;
  timed_points.reserve(body_points.size());
  for (const Eigen::Vector3d &point : body_points)
  {
    TimedPoint timed_point;
    timed_point.point = point;
    timed_points.push_back(timed_point);
  }
  return processScan(timed_points, scan_end_stamp);
}

VisualUpdateResult LidarOdometry::processVisual(
    double stamp, const VisualPoseLinearizer &linearizer)
{
  VisualUpdateResult result;
  result.stamp = stamp;
  result.pose = pose_cache_;
  if (!options_.visual_enabled)
  {
    result.reason = "visual_disabled";
    return result;
  }
  if (!linearizer || !std::isfinite(stamp))
  {
    result.reason = "invalid_visual_measurement";
    return result;
  }
  if (!map_initialized_ || (options_.imu_enabled && !imu_initialized_))
  {
    result.reason = "visual_waiting_for_lio";
    return result;
  }
  if (state_stamp_ > 0.0 && stamp + 1e-8 < state_stamp_)
  {
    result.reason = "visual_precedes_filter_state";
    return result;
  }

  std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> trajectory;
  if (options_.imu_enabled)
  {
    if (!propagateTo(stamp, trajectory))
    {
      result.reason = "visual_insufficient_imu_coverage";
      return result;
    }
  }
  else
  {
    if (state_stamp_ <= 0.0) state_stamp_ = stamp;
    predictWithoutImu(stamp, trajectory);
  }
  result.propagated = true;
  const State propagated_state = state_;
  const Eigen::Isometry3d propagated_pose = statePose(propagated_state);
  Matrix18d regularized_covariance = propagated_state.covariance;
  regularized_covariance.diagonal().array() += 1e-10;
  const Eigen::LDLT<Matrix18d> covariance_solver(regularized_covariance);
  if (covariance_solver.info() != Eigen::Success)
  {
    result.reason = "visual_invalid_predicted_covariance";
    return result;
  }
  const Matrix18d prior_information =
      covariance_solver.solve(Matrix18d::Identity());
  State estimate = propagated_state;
  Matrix18d final_information = prior_information;
  VisualPoseLinearization final_linearization;
  bool observability_rejected = false;
  bool two_mode_quality_rejected = false;
  // Decide the optional automatic Z channel from the first valid direct
  // linearization, then keep that axis set fixed for this ESKF update. This
  // keeps the final covariance consistent with the measurement that was
  // actually applied instead of toggling Z between Gauss-Newton iterations.
  LidarOdometryOptions visual_options = options_;
  const bool automatic_z_requested =
      options_.visual_fuse_translation_z_when_observable &&
      !options_.visual_fuse_translation_z;
  bool automatic_z_decided = !automatic_z_requested;
  bool automatic_z_enabled = false;

  for (int iteration = 0; iteration < options_.visual_max_iterations; ++iteration)
  {
    const VisualPoseLinearization linearization = linearizer(statePose(estimate));
    final_linearization = linearization;
    if (!linearization.valid ||
        linearization.landmarks < options_.visual_min_landmarks ||
        linearization.residuals < options_.visual_min_residuals ||
        !linearization.hessian.allFinite() ||
        !linearization.gradient.allFinite())
    {
      break;
    }
    const VisualMeasurementProjection projected_measurement =
        projectVisualMeasurement(linearization, visual_options);
    result.observable_directions = projected_measurement.observable_directions;
    if (!projected_measurement.valid ||
        projected_measurement.observable_directions <
            options_.visual_min_observable_directions)
    {
      observability_rejected = true;
      break;
    }
    if (!passesTwoModeVisualQuality(linearization,
                                    projected_measurement.observable_directions,
                                    options_))
    {
      two_mode_quality_rejected = true;
      break;
    }
    if (!automatic_z_decided)
    {
      automatic_z_decided = true;
      automatic_z_enabled = projected_measurement.z_fused;
      // Freeze the accepted axis set for all following iterations. When the
      // Z gate fails, the helper has already re-projected yaw/XY without the
      // coupled Z component.
      visual_options.visual_fuse_translation_z = automatic_z_enabled;
      visual_options.visual_fuse_translation_z_when_observable = false;
    }
    result.z_observable = projected_measurement.z_observable;
    result.z_projection = projected_measurement.z_projection;
    result.z_conditional_information_ratio =
        projected_measurement.z_conditional_information_ratio;
    Matrix18d measurement_hessian = Matrix18d::Zero();
    Vector18d measurement_gradient = Vector18d::Zero();
    measurement_hessian.block<6, 6>(0, 0) = projected_measurement.hessian;
    measurement_gradient.segment<6>(0) = projected_measurement.gradient;
    const Vector18d displacement = stateDifference(estimate, propagated_state);
    Matrix18d information = prior_information + measurement_hessian;
    information.diagonal().array() += options_.visual_solver_damping;
    const Vector18d gradient =
        prior_information * displacement + measurement_gradient;
    const Eigen::LDLT<Matrix18d> solver(information);
    if (solver.info() != Eigen::Success) break;
    Vector18d step = solver.solve(-gradient);
    if (!step.allFinite()) break;
    // Keep the prediction in the LiDAR/IMU state for axes that the current
    // image cannot observe. Without this projection, the prior term can turn
    // a shallow photometric valley into a spurious Z or roll/pitch update.
    step.segment<6>(0) = projected_measurement.projection * step.segment<6>(0);

    const double rotation_limit = options_.visual_max_rotation_step_deg * kPi / 180.0;
    const double rotation_norm = step.segment<3>(0).norm();
    if (rotation_norm > rotation_limit && rotation_norm > kSmall)
    {
      step.segment<3>(0) *= rotation_limit / rotation_norm;
    }
    const double translation_norm = step.segment<3>(3).norm();
    if (translation_norm > options_.visual_max_translation_step &&
        translation_norm > kSmall)
    {
      step.segment<3>(3) *= options_.visual_max_translation_step /
                            translation_norm;
    }
    if (automatic_z_enabled && options_.visual_max_z_step > 0.0)
    {
      const double current_z_correction =
          estimate.position.z() - propagated_state.position.z();
      const double bounded_z_correction = std::max(
          -options_.visual_max_z_step, std::min(options_.visual_max_z_step,
              current_z_correction + step(5)));
      step(5) = bounded_z_correction - current_z_correction;
    }
    if (options_.visual_fuse_correlated_states)
    {
      const Vector18d current_correction =
          stateDifference(estimate, propagated_state);
      const auto bound_total_correction =
          [&step, &current_correction](int start, double maximum)
          {
            if (maximum <= 0.0) return;
            Eigen::Vector3d bounded =
                current_correction.segment<3>(start) +
                step.segment<3>(start);
            const double norm = bounded.norm();
            if (norm > maximum && norm > kSmall)
            {
              bounded *= maximum / norm;
            }
            step.segment<3>(start) =
                bounded - current_correction.segment<3>(start);
          };
      bound_total_correction(6, options_.visual_max_velocity_step);
      bound_total_correction(9, options_.visual_max_gyro_bias_step);
      bound_total_correction(12, options_.visual_max_acceleration_bias_step);
      bound_total_correction(15, options_.visual_max_gravity_step);
    }
    else
    {
      step.segment<12>(6).setZero();
    }
    applyError(estimate, step);
    final_information = information;
    result.iterations = iteration + 1;
    if (step.segment<3>(3).norm() < options_.visual_convergence_translation &&
        step.segment<3>(0).norm() * 180.0 / kPi <
            options_.visual_convergence_rotation_deg)
    {
      result.converged = true;
      break;
    }
  }

  final_linearization = linearizer(statePose(estimate));
  result.landmarks = final_linearization.landmarks;
  result.residuals = final_linearization.residuals;
  result.rmse = final_linearization.rmse;
  result.mean_ncc = final_linearization.mean_ncc;
  const VisualMeasurementProjection final_projected_measurement =
      projectVisualMeasurement(final_linearization, visual_options);
  if (final_projected_measurement.valid)
  {
    final_information = prior_information;
    final_information.block<6, 6>(0, 0).noalias() +=
        final_projected_measurement.hessian;
    final_information.diagonal().array() += options_.visual_solver_damping;
  }
  result.observable_directions = final_projected_measurement.observable_directions;
  result.z_observable = final_projected_measurement.z_observable;
  result.z_projection = final_projected_measurement.z_projection;
  result.z_conditional_information_ratio =
      final_projected_measurement.z_conditional_information_ratio;
  const bool observability_valid = !observability_rejected &&
      final_projected_measurement.valid &&
      final_projected_measurement.observable_directions >=
          options_.visual_min_observable_directions;
  const bool two_mode_quality_valid =
      !two_mode_quality_rejected &&
      passesTwoModeVisualQuality(final_linearization,
                                 final_projected_measurement.observable_directions,
                                 options_);
  const Eigen::Isometry3d estimate_pose = statePose(estimate);
  const Eigen::Isometry3d correction = propagated_pose.inverse() * estimate_pose;
  const double translation_step = correction.translation().norm();
  const double rotation_step = rotationDegrees(correction.rotation());
  const bool measurement_valid = final_linearization.valid &&
      final_linearization.landmarks >= options_.visual_min_landmarks &&
      final_linearization.residuals >= options_.visual_min_residuals &&
      std::isfinite(final_linearization.rmse) &&
      final_linearization.rmse <= options_.visual_max_rmse &&
      std::isfinite(final_linearization.mean_ncc) &&
      final_linearization.mean_ncc >= options_.visual_min_mean_ncc;
  const bool motion_valid = translation_step <= options_.visual_max_translation_step &&
      rotation_step <= options_.visual_max_rotation_step_deg;
  const bool z_measurement_enabled = final_projected_measurement.z_fused &&
      (!automatic_z_requested ||
       (automatic_z_decided && automatic_z_enabled));
  const bool convergence_valid =
      !options_.visual_require_convergence || result.converged;
  if (measurement_valid && motion_valid && observability_valid &&
      two_mode_quality_valid && convergence_valid && result.iterations > 0)
  {
    state_ = estimate;
    const Eigen::LDLT<Matrix18d> information_solver(final_information);
    if (information_solver.info() == Eigen::Success)
    {
      state_.covariance = information_solver.solve(Matrix18d::Identity());
      state_.covariance = 0.5 *
          (state_.covariance + state_.covariance.transpose());
    }
    resetCovarianceAfterInjection(
        state_, stateDifference(state_, propagated_state));
    result.accepted = true;
    result.z_fused = z_measurement_enabled;
    result.reason = "visual_accepted";
  }
  else
  {
    state_ = propagated_state;
    if (!final_linearization.valid)
    {
      result.reason = final_linearization.reason;
    }
    else if (!observability_valid)
    {
      result.reason = "visual_observability_gate";
    }
    else if (!two_mode_quality_valid)
    {
      result.reason = "visual_two_mode_quality_gate";
    }
    else if (!measurement_valid)
    {
      result.reason = "visual_quality_gate";
    }
    else if (!convergence_valid)
    {
      result.reason = "visual_not_converged";
    }
    else
    {
      result.reason = "visual_motion_gate";
    }
  }
  pose_cache_ = statePose(state_);
  recordCurrentPose();
  result.pose = pose_cache_;
  result.correction = propagated_pose.inverse() * pose_cache_;
  result.z_correction = result.correction.translation().z();
  result.velocity_correction =
      state_.velocity - propagated_state.velocity;
  result.gyro_bias_correction =
      state_.gyro_bias - propagated_state.gyro_bias;
  result.acceleration_bias_correction =
      state_.acceleration_bias - propagated_state.acceleration_bias;
  result.gravity_correction =
      state_.gravity - propagated_state.gravity;
  result.velocity = state_.velocity;
  result.covariance.setZero();
  result.covariance.block<3, 3>(0, 0) = state_.covariance.block<3, 3>(0, 0);
  result.covariance.block<3, 3>(0, 3) = state_.covariance.block<3, 3>(0, 3);
  result.covariance.block<3, 3>(3, 0) = state_.covariance.block<3, 3>(3, 0);
  result.covariance.block<3, 3>(3, 3) = state_.covariance.block<3, 3>(3, 3);
  return result;
}

LidarOdometryResult LidarOdometry::processScan(const TimedPointVector &points,
                                               double scan_end_stamp)
{
  LidarOdometryResult result;
  result.stamp = scan_end_stamp;
  fillResultState(result);
  if (!std::isfinite(scan_end_stamp))
  {
    result.reject_reason = "invalid_scan_stamp";
    return result;
  }
  if (options_.imu_enabled && !imu_initialized_)
  {
    result.reject_reason = "waiting_for_imu_initialization";
    fillResultState(result);
    return result;
  }
  if (state_stamp_ > 0.0 && scan_end_stamp <= state_stamp_ + 1e-8)
  {
    result.reject_reason = "scan_precedes_filter_state";
    fillResultState(result);
    return result;
  }

  const State state_before_scan = state_;
  std::vector<ImuPose, Eigen::aligned_allocator<ImuPose>> trajectory;
  if (options_.imu_enabled)
  {
    if (!propagateTo(scan_end_stamp, trajectory))
    {
      result.reject_reason = "insufficient_imu_coverage";
      fillResultState(result);
      return result;
    }
    result.used_imu = true;
  }
  else
  {
    if (state_stamp_ <= 0.0)
    {
      state_stamp_ = scan_end_stamp;
    }
    predictWithoutImu(scan_end_stamp, trajectory);
  }

  const Eigen::Isometry3d pose_before_scan = last_scan_pose_;
  trajectory = deskewTrajectory(scan_end_stamp);
  result.deskewed_points = deskewScan(points, scan_end_stamp, trajectory);
  const PointVector scan = voxelDownsample(result.deskewed_points,
                                            options_.scan_voxel_size,
                                            options_.max_scan_points);
  result.scan_points = static_cast<int>(scan.size());
  if (scan.size() < static_cast<std::size_t>(options_.min_scan_points))
  {
    result.loss_limited = applyLidarLossProtection(state_before_scan, scan_end_stamp);
    result.reject_reason = "insufficient_scan_points";
    pose_cache_ = statePose(state_);
    result.relative_pose = pose_before_scan.inverse() * pose_cache_;
    previous_scan_pose_ = last_scan_pose_;
    last_scan_pose_ = pose_cache_;
    previous_scan_stamp_ = scan_end_stamp;
    fillResultState(result);
    pruneImu(scan_end_stamp);
    return result;
  }

  if (!map_initialized_)
  {
    map_initialized_ = true;
    ++accepted_scan_count_;
    consecutive_rejections_ = 0;
    lidar_loss_limited_ = false;
    lidar_loss_frozen_ = false;
    last_accepted_state_ = state_;
    insertMapPoints(scan, state_, false);
    have_last_map_insert_pose_ = true;
    last_map_insert_pose_ = statePose(state_);
    last_map_insert_stamp_ = scan_end_stamp;
    pose_cache_ = statePose(state_);
    last_scan_pose_ = pose_cache_;
    previous_scan_pose_ = pose_cache_;
    previous_scan_stamp_ = scan_end_stamp;
    result.accepted = true;
    result.converged = true;
    result.map_updated = true;
    result.map_keyframe_selected = true;
    result.map_update_reason = "initialized";
    result.rmse = 0.0;
    result.inlier_ratio = 1.0;
    result.correspondences = static_cast<int>(scan.size());
    result.reject_reason = "initialized";
    result.relative_pose.setIdentity();
    fillResultState(result);
    pruneImu(scan_end_stamp);
    return result;
  }

  const State propagated_state = state_;
  const Matrix18d propagated_covariance = state_.covariance;
  Matrix18d regularized_covariance = propagated_covariance;
  regularized_covariance.diagonal().array() += 1e-10;
  const Eigen::LDLT<Matrix18d> covariance_solver(regularized_covariance);
  if (covariance_solver.info() != Eigen::Success)
  {
    result.reject_reason = "invalid_predicted_covariance";
    state_ = propagated_state;
    if (!state_.covariance.allFinite())
    {
      state_.covariance = state_before_scan.covariance;
    }
    result.loss_limited = applyLidarLossProtection(state_before_scan, scan_end_stamp);
    pose_cache_ = statePose(state_);
    result.relative_pose = pose_before_scan.inverse() * pose_cache_;
    previous_scan_pose_ = last_scan_pose_;
    last_scan_pose_ = pose_cache_;
    previous_scan_stamp_ = scan_end_stamp;
    recordCurrentPose();
    fillResultState(result);
    pruneImu(scan_end_stamp);
    return result;
  }
  const Matrix18d prior_information = covariance_solver.solve(Matrix18d::Identity());
  State estimate = propagated_state;
  Matrix18d final_information = prior_information;
  Eigen::Matrix<double, 6, 6> final_measurement_hessian =
      Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> final_observability_projection =
      Eigen::Matrix<double, 6, 6>::Identity();
  double final_squared_error = std::numeric_limits<double>::infinity();
  double final_normalized_squared_error =
      std::numeric_limits<double>::infinity();
  int final_correspondences = 0;
  int final_observable_directions = 0;
  int final_observable_rotation_directions = 0;
  double final_measurement_condition =
      std::numeric_limits<double>::infinity();
  double final_rotation_measurement_condition =
      std::numeric_limits<double>::infinity();
  double final_yaw_observability = 0.0;
  int final_correspondence_sectors = 0;
  int final_point_knn_fallback_queries = 0;
  int final_point_knn_fallback_matches = 0;
  int final_innovation_rejections = 0;
  double final_robust_weight_sum = 0.0;
  bool final_wheel_forward_rejected = false;
  bool final_wheel_yaw_rate_rejected = false;
  double final_wheel_yaw_rate_effective_noise =
      options_.wheel_yaw_rate_noise;
  bool final_used_wheel_yaw_rate = false;
  double final_wheel_yaw_rate_residual = 0.0;
  WheelSample wheel_measurement;
  const bool have_wheel_measurement = wheelMeasurement(scan_end_stamp,
                                                        &wheel_measurement);
  const double measured_forward_speed = have_wheel_measurement
      ? wheel_measurement.forward_speed : 0.0;
  result.used_wheel = have_wheel_measurement;
  result.wheel_speed = measured_forward_speed;
  Eigen::Vector3d measured_angular_velocity = Eigen::Vector3d::Zero();
  const bool have_angular_velocity = angularVelocityMeasurement(
      scan_end_stamp, &measured_angular_velocity);
  const bool wheel_yaw_rate_requested = have_wheel_measurement &&
      std::abs(options_.wheel_yaw_rate_scale) > kSmall &&
      std::isfinite(wheel_measurement.differential_speed);
  const double corrected_differential_speed = wheel_yaw_rate_requested
      ? wheel_measurement.differential_speed -
          options_.wheel_differential_forward_leakage *
              measured_forward_speed
      : 0.0;
  const double measured_wheel_yaw_rate = wheel_yaw_rate_requested
      ? options_.wheel_yaw_rate_scale * corrected_differential_speed
      : 0.0;
  const bool differential_consistent = wheel_yaw_rate_requested &&
      (options_.wheel_differential_max_disagreement <= 0.0 ||
       (std::isfinite(wheel_measurement.differential_disagreement) &&
        wheel_measurement.differential_disagreement <=
            options_.wheel_differential_max_disagreement));
  const double wheel_yaw_rate_prefit_residual =
      wheel_yaw_rate_requested && have_angular_velocity
      ? measured_angular_velocity.z() - propagated_state.gyro_bias.z() -
          measured_wheel_yaw_rate
      : 0.0;
  const bool wheel_imu_yaw_rate_consistent =
      wheel_yaw_rate_requested && have_angular_velocity &&
      (options_.wheel_yaw_rate_max_imu_difference <= 0.0 ||
       std::abs(wheel_yaw_rate_prefit_residual) <=
           options_.wheel_yaw_rate_max_imu_difference);
  const bool wheel_yaw_rate_input_valid = wheel_yaw_rate_requested &&
      have_angular_velocity && differential_consistent &&
      wheel_imu_yaw_rate_consistent &&
      std::abs(measured_forward_speed) >=
          options_.wheel_yaw_rate_min_speed &&
      std::isfinite(measured_wheel_yaw_rate) &&
      std::abs(measured_wheel_yaw_rate) <=
          options_.wheel_yaw_rate_max_abs;
  const bool robust_wheel_yaw_bias_enabled =
      options_.wheel_yaw_bias_window_sec > 0.0;
  const bool low_curvature_bias_sample = wheel_yaw_rate_input_valid &&
      (options_.wheel_yaw_bias_max_abs_rate <= 0.0 ||
       std::abs(measured_wheel_yaw_rate) <=
           options_.wheel_yaw_bias_max_abs_rate);
  if (robust_wheel_yaw_bias_enabled)
  {
    const double oldest_bias_stamp =
        scan_end_stamp - options_.wheel_yaw_bias_window_sec;
    while (!wheel_yaw_bias_history_.empty() &&
           wheel_yaw_bias_history_.front().stamp < oldest_bias_stamp)
    {
      wheel_yaw_bias_history_.pop_front();
    }
    if (low_curvature_bias_sample &&
        (wheel_yaw_bias_history_.empty() ||
         scan_end_stamp > wheel_yaw_bias_history_.back().stamp + kSmall))
    {
      wheel_yaw_bias_history_.push_back(WheelYawBiasSample{
          scan_end_stamp,
          measured_angular_velocity.z() - measured_wheel_yaw_rate});
    }
  }
  double robust_wheel_yaw_bias = 0.0;
  double robust_wheel_yaw_bias_raw = 0.0;
  double robust_wheel_yaw_bias_mad =
      std::numeric_limits<double>::infinity();
  double robust_wheel_yaw_bias_noise = options_.wheel_yaw_rate_noise;
  bool robust_wheel_yaw_bias_valid = false;
  if (robust_wheel_yaw_bias_enabled &&
      wheel_yaw_bias_history_.size() >=
          static_cast<std::size_t>(options_.wheel_yaw_bias_min_samples))
  {
    std::vector<double> bias_samples;
    bias_samples.reserve(wheel_yaw_bias_history_.size());
    for (const WheelYawBiasSample &sample : wheel_yaw_bias_history_)
    {
      bias_samples.push_back(sample.bias);
    }
    robust_wheel_yaw_bias_raw = median(bias_samples);
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(bias_samples.size());
    for (const double bias : bias_samples)
    {
      absolute_deviations.push_back(
          std::abs(bias - robust_wheel_yaw_bias_raw));
    }
    robust_wheel_yaw_bias_mad = median(absolute_deviations);
    const double independent_sample_noise = std::sqrt(
        options_.wheel_yaw_rate_noise * options_.wheel_yaw_rate_noise +
        options_.gyro_noise * options_.gyro_noise) /
        std::sqrt(static_cast<double>(bias_samples.size()));
    robust_wheel_yaw_bias_noise = std::max(
        options_.wheel_yaw_bias_noise_floor,
        std::max(independent_sample_noise,
                 1.4826 * robust_wheel_yaw_bias_mad));
    robust_wheel_yaw_bias_valid =
        std::isfinite(robust_wheel_yaw_bias_raw) &&
        std::isfinite(robust_wheel_yaw_bias_mad) &&
        (options_.wheel_yaw_bias_max_mad <= 0.0 ||
         robust_wheel_yaw_bias_mad <= options_.wheel_yaw_bias_max_mad);
    if (robust_wheel_yaw_bias_valid && low_curvature_bias_sample &&
        options_.wheel_yaw_bias_calibrate_offset &&
        !wheel_yaw_bias_offset_initialized_)
    {
      wheel_yaw_bias_offset_ =
          robust_wheel_yaw_bias_raw - propagated_state.gyro_bias.z();
      wheel_yaw_bias_offset_initialized_ =
          std::isfinite(wheel_yaw_bias_offset_);
      if (!wheel_yaw_bias_offset_initialized_)
      {
        wheel_yaw_bias_offset_ = 0.0;
      }
    }
    robust_wheel_yaw_bias = robust_wheel_yaw_bias_raw -
        (wheel_yaw_bias_offset_initialized_ ? wheel_yaw_bias_offset_ : 0.0);
  }
  result.wheel_yaw_rate = measured_wheel_yaw_rate;
  result.imu_yaw_rate = have_angular_velocity
      ? measured_angular_velocity.z() - propagated_state.gyro_bias.z()
      : 0.0;
  result.wheel_yaw_bias_window_samples =
      static_cast<int>(wheel_yaw_bias_history_.size());
  result.wheel_yaw_bias_observation = robust_wheel_yaw_bias;
  result.wheel_yaw_bias_raw_observation = robust_wheel_yaw_bias_raw;
  result.wheel_yaw_bias_offset = wheel_yaw_bias_offset_;
  result.wheel_yaw_bias_offset_calibrated =
      wheel_yaw_bias_offset_initialized_;
  result.wheel_yaw_bias_mad = robust_wheel_yaw_bias_mad;
  const bool stable_wheel_motion_for_acceleration_bias =
      robust_wheel_yaw_bias_enabled && low_curvature_bias_sample &&
      robust_wheel_yaw_bias_valid;
  const bool acceleration_bias_update_allowed =
      options_.lidar_update_acceleration_bias &&
      (!options_.lidar_acceleration_bias_require_stable_wheel_motion ||
       stable_wheel_motion_for_acceleration_bias);
  result.acceleration_bias_update_allowed =
      acceleration_bias_update_allowed;
  final_wheel_yaw_rate_residual = wheel_yaw_rate_prefit_residual;
  const bool wheel_yaw_rate_should_be_tested =
      wheel_yaw_rate_requested &&
      std::abs(measured_forward_speed) >=
          options_.wheel_yaw_rate_min_speed;
  final_wheel_yaw_rate_rejected =
      wheel_yaw_rate_should_be_tested && !wheel_yaw_rate_input_valid;
  Eigen::Vector3d world_up = Eigen::Vector3d::UnitZ();
  if (propagated_state.gravity.norm() > kSmall)
  {
    world_up = -propagated_state.gravity.normalized();
  }
  const Eigen::Vector3d up_in_propagated_body =
      propagated_state.rotation.transpose() * world_up;
  if (options_.lidar_yaw_correction_window_sec > 0.0)
  {
    const double oldest_stamp =
        scan_end_stamp - options_.lidar_yaw_correction_window_sec;
    while (!lidar_yaw_correction_history_.empty() &&
           lidar_yaw_correction_history_.front().stamp < oldest_stamp)
    {
      lidar_yaw_correction_history_.pop_front();
    }
  }
  double historical_lidar_yaw_correction = 0.0;
  for (const YawCorrectionSample &sample : lidar_yaw_correction_history_)
  {
    historical_lidar_yaw_correction += sample.correction;
  }
  const bool cumulative_yaw_limit_enabled = result.used_imu &&
      options_.limit_cumulative_lidar_yaw_correction &&
      options_.lidar_yaw_correction_window_sec > 0.0 &&
      options_.max_cumulative_lidar_yaw_correction_deg > 0.0;
  const double cumulative_yaw_limit =
      options_.max_cumulative_lidar_yaw_correction_deg * kPi / 180.0;
  bool lidar_yaw_correction_limited = false;
  bool lidar_yaw_information_guard_active = false;
  result.lidar_yaw_correction_limit_deg =
      options_.max_cumulative_lidar_yaw_correction_deg;
  int convergence_confirmations = 0;
  int applied_update_iterations = 0;
  bool final_linearization_valid = false;

  // Reserve one final relinearization after the configured number of update
  // steps.  The old path applied its last increment and then validated stale
  // residuals from the previous pose, which was particularly unsafe during a
  // turn when nearest planes can change between iterations.
  for (int iteration = 0; iteration <= options_.max_iterations; ++iteration)
  {
    Matrix18d lidar_measurement_hessian = Matrix18d::Zero();
    Vector18d lidar_measurement_gradient = Vector18d::Zero();
    Matrix18d wheel_measurement_hessian = Matrix18d::Zero();
    Vector18d wheel_measurement_gradient = Vector18d::Zero();
    double squared_error = 0.0;
    double normalized_squared_error = 0.0;
    int correspondences = 0;
    int point_knn_fallback_queries = 0;
    int point_knn_fallback_matches = 0;
    int innovation_rejections = 0;
    double robust_weight_sum = 0.0;
    bool wheel_forward_rejected = false;
    constexpr int kAzimuthSectorCount = 12;
    std::array<bool, kAzimuthSectorCount> correspondence_sectors{};

    struct PointLinearization
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      Eigen::Vector3d world_point = Eigen::Vector3d::Zero();
      PlaneMatch match;
      Eigen::Matrix<double, 1, 18> jacobian =
          Eigen::Matrix<double, 1, 18>::Zero();
      double residual = 0.0;
      double measurement_variance = 0.0;
      double robust_weight = 0.0;
      bool matched = false;
      bool fallback_matched = false;
      bool accepted = false;
      bool innovation_rejected = false;
      int azimuth_sector = -1;
    };
    using PointLinearizationVector = std::vector<
        PointLinearization, Eigen::aligned_allocator<PointLinearization>>;
    PointLinearizationVector point_linearizations(scan.size());
    const int scan_point_count = static_cast<int>(scan.size());

    // FAST-LIVO2 distributes the read-only point-to-plane search over a small
    // worker set. Map mutation remains outside this region, so concurrent
    // unordered_map lookups never overlap an insert, erase, or plane refit.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options_.registration_threads) \
    if(options_.registration_threads > 1)
#endif
    for (int point_index = 0; point_index < scan_point_count; ++point_index)
    {
      PointLinearization &point =
          point_linearizations[static_cast<std::size_t>(point_index)];
      const Eigen::Vector3d &body_point =
          scan[static_cast<std::size_t>(point_index)];
      point.world_point =
          estimate.rotation * body_point + estimate.position;
      point.matched = findLocalPlane(point.world_point, point.match);
    }

    // Select KNN fallback work in scan order before dispatching it. This keeps
    // the configured budget and resulting associations deterministic even
    // when worker completion order changes.
    std::vector<int> fallback_indices;
    if (!options_.use_point_knn_plane && options_.point_knn_fallback)
    {
      const int fallback_budget = options_.point_knn_fallback_max_queries;
      fallback_indices.reserve(fallback_budget > 0
          ? static_cast<std::size_t>(std::min(fallback_budget, scan_point_count))
          : scan.size());
      for (int point_index = 0; point_index < scan_point_count; ++point_index)
      {
        if (point_linearizations[static_cast<std::size_t>(point_index)].matched)
        {
          continue;
        }
        if (fallback_budget > 0 &&
            static_cast<int>(fallback_indices.size()) >= fallback_budget)
        {
          break;
        }
        fallback_indices.push_back(point_index);
      }
    }
    point_knn_fallback_queries = static_cast<int>(fallback_indices.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options_.registration_threads) \
    if(options_.registration_threads > 1)
#endif
    for (int fallback_index = 0;
         fallback_index < static_cast<int>(fallback_indices.size());
         ++fallback_index)
    {
      PointLinearization &point = point_linearizations[static_cast<std::size_t>(
          fallback_indices[static_cast<std::size_t>(fallback_index)])];
      point.matched = findPointKnnPlane(point.world_point, point.match);
      point.fallback_matched = point.matched;
    }

    // Measurement preparation is independent per point. The final normal
    // equation is accumulated serially below in scan order to avoid floating
    // point reduction jitter in convergence and map-write decisions.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options_.registration_threads) \
    if(options_.registration_threads > 1)
#endif
    for (int point_index = 0; point_index < scan_point_count; ++point_index)
    {
      PointLinearization &point =
          point_linearizations[static_cast<std::size_t>(point_index)];
      if (!point.matched)
      {
        continue;
      }
      const Eigen::Vector3d &body_point =
          scan[static_cast<std::size_t>(point_index)];
      point.residual = point.match.normal.dot(
          point.world_point - point.match.center);
      if (std::abs(point.residual) > options_.max_plane_distance)
      {
        continue;
      }
      point.jacobian.block<1, 3>(0, 0) =
          -point.match.normal.transpose() * estimate.rotation * skew(body_point);
      point.jacobian.block<1, 3>(0, 3) = point.match.normal.transpose();
      const Eigen::Vector3d lidar_ray =
          body_point - options_.lidar_origin_in_body;
      const double range = lidar_ray.norm();
      double point_variance =
          options_.lidar_range_noise * options_.lidar_range_noise +
          std::pow(options_.lidar_beam_noise * range, 2.0);
      if (options_.use_directional_lidar_covariance && range > kSmall)
      {
        const Eigen::Vector3d beam_direction = lidar_ray / range;
        const Eigen::Vector3d normal_in_body =
            estimate.rotation.transpose() * point.match.normal;
        const double radial_projection =
            normal_in_body.dot(beam_direction);
        const double radial_variance =
            options_.lidar_range_noise * options_.lidar_range_noise;
        const double tangential_variance =
            std::pow(options_.lidar_beam_noise * range, 2.0);
        point_variance = tangential_variance +
            (radial_variance - tangential_variance) *
                radial_projection * radial_projection;
      }
      point.measurement_variance = std::max(1e-8,
          options_.lidar_measurement_noise *
              options_.lidar_measurement_noise +
          std::max(0.0, point_variance) + point.match.variance);
      if (options_.lidar_innovation_gate > 0.0)
      {
        const double predicted_variance = std::max(0.0,
            (point.jacobian * propagated_covariance *
             point.jacobian.transpose())(0, 0));
        const double innovation_variance =
            point.measurement_variance + predicted_variance;
        const double normalized_innovation =
            point.residual * point.residual /
                std::max(1e-8, innovation_variance);
        if (!std::isfinite(normalized_innovation) ||
            normalized_innovation > options_.lidar_innovation_gate)
        {
          point.innovation_rejected = true;
          continue;
        }
      }
      point.robust_weight =
          options_.lidar_normalized_huber_delta > 0.0
          ? huberWeight(point.residual /
                            std::sqrt(point.measurement_variance),
                        options_.lidar_normalized_huber_delta)
          : huberWeight(point.residual, options_.huber_delta);
      point.accepted = true;
      const double azimuth = std::atan2(body_point.y(), body_point.x());
      const double normalized_azimuth = std::max(0.0, std::min(
          1.0 - 1e-12, (azimuth + kPi) / (2.0 * kPi)));
      point.azimuth_sector = std::min(kAzimuthSectorCount - 1,
          static_cast<int>(std::floor(
              normalized_azimuth * kAzimuthSectorCount)));
    }

    for (const PointLinearization &point : point_linearizations)
    {
      if (point.fallback_matched)
      {
        ++point_knn_fallback_matches;
      }
      if (point.innovation_rejected)
      {
        ++innovation_rejections;
      }
      if (!point.accepted)
      {
        continue;
      }
      const double robust_information =
          point.robust_weight / point.measurement_variance;
      lidar_measurement_hessian.noalias() +=
          robust_information * point.jacobian.transpose() * point.jacobian;
      lidar_measurement_gradient.noalias() +=
          robust_information * point.jacobian.transpose() * point.residual;
      squared_error += point.residual * point.residual;
      normalized_squared_error += point.residual * point.residual /
          point.measurement_variance;
      robust_weight_sum += point.robust_weight;
      ++correspondences;
      correspondence_sectors[static_cast<std::size_t>(
          point.azimuth_sector)] = true;
    }

    if (have_wheel_measurement)
    {
      // The Ranger measurement is the velocity of the odometer reference
      // point in the body frame.  Applying the non-holonomic constraint in
      // world-Z (the old implementation) couples a small pitch error into a
      // persistent height drift.  Keep all three components in body frame and
      // compensate the configured IMU-to-odometer lever arm when an IMU rate
      // at the scan stamp is available.
      const Eigen::Matrix3d body_from_world = estimate.rotation.transpose();
      const Eigen::Vector3d body_velocity = body_from_world * estimate.velocity;
      Eigen::Vector3d wheel_velocity = body_velocity;
      Eigen::Matrix<double, 3, 18> jacobian =
          Eigen::Matrix<double, 3, 18>::Zero();
      jacobian.block<3, 3>(0, 0) = skew(body_velocity);
      jacobian.block<3, 3>(0, 6) = body_from_world;
      if (options_.wheel_compensate_angular_velocity &&
          options_.wheel_lever_arm.squaredNorm() > kSmall)
      {
        Eigen::Vector3d measured_angular_velocity;
        if (angularVelocityMeasurement(scan_end_stamp, &measured_angular_velocity))
        {
          const Eigen::Vector3d body_angular_velocity =
              measured_angular_velocity - estimate.gyro_bias;
          wheel_velocity += body_angular_velocity.cross(options_.wheel_lever_arm);
          // d((omega - bg) x r) / d(bg) = skew(r).
          jacobian.block<3, 3>(0, 9) = skew(options_.wheel_lever_arm);
        }
      }
      const Eigen::Vector3d residual = wheel_velocity -
          Eigen::Vector3d(measured_forward_speed, 0.0, 0.0);
      const Eigen::Vector3d sigma(options_.wheel_forward_noise,
                                  options_.wheel_lateral_noise,
                                  options_.wheel_vertical_noise);
      for (int axis = 0; axis < 3; ++axis)
      {
        const Eigen::Matrix<double, 1, 18> row = jacobian.row(axis);
        const double measurement_variance = sigma(axis) * sigma(axis);
        if (axis == 0 && options_.wheel_forward_innovation_gate > 0.0)
        {
          const double predicted_variance = std::max(0.0,
              (row * propagated_covariance * row.transpose())(0, 0));
          const double normalized_innovation = residual(axis) * residual(axis) /
              std::max(1e-8, measurement_variance + predicted_variance);
          if (!std::isfinite(normalized_innovation) ||
              normalized_innovation >
                  options_.wheel_forward_innovation_gate)
          {
            wheel_forward_rejected = true;
            continue;
          }
        }
        const double normalized = residual(axis) / sigma(axis);
        const double information =
            huberWeight(normalized, options_.wheel_huber_delta) /
            measurement_variance;
        wheel_measurement_hessian.noalias() +=
            information * row.transpose() * row;
        wheel_measurement_gradient.noalias() +=
            information * row.transpose() * residual(axis);
      }
      result.wheel_velocity_residual = residual.norm();

      if (wheel_yaw_rate_input_valid)
      {
        const bool robust_bias_factor_available =
            robust_wheel_yaw_bias_enabled && low_curvature_bias_sample &&
            robust_wheel_yaw_bias_valid;
        const bool direct_bias_factor_available =
            !robust_wheel_yaw_bias_enabled;
        if (robust_bias_factor_available || direct_bias_factor_available)
        {
          Eigen::Matrix<double, 1, 18> yaw_jacobian =
              Eigen::Matrix<double, 1, 18>::Zero();
          // h(x) = gyro_z - wheel_yaw_rate - gyro_bias_z. In robust-window
          // mode the first two terms are replaced by their low-curvature
          // median, while the state Jacobian remains unchanged.
          yaw_jacobian(0, 11) = -1.0;
          const double yaw_residual = robust_bias_factor_available
              ? robust_wheel_yaw_bias - estimate.gyro_bias.z()
              : measured_angular_velocity.z() - estimate.gyro_bias.z() -
                  measured_wheel_yaw_rate;
          const double wheel_scale_standard_deviation =
              options_.wheel_yaw_rate_relative_scale_uncertainty *
              std::abs(measured_wheel_yaw_rate);
          const double yaw_noise = robust_bias_factor_available
              ? robust_wheel_yaw_bias_noise
              : std::sqrt(options_.wheel_yaw_rate_noise *
                              options_.wheel_yaw_rate_noise +
                          options_.gyro_noise * options_.gyro_noise);
          const double yaw_variance = yaw_noise * yaw_noise +
              wheel_scale_standard_deviation *
                  wheel_scale_standard_deviation;
          final_wheel_yaw_rate_effective_noise = std::sqrt(yaw_variance);
          bool reject_yaw_rate = false;
          if (options_.wheel_yaw_rate_innovation_gate > 0.0)
          {
            const double predicted_variance = std::max(
                0.0, (yaw_jacobian * propagated_covariance *
                      yaw_jacobian.transpose())(0, 0));
            const double normalized_innovation =
                yaw_residual * yaw_residual /
                std::max(1e-8, yaw_variance + predicted_variance);
            reject_yaw_rate = !std::isfinite(normalized_innovation) ||
                normalized_innovation >
                    options_.wheel_yaw_rate_innovation_gate;
          }
          if (!reject_yaw_rate && options_.wheel_yaw_rate_fuse_gyro_bias)
          {
            const double normalized_yaw_residual =
                yaw_residual / std::sqrt(yaw_variance);
            const double yaw_information =
                huberWeight(normalized_yaw_residual,
                            options_.wheel_yaw_rate_huber_delta) /
                yaw_variance;
            wheel_measurement_hessian.noalias() +=
                yaw_information * yaw_jacobian.transpose() * yaw_jacobian;
            wheel_measurement_gradient.noalias() +=
                yaw_information * yaw_jacobian.transpose() * yaw_residual;
            final_used_wheel_yaw_rate = true;
          }
          final_wheel_yaw_rate_rejected = reject_yaw_rate;
        }
      }
    }

    final_correspondences = correspondences;
    final_correspondence_sectors = static_cast<int>(std::count(
        correspondence_sectors.begin(), correspondence_sectors.end(), true));
    final_point_knn_fallback_queries = point_knn_fallback_queries;
    final_point_knn_fallback_matches = point_knn_fallback_matches;
    final_innovation_rejections = innovation_rejections;
    final_robust_weight_sum = robust_weight_sum;
    final_wheel_forward_rejected = wheel_forward_rejected;
    if (correspondences < options_.min_correspondences)
    {
      break;
    }

    const Eigen::Matrix<double, 6, 6> pose_hessian =
        0.5 * (lidar_measurement_hessian.block<6, 6>(0, 0) +
               lidar_measurement_hessian.block<6, 6>(0, 0).transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
        observability_solver(pose_hessian);
    if (observability_solver.info() != Eigen::Success)
    {
      break;
    }
    const Eigen::Matrix<double, 6, 1> eigenvalues =
        observability_solver.eigenvalues().cwiseMax(0.0);
    const double largest_eigenvalue = std::max(kSmall, eigenvalues.maxCoeff());
    const double observable_threshold = largest_eigenvalue *
        options_.observability_eigen_ratio;
    Eigen::Matrix<double, 6, 6> observability_projection =
        Eigen::Matrix<double, 6, 6>::Zero();
    int observable_directions = 0;
    double smallest_observable_eigenvalue =
        std::numeric_limits<double>::infinity();
    for (int index = 0; index < eigenvalues.rows(); ++index)
    {
      if (eigenvalues(index) < observable_threshold)
      {
        continue;
      }
      const Eigen::Matrix<double, 6, 1> eigenvector =
          observability_solver.eigenvectors().col(index);
      observability_projection.noalias() += eigenvector * eigenvector.transpose();
      ++observable_directions;
      smallest_observable_eigenvalue = std::min(
          smallest_observable_eigenvalue, eigenvalues(index));
    }
    if (observable_directions == 0)
    {
      break;
    }

    // Translation can make the full 6-DoF Hessian look healthy while a
    // rotational mode remains weak. Marginalize translation and inspect the
    // conditional rotation information before allowing LiDAR to correct the
    // IMU attitude or write the scan into the persistent map.
    const Eigen::Matrix3d translation_hessian =
        0.5 * (pose_hessian.block<3, 3>(3, 3) +
               pose_hessian.block<3, 3>(3, 3).transpose());
    Eigen::Matrix3d translation_hessian_inverse = Eigen::Matrix3d::Zero();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
        translation_solver(translation_hessian);
    if (translation_solver.info() == Eigen::Success)
    {
      const Eigen::Vector3d translation_eigenvalues =
          translation_solver.eigenvalues().cwiseMax(0.0);
      const double translation_largest = std::max(
          kSmall, translation_eigenvalues.maxCoeff());
      const double inverse_threshold = std::max(
          kSmall, 1e-8 * translation_largest);
      for (int index = 0; index < 3; ++index)
      {
        if (translation_eigenvalues(index) <= inverse_threshold) continue;
        const Eigen::Vector3d direction =
            translation_solver.eigenvectors().col(index);
        translation_hessian_inverse.noalias() +=
            direction * direction.transpose() /
            translation_eigenvalues(index);
      }
    }
    Eigen::Matrix3d rotation_hessian =
        pose_hessian.block<3, 3>(0, 0) -
        pose_hessian.block<3, 3>(0, 3) *
            translation_hessian_inverse *
            pose_hessian.block<3, 3>(3, 0);
    rotation_hessian = 0.5 *
        (rotation_hessian + rotation_hessian.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
        rotation_solver(rotation_hessian);
    if (rotation_solver.info() != Eigen::Success)
    {
      break;
    }
    const Eigen::Vector3d rotation_eigenvalues =
        rotation_solver.eigenvalues().cwiseMax(0.0);
    const double largest_rotation_eigenvalue = std::max(
        kSmall, rotation_eigenvalues.maxCoeff());
    const double rotation_observable_threshold =
        largest_rotation_eigenvalue *
        options_.rotation_observability_eigen_ratio;
    Eigen::Matrix3d rotation_observability_projection =
        Eigen::Matrix3d::Zero();
    int observable_rotation_directions = 0;
    double smallest_observable_rotation_eigenvalue =
        std::numeric_limits<double>::infinity();
    for (int index = 0; index < 3; ++index)
    {
      if (rotation_eigenvalues(index) < rotation_observable_threshold)
      {
        continue;
      }
      const Eigen::Vector3d direction =
          rotation_solver.eigenvectors().col(index);
      rotation_observability_projection.noalias() +=
          direction * direction.transpose();
      ++observable_rotation_directions;
      smallest_observable_rotation_eigenvalue = std::min(
          smallest_observable_rotation_eigenvalue,
          rotation_eigenvalues(index));
    }
    const double yaw_observability = std::max(
        0.0, std::min(1.0, rotation_observability_projection(2, 2)));
    Eigen::Matrix3d rotation_information_filter = Eigen::Matrix3d::Identity();
    if (options_.project_lidar_information_to_observable_rotation)
    {
      const double weak_scale = options_.weak_rotation_information_scale;
      rotation_information_filter =
          weak_scale * Eigen::Matrix3d::Identity() +
          (1.0 - weak_scale) * rotation_observability_projection;
    }
    Eigen::Matrix<double, 6, 6> rotation_pose_filter =
        Eigen::Matrix<double, 6, 6>::Identity();
    rotation_pose_filter.block<3, 3>(0, 0) = rotation_information_filter;
    Eigen::Matrix<double, 6, 6> filtered_pose_hessian =
        rotation_pose_filter * pose_hessian * rotation_pose_filter;
    filtered_pose_hessian = 0.5 *
        (filtered_pose_hessian + filtered_pose_hessian.transpose());
    Eigen::Matrix<double, 6, 1> filtered_pose_gradient =
        rotation_pose_filter * lidar_measurement_gradient.head<6>();
    if (lidar_yaw_information_guard_active)
    {
      const Eigen::Vector3d up_in_estimate_body =
          estimate.rotation.transpose() * world_up;
      const Eigen::Matrix3d yaw_information_filter =
          Eigen::Matrix3d::Identity() -
          (1.0 - options_.limited_lidar_yaw_information_scale) *
              up_in_estimate_body * up_in_estimate_body.transpose();
      Eigen::Matrix<double, 6, 6> yaw_pose_filter =
          Eigen::Matrix<double, 6, 6>::Identity();
      yaw_pose_filter.block<3, 3>(0, 0) = yaw_information_filter;
      filtered_pose_hessian = yaw_pose_filter * filtered_pose_hessian *
          yaw_pose_filter;
      filtered_pose_hessian = 0.5 *
          (filtered_pose_hessian + filtered_pose_hessian.transpose());
      filtered_pose_gradient = yaw_pose_filter * filtered_pose_gradient;
    }

    Matrix18d measurement_hessian = wheel_measurement_hessian;
    Vector18d measurement_gradient = wheel_measurement_gradient;
    if (options_.project_lidar_information_to_observable_subspace)
    {
      measurement_hessian.block<6, 6>(0, 0).noalias() +=
          observability_projection * filtered_pose_hessian *
          observability_projection;
      measurement_gradient.head<6>().noalias() +=
          observability_projection *
          filtered_pose_gradient;
    }
    else
    {
      measurement_hessian.block<6, 6>(0, 0).noalias() +=
          filtered_pose_hessian;
      measurement_gradient.head<6>().noalias() +=
          filtered_pose_gradient;
    }
    const Vector18d displacement = stateDifference(estimate, propagated_state);
    Matrix18d information = prior_information + measurement_hessian;
    information.diagonal().array() += options_.solver_damping;
    const Vector18d gradient = prior_information * displacement + measurement_gradient;
    const Eigen::LDLT<Matrix18d> solver(information);
    if (solver.info() != Eigen::Success)
    {
      break;
    }
    Vector18d step = solver.solve(-gradient);
    if (!step.allFinite())
    {
      break;
    }
    // In the normal full-information path, retain the complete iterated ESKF
    // correction as FAST-LIVO2 does. The prior information already limits weak
    // scan directions; hard-zeroing them here caused systematic under-travel
    // through long turns and corridors. The optional projected-information
    // branch above remains available for controlled degeneracy experiments.
    if (options_.project_gyro_bias_update_to_observable_rotation)
    {
      // Rotation error and gyro bias both use body-frame coordinates. Reuse
      // the marginalized rotational filter so strong translation support
      // cannot manufacture a gyro-bias correction in a weak yaw mode.
      step.segment<3>(9) =
          rotation_information_filter *
          step.segment<3>(9);
    }
    if (lidar_yaw_information_guard_active)
    {
      const Eigen::Vector3d up_in_estimate_body =
          estimate.rotation.transpose() * world_up;
      const Eigen::Matrix3d yaw_information_filter =
          Eigen::Matrix3d::Identity() -
          (1.0 - options_.limited_lidar_yaw_information_scale) *
              up_in_estimate_body * up_in_estimate_body.transpose();
      step.segment<3>(9) =
          yaw_information_filter * step.segment<3>(9);
    }
    const auto clamp_segment = [&step](int start, double maximum)
    {
      const double norm = step.segment<3>(start).norm();
      if (norm > maximum && norm > kSmall)
      {
        step.segment<3>(start) *= maximum / norm;
      }
    };
    clamp_segment(0, options_.max_iteration_rotation_deg * kPi / 180.0);
    clamp_segment(3, options_.max_iteration_translation);
    // Hidden-state limits are scan limits, not iteration limits. Applying the
    // full allowance in every IEKF iteration let a five-iteration scan change
    // gyro bias and velocity by five times the configured bound.
    const Vector18d accumulated_update =
        stateDifference(estimate, propagated_state);
    if (cumulative_yaw_limit_enabled)
    {
      const Eigen::Matrix3d proposed_rotation =
          estimate.rotation * expSO3(step.segment<3>(0));
      Eigen::Vector3d proposed_correction = logSO3(
          propagated_state.rotation.transpose() * proposed_rotation);
      const double proposed_yaw_correction =
          proposed_correction.dot(up_in_propagated_body);
      const double minimum_yaw_correction =
          -cumulative_yaw_limit - historical_lidar_yaw_correction;
      const double maximum_yaw_correction =
          cumulative_yaw_limit - historical_lidar_yaw_correction;
      const double limited_yaw_correction = std::max(
          minimum_yaw_correction,
          std::min(maximum_yaw_correction, proposed_yaw_correction));
      if (std::abs(limited_yaw_correction - proposed_yaw_correction) >
          1e-10)
      {
        proposed_correction +=
            (limited_yaw_correction - proposed_yaw_correction) *
            up_in_propagated_body;
        const Eigen::Matrix3d limited_rotation =
            propagated_state.rotation * expSO3(proposed_correction);
        step.segment<3>(0) = logSO3(
            estimate.rotation.transpose() * limited_rotation);
        lidar_yaw_correction_limited = true;
        lidar_yaw_information_guard_active = true;
      }
    }
    const auto clamp_total_segment =
        [&step, &accumulated_update](int start, double maximum)
    {
      Eigen::Vector3d total = accumulated_update.segment<3>(start) +
          step.segment<3>(start);
      const double norm = total.norm();
      if (norm > maximum && norm > kSmall)
      {
        total *= maximum / norm;
        step.segment<3>(start) =
            total - accumulated_update.segment<3>(start);
      }
    };
    clamp_total_segment(6, options_.max_lidar_velocity_step);
    clamp_total_segment(9, options_.max_lidar_gyro_bias_step);
    if (acceleration_bias_update_allowed)
    {
      clamp_total_segment(12, options_.max_lidar_acceleration_bias_step);
    }
    else
    {
      step.segment<3>(12).setZero();
    }
    if (options_.lidar_update_gravity)
    {
      clamp_total_segment(15, options_.max_lidar_gravity_step);
    }
    else
    {
      step.segment<3>(15).setZero();
    }
    final_information = information;
    final_measurement_hessian = pose_hessian;
    final_observability_projection = observability_projection;
    final_squared_error = squared_error;
    final_normalized_squared_error = normalized_squared_error;
    final_observable_directions = observable_directions;
    final_observable_rotation_directions =
        observable_rotation_directions;
    final_measurement_condition = largest_eigenvalue /
        std::max(kSmall, smallest_observable_eigenvalue);
    final_rotation_measurement_condition =
        largest_rotation_eigenvalue /
        std::max(kSmall, smallest_observable_rotation_eigenvalue);
    final_yaw_observability = yaw_observability;
    result.iterations = iteration + 1;
    const bool small_increment =
        step.segment<3>(3).norm() < options_.convergence_translation &&
        step.segment<3>(0).norm() * 180.0 / kPi <
            options_.convergence_rotation_deg;
    convergence_confirmations = small_increment
        ? convergence_confirmations + 1 : 0;
    result.convergence_confirmations = convergence_confirmations;
    if (convergence_confirmations >=
        options_.convergence_confirmation_iterations)
    {
      result.converged = true;
      final_linearization_valid = true;
      break;
    }
    if (applied_update_iterations >= options_.max_iterations)
    {
      // Keep the estimate at this already-evaluated state.  A non-converged
      // scan can still be accepted by a permissive profile, but it cannot use
      // residuals from an unapplied linearization step.
      final_linearization_valid = true;
      break;
    }
    applyError(estimate, step);
    ++applied_update_iterations;
  }

  result.final_linearization_valid = final_linearization_valid;
  result.correspondences = final_correspondences;
  result.correspondence_azimuth_sectors = final_correspondence_sectors;
  result.point_knn_fallback_queries = final_point_knn_fallback_queries;
  result.point_knn_fallback_matches = final_point_knn_fallback_matches;
  result.innovation_rejections = final_innovation_rejections;
  result.mean_robust_weight = final_correspondences > 0
      ? final_robust_weight_sum /
          static_cast<double>(final_correspondences)
      : 0.0;
  result.wheel_forward_rejected = final_wheel_forward_rejected;
  result.used_wheel_yaw_rate = final_used_wheel_yaw_rate;
  result.wheel_yaw_rate_rejected = final_wheel_yaw_rate_rejected;
  result.wheel_yaw_rate_effective_noise =
      final_wheel_yaw_rate_effective_noise;
  result.wheel_yaw_rate_residual = final_wheel_yaw_rate_residual;
  result.inlier_ratio = static_cast<double>(final_correspondences) /
      static_cast<double>(std::max<std::size_t>(1U, scan.size()));
  result.rmse = final_correspondences > 0 && std::isfinite(final_squared_error)
      ? std::sqrt(final_squared_error / static_cast<double>(final_correspondences))
      : std::numeric_limits<double>::infinity();
  result.mean_normalized_residual = final_correspondences > 0 &&
      std::isfinite(final_normalized_squared_error)
      ? final_normalized_squared_error /
          static_cast<double>(final_correspondences)
      : std::numeric_limits<double>::infinity();
  result.observable_directions = final_observable_directions;
  result.observable_rotation_directions =
      final_observable_rotation_directions;
  result.measurement_condition = final_measurement_condition;
  result.rotation_measurement_condition =
      final_rotation_measurement_condition;
  result.yaw_observability = final_yaw_observability;
  if (final_correspondences >= options_.min_correspondences)
  {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigen_solver(
        final_measurement_hessian);
    if (eigen_solver.info() == Eigen::Success)
    {
      const double largest = std::max(kSmall, eigen_solver.eigenvalues().maxCoeff());
      result.degenerate = eigen_solver.eigenvalues().minCoeff() <
          largest * options_.degeneracy_eigen_ratio ||
          result.observable_directions < 6;
    }
  }

  const Eigen::Isometry3d estimated_pose = statePose(estimate);
  const Eigen::Isometry3d propagated_pose = statePose(propagated_state);
  const Eigen::Isometry3d relative_motion = pose_before_scan.inverse() * estimated_pose;
  const Eigen::Isometry3d propagated_relative_motion =
      pose_before_scan.inverse() * propagated_pose;
  const Eigen::Isometry3d lidar_correction = propagated_pose.inverse() * estimated_pose;
  const Eigen::Vector3d lidar_correction_vector =
      logSO3(lidar_correction.rotation());
  const double lidar_yaw_correction =
      lidar_correction_vector.dot(up_in_propagated_body);
  result.lidar_yaw_correction_deg =
      lidar_yaw_correction * 180.0 / kPi;
  const double cumulative_lidar_yaw_correction =
      historical_lidar_yaw_correction + lidar_yaw_correction;
  result.cumulative_lidar_yaw_correction_deg =
      cumulative_lidar_yaw_correction * 180.0 / kPi;
  result.lidar_yaw_correction_limited =
      lidar_yaw_correction_limited;
  result.lidar_yaw_information_scale =
      lidar_yaw_information_guard_active
      ? options_.limited_lidar_yaw_information_scale : 1.0;
  const bool cumulative_yaw_correction_valid =
      !result.used_imu || options_.lidar_yaw_correction_window_sec <= 0.0 ||
      options_.max_cumulative_lidar_yaw_correction_deg <= 0.0 ||
      std::abs(result.cumulative_lidar_yaw_correction_deg) <=
          options_.max_cumulative_lidar_yaw_correction_deg + 1e-6;
  bool rotation_correction_nis_valid = true;
  if (result.used_imu && options_.lidar_rotation_correction_nis_gate > 0.0)
  {
    const Eigen::Vector3d rotation_correction =
        logSO3(lidar_correction.rotation());
    const double angular_floor =
        options_.lidar_rotation_correction_std_floor_deg * kPi / 180.0;
    Eigen::Matrix3d rotation_covariance = 0.5 *
        (propagated_covariance.block<3, 3>(0, 0) +
         propagated_covariance.block<3, 3>(0, 0).transpose());
    rotation_covariance.diagonal().array() += angular_floor * angular_floor;
    const Eigen::LDLT<Eigen::Matrix3d> rotation_covariance_solver(
        rotation_covariance);
    if (rotation_covariance_solver.info() == Eigen::Success)
    {
      const Eigen::Vector3d whitened_correction =
          rotation_covariance_solver.solve(rotation_correction);
      result.lidar_rotation_correction_nis =
          rotation_correction.dot(whitened_correction);
      rotation_correction_nis_valid =
          std::isfinite(result.lidar_rotation_correction_nis) &&
          result.lidar_rotation_correction_nis <=
              options_.lidar_rotation_correction_nis_gate;
    }
    else
    {
      rotation_correction_nis_valid = false;
    }
  }
  const double scan_dt = previous_scan_stamp_ > 0.0
      ? std::max(0.0, scan_end_stamp - previous_scan_stamp_) : 0.0;
  const double translation_motion_gate = std::max(
      options_.max_translation_per_scan, options_.max_translation_speed * scan_dt);
  const double expected_rotation_deg =
      rotationDegrees(propagated_relative_motion.rotation());
  double rotation_motion_gate = std::max(
      options_.max_rotation_per_scan_deg, options_.max_rotation_speed_deg * scan_dt);
  const bool turn_aware_interval_valid =
      options_.turn_aware_motion_gate_enabled && result.used_imu &&
      scan_dt > 0.0 && options_.turn_aware_max_scan_dt > 0.0 &&
      scan_dt <= options_.turn_aware_max_scan_dt;
  if (turn_aware_interval_valid)
  {
    const double propagated_gate = std::min(
        options_.turn_aware_max_rotation_deg,
        expected_rotation_deg + options_.turn_aware_rotation_margin_deg);
    rotation_motion_gate = std::max(rotation_motion_gate, propagated_gate);
  }
  result.expected_rotation_deg = expected_rotation_deg;
  result.rotation_motion_gate_deg = rotation_motion_gate;
  result.turn_aware_gate_active = turn_aware_interval_valid &&
      rotation_motion_gate >
          std::max(options_.max_rotation_per_scan_deg,
                   options_.max_rotation_speed_deg * scan_dt) + 1e-6;
  const bool ratio_support = final_correspondences >= options_.min_correspondences &&
      result.inlier_ratio >= options_.min_inlier_ratio;
  const bool strong_support = options_.strong_support_min_correspondences > 0 &&
      options_.strong_support_max_rmse > 0.0 &&
      final_correspondences >= options_.strong_support_min_correspondences &&
      final_correspondence_sectors >= options_.strong_support_min_azimuth_sectors &&
      std::isfinite(result.rmse) && result.rmse <= options_.strong_support_max_rmse;
  result.strong_support = strong_support;
  const bool recovery_mode = options_.recovery_after_rejections > 0 &&
      consecutive_rejections_ >= options_.recovery_after_rejections;
  result.recovery_mode = recovery_mode;
  const double lidar_correction_translation_gate = recovery_mode && strong_support &&
      options_.recovery_max_lidar_correction_translation > 0.0
      ? std::max(options_.max_lidar_correction_translation,
                 options_.recovery_max_lidar_correction_translation)
      : options_.max_lidar_correction_translation;
  double lidar_correction_rotation_gate = recovery_mode && strong_support &&
      options_.recovery_max_lidar_correction_rotation_deg > 0.0
      ? std::max(options_.max_lidar_correction_rotation_deg,
                 options_.recovery_max_lidar_correction_rotation_deg)
      : options_.max_lidar_correction_rotation_deg;
  const bool trusted_wheel_turn = turn_aware_interval_valid &&
      wheel_yaw_rate_input_valid && !result.wheel_yaw_rate_rejected &&
      std::abs(result.wheel_yaw_rate) >=
          options_.turn_aware_min_yaw_rate &&
      std::abs(result.imu_yaw_rate - result.wheel_yaw_rate) <=
          options_.turn_aware_wheel_imu_max_yaw_rate_difference;
  if (trusted_wheel_turn &&
      options_.turn_aware_lidar_correction_rotation_deg > 0.0)
  {
    lidar_correction_rotation_gate = std::max(
        lidar_correction_rotation_gate,
        options_.turn_aware_lidar_correction_rotation_deg);
  }
  result.lidar_correction_rotation_gate_deg =
      lidar_correction_rotation_gate;
  if (final_correspondences < options_.min_correspondences)
  {
    result.reject_reason = "too_few_plane_correspondences";
  }
  else if (!result.final_linearization_valid)
  {
    result.reject_reason = "invalid_final_linearization";
  }
  else if (options_.require_convergence_for_acceptance && !result.converged)
  {
    result.reject_reason = "registration_not_converged";
  }
  else if (!ratio_support && !strong_support)
  {
    result.reject_reason = "low_plane_support";
  }
  else if (!std::isfinite(result.rmse) || result.rmse > options_.max_rmse)
  {
    result.reject_reason = "high_plane_residual";
  }
  else if (options_.max_mean_normalized_residual > 0.0 &&
           (!std::isfinite(result.mean_normalized_residual) ||
            result.mean_normalized_residual >
                options_.max_mean_normalized_residual))
  {
    result.reject_reason = "high_normalized_residual";
  }
  else if (result.observable_directions < options_.min_observable_directions &&
           !strong_support)
  {
    result.reject_reason = "insufficient_observable_directions";
  }
  else if (relative_motion.translation().norm() > translation_motion_gate ||
           rotationDegrees(relative_motion.rotation()) > rotation_motion_gate)
  {
    result.reject_reason = "implausible_scan_motion";
  }
  else if (!rotation_correction_nis_valid)
  {
    result.reject_reason = "rotation_correction_innovation";
  }
  else if (!cumulative_yaw_correction_valid)
  {
    result.reject_reason = "persistent_lidar_yaw_correction";
  }
  else if (lidar_correction.translation().norm() > lidar_correction_translation_gate ||
           rotationDegrees(lidar_correction.rotation()) >
               lidar_correction_rotation_gate)
  {
    result.reject_reason = "implausible_lidar_correction";
  }
  else if (result.degenerate && !strong_support &&
           (result.inlier_ratio < options_.degenerate_min_inlier_ratio ||
            result.rmse > options_.degenerate_max_rmse))
  {
    result.reject_reason = "weak_degenerate_registration";
  }
  else
  {
    result.accepted = true;
    result.reject_reason = "accepted";
  }

  if (result.accepted)
  {
    if (options_.lidar_yaw_correction_window_sec > 0.0)
    {
      lidar_yaw_correction_history_.push_back(
          YawCorrectionSample{scan_end_stamp, lidar_yaw_correction});
    }
    const double dt = previous_scan_stamp_ > 0.0
        ? scan_end_stamp - previous_scan_stamp_ : 0.0;
    state_ = estimate;
    const Eigen::LDLT<Matrix18d> information_solver(final_information);
    if (information_solver.info() == Eigen::Success)
    {
      state_.covariance = information_solver.solve(Matrix18d::Identity());
      state_.covariance = 0.5 * (state_.covariance + state_.covariance.transpose());
      if (options_.preserve_unobservable_covariance &&
          !options_.project_lidar_information_to_observable_subspace &&
          result.observable_directions < 6)
      {
        // Retain predicted uncertainty in the scan-unobservable pose
        // directions. The two projected PSD terms avoid manufacturing a
        // confident covariance merely because the normal equations were
        // numerically invertible after damping.
        Matrix18d observed_projection = Matrix18d::Identity();
        observed_projection.block<6, 6>(0, 0) =
            final_observability_projection;
        Matrix18d unobservable_projection = Matrix18d::Zero();
        unobservable_projection.block<6, 6>(0, 0) =
            Eigen::Matrix<double, 6, 6>::Identity() -
            final_observability_projection;
        state_.covariance = observed_projection * state_.covariance *
                observed_projection.transpose() +
            unobservable_projection * propagated_covariance *
                unobservable_projection.transpose();
        state_.covariance = 0.5 *
            (state_.covariance + state_.covariance.transpose());
      }
    }
    else
    {
      state_.covariance = propagated_covariance;
    }
    if (!options_.imu_enabled && dt > 1e-6)
    {
      state_.velocity = (state_.position - pose_before_scan.translation()) / dt;
    }
    resetCovarianceAfterInjection(
        state_, stateDifference(state_, propagated_state));
    ++accepted_scan_count_;
    consecutive_rejections_ = 0;
    lidar_loss_limited_ = false;
    lidar_loss_frozen_ = false;
    last_accepted_state_ = state_;
    const bool observable_for_map = result.observable_directions >=
        options_.map_insertion_min_observable_directions;
    const bool rotation_observable_for_map =
        options_.map_insertion_min_observable_rotation_directions <= 0 ||
        result.observable_rotation_directions >=
            options_.map_insertion_min_observable_rotation_directions;
    const bool yaw_observable_for_map =
        options_.map_insertion_min_yaw_observability <= 0.0 ||
        result.yaw_observability >=
            options_.map_insertion_min_yaw_observability;
    const bool innovation_for_map =
        options_.map_insertion_max_mean_normalized_residual <= 0.0 ||
        (std::isfinite(result.mean_normalized_residual) &&
         result.mean_normalized_residual <=
             options_.map_insertion_max_mean_normalized_residual);
    const bool convergence_for_map =
        !options_.map_insertion_require_convergence || result.converged;
    const bool yaw_guard_for_map =
        !options_.defer_map_when_lidar_yaw_limited ||
        !result.lidar_yaw_correction_limited;
    const bool correction_for_map =
        (options_.map_insertion_max_lidar_correction_translation <= 0.0 ||
         lidar_correction.translation().norm() <=
             options_.map_insertion_max_lidar_correction_translation) &&
        (options_.map_insertion_max_lidar_correction_rotation_deg <= 0.0 ||
         rotationDegrees(lidar_correction.rotation()) <=
             options_.map_insertion_max_lidar_correction_rotation_deg);
    bool recovery_trusted_for_map = true;
    if (recovery_map_guard_active_)
    {
      if (result.strong_support)
      {
        ++recovery_map_trusted_scan_count_;
      }
      else
      {
        recovery_map_trusted_scan_count_ = 0;
      }
      const int required_strong_scans = std::max(
          1, options_.recovery_map_insert_min_consecutive_strong_support);
      recovery_trusted_for_map = recovery_map_trusted_scan_count_ >= required_strong_scans;
      if (recovery_trusted_for_map)
      {
        recovery_map_guard_active_ = false;
      }
    }
    const bool map_keyframe_selected =
        shouldInsertMap(state_, scan_end_stamp);
    result.map_keyframe_selected = map_keyframe_selected;
    if (observable_for_map && rotation_observable_for_map &&
        yaw_observable_for_map && innovation_for_map && convergence_for_map &&
        yaw_guard_for_map && correction_for_map && recovery_trusted_for_map &&
        map_keyframe_selected)
    {
      insertMapPoints(scan, state_, true);
      have_last_map_insert_pose_ = true;
      last_map_insert_pose_ = statePose(state_);
      last_map_insert_stamp_ = scan_end_stamp;
      result.map_updated = true;
      result.map_update_reason = "updated";
    }
    else
    {
      result.map_update_deferred = true;
      if (!observable_for_map)
      {
        result.map_update_reason = "insufficient_observability";
      }
      else if (!rotation_observable_for_map)
      {
        result.map_update_reason = "insufficient_rotation_observability";
      }
      else if (!yaw_observable_for_map)
      {
        result.map_update_reason = "insufficient_yaw_observability";
      }
      else if (!innovation_for_map)
      {
        result.map_update_reason = "high_normalized_residual";
      }
      else if (!convergence_for_map)
      {
        result.map_update_reason = "registration_not_converged";
      }
      else if (!yaw_guard_for_map)
      {
        result.map_update_reason = "limited_lidar_yaw_correction";
      }
      else if (!correction_for_map)
      {
        result.map_update_reason = "large_lidar_correction";
      }
      else if (!recovery_trusted_for_map)
      {
        result.map_update_reason = "recovery_confirmation";
      }
      else
      {
        result.map_update_reason = "not_keyframe";
      }
    }
  }
  else
  {
    state_ = propagated_state;
    state_.covariance = propagated_covariance;
    result.loss_limited = applyLidarLossProtection(state_before_scan, scan_end_stamp);
    result.map_update_reason = "registration_rejected";
  }

  pose_cache_ = statePose(state_);
  result.acceleration_bias_correction =
      state_.acceleration_bias - propagated_state.acceleration_bias;
  result.relative_pose = pose_before_scan.inverse() * pose_cache_;
  previous_scan_pose_ = last_scan_pose_;
  last_scan_pose_ = pose_cache_;
  previous_scan_stamp_ = scan_end_stamp;
  recordCurrentPose();
  fillResultState(result);
  pruneImu(scan_end_stamp);
  return result;
}

}  // namespace hybrid_localization
