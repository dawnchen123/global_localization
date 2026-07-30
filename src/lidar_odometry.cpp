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
  options_.plane_fit_residual_gate = std::max(0.02, options_.plane_fit_residual_gate);
  options_.strong_support_min_correspondences = std::max(
      0, options_.strong_support_min_correspondences);
  options_.point_knn_fallback_max_queries = std::max(
      0, options_.point_knn_fallback_max_queries);
  options_.strong_support_min_azimuth_sectors = std::max(
      1, options_.strong_support_min_azimuth_sectors);
  options_.strong_support_max_rmse = std::max(0.0, options_.strong_support_max_rmse);
  options_.recovery_after_rejections = std::max(0, options_.recovery_after_rejections);
  options_.recovery_max_lidar_correction_translation = std::max(
      0.0, options_.recovery_max_lidar_correction_translation);
  options_.recovery_max_lidar_correction_rotation_deg = std::max(
      0.0, options_.recovery_max_lidar_correction_rotation_deg);
  options_.lidar_measurement_noise = std::max(0.005, options_.lidar_measurement_noise);
  options_.huber_delta = std::max(0.01, options_.huber_delta);
  options_.max_translation_per_scan = std::max(0.05, options_.max_translation_per_scan);
  options_.max_rotation_per_scan_deg = std::max(0.1, options_.max_rotation_per_scan_deg);
  options_.max_translation_speed = std::max(0.0, options_.max_translation_speed);
  options_.max_rotation_speed_deg = std::max(0.0, options_.max_rotation_speed_deg);
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
  options_.max_iterations = std::max(1, options_.max_iterations);
  options_.min_scan_points = std::max(20, options_.min_scan_points);
  options_.min_correspondences = std::max(20, options_.min_correspondences);
  options_.max_scan_points = std::max(options_.min_scan_points, options_.max_scan_points);
  options_.max_map_points = std::max(options_.min_correspondences, options_.max_map_points);
  options_.normal_neighbor_voxels = std::max(1, options_.normal_neighbor_voxels);
  options_.min_normal_neighbors = std::max(4, options_.min_normal_neighbors);
  options_.max_plane_neighbors = std::max(options_.min_normal_neighbors,
                                           options_.max_plane_neighbors);
  options_.min_voxel_plane_points = std::max(4, options_.min_voxel_plane_points);
  options_.max_voxel_points = std::max(options_.min_voxel_plane_points,
                                        options_.max_voxel_points);
  options_.max_voxel_samples = std::max(4, options_.max_voxel_samples);
  options_.imu_init_samples = std::max(20, options_.imu_init_samples);
  options_.imu_init_duration = std::max(0.1, options_.imu_init_duration);
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
  imu_buffer_.clear();
  propagation_history_.clear();
  wheel_buffer_.clear();
  map_voxels_.clear();
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

bool LidarOdometry::wheelMeasurement(double stamp, double *forward_speed) const
{
  if (!options_.wheel_enabled || wheel_buffer_.empty() || forward_speed == nullptr)
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
  *forward_speed = best->forward_speed;
  return true;
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
  state_.covariance.block<3, 3>(9, 9).diagonal() =
      angular_velocity_variance.cwiseMax(Eigen::Vector3d::Constant(1e-8));
  state_.covariance.block<3, 3>(12, 12).diagonal() =
      (acceleration_scale_ * acceleration_scale_ * acceleration_variance)
          .cwiseMax(Eigen::Vector3d::Constant(1e-5));
  state_.covariance.block<3, 3>(15, 15).diagonal().setConstant(1e-3);
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
  if (previous_scan_stamp_ > 0.0 && state_stamp_ > previous_scan_stamp_)
  {
    const double previous_dt = state_stamp_ - previous_scan_stamp_;
    const Eigen::Matrix3d relative_rotation =
        previous_scan_pose_.rotation().transpose() * last_scan_pose_.rotation();
    state_.rotation *= expSO3(logSO3(relative_rotation) * dt / previous_dt);
  }
  state_.covariance.block<3, 3>(0, 0).diagonal().array() += 1e-3 * dt;
  state_.covariance.block<3, 3>(3, 3).diagonal().array() += 1e-2 * dt;
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

bool LidarOdometry::findLocalPlane(const Eigen::Vector3d &world_point,
                                   PlaneMatch &match) const
{
  if (options_.use_point_knn_plane) return findPointKnnPlane(world_point, match);
  if (options_.use_compatible_voxel_plane)
  {
    if (findCompatibleVoxelPlane(world_point, match)) return true;
  }
  return findSmoothVoxelPlane(world_point, match);
}

bool LidarOdometry::findCompatibleVoxelPlane(const Eigen::Vector3d &world_point,
                                              PlaneMatch &match) const
{
  if (map_voxels_.empty()) return false;
  const VoxelKey center = voxelKey(world_point, options_.map_voxel_size);
  const int radius = std::max(1, static_cast<int>(std::ceil(
      options_.max_correspondence_distance / options_.map_voxel_size)));
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
    *normal = solver.eigenvectors().col(0).normalized();
    return normal->allFinite();
  };

  std::vector<int> inliers(neighbors.size());
  std::iota(inliers.begin(), inliers.end(), 0);
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  double total_weight = 0.0;
  if (!fit_plane(inliers, &mean, &normal, &eigenvalues, &total_weight)) return false;

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
      !fit_plane(robust_inliers, &mean, &normal, &eigenvalues, &total_weight))
  {
    return false;
  }
  if (eigenvalues.y() < 1e-6 || eigenvalues.x() > options_.max_plane_variance ||
      eigenvalues.x() > options_.plane_max_eigen_ratio * eigenvalues.y())
  {
    return false;
  }
  match.center = mean;
  match.normal = normal;
  const double ratio = eigenvalues.x() / std::max(1e-9, eigenvalues.y());
  match.variance = std::max(1e-6, eigenvalues.x()) +
      options_.plane_uncertainty_scale * ratio / total_weight;
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

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  double total_weight = 0.0;
  const double kernel_sigma = std::max(options_.map_voxel_size,
                                        0.5 * options_.max_correspondence_distance);
  const double inverse_two_sigma_squared = 0.5 / (kernel_sigma * kernel_sigma);
  for (const VoxelNeighbor &neighbor : neighbors)
  {
    const double maturity = std::min(1.0, static_cast<double>(neighbor.voxel->count) /
                                          options_.min_voxel_plane_points);
    const double weight = maturity * std::exp(
        -neighbor.squared_distance * inverse_two_sigma_squared);
    mean += weight * neighbor.voxel->mean;
    total_weight += weight;
  }
  if (total_weight < 1e-3) return false;
  mean /= total_weight;

  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  for (const VoxelNeighbor &neighbor : neighbors)
  {
    const MapVoxel &voxel = *neighbor.voxel;
    const double maturity = std::min(1.0, static_cast<double>(voxel.count) /
                                          options_.min_voxel_plane_points);
    const double weight = maturity * std::exp(
        -neighbor.squared_distance * inverse_two_sigma_squared);
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    if (voxel.count > 1)
    {
      covariance = voxel.scatter / static_cast<double>(voxel.count - 1);
    }
    const Eigen::Vector3d centered = voxel.mean - mean;
    scatter.noalias() += weight * (covariance + centered * centered.transpose());
  }
  scatter /= total_weight;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
  if (solver.info() != Eigen::Success) return false;
  const Eigen::Vector3d eigenvalues = solver.eigenvalues();
  if (eigenvalues.y() < 1e-6 || eigenvalues.x() > options_.max_plane_variance ||
      eigenvalues.x() > options_.plane_max_eigen_ratio * eigenvalues.y())
  {
    return false;
  }
  match.center = mean;
  match.normal = solver.eigenvectors().col(0).normalized();
  const double ratio = eigenvalues.x() / std::max(1e-9, eigenvalues.y());
  match.variance = std::max(1e-6, eigenvalues.x()) +
      options_.plane_uncertainty_scale * ratio / total_weight;
  match.nearest_squared_distance = std::min_element(
      neighbors.begin(), neighbors.end(),
      [](const VoxelNeighbor &first, const VoxelNeighbor &second)
      { return first.squared_distance < second.squared_distance; })->squared_distance;
  return true;
}

void LidarOdometry::updateVoxel(MapVoxel &voxel, const Eigen::Vector3d &point)
{
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
    const double alpha = 1.0 / effective_count;
    voxel.mean += alpha * delta;
    const Eigen::Vector3d centered = point - voxel.mean;
    covariance = (1.0 - alpha) * covariance + alpha * centered * centered.transpose();
    voxel.scatter = (effective_count - 1.0) *
                    0.5 * (covariance + covariance.transpose());
  }
  voxel.last_seen_scan = accepted_scan_count_;
}

void LidarOdometry::updateVoxelPlane(MapVoxel &voxel)
{
  voxel.plane_valid = false;
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
    if (filter_existing && existing != map_voxels_.end() && existing->second.plane_valid)
    {
      const MapVoxel &voxel = existing->second;
      if (std::abs(voxel.normal.dot(world_point - voxel.mean)) >
          options_.map_insertion_max_plane_distance)
      {
        continue;
      }
    }
    updateVoxel(map_voxels_[key], world_point);
    touched.insert(key);
  }
  for (const VoxelKey &key : touched)
  {
    const auto iterator = map_voxels_.find(key);
    if (iterator != map_voxels_.end()) updateVoxelPlane(iterator->second);
  }
  if (accepted_scan_count_ % 10 == 0 ||
      map_voxels_.size() > static_cast<std::size_t>(1.10 * options_.max_map_points))
  {
    pruneMap();
  }
}

void LidarOdometry::pruneMap()
{
  const double radius_squared = options_.local_map_radius * options_.local_map_radius;
  for (auto iterator = map_voxels_.begin(); iterator != map_voxels_.end();)
  {
    if ((iterator->second.mean - state_.position).squaredNorm() > radius_squared)
    {
      iterator = map_voxels_.erase(iterator);
    }
    else
    {
      ++iterator;
    }
  }
  if (map_voxels_.size() <= static_cast<std::size_t>(options_.max_map_points))
  {
    return;
  }
  struct RankedKey
  {
    VoxelKey key;
    double squared_distance = 0.0;
  };
  std::vector<RankedKey> ranked;
  ranked.reserve(map_voxels_.size());
  for (const auto &entry : map_voxels_)
  {
    ranked.push_back(RankedKey{entry.first,
        (entry.second.mean - state_.position).squaredNorm()});
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedKey &first, const RankedKey &second)
            { return first.squared_distance < second.squared_distance; });
  for (std::size_t index = static_cast<std::size_t>(options_.max_map_points);
       index < ranked.size(); ++index)
  {
    map_voxels_.erase(ranked[index].key);
  }
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
  lidar_loss_limited_ = false;
  lidar_loss_frozen_ = false;
  if (!map_initialized_ || options_.lidar_loss_hold_after_rejections <= 0 ||
      consecutive_rejections_ < options_.lidar_loss_hold_after_rejections)
  {
    return false;
  }

  if (options_.lidar_loss_freeze_after_rejections > 0 &&
      consecutive_rejections_ >= options_.lidar_loss_freeze_after_rejections)
  {
    // Continued propagation is no longer an estimate once the local map has
    // provided no support for this long. Freeze the full pose so downstream
    // semantic mapping cannot accumulate data at a fabricated location.
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
  state_.velocity.head<2>() = horizontal_velocity * options_.lidar_loss_velocity_decay;

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
    Matrix18d measurement_hessian = Matrix18d::Zero();
    Vector18d measurement_gradient = Vector18d::Zero();
    measurement_hessian.block<6, 6>(0, 0) = linearization.hessian;
    measurement_gradient.segment<6>(0) = linearization.gradient;
    const Vector18d displacement = stateDifference(estimate, propagated_state);
    Matrix18d information = prior_information + measurement_hessian;
    information.diagonal().array() += options_.visual_solver_damping;
    const Vector18d gradient =
        prior_information * displacement + measurement_gradient;
    const Eigen::LDLT<Matrix18d> solver(information);
    if (solver.info() != Eigen::Success) break;
    Vector18d step = solver.solve(-gradient);
    if (!step.allFinite()) break;

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
    step.segment<12>(6).setZero();
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
  const Eigen::Isometry3d estimate_pose = statePose(estimate);
  const Eigen::Isometry3d correction = propagated_pose.inverse() * estimate_pose;
  const double translation_step = correction.translation().norm();
  const double rotation_step = rotationDegrees(correction.rotation());
  const bool measurement_valid = final_linearization.valid &&
      final_linearization.landmarks >= options_.visual_min_landmarks &&
      final_linearization.residuals >= options_.visual_min_residuals &&
      std::isfinite(final_linearization.rmse) &&
      final_linearization.rmse <= options_.visual_max_rmse;
  const bool motion_valid = translation_step <= options_.visual_max_translation_step &&
      rotation_step <= options_.visual_max_rotation_step_deg;
  if (measurement_valid && motion_valid && result.iterations > 0)
  {
    state_ = estimate;
    const Eigen::LDLT<Matrix18d> information_solver(final_information);
    if (information_solver.info() == Eigen::Success)
    {
      state_.covariance = information_solver.solve(Matrix18d::Identity());
      state_.covariance = 0.5 *
          (state_.covariance + state_.covariance.transpose());
    }
    result.accepted = true;
    result.reason = "visual_accepted";
  }
  else
  {
    state_ = propagated_state;
    if (!final_linearization.valid)
    {
      result.reason = final_linearization.reason;
    }
    else if (!measurement_valid)
    {
      result.reason = "visual_quality_gate";
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
    pose_cache_ = statePose(state_);
    last_scan_pose_ = pose_cache_;
    previous_scan_pose_ = pose_cache_;
    previous_scan_stamp_ = scan_end_stamp;
    result.accepted = true;
    result.converged = true;
    result.map_updated = true;
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
  double final_squared_error = std::numeric_limits<double>::infinity();
  int final_correspondences = 0;
  int final_correspondence_sectors = 0;
  int final_point_knn_fallback_queries = 0;
  int final_point_knn_fallback_matches = 0;
  double measured_forward_speed = 0.0;
  const bool have_wheel_measurement = wheelMeasurement(scan_end_stamp,
                                                        &measured_forward_speed);
  result.used_wheel = have_wheel_measurement;
  result.wheel_speed = measured_forward_speed;

  for (int iteration = 0; iteration < options_.max_iterations; ++iteration)
  {
    Matrix18d measurement_hessian = Matrix18d::Zero();
    Vector18d measurement_gradient = Vector18d::Zero();
    double squared_error = 0.0;
    int correspondences = 0;
    int point_knn_fallback_queries = 0;
    int point_knn_fallback_matches = 0;
    constexpr int kAzimuthSectorCount = 12;
    std::array<bool, kAzimuthSectorCount> correspondence_sectors{};
    for (const Eigen::Vector3d &body_point : scan)
    {
      const Eigen::Vector3d world_point =
          estimate.rotation * body_point + estimate.position;
      PlaneMatch match;
      bool found_match = findLocalPlane(world_point, match);
      if (!found_match && !options_.use_point_knn_plane && options_.point_knn_fallback &&
          (options_.point_knn_fallback_max_queries == 0 ||
           point_knn_fallback_queries < options_.point_knn_fallback_max_queries))
      {
        ++point_knn_fallback_queries;
        found_match = findPointKnnPlane(world_point, match);
        if (found_match)
        {
          ++point_knn_fallback_matches;
        }
      }
      if (!found_match)
      {
        continue;
      }
      const double residual = match.normal.dot(world_point - match.center);
      if (std::abs(residual) > options_.max_plane_distance)
      {
        continue;
      }
      Eigen::Matrix<double, 1, 18> jacobian =
          Eigen::Matrix<double, 1, 18>::Zero();
      jacobian.block<1, 3>(0, 0) =
          -match.normal.transpose() * estimate.rotation * skew(body_point);
      jacobian.block<1, 3>(0, 3) = match.normal.transpose();
      const double range = body_point.norm();
      const double measurement_variance =
          options_.lidar_measurement_noise * options_.lidar_measurement_noise +
          options_.lidar_range_noise * options_.lidar_range_noise +
          std::pow(options_.lidar_beam_noise * range, 2.0) + match.variance;
      const double robust_information = huberWeight(residual, options_.huber_delta) /
          std::max(1e-8, measurement_variance);
      measurement_hessian.noalias() +=
          robust_information * jacobian.transpose() * jacobian;
      measurement_gradient.noalias() +=
          robust_information * jacobian.transpose() * residual;
      squared_error += residual * residual;
      ++correspondences;
      const double azimuth = std::atan2(body_point.y(), body_point.x());
      const double normalized_azimuth = std::max(0.0, std::min(
          1.0 - 1e-12, (azimuth + kPi) / (2.0 * kPi)));
      const int sector = std::min(kAzimuthSectorCount - 1,
          static_cast<int>(std::floor(normalized_azimuth * kAzimuthSectorCount)));
      correspondence_sectors[static_cast<std::size_t>(sector)] = true;
    }

    if (have_wheel_measurement)
    {
      const Eigen::Vector3d body_velocity = estimate.rotation.transpose() * estimate.velocity;
      Eigen::Vector3d residual;
      residual << body_velocity.x() - measured_forward_speed,
                  body_velocity.y(), estimate.velocity.z();
      Eigen::Matrix<double, 3, 18> jacobian = Eigen::Matrix<double, 3, 18>::Zero();
      jacobian.block<2, 3>(0, 6) = estimate.rotation.transpose().topRows<2>();
      jacobian(2, 8) = 1.0;
      const Eigen::Vector3d sigma(options_.wheel_forward_noise,
                                  options_.wheel_lateral_noise,
                                  options_.wheel_vertical_noise);
      for (int axis = 0; axis < 3; ++axis)
      {
        const double normalized = residual(axis) / sigma(axis);
        const double information = huberWeight(normalized, options_.wheel_huber_delta) /
                                   (sigma(axis) * sigma(axis));
        const Eigen::Matrix<double, 1, 18> row = jacobian.row(axis);
        measurement_hessian.noalias() += information * row.transpose() * row;
        measurement_gradient.noalias() += information * row.transpose() * residual(axis);
      }
      result.wheel_velocity_residual = residual.norm();
    }

    final_correspondences = correspondences;
    final_correspondence_sectors = static_cast<int>(std::count(
        correspondence_sectors.begin(), correspondence_sectors.end(), true));
    final_point_knn_fallback_queries = point_knn_fallback_queries;
    final_point_knn_fallback_matches = point_knn_fallback_matches;
    if (correspondences < options_.min_correspondences)
    {
      break;
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
    const auto clamp_segment = [&step](int start, double maximum)
    {
      const double norm = step.segment<3>(start).norm();
      if (norm > maximum && norm > kSmall)
      {
        step.segment<3>(start) *= maximum / norm;
      }
    };
    clamp_segment(0, 3.0 * kPi / 180.0);
    clamp_segment(3, 0.40);
    clamp_segment(6, 1.0);
    clamp_segment(9, 0.02);
    clamp_segment(12, 0.10);
    clamp_segment(15, 0.05);
    applyError(estimate, step);
    final_information = information;
    final_measurement_hessian = measurement_hessian.block<6, 6>(0, 0);
    final_squared_error = squared_error;
    result.iterations = iteration + 1;
    if (step.segment<3>(3).norm() < options_.convergence_translation &&
        step.segment<3>(0).norm() * 180.0 / kPi < options_.convergence_rotation_deg)
    {
      result.converged = true;
      break;
    }
  }

  result.correspondences = final_correspondences;
  result.correspondence_azimuth_sectors = final_correspondence_sectors;
  result.point_knn_fallback_queries = final_point_knn_fallback_queries;
  result.point_knn_fallback_matches = final_point_knn_fallback_matches;
  result.inlier_ratio = static_cast<double>(final_correspondences) /
      static_cast<double>(std::max<std::size_t>(1U, scan.size()));
  result.rmse = final_correspondences > 0 && std::isfinite(final_squared_error)
      ? std::sqrt(final_squared_error / static_cast<double>(final_correspondences))
      : std::numeric_limits<double>::infinity();
  if (final_correspondences >= options_.min_correspondences)
  {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigen_solver(
        final_measurement_hessian);
    if (eigen_solver.info() == Eigen::Success)
    {
      const double largest = std::max(kSmall, eigen_solver.eigenvalues().maxCoeff());
      result.degenerate = eigen_solver.eigenvalues().minCoeff() <
          largest * options_.degeneracy_eigen_ratio;
    }
  }

  const Eigen::Isometry3d estimated_pose = statePose(estimate);
  const Eigen::Isometry3d propagated_pose = statePose(propagated_state);
  const Eigen::Isometry3d relative_motion = pose_before_scan.inverse() * estimated_pose;
  const Eigen::Isometry3d lidar_correction = propagated_pose.inverse() * estimated_pose;
  const double scan_dt = previous_scan_stamp_ > 0.0
      ? std::max(0.0, scan_end_stamp - previous_scan_stamp_) : 0.0;
  const double translation_motion_gate = std::max(
      options_.max_translation_per_scan, options_.max_translation_speed * scan_dt);
  const double rotation_motion_gate = std::max(
      options_.max_rotation_per_scan_deg, options_.max_rotation_speed_deg * scan_dt);
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
  const double lidar_correction_rotation_gate = recovery_mode && strong_support &&
      options_.recovery_max_lidar_correction_rotation_deg > 0.0
      ? std::max(options_.max_lidar_correction_rotation_deg,
                 options_.recovery_max_lidar_correction_rotation_deg)
      : options_.max_lidar_correction_rotation_deg;
  if (final_correspondences < options_.min_correspondences)
  {
    result.reject_reason = "too_few_plane_correspondences";
  }
  else if (!ratio_support && !strong_support)
  {
    result.reject_reason = "low_plane_support";
  }
  else if (!std::isfinite(result.rmse) || result.rmse > options_.max_rmse)
  {
    result.reject_reason = "high_plane_residual";
  }
  else if (relative_motion.translation().norm() > translation_motion_gate ||
           rotationDegrees(relative_motion.rotation()) > rotation_motion_gate)
  {
    result.reject_reason = "implausible_scan_motion";
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
    const double dt = previous_scan_stamp_ > 0.0
        ? scan_end_stamp - previous_scan_stamp_ : 0.0;
    state_ = estimate;
    const Eigen::LDLT<Matrix18d> information_solver(final_information);
    if (information_solver.info() == Eigen::Success)
    {
      state_.covariance = information_solver.solve(Matrix18d::Identity());
      state_.covariance = 0.5 * (state_.covariance + state_.covariance.transpose());
    }
    else
    {
      state_.covariance = propagated_covariance;
    }
    if (!options_.imu_enabled && dt > 1e-6)
    {
      state_.velocity = (state_.position - pose_before_scan.translation()) / dt;
    }
    ++accepted_scan_count_;
    consecutive_rejections_ = 0;
    lidar_loss_limited_ = false;
    lidar_loss_frozen_ = false;
    last_accepted_state_ = state_;
    insertMapPoints(scan, state_, true);
    result.map_updated = true;
  }
  else
  {
    state_ = propagated_state;
    state_.covariance = propagated_covariance;
    result.loss_limited = applyLidarLossProtection(state_before_scan, scan_end_stamp);
  }

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

}  // namespace hybrid_localization
