#include "hybrid_localization/core.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hybrid_localization
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-9;

Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}

Eigen::Matrix3d rotationFromYaw(double yaw)
{
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

double yawFromRotation(const Eigen::Matrix3d &rotation)
{
  return std::atan2(rotation(1, 0), rotation(0, 0));
}

Eigen::MatrixXd safeInformation(const Eigen::MatrixXd &information, int dimension)
{
  if (information.rows() == dimension && information.cols() == dimension)
  {
    return information;
  }
  return Eigen::MatrixXd::Identity(dimension, dimension);
}
}  // namespace

void ImuPreintegration::reset()
{
  valid = false;
  dt = 0.0;
  delta_rotation = Eigen::Quaterniond::Identity();
  delta_position.setZero();
  delta_velocity.setZero();
}

void ImuPreintegration::integrate(const Eigen::Vector3d &accel, const Eigen::Vector3d &gyro,
                                  double dt_sec)
{
  if (dt_sec <= 0.0 || dt_sec > 0.2)
  {
    return;
  }
  const Eigen::Vector3d acceleration_in_start = delta_rotation * accel;
  const Eigen::Vector3d old_velocity = delta_velocity;
  delta_position += old_velocity * dt_sec + 0.5 * acceleration_in_start * dt_sec * dt_sec;
  delta_velocity += acceleration_in_start * dt_sec;
  const double angle = gyro.norm() * dt_sec;
  if (angle > kEpsilon)
  {
    delta_rotation = (delta_rotation * Eigen::Quaterniond(
        Eigen::AngleAxisd(angle, gyro.normalized()))).normalized();
  }
  dt += dt_sec;
  valid = dt > 0.0;
}

Eigen::Isometry3d expSE3(const Eigen::Matrix<double, 6, 1> &xi)
{
  const Eigen::Vector3d omega = xi.head<3>();
  const Eigen::Vector3d translation = xi.tail<3>();
  const double theta = omega.norm();
  const Eigen::Matrix3d omega_hat = skew(omega);
  const Eigen::Matrix3d omega_hat2 = omega_hat * omega_hat;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
  if (theta > 1e-8)
  {
    rotation = Eigen::AngleAxisd(theta, omega / theta).toRotationMatrix();
    V += ((1.0 - std::cos(theta)) / (theta * theta)) * omega_hat;
    V += ((theta - std::sin(theta)) / (theta * theta * theta)) * omega_hat2;
  }
  else
  {
    rotation += omega_hat;
    V += 0.5 * omega_hat;
  }
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = rotation;
  result.translation() = V * translation;
  return result;
}

Eigen::Isometry3d projectToSE3(const Eigen::Isometry3d &transform)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  if (transform.translation().allFinite())
  {
    result.translation() = transform.translation();
  }
  if (!transform.linear().allFinite())
  {
    return result;
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      transform.linear(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d u = svd.matrixU();
  const Eigen::Matrix3d v = svd.matrixV();
  Eigen::Matrix3d rotation = u * v.transpose();
  if (rotation.determinant() < 0.0)
  {
    u.col(2) *= -1.0;
    rotation = u * v.transpose();
  }
  if (rotation.allFinite())
  {
    result.linear() = rotation;
  }
  return result;
}

Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Isometry3d &transform)
{
  Eigen::Matrix<double, 6, 1> result;
  const Eigen::AngleAxisd angle_axis(transform.rotation());
  const double theta = angle_axis.angle();
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  if (theta > 1e-8)
  {
    omega = theta * angle_axis.axis();
  }
  const Eigen::Matrix3d omega_hat = skew(omega);
  const Eigen::Matrix3d omega_hat2 = omega_hat * omega_hat;
  Eigen::Matrix3d V_inverse = Eigen::Matrix3d::Identity();
  if (theta > 1e-8)
  {
    const double half_theta = 0.5 * theta;
    const double cot_half_theta = std::cos(half_theta) / std::sin(half_theta);
    V_inverse -= 0.5 * omega_hat;
    V_inverse += (1.0 - half_theta * cot_half_theta) / (theta * theta) * omega_hat2;
  }
  else
  {
    V_inverse -= 0.5 * omega_hat;
  }
  result.head<3>() = omega;
  result.tail<3>() = V_inverse * transform.translation();
  return result;
}

double wrapAngle(double angle)
{
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

double yawOf(const Eigen::Isometry3d &pose)
{
  return yawFromRotation(pose.rotation());
}

Eigen::Isometry3d planarTransform(double x, double y, double yaw)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = rotationFromYaw(yaw);
  result.translation() = Eigen::Vector3d(x, y, 0.0);
  return result;
}

uint8_t convertSemanticLabel(int label, const std::string &mode)
{
  if (mode == "internal")
  {
    return static_cast<uint8_t>(std::max(0, std::min(6, label)));
  }
  if (mode == "sam3")
  {
    switch (label)
    {
      case 0: return 0U;
      case 1: return 1U;  // road
      case 2: return 3U;  // building
      case 3:             // tree
      case 4: return 4U;  // grass
      case 5: return 5U;  // dynamic object
      default: return 6U;
    }
  }
  if (label == 0)
  {
    return 0U;
  }
  if (label == 40 || label == 44)
  {
    return 1U;
  }
  if (label == 48 || label == 49)
  {
    return 2U;
  }
  if (label == 50)
  {
    return 3U;
  }
  if (label == 70 || label == 71 || label == 72 || label == 80 || label == 81)
  {
    return 4U;
  }
  if (label == 10 || label == 11 || label == 13 || label == 15 || label == 18 ||
      label == 20 || label == 30 || label == 31 || label == 32)
  {
    return 5U;
  }
  return 6U;
}

SlidingWindowOptimizer::SlidingWindowOptimizer(std::size_t max_states)
    : max_states_(std::max<std::size_t>(2, max_states))
{
}

void SlidingWindowOptimizer::reset()
{
  states_.clear();
  factors_.clear();
}

int SlidingWindowOptimizer::addState(const PoseState &state)
{
  PoseState copy = state;
  copy.pose = projectToSE3(copy.pose);
  copy.id = static_cast<int>(states_.size());
  states_.push_back(copy);
  while (states_.size() > max_states_)
  {
    marginalizeOldest();
  }
  return static_cast<int>(states_.size() - 1);
}

void SlidingWindowOptimizer::addRelativeFactor(int first, int second,
                                                const Eigen::Isometry3d &measurement,
                                                const Eigen::Matrix<double, 6, 6> &information,
                                                FactorType type, double huber_delta)
{
  Factor factor;
  factor.type = type;
  factor.first = first;
  factor.second = second;
  factor.measurement = measurement;
  factor.information = information;
  factor.huber_delta = std::max(1e-6, huber_delta);
  factors_.push_back(factor);
}

void SlidingWindowOptimizer::addWheelFactor(int first, int second, const Eigen::Vector3d &measurement,
                                             const Eigen::Matrix3d &information, double huber_delta)
{
  Factor factor;
  factor.type = FactorType::Wheel;
  factor.first = first;
  factor.second = second;
  factor.wheel_measurement = measurement;
  factor.information = information;
  factor.huber_delta = std::max(1e-6, huber_delta);
  factors_.push_back(factor);
}

void SlidingWindowOptimizer::addImuFactor(int first, int second, const ImuPreintegration &measurement,
                                          const Eigen::Matrix<double, 9, 9> &information,
                                          double huber_delta)
{
  Factor factor;
  factor.type = FactorType::ImuPreintegration;
  factor.first = first;
  factor.second = second;
  factor.imu = measurement;
  factor.measurement = Eigen::Isometry3d::Identity();
  factor.measurement.linear() = measurement.delta_rotation.toRotationMatrix();
  factor.measurement.translation() = measurement.delta_position;
  factor.information = information;
  factor.huber_delta = std::max(1e-6, huber_delta);
  factors_.push_back(factor);
}

void SlidingWindowOptimizer::addAbsoluteFactor(int state_index, const Eigen::Isometry3d &measurement,
                                                const Eigen::Matrix<double, 6, 6> &information,
                                                FactorType type, double confidence, double huber_delta)
{
  Factor factor;
  factor.type = type;
  factor.first = state_index;
  factor.second = -1;
  factor.measurement = measurement;
  factor.information = information;
  factor.confidence = std::max(0.0, std::min(1.0, confidence));
  factor.huber_delta = std::max(1e-6, huber_delta);
  factors_.push_back(factor);
}

Eigen::VectorXd SlidingWindowOptimizer::residual(const Factor &factor,
                                                 const std::vector<PoseState> &states) const
{
  if (factor.first < 0 || factor.first >= static_cast<int>(states.size()))
  {
    return Eigen::VectorXd();
  }
  if (factor.type == FactorType::Prior || factor.second < 0)
  {
    return residualForAbsolute(factor, states[static_cast<std::size_t>(factor.first)]);
  }
  if (factor.second >= static_cast<int>(states.size()))
  {
    return Eigen::VectorXd();
  }
  return residualForType(factor, states[static_cast<std::size_t>(factor.first)],
                         states[static_cast<std::size_t>(factor.second)]);
}

Eigen::VectorXd SlidingWindowOptimizer::residualForType(const Factor &factor,
                                                        const PoseState &first,
                                                        const PoseState &second) const
{
  const Eigen::Isometry3d relative = first.pose.inverse() * second.pose;
  if (factor.type == FactorType::Wheel)
  {
    Eigen::Vector3d result;
    result << relative.translation().x() - factor.wheel_measurement.x(),
        relative.translation().y() - factor.wheel_measurement.y(),
        wrapAngle(yawOf(relative) - factor.wheel_measurement.z());
    return result;
  }
  if (factor.type == FactorType::ImuPreintegration)
  {
    Eigen::VectorXd result(9);
    result.head<3>() = logSE3(factor.measurement.inverse() * relative).head<3>();
    result.segment<3>(3) = relative.translation() - factor.imu.delta_position;
    result.tail<3>() = (second.velocity - first.velocity) - factor.imu.delta_velocity;
    return result;
  }
  return logSE3(factor.measurement.inverse() * relative);
}

Eigen::VectorXd SlidingWindowOptimizer::residualForAbsolute(const Factor &factor,
                                                             const PoseState &state) const
{
  return logSE3(factor.measurement.inverse() * state.pose);
}

Eigen::MatrixXd SlidingWindowOptimizer::numericalJacobian(
    const Factor &factor, int state_index, const std::vector<PoseState> &states) const
{
  const Eigen::VectorXd base = residual(factor, states);
  Eigen::MatrixXd jacobian(base.size(), 15);
  jacobian.setZero();
  if (state_index < 0 || state_index >= static_cast<int>(states.size()))
  {
    return jacobian;
  }
  constexpr double eps = 1e-5;
  for (int column = 0; column < 15; ++column)
  {
    std::vector<PoseState> perturbed = states;
    Eigen::Matrix<double, 15, 1> delta = Eigen::Matrix<double, 15, 1>::Zero();
    delta(column) = eps;
    boxPlus(perturbed[static_cast<std::size_t>(state_index)], delta);
    jacobian.col(column) = (residual(factor, perturbed) - base) / eps;
  }
  return jacobian;
}

void SlidingWindowOptimizer::boxPlus(PoseState &state, const Eigen::Matrix<double, 15, 1> &delta)
{
  const Eigen::Isometry3d updated_pose = state.pose * expSE3(delta.head<6>());
  state.pose = projectToSE3(updated_pose);
  state.velocity += delta.segment<3>(6);
  state.gyro_bias += delta.segment<3>(9);
  state.accel_bias += delta.segment<3>(12);
}

double SlidingWindowOptimizer::robustWeight(double squared_norm, double huber_delta)
{
  const double norm = std::sqrt(std::max(0.0, squared_norm));
  if (norm <= huber_delta || norm < 1e-12)
  {
    return 1.0;
  }
  return huber_delta / norm;
}

bool SlidingWindowOptimizer::optimize(int iterations, double max_step_norm)
{
  if (states_.empty() || factors_.empty())
  {
    return false;
  }
  const bool keep_first_fixed = states_.size() > 1;
  const std::size_t first_optimizable = keep_first_fixed ? 1 : 0;
  const int state_count = static_cast<int>(states_.size() - first_optimizable);
  if (state_count <= 0)
  {
    return false;
  }
  const int dimension = state_count * 15;
  bool changed = false;
  for (int iteration = 0; iteration < std::max(1, iterations); ++iteration)
  {
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(dimension, dimension);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dimension);
    double total_error = 0.0;
    int used_factors = 0;
    for (const Factor &factor : factors_)
    {
      if (!factor.active || factor.first < 0 || factor.first >= static_cast<int>(states_.size()))
      {
        continue;
      }
      const Eigen::VectorXd r = residual(factor, states_);
      if (r.size() == 0)
      {
        continue;
      }
      const Eigen::MatrixXd information = safeInformation(factor.information, r.size());
      const double squared_norm = (r.transpose() * information * r)(0, 0);
      const double weight = robustWeight(squared_norm, factor.huber_delta) * factor.confidence;
      const Eigen::MatrixXd weighted_information = weight * information;
      total_error += weight * squared_norm;
      ++used_factors;

      const bool first_is_optimizable = factor.first >= static_cast<int>(first_optimizable);
      const bool second_is_optimizable = factor.second >= static_cast<int>(first_optimizable);
      Eigen::MatrixXd first_jacobian;
      Eigen::MatrixXd second_jacobian;
      if (first_is_optimizable)
      {
        first_jacobian = numericalJacobian(factor, factor.first, states_);
        const int block = (factor.first - static_cast<int>(first_optimizable)) * 15;
        hessian.block(block, block, 15, 15).noalias() +=
            first_jacobian.transpose() * weighted_information * first_jacobian;
        gradient.segment(block, 15).noalias() +=
            first_jacobian.transpose() * weighted_information * r;
      }
      if (second_is_optimizable)
      {
        second_jacobian = numericalJacobian(factor, factor.second, states_);
        const int block = (factor.second - static_cast<int>(first_optimizable)) * 15;
        hessian.block(block, block, 15, 15).noalias() +=
            second_jacobian.transpose() * weighted_information * second_jacobian;
        gradient.segment(block, 15).noalias() +=
            second_jacobian.transpose() * weighted_information * r;
      }
      if (first_is_optimizable && second_is_optimizable)
      {
        const int first_block = (factor.first - static_cast<int>(first_optimizable)) * 15;
        const int second_block = (factor.second - static_cast<int>(first_optimizable)) * 15;
        const Eigen::MatrixXd cross = first_jacobian.transpose() * weighted_information * second_jacobian;
        hessian.block(first_block, second_block, 15, 15).noalias() += cross;
        hessian.block(second_block, first_block, 15, 15).noalias() += cross.transpose();
      }
    }
    if (used_factors == 0)
    {
      return changed;
    }
    hessian.diagonal().array() += 1e-8;
    const Eigen::VectorXd step = hessian.ldlt().solve(-gradient);
    if (!step.allFinite())
    {
      return changed;
    }
    double norm = step.norm();
    const double scale = norm > max_step_norm && norm > 1e-9 ? max_step_norm / norm : 1.0;
    for (std::size_t state_index = first_optimizable; state_index < states_.size(); ++state_index)
    {
      const int block = static_cast<int>(state_index - first_optimizable) * 15;
      Eigen::Matrix<double, 15, 1> state_step =
          (scale * step.segment(block, 15)).template cast<double>();
      boxPlus(states_[state_index], state_step);
      changed = true;
    }
    if (norm < 1e-5 || std::abs(total_error) < 1e-12)
    {
      break;
    }
  }
  if (changed && !states_.empty())
  {
    states_.back().covariance.diagonal().array() *= 0.98;
  }
  return changed;
}

void SlidingWindowOptimizer::marginalizeOldest()
{
  if (states_.empty())
  {
    return;
  }
  states_.erase(states_.begin());
  for (PoseState &state : states_)
  {
    --state.id;
  }
  std::vector<Factor> kept;
  kept.reserve(factors_.size());
  for (Factor factor : factors_)
  {
    if (factor.first == 0 || factor.second == 0)
    {
      continue;
    }
    --factor.first;
    if (factor.second >= 0)
    {
      --factor.second;
    }
    kept.push_back(factor);
  }
  factors_.swap(kept);
}

void BevGrid::reset(int cells_x, int cells_y, double cell_resolution, double center_x, double center_y)
{
  width = std::max(1, cells_x);
  height = std::max(1, cells_y);
  resolution = std::max(1e-3, cell_resolution);
  origin_x = center_x - 0.5 * width * resolution;
  origin_y = center_y - 0.5 * height * resolution;
  const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  occupancy.assign(count, 0.0F);
  height_min.assign(count, std::numeric_limits<float>::infinity());
  height_max.assign(count, -std::numeric_limits<float>::infinity());
  labels.assign(count, 0U);
  quality.assign(count, 0.0F);
}

bool BevGrid::worldToCell(double x, double y, int &ix, int &iy) const
{
  ix = static_cast<int>(std::floor((x - origin_x) / resolution));
  iy = static_cast<int>(std::floor((y - origin_y) / resolution));
  return ix >= 0 && ix < width && iy >= 0 && iy < height;
}

Eigen::Vector2d BevGrid::cellCenter(int ix, int iy) const
{
  return Eigen::Vector2d(origin_x + (static_cast<double>(ix) + 0.5) * resolution,
                         origin_y + (static_cast<double>(iy) + 0.5) * resolution);
}

void BevGrid::insert(const BevPoint &point, double ground_z, double max_height)
{
  int ix = 0;
  int iy = 0;
  if (!worldToCell(point.point.x(), point.point.y(), ix, iy) ||
      point.point.z() < ground_z || point.point.z() > max_height)
  {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
                            static_cast<std::size_t>(ix);
  occupancy[index] = std::min(1.0F, occupancy[index] + 0.20F * point.confidence);
  height_min[index] = std::min(height_min[index], static_cast<float>(point.point.z()));
  height_max[index] = std::max(height_max[index], static_cast<float>(point.point.z()));
  quality[index] = std::min(1.0F, quality[index] + 0.10F * point.confidence);
  if (point.confidence >= quality[index] || labels[index] == 0U)
  {
    labels[index] = point.label;
  }
}

bool BevGrid::occupiedAt(double x, double y) const
{
  int ix = 0;
  int iy = 0;
  if (!worldToCell(x, y, ix, iy))
  {
    return false;
  }
  return occupancy[static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(ix)] > 0.35F;
}

void PriorMap::clear()
{
  width = 0;
  height = 0;
  occupancy.clear();
  labels.clear();
  edge.clear();
}

bool PriorMap::valid() const
{
  return width > 0 && height > 0 && occupancy.size() ==
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

bool PriorMap::worldToCell(double x, double y, int &ix, int &iy) const
{
  ix = static_cast<int>(std::floor((x - origin_x) / resolution));
  iy = static_cast<int>(std::floor((y - origin_y) / resolution));
  return ix >= 0 && ix < width && iy >= 0 && iy < height;
}

Eigen::Vector2d PriorMap::cellCenter(int ix, int iy) const
{
  return Eigen::Vector2d(origin_x + (static_cast<double>(ix) + 0.5) * resolution,
                         origin_y + (static_cast<double>(iy) + 0.5) * resolution);
}

float PriorMap::occupancyAt(double x, double y) const
{
  int ix = 0;
  int iy = 0;
  if (!worldToCell(x, y, ix, iy))
  {
    return 0.0F;
  }
  return occupancy[static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(ix)];
}

uint8_t PriorMap::labelAt(double x, double y) const
{
  if (labels.size() != occupancy.size())
  {
    return 0U;
  }
  int ix = 0;
  int iy = 0;
  if (!worldToCell(x, y, ix, iy))
  {
    return 0U;
  }
  return labels[static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(ix)];
}

float PriorMap::edgeAt(double x, double y) const
{
  if (edge.size() != occupancy.size())
  {
    return 0.0F;
  }
  int ix = 0;
  int iy = 0;
  if (!worldToCell(x, y, ix, iy))
  {
    return 0.0F;
  }
  return edge[static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(ix)];
}

void PriorMap::recomputeEdges()
{
  edge.assign(occupancy.size(), 0.0F);
  if (!valid())
  {
    return;
  }
  for (int iy = 1; iy + 1 < height; ++iy)
  {
    for (int ix = 1; ix + 1 < width; ++ix)
    {
      const std::size_t index = static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
                                static_cast<std::size_t>(ix);
      const float dx = std::abs(occupancy[index + 1] - occupancy[index - 1]);
      const float dy = std::abs(occupancy[index + width] - occupancy[index - width]);
      edge[index] = std::min(1.0F, dx + dy);
    }
  }
}

PriorMatcher::PriorMatcher(const MatcherOptions &options) : options_(options)
{
}

std::vector<double> PriorMatcher::axisSamples(double center, double radius, double step)
{
  std::vector<double> result;
  const double safe_step = std::max(0.05, step);
  const int count = std::min(121, static_cast<int>(std::ceil(2.0 * radius / safe_step)) + 1);
  if (count <= 1)
  {
    result.push_back(center);
    return result;
  }
  const double actual_step = 2.0 * radius / static_cast<double>(count - 1);
  result.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i)
  {
    result.push_back(center - radius + actual_step * static_cast<double>(i));
  }
  return result;
}

double PriorMatcher::evaluate(
    const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points,
    const PriorMap &prior, const Eigen::Isometry3d &global_from_odom) const
{
  if (points.empty() || !prior.valid())
  {
    return -1.0;
  }
  double score = 0.0;
  int considered = 0;
  for (const BevPoint &point : points)
  {
    const Eigen::Vector3d transformed = global_from_odom * point.point;
    const float occupancy = prior.occupancyAt(transformed.x(), transformed.y());
    const float edge = prior.edgeAt(transformed.x(), transformed.y());
    if (occupancy <= 0.01F && edge <= 0.01F)
    {
      continue;
    }
    ++considered;
    double point_score = options_.occupancy_weight * occupancy + options_.edge_weight * edge;
    const uint8_t prior_label = prior.labelAt(transformed.x(), transformed.y());
    if (point.label != 0U && prior_label != 0U)
    {
      point_score += options_.semantic_weight * (point.label == prior_label ? 1.0 : -1.0);
    }
    if (point.dynamic)
    {
      point_score -= options_.dynamic_penalty;
    }
    score += point_score;
  }
  if (considered == 0)
  {
    return -1.0;
  }
  return score / static_cast<double>(points.size());
}

void PriorMatcher::buildPairs(
    const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points,
    const PriorMap &prior, const Eigen::Isometry3d &global_from_odom,
    MapMatchResult &result) const
{
  result.pairs.clear();
  const int search_cells = std::max(1, static_cast<int>(std::ceil(options_.match_distance / prior.resolution)));
  int index = 0;
  for (const BevPoint &point : points)
  {
    if (index++ >= options_.max_points)
    {
      break;
    }
    const Eigen::Vector3d transformed = global_from_odom * point.point;
    int center_x = 0;
    int center_y = 0;
    if (!prior.worldToCell(transformed.x(), transformed.y(), center_x, center_y))
    {
      continue;
    }
    double best_distance = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_target = Eigen::Vector2d::Zero();
    uint8_t target_label = 0U;
    for (int dy = -search_cells; dy <= search_cells; ++dy)
    {
      for (int dx = -search_cells; dx <= search_cells; ++dx)
      {
        const int ix = center_x + dx;
        const int iy = center_y + dy;
        if (ix < 0 || ix >= prior.width || iy < 0 || iy >= prior.height)
        {
          continue;
        }
        const Eigen::Vector2d target = prior.cellCenter(ix, iy);
        const double distance = (target - transformed.head<2>()).norm();
        const std::size_t cell = static_cast<std::size_t>(iy) * static_cast<std::size_t>(prior.width) +
                                 static_cast<std::size_t>(ix);
        if (prior.occupancy[cell] > 0.35F && distance < best_distance)
        {
          best_distance = distance;
          best_target = target;
          target_label = prior.labels.size() == prior.occupancy.size() ? prior.labels[cell] : 0U;
        }
      }
    }
    if (!std::isfinite(best_distance))
    {
      continue;
    }
    MatchPair pair;
    pair.source = point.point;
    pair.target = Eigen::Vector3d(best_target.x(), best_target.y(), transformed.z());
    pair.residual = best_distance;
    pair.source_label = point.label;
    pair.target_label = target_label;
    pair.candidate = true;
    pair.inlier = best_distance <= options_.inlier_distance;
    pair.outlier = !pair.inlier;
    result.pairs.push_back(pair);
  }
  if (!result.pairs.empty())
  {
    int inliers = 0;
    for (const MatchPair &pair : result.pairs)
    {
      if (pair.inlier) ++inliers;
    }
    result.inlier_ratio = static_cast<double>(inliers) / static_cast<double>(result.pairs.size());
  }
}

MapMatchResult PriorMatcher::match(
    const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &local_points,
    const PriorMap &prior, const Eigen::Isometry3d &predicted_global_from_odom,
    const Eigen::Matrix3d &position_covariance) const
{
  MapMatchResult result;
  result.global_from_odom = predicted_global_from_odom;
  if (local_points.empty())
  {
    result.reject_reason = "empty_local_bev";
    return result;
  }
  if (!prior.valid())
  {
    result.reject_reason = "prior_map_unavailable";
    return result;
  }

  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> points;
  const int stride = std::max(1, static_cast<int>(local_points.size()) / std::max(1, options_.max_points));
  for (std::size_t i = 0; i < local_points.size() && static_cast<int>(points.size()) < options_.max_points;
       i += static_cast<std::size_t>(stride))
  {
    points.push_back(local_points[i]);
  }
  const double covariance_scale = std::sqrt(std::max(0.01, position_covariance.topLeftCorner<2, 2>().trace() *
                                                               options_.confidence_gamma));
  const double radius = std::min(options_.max_search_radius,
                                 std::max(options_.min_search_radius, covariance_scale));
  result.search_radius = radius;
  const double center_x = predicted_global_from_odom.translation().x();
  const double center_y = predicted_global_from_odom.translation().y();
  const double base_yaw = yawOf(predicted_global_from_odom);

  auto evaluateCandidates = [&](double translation_radius, double translation_step,
                                double yaw_step_deg, double yaw_radius_deg,
                                double &best_x, double &best_y, double &best_yaw,
                                double &best_score, double &second_score) {
    const std::vector<double> xs = axisSamples(center_x, translation_radius, translation_step);
    const std::vector<double> ys = axisSamples(center_y, translation_radius, translation_step);
    const int yaw_count = std::max(1, static_cast<int>(std::ceil(2.0 * yaw_radius_deg / yaw_step_deg)) + 1);
    const double actual_yaw_step = 2.0 * yaw_radius_deg / static_cast<double>(std::max(1, yaw_count - 1));
    int candidates = 0;
    for (double yaw_delta_deg = -yaw_radius_deg; yaw_delta_deg <= yaw_radius_deg + 1e-6;
         yaw_delta_deg += actual_yaw_step)
    {
      const double candidate_yaw = base_yaw + yaw_delta_deg * kPi / 180.0;
      for (double x : xs)
      {
        for (double y : ys)
        {
          if (++candidates > options_.max_candidates)
          {
            return;
          }
          const Eigen::Matrix3d roll_pitch =
              rotationFromYaw(base_yaw).transpose() * predicted_global_from_odom.rotation();
          Eigen::Isometry3d candidate = Eigen::Isometry3d::Identity();
          candidate.linear() = rotationFromYaw(candidate_yaw) * roll_pitch;
          candidate.translation() = predicted_global_from_odom.translation();
          candidate.translation().x() = x;
          candidate.translation().y() = y;
          const double score = evaluate(points, prior, candidate);
          if (score > best_score)
          {
            second_score = best_score;
            best_score = score;
            best_x = x;
            best_y = y;
            best_yaw = candidate_yaw;
          }
          else if (score > second_score)
          {
            second_score = score;
          }
        }
      }
    }
  };

  double best_x = center_x;
  double best_y = center_y;
  double best_yaw = base_yaw;
  double best_score = -std::numeric_limits<double>::infinity();
  double second_score = -std::numeric_limits<double>::infinity();
  evaluateCandidates(radius, options_.coarse_translation_step, options_.coarse_yaw_step_deg,
                     options_.yaw_search_deg, best_x, best_y, best_yaw, best_score, second_score);
  const double fine_radius = std::max(options_.fine_translation_step * 2.0,
                                      options_.coarse_translation_step * 1.5);
  const double saved_center_x = center_x;
  const double saved_center_y = center_y;
  (void)saved_center_x;
  (void)saved_center_y;
  const std::vector<double> fine_xs = axisSamples(best_x, fine_radius, options_.fine_translation_step);
  const std::vector<double> fine_ys = axisSamples(best_y, fine_radius, options_.fine_translation_step);
  const int fine_yaw_count = std::max(1, static_cast<int>(std::ceil(10.0 / options_.fine_yaw_step_deg)) + 1);
  const double fine_yaw_step = 10.0 / static_cast<double>(std::max(1, fine_yaw_count - 1));
  for (int yi = 0; yi < fine_yaw_count; ++yi)
  {
    const double yaw_delta = (-5.0 + fine_yaw_step * static_cast<double>(yi)) * kPi / 180.0;
    const double candidate_yaw = best_yaw + yaw_delta;
    for (double x : fine_xs)
    {
      for (double y : fine_ys)
      {
        const Eigen::Matrix3d roll_pitch =
            rotationFromYaw(base_yaw).transpose() * predicted_global_from_odom.rotation();
        Eigen::Isometry3d candidate = Eigen::Isometry3d::Identity();
        candidate.linear() = rotationFromYaw(candidate_yaw) * roll_pitch;
        candidate.translation() = predicted_global_from_odom.translation();
        candidate.translation().x() = x;
        candidate.translation().y() = y;
        const double score = evaluate(points, prior, candidate);
        if (score > best_score)
        {
          second_score = best_score;
          best_score = score;
          best_x = x;
          best_y = y;
          best_yaw = candidate_yaw;
        }
        else if (score > second_score)
        {
          second_score = score;
        }
      }
    }
  }

  const Eigen::Matrix3d roll_pitch =
      rotationFromYaw(base_yaw).transpose() * predicted_global_from_odom.rotation();
  result.global_from_odom = Eigen::Isometry3d::Identity();
  result.global_from_odom.linear() = rotationFromYaw(best_yaw) * roll_pitch;
  result.global_from_odom.translation() = predicted_global_from_odom.translation();
  result.global_from_odom.translation().x() = best_x;
  result.global_from_odom.translation().y() = best_y;
  result.best_score = best_score;
  result.second_score = second_score;
  const double score_gap = std::isfinite(second_score) ? best_score - second_score : 0.0;
  const double gap_confidence = 1.0 / (1.0 + std::exp(-20.0 * (score_gap - options_.min_score_gap)));
  result.confidence = gap_confidence;
  buildPairs(points, prior, result.global_from_odom, result);
  result.valid = !result.pairs.empty() && best_score > -0.5;
  result.gate_passed = result.valid && result.inlier_ratio >= options_.min_inlier_ratio &&
                       result.confidence >= options_.min_confidence && score_gap >= options_.min_score_gap;
  if (!result.valid)
  {
    result.reject_reason = "no_geometric_overlap";
  }
  else if (result.inlier_ratio < options_.min_inlier_ratio)
  {
    result.reject_reason = "low_inlier_ratio";
  }
  else if (result.confidence < options_.min_confidence || score_gap < options_.min_score_gap)
  {
    result.reject_reason = "ambiguous_match";
  }
  if (result.gate_passed)
  {
    for (MatchPair &pair : result.pairs)
    {
      pair.applied = pair.inlier;
    }
  }
  return result;
}

}  // namespace hybrid_localization
