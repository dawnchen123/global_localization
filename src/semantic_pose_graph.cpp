#include "hybrid_localization/semantic_pose_graph.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace hybrid_localization
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double clampValue(double value, double minimum, double maximum)
{
  return std::max(minimum, std::min(maximum, value));
}

double wrapAngle(double angle)
{
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

double yawOf(const Eigen::Matrix3d &rotation)
{
  return std::atan2(rotation(1, 0), rotation(0, 0));
}

double yawOf(const Eigen::Isometry3d &pose)
{
  return yawOf(pose.rotation());
}

double rotationAngle(const Eigen::Matrix3d &rotation)
{
  return std::acos(clampValue(0.5 * (rotation.trace() - 1.0), -1.0, 1.0));
}

Eigen::Isometry3d projectToSE3(const Eigen::Isometry3d &pose)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(pose.rotation(), Eigen::ComputeFullU |
                                                        Eigen::ComputeFullV);
  Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
  if (rotation.determinant() < 0.0)
  {
    Eigen::Matrix3d u = svd.matrixU();
    u.col(2) *= -1.0;
    rotation = u * svd.matrixV().transpose();
  }
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = rotation;
  result.translation() = pose.translation();
  return result;
}

Eigen::Isometry3d planarDelta(double x, double y, double yaw)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  result.translation() = Eigen::Vector3d(x, y, 0.0);
  return result;
}

gtsam::Pose3 toGtsam(const Eigen::Isometry3d &pose)
{
  const Eigen::Isometry3d clean = projectToSE3(pose);
  return gtsam::Pose3(gtsam::Rot3(clean.rotation()),
                      gtsam::Point3(clean.translation().x(), clean.translation().y(),
                                    clean.translation().z()));
}

Eigen::Isometry3d fromGtsam(const gtsam::Pose3 &pose)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = pose.rotation().matrix();
  result.translation() = pose.translation();
  return projectToSE3(result);
}

gtsam::Key poseKey(int id)
{
  return gtsam::Symbol('x', static_cast<std::uint64_t>(id));
}

double median(std::vector<double> values)
{
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const std::size_t middle = values.size() / 2U;
  std::nth_element(values.begin(), values.begin() + static_cast<long>(middle), values.end());
  double result = values[middle];
  if (values.size() % 2U == 0U)
  {
    const auto lower = std::max_element(values.begin(), values.begin() + static_cast<long>(middle));
    result = 0.5 * (result + *lower);
  }
  return result;
}

double mad(const std::vector<double> &values, double center)
{
  std::vector<double> deviations;
  deviations.reserve(values.size());
  for (double value : values) deviations.push_back(std::abs(value - center));
  return median(deviations);
}

bool isGroundLabel(uint8_t label)
{
  return label == 1U || label == 2U;
}

bool isStructuralLabel(uint8_t label)
{
  return label == 3U || label == 4U || label == 6U;
}

bool semanticCompatible(uint8_t first, uint8_t second, bool use_semantics)
{
  if (!use_semantics || first == 0U || second == 0U) return true;
  if (isGroundLabel(first) && isGroundLabel(second)) return true;
  return first == second;
}

struct CellKey
{
  int x = 0;
  int y = 0;

  bool operator==(const CellKey &other) const
  {
    return x == other.x && y == other.y;
  }
};

struct CellKeyHash
{
  std::size_t operator()(const CellKey &key) const
  {
    const std::size_t x = std::hash<int>()(key.x);
    const std::size_t y = std::hash<int>()(key.y);
    return x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6U) + (x >> 2U));
  }
};

struct SemanticVoxelKey
{
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const SemanticVoxelKey &other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct SemanticVoxelKeyHash
{
  std::size_t operator()(const SemanticVoxelKey &key) const
  {
    const std::size_t xy = CellKeyHash()(CellKey{key.x, key.y});
    const std::size_t z = std::hash<int>()(key.z);
    return xy ^ (z + 0x9e3779b97f4a7c15ULL + (xy << 6U) + (xy >> 2U));
  }
};

struct SemanticMapCell
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  double weight = 0.0;
  std::array<double, 7> labels{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
};

struct CellStats
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  double weight = 0.0;
  double min_z = std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  int count = 0;
  std::array<double, 7> labels{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
};

struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  uint8_t label = 0U;
  bool structural = false;
  bool ground = false;
  double weight = 1.0;
};

using FeatureVector = std::vector<Feature, Eigen::aligned_allocator<Feature>>;

struct Descriptor
{
  std::vector<float> ring_key;
};

struct Frame
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stamp = 0.0;
  Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
  SemanticGraphPointVector points;
};

struct WheelObservation
{
  double stamp = 0.0;
  double speed = 0.0;
};

struct VisualObservation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stamp = 0.0;
  Eigen::Isometry3d visual_from_body_pose = Eigen::Isometry3d::Identity();
  int segment = 0;
  double quality = 0.0;
  bool metric_pose = false;
};

struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  double stamp = 0.0;
  Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d optimized_pose = Eigen::Isometry3d::Identity();
  FeatureVector features;
  Descriptor descriptor;
  bool has_semantics = false;
};

struct PairRecord
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int source_index = -1;
  int target_index = -1;
  Eigen::Vector3d source = Eigen::Vector3d::Zero();
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  double residual = std::numeric_limits<double>::infinity();
  DebugPairStage stage = DebugPairStage::Candidate;
};

using PairVector = std::vector<PairRecord, Eigen::aligned_allocator<PairRecord>>;

struct MatchResult
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  bool xy_accepted = false;
  bool z_accepted = false;
  Eigen::Isometry3d measurement = Eigen::Isometry3d::Identity();
  int xy_candidates = 0;
  int xy_inliers = 0;
  int z_candidates = 0;
  int z_inliers = 0;
  double xy_rmse = std::numeric_limits<double>::infinity();
  double xy_inlier_ratio = 0.0;
  double spread = 0.0;
  double spread_ratio = 0.0;
  double z_median = 0.0;
  double z_mad = std::numeric_limits<double>::infinity();
  double descriptor_similarity = 0.0;
  double score = -std::numeric_limits<double>::infinity();
  std::string reason = "not_matched";
  PairVector xy_pairs;
  PairVector z_pairs;
};

struct Candidate
{
  int id = -1;
  double similarity = 0.0;
  double distance = 0.0;
};

class SpatialIndex
{
public:
  SpatialIndex(const FeatureVector &features, bool structural, double resolution)
      : features_(features), structural_(structural), resolution_(std::max(0.05, resolution))
  {
    for (std::size_t i = 0; i < features_.size(); ++i)
    {
      if ((structural_ && !features_[i].structural) || (!structural_ && !features_[i].ground))
      {
        continue;
      }
      cells_[key(features_[i].point.head<2>())].push_back(static_cast<int>(i));
    }
  }

  int nearest(const Eigen::Vector2d &point, uint8_t label, double maximum_distance,
              bool use_semantics, double *distance = nullptr) const
  {
    const CellKey center = key(point);
    const int radius = static_cast<int>(std::ceil(maximum_distance / resolution_));
    int best = -1;
    double best_squared = maximum_distance * maximum_distance;
    for (int dx = -radius; dx <= radius; ++dx)
    {
      for (int dy = -radius; dy <= radius; ++dy)
      {
        const auto it = cells_.find(CellKey{center.x + dx, center.y + dy});
        if (it == cells_.end()) continue;
        for (int index : it->second)
        {
          const Feature &feature = features_[static_cast<std::size_t>(index)];
          if (!semanticCompatible(label, feature.label, use_semantics)) continue;
          const double squared = (feature.point.head<2>() - point).squaredNorm();
          if (squared < best_squared)
          {
            best_squared = squared;
            best = index;
          }
        }
      }
    }
    if (distance) *distance = best >= 0 ? std::sqrt(best_squared)
                                        : std::numeric_limits<double>::infinity();
    return best;
  }

private:
  CellKey key(const Eigen::Vector2d &point) const
  {
    return CellKey{static_cast<int>(std::floor(point.x() / resolution_)),
                   static_cast<int>(std::floor(point.y() / resolution_))};
  }

  const FeatureVector &features_;
  bool structural_ = true;
  double resolution_ = 1.0;
  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> cells_;
};

FeatureVector buildFeatures(const std::deque<Frame, Eigen::aligned_allocator<Frame>> &frames,
                            const Eigen::Isometry3d &keyframe_pose,
                            const SemanticPoseGraphOptions &options,
                            bool *has_semantics)
{
  using CellPair = std::pair<const CellKey, CellStats>;
  std::unordered_map<CellKey, CellStats, CellKeyHash, std::equal_to<CellKey>,
                     Eigen::aligned_allocator<CellPair>> cells;
  const Eigen::Isometry3d keyframe_from_world = keyframe_pose.inverse();
  *has_semantics = false;
  for (const Frame &frame : frames)
  {
    const Eigen::Isometry3d keyframe_from_frame = keyframe_from_world * frame.raw_pose;
    for (const SemanticGraphPoint &source : frame.points)
    {
      if (!source.point.allFinite() || source.label == 5U || source.confidence < 0.15F) continue;
      const Eigen::Vector3d point = keyframe_from_frame * source.point;
      const double range = point.head<2>().norm();
      if (!point.allFinite() || range < 1.0 || range > options.descriptor_max_radius) continue;
      const CellKey key{static_cast<int>(std::floor(point.x() / options.feature_resolution)),
                        static_cast<int>(std::floor(point.y() / options.feature_resolution))};
      CellStats &cell = cells[key];
      const double weight = clampValue(static_cast<double>(source.confidence), 0.15, 1.0);
      cell.sum += weight * point;
      cell.weight += weight;
      cell.min_z = std::min(cell.min_z, point.z());
      cell.max_z = std::max(cell.max_z, point.z());
      ++cell.count;
      if (source.label < cell.labels.size())
      {
        cell.labels[source.label] += weight;
        if (source.label != 0U) *has_semantics = true;
      }
    }
  }

  FeatureVector features;
  features.reserve(cells.size());
  for (const auto &entry : cells)
  {
    const CellStats &cell = entry.second;
    if (cell.count < options.feature_min_points || cell.weight <= 1e-6) continue;
    Feature feature;
    feature.point = cell.sum / cell.weight;
    feature.weight = std::sqrt(static_cast<double>(cell.count));
    const auto label_it = std::max_element(cell.labels.begin() + 1, cell.labels.end());
    if (label_it != cell.labels.end() && *label_it > 0.5)
    {
      feature.label = static_cast<uint8_t>(std::distance(cell.labels.begin(), label_it));
    }
    const double span = cell.max_z - cell.min_z;
    const bool semantic_ground = isGroundLabel(feature.label);
    const bool semantic_structure = isStructuralLabel(feature.label);
    feature.ground = (semantic_ground && cell.count >= options.ground_min_points) ||
        (cell.count >= options.ground_min_points && span <= options.ground_max_height_span &&
         feature.point.z() <= options.ground_max_height);
    feature.structural = semantic_structure || span >= options.structural_min_height_span ||
        (feature.point.z() >= options.structural_min_z && !feature.ground);
    if (feature.structural || feature.ground) features.push_back(feature);
  }

  if (features.size() > static_cast<std::size_t>(options.max_features_per_keyframe))
  {
    FeatureVector reduced;
    reduced.reserve(static_cast<std::size_t>(options.max_features_per_keyframe));
    const double stride = static_cast<double>(features.size()) /
                          static_cast<double>(options.max_features_per_keyframe);
    for (int i = 0; i < options.max_features_per_keyframe; ++i)
    {
      reduced.push_back(features[static_cast<std::size_t>(std::floor(i * stride))]);
    }
    features.swap(reduced);
  }
  return features;
}

Descriptor buildDescriptor(const FeatureVector &features,
                           const SemanticPoseGraphOptions &options)
{
  const int rings = std::max(1, options.descriptor_rings);
  Descriptor descriptor;
  descriptor.ring_key.assign(static_cast<std::size_t>(rings * 3), 0.0F);
  std::vector<float> counts(static_cast<std::size_t>(rings), 0.0F);
  for (const Feature &feature : features)
  {
    const double range = feature.point.head<2>().norm();
    if (range >= options.descriptor_max_radius) continue;
    const int ring = std::min(rings - 1, static_cast<int>(
        std::floor(range / options.descriptor_max_radius * rings)));
    counts[static_cast<std::size_t>(ring)] += 1.0F;
    if (feature.structural)
    {
      descriptor.ring_key[static_cast<std::size_t>(ring)] += 1.0F;
    }
    if (feature.ground)
    {
      descriptor.ring_key[static_cast<std::size_t>(rings + ring)] += 1.0F;
    }
    if (feature.label != 0U)
    {
      descriptor.ring_key[static_cast<std::size_t>(2 * rings + ring)] +=
          static_cast<float>(options.semantic_weight);
    }
  }
  for (int ring = 0; ring < rings; ++ring)
  {
    const float denominator = std::max(1.0F, counts[static_cast<std::size_t>(ring)]);
    for (int channel = 0; channel < 3; ++channel)
    {
      descriptor.ring_key[static_cast<std::size_t>(channel * rings + ring)] /= denominator;
    }
  }
  return descriptor;
}

Keyframe semanticOnlyKeyframe(const Keyframe &source,
                              const SemanticPoseGraphOptions &options)
{
  Keyframe result = source;
  result.features.clear();
  result.features.reserve(source.features.size());
  for (const Feature &feature : source.features)
  {
    if (feature.label != 0U) result.features.push_back(feature);
  }
  result.has_semantics = !result.features.empty();
  result.descriptor = buildDescriptor(result.features, options);
  return result;
}

double descriptorSimilarity(const Descriptor &first, const Descriptor &second)
{
  if (first.ring_key.size() != second.ring_key.size() || first.ring_key.empty()) return 0.0;
  double dot = 0.0;
  double norm_first = 0.0;
  double norm_second = 0.0;
  for (std::size_t i = 0; i < first.ring_key.size(); ++i)
  {
    dot += static_cast<double>(first.ring_key[i]) * second.ring_key[i];
    norm_first += static_cast<double>(first.ring_key[i]) * first.ring_key[i];
    norm_second += static_cast<double>(second.ring_key[i]) * second.ring_key[i];
  }
  if (norm_first <= 1e-9 || norm_second <= 1e-9) return 0.0;
  return dot / std::sqrt(norm_first * norm_second);
}

FeatureVector transformedFeatures(const FeatureVector &source, const Eigen::Isometry3d &transform,
                                  bool structural)
{
  FeatureVector result = source;
  for (Feature &feature : result)
  {
    if ((structural && feature.structural) || (!structural && feature.ground))
    {
      feature.point = transform * feature.point;
    }
  }
  return result;
}

PairVector mutualNearestPairs(const FeatureVector &target, const FeatureVector &source,
                              const Eigen::Isometry3d &target_from_source,
                              bool structural, double maximum_distance,
                              bool use_semantics)
{
  const FeatureVector transformed = transformedFeatures(source, target_from_source, structural);
  const SpatialIndex target_index(target, structural, maximum_distance);
  const SpatialIndex source_index(transformed, structural, maximum_distance);
  std::vector<int> source_to_target(source.size(), -1);
  std::vector<int> target_to_source(target.size(), -1);
  for (std::size_t i = 0; i < transformed.size(); ++i)
  {
    if ((structural && !transformed[i].structural) || (!structural && !transformed[i].ground)) continue;
    source_to_target[i] = target_index.nearest(transformed[i].point.head<2>(), transformed[i].label,
                                               maximum_distance, use_semantics);
  }
  for (std::size_t i = 0; i < target.size(); ++i)
  {
    if ((structural && !target[i].structural) || (!structural && !target[i].ground)) continue;
    target_to_source[i] = source_index.nearest(target[i].point.head<2>(), target[i].label,
                                               maximum_distance, use_semantics);
  }
  PairVector pairs;
  for (std::size_t source_id = 0; source_id < source_to_target.size(); ++source_id)
  {
    const int target_id = source_to_target[source_id];
    if (target_id < 0 || target_to_source[static_cast<std::size_t>(target_id)] !=
                             static_cast<int>(source_id))
    {
      continue;
    }
    PairRecord pair;
    pair.source_index = static_cast<int>(source_id);
    pair.target_index = target_id;
    pair.source = source[source_id].point;
    pair.target = target[static_cast<std::size_t>(target_id)].point;
    pair.residual = (transformed[source_id].point.head<2>() - pair.target.head<2>()).norm();
    pairs.push_back(pair);
  }
  return pairs;
}

Eigen::Isometry3d fitPlanarTransform(const PairVector &pairs,
                                    const std::vector<double> &weights,
                                    const Eigen::Isometry3d &fallback)
{
  if (pairs.size() < 2U || weights.size() != pairs.size()) return fallback;
  double sum_weight = 0.0;
  Eigen::Vector2d source_center = Eigen::Vector2d::Zero();
  Eigen::Vector2d target_center = Eigen::Vector2d::Zero();
  for (std::size_t i = 0; i < pairs.size(); ++i)
  {
    const double weight = std::max(0.0, weights[i]);
    sum_weight += weight;
    source_center += weight * pairs[i].source.head<2>();
    target_center += weight * pairs[i].target.head<2>();
  }
  if (sum_weight <= 1e-9) return fallback;
  source_center /= sum_weight;
  target_center /= sum_weight;
  Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
  for (std::size_t i = 0; i < pairs.size(); ++i)
  {
    covariance += weights[i] * (pairs[i].source.head<2>() - source_center) *
                  (pairs[i].target.head<2>() - target_center).transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix2d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d rotation = svd.matrixV() * svd.matrixU().transpose();
  if (rotation.determinant() < 0.0)
  {
    Eigen::Matrix2d v = svd.matrixV();
    v.col(1) *= -1.0;
    rotation = v * svd.matrixU().transpose();
  }
  const Eigen::Vector2d translation = target_center - rotation * source_center;
  Eigen::Isometry3d result = fallback;
  const double fitted_yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  const double current_yaw = yawOf(fallback);
  result.linear() = Eigen::AngleAxisd(wrapAngle(fitted_yaw - current_yaw),
                                      Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                    fallback.rotation();
  result.translation().x() = translation.x();
  result.translation().y() = translation.y();
  return projectToSE3(result);
}

bool transformWithinCorrection(const Eigen::Isometry3d &estimate,
                               const Eigen::Isometry3d &raw,
                               const SemanticPoseGraphOptions &options)
{
  const Eigen::Isometry3d delta = estimate * raw.inverse();
  return delta.translation().head<2>().norm() <= options.max_xy_correction &&
      std::abs(wrapAngle(yawOf(estimate) - yawOf(raw))) <=
          options.max_yaw_correction_deg * kPi / 180.0;
}

Eigen::Isometry3d ransacAndHuber(const PairVector &candidates,
                                const Eigen::Isometry3d &initial,
                                const Eigen::Isometry3d &raw,
                                const SemanticPoseGraphOptions &options,
                                int seed)
{
  if (candidates.size() < 2U) return initial;
  Eigen::Isometry3d best = initial;
  int best_inliers = 0;
  double best_error = std::numeric_limits<double>::infinity();
  std::mt19937 generator(static_cast<std::uint32_t>(seed));
  std::uniform_int_distribution<int> distribution(0, static_cast<int>(candidates.size()) - 1);
  for (int iteration = 0; iteration < options.ransac_iterations; ++iteration)
  {
    int first = distribution(generator);
    int second = distribution(generator);
    if (first == second) continue;
    const Eigen::Vector2d source_delta = candidates[static_cast<std::size_t>(second)].source.head<2>() -
                                         candidates[static_cast<std::size_t>(first)].source.head<2>();
    const Eigen::Vector2d target_delta = candidates[static_cast<std::size_t>(second)].target.head<2>() -
                                         candidates[static_cast<std::size_t>(first)].target.head<2>();
    if (source_delta.norm() < 2.0 || target_delta.norm() < 2.0) continue;
    const double yaw = std::atan2(target_delta.y(), target_delta.x()) -
                       std::atan2(source_delta.y(), source_delta.x());
    const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(yaw).toRotationMatrix();
    const Eigen::Vector2d translation = candidates[static_cast<std::size_t>(first)].target.head<2>() -
                                        rotation * candidates[static_cast<std::size_t>(first)].source.head<2>();
    Eigen::Isometry3d model = raw;
    const double raw_yaw = yawOf(raw);
    model.linear() = Eigen::AngleAxisd(wrapAngle(yaw - raw_yaw), Eigen::Vector3d::UnitZ())
                         .toRotationMatrix() * raw.rotation();
    model.translation().x() = translation.x();
    model.translation().y() = translation.y();
    if (!transformWithinCorrection(model, raw, options)) continue;
    int inliers = 0;
    double squared_error = 0.0;
    for (const PairRecord &pair : candidates)
    {
      const double residual = (model * pair.source - pair.target).head<2>().norm();
      if (residual <= options.ransac_inlier_distance)
      {
        ++inliers;
        squared_error += residual * residual;
      }
    }
    if (inliers > best_inliers || (inliers == best_inliers && squared_error < best_error))
    {
      best = model;
      best_inliers = inliers;
      best_error = squared_error;
    }
  }

  for (int iteration = 0; iteration < options.huber_iterations; ++iteration)
  {
    std::vector<double> weights(candidates.size(), 0.0);
    int active = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
      const double residual = (best * candidates[i].source - candidates[i].target).head<2>().norm();
      if (residual <= options.correspondence_distance)
      {
        weights[i] = residual <= options.huber_delta ? 1.0 : options.huber_delta / residual;
        ++active;
      }
    }
    if (active < 2) break;
    const Eigen::Isometry3d updated = fitPlanarTransform(candidates, weights, best);
    if (!transformWithinCorrection(updated, raw, options)) break;
    const double translation_change = (updated.translation() - best.translation()).head<2>().norm();
    const double yaw_change = std::abs(wrapAngle(yawOf(updated) - yawOf(best)));
    best = updated;
    if (translation_change < 1e-3 && yaw_change < 1e-4) break;
  }
  return best;
}

double coarseScore(const FeatureVector &target, const FeatureVector &source,
                   const SpatialIndex &index, const Eigen::Isometry3d &transform,
                   const SemanticPoseGraphOptions &options)
{
  int tested = 0;
  double score = 0.0;
  const int structural_count = static_cast<int>(std::count_if(source.begin(), source.end(),
      [](const Feature &feature) { return feature.structural; }));
  const int stride = std::max(1, structural_count / std::max(1, options.coarse_max_points));
  int structural_index = 0;
  for (const Feature &feature : source)
  {
    if (!feature.structural) continue;
    if (structural_index++ % stride != 0) continue;
    const Eigen::Vector3d point = transform * feature.point;
    double distance = 0.0;
    const int nearest = index.nearest(point.head<2>(), feature.label,
                                      options.correspondence_distance,
                                      options.use_semantics, &distance);
    ++tested;
    if (nearest >= 0)
    {
      const bool semantic = feature.label != 0U &&
                            target[static_cast<std::size_t>(nearest)].label == feature.label;
      score += std::max(0.0, 1.0 - distance / options.correspondence_distance) *
               (1.0 + (semantic ? options.semantic_weight : 0.0));
    }
  }
  return tested > 0 ? score : 0.0;
}

Eigen::Isometry3d coarseAlign(const FeatureVector &target, const FeatureVector &source,
                             const Eigen::Isometry3d &raw,
                             const SemanticPoseGraphOptions &options,
                             double *best_score)
{
  const SpatialIndex index(target, true, options.correspondence_distance);
  Eigen::Isometry3d best = raw;
  *best_score = coarseScore(target, source, index, raw, options);
  const double yaw_radius = options.coarse_yaw_radius_deg * kPi / 180.0;
  const double yaw_step = std::max(0.1, options.coarse_yaw_step_deg) * kPi / 180.0;
  for (double yaw = -yaw_radius; yaw <= yaw_radius + 1e-9; yaw += yaw_step)
  {
    for (double x = -options.coarse_xy_radius; x <= options.coarse_xy_radius + 1e-9;
         x += options.coarse_xy_step)
    {
      for (double y = -options.coarse_xy_radius; y <= options.coarse_xy_radius + 1e-9;
           y += options.coarse_xy_step)
      {
        const Eigen::Isometry3d candidate = planarDelta(x, y, yaw) * raw;
        const double score = coarseScore(target, source, index, candidate, options);
        if (score > *best_score)
        {
          *best_score = score;
          best = candidate;
        }
      }
    }
  }

  const Eigen::Isometry3d coarse = best;
  const double fine_xy_step = std::max(0.15, options.coarse_xy_step / 3.0);
  const double fine_yaw_step = std::max(0.25, options.coarse_yaw_step_deg / 3.0) * kPi / 180.0;
  for (double yaw = -yaw_step; yaw <= yaw_step + 1e-9; yaw += fine_yaw_step)
  {
    for (double x = -options.coarse_xy_step; x <= options.coarse_xy_step + 1e-9;
         x += fine_xy_step)
    {
      for (double y = -options.coarse_xy_step; y <= options.coarse_xy_step + 1e-9;
           y += fine_xy_step)
      {
        const Eigen::Isometry3d candidate = planarDelta(x, y, yaw) * coarse;
        const double score = coarseScore(target, source, index, candidate, options);
        if (score > *best_score)
        {
          *best_score = score;
          best = candidate;
        }
      }
    }
  }
  return best;
}

void computeSpread(const PairVector &pairs, const Eigen::Isometry3d &transform,
                   double *spread, double *spread_ratio)
{
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  int count = 0;
  for (const PairRecord &pair : pairs)
  {
    if (pair.stage == DebugPairStage::Outlier) continue;
    mean += (transform * pair.source).head<2>();
    ++count;
  }
  if (count < 2)
  {
    *spread = 0.0;
    *spread_ratio = 0.0;
    return;
  }
  mean /= static_cast<double>(count);
  Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
  for (const PairRecord &pair : pairs)
  {
    if (pair.stage == DebugPairStage::Outlier) continue;
    const Eigen::Vector2d delta = (transform * pair.source).head<2>() - mean;
    covariance += delta * delta.transpose();
  }
  covariance /= static_cast<double>(count);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
  if (solver.info() != Eigen::Success)
  {
    *spread = 0.0;
    *spread_ratio = 0.0;
    return;
  }
  const double small = std::max(0.0, solver.eigenvalues().x());
  const double large = std::max(1e-9, solver.eigenvalues().y());
  *spread = std::sqrt(small + large);
  *spread_ratio = std::sqrt(small / large);
}

void evaluateZ(const Keyframe &reference, const Keyframe &current,
               const Eigen::Isometry3d &xy_measurement,
               const SemanticPoseGraphOptions &options,
               MatchResult *result)
{
  PairVector pairs = mutualNearestPairs(reference.features, current.features, xy_measurement,
                                        false, options.z_correspondence_distance,
                                        options.use_semantics);
  result->z_candidates = static_cast<int>(pairs.size());
  std::vector<double> broad_residuals;
  broad_residuals.reserve(pairs.size());
  for (PairRecord &pair : pairs)
  {
    const Eigen::Vector3d transformed = xy_measurement * pair.source;
    pair.residual = pair.target.z() - transformed.z();
    if (std::abs(pair.residual) <= options.z_candidate_residual_gate)
    {
      broad_residuals.push_back(pair.residual);
      pair.stage = DebugPairStage::Candidate;
    }
    else
    {
      pair.stage = DebugPairStage::Outlier;
    }
  }
  if (broad_residuals.empty())
  {
    result->z_pairs = pairs;
    return;
  }
  const double center = median(broad_residuals);
  std::vector<double> inlier_residuals;
  for (PairRecord &pair : pairs)
  {
    if (pair.stage == DebugPairStage::Outlier) continue;
    if (std::abs(pair.residual - center) <= options.z_inlier_residual_gate)
    {
      pair.stage = DebugPairStage::Inlier;
      inlier_residuals.push_back(pair.residual);
    }
    else
    {
      pair.stage = DebugPairStage::Outlier;
    }
  }
  result->z_inliers = static_cast<int>(inlier_residuals.size());
  result->z_median = inlier_residuals.empty() ? center : median(inlier_residuals);
  result->z_mad = inlier_residuals.empty() ? std::numeric_limits<double>::infinity()
                                           : mad(inlier_residuals, result->z_median);
  result->z_accepted = result->z_inliers >= options.min_z_inliers &&
      std::abs(result->z_median) <= options.max_z_correction &&
      result->z_mad <= options.max_z_mad;
  result->z_pairs = pairs;
}

MatchResult matchKeyframes(const Keyframe &reference, const Keyframe &current,
                           const SemanticPoseGraphOptions &options)
{
  MatchResult result;
  result.valid = true;
  result.measurement = reference.raw_pose.inverse() * current.raw_pose;
  result.descriptor_similarity = descriptorSimilarity(reference.descriptor, current.descriptor);
  const Eigen::Isometry3d raw = result.measurement;
  double coarse_score = 0.0;
  Eigen::Isometry3d estimate = coarseAlign(reference.features, current.features, raw,
                                           options, &coarse_score);
  PairVector initial_pairs = mutualNearestPairs(reference.features, current.features, estimate,
                                                true, options.correspondence_distance,
                                                options.use_semantics);
  result.xy_candidates = static_cast<int>(initial_pairs.size());
  if (result.xy_candidates >= options.coarse_min_inliers)
  {
    estimate = ransacAndHuber(initial_pairs, estimate, raw, options,
                              7919 + 101 * reference.id + current.id);
  }
  PairVector final_pairs = mutualNearestPairs(reference.features, current.features, estimate,
                                              true, options.correspondence_distance,
                                              options.use_semantics);
  double squared_error = 0.0;
  int inliers = 0;
  for (PairRecord &pair : final_pairs)
  {
    pair.residual = (estimate * pair.source - pair.target).head<2>().norm();
    if (pair.residual <= options.ransac_inlier_distance)
    {
      pair.stage = DebugPairStage::Inlier;
      squared_error += pair.residual * pair.residual;
      ++inliers;
    }
    else
    {
      pair.stage = DebugPairStage::Outlier;
    }
  }
  result.xy_inliers = inliers;
  result.xy_inlier_ratio = final_pairs.empty() ? 0.0 :
      static_cast<double>(inliers) / static_cast<double>(final_pairs.size());
  result.xy_rmse = inliers > 0 ? std::sqrt(squared_error / static_cast<double>(inliers))
                               : std::numeric_limits<double>::infinity();
  computeSpread(final_pairs, estimate, &result.spread, &result.spread_ratio);
  const Eigen::Isometry3d correction = estimate * raw.inverse();
  const double correction_xy = correction.translation().head<2>().norm();
  const double correction_yaw = std::abs(wrapAngle(yawOf(estimate) - yawOf(raw)));
  result.xy_accepted = options.enable_xy_loops &&
      result.xy_inliers >= options.min_xy_inliers &&
      result.xy_inlier_ratio >= options.min_xy_inlier_ratio &&
      result.xy_rmse <= options.max_xy_rmse &&
      result.spread >= options.min_xy_spread &&
      result.spread_ratio >= options.min_xy_spread_ratio &&
      correction_xy <= options.max_xy_correction &&
      correction_yaw <= options.max_yaw_correction_deg * kPi / 180.0;
  result.measurement = estimate;
  result.xy_pairs = final_pairs;

  if (options.enable_z_loops)
  {
    evaluateZ(reference, current, estimate, options, &result);
    if (result.z_accepted) result.measurement.translation().z() += result.z_median;
  }
  result.score = 2.0 * static_cast<double>(result.xy_inliers) -
      180.0 * std::min(2.0, result.xy_rmse) +
      0.8 * static_cast<double>(result.z_inliers) +
      80.0 * result.descriptor_similarity + 0.05 * coarse_score;
  if (result.xy_accepted || result.z_accepted)
  {
    result.reason = "measurement_gate_passed";
  }
  else if (result.xy_candidates < options.coarse_min_inliers)
  {
    result.reason = "insufficient_mutual_candidates";
  }
  else if (result.xy_inliers < options.min_xy_inliers)
  {
    result.reason = "insufficient_ransac_inliers";
  }
  else if (result.xy_rmse > options.max_xy_rmse)
  {
    result.reason = "xy_rmse_gate";
  }
  else
  {
    result.reason = "geometric_gate";
  }
  return result;
}

gtsam::SharedNoiseModel robustDiagonal(const gtsam::Vector6 &sigmas, double huber_k)
{
  const auto diagonal = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  if (huber_k <= 0.0) return diagonal;
  return gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(huber_k), diagonal);
}

Eigen::Isometry3d interpolateTransform(const Eigen::Isometry3d &first,
                                      const Eigen::Isometry3d &second, double ratio)
{
  ratio = clampValue(ratio, 0.0, 1.0);
  Eigen::Quaterniond first_q(first.rotation());
  Eigen::Quaterniond second_q(second.rotation());
  if (first_q.dot(second_q) < 0.0) second_q.coeffs() *= -1.0;
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = first_q.slerp(ratio, second_q).normalized().toRotationMatrix();
  result.translation() = (1.0 - ratio) * first.translation() + ratio * second.translation();
  return projectToSE3(result);
}

}  // namespace

struct SemanticPoseGraph::Impl
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit Impl(const SemanticPoseGraphOptions &input_options) : options(input_options)
  {
    initializeIsam();
  }

  void initializeIsam()
  {
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = options.isam_relinearize_threshold;
    parameters.relinearizeSkip = std::max(1, options.isam_relinearize_skip);
    isam.reset(new gtsam::ISAM2(parameters));
  }

  void updateOptimizedPoses()
  {
    if (keyframes.empty()) return;
    const gtsam::Values estimate = isam->calculateEstimate();
    for (Keyframe &keyframe : keyframes)
    {
      if (estimate.exists(poseKey(keyframe.id)))
      {
        keyframe.optimized_pose = fromGtsam(estimate.at<gtsam::Pose3>(poseKey(keyframe.id)));
      }
    }
  }

  Eigen::Isometry3d effectivePose(const Keyframe &keyframe) const
  {
    if (stats.wheel_factors > 0 || stats.visual_rotation_factors > 0 ||
        stats.visual_loop_factors > 0 ||
        stats.xy_loop_factors >= options.min_loops_for_xy_output ||
        stats.semantic_observation_xy_factors >=
            options.min_semantic_observation_factors_for_xy_output)
    {
      return keyframe.optimized_pose;
    }
    Eigen::Isometry3d result = keyframe.raw_pose;
    result.translation().z() = keyframe.optimized_pose.translation().z();
    return result;
  }

  const Keyframe *nearestKeyframe(double stamp) const
  {
    if (keyframes.empty() || !std::isfinite(stamp)) return nullptr;
    const auto nearest = std::min_element(
        keyframes.begin(), keyframes.end(), [stamp](const Keyframe &first,
                                                    const Keyframe &second)
        {
          return std::abs(first.stamp - stamp) < std::abs(second.stamp - stamp);
        });
    if (nearest == keyframes.end() ||
        std::abs(nearest->stamp - stamp) > options.visual_loop_max_time_offset)
    {
      return nullptr;
    }
    return &*nearest;
  }

  bool addVisualLoopConstraint(double reference_stamp, double current_stamp,
                               const Eigen::Isometry3d &reference_from_current,
                               double quality)
  {
    if (!options.enable_visual_loop_factors) return false;
    ++stats.visual_loop_attempts;
    if (!std::isfinite(reference_stamp) || !std::isfinite(current_stamp) ||
        !std::isfinite(quality) || !reference_from_current.matrix().allFinite() ||
        quality < options.visual_loop_min_quality ||
        current_stamp <= reference_stamp)
    {
      ++stats.visual_loop_rejections;
      return false;
    }
    const Keyframe *reference = nearestKeyframe(reference_stamp);
    const Keyframe *current = nearestKeyframe(current_stamp);
    if (reference == nullptr || current == nullptr ||
        current->id - reference->id < options.visual_loop_min_index_gap)
    {
      ++stats.visual_loop_rejections;
      return false;
    }
    const std::uint64_t pair_key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(reference->id)) << 32U) |
        static_cast<std::uint32_t>(current->id);
    if (!visual_loop_pairs.insert(pair_key).second)
    {
      ++stats.visual_loop_rejections;
      return false;
    }

    const Eigen::Isometry3d measurement = projectToSE3(reference_from_current);
    const Eigen::Isometry3d raw_relative =
        projectToSE3(reference->raw_pose.inverse() * current->raw_pose);
    const Eigen::Isometry3d disagreement =
        projectToSE3(raw_relative.inverse() * measurement);
    const double translation_disagreement = disagreement.translation().norm();
    const double rotation_disagreement = rotationAngle(disagreement.rotation());
    if (!std::isfinite(translation_disagreement) ||
        !std::isfinite(rotation_disagreement) ||
        translation_disagreement >
            options.visual_loop_max_translation_disagreement ||
        rotation_disagreement >
            options.visual_loop_max_rotation_disagreement_deg * kPi / 180.0)
    {
      visual_loop_pairs.erase(pair_key);
      ++stats.visual_loop_rejections;
      return false;
    }

    const double quality_scale = 1.0 + options.visual_loop_quality_sigma_scale *
        (1.0 - clampValue(quality, 0.0, 1.0));
    gtsam::Vector6 sigmas;
    sigmas << options.visual_loop_sigma_roll_pitch_deg * quality_scale * kPi / 180.0,
              options.visual_loop_sigma_roll_pitch_deg * quality_scale * kPi / 180.0,
              options.visual_loop_sigma_yaw_deg * quality_scale * kPi / 180.0,
              options.visual_loop_sigma_xy * quality_scale,
              options.visual_loop_sigma_xy * quality_scale,
              options.visual_loop_sigma_z * quality_scale;
    gtsam::NonlinearFactorGraph factors;
    factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
        poseKey(reference->id), poseKey(current->id), toGtsam(measurement),
        robustDiagonal(sigmas, options.visual_loop_huber_k)));
    isam->update(factors, gtsam::Values());
    isam->update();
    updateOptimizedPoses();
    ++stats.visual_loop_factors;
    return true;
  }

  SemanticGraphPointVector buildSemanticMap(double voxel_size,
                                             int maximum_points) const
  {
    SemanticGraphPointVector output;
    if (semantic_observations.empty() || maximum_points <= 0) return output;
    voxel_size = std::max(0.05, voxel_size);
    std::size_t total = 0U;
    for (const Keyframe &observation : semantic_observations)
    {
      total += static_cast<std::size_t>(std::count_if(
          observation.features.begin(), observation.features.end(),
          [](const Feature &feature) { return feature.label != 0U; }));
    }
    if (total == 0U) return output;
    const std::size_t limit = static_cast<std::size_t>(std::max(1, maximum_points));
    const std::size_t stride = std::max<std::size_t>(
        1U, (total + limit - 1U) / limit);
    using VoxelPair = std::pair<const SemanticVoxelKey, SemanticMapCell>;
    std::unordered_map<SemanticVoxelKey, SemanticMapCell, SemanticVoxelKeyHash,
                       std::equal_to<SemanticVoxelKey>,
                       Eigen::aligned_allocator<VoxelPair>> voxels;
    voxels.reserve(std::min(total / stride, limit));
    std::size_t semantic_index = 0U;
    for (const Keyframe &observation : semantic_observations)
    {
      if (observation.id < 0 || observation.id >= static_cast<int>(keyframes.size())) continue;
      const Eigen::Isometry3d world_from_keyframe =
          effectivePose(keyframes[static_cast<std::size_t>(observation.id)]);
      for (const Feature &feature : observation.features)
      {
        if (feature.label == 0U) continue;
        if (semantic_index++ % stride != 0U) continue;
        const Eigen::Vector3d point = world_from_keyframe * feature.point;
        if (!point.allFinite()) continue;
        const SemanticVoxelKey key{
            static_cast<int>(std::floor(point.x() / voxel_size)),
            static_cast<int>(std::floor(point.y() / voxel_size)),
            static_cast<int>(std::floor(point.z() / voxel_size))};
        SemanticMapCell &cell = voxels[key];
        const double weight = std::max(0.1, feature.weight);
        cell.sum += weight * point;
        cell.weight += weight;
        if (feature.label < cell.labels.size()) cell.labels[feature.label] += weight;
      }
    }
    output.reserve(std::min(voxels.size(), limit));
    for (const auto &entry : voxels)
    {
      const SemanticMapCell &cell = entry.second;
      if (cell.weight <= 1e-9) continue;
      const auto label = std::max_element(cell.labels.begin() + 1, cell.labels.end());
      if (label == cell.labels.end() || *label <= 0.0) continue;
      SemanticGraphPoint point;
      point.point = cell.sum / cell.weight;
      point.label = static_cast<uint8_t>(std::distance(cell.labels.begin(), label));
      point.confidence = static_cast<float>(clampValue(*label / cell.weight, 0.0, 1.0));
      output.push_back(point);
      if (output.size() >= limit) break;
    }
    return output;
  }

  std::vector<Candidate> findCandidates(const Keyframe &current) const
  {
    std::vector<Candidate> candidates;
    const int maximum_id = current.id - std::max(1, options.loop_min_index_gap);
    for (int id = 0; id <= maximum_id; ++id)
    {
      const Keyframe &reference = keyframes[static_cast<std::size_t>(id)];
      const double distance = (reference.raw_pose.translation() -
                               current.raw_pose.translation()).head<2>().norm();
      if (distance > options.loop_search_radius) continue;
      const double yaw_difference = std::abs(wrapAngle(yawOf(reference.raw_pose) -
                                                        yawOf(current.raw_pose)));
      if (yaw_difference > options.loop_max_yaw_difference_deg * kPi / 180.0) continue;
      const double similarity = descriptorSimilarity(reference.descriptor, current.descriptor);
      if (similarity < options.descriptor_min_similarity) continue;
      candidates.push_back(Candidate{id, similarity, distance});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &first,
                                                       const Candidate &second)
    {
      if (std::abs(first.similarity - second.similarity) > 1e-6)
      {
        return first.similarity > second.similarity;
      }
      return first.distance < second.distance;
    });

    std::vector<Candidate> suppressed;
    for (const Candidate &candidate : candidates)
    {
      bool same_neighborhood = false;
      for (const Candidate &kept : suppressed)
      {
        if (std::abs(candidate.id - kept.id) < 8)
        {
          same_neighborhood = true;
          break;
        }
      }
      if (!same_neighborhood) suppressed.push_back(candidate);
      if (static_cast<int>(suppressed.size()) >= options.loop_max_candidates) break;
    }
    return suppressed;
  }

  bool graphConsistency(const Keyframe &reference, const Keyframe &current,
                        MatchResult *match) const
  {
    const Eigen::Isometry3d predicted = reference.optimized_pose.inverse() *
                                        current.optimized_pose;
    const double xy = (match->measurement.translation() -
                       predicted.translation()).head<2>().norm();
    const double yaw = std::abs(wrapAngle(yawOf(match->measurement) - yawOf(predicted)));
    const double z = std::abs(match->measurement.translation().z() -
                              predicted.translation().z());
    if (xy > options.graph_consistency_max_xy)
    {
      match->xy_accepted = false;
      match->reason = "graph_xy_consistency_gate";
    }
    if (yaw > options.graph_consistency_max_yaw_deg * kPi / 180.0)
    {
      match->xy_accepted = false;
      match->reason = "graph_yaw_consistency_gate";
    }
    if (z > options.graph_consistency_max_z)
    {
      match->z_accepted = false;
      if (!match->xy_accepted) match->reason = "graph_z_consistency_gate";
    }
    return match->xy_accepted || match->z_accepted;
  }

  void populateDebug(const Keyframe &reference, const Keyframe &current,
                     const MatchResult &match, bool applied,
                     SemanticLoopDebug *destination = nullptr)
  {
    SemanticLoopDebug &output = destination != nullptr ? *destination : debug;
    output = SemanticLoopDebug();
    output.valid = true;
    output.accepted = applied;
    output.xy_accepted = match.xy_accepted && applied;
    output.z_accepted = match.z_accepted && applied;
    output.reference_id = reference.id;
    output.current_id = current.id;
    output.descriptor_similarity = match.descriptor_similarity;
    output.xy_rmse = match.xy_rmse;
    output.z_median = match.z_median;
    output.z_mad = match.z_mad;
    output.reason = match.reason;
    const auto graph_pose = [&](const Keyframe &keyframe)
    {
      return keyframe.id >= 0 && keyframe.id < static_cast<int>(keyframes.size()) ?
          effectivePose(keyframes[static_cast<std::size_t>(keyframe.id)]) :
          effectivePose(keyframe);
    };
    const Eigen::Isometry3d reference_world = graph_pose(reference);
    const Eigen::Isometry3d current_world = graph_pose(current);
    const auto convert = [&](const PairVector &pairs, bool factor_applied,
                             GraphDebugPairVector *output)
    {
      output->reserve(pairs.size());
      for (const PairRecord &pair : pairs)
      {
        GraphDebugPair converted;
        converted.source_world = current_world * pair.source;
        converted.target_world = reference_world * pair.target;
        converted.residual = pair.residual;
        converted.stage = pair.stage;
        if (factor_applied && pair.stage == DebugPairStage::Inlier)
        {
          converted.stage = DebugPairStage::Applied;
        }
        output->push_back(converted);
      }
    };
    convert(match.xy_pairs, applied && match.xy_accepted, &output.xy_pairs);
    convert(match.z_pairs, applied && match.z_accepted, &output.z_pairs);
  }

  void setSemanticObservationStatus(const Keyframe &current,
                                    const std::string &reason)
  {
    semantic_debug = SemanticLoopDebug();
    semantic_debug.valid = true;
    semantic_debug.current_id = current.id;
    semantic_debug.reason = reason;
  }

  SemanticPoseGraphOptions semanticObservationOptions() const
  {
    SemanticPoseGraphOptions local = options;
    local.use_semantics = true;
    // Semantic observations use the same matcher as generic loop proposals,
    // but their enable flags must remain independent. Production disables
    // generic geometric loops while still allowing the locally gated semantic
    // factors below to reach iSAM.
    local.enable_xy_loops = options.enable_semantic_observation_xy_factors;
    local.enable_z_loops = options.enable_semantic_observation_z_factors;
    local.coarse_xy_radius = options.semantic_observation_max_xy_correction;
    local.coarse_xy_step = std::max(0.15, std::min(
        0.30, options.semantic_observation_max_xy_correction / 3.0));
    local.coarse_yaw_radius_deg = options.semantic_observation_max_yaw_correction_deg;
    local.coarse_yaw_step_deg = std::max(
        0.35, options.semantic_observation_max_yaw_correction_deg / 4.0);
    local.coarse_min_inliers = std::max(
        8, options.semantic_observation_min_inliers / 2);
    local.correspondence_distance =
        options.semantic_observation_correspondence_distance;
    local.ransac_inlier_distance =
        options.semantic_observation_ransac_inlier_distance;
    local.min_xy_inliers = options.semantic_observation_min_inliers;
    local.min_xy_inlier_ratio = options.semantic_observation_min_inlier_ratio;
    local.min_xy_spread = options.semantic_observation_min_spread;
    local.min_xy_spread_ratio = options.semantic_observation_min_spread_ratio;
    local.max_xy_rmse = options.semantic_observation_max_rmse;
    local.huber_delta = std::min(options.huber_delta,
                                 options.semantic_observation_ransac_inlier_distance);
    local.max_xy_correction = options.semantic_observation_max_xy_correction;
    local.max_yaw_correction_deg =
        options.semantic_observation_max_yaw_correction_deg;
    local.z_correspondence_distance =
        options.semantic_observation_correspondence_distance;
    local.z_candidate_residual_gate = std::max(
        0.45, options.semantic_observation_max_z_correction + 0.15);
    local.z_inlier_residual_gate = std::min(options.z_inlier_residual_gate, 0.16);
    local.min_z_inliers = options.semantic_observation_min_z_inliers;
    local.max_z_mad = std::min(options.max_z_mad, 0.10);
    local.max_z_correction = options.semantic_observation_max_z_correction;
    return local;
  }

  bool addSemanticObservationFactor(const Keyframe &current,
                                    gtsam::NonlinearFactorGraph *factors)
  {
    if (!options.enable_semantic_observation_factors ||
        (!options.enable_semantic_observation_xy_factors &&
         !options.enable_semantic_observation_z_factors) ||
        !options.use_semantics || factors == nullptr)
    {
      return false;
    }
    if (options.semantic_observation_interval > 1 &&
        (semantic_observations.size() - 1U) %
            static_cast<std::size_t>(options.semantic_observation_interval) != 0U)
    {
      ++stats.semantic_observation_skips;
      return false;
    }
    if (!current.has_semantics)
    {
      ++stats.semantic_observation_skips;
      setSemanticObservationStatus(current, "current_has_no_semantics");
      return false;
    }
    const int minimum_gap = std::max(1, options.semantic_observation_min_index_gap);
    const int maximum_gap = std::max(
        minimum_gap, options.semantic_observation_max_index_gap);
    const int maximum_reference_uses = std::max(
        0, options.semantic_observation_max_reference_uses);
    if (current.id < minimum_gap)
    {
      ++stats.semantic_observation_skips;
      setSemanticObservationStatus(current, "waiting_for_independent_semantic_observation");
      return false;
    }

    const SemanticPoseGraphOptions local_options = semanticObservationOptions();
    const Keyframe semantic_current = semanticOnlyKeyframe(current, local_options);
    if (semantic_current.features.size() < static_cast<std::size_t>(
            std::max(1, options.semantic_observation_min_features)))
    {
      ++stats.semantic_observation_skips;
      setSemanticObservationStatus(current, "insufficient_current_semantic_features");
      return false;
    }

    MatchResult best_match;
    int best_reference_id = -1;
    int best_rank = -1;
    for (const Keyframe &reference : semantic_observations)
    {
      const auto reference_uses = semantic_reference_uses.find(reference.id);
      if (maximum_reference_uses > 0 && reference_uses != semantic_reference_uses.end() &&
          reference_uses->second >= maximum_reference_uses)
      {
        ++stats.semantic_observation_reference_rejections;
        continue;
      }
      const int gap = current.id - reference.id;
      if (gap < minimum_gap || gap > maximum_gap) continue;
      const Eigen::Isometry3d raw_measurement =
          reference.raw_pose.inverse() * current.raw_pose;
      if (raw_measurement.translation().head<2>().norm() <
          options.semantic_observation_min_baseline)
      {
        ++stats.semantic_observation_reference_rejections;
        continue;
      }
      const double distance = (reference.raw_pose.translation() -
                               current.raw_pose.translation()).head<2>().norm();
      if (distance > options.semantic_observation_search_radius) continue;
      const Keyframe semantic_reference = semanticOnlyKeyframe(reference, local_options);
      if (semantic_reference.features.size() < static_cast<std::size_t>(
              std::max(1, options.semantic_observation_min_features)))
      {
        continue;
      }
      ++stats.semantic_observation_attempts;
      MatchResult match = matchKeyframes(semantic_reference, semantic_current,
                                         local_options);
      // A full planar semantic constraint carries more independent information
      // than a Z-only road-height constraint. Do not let many ground pairs hide
      // a valid structural XY match merely because their aggregate score is high.
      const bool xy_available = match.xy_accepted &&
          options.enable_semantic_observation_xy_factors;
      const bool z_available = match.z_accepted &&
          options.enable_semantic_observation_z_factors &&
          (!options.semantic_observation_require_xy_for_z || xy_available);
      const int rank = xy_available ? 2 : (z_available ? 1 : 0);
      if (best_reference_id < 0 || rank > best_rank ||
          (rank == best_rank && match.score > best_match.score))
      {
        best_match = std::move(match);
        best_reference_id = reference.id;
        best_rank = rank;
      }
    }

    if (best_reference_id < 0)
    {
      ++stats.semantic_observation_skips;
      setSemanticObservationStatus(current, "no_independent_semantic_reference");
      return false;
    }
    const auto reference_it = std::find_if(
        semantic_observations.begin(), semantic_observations.end(),
        [&](const Keyframe &item) { return item.id == best_reference_id; });
    if (reference_it == semantic_observations.end()) return false;
    const Keyframe &reference = *reference_it;
    stats.last_semantic_xy_candidates = best_match.xy_candidates;
    stats.last_semantic_xy_inliers = best_match.xy_inliers;
    stats.last_semantic_z_candidates = best_match.z_candidates;
    stats.last_semantic_z_inliers = best_match.z_inliers;
    stats.last_semantic_xy_inlier_ratio = best_match.xy_inlier_ratio;
    stats.last_semantic_xy_rmse = best_match.xy_rmse;
    stats.last_semantic_xy_spread = best_match.spread;
    stats.last_semantic_xy_spread_ratio = best_match.spread_ratio;
    const Eigen::Isometry3d raw_measurement =
        reference.raw_pose.inverse() * current.raw_pose;
    stats.last_semantic_baseline = raw_measurement.translation().head<2>().norm();
    const Eigen::Isometry3d correction = best_match.measurement * raw_measurement.inverse();
    stats.last_semantic_xy_correction = correction.translation().head<2>().norm();
    stats.last_semantic_yaw_correction_deg =
        std::abs(wrapAngle(yawOf(best_match.measurement) - yawOf(raw_measurement))) *
        180.0 / kPi;
    stats.last_semantic_z_median = best_match.z_median;
    stats.last_semantic_z_mad = best_match.z_mad;
    if (best_rank <= 0)
    {
      ++stats.semantic_observation_rejections;
      best_match.reason = "semantic_" + best_match.reason;
      populateDebug(reference, current, best_match, false, &semantic_debug);
      return false;
    }
    // The semantic matcher is evaluated in a local submap.  Before its
    // relative measurement reaches iSAM, compare it with the current graph
    // prediction as well.  This rejects a visually plausible but globally
    // incompatible association instead of allowing a single factor to pull
    // the trajectory into a distant repeated streetscape.
    if (!graphConsistency(reference, current, &best_match))
    {
      ++stats.semantic_observation_rejections;
      populateDebug(reference, current, best_match, false, &semantic_debug);
      return false;
    }

    if (options.semantic_observation_require_xy_for_z && !best_match.xy_accepted &&
        best_match.z_accepted)
    {
      best_match.z_accepted = false;
      best_match.reason = "semantic_z_requires_xy_support";
    }
    if (!best_match.xy_accepted && !best_match.z_accepted)
    {
      ++stats.semantic_observation_rejections;
      populateDebug(reference, current, best_match, false, &semantic_debug);
      return false;
    }

    int applied_inliers = 0;
    if (best_match.xy_accepted && options.enable_semantic_observation_xy_factors)
    {
      gtsam::Vector6 sigmas;
      sigmas << 1e3, 1e3,
          options.semantic_observation_sigma_yaw_deg * kPi / 180.0,
          options.semantic_observation_sigma_xy,
          options.semantic_observation_sigma_xy, 1e3;
      factors->add(gtsam::BetweenFactor<gtsam::Pose3>(
          poseKey(reference.id), poseKey(current.id), toGtsam(best_match.measurement),
          robustDiagonal(sigmas, options.semantic_observation_huber_k)));
      ++stats.semantic_observation_xy_factors;
      applied_inliers += best_match.xy_inliers;
    }
    if (best_match.z_accepted && options.enable_semantic_observation_z_factors)
    {
      gtsam::Vector6 sigmas;
      sigmas << 1e3, 1e3, 1e3, 1e3, 1e3,
          options.semantic_observation_sigma_z;
      factors->add(gtsam::BetweenFactor<gtsam::Pose3>(
          poseKey(reference.id), poseKey(current.id), toGtsam(best_match.measurement),
          robustDiagonal(sigmas, options.semantic_observation_huber_k)));
      ++stats.semantic_observation_z_factors;
      applied_inliers += best_match.z_inliers;
    }
    if (applied_inliers <= 0) return false;
    ++semantic_reference_uses[reference.id];
    ++stats.semantic_observation_factors;
    stats.semantic_observation_inliers += applied_inliers;
    best_match.reason = "semantic_observation_applied";
    populateDebug(reference, current, best_match, true, &semantic_debug);
    return true;
  }

  bool addSequentialGroundFactor(const Keyframe &previous, const Keyframe &current,
                                 gtsam::NonlinearFactorGraph *factors)
  {
    if (!options.enable_sequential_ground_z) return false;
    MatchResult result;
    result.measurement = previous.raw_pose.inverse() * current.raw_pose;
    evaluateZ(previous, current, result.measurement, options, &result);
    if (!result.z_accepted) return false;
    result.measurement.translation().z() += result.z_median;
    gtsam::Vector6 sigmas;
    sigmas << 1e3, 1e3, 1e3, 1e3, 1e3, options.sequential_ground_sigma_z;
    factors->add(gtsam::BetweenFactor<gtsam::Pose3>(
        poseKey(previous.id), poseKey(current.id), toGtsam(result.measurement),
        robustDiagonal(sigmas, options.sequential_ground_huber_k)));
    ++stats.sequential_ground_factors;
    return true;
  }

  bool integrateWheel(double start_stamp, double end_stamp, double *distance) const
  {
    if (!options.enable_wheel_factors || distance == nullptr ||
        end_stamp <= start_stamp || wheel.size() < 2U)
    {
      return false;
    }
    auto interpolate = [&](double stamp, double *speed) -> bool
    {
      auto upper = std::lower_bound(wheel.begin(), wheel.end(), stamp,
          [](const WheelObservation &sample, double value) { return sample.stamp < value; });
      if (upper == wheel.begin())
      {
        if (std::abs(upper->stamp - stamp) > options.wheel_max_gap) return false;
        *speed = upper->speed;
        return true;
      }
      if (upper == wheel.end())
      {
        const WheelObservation &last = wheel.back();
        if (std::abs(last.stamp - stamp) > options.wheel_max_gap) return false;
        *speed = last.speed;
        return true;
      }
      const WheelObservation &second = *upper;
      const WheelObservation &first = *(upper - 1);
      if (stamp - first.stamp > options.wheel_max_gap ||
          second.stamp - stamp > options.wheel_max_gap ||
          second.stamp <= first.stamp)
      {
        return false;
      }
      const double ratio = (stamp - first.stamp) / (second.stamp - first.stamp);
      *speed = (1.0 - ratio) * first.speed + ratio * second.speed;
      return true;
    };

    double start_speed = 0.0;
    double end_speed = 0.0;
    if (!interpolate(start_stamp, &start_speed) || !interpolate(end_stamp, &end_speed))
    {
      return false;
    }
    std::vector<WheelObservation> interval;
    interval.push_back(WheelObservation{start_stamp, start_speed});
    for (const WheelObservation &sample : wheel)
    {
      if (sample.stamp > start_stamp && sample.stamp < end_stamp)
      {
        interval.push_back(sample);
      }
    }
    interval.push_back(WheelObservation{end_stamp, end_speed});
    if (static_cast<int>(interval.size()) < options.wheel_min_samples + 2) return false;
    double integrated = 0.0;
    for (std::size_t index = 1; index < interval.size(); ++index)
    {
      const double dt = interval[index].stamp - interval[index - 1U].stamp;
      if (dt <= 0.0 || dt > options.wheel_max_gap) return false;
      integrated += 0.5 * dt * (interval[index - 1U].speed + interval[index].speed);
    }
    *distance = options.wheel_speed_scale * integrated;
    return std::isfinite(*distance);
  }

  bool addWheelFactor(const Keyframe &previous, const Keyframe &current,
                      const Eigen::Isometry3d &raw_relative,
                      gtsam::NonlinearFactorGraph *factors)
  {
    double arc_length = 0.0;
    if (!integrateWheel(previous.stamp, current.stamp, &arc_length)) return false;
    const double delta_yaw = wrapAngle(yawOf(raw_relative));
    Eigen::Isometry3d measurement = raw_relative;
    if (std::abs(delta_yaw) < 1e-5)
    {
      measurement.translation().x() = arc_length;
      measurement.translation().y() = 0.0;
    }
    else
    {
      measurement.translation().x() = arc_length * std::sin(delta_yaw) / delta_yaw;
      measurement.translation().y() = arc_length * (1.0 - std::cos(delta_yaw)) / delta_yaw;
    }
    const double forward_sigma = options.wheel_sigma_base +
                                 options.wheel_sigma_per_meter * std::abs(arc_length);
    gtsam::Vector6 sigmas;
    sigmas << 1e3, 1e3, 1e3, std::max(0.02, forward_sigma),
              std::max(0.02, options.wheel_lateral_sigma), 1e3;
    factors->add(gtsam::BetweenFactor<gtsam::Pose3>(
        poseKey(previous.id), poseKey(current.id), toGtsam(measurement),
        robustDiagonal(sigmas, options.wheel_huber_k)));
    ++stats.wheel_factors;
    return true;
  }

  const VisualObservation *nearestVisualObservation(double stamp) const
  {
    const VisualObservation *best = nullptr;
    double best_difference = std::numeric_limits<double>::infinity();
    for (auto observation = visual.rbegin(); observation != visual.rend(); ++observation)
    {
      const double difference = std::abs(observation->stamp - stamp);
      if (difference < best_difference)
      {
        best = &*observation;
        best_difference = difference;
      }
      if (observation->stamp < stamp - options.visual_max_time_offset) break;
    }
    return best && best_difference <= options.visual_max_time_offset ? best : nullptr;
  }

  bool addVisualRotationFactor(const Keyframe &previous, const Keyframe &current,
                               const Eigen::Isometry3d &raw_relative,
                               gtsam::NonlinearFactorGraph *factors)
  {
    if (!options.enable_visual_rotation_factors || factors == nullptr) return false;
    const VisualObservation *first = nearestVisualObservation(previous.stamp);
    const VisualObservation *second = nearestVisualObservation(current.stamp);
    if (first == nullptr || second == nullptr || first == second ||
        first->segment != second->segment || second->stamp <= first->stamp)
    {
      return false;
    }
    const double quality = std::min(first->quality, second->quality);
    if (quality < options.visual_min_quality)
    {
      ++stats.visual_rotation_rejections;
      return false;
    }
    const Eigen::Isometry3d visual_relative_pose =
        first->visual_from_body_pose.inverse() * second->visual_from_body_pose;
    const Eigen::Matrix3d visual_relative = visual_relative_pose.rotation();
    const double disagreement = rotationAngle(
        visual_relative.transpose() * raw_relative.rotation());
    if (!std::isfinite(disagreement) || disagreement >
        options.visual_max_angular_disagreement_deg * kPi / 180.0)
    {
      ++stats.visual_rotation_rejections;
      return false;
    }
    Eigen::Isometry3d measurement = raw_relative;
    Eigen::Isometry3d visual_pose = Eigen::Isometry3d::Identity();
    visual_pose.linear() = visual_relative;
    measurement.linear() = projectToSE3(visual_pose).rotation();
    const bool use_translation = options.enable_visual_translation_factors &&
        first->metric_pose && second->metric_pose;
    if (use_translation)
    {
      const double translation_disagreement =
          (visual_relative_pose.translation() - raw_relative.translation()).norm();
      if (!std::isfinite(translation_disagreement) || translation_disagreement >
          options.visual_max_translation_disagreement)
      {
        ++stats.visual_rotation_rejections;
        return false;
      }
      measurement.translation() = visual_relative_pose.translation();
    }
    const double quality_scale = 1.0 + options.visual_quality_sigma_scale *
        (1.0 - clampValue(quality, 0.0, 1.0));
    const double distance = std::max(0.0, visual_relative_pose.translation().norm());
    const double visual_xy_sigma = options.visual_sigma_xy_base +
        options.visual_sigma_translation_per_meter * distance;
    const double visual_z_sigma = options.visual_sigma_z_base +
        options.visual_sigma_translation_per_meter * distance;
    gtsam::Vector6 sigmas;
    sigmas << options.visual_sigma_roll_pitch_deg * quality_scale * kPi / 180.0,
              options.visual_sigma_roll_pitch_deg * quality_scale * kPi / 180.0,
              options.visual_sigma_yaw_deg * quality_scale * kPi / 180.0,
              use_translation ? visual_xy_sigma * quality_scale : 1e3,
              use_translation ? visual_xy_sigma * quality_scale : 1e3,
              use_translation ? visual_z_sigma * quality_scale : 1e3;
    factors->add(gtsam::BetweenFactor<gtsam::Pose3>(
        poseKey(previous.id), poseKey(current.id), toGtsam(measurement),
        robustDiagonal(sigmas, options.visual_huber_k)));
    ++stats.visual_rotation_factors;
    if (use_translation) ++stats.visual_translation_factors;
    return true;
  }

  bool addLoop(const Keyframe &reference, Keyframe *current, MatchResult *match)
  {
    if (!graphConsistency(reference, *current, match))
    {
      populateDebug(reference, *current, *match, false);
      ++stats.loop_rejections;
      return false;
    }
    gtsam::NonlinearFactorGraph factors;
    if (match->xy_accepted)
    {
      gtsam::Vector6 sigmas;
      sigmas << 1e3, 1e3, options.loop_sigma_yaw_deg * kPi / 180.0,
                options.loop_sigma_xy, options.loop_sigma_xy, 1e3;
      factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
          poseKey(reference.id), poseKey(current->id), toGtsam(match->measurement),
          robustDiagonal(sigmas, options.loop_huber_k)));
      ++stats.xy_loop_factors;
    }
    if (match->z_accepted)
    {
      gtsam::Vector6 sigmas;
      sigmas << 1e3, 1e3, 1e3, 1e3, 1e3, options.loop_sigma_z;
      factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
          poseKey(reference.id), poseKey(current->id), toGtsam(match->measurement),
          robustDiagonal(sigmas, options.loop_huber_k)));
      ++stats.z_loop_factors;
    }
    if (factors.empty()) return false;
    isam->update(factors, gtsam::Values());
    isam->update();
    updateOptimizedPoses();
    ++stats.loop_factors;
    match->reason = "applied";
    populateDebug(reference, *current, *match, true);
    return true;
  }

  bool addSemanticObservation(double stamp, const Eigen::Isometry3d &raw_pose,
                              const SemanticGraphPointVector &local_points)
  {
    ++stats.semantic_observations_received;
    if (!options.use_semantics || keyframes.empty() || local_points.empty()) return false;
    auto associated = std::min_element(
        keyframes.begin(), keyframes.end(), [&](const Keyframe &first,
                                                const Keyframe &second)
        {
          return std::abs(first.stamp - stamp) < std::abs(second.stamp - stamp);
        });
    if (associated == keyframes.end() ||
        std::abs(associated->stamp - stamp) > options.semantic_observation_max_time_offset)
    {
      ++stats.semantic_observation_skips;
      semantic_debug = SemanticLoopDebug();
      semantic_debug.valid = true;
      semantic_debug.reason = "semantic_pose_association_failed";
      return false;
    }

    Frame frame;
    frame.stamp = stamp;
    frame.raw_pose = projectToSE3(raw_pose);
    frame.points.reserve(local_points.size());
    for (const SemanticGraphPoint &point : local_points)
    {
      if (point.label != 0U && point.point.allFinite()) frame.points.push_back(point);
    }
    if (frame.points.empty()) return false;

    ++stats.semantic_observations_associated;
    associated->has_semantics = true;
    semantic_pending_frames.push_back(std::move(frame));
    const std::size_t observations_per_submap = static_cast<std::size_t>(
        std::max(1, options.semantic_submap_observations));
    while (semantic_pending_frames.size() > observations_per_submap)
    {
      semantic_pending_frames.pop_front();
    }
    stats.semantic_pending_observations =
        static_cast<int>(semantic_pending_frames.size());
    if (semantic_pending_frames.size() < observations_per_submap)
    {
      Keyframe pending;
      pending.id = associated->id;
      pending.stamp = stamp;
      std::ostringstream reason;
      reason << "collecting_semantic_submap_" << semantic_pending_frames.size()
             << "_of_" << observations_per_submap;
      setSemanticObservationStatus(pending, reason.str());
      return true;
    }

    Keyframe observation;
    observation.id = associated->id;
    observation.stamp = associated->stamp;
    observation.raw_pose = associated->raw_pose;
    observation.optimized_pose = associated->optimized_pose;
    observation.features = buildFeatures(semantic_pending_frames, observation.raw_pose,
                                         options, &observation.has_semantics);
    observation = semanticOnlyKeyframe(observation, options);
    // The batch is consumed regardless of whether it passes feature gates. This
    // guarantees that no original SAM observation is counted in two factors.
    semantic_pending_frames.clear();
    stats.semantic_pending_observations = 0;
    if (!observation.has_semantics || observation.features.empty())
    {
      ++stats.semantic_observation_skips;
      setSemanticObservationStatus(observation, "semantic_observation_has_no_features");
      return false;
    }

    auto duplicate = std::find_if(
        semantic_observations.begin(), semantic_observations.end(),
        [&](const Keyframe &item) { return item.id == observation.id; });
    if (duplicate != semantic_observations.end())
    {
      *duplicate = std::move(observation);
      return true;
    }
    semantic_observations.push_back(observation);
    stats.semantic_keyframes = static_cast<int>(semantic_observations.size());

    gtsam::NonlinearFactorGraph factors;
    if (addSemanticObservationFactor(semantic_observations.back(), &factors) &&
        !factors.empty())
    {
      isam->update(factors, gtsam::Values());
      isam->update();
      updateOptimizedPoses();
    }
    return true;
  }

  bool createKeyframe(double stamp, const Eigen::Isometry3d &raw_pose)
  {
    if (static_cast<int>(keyframes.size()) >= options.max_keyframes) return false;
    Keyframe current;
    current.id = static_cast<int>(keyframes.size());
    current.stamp = stamp;
    current.raw_pose = projectToSE3(raw_pose);
    current.optimized_pose = current.raw_pose;
    current.features = buildFeatures(frames, current.raw_pose, options, &current.has_semantics);
    current.descriptor = buildDescriptor(current.features, options);
    if (current.features.size() < static_cast<std::size_t>(std::max(30, options.coarse_min_inliers / 2)))
    {
      return false;
    }

    const auto start = std::chrono::steady_clock::now();
    gtsam::NonlinearFactorGraph factors;
    gtsam::Values values;
    if (keyframes.empty())
    {
      gtsam::Vector6 prior_sigmas;
      prior_sigmas << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
      factors.add(gtsam::PriorFactor<gtsam::Pose3>(poseKey(current.id),
                                                   toGtsam(current.raw_pose),
                                                   gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas)));
      values.insert(poseKey(current.id), toGtsam(current.raw_pose));
    }
    else
    {
      const Keyframe &previous = keyframes.back();
      const Eigen::Isometry3d relative = previous.raw_pose.inverse() * current.raw_pose;
      const double distance = std::max(0.1, relative.translation().norm());
      gtsam::Vector6 sigmas;
      sigmas << options.odom_sigma_roll_pitch, options.odom_sigma_roll_pitch,
                options.odom_sigma_yaw,
                options.odom_sigma_xy_base + options.odom_sigma_xy_per_meter * distance,
                options.odom_sigma_xy_base + options.odom_sigma_xy_per_meter * distance,
                options.odom_sigma_z_base + options.odom_sigma_xy_per_meter * distance;
      factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
          poseKey(previous.id), poseKey(current.id), toGtsam(relative),
          gtsam::noiseModel::Diagonal::Sigmas(sigmas)));
      ++stats.odometry_factors;
      addVisualRotationFactor(previous, current, relative, &factors);
      addWheelFactor(previous, current, relative, &factors);
      addSequentialGroundFactor(previous, current, &factors);
      const Eigen::Isometry3d initial = previous.optimized_pose * relative;
      values.insert(poseKey(current.id), toGtsam(initial));
    }
    isam->update(factors, values);
    keyframes.push_back(current);
    updateOptimizedPoses();
    stats.keyframes = static_cast<int>(keyframes.size());

    if (keyframes.size() > static_cast<std::size_t>(options.loop_min_index_gap + 1))
    {
      Keyframe &stored_current = keyframes.back();
      const std::vector<Candidate> candidates = findCandidates(stored_current);
      stats.loop_attempts += static_cast<int>(candidates.size());
      MatchResult best_match;
      int best_id = -1;
      double second_score = -std::numeric_limits<double>::infinity();
      for (const Candidate &candidate : candidates)
      {
        MatchResult match = matchKeyframes(keyframes[static_cast<std::size_t>(candidate.id)],
                                           stored_current, options);
        match.descriptor_similarity = candidate.similarity;
        if (match.score > best_match.score)
        {
          second_score = best_match.score;
          best_match = match;
          best_id = candidate.id;
        }
        else
        {
          second_score = std::max(second_score, match.score);
        }
      }
      if (best_id >= 0)
      {
        const double score_scale = std::isfinite(second_score) ?
            std::max({1.0, std::abs(best_match.score), std::abs(second_score)}) : 1.0;
        const double descriptor_score_gap = std::isfinite(second_score) ?
            (best_match.score - second_score) / score_scale : 1.0;
        const bool ambiguous = std::isfinite(second_score) &&
            descriptor_score_gap < options.descriptor_min_score_gap;
        if (ambiguous)
        {
          best_match.xy_accepted = false;
          best_match.z_accepted = false;
          best_match.reason = "ambiguous_candidate_score";
          populateDebug(keyframes[static_cast<std::size_t>(best_id)], stored_current,
                        best_match, false);
          ++stats.loop_rejections;
        }
        else if (best_match.xy_accepted || best_match.z_accepted)
        {
          addLoop(keyframes[static_cast<std::size_t>(best_id)], &stored_current, &best_match);
        }
        else
        {
          populateDebug(keyframes[static_cast<std::size_t>(best_id)], stored_current,
                        best_match, false);
          ++stats.loop_rejections;
        }
        debug.descriptor_score_gap = descriptor_score_gap;
      }
    }
    stats.last_optimization_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return true;
  }

  SemanticPoseGraphOptions options;
  std::unique_ptr<gtsam::ISAM2> isam;
  std::deque<Frame, Eigen::aligned_allocator<Frame>> frames;
  std::deque<Frame, Eigen::aligned_allocator<Frame>> semantic_pending_frames;
  std::vector<Keyframe, Eigen::aligned_allocator<Keyframe>> keyframes;
  std::vector<Keyframe, Eigen::aligned_allocator<Keyframe>> semantic_observations;
  std::unordered_map<int, int> semantic_reference_uses;
  std::vector<GraphTrajectorySample, Eigen::aligned_allocator<GraphTrajectorySample>> odometry;
  std::deque<WheelObservation> wheel;
  std::deque<VisualObservation, Eigen::aligned_allocator<VisualObservation>> visual;
  std::unordered_set<std::uint64_t> visual_loop_pairs;
  SemanticLoopDebug debug;
  SemanticLoopDebug semantic_debug;
  SemanticPoseGraphStats stats;
};

SemanticPoseGraph::SemanticPoseGraph(const SemanticPoseGraphOptions &options)
    : impl_(new Impl(options))
{
}

SemanticPoseGraph::~SemanticPoseGraph() = default;

void SemanticPoseGraph::reset()
{
  const SemanticPoseGraphOptions options = impl_->options;
  impl_.reset(new Impl(options));
}

void SemanticPoseGraph::addWheelSample(double stamp, double forward_speed)
{
  if (!impl_->options.enabled || !impl_->options.enable_wheel_factors ||
      !std::isfinite(stamp) || !std::isfinite(forward_speed)) return;
  if (!impl_->wheel.empty() && stamp <= impl_->wheel.back().stamp) return;
  impl_->wheel.push_back(WheelObservation{stamp, forward_speed});
  const double keep_after = stamp - 60.0;
  while (impl_->wheel.size() > 2U && impl_->wheel[1].stamp < keep_after)
  {
    impl_->wheel.pop_front();
  }
}

void SemanticPoseGraph::addVisualRotationSample(
    double stamp, const Eigen::Matrix3d &visual_from_body_rotation,
    int segment, double quality)
{
  if (!impl_->options.enabled || !impl_->options.enable_visual_rotation_factors ||
      !std::isfinite(stamp) || !visual_from_body_rotation.allFinite() ||
      !std::isfinite(quality)) return;
  if (!impl_->visual.empty() && stamp <= impl_->visual.back().stamp) return;
  VisualObservation observation;
  observation.stamp = stamp;
  Eigen::Isometry3d visual_pose = Eigen::Isometry3d::Identity();
  visual_pose.linear() = visual_from_body_rotation;
  observation.visual_from_body_pose = projectToSE3(visual_pose);
  observation.segment = segment;
  observation.quality = clampValue(quality, 0.0, 1.0);
  observation.metric_pose = false;
  impl_->visual.push_back(observation);
  const double keep_after = stamp - 60.0;
  while (impl_->visual.size() > 2U && impl_->visual[1].stamp < keep_after)
  {
    impl_->visual.pop_front();
  }
}

void SemanticPoseGraph::addVisualPoseSample(
    double stamp, const Eigen::Isometry3d &visual_from_body_pose,
    int segment, double quality)
{
  if (!impl_->options.enabled || !impl_->options.enable_visual_rotation_factors ||
      !std::isfinite(stamp) || !visual_from_body_pose.matrix().allFinite() ||
      !std::isfinite(quality)) return;
  if (!impl_->visual.empty() && stamp <= impl_->visual.back().stamp) return;
  VisualObservation observation;
  observation.stamp = stamp;
  observation.visual_from_body_pose = projectToSE3(visual_from_body_pose);
  observation.segment = segment;
  observation.quality = clampValue(quality, 0.0, 1.0);
  observation.metric_pose = true;
  impl_->visual.push_back(observation);
  const double keep_after = stamp - 60.0;
  while (impl_->visual.size() > 2U && impl_->visual[1].stamp < keep_after)
  {
    impl_->visual.pop_front();
  }
}

void SemanticPoseGraph::addOdometrySample(double stamp, const Eigen::Isometry3d &raw_pose)
{
  if (!impl_->options.enabled || !std::isfinite(stamp)) return;
  if (!impl_->odometry.empty() && stamp <= impl_->odometry.back().stamp + 1e-9) return;
  GraphTrajectorySample sample;
  sample.stamp = stamp;
  sample.raw_pose = projectToSE3(raw_pose);
  sample.optimized_pose = correctedPose(sample.raw_pose);
  impl_->odometry.push_back(sample);
}

bool SemanticPoseGraph::addFrame(double stamp, const Eigen::Isometry3d &raw_pose,
                                 const SemanticGraphPointVector &local_points)
{
  if (!impl_->options.enabled || local_points.empty() || !std::isfinite(stamp)) return false;
  Frame frame;
  frame.stamp = stamp;
  frame.raw_pose = projectToSE3(raw_pose);
  const int maximum = std::max(100, impl_->options.max_points_per_frame);
  const int stride = std::max(1, static_cast<int>(local_points.size()) / maximum);
  frame.points.reserve(std::min(local_points.size(), static_cast<std::size_t>(maximum)));
  for (std::size_t i = 0; i < local_points.size(); i += static_cast<std::size_t>(stride))
  {
    frame.points.push_back(local_points[i]);
    if (static_cast<int>(frame.points.size()) >= maximum) break;
  }
  impl_->frames.push_back(frame);
  while (static_cast<int>(impl_->frames.size()) > std::max(1, impl_->options.submap_frames))
  {
    impl_->frames.pop_front();
  }

  bool make_keyframe = impl_->keyframes.empty();
  if (!make_keyframe)
  {
    const Keyframe &last = impl_->keyframes.back();
    const Eigen::Isometry3d motion = last.raw_pose.inverse() * frame.raw_pose;
    make_keyframe = motion.translation().head<2>().norm() >= impl_->options.keyframe_distance ||
        std::abs(wrapAngle(yawOf(motion))) >= impl_->options.keyframe_yaw_deg * kPi / 180.0 ||
        stamp - last.stamp >= impl_->options.keyframe_interval_sec;
  }
  if (!make_keyframe) return false;
  return impl_->createKeyframe(stamp, frame.raw_pose);
}

bool SemanticPoseGraph::addSemanticObservation(
    double stamp, const Eigen::Isometry3d &raw_pose,
    const SemanticGraphPointVector &local_points)
{
  if (!impl_->options.enabled || !impl_->options.use_semantics ||
      !std::isfinite(stamp) || local_points.empty())
  {
    return false;
  }
  return impl_->addSemanticObservation(stamp, raw_pose, local_points);
}

bool SemanticPoseGraph::addVisualLoopConstraint(
    double reference_stamp, double current_stamp,
    const Eigen::Isometry3d &reference_from_current, double quality)
{
  if (!impl_->options.enabled || !impl_->options.enable_visual_loop_factors)
    return false;
  return impl_->addVisualLoopConstraint(reference_stamp, current_stamp,
                                        reference_from_current, quality);
}

bool SemanticPoseGraph::initialized() const
{
  return !impl_->keyframes.empty();
}

Eigen::Isometry3d SemanticPoseGraph::correctedPose(const Eigen::Isometry3d &raw_pose) const
{
  if (impl_->keyframes.empty()) return projectToSE3(raw_pose);
  const Keyframe &last = impl_->keyframes.back();
  const Eigen::Isometry3d correction = impl_->effectivePose(last) * last.raw_pose.inverse();
  return projectToSE3(correction * raw_pose);
}

std::vector<GraphTrajectorySample, Eigen::aligned_allocator<GraphTrajectorySample>>
SemanticPoseGraph::optimizedTrajectory() const
{
  std::vector<GraphTrajectorySample, Eigen::aligned_allocator<GraphTrajectorySample>> result =
      impl_->odometry;
  if (result.empty() || impl_->keyframes.empty()) return result;
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> corrections;
  corrections.reserve(impl_->keyframes.size());
  for (const Keyframe &keyframe : impl_->keyframes)
  {
    corrections.push_back(impl_->effectivePose(keyframe) * keyframe.raw_pose.inverse());
  }
  std::size_t upper = 0U;
  for (GraphTrajectorySample &sample : result)
  {
    while (upper < impl_->keyframes.size() && impl_->keyframes[upper].stamp < sample.stamp) ++upper;
    Eigen::Isometry3d correction;
    if (upper == 0U)
    {
      correction = corrections.front();
    }
    else if (upper >= impl_->keyframes.size())
    {
      correction = corrections.back();
    }
    else
    {
      const double start = impl_->keyframes[upper - 1U].stamp;
      const double end = impl_->keyframes[upper].stamp;
      const double ratio = end > start ? (sample.stamp - start) / (end - start) : 0.0;
      correction = interpolateTransform(corrections[upper - 1U], corrections[upper], ratio);
    }
    sample.optimized_pose = projectToSE3(correction * sample.raw_pose);
  }
  return result;
}

bool SemanticPoseGraph::saveOptimizedTrajectory(const std::string &path) const
{
  if (path.empty()) return false;
  const auto trajectory = optimizedTrajectory();
  if (trajectory.empty()) return false;
  const std::string temporary_path = path + ".tmp";
  std::ofstream output(temporary_path.c_str(), std::ios::out | std::ios::trunc);
  if (!output.is_open()) return false;
  output << "timestamp,x,y,z,qx,qy,qz,qw\n";
  output << std::fixed << std::setprecision(9);
  for (const GraphTrajectorySample &sample : trajectory)
  {
    Eigen::Quaterniond quaternion(sample.optimized_pose.rotation());
    quaternion.normalize();
    output << sample.stamp << ',' << sample.optimized_pose.translation().x() << ','
           << sample.optimized_pose.translation().y() << ','
           << sample.optimized_pose.translation().z() << ',' << quaternion.x() << ','
           << quaternion.y() << ',' << quaternion.z() << ',' << quaternion.w() << '\n';
  }
  output.flush();
  const bool good = output.good();
  output.close();
  if (!good) return false;
  if (std::rename(temporary_path.c_str(), path.c_str()) != 0)
  {
    std::remove(temporary_path.c_str());
    return false;
  }
  return true;
}

SemanticGraphPointVector SemanticPoseGraph::semanticMap(double voxel_size,
                                                        int maximum_points) const
{
  return impl_->buildSemanticMap(voxel_size, maximum_points);
}

const SemanticLoopDebug &SemanticPoseGraph::lastDebug() const
{
  return impl_->debug;
}

const SemanticLoopDebug &SemanticPoseGraph::lastSemanticDebug() const
{
  return impl_->semantic_debug;
}

SemanticPoseGraphStats SemanticPoseGraph::stats() const
{
  return impl_->stats;
}

}  // namespace hybrid_localization
