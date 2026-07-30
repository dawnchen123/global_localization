#include "hybrid_localization/sparse_visual_map.h"

#include <Eigen/SVD>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hybrid_localization
{
namespace
{

constexpr double kSmall = 1e-9;

double clampValue(double value, double minimum, double maximum)
{
  return std::max(minimum, std::min(maximum, value));
}

Eigen::Matrix3d skew(const Eigen::Vector3d &vector)
{
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(),
            vector.z(), 0.0, -vector.x(),
            -vector.y(), vector.x(), 0.0;
  return matrix;
}

Eigen::Matrix3d projectToRotation(const Eigen::Matrix3d &matrix)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
  if (rotation.determinant() < 0.0)
  {
    Eigen::Matrix3d u = svd.matrixU();
    u.col(2) *= -1.0;
    rotation = u * svd.matrixV().transpose();
  }
  return rotation;
}

double huberWeight(double residual, double delta)
{
  const double magnitude = std::abs(residual);
  return magnitude <= delta || magnitude < kSmall ? 1.0 : delta / magnitude;
}

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
  std::size_t operator()(const VoxelKey &key) const
  {
    std::size_t seed = std::hash<int>()(key.x);
    seed ^= std::hash<int>()(key.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int>()(key.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

VoxelKey voxelKey(const Eigen::Vector3d &point, double resolution)
{
  VoxelKey key;
  key.x = static_cast<int>(std::floor(point.x() / resolution));
  key.y = static_cast<int>(std::floor(point.y() / resolution));
  key.z = static_cast<int>(std::floor(point.z() / resolution));
  return key;
}

}  // namespace

SparseVisualMap::SparseVisualMap(const SparseVisualMapOptions &options)
    : options_(options)
{
  options_.image_scale = clampValue(options_.image_scale, 0.1, 1.0);
  options_.patch_half_size = std::max(1, options_.patch_half_size);
  options_.grid_size_pixels = std::max(4, options_.grid_size_pixels);
  options_.max_landmarks = std::max(50, options_.max_landmarks);
  options_.max_active_landmarks = std::max(20, std::min(
      options_.max_landmarks, options_.max_active_landmarks));
  options_.max_lidar_candidates = std::max(100, options_.max_lidar_candidates);
  options_.max_new_landmarks_per_frame = std::max(
      10, options_.max_new_landmarks_per_frame);
  options_.max_missed_frames = std::max(2, options_.max_missed_frames);
  options_.reference_refresh_observations = std::max(
      2, options_.reference_refresh_observations);
  options_.minimum_depth = std::max(0.1, options_.minimum_depth);
  options_.maximum_depth = std::max(
      options_.minimum_depth + 1.0, options_.maximum_depth);
  options_.minimum_gradient = std::max(0.0, options_.minimum_gradient);
  options_.minimum_patch_stddev = std::max(0.1, options_.minimum_patch_stddev);
  options_.minimum_ncc = clampValue(options_.minimum_ncc, -1.0, 1.0);
  options_.photometric_huber_delta = std::max(
      0.1, options_.photometric_huber_delta);
  options_.photometric_noise = std::max(0.05, options_.photometric_noise);
  options_.information_scale = std::max(1e-6, options_.information_scale);
  options_.landmark_voxel_size = std::max(0.05, options_.landmark_voxel_size);
  options_.local_map_radius = std::max(5.0, options_.local_map_radius);
  for (float &weight : options_.semantic_class_weights)
  {
    weight = static_cast<float>(clampValue(weight, 0.0, 2.0));
  }
  options_.body_from_camera_rotation = projectToRotation(
      options_.body_from_camera_rotation);
  if (!options_.body_from_camera_translation.allFinite())
  {
    options_.body_from_camera_translation.setZero();
  }
  reset();
}

void SparseVisualMap::reset()
{
  candidates_.clear();
  landmarks_.clear();
  last_inlier_ids_.clear();
  debug_image_.release();
  next_landmark_id_ = 0;
  stats_ = SparseVisualMapStats();
}

SparseVisualFrame SparseVisualMap::prepareFrame(
    double stamp, const cv::Mat &image, const cv::Mat &dynamic_mask,
    const cv::Mat &semantic_labels) const
{
  SparseVisualFrame frame;
  frame.stamp = stamp;
  if (!options_.enabled || !std::isfinite(stamp) || image.empty()) return frame;

  cv::Mat gray;
  if (image.channels() == 1)
  {
    gray = image;
  }
  else if (image.channels() == 3)
  {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  else if (image.channels() == 4)
  {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  }
  else
  {
    return frame;
  }
  if (gray.depth() != CV_8U)
  {
    cv::Mat converted;
    gray.convertTo(converted, CV_8U);
    gray = converted;
  }
  if (options_.image_scale < 0.999)
  {
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(), options_.image_scale,
               options_.image_scale, cv::INTER_AREA);
    gray = resized;
  }
  const double scale = options_.image_scale;
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
      options_.fx * scale, 0.0, options_.cx * scale,
      0.0, options_.fy * scale, options_.cy * scale,
      0.0, 0.0, 1.0);
  const cv::Mat distortion = (cv::Mat_<double>(1, 5) <<
      options_.distortion[0], options_.distortion[1], options_.distortion[2],
      options_.distortion[3], options_.distortion[4]);
  cv::Mat rectified;
  cv::undistort(gray, rectified, camera_matrix, distortion, camera_matrix);
  rectified.convertTo(frame.gray, CV_32F);
  cv::Sobel(frame.gray, frame.gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(frame.gray, frame.gradient_y, CV_32F, 0, 1, 3, 0.125);

  if (!dynamic_mask.empty())
  {
    cv::Mat mask;
    if (dynamic_mask.channels() == 1)
    {
      mask = dynamic_mask;
    }
    else
    {
      cv::cvtColor(dynamic_mask, mask, cv::COLOR_BGR2GRAY);
    }
    if (mask.depth() != CV_8U) mask.convertTo(mask, CV_8U);
    if (mask.size() != gray.size())
    {
      cv::resize(mask, mask, gray.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::undistort(mask, frame.dynamic_mask, camera_matrix, distortion,
                  camera_matrix);
    cv::threshold(frame.dynamic_mask, frame.dynamic_mask, 0.0, 255.0,
                  cv::THRESH_BINARY);
  }
  if (!semantic_labels.empty())
  {
    cv::Mat labels;
    if (semantic_labels.channels() == 1)
    {
      labels = semantic_labels;
    }
    else
    {
      cv::cvtColor(semantic_labels, labels, cv::COLOR_BGR2GRAY);
    }
    if (labels.depth() != CV_8U) labels.convertTo(labels, CV_8U);
    if (labels.size() != gray.size())
    {
      cv::resize(labels, labels, gray.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::undistort(labels, frame.semantic_labels, camera_matrix, distortion,
                  camera_matrix);
  }
  return frame;
}

void SparseVisualMap::addLidarFrame(
    double stamp, const Eigen::Isometry3d &world_from_body,
    const PointVector &body_points)
{
  addLidarFrame(stamp, world_from_body, body_points,
                std::vector<uint8_t>(), std::vector<float>());
}

void SparseVisualMap::addLidarFrame(
    double stamp, const Eigen::Isometry3d &world_from_body,
    const PointVector &body_points, const std::vector<uint8_t> &labels,
    const std::vector<float> &semantic_weights)
{
  candidates_.clear();
  if (!options_.enabled || !std::isfinite(stamp) ||
      !world_from_body.matrix().allFinite() || body_points.empty())
  {
    return;
  }
  const std::size_t stride = std::max<std::size_t>(
      1U, body_points.size() /
          static_cast<std::size_t>(options_.max_lidar_candidates));
  std::unordered_set<VoxelKey, VoxelKeyHash> occupied;
  candidates_.reserve(std::min<std::size_t>(
      body_points.size(), static_cast<std::size_t>(options_.max_lidar_candidates)));
  for (std::size_t index = 0; index < body_points.size(); index += stride)
  {
    const Eigen::Vector3d &body_point = body_points[index];
    if (!body_point.allFinite()) continue;
    Candidate candidate;
    candidate.point = world_from_body * body_point;
    if (!candidate.point.allFinite() ||
        !occupied.insert(voxelKey(candidate.point,
                                  options_.landmark_voxel_size)).second)
    {
      continue;
    }
    if (index < labels.size()) candidate.label = labels[index];
    if (index < semantic_weights.size() &&
        std::isfinite(semantic_weights[index]))
    {
      candidate.semantic_weight = static_cast<float>(clampValue(
          semantic_weights[index], 0.0, 2.0));
    }
    candidates_.push_back(candidate);
    if (candidates_.size() >=
        static_cast<std::size_t>(options_.max_lidar_candidates))
    {
      break;
    }
  }
}

bool SparseVisualMap::project(
    const Eigen::Vector3d &world_point,
    const Eigen::Isometry3d &world_from_body, cv::Point2f *pixel,
    Eigen::Vector3d *camera_point, Eigen::Vector3d *body_point) const
{
  const Eigen::Vector3d point_body = world_from_body.rotation().transpose() *
      (world_point - world_from_body.translation());
  const Eigen::Matrix3d camera_from_body =
      options_.body_from_camera_rotation.transpose();
  const Eigen::Vector3d point_camera = camera_from_body *
      (point_body - options_.body_from_camera_translation);
  if (!point_camera.allFinite() || point_camera.z() < options_.minimum_depth ||
      point_camera.z() > options_.maximum_depth)
  {
    return false;
  }
  const double scale = options_.image_scale;
  if (pixel != nullptr)
  {
    pixel->x = static_cast<float>(options_.fx * scale * point_camera.x() /
                                 point_camera.z() + options_.cx * scale);
    pixel->y = static_cast<float>(options_.fy * scale * point_camera.y() /
                                 point_camera.z() + options_.cy * scale);
  }
  if (camera_point != nullptr) *camera_point = point_camera;
  if (body_point != nullptr) *body_point = point_body;
  return true;
}

float SparseVisualMap::bilinear(const cv::Mat &image, float x, float y)
{
  const int left = static_cast<int>(std::floor(x));
  const int top = static_cast<int>(std::floor(y));
  if (left < 0 || top < 0 || left + 1 >= image.cols || top + 1 >= image.rows)
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float dx = x - static_cast<float>(left);
  const float dy = y - static_cast<float>(top);
  const float top_value = (1.0F - dx) * image.at<float>(top, left) +
                          dx * image.at<float>(top, left + 1);
  const float bottom_value = (1.0F - dx) * image.at<float>(top + 1, left) +
                             dx * image.at<float>(top + 1, left + 1);
  return (1.0F - dy) * top_value + dy * bottom_value;
}

bool SparseVisualMap::normalizedPatch(
    const cv::Mat &image, const cv::Point2f &center,
    std::vector<float> *patch, double *stddev) const
{
  if (patch == nullptr || image.empty() || image.type() != CV_32F) return false;
  const int half = options_.patch_half_size;
  const int border = half + 2;
  if (center.x < border || center.y < border ||
      center.x >= image.cols - border || center.y >= image.rows - border)
  {
    return false;
  }
  patch->clear();
  patch->reserve(static_cast<std::size_t>((2 * half + 1) * (2 * half + 1)));
  double sum = 0.0;
  for (int y = -half; y <= half; ++y)
  {
    for (int x = -half; x <= half; ++x)
    {
      const float value = bilinear(image, center.x + static_cast<float>(x),
                                   center.y + static_cast<float>(y));
      if (!std::isfinite(value)) return false;
      patch->push_back(value);
      sum += value;
    }
  }
  const double mean = sum / static_cast<double>(patch->size());
  double variance = 0.0;
  for (float value : *patch)
  {
    const double difference = static_cast<double>(value) - mean;
    variance += difference * difference;
  }
  const double sigma = std::sqrt(variance /
      static_cast<double>(patch->size()));
  if (!std::isfinite(sigma) || sigma < options_.minimum_patch_stddev)
  {
    return false;
  }
  for (float &value : *patch)
  {
    value = static_cast<float>((static_cast<double>(value) - mean) / sigma);
  }
  if (stddev != nullptr) *stddev = sigma;
  return true;
}

bool SparseVisualMap::masked(
    const SparseVisualFrame &frame, const cv::Point2f &pixel) const
{
  if (frame.dynamic_mask.empty()) return false;
  const int x = static_cast<int>(std::lround(pixel.x));
  const int y = static_cast<int>(std::lround(pixel.y));
  return x >= 0 && y >= 0 && x < frame.dynamic_mask.cols &&
         y < frame.dynamic_mask.rows && frame.dynamic_mask.at<uint8_t>(y, x) != 0U;
}

uint8_t SparseVisualMap::semanticLabel(
    const SparseVisualFrame &frame, const cv::Point2f &pixel) const
{
  if (frame.semantic_labels.empty()) return 0U;
  const int x = static_cast<int>(std::lround(pixel.x));
  const int y = static_cast<int>(std::lround(pixel.y));
  if (x < 0 || y < 0 || x >= frame.semantic_labels.cols ||
      y >= frame.semantic_labels.rows)
  {
    return 0U;
  }
  return frame.semantic_labels.at<uint8_t>(y, x);
}

float SparseVisualMap::semanticClassWeight(uint8_t label) const
{
  const std::size_t index = static_cast<std::size_t>(label);
  return index < options_.semantic_class_weights.size() ?
      options_.semantic_class_weights[index] : options_.semantic_class_weights[0];
}

VisualPoseLinearization SparseVisualMap::linearize(
    const SparseVisualFrame &frame,
    const Eigen::Isometry3d &world_from_body)
{
  VisualPoseLinearization result;
  last_inlier_ids_.clear();
  if (!options_.enabled)
  {
    result.reason = "sparse_visual_map_disabled";
    return result;
  }
  if (!frame.valid() || !world_from_body.matrix().allFinite())
  {
    result.reason = "invalid_visual_frame";
    return result;
  }
  cv::Mat debug_gray;
  frame.gray.convertTo(debug_gray, CV_8U);
  cv::cvtColor(debug_gray, debug_image_, cv::COLOR_GRAY2BGR);
  if (landmarks_.empty())
  {
    result.reason = "visual_map_empty";
    stats_.last_reason = result.reason;
    return result;
  }

  struct Projection
  {
    std::size_t landmark = 0U;
    cv::Point2f pixel;
    Eigen::Vector3d camera_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d body_point = Eigen::Vector3d::Zero();
  };
  std::unordered_map<int, Projection> grid;
  const int grid_cols = std::max(1, frame.gray.cols / options_.grid_size_pixels);
  int dynamic_rejections = 0;
  for (std::size_t index = 0; index < landmarks_.size(); ++index)
  {
    Projection projection;
    projection.landmark = index;
    if (!project(landmarks_[index].point, world_from_body, &projection.pixel,
                 &projection.camera_point, &projection.body_point))
    {
      continue;
    }
    const int border = options_.patch_half_size + 2;
    if (projection.pixel.x < border || projection.pixel.y < border ||
        projection.pixel.x >= frame.gray.cols - border ||
        projection.pixel.y >= frame.gray.rows - border)
    {
      continue;
    }
    const uint8_t current_label = semanticLabel(frame, projection.pixel);
    if (masked(frame, projection.pixel) || semanticClassWeight(current_label) <= 0.0F ||
        landmarks_[index].semantic_weight <= 0.0F)
    {
      ++dynamic_rejections;
      continue;
    }
    const int cell_x = static_cast<int>(projection.pixel.x) /
        options_.grid_size_pixels;
    const int cell_y = static_cast<int>(projection.pixel.y) /
        options_.grid_size_pixels;
    const int cell = cell_y * grid_cols + cell_x;
    const auto existing = grid.find(cell);
    if (existing == grid.end() || projection.camera_point.z() <
        existing->second.camera_point.z())
    {
      grid[cell] = projection;
    }
  }
  std::vector<Projection> active;
  active.reserve(grid.size());
  for (const auto &entry : grid) active.push_back(entry.second);
  std::sort(active.begin(), active.end(),
            [this](const Projection &left, const Projection &right)
            {
              const Landmark &first = landmarks_[left.landmark];
              const Landmark &second = landmarks_[right.landmark];
              const double first_score = first.semantic_weight *
                  (1.0 + std::min(10, first.observations));
              const double second_score = second.semantic_weight *
                  (1.0 + std::min(10, second.observations));
              return first_score > second_score;
            });
  if (active.size() > static_cast<std::size_t>(options_.max_active_landmarks))
  {
    active.resize(static_cast<std::size_t>(options_.max_active_landmarks));
  }

  const Eigen::Matrix3d camera_from_body =
      options_.body_from_camera_rotation.transpose();
  const Eigen::Matrix3d camera_from_world_rotation =
      camera_from_body * world_from_body.rotation().transpose();
  const double fx = options_.fx * options_.image_scale;
  const double fy = options_.fy * options_.image_scale;
  double squared_error = 0.0;
  double ncc_sum = 0.0;
  int inlier_landmarks = 0;
  int residuals = 0;
  const int half = options_.patch_half_size;
  for (const Projection &projection : active)
  {
    const Landmark &landmark = landmarks_[projection.landmark];
    std::vector<float> current_patch;
    double current_stddev = 0.0;
    if (!normalizedPatch(frame.gray, projection.pixel, &current_patch,
                         &current_stddev) ||
        current_patch.size() != landmark.patch.size())
    {
      cv::circle(debug_image_, projection.pixel, 2, cv::Scalar(0, 165, 255),
                 cv::FILLED, cv::LINE_AA);
      continue;
    }
    double ncc = 0.0;
    for (std::size_t index = 0; index < current_patch.size(); ++index)
    {
      ncc += static_cast<double>(current_patch[index]) * landmark.patch[index];
    }
    ncc /= static_cast<double>(current_patch.size());
    if (!std::isfinite(ncc) || ncc < options_.minimum_ncc)
    {
      cv::circle(debug_image_, projection.pixel, 2, cv::Scalar(30, 30, 230),
                 cv::FILLED, cv::LINE_AA);
      continue;
    }

    const double x = projection.camera_point.x();
    const double y = projection.camera_point.y();
    const double z = projection.camera_point.z();
    Eigen::Matrix<double, 2, 3> projection_jacobian;
    projection_jacobian << fx / z, 0.0, -fx * x / (z * z),
                           0.0, fy / z, -fy * y / (z * z);
    Eigen::Matrix<double, 3, 6> point_jacobian;
    point_jacobian.block<3, 3>(0, 0) =
        camera_from_body * skew(projection.body_point);
    point_jacobian.block<3, 3>(0, 3) = -camera_from_world_rotation;
    const Eigen::Matrix<double, 2, 6> pixel_jacobian =
        projection_jacobian * point_jacobian;
    const uint8_t current_label = semanticLabel(frame, projection.pixel);
    const double semantic_weight = std::sqrt(clampValue(
        landmark.semantic_weight * semanticClassWeight(current_label), 0.0, 4.0));
    const double base_information = options_.information_scale * semantic_weight /
        (options_.photometric_noise * options_.photometric_noise);
    std::size_t patch_index = 0U;
    for (int offset_y = -half; offset_y <= half; ++offset_y)
    {
      for (int offset_x = -half; offset_x <= half; ++offset_x, ++patch_index)
      {
        const float pixel_x = projection.pixel.x + static_cast<float>(offset_x);
        const float pixel_y = projection.pixel.y + static_cast<float>(offset_y);
        const double gradient_x = bilinear(frame.gradient_x, pixel_x, pixel_y) /
            current_stddev;
        const double gradient_y = bilinear(frame.gradient_y, pixel_x, pixel_y) /
            current_stddev;
        if (!std::isfinite(gradient_x) || !std::isfinite(gradient_y)) continue;
        const double residual = static_cast<double>(current_patch[patch_index]) -
                                landmark.patch[patch_index];
        Eigen::Matrix<double, 1, 6> jacobian =
            gradient_x * pixel_jacobian.row(0) +
            gradient_y * pixel_jacobian.row(1);
        const double information = base_information *
            huberWeight(residual, options_.photometric_huber_delta);
        result.hessian.noalias() +=
            information * jacobian.transpose() * jacobian;
        result.gradient.noalias() +=
            information * jacobian.transpose() * residual;
        squared_error += residual * residual;
        ++residuals;
      }
    }
    last_inlier_ids_.push_back(landmark.id);
    ++inlier_landmarks;
    ncc_sum += ncc;
    cv::circle(debug_image_, projection.pixel, 2, cv::Scalar(40, 220, 40),
               cv::FILLED, cv::LINE_AA);
  }

  result.landmarks = inlier_landmarks;
  result.residuals = residuals;
  result.rmse = residuals > 0 ?
      std::sqrt(squared_error / static_cast<double>(residuals)) :
      std::numeric_limits<double>::infinity();
  result.mean_ncc = inlier_landmarks > 0 ?
      ncc_sum / static_cast<double>(inlier_landmarks) : 0.0;
  result.valid = inlier_landmarks > 0 && residuals > 0 &&
      result.hessian.allFinite() && result.gradient.allFinite();
  result.reason = result.valid ? "visual_linearized" :
                                 "insufficient_visual_inliers";
  stats_.dynamic_rejections += static_cast<std::uint64_t>(dynamic_rejections);
  stats_.last_visible = static_cast<int>(active.size());
  stats_.last_inliers = inlier_landmarks;
  stats_.last_residuals = residuals;
  stats_.last_rmse = result.rmse;
  stats_.last_mean_ncc = result.mean_ncc;
  stats_.last_reason = result.reason;
  cv::putText(debug_image_,
              "visible=" + std::to_string(active.size()) +
                  " inliers=" + std::to_string(inlier_landmarks) +
                  " rmse=" + std::to_string(result.rmse),
              cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  return result;
}

void SparseVisualMap::commitFrame(
    const SparseVisualFrame &frame,
    const Eigen::Isometry3d &world_from_body,
    bool visual_update_accepted)
{
  if (!options_.enabled || !frame.valid()) return;
  ++stats_.frames;
  if (visual_update_accepted)
  {
    ++stats_.accepted_updates;
  }
  else
  {
    ++stats_.rejected_updates;
  }
  const std::unordered_set<int> inliers(last_inlier_ids_.begin(),
                                        last_inlier_ids_.end());
  for (Landmark &landmark : landmarks_)
  {
    cv::Point2f pixel;
    Eigen::Vector3d camera_point;
    if (!project(landmark.point, world_from_body, &pixel, &camera_point))
    {
      ++landmark.missed_frames;
      continue;
    }
    const uint8_t current_label = semanticLabel(frame, pixel);
    if (masked(frame, pixel) || semanticClassWeight(current_label) <= 0.0F)
    {
      landmark.semantic_weight = 0.0F;
      landmark.missed_frames = options_.max_missed_frames + 1;
      continue;
    }
    if (inliers.count(landmark.id) == 0U)
    {
      ++landmark.missed_frames;
      continue;
    }
    landmark.missed_frames = 0;
    ++landmark.observations;
    landmark.last_seen_stamp = frame.stamp;
    if (current_label != 0U)
    {
      landmark.label = current_label;
      landmark.semantic_weight = semanticClassWeight(current_label);
    }
    if (visual_update_accepted &&
        landmark.observations % options_.reference_refresh_observations == 0)
    {
      std::vector<float> refreshed;
      if (normalizedPatch(frame.gray, pixel, &refreshed))
      {
        landmark.patch.swap(refreshed);
      }
    }
  }
  cullLandmarks(world_from_body.translation());
  seedLandmarks(frame, world_from_body);
  stats_.landmarks = static_cast<int>(landmarks_.size());
}

void SparseVisualMap::seedLandmarks(
    const SparseVisualFrame &frame,
    const Eigen::Isometry3d &world_from_body)
{
  if (candidates_.empty() ||
      landmarks_.size() >= static_cast<std::size_t>(options_.max_landmarks))
  {
    candidates_.clear();
    return;
  }
  const int grid_cols = std::max(1, frame.gray.cols / options_.grid_size_pixels);
  std::unordered_set<int> occupied_cells;
  std::unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels;
  for (const Landmark &landmark : landmarks_)
  {
    occupied_voxels.insert(voxelKey(landmark.point, options_.landmark_voxel_size));
    cv::Point2f pixel;
    Eigen::Vector3d camera_point;
    if (project(landmark.point, world_from_body, &pixel, &camera_point))
    {
      const int cell_x = static_cast<int>(pixel.x) / options_.grid_size_pixels;
      const int cell_y = static_cast<int>(pixel.y) / options_.grid_size_pixels;
      occupied_cells.insert(cell_y * grid_cols + cell_x);
    }
  }

  struct Proposal
  {
    std::size_t candidate = 0U;
    cv::Point2f pixel;
    double score = 0.0;
  };
  std::vector<Proposal> proposals;
  proposals.reserve(candidates_.size());
  for (std::size_t index = 0; index < candidates_.size(); ++index)
  {
    const Candidate &candidate = candidates_[index];
    if (candidate.semantic_weight <= 0.0F ||
        occupied_voxels.count(voxelKey(candidate.point,
                                       options_.landmark_voxel_size)) > 0U)
    {
      continue;
    }
    cv::Point2f pixel;
    Eigen::Vector3d camera_point;
    if (!project(candidate.point, world_from_body, &pixel, &camera_point) ||
        masked(frame, pixel))
    {
      continue;
    }
    const uint8_t image_label = semanticLabel(frame, pixel);
    const float image_weight = semanticClassWeight(image_label);
    if (image_weight <= 0.0F) continue;
    const int border = options_.patch_half_size + 2;
    if (pixel.x < border || pixel.y < border ||
        pixel.x >= frame.gray.cols - border ||
        pixel.y >= frame.gray.rows - border)
    {
      continue;
    }
    const int cell_x = static_cast<int>(pixel.x) / options_.grid_size_pixels;
    const int cell_y = static_cast<int>(pixel.y) / options_.grid_size_pixels;
    const int cell = cell_y * grid_cols + cell_x;
    if (occupied_cells.count(cell) > 0U) continue;
    const double gx = bilinear(frame.gradient_x, pixel.x, pixel.y);
    const double gy = bilinear(frame.gradient_y, pixel.x, pixel.y);
    const double gradient = std::hypot(gx, gy);
    if (!std::isfinite(gradient) || gradient < options_.minimum_gradient) continue;
    Proposal proposal;
    proposal.candidate = index;
    proposal.pixel = pixel;
    proposal.score = gradient * candidate.semantic_weight * image_weight /
        std::sqrt(std::max(1.0, camera_point.z()));
    proposals.push_back(proposal);
  }
  std::sort(proposals.begin(), proposals.end(),
            [](const Proposal &left, const Proposal &right)
            {
              return left.score > right.score;
            });

  int added = 0;
  for (const Proposal &proposal : proposals)
  {
    if (added >= options_.max_new_landmarks_per_frame ||
        landmarks_.size() >= static_cast<std::size_t>(options_.max_landmarks))
    {
      break;
    }
    const int cell_x = static_cast<int>(proposal.pixel.x) /
        options_.grid_size_pixels;
    const int cell_y = static_cast<int>(proposal.pixel.y) /
        options_.grid_size_pixels;
    const int cell = cell_y * grid_cols + cell_x;
    if (!occupied_cells.insert(cell).second) continue;
    const Candidate &candidate = candidates_[proposal.candidate];
    const VoxelKey key = voxelKey(candidate.point, options_.landmark_voxel_size);
    if (!occupied_voxels.insert(key).second) continue;
    Landmark landmark;
    if (!normalizedPatch(frame.gray, proposal.pixel, &landmark.patch)) continue;
    landmark.id = next_landmark_id_++;
    landmark.point = candidate.point;
    const uint8_t image_label = semanticLabel(frame, proposal.pixel);
    landmark.label = image_label != 0U ? image_label : candidate.label;
    landmark.semantic_weight = candidate.semantic_weight *
        semanticClassWeight(landmark.label);
    landmark.observations = 1;
    landmark.last_seen_stamp = frame.stamp;
    landmarks_.push_back(std::move(landmark));
    ++added;
  }
  stats_.seeded_landmarks += static_cast<std::uint64_t>(added);
  candidates_.clear();
}

void SparseVisualMap::cullLandmarks(const Eigen::Vector3d &position)
{
  const std::size_t before = landmarks_.size();
  landmarks_.erase(std::remove_if(landmarks_.begin(), landmarks_.end(),
      [this, &position](const Landmark &landmark)
      {
        return landmark.missed_frames > options_.max_missed_frames ||
               !landmark.point.allFinite() ||
               (landmark.point - position).norm() > options_.local_map_radius;
      }), landmarks_.end());
  if (landmarks_.size() > static_cast<std::size_t>(options_.max_landmarks))
  {
    std::sort(landmarks_.begin(), landmarks_.end(),
              [](const Landmark &left, const Landmark &right)
              {
                if (left.observations != right.observations)
                  return left.observations > right.observations;
                return left.last_seen_stamp > right.last_seen_stamp;
              });
    landmarks_.resize(static_cast<std::size_t>(options_.max_landmarks));
  }
  stats_.culled_landmarks += static_cast<std::uint64_t>(
      before - landmarks_.size());
}

}  // namespace hybrid_localization
