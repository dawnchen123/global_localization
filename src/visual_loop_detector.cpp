#include "hybrid_localization/visual_loop_detector.h"

#include <Eigen/SVD>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

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

double yawOf(const Eigen::Isometry3d &pose)
{
  return std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
}

double rotationDegrees(const Eigen::Matrix3d &rotation)
{
  return std::acos(clampValue(0.5 * (rotation.trace() - 1.0), -1.0, 1.0)) *
         180.0 / kPi;
}

Eigen::Matrix3d projectRotation(const Eigen::Matrix3d &matrix)
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

Eigen::Isometry3d cleanPose(const Eigen::Isometry3d &pose)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = projectRotation(pose.rotation());
  result.translation() = pose.translation();
  return result;
}

}  // namespace

VisualLoopDetector::VisualLoopDetector(
    const VisualLoopDetectorOptions &options) : options_(options)
{
  options_.image_scale = clampValue(options_.image_scale, 0.1, 1.0);
  options_.maximum_features = std::max(200, options_.maximum_features);
  options_.minimum_depth_features = std::max(10, options_.minimum_depth_features);
  options_.maximum_database_size = std::max(10, options_.maximum_database_size);
  options_.keyframe_distance = std::max(0.0, options_.keyframe_distance);
  options_.keyframe_interval_sec = std::max(0.05, options_.keyframe_interval_sec);
  options_.minimum_index_gap = std::max(2, options_.minimum_index_gap);
  options_.search_radius = std::max(1.0, options_.search_radius);
  options_.maximum_retrieval_candidates = std::max(
      1, options_.maximum_retrieval_candidates);
  options_.maximum_geometric_candidates = std::max(
      1, options_.maximum_geometric_candidates);
  options_.descriptor_ratio = clampValue(options_.descriptor_ratio, 0.3, 0.95);
  options_.minimum_descriptor_matches = std::max(
      8, options_.minimum_descriptor_matches);
  options_.depth_association_radius_pixels = std::max(
      1, options_.depth_association_radius_pixels);
  options_.minimum_depth = std::max(0.1, options_.minimum_depth);
  options_.maximum_depth = std::max(
      options_.minimum_depth + 1.0, options_.maximum_depth);
  options_.minimum_pnp_inliers = std::max(6, options_.minimum_pnp_inliers);
  options_.minimum_pnp_inlier_ratio = clampValue(
      options_.minimum_pnp_inlier_ratio, 0.05, 1.0);
  options_.grid_rows = std::max(1, options_.grid_rows);
  options_.grid_cols = std::max(1, options_.grid_cols);
  options_.minimum_occupied_cells = std::max(1,
      std::min(options_.grid_rows * options_.grid_cols,
               options_.minimum_occupied_cells));
  options_.body_from_camera_rotation = projectRotation(
      options_.body_from_camera_rotation);
  orb_ = cv::ORB::create(options_.maximum_features, 1.2F, 8, 20, 0, 2,
                         cv::ORB::HARRIS_SCORE, 20, 12);
  reset();
}

void VisualLoopDetector::reset()
{
  keyframes_.clear();
  next_id_ = 0;
}

cv::Mat VisualLoopDetector::prepareImage(const cv::Mat &image) const
{
  if (image.empty()) return cv::Mat();
  cv::Mat gray;
  if (image.channels() == 1) gray = image;
  else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  else return cv::Mat();
  if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U);
  if (options_.image_scale < 0.999)
  {
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(), options_.image_scale,
               options_.image_scale, cv::INTER_AREA);
    gray = resized;
  }
  cv::equalizeHist(gray, gray);
  return gray;
}

cv::Mat VisualLoopDetector::featureMask(const cv::Mat &dynamic_mask,
                                        const cv::Size &size) const
{
  if (dynamic_mask.empty()) return cv::Mat();
  cv::Mat mask;
  if (dynamic_mask.channels() == 1) mask = dynamic_mask;
  else cv::cvtColor(dynamic_mask, mask, cv::COLOR_BGR2GRAY);
  if (mask.depth() != CV_8U) mask.convertTo(mask, CV_8U);
  if (mask.size() != size)
  {
    cv::resize(mask, mask, size, 0.0, 0.0, cv::INTER_NEAREST);
  }
  cv::threshold(mask, mask, 0.0, 255.0, cv::THRESH_BINARY_INV);
  return mask;
}

VisualLoopDetector::Keyframe VisualLoopDetector::buildKeyframe(
    double stamp, const cv::Mat &gray,
    const VisualLidarPointVector &body_points,
    const Eigen::Isometry3d &raw_pose, const cv::Mat &mask)
{
  Keyframe keyframe;
  keyframe.id = next_id_;
  keyframe.stamp = stamp;
  keyframe.raw_pose = cleanPose(raw_pose);
  keyframe.gray = gray.clone();
  orb_->detectAndCompute(keyframe.gray, mask, keyframe.keypoints,
                         keyframe.descriptors, false);
  associateDepth(&keyframe, body_points);
  return keyframe;
}

void VisualLoopDetector::associateDepth(
    Keyframe *keyframe, const VisualLidarPointVector &body_points) const
{
  if (keyframe == nullptr) return;
  keyframe->depth_points.assign(keyframe->keypoints.size(), Eigen::Vector3d::Zero());
  keyframe->has_depth.assign(keyframe->keypoints.size(), 0U);
  if (keyframe->gray.empty() || keyframe->keypoints.empty() || body_points.empty()) return;

  const int width = keyframe->gray.cols;
  const int height = keyframe->gray.rows;
  std::vector<int> point_index(static_cast<std::size_t>(width * height), -1);
  std::vector<double> depth(static_cast<std::size_t>(width * height),
                            std::numeric_limits<double>::infinity());
  const Eigen::Matrix3d camera_from_body =
      options_.body_from_camera_rotation.transpose();
  const double fx = options_.fx * options_.image_scale;
  const double fy = options_.fy * options_.image_scale;
  const double cx = options_.cx * options_.image_scale;
  const double cy = options_.cy * options_.image_scale;
  for (std::size_t index = 0U; index < body_points.size(); ++index)
  {
    const Eigen::Vector3d camera_point = camera_from_body *
        (body_points[index] - options_.body_from_camera_translation);
    if (!camera_point.allFinite() || camera_point.z() < options_.minimum_depth ||
        camera_point.z() > options_.maximum_depth) continue;
    const double x = camera_point.x() / camera_point.z();
    const double y = camera_point.y() / camera_point.z();
    const double r2 = x * x + y * y;
    const double radial = 1.0 + options_.distortion[0] * r2 +
        options_.distortion[1] * r2 * r2 +
        options_.distortion[4] * r2 * r2 * r2;
    const double distorted_x = x * radial +
        2.0 * options_.distortion[2] * x * y +
        options_.distortion[3] * (r2 + 2.0 * x * x);
    const double distorted_y = y * radial +
        options_.distortion[2] * (r2 + 2.0 * y * y) +
        2.0 * options_.distortion[3] * x * y;
    const int u = static_cast<int>(std::lround(fx * distorted_x + cx));
    const int v = static_cast<int>(std::lround(fy * distorted_y + cy));
    if (u < 0 || v < 0 || u >= width || v >= height) continue;
    const std::size_t cell = static_cast<std::size_t>(v * width + u);
    if (camera_point.z() < depth[cell])
    {
      depth[cell] = camera_point.z();
      point_index[cell] = static_cast<int>(index);
    }
  }

  const int radius = options_.depth_association_radius_pixels;
  for (std::size_t feature = 0U; feature < keyframe->keypoints.size(); ++feature)
  {
    const cv::Point2f pixel = keyframe->keypoints[feature].pt;
    int best = -1;
    double best_distance = static_cast<double>(radius * radius + 1);
    double best_depth = std::numeric_limits<double>::infinity();
    for (int dy = -radius; dy <= radius; ++dy)
    {
      for (int dx = -radius; dx <= radius; ++dx)
      {
        const int u = static_cast<int>(std::lround(pixel.x)) + dx;
        const int v = static_cast<int>(std::lround(pixel.y)) + dy;
        if (u < 0 || v < 0 || u >= width || v >= height) continue;
        const std::size_t cell = static_cast<std::size_t>(v * width + u);
        if (point_index[cell] < 0) continue;
        const double squared = std::pow(static_cast<double>(u) - pixel.x, 2.0) +
                               std::pow(static_cast<double>(v) - pixel.y, 2.0);
        if (squared < best_distance ||
            (std::abs(squared - best_distance) < 1e-9 && depth[cell] < best_depth))
        {
          best = point_index[cell];
          best_distance = squared;
          best_depth = depth[cell];
        }
      }
    }
    if (best >= 0)
    {
      keyframe->depth_points[feature] = body_points[static_cast<std::size_t>(best)];
      keyframe->has_depth[feature] = 1U;
    }
  }
}

VisualLoopDetector::RetrievalCandidate VisualLoopDetector::retrieve(
    const Keyframe &reference, const Keyframe &current,
    std::size_t database_index) const
{
  RetrievalCandidate candidate;
  candidate.database_index = database_index;
  if (reference.descriptors.empty() || current.descriptors.empty()) return candidate;
  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> nearest;
  matcher.knnMatch(current.descriptors, reference.descriptors, nearest, 2);
  std::vector<cv::DMatch> filtered;
  filtered.reserve(nearest.size());
  for (const auto &pair : nearest)
  {
    if (pair.size() < 2U || pair[0].distance >=
        options_.descriptor_ratio * pair[1].distance) continue;
    const int reference_index = pair[0].trainIdx;
    if (reference_index < 0 || static_cast<std::size_t>(reference_index) >=
        reference.has_depth.size() || !reference.has_depth[reference_index]) continue;
    filtered.push_back(pair[0]);
  }
  std::sort(filtered.begin(), filtered.end(),
            [](const cv::DMatch &left, const cv::DMatch &right)
            {
              return left.distance < right.distance;
            });
  std::unordered_set<int> used_current;
  std::unordered_set<int> used_reference;
  for (const cv::DMatch &match : filtered)
  {
    if (!used_current.insert(match.queryIdx).second ||
        !used_reference.insert(match.trainIdx).second) continue;
    candidate.correspondences.push_back(match);
  }
  candidate.matches = static_cast<int>(candidate.correspondences.size());
  candidate.score = candidate.matches / static_cast<double>(std::max(
      1, std::min(current.descriptors.rows, reference.descriptors.rows)));
  return candidate;
}

VisualLoopResult VisualLoopDetector::verify(
    const Keyframe &reference, const Keyframe &current,
    const RetrievalCandidate &candidate) const
{
  VisualLoopResult result;
  result.keyframe_created = true;
  result.candidate_found = true;
  result.reference_id = reference.id;
  result.current_id = current.id;
  result.reference_stamp = reference.stamp;
  result.current_stamp = current.stamp;
  result.descriptor_matches = candidate.matches;
  result.descriptor_score = candidate.score;
  if (candidate.matches < options_.minimum_descriptor_matches)
  {
    result.reason = "visual_loop_insufficient_descriptor_matches";
    return result;
  }

  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(candidate.correspondences.size());
  image_points.reserve(candidate.correspondences.size());
  for (const cv::DMatch &match : candidate.correspondences)
  {
    const Eigen::Vector3d &point = reference.depth_points[
        static_cast<std::size_t>(match.trainIdx)];
    object_points.emplace_back(static_cast<float>(point.x()),
                               static_cast<float>(point.y()),
                               static_cast<float>(point.z()));
    image_points.push_back(current.keypoints[static_cast<std::size_t>(match.queryIdx)].pt);
  }
  const double scale = options_.image_scale;
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
      options_.fx * scale, 0.0, options_.cx * scale,
      0.0, options_.fy * scale, options_.cy * scale,
      0.0, 0.0, 1.0);
  const cv::Mat distortion = (cv::Mat_<double>(1, 5) <<
      options_.distortion[0], options_.distortion[1], options_.distortion[2],
      options_.distortion[3], options_.distortion[4]);
  cv::Mat rotation_vector;
  cv::Mat translation_vector;
  cv::Mat inliers;
  const bool solved = cv::solvePnPRansac(
      object_points, image_points, camera_matrix, distortion, rotation_vector,
      translation_vector, false, options_.pnp_iterations,
      options_.pnp_reprojection_error, 0.999, inliers, cv::SOLVEPNP_EPNP);
  if (!solved || inliers.empty())
  {
    result.reason = "visual_loop_pnp_failed";
    return result;
  }
  result.pnp_inliers = inliers.rows;
  result.inlier_ratio = result.pnp_inliers /
      static_cast<double>(std::max(1, candidate.matches));
  if (result.pnp_inliers < options_.minimum_pnp_inliers ||
      result.inlier_ratio < options_.minimum_pnp_inlier_ratio)
  {
    result.reason = "visual_loop_pnp_inlier_gate";
    return result;
  }

  std::vector<cv::Point3f> inlier_objects;
  std::vector<cv::Point2f> inlier_images;
  std::set<int> occupied;
  std::vector<uint8_t> inlier_mask(candidate.correspondences.size(), 0U);
  for (int row = 0; row < inliers.rows; ++row)
  {
    const int index = inliers.at<int>(row, 0);
    if (index < 0 || static_cast<std::size_t>(index) >= object_points.size()) continue;
    inlier_objects.push_back(object_points[static_cast<std::size_t>(index)]);
    inlier_images.push_back(image_points[static_cast<std::size_t>(index)]);
    inlier_mask[static_cast<std::size_t>(index)] = 1U;
    const cv::Point2f pixel = image_points[static_cast<std::size_t>(index)];
    const int col = std::min(options_.grid_cols - 1, std::max(0,
        static_cast<int>(pixel.x * options_.grid_cols /
                         std::max(1, current.gray.cols))));
    const int grid_row = std::min(options_.grid_rows - 1, std::max(0,
        static_cast<int>(pixel.y * options_.grid_rows /
                         std::max(1, current.gray.rows))));
    occupied.insert(grid_row * options_.grid_cols + col);
  }
  result.occupied_cells = static_cast<int>(occupied.size());
  if (result.occupied_cells < options_.minimum_occupied_cells)
  {
    result.reason = "visual_loop_spatial_coverage_gate";
    return result;
  }
  cv::solvePnP(inlier_objects, inlier_images, camera_matrix, distortion,
               rotation_vector, translation_vector, true,
               cv::SOLVEPNP_ITERATIVE);
  std::vector<cv::Point2f> projected;
  cv::projectPoints(inlier_objects, rotation_vector, translation_vector,
                    camera_matrix, distortion, projected);
  double squared_error = 0.0;
  for (std::size_t index = 0U; index < projected.size(); ++index)
  {
    const cv::Point2f difference = projected[index] - inlier_images[index];
    squared_error += difference.dot(difference);
  }
  result.reprojection_rmse = projected.empty() ?
      std::numeric_limits<double>::infinity() :
      std::sqrt(squared_error / static_cast<double>(projected.size()));
  if (!std::isfinite(result.reprojection_rmse) ||
      result.reprojection_rmse > options_.maximum_reprojection_rmse)
  {
    result.reason = "visual_loop_reprojection_gate";
    return result;
  }

  cv::Mat rotation_cv;
  cv::Rodrigues(rotation_vector, rotation_cv);
  Eigen::Isometry3d camera_current_from_body_reference =
      Eigen::Isometry3d::Identity();
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
      camera_current_from_body_reference.linear()(row, col) =
          rotation_cv.at<double>(row, col);
    camera_current_from_body_reference.translation()(row) =
        translation_vector.at<double>(row, 0);
  }
  Eigen::Isometry3d body_current_from_camera_current =
      Eigen::Isometry3d::Identity();
  body_current_from_camera_current.linear() =
      options_.body_from_camera_rotation;
  body_current_from_camera_current.translation() =
      options_.body_from_camera_translation;
  const Eigen::Isometry3d body_current_from_body_reference = cleanPose(
      body_current_from_camera_current * camera_current_from_body_reference);
  result.reference_from_current = cleanPose(
      body_current_from_body_reference.inverse());
  const Eigen::Isometry3d raw_relative = cleanPose(
      reference.raw_pose.inverse() * current.raw_pose);
  const Eigen::Isometry3d disagreement = cleanPose(
      raw_relative.inverse() * result.reference_from_current);
  result.translation_disagreement = disagreement.translation().norm();
  result.rotation_disagreement_deg = rotationDegrees(disagreement.rotation());
  if (result.translation_disagreement >
          options_.maximum_translation_disagreement ||
      result.rotation_disagreement_deg >
          options_.maximum_rotation_disagreement_deg)
  {
    result.reason = "visual_loop_pose_consistency_gate";
    return result;
  }

  const double match_quality = std::min(1.0, candidate.matches /
      static_cast<double>(std::max(1, options_.minimum_descriptor_matches * 2)));
  const double coverage_quality = std::min(1.0, result.occupied_cells /
      static_cast<double>(std::max(1, options_.minimum_occupied_cells * 2)));
  const double reprojection_quality = std::exp(-result.reprojection_rmse /
      std::max(0.1, options_.maximum_reprojection_rmse));
  result.quality = 0.25 * match_quality + 0.35 * result.inlier_ratio +
                   0.20 * coverage_quality + 0.20 * reprojection_quality;
  if (result.quality < options_.minimum_quality)
  {
    result.reason = "visual_loop_quality_gate";
    return result;
  }

  std::vector<cv::DMatch> draw_matches;
  std::vector<char> draw_mask;
  draw_matches.reserve(candidate.correspondences.size());
  draw_mask.reserve(candidate.correspondences.size());
  for (std::size_t index = 0U; index < candidate.correspondences.size(); ++index)
  {
    const cv::DMatch &match = candidate.correspondences[index];
    draw_matches.emplace_back(match.trainIdx, match.queryIdx, match.distance);
    draw_mask.push_back(inlier_mask[index] ? 1 : 0);
  }
  cv::drawMatches(reference.gray, reference.keypoints, current.gray,
                  current.keypoints, draw_matches, result.debug_image,
                  cv::Scalar(40, 220, 40), cv::Scalar(100, 100, 100), draw_mask,
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  result.accepted = true;
  result.reason = "visual_loop_accepted";
  return result;
}

VisualLoopResult VisualLoopDetector::process(
    double stamp, const cv::Mat &image,
    const VisualLidarPointVector &body_points,
    const Eigen::Isometry3d &world_from_body, const cv::Mat &dynamic_mask)
{
  VisualLoopResult result;
  result.current_stamp = stamp;
  if (!options_.enabled)
  {
    result.reason = "visual_loop_disabled";
    return result;
  }
  if (!std::isfinite(stamp) || image.empty() || body_points.empty() ||
      !world_from_body.matrix().allFinite())
  {
    result.reason = "visual_loop_invalid_input";
    return result;
  }
  const cv::Mat gray = prepareImage(image);
  if (gray.empty())
  {
    result.reason = "visual_loop_invalid_image";
    return result;
  }
  if (!keyframes_.empty())
  {
    const Keyframe &last = keyframes_.back();
    const Eigen::Isometry3d motion = last.raw_pose.inverse() * world_from_body;
    if (motion.translation().norm() < options_.keyframe_distance &&
        stamp - last.stamp < options_.keyframe_interval_sec)
    {
      result.reason = "visual_loop_minimum_interval";
      return result;
    }
  }

  Keyframe current = buildKeyframe(
      stamp, gray, body_points, world_from_body,
      featureMask(dynamic_mask, gray.size()));
  result.current_id = current.id;
  const int depth_features = static_cast<int>(std::count(
      current.has_depth.begin(), current.has_depth.end(), 1U));
  if (current.descriptors.empty() ||
      depth_features < options_.minimum_depth_features)
  {
    result.reason = "visual_loop_insufficient_depth_features";
    return result;
  }
  result.keyframe_created = true;

  std::vector<std::pair<double, std::size_t>> spatial;
  for (std::size_t index = 0U; index < keyframes_.size(); ++index)
  {
    const Keyframe &reference = keyframes_[index];
    if (current.id - reference.id < options_.minimum_index_gap) continue;
    const double distance = (reference.raw_pose.translation() -
                             current.raw_pose.translation()).head<2>().norm();
    const double yaw_difference = std::abs(wrapAngle(
        yawOf(reference.raw_pose) - yawOf(current.raw_pose))) * 180.0 / kPi;
    if (distance <= options_.search_radius &&
        yaw_difference <= options_.maximum_yaw_difference_deg)
    {
      spatial.emplace_back(distance, index);
    }
  }
  std::sort(spatial.begin(), spatial.end());
  if (spatial.size() >
      static_cast<std::size_t>(options_.maximum_retrieval_candidates))
  {
    spatial.resize(static_cast<std::size_t>(
        options_.maximum_retrieval_candidates));
  }
  std::vector<RetrievalCandidate> candidates;
  candidates.reserve(spatial.size());
  for (const auto &item : spatial)
  {
    RetrievalCandidate candidate = retrieve(keyframes_[item.second], current,
                                            item.second);
    if (candidate.matches >= options_.minimum_descriptor_matches)
      candidates.push_back(std::move(candidate));
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const RetrievalCandidate &left,
               const RetrievalCandidate &right)
            {
              if (left.matches != right.matches) return left.matches > right.matches;
              return left.score > right.score;
            });
  if (candidates.size() >
      static_cast<std::size_t>(options_.maximum_geometric_candidates))
  {
    candidates.resize(static_cast<std::size_t>(
        options_.maximum_geometric_candidates));
  }

  VisualLoopResult best;
  best.current_id = current.id;
  best.current_stamp = stamp;
  best.keyframe_created = true;
  best.reason = candidates.empty() ? "visual_loop_no_candidate" :
                                     "visual_loop_geometric_rejection";
  for (const RetrievalCandidate &candidate : candidates)
  {
    const Keyframe &reference = keyframes_[candidate.database_index];
    VisualLoopResult verified = verify(reference, current, candidate);
    if (verified.accepted && (!best.accepted || verified.quality > best.quality))
      best = std::move(verified);
    else if (!best.accepted && verified.pnp_inliers > best.pnp_inliers)
      best = std::move(verified);
  }

  keyframes_.push_back(std::move(current));
  ++next_id_;
  while (keyframes_.size() >
      static_cast<std::size_t>(options_.maximum_database_size))
  {
    keyframes_.pop_front();
  }
  return best;
}

}  // namespace hybrid_localization
