#include "hybrid_localization/visual_rotation_tracker.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
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

double median(std::vector<double> values)
{
  if (values.empty()) return 0.0;
  const std::size_t middle = values.size() / 2U;
  std::nth_element(values.begin(), values.begin() + static_cast<long>(middle), values.end());
  double result = values[middle];
  if (values.size() % 2U == 0U)
  {
    const auto lower = std::max_element(values.begin(),
                                        values.begin() + static_cast<long>(middle));
    result = 0.5 * (result + *lower);
  }
  return result;
}

Eigen::Matrix3d projectToRotation(const Eigen::Matrix3d &matrix)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
  if (rotation.determinant() < 0.0)
  {
    Eigen::Matrix3d u = svd.matrixU();
    u.col(2) *= -1.0;
    rotation = u * svd.matrixV().transpose();
  }
  return rotation;
}

Eigen::Matrix3d eigenRotation(const cv::Mat &rotation)
{
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      result(row, col) = rotation.at<double>(row, col);
    }
  }
  return projectToRotation(result);
}

cv::Mat bgrImage(const cv::Mat &gray)
{
  if (gray.empty()) return cv::Mat();
  cv::Mat bgr;
  if (gray.channels() == 1)
  {
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
  }
  else
  {
    bgr = gray.clone();
  }
  return bgr;
}

void drawTracks(const cv::Mat &gray,
                const std::vector<cv::Point2f> &reference_points,
                const std::vector<cv::Point2f> &current_points,
                const cv::Mat &inlier_mask, cv::Mat *debug)
{
  if (debug == nullptr) return;
  *debug = bgrImage(gray);
  const std::size_t count = std::min(reference_points.size(), current_points.size());
  for (std::size_t index = 0; index < count; ++index)
  {
    bool inlier = false;
    bool classified = !inlier_mask.empty();
    if (classified)
    {
      const int flat_index = static_cast<int>(index);
      if (inlier_mask.rows == 1 && flat_index < inlier_mask.cols)
      {
        inlier = inlier_mask.at<uint8_t>(0, flat_index) != 0U;
      }
      else if (flat_index < inlier_mask.rows)
      {
        inlier = inlier_mask.at<uint8_t>(flat_index, 0) != 0U;
      }
    }
    const cv::Scalar color = classified ?
        (inlier ? cv::Scalar(50, 220, 50) : cv::Scalar(40, 40, 230)) :
        cv::Scalar(0, 210, 255);
    cv::line(*debug, reference_points[index], current_points[index], color, 1,
             cv::LINE_AA);
    cv::circle(*debug, current_points[index], 2, color, cv::FILLED, cv::LINE_AA);
  }
}

void drawPnpCorrespondences(const cv::Mat &gray,
                            const std::vector<cv::Point2f> &image_points,
                            const cv::Mat &inliers, cv::Mat *debug)
{
  if (debug == nullptr) return;
  *debug = bgrImage(gray);
  std::vector<uint8_t> is_inlier(image_points.size(), 0U);
  for (int row = 0; row < inliers.rows; ++row)
  {
    const int index = inliers.at<int>(row, 0);
    if (index >= 0 && static_cast<std::size_t>(index) < is_inlier.size())
    {
      is_inlier[static_cast<std::size_t>(index)] = 1U;
    }
  }
  for (std::size_t index = 0; index < image_points.size(); ++index)
  {
    const cv::Scalar color = is_inlier[index] ? cv::Scalar(50, 220, 50) :
                                               cv::Scalar(120, 120, 120);
    cv::circle(*debug, image_points[index], is_inlier[index] ? 3 : 2,
               color, cv::FILLED, cv::LINE_AA);
  }
}

}  // namespace

VisualRotationTracker::VisualRotationTracker(const VisualRotationTrackerOptions &options)
    : options_(options)
{
  options_.image_scale = clampValue(options_.image_scale, 0.1, 1.0);
  options_.minimum_interval = std::max(0.0, options_.minimum_interval);
  options_.maximum_reference_gap = std::max(0.1, options_.maximum_reference_gap);
  options_.maximum_features = std::max(100, options_.maximum_features);
  options_.minimum_tracks = std::max(12, options_.minimum_tracks);
  options_.minimum_inliers = std::max(8, options_.minimum_inliers);
  options_.feature_quality = clampValue(options_.feature_quality, 1e-5, 0.2);
  options_.feature_minimum_distance = std::max(2.0, options_.feature_minimum_distance);
  options_.forward_backward_error = std::max(0.1, options_.forward_backward_error);
  options_.minimum_median_parallax = std::max(0.0, options_.minimum_median_parallax);
  options_.ransac_probability = clampValue(options_.ransac_probability, 0.8, 0.9999);
  options_.ransac_threshold_pixels = std::max(0.2, options_.ransac_threshold_pixels);
  options_.minimum_inlier_ratio = clampValue(options_.minimum_inlier_ratio, 0.05, 1.0);
  options_.grid_rows = std::max(1, options_.grid_rows);
  options_.grid_cols = std::max(1, options_.grid_cols);
  options_.minimum_occupied_cells = std::max(1, options_.minimum_occupied_cells);
  options_.body_from_camera_rotation =
      projectToRotation(options_.body_from_camera_rotation);
  if (!options_.body_from_camera_translation.allFinite())
  {
    options_.body_from_camera_translation.setZero();
  }
  options_.pnp_association_radius_pixels = std::max(1, options_.pnp_association_radius_pixels);
  options_.pnp_minimum_correspondences = std::max(6, options_.pnp_minimum_correspondences);
  options_.pnp_minimum_inliers = std::max(6, options_.pnp_minimum_inliers);
  options_.pnp_minimum_occupied_cells = std::max(1, options_.pnp_minimum_occupied_cells);
  options_.pnp_minimum_inlier_ratio = clampValue(
      options_.pnp_minimum_inlier_ratio, 0.05, 1.0);
  options_.pnp_ransac_reprojection_error = std::max(
      0.2, options_.pnp_ransac_reprojection_error);
  options_.pnp_ransac_iterations = std::max(20, options_.pnp_ransac_iterations);
  options_.pnp_maximum_reprojection_rmse = std::max(
      0.2, options_.pnp_maximum_reprojection_rmse);
  options_.pnp_maximum_local_depth_difference = std::max(
      0.1, options_.pnp_maximum_local_depth_difference);
  options_.pnp_maximum_local_depth_ratio = clampValue(
      options_.pnp_maximum_local_depth_ratio, 0.0, 1.0);
  options_.pnp_minimum_depth = std::max(0.1, options_.pnp_minimum_depth);
  options_.pnp_maximum_depth = std::max(
      options_.pnp_minimum_depth + 1.0, options_.pnp_maximum_depth);
  options_.pnp_maximum_translation_speed = std::max(
      0.1, options_.pnp_maximum_translation_speed);
  options_.pnp_maximum_translation_step = std::max(
      0.1, options_.pnp_maximum_translation_step);
}

void VisualRotationTracker::reset()
{
  reference_gray_.release();
  reference_body_points_.clear();
  reference_stamp_ = 0.0;
  initialized_ = false;
  segment_ = 0;
  visual_from_body_rotation_.setIdentity();
  visual_from_body_pose_.setIdentity();
}

cv::Mat VisualRotationTracker::prepareImage(const cv::Mat &image) const
{
  if (image.empty()) return cv::Mat();
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
    return cv::Mat();
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
    cv::resize(gray, resized, cv::Size(), options_.image_scale, options_.image_scale,
               cv::INTER_AREA);
    gray = resized;
  }
  if (options_.equalize_histogram)
  {
    cv::Mat equalized;
    cv::equalizeHist(gray, equalized);
    gray = equalized;
  }
  return gray.clone();
}

cv::Mat VisualRotationTracker::projectionDebugImage(
    const cv::Mat &gray, const VisualLidarPointVector &body_points) const
{
  if (!options_.generate_debug_images) return cv::Mat();
  cv::Mat debug = bgrImage(gray);
  if (debug.empty() || body_points.empty()) return debug;

  const double scale = options_.image_scale;
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
      options_.fx * scale, 0.0, options_.cx * scale,
      0.0, options_.fy * scale, options_.cy * scale,
      0.0, 0.0, 1.0);
  const cv::Mat distortion = (cv::Mat_<double>(1, 5) <<
      options_.distortion[0], options_.distortion[1], options_.distortion[2],
      options_.distortion[3], options_.distortion[4]);
  const Eigen::Matrix3d camera_from_body_rotation =
      options_.body_from_camera_rotation.transpose();
  std::vector<cv::Point3f> camera_points;
  camera_points.reserve(body_points.size());
  for (const Eigen::Vector3d &body_point : body_points)
  {
    const Eigen::Vector3d camera_point = camera_from_body_rotation *
        (body_point - options_.body_from_camera_translation);
    if (!camera_point.allFinite() || camera_point.z() < options_.pnp_minimum_depth ||
        camera_point.z() > options_.pnp_maximum_depth) continue;
    camera_points.emplace_back(static_cast<float>(camera_point.x()),
                               static_cast<float>(camera_point.y()),
                               static_cast<float>(camera_point.z()));
  }
  if (camera_points.empty()) return debug;

  std::vector<cv::Point2f> projected;
  cv::projectPoints(camera_points, cv::Mat::zeros(3, 1, CV_64F),
                    cv::Mat::zeros(3, 1, CV_64F), camera_matrix, distortion,
                    projected);
  cv::Mat z_buffer(gray.rows, gray.cols, CV_32F,
                   cv::Scalar(std::numeric_limits<float>::infinity()));
  for (std::size_t index = 0; index < projected.size(); ++index)
  {
    const int x = static_cast<int>(std::lround(projected[index].x));
    const int y = static_cast<int>(std::lround(projected[index].y));
    if (x < 0 || y < 0 || x >= gray.cols || y >= gray.rows) continue;
    z_buffer.at<float>(y, x) = std::min(z_buffer.at<float>(y, x),
                                       camera_points[index].z);
  }
  for (std::size_t index = 0; index < projected.size(); ++index)
  {
    const int x = static_cast<int>(std::lround(projected[index].x));
    const int y = static_cast<int>(std::lround(projected[index].y));
    if (x < 0 || y < 0 || x >= gray.cols || y >= gray.rows) continue;
    if (camera_points[index].z > z_buffer.at<float>(y, x) + 0.05F) continue;
    const double normalized_depth = clampValue(
        (camera_points[index].z - options_.pnp_minimum_depth) /
            std::max(1.0, options_.pnp_maximum_depth - options_.pnp_minimum_depth),
        0.0, 1.0);
    const cv::Scalar color(255.0 * (1.0 - normalized_depth),
                           255.0 * (1.0 - std::abs(2.0 * normalized_depth - 1.0)),
                           255.0 * normalized_depth);
    cv::circle(debug, projected[index], 1, color, cv::FILLED, cv::LINE_AA);
  }
  return debug;
}

VisualRotationEstimate VisualRotationTracker::initializeReference(
    double stamp, const cv::Mat &gray, const VisualLidarPointVector &body_points,
    const std::string &reason, bool new_segment)
{
  if (new_segment) ++segment_;
  reference_gray_ = gray.clone();
  reference_body_points_ = body_points;
  reference_stamp_ = stamp;
  initialized_ = true;
  visual_from_body_rotation_.setIdentity();
  visual_from_body_pose_.setIdentity();
  VisualRotationEstimate result;
  result.observation_valid = true;
  result.metric_pose_valid = options_.enable_lidar_pnp && !body_points.empty();
  result.stamp = stamp;
  result.segment = segment_;
  result.quality = 1.0;
  result.visual_from_body_rotation = visual_from_body_rotation_;
  result.visual_from_body_pose = visual_from_body_pose_;
  result.reason = reason;
  return result;
}

VisualRotationEstimate VisualRotationTracker::process(double stamp, const cv::Mat &image)
{
  return process(stamp, image, VisualLidarPointVector());
}

VisualRotationEstimate VisualRotationTracker::process(
    double stamp, const cv::Mat &image, const VisualLidarPointVector &body_points)
{
  VisualRotationEstimate result;
  result.stamp = stamp;
  result.segment = segment_;
  if (!std::isfinite(stamp))
  {
    result.reason = "invalid_stamp";
    return result;
  }
  const cv::Mat gray = prepareImage(image);
  if (gray.empty())
  {
    result.reason = "invalid_image";
    return result;
  }
  if (options_.generate_debug_images)
  {
    result.projection_debug_image = projectionDebugImage(gray, body_points);
    result.tracking_debug_image = bgrImage(gray);
    result.pnp_debug_image = bgrImage(gray);
  }
  if (!initialized_)
  {
    VisualRotationEstimate initialized = initializeReference(
        stamp, gray, body_points, "initialized", true);
    initialized.projection_debug_image = result.projection_debug_image;
    initialized.tracking_debug_image = result.tracking_debug_image;
    initialized.pnp_debug_image = result.pnp_debug_image;
    return initialized;
  }
  const double dt = stamp - reference_stamp_;
  if (dt <= 0.0)
  {
    result.reason = "non_monotonic_stamp";
    return result;
  }
  if (dt < options_.minimum_interval)
  {
    result.reason = "minimum_interval";
    return result;
  }
  if (dt > options_.maximum_reference_gap || gray.size() != reference_gray_.size())
  {
    VisualRotationEstimate reset = initializeReference(
        stamp, gray, body_points, "reference_reset", true);
    reset.projection_debug_image = result.projection_debug_image;
    reset.tracking_debug_image = result.tracking_debug_image;
    reset.pnp_debug_image = result.pnp_debug_image;
    return reset;
  }

  std::vector<cv::Point2f> reference_points;
  cv::goodFeaturesToTrack(reference_gray_, reference_points, options_.maximum_features,
                          options_.feature_quality, options_.feature_minimum_distance);
  if (static_cast<int>(reference_points.size()) < options_.minimum_tracks)
  {
    result.reason = "insufficient_features";
    return result;
  }
  cv::cornerSubPix(reference_gray_, reference_points, cv::Size(5, 5), cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                    20, 0.01));

  std::vector<cv::Point2f> current_points;
  std::vector<uint8_t> forward_status;
  std::vector<float> forward_error;
  cv::calcOpticalFlowPyrLK(reference_gray_, gray, reference_points, current_points,
                           forward_status, forward_error, cv::Size(21, 21), 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                            30, 0.01));
  std::vector<cv::Point2f> backward_points;
  std::vector<uint8_t> backward_status;
  std::vector<float> backward_error;
  cv::calcOpticalFlowPyrLK(gray, reference_gray_, current_points, backward_points,
                           backward_status, backward_error, cv::Size(21, 21), 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                            30, 0.01));

  std::vector<cv::Point2f> first;
  std::vector<cv::Point2f> second;
  std::vector<double> parallaxes;
  std::vector<uint8_t> occupied(static_cast<std::size_t>(options_.grid_rows *
                                                         options_.grid_cols), 0U);
  const float width = static_cast<float>(gray.cols);
  const float height = static_cast<float>(gray.rows);
  const std::size_t tracked_count = std::min(
      std::min(reference_points.size(), current_points.size()),
      std::min(std::min(forward_status.size(), backward_status.size()),
               backward_points.size()));
  for (std::size_t index = 0; index < tracked_count; ++index)
  {
    if (!forward_status[index] || !backward_status[index]) continue;
    const cv::Point2f &current = current_points[index];
    if (current.x < 1.0F || current.y < 1.0F || current.x >= width - 1.0F ||
        current.y >= height - 1.0F) continue;
    if (cv::norm(reference_points[index] - backward_points[index]) >
        options_.forward_backward_error) continue;
    first.push_back(reference_points[index]);
    second.push_back(current);
    parallaxes.push_back(cv::norm(reference_points[index] - current));
    const int row = std::min(options_.grid_rows - 1,
        static_cast<int>(current.y / std::max(1.0F, height) * options_.grid_rows));
    const int col = std::min(options_.grid_cols - 1,
        static_cast<int>(current.x / std::max(1.0F, width) * options_.grid_cols));
    occupied[static_cast<std::size_t>(row * options_.grid_cols + col)] = 1U;
  }
  result.tracks = static_cast<int>(first.size());
  result.occupied_cells = static_cast<int>(std::count(occupied.begin(), occupied.end(), 1U));
  result.median_parallax = median(parallaxes);
  if (options_.generate_debug_images)
  {
    drawTracks(gray, first, second, cv::Mat(), &result.tracking_debug_image);
  }
  if (result.tracks < options_.minimum_tracks)
  {
    result.reason = "insufficient_tracks";
    return result;
  }
  if (result.occupied_cells < options_.minimum_occupied_cells)
  {
    result.reason = "insufficient_spatial_coverage";
    return result;
  }
  if (result.median_parallax < options_.minimum_median_parallax)
  {
    result.reason = "insufficient_parallax";
    return result;
  }

  const double scale = options_.image_scale;
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
      options_.fx * scale, 0.0, options_.cx * scale,
      0.0, options_.fy * scale, options_.cy * scale,
      0.0, 0.0, 1.0);
  const cv::Mat distortion = (cv::Mat_<double>(1, 5) <<
      options_.distortion[0], options_.distortion[1], options_.distortion[2],
      options_.distortion[3], options_.distortion[4]);

  if (options_.enable_lidar_pnp)
  {
    std::string pnp_failure = "pnp_reference_cloud_missing";
    if (!reference_body_points_.empty() && !body_points.empty())
    {
      try
      {
        const Eigen::Matrix3d camera_from_body_rotation =
            options_.body_from_camera_rotation.transpose();
        std::vector<cv::Point3f> reference_camera_points;
        reference_camera_points.reserve(reference_body_points_.size());
        for (const Eigen::Vector3d &body_point : reference_body_points_)
        {
          const Eigen::Vector3d camera_point = camera_from_body_rotation *
              (body_point - options_.body_from_camera_translation);
          if (!camera_point.allFinite() || camera_point.z() < options_.pnp_minimum_depth ||
              camera_point.z() > options_.pnp_maximum_depth) continue;
          reference_camera_points.emplace_back(static_cast<float>(camera_point.x()),
                                                static_cast<float>(camera_point.y()),
                                                static_cast<float>(camera_point.z()));
        }
        std::vector<cv::Point2f> projected_points;
        if (!reference_camera_points.empty())
        {
          cv::projectPoints(reference_camera_points, cv::Mat::zeros(3, 1, CV_64F),
                            cv::Mat::zeros(3, 1, CV_64F), camera_matrix,
                            distortion, projected_points);
        }
        cv::Mat depth(gray.rows, gray.cols, CV_32F,
                      cv::Scalar(std::numeric_limits<float>::infinity()));
        cv::Mat point_index(gray.rows, gray.cols, CV_32S, cv::Scalar(-1));
        for (std::size_t index = 0; index < projected_points.size(); ++index)
        {
          const int x = static_cast<int>(std::lround(projected_points[index].x));
          const int y = static_cast<int>(std::lround(projected_points[index].y));
          if (x < 0 || y < 0 || x >= gray.cols || y >= gray.rows) continue;
          const float z = reference_camera_points[index].z;
          if (z < depth.at<float>(y, x))
          {
            depth.at<float>(y, x) = z;
            point_index.at<int>(y, x) = static_cast<int>(index);
          }
        }

        struct Association
        {
          int feature = -1;
          int point = -1;
          double squared_pixel_distance = std::numeric_limits<double>::infinity();
          double depth = 0.0;
        };
        std::vector<Association> associations;
        associations.reserve(first.size());
        const int association_radius = options_.pnp_association_radius_pixels;
        for (std::size_t feature = 0; feature < first.size(); ++feature)
        {
          const int center_x = static_cast<int>(std::lround(first[feature].x));
          const int center_y = static_cast<int>(std::lround(first[feature].y));
          Association best;
          best.feature = static_cast<int>(feature);
          std::vector<double> local_depths;
          for (int dy = -association_radius; dy <= association_radius; ++dy)
          {
            const int y = center_y + dy;
            if (y < 0 || y >= gray.rows) continue;
            for (int dx = -association_radius; dx <= association_radius; ++dx)
            {
              const int x = center_x + dx;
              if (x < 0 || x >= gray.cols) continue;
              const int candidate = point_index.at<int>(y, x);
              if (candidate < 0) continue;
              local_depths.push_back(reference_camera_points[
                  static_cast<std::size_t>(candidate)].z);
              const double squared = static_cast<double>(dx * dx + dy * dy);
              if (squared < best.squared_pixel_distance)
              {
                best.point = candidate;
                best.squared_pixel_distance = squared;
                best.depth = reference_camera_points[
                    static_cast<std::size_t>(candidate)].z;
              }
            }
          }
          if (best.point < 0) continue;
          if (local_depths.size() >= 3U)
          {
            const double local_median = median(local_depths);
            const double allowed_difference = std::max(
                options_.pnp_maximum_local_depth_difference,
                options_.pnp_maximum_local_depth_ratio * local_median);
            if (!std::isfinite(local_median) ||
                std::abs(best.depth - local_median) > allowed_difference)
            {
              continue;
            }
          }
          associations.push_back(best);
        }
        std::sort(associations.begin(), associations.end(),
                  [](const Association &first_association,
                     const Association &second_association)
                  {
                    return first_association.squared_pixel_distance <
                           second_association.squared_pixel_distance;
                  });
        std::unordered_set<int> used_points;
        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        std::vector<int> object_feature_indices;
        std::vector<cv::Point2f> normalized_reference_points;
        cv::undistortPoints(first, normalized_reference_points,
                            camera_matrix, distortion);
        object_points.reserve(associations.size());
        image_points.reserve(associations.size());
        object_feature_indices.reserve(associations.size());
        for (const Association &association : associations)
        {
          if (!used_points.insert(association.point).second) continue;
          const std::size_t feature = static_cast<std::size_t>(association.feature);
          if (feature >= normalized_reference_points.size() ||
              !std::isfinite(association.depth)) continue;
          const cv::Point2f &ray = normalized_reference_points[feature];
          object_points.emplace_back(ray.x * static_cast<float>(association.depth),
                                     ray.y * static_cast<float>(association.depth),
                                     static_cast<float>(association.depth));
          image_points.push_back(second[static_cast<std::size_t>(association.feature)]);
          object_feature_indices.push_back(association.feature);
        }
        result.pnp_correspondences = static_cast<int>(object_points.size());
        pnp_failure = "pnp_insufficient_correspondences";
        if (result.pnp_correspondences >= options_.pnp_minimum_correspondences)
        {
          cv::Mat rotation_vector;
          cv::Mat translation_vector;
          cv::Mat pnp_inliers;
          const bool solved = cv::solvePnPRansac(
              object_points, image_points, camera_matrix, distortion,
              rotation_vector, translation_vector, false,
              options_.pnp_ransac_iterations,
              static_cast<float>(options_.pnp_ransac_reprojection_error),
              options_.ransac_probability, pnp_inliers, cv::SOLVEPNP_EPNP);
          if (options_.generate_debug_images)
          {
            drawPnpCorrespondences(gray, image_points, pnp_inliers,
                                   &result.pnp_debug_image);
            cv::Mat track_inliers = cv::Mat::zeros(
                static_cast<int>(first.size()), 1, CV_8U);
            for (int row = 0; row < pnp_inliers.rows; ++row)
            {
              const int pnp_index = pnp_inliers.at<int>(row, 0);
              if (pnp_index < 0 ||
                  static_cast<std::size_t>(pnp_index) >= object_feature_indices.size())
              {
                continue;
              }
              const int feature_index = object_feature_indices[
                  static_cast<std::size_t>(pnp_index)];
              if (feature_index >= 0 && feature_index < track_inliers.rows)
              {
                track_inliers.at<uint8_t>(feature_index, 0) = 1U;
              }
            }
            drawTracks(gray, first, second, track_inliers,
                       &result.tracking_debug_image);
          }
          result.pnp_inliers = solved ? pnp_inliers.rows : 0;
          result.pnp_inlier_ratio = result.pnp_correspondences > 0 ?
              static_cast<double>(result.pnp_inliers) /
                  static_cast<double>(result.pnp_correspondences) : 0.0;
          std::vector<uint8_t> pnp_occupied(static_cast<std::size_t>(
              options_.grid_rows * options_.grid_cols), 0U);
          for (int row = 0; row < pnp_inliers.rows; ++row)
          {
            const int index = pnp_inliers.at<int>(row, 0);
            if (index < 0 || index >= result.pnp_correspondences) continue;
            const cv::Point2f &point = image_points[static_cast<std::size_t>(index)];
            const int grid_row = std::max(0, std::min(options_.grid_rows - 1,
                static_cast<int>(point.y / std::max(1.0F, height) * options_.grid_rows)));
            const int grid_col = std::max(0, std::min(options_.grid_cols - 1,
                static_cast<int>(point.x / std::max(1.0F, width) * options_.grid_cols)));
            pnp_occupied[static_cast<std::size_t>(
                grid_row * options_.grid_cols + grid_col)] = 1U;
          }
          result.pnp_occupied_cells = static_cast<int>(
              std::count(pnp_occupied.begin(), pnp_occupied.end(), 1U));
          pnp_failure = "pnp_ransac_gate";
          if (solved && result.pnp_inliers >= options_.pnp_minimum_inliers &&
              result.pnp_inlier_ratio >= options_.pnp_minimum_inlier_ratio &&
              result.pnp_occupied_cells >= options_.pnp_minimum_occupied_cells)
          {
            std::vector<cv::Point3f> inlier_objects;
            std::vector<cv::Point2f> inlier_images;
            inlier_objects.reserve(static_cast<std::size_t>(result.pnp_inliers));
            inlier_images.reserve(static_cast<std::size_t>(result.pnp_inliers));
            for (int row = 0; row < pnp_inliers.rows; ++row)
            {
              const int index = pnp_inliers.at<int>(row, 0);
              if (index < 0 || index >= result.pnp_correspondences) continue;
              inlier_objects.push_back(object_points[static_cast<std::size_t>(index)]);
              inlier_images.push_back(image_points[static_cast<std::size_t>(index)]);
            }
            const bool refined = cv::solvePnP(
                inlier_objects, inlier_images, camera_matrix, distortion,
                rotation_vector, translation_vector, true, cv::SOLVEPNP_ITERATIVE);
            if (!refined)
            {
              pnp_failure = "pnp_refinement_failed";
            }
            else
            {
              std::vector<cv::Point2f> reprojections;
              cv::projectPoints(inlier_objects, rotation_vector, translation_vector,
                                camera_matrix, distortion, reprojections);
              double squared_reprojection_error = 0.0;
              for (std::size_t index = 0; index < reprojections.size(); ++index)
              {
                const double error = cv::norm(reprojections[index] - inlier_images[index]);
                squared_reprojection_error += error * error;
              }
              result.pnp_reprojection_rmse = reprojections.empty() ?
                  std::numeric_limits<double>::infinity() :
                  std::sqrt(squared_reprojection_error /
                            static_cast<double>(reprojections.size()));
              pnp_failure = "pnp_reprojection_gate";
              if (std::isfinite(result.pnp_reprojection_rmse) &&
                  result.pnp_reprojection_rmse <= options_.pnp_maximum_reprojection_rmse)
              {
                cv::Mat current_camera_rotation_cv;
                cv::Rodrigues(rotation_vector, current_camera_rotation_cv);
                Eigen::Isometry3d current_camera_from_reference_camera =
                    Eigen::Isometry3d::Identity();
                current_camera_from_reference_camera.linear() =
                    eigenRotation(current_camera_rotation_cv);
                current_camera_from_reference_camera.translation() = Eigen::Vector3d(
                    translation_vector.at<double>(0), translation_vector.at<double>(1),
                    translation_vector.at<double>(2));
                Eigen::Isometry3d body_from_camera = Eigen::Isometry3d::Identity();
                body_from_camera.linear() = options_.body_from_camera_rotation;
                body_from_camera.translation() = options_.body_from_camera_translation;
                result.relative_body_pose = body_from_camera *
                    current_camera_from_reference_camera.inverse() *
                    body_from_camera.inverse();
                result.relative_body_rotation =
                    projectToRotation(result.relative_body_pose.rotation());
                result.relative_body_pose.linear() = result.relative_body_rotation;
                const double cosine = clampValue(
                    0.5 * (result.relative_body_rotation.trace() - 1.0), -1.0, 1.0);
                result.rotation_deg = std::acos(cosine) * 180.0 / kPi;
                const double maximum_rotation = std::min(options_.maximum_rotation_step_deg,
                    options_.maximum_rotation_rate_deg * dt + 0.5);
                const double maximum_translation = std::min(
                    options_.pnp_maximum_translation_step,
                    options_.pnp_maximum_translation_speed * dt + 0.25);
                pnp_failure = "pnp_motion_gate";
                if (std::isfinite(result.rotation_deg) &&
                    result.rotation_deg <= maximum_rotation &&
                    result.relative_body_pose.translation().allFinite() &&
                    result.relative_body_pose.translation().norm() <= maximum_translation)
                {
                  const double count_quality = clampValue(
                      static_cast<double>(result.pnp_inliers) /
                          std::max(1, 2 * options_.pnp_minimum_inliers), 0.0, 1.0);
                  const double ratio_quality = clampValue(
                      result.pnp_inlier_ratio / std::max(
                          0.5, 2.0 * options_.pnp_minimum_inlier_ratio), 0.0, 1.0);
                  const double coverage_quality = clampValue(
                      static_cast<double>(result.pnp_occupied_cells) /
                          std::max(1, 2 * options_.pnp_minimum_occupied_cells), 0.0, 1.0);
                  const double normalized_rmse = result.pnp_reprojection_rmse /
                      options_.pnp_maximum_reprojection_rmse;
                  const double reprojection_quality = std::exp(
                      -0.5 * normalized_rmse * normalized_rmse);
                  result.quality = clampValue(std::sqrt(
                      ratio_quality * count_quality * coverage_quality) *
                      reprojection_quality, 0.0, 1.0);
                  visual_from_body_pose_ = visual_from_body_pose_ * result.relative_body_pose;
                  visual_from_body_pose_.linear() =
                      projectToRotation(visual_from_body_pose_.rotation());
                  visual_from_body_rotation_ = visual_from_body_pose_.rotation();
                  reference_gray_ = gray;
                  reference_body_points_ = body_points;
                  reference_stamp_ = stamp;
                  result.observation_valid = true;
                  result.motion_valid = true;
                  result.metric_pose_valid = true;
                  result.segment = segment_;
                  result.visual_from_body_rotation = visual_from_body_rotation_;
                  result.visual_from_body_pose = visual_from_body_pose_;
                  result.inliers = result.pnp_inliers;
                  result.inlier_ratio = result.pnp_inlier_ratio;
                  result.reason = "pnp_accepted";
                  return result;
                }
              }
            }
          }
        }
      }
      catch (const cv::Exception &)
      {
        pnp_failure = "pnp_opencv_exception";
      }
    }
    if (options_.require_lidar_pnp)
    {
      result.reason = pnp_failure;
      return result;
    }
  }

  std::vector<cv::Point2f> first_normalized;
  std::vector<cv::Point2f> second_normalized;
  cv::undistortPoints(first, first_normalized, camera_matrix, distortion);
  cv::undistortPoints(second, second_normalized, camera_matrix, distortion);
  cv::Mat inlier_mask;
  const double normalized_threshold = options_.ransac_threshold_pixels /
      std::max(1.0, 0.5 * (options_.fx + options_.fy) * scale);
  const cv::Mat essential = cv::findEssentialMat(
      first_normalized, second_normalized, 1.0, cv::Point2d(0.0, 0.0),
      cv::RANSAC, options_.ransac_probability, normalized_threshold, inlier_mask);
  if (essential.empty())
  {
    result.reason = "essential_matrix_failed";
    return result;
  }
  cv::Mat camera_rotation;
  cv::Mat camera_translation;
  const int inliers = cv::recoverPose(essential, first_normalized, second_normalized,
                                      camera_rotation, camera_translation, 1.0,
                                      cv::Point2d(0.0, 0.0), inlier_mask);
  if (options_.generate_debug_images)
  {
    drawTracks(gray, first, second, inlier_mask, &result.tracking_debug_image);
  }
  result.inliers = inliers;
  result.inlier_ratio = result.tracks > 0 ?
      static_cast<double>(result.inliers) / static_cast<double>(result.tracks) : 0.0;
  if (result.inliers < options_.minimum_inliers ||
      result.inlier_ratio < options_.minimum_inlier_ratio)
  {
    result.reason = "visual_ransac_gate";
    return result;
  }

  const Eigen::Matrix3d current_camera_from_reference_camera =
      eigenRotation(camera_rotation);
  const Eigen::Matrix3d reference_body_from_current_body =
      options_.body_from_camera_rotation *
      current_camera_from_reference_camera.transpose() *
      options_.body_from_camera_rotation.transpose();
  result.relative_body_rotation = projectToRotation(reference_body_from_current_body);
  const double cosine = clampValue(0.5 * (result.relative_body_rotation.trace() - 1.0),
                                   -1.0, 1.0);
  result.rotation_deg = std::acos(cosine) * 180.0 / kPi;
  const double maximum_rotation = std::min(options_.maximum_rotation_step_deg,
      options_.maximum_rotation_rate_deg * dt + 0.5);
  if (!std::isfinite(result.rotation_deg) || result.rotation_deg > maximum_rotation)
  {
    result.reason = "visual_rotation_rate_gate";
    return result;
  }

  const double inlier_quality = clampValue(
      static_cast<double>(result.inliers) / std::max(1, 2 * options_.minimum_inliers),
      0.0, 1.0);
  const double coverage_quality = clampValue(
      static_cast<double>(result.occupied_cells) /
          std::max(1, 2 * options_.minimum_occupied_cells), 0.0, 1.0);
  result.quality = clampValue(result.inlier_ratio * std::sqrt(inlier_quality * coverage_quality),
                              0.0, 1.0);
  result.relative_body_pose.setIdentity();
  result.relative_body_pose.linear() = result.relative_body_rotation;
  visual_from_body_pose_ = visual_from_body_pose_ * result.relative_body_pose;
  visual_from_body_pose_.linear() = projectToRotation(visual_from_body_pose_.rotation());
  visual_from_body_rotation_ = projectToRotation(
      visual_from_body_rotation_ * result.relative_body_rotation);
  reference_gray_ = gray;
  reference_body_points_ = body_points;
  reference_stamp_ = stamp;
  result.observation_valid = true;
  result.motion_valid = true;
  result.segment = segment_;
  result.visual_from_body_rotation = visual_from_body_rotation_;
  result.visual_from_body_pose = visual_from_body_pose_;
  result.reason = "accepted";
  return result;
}

}  // namespace hybrid_localization
