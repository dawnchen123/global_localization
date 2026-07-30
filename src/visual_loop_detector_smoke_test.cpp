#include "hybrid_localization/visual_loop_detector.h"

#include <Eigen/Geometry>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>

int main()
{
  hybrid_localization::VisualLoopDetectorOptions options;
  options.enabled = true;
  options.fx = 220.0;
  options.fy = 220.0;
  options.cx = 160.0;
  options.cy = 120.0;
  options.distortion.fill(0.0);
  options.image_scale = 1.0;
  options.maximum_features = 1200;
  options.minimum_depth_features = 40;
  options.keyframe_distance = 0.0;
  options.keyframe_interval_sec = 0.1;
  options.minimum_index_gap = 2;
  options.search_radius = 5.0;
  options.maximum_yaw_difference_deg = 20.0;
  options.minimum_descriptor_matches = 30;
  options.depth_association_radius_pixels = 3;
  options.minimum_pnp_inliers = 25;
  options.minimum_pnp_inlier_ratio = 0.5;
  options.minimum_occupied_cells = 5;
  options.maximum_reprojection_rmse = 0.8;
  options.maximum_translation_disagreement = 0.2;
  options.maximum_rotation_disagreement_deg = 2.0;
  options.minimum_quality = 0.35;

  cv::Mat image(240, 320, CV_8UC1, cv::Scalar(20));
  for (int y = 12; y < image.rows - 12; y += 18)
  {
    for (int x = 12; x < image.cols - 12; x += 18)
    {
      const int value = ((x / 18 + y / 18) % 2 == 0) ? 235 : 75;
      cv::rectangle(image, cv::Rect(x - 5, y - 5, 11, 11),
                    cv::Scalar(value), cv::FILLED);
      cv::line(image, cv::Point(x - 6, y + 7), cv::Point(x + 7, y - 6),
               cv::Scalar(255 - value), 1);
    }
  }

  hybrid_localization::VisualLidarPointVector points;
  points.reserve(static_cast<std::size_t>(image.rows * image.cols / 4));
  for (int v = 0; v < image.rows; v += 2)
  {
    for (int u = 0; u < image.cols; u += 2)
    {
      const double depth = 4.0 + 0.35 * std::sin(0.031 * u) +
                           0.25 * std::cos(0.043 * v);
      points.emplace_back((u - options.cx) * depth / options.fx,
                          (v - options.cy) * depth / options.fy, depth);
    }
  }

  hybrid_localization::VisualLoopDetector detector(options);
  const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  const auto first = detector.process(0.0, image, points, pose);
  const auto second = detector.process(1.0, image, points, pose);
  const auto third = detector.process(2.0, image, points, pose);
  if (!first.keyframe_created || !second.keyframe_created ||
      !third.keyframe_created || detector.keyframeCount() != 3U)
  {
    std::cerr << "visual keyframe creation failed: " << first.reason << ", "
              << second.reason << ", " << third.reason << std::endl;
    return 1;
  }
  if (!third.accepted || third.reference_id != 0 || third.current_id != 2)
  {
    std::cerr << "visual loop was not accepted: " << third.reason
              << " matches=" << third.descriptor_matches
              << " inliers=" << third.pnp_inliers
              << " cells=" << third.occupied_cells
              << " rmse=" << third.reprojection_rmse << std::endl;
    return 2;
  }
  if (third.reference_from_current.translation().norm() > 0.03 ||
      Eigen::AngleAxisd(third.reference_from_current.rotation()).angle() > 0.01)
  {
    std::cerr << "visual loop pose is inconsistent with identity" << std::endl;
    return 3;
  }
  std::cout << "visual loop detector smoke test passed: matches="
            << third.descriptor_matches << " inliers=" << third.pnp_inliers
            << " quality=" << third.quality << std::endl;
  return 0;
}
