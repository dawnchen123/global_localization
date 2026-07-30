#include "hybrid_localization/visual_rotation_tracker.h"

#include <Eigen/Core>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>

namespace
{

cv::Point project(const Eigen::Vector3d &point, double fx, double fy,
                  double cx, double cy)
{
  return cv::Point(static_cast<int>(std::lround(fx * point.x() / point.z() + cx)),
                   static_cast<int>(std::lround(fy * point.y() / point.z() + cy)));
}

void drawFeature(cv::Mat *image, const cv::Point &center, int index)
{
  const int intensity = 170 + (index % 4) * 20;
  cv::line(*image, center + cv::Point(-3, 0), center + cv::Point(3, 0),
           cv::Scalar(intensity), 1, cv::LINE_AA);
  cv::line(*image, center + cv::Point(0, -3), center + cv::Point(0, 3),
           cv::Scalar(255), 1, cv::LINE_AA);
  cv::circle(*image, center, 1, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
}

}  // namespace

int main()
{
  hybrid_localization::VisualRotationTrackerOptions options;
  options.fx = 400.0;
  options.fy = 400.0;
  options.cx = 320.0;
  options.cy = 240.0;
  options.distortion.fill(0.0);
  options.image_scale = 1.0;
  options.minimum_interval = 0.05;
  options.maximum_reference_gap = 1.0;
  options.maximum_features = 500;
  options.minimum_tracks = 16;
  options.minimum_inliers = 10;
  options.feature_quality = 0.001;
  options.feature_minimum_distance = 8.0;
  options.forward_backward_error = 2.0;
  options.minimum_median_parallax = 0.25;
  options.minimum_occupied_cells = 4;
  options.enable_lidar_pnp = true;
  options.require_lidar_pnp = true;
  options.pnp_association_radius_pixels = 7;
  options.pnp_minimum_correspondences = 12;
  options.pnp_minimum_inliers = 8;
  options.pnp_minimum_occupied_cells = 4;
  options.pnp_minimum_inlier_ratio = 0.35;
  options.pnp_ransac_reprojection_error = 4.0;
  options.pnp_maximum_reprojection_rmse = 3.0;

  cv::Mat reference = cv::Mat::zeros(480, 640, CV_8U);
  cv::Mat current = cv::Mat::zeros(480, 640, CV_8U);
  hybrid_localization::VisualLidarPointVector points;
  const Eigen::Vector3d camera_translation(-0.10, 0.01, 0.0);
  int feature_index = 0;
  for (int row = 0; row < 6; ++row)
  {
    for (int col = 0; col < 8; ++col)
    {
      const double u = 100.0 + 60.0 * col;
      const double v = 75.0 + 62.0 * row;
      const double depth = 6.0 + 0.45 * ((row + 2 * col) % 6);
      const Eigen::Vector3d point((u - options.cx) * depth / options.fx,
                                  (v - options.cy) * depth / options.fy,
                                  depth);
      points.push_back(point);
      drawFeature(&reference, project(point, options.fx, options.fy,
                                      options.cx, options.cy), feature_index);
      drawFeature(&current, project(point + camera_translation, options.fx, options.fy,
                                    options.cx, options.cy), feature_index);
      ++feature_index;
    }
  }

  hybrid_localization::VisualRotationTracker tracker(options);
  const auto initialized = tracker.process(1.0, reference, points);
  if (!initialized.observation_valid || !initialized.metric_pose_valid ||
      initialized.reason != "initialized")
  {
    std::cerr << "visual tracker did not initialize with LiDAR depth\n";
    return 1;
  }

  const auto estimate = tracker.process(1.2, current, points);
  if (!estimate.motion_valid || !estimate.metric_pose_valid)
  {
    std::cerr << "LiDAR PnP rejected synthetic motion: " << estimate.reason
              << " correspondences=" << estimate.pnp_correspondences
              << " inliers=" << estimate.pnp_inliers << '\n';
    return 2;
  }
  if (!estimate.relative_body_pose.matrix().allFinite() ||
      estimate.pnp_inliers < options.pnp_minimum_inliers)
  {
    std::cerr << "LiDAR PnP returned an invalid metric pose\n";
    return 3;
  }
  if (estimate.projection_debug_image.empty() ||
      estimate.tracking_debug_image.empty() || estimate.pnp_debug_image.empty())
  {
    std::cerr << "visual tracker did not produce observation debug images\n";
    return 4;
  }
  std::cout << "visual tracker smoke test passed: correspondences="
            << estimate.pnp_correspondences << " inliers=" << estimate.pnp_inliers
            << " rmse=" << estimate.pnp_reprojection_rmse << '\n';
  return 0;
}
