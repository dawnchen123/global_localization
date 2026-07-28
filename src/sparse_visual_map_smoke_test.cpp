#include "hybrid_localization/sparse_visual_map.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
  using namespace hybrid_localization;

  SparseVisualMapOptions options;
  options.enabled = true;
  options.fx = 200.0;
  options.fy = 200.0;
  options.cx = 160.0;
  options.cy = 120.0;
  options.distortion = {{0.0, 0.0, 0.0, 0.0, 0.0}};
  options.image_scale = 1.0;
  options.patch_half_size = 2;
  options.grid_size_pixels = 12;
  options.max_landmarks = 400;
  options.max_active_landmarks = 400;
  options.max_new_landmarks_per_frame = 300;
  options.minimum_gradient = 0.0;
  options.minimum_patch_stddev = 0.1;
  options.minimum_ncc = 0.95;
  options.local_map_radius = 100.0;
  SparseVisualMap visual_map(options);

  // Pixel sampling must remain in the exact same camera model used by the
  // projection/Jacobian. A pre-rectified stream and an intentionally raw
  // radial-tangential stream must both retain their own pixel coordinates.
  cv::Mat calibration_probe(240, 320, CV_8U, cv::Scalar(0));
  calibration_probe.at<uint8_t>(24, 24) = 255U;
  SparseVisualMapOptions probe_options = options;
  probe_options.distortion = {{-0.15, 0.09, 0.001, -0.001, -0.02}};
  probe_options.rectify_input = false;
  probe_options.apply_distortion = false;
  SparseVisualMap rectified_probe_map(probe_options);
  const SparseVisualFrame rectified_probe = rectified_probe_map.prepareFrame(
      0.5, calibration_probe);
  assert(rectified_probe.valid());
  assert(rectified_probe.gray.at<float>(24, 24) == 255.0F);
  probe_options.apply_distortion = true;
  SparseVisualMap raw_probe_map(probe_options);
  const SparseVisualFrame raw_probe = raw_probe_map.prepareFrame(0.5,
                                                                   calibration_probe);
  assert(raw_probe.valid());
  assert(raw_probe.gray.at<float>(24, 24) == 255.0F);

  cv::Mat image(240, 320, CV_8U);
  for (int row = 0; row < image.rows; ++row)
  {
    for (int col = 0; col < image.cols; ++col)
    {
      image.at<uint8_t>(row, col) = static_cast<uint8_t>(
          (37 * col + 17 * row + (col * row) % 251) % 256);
    }
  }

  PointVector points;
  constexpr double depth = 5.0;
  for (int row = 18; row < image.rows - 18; row += 12)
  {
    for (int col = 18; col < image.cols - 18; col += 12)
    {
      points.emplace_back((static_cast<double>(col) - options.cx) * depth /
                              options.fx,
                          (static_cast<double>(row) - options.cy) * depth /
                              options.fy,
                          depth);
    }
  }

  const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  visual_map.addLidarFrame(1.0, pose, points);
  const SparseVisualFrame reference = visual_map.prepareFrame(1.0, image);
  assert(reference.valid());
  visual_map.commitFrame(reference, pose, false);
  assert(visual_map.landmarkCount() > 100U);

  const SparseVisualFrame current = visual_map.prepareFrame(1.1, image);
  const VisualPoseLinearization linearization = visual_map.linearize(current, pose);
  assert(linearization.valid);
  assert(linearization.landmarks > 100);
  assert(linearization.residuals > 1000);
  assert(linearization.rmse < 1e-5);
  assert(linearization.mean_ncc > 0.999);
  assert(!visual_map.debugImage().empty());

  const cv::Mat dynamic_mask(image.size(), CV_8U, cv::Scalar(255));
  const SparseVisualFrame masked = visual_map.prepareFrame(1.2, image,
                                                           dynamic_mask);
  const VisualPoseLinearization masked_result = visual_map.linearize(masked, pose);
  assert(!masked_result.valid);
  assert(masked_result.landmarks == 0);
  assert(visual_map.stats().dynamic_rejections > 0U);

  const cv::Mat dynamic_labels(image.size(), CV_8U, cv::Scalar(5));
  const SparseVisualFrame semantic_dynamic = visual_map.prepareFrame(
      1.25, image, cv::Mat(), dynamic_labels);
  const VisualPoseLinearization semantic_dynamic_result =
      visual_map.linearize(semantic_dynamic, pose);
  assert(!semantic_dynamic_result.valid);
  assert(semantic_dynamic_result.landmarks == 0);

  SparseVisualMap weighted_map(options);
  const std::vector<uint8_t> labels(points.size(), 5U);
  const std::vector<float> zero_weights(points.size(), 0.0F);
  weighted_map.addLidarFrame(1.0, pose, points, labels, zero_weights);
  weighted_map.commitFrame(reference, pose, false);
  assert(weighted_map.landmarkCount() == 0U);

  // Render texture only at the radial-tangential projections.  A pinhole-only
  // projection cannot seed these landmarks reliably away from the principal
  // point, while the calibrated model must reproduce the same reference view.
  SparseVisualMapOptions distorted_options = options;
  distorted_options.distortion = {{-0.15, 0.09, 0.001, -0.001, -0.02}};
  distorted_options.rectify_input = false;
  distorted_options.apply_distortion = true;
  SparseVisualMap distorted_map(distorted_options);
  cv::Mat distorted_image(image.size(), CV_8U, cv::Scalar(0));
  PointVector distorted_points;
  const auto distort = [&distorted_options](double x, double y)
  {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + distorted_options.distortion[0] * r2 +
        distorted_options.distortion[1] * r2 * r2 +
        distorted_options.distortion[4] * r2 * r2 * r2;
    return cv::Point2d(
        x * radial + 2.0 * distorted_options.distortion[2] * x * y +
            distorted_options.distortion[3] * (r2 + 2.0 * x * x),
        y * radial + distorted_options.distortion[2] * (r2 + 2.0 * y * y) +
            2.0 * distorted_options.distortion[3] * x * y);
  };
  for (int row = 30; row < image.rows - 30; row += 30)
  {
    for (int col = 30; col < image.cols - 30; col += 30)
    {
      const double x = (static_cast<double>(col) - options.cx) / options.fx;
      const double y = (static_cast<double>(row) - options.cy) / options.fy;
      const cv::Point2d projected = distort(x, y);
      const int u = static_cast<int>(std::lround(
          distorted_options.fx * projected.x + distorted_options.cx));
      const int v = static_cast<int>(std::lround(
          distorted_options.fy * projected.y + distorted_options.cy));
      if (u < 8 || v < 8 || u >= image.cols - 8 || v >= image.rows - 8) continue;
      distorted_points.emplace_back(x * depth, y * depth, depth);
      for (int dy = -5; dy <= 5; ++dy)
      {
        for (int dx = -5; dx <= 5; ++dx)
        {
          distorted_image.at<uint8_t>(v + dy, u + dx) = static_cast<uint8_t>(
              30 + (19 * (dx + 5) + 37 * (dy + 5) + 11 * row + col) % 220);
        }
      }
    }
  }
  distorted_map.addLidarFrame(2.0, pose, distorted_points);
  const SparseVisualFrame distorted_reference = distorted_map.prepareFrame(
      2.0, distorted_image);
  distorted_map.commitFrame(distorted_reference, pose, false);
  assert(distorted_map.landmarkCount() > 20U);
  const VisualPoseLinearization distorted_linearization = distorted_map.linearize(
      distorted_map.prepareFrame(2.1, distorted_image), pose);
  assert(distorted_linearization.valid);
  assert(distorted_linearization.landmarks > 20);
  assert(distorted_linearization.rmse < 1e-5);

  std::cout << "sparse_visual_map_smoke_test: PASS landmarks="
            << visual_map.landmarkCount() << "\n";
  return 0;
}
