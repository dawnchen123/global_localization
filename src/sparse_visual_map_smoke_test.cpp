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

  std::cout << "sparse_visual_map_smoke_test: PASS landmarks="
            << visual_map.landmarkCount() << "\n";
  return 0;
}
