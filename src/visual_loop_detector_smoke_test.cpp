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
  options.debug_image_history_size = 1;
  options.keyframe_distance = 0.0;
  options.keyframe_interval_sec = 0.1;
  options.retrieval_interval_sec = 0.0;
  options.minimum_index_gap = 2;
  options.minimum_time_separation_sec = 0.0;
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
  if (third.debug_image.empty())
  {
    std::cerr << "archived visual-loop debug rendering failed" << std::endl;
    return 6;
  }
  if (third.reference_from_current.translation().norm() > 0.03 ||
      Eigen::AngleAxisd(third.reference_from_current.rotation()).angle() > 0.01)
  {
    std::cerr << "visual loop pose is inconsistent with identity" << std::endl;
    return 3;
  }
  hybrid_localization::VisualLoopDetectorOptions bounded_options = options;
  bounded_options.keyframe_distance = 0.2;
  bounded_options.keyframe_interval_sec = 1.0;
  hybrid_localization::VisualLoopDetector bounded_detector(bounded_options);
  bounded_detector.process(0.0, image, points, pose);
  Eigen::Isometry3d moved_pose = pose;
  moved_pose.translation().x() = 1.0;
  const auto dense_image = bounded_detector.process(0.2, image, points, moved_pose);
  if (dense_image.keyframe_created || bounded_detector.keyframeCount() != 1U)
  {
    std::cerr << "visual loop temporal database bound regression" << std::endl;
    return 7;
  }
  const auto bounded_second = bounded_detector.process(1.1, image, points, moved_pose);
  if (!bounded_second.keyframe_created || bounded_detector.keyframeCount() != 2U)
  {
    std::cerr << "visual loop minimum interval acceptance regression" << std::endl;
    return 8;
  }
  hybrid_localization::VisualLoopDetectorOptions throttled_options = options;
  throttled_options.retrieval_interval_sec = 3.0;
  hybrid_localization::VisualLoopDetector throttled_detector(throttled_options);
  const auto throttled_first = throttled_detector.process(0.0, image, points, pose);
  const auto throttled_second = throttled_detector.process(1.0, image, points, pose);
  const auto throttled_third = throttled_detector.process(2.0, image, points, pose);
  const auto throttled_fourth = throttled_detector.process(3.0, image, points, pose);
  if (!throttled_first.keyframe_created || !throttled_second.keyframe_created ||
      !throttled_third.keyframe_created ||
      throttled_second.reason != "visual_loop_retrieval_interval" ||
      throttled_third.reason != "visual_loop_retrieval_interval" ||
      throttled_detector.keyframeCount() != 4U || !throttled_fourth.accepted)
  {
    std::cerr << "visual loop retrieval throttling regression: second="
              << throttled_second.reason << " third=" << throttled_third.reason
              << " fourth=" << throttled_fourth.reason << std::endl;
    return 9;
  }
  for (int iteration = 0; iteration < 25; ++iteration)
  {
    hybrid_localization::VisualLoopDetector stress_detector(options);
    stress_detector.process(0.0, image, points, pose);
    stress_detector.process(1.0, image, points, pose);
    const auto stress_result = stress_detector.process(2.0, image, points, pose);
    if (!stress_result.accepted || stress_result.reference_id != 0 ||
        stress_result.current_id != 2)
    {
      std::cerr << "visual loop stress iteration failed: " << iteration
                << " reason=" << stress_result.reason << std::endl;
      return 4;
    }
  }
  hybrid_localization::VisualLoopDetectorOptions temporal_options = options;
  temporal_options.minimum_time_separation_sec = 3.0;
  hybrid_localization::VisualLoopDetector temporal_detector(temporal_options);
  temporal_detector.process(0.0, image, points, pose);
  temporal_detector.process(1.0, image, points, pose);
  const auto short_gap = temporal_detector.process(2.0, image, points, pose);
  const auto long_gap = temporal_detector.process(3.0, image, points, pose);
  if (short_gap.accepted || !long_gap.accepted || long_gap.reference_id != 0 ||
      long_gap.temporal_separation_sec < 3.0)
  {
    std::cerr << "visual loop temporal-separation gate regression: short="
              << short_gap.reason << " long=" << long_gap.reason
              << " separation=" << long_gap.temporal_separation_sec << std::endl;
    return 5;
  }
  hybrid_localization::VisualLoopDetectorOptions multi_candidate_options = options;
  multi_candidate_options.maximum_verified_candidates = 3;
  multi_candidate_options.candidate_reference_min_separation_sec = 0.0;
  hybrid_localization::VisualLoopDetector multi_candidate_detector(multi_candidate_options);
  multi_candidate_detector.process(0.0, image, points, pose);
  multi_candidate_detector.process(1.0, image, points, pose);
  multi_candidate_detector.process(2.0, image, points, pose);
  const auto multi_candidate_result =
      multi_candidate_detector.process(3.0, image, points, pose);
  const auto &multi_candidates = multi_candidate_detector.lastAcceptedCandidates();
  if (!multi_candidate_result.accepted || multi_candidates.size() < 2U ||
      multi_candidates[0].reference_id == multi_candidates[1].reference_id)
  {
    std::cerr << "visual loop verified-candidate fanout regression: count="
              << multi_candidates.size() << std::endl;
    return 10;
  }
  hybrid_localization::VisualLoopDetectorOptions diverse_candidate_options =
      multi_candidate_options;
  diverse_candidate_options.candidate_reference_min_separation_sec = 2.0;
  hybrid_localization::VisualLoopDetector diverse_candidate_detector(
      diverse_candidate_options);
  diverse_candidate_detector.process(0.0, image, points, pose);
  diverse_candidate_detector.process(1.0, image, points, pose);
  diverse_candidate_detector.process(2.0, image, points, pose);
  diverse_candidate_detector.process(3.0, image, points, pose);
  if (diverse_candidate_detector.lastAcceptedCandidates().size() != 1U)
  {
    std::cerr << "visual loop reference-diversity regression: count="
              << diverse_candidate_detector.lastAcceptedCandidates().size() << std::endl;
    return 11;
  }
  hybrid_localization::VisualLoopDetectorOptions global_options = options;
  global_options.search_radius = 1.0;
  global_options.enable_global_retrieval_fallback = true;
  global_options.maximum_global_retrieval_candidates = 2;
  global_options.minimum_global_geometric_candidates = 1;
  global_options.global_retrieval_feature_count = 300;
  global_options.global_retrieval_min_votes = 1;
  global_options.global_retrieval_min_table_count = 1;
  // This synthetic revisit deliberately has an incorrect raw translation.
  // It verifies that the global path supplies the candidate; production still
  // uses the much tighter PnP and LiDAR graph consistency gates.
  global_options.maximum_translation_disagreement = 20.0;
  hybrid_localization::VisualLoopDetector global_detector(global_options);
  Eigen::Isometry3d global_middle_pose = pose;
  global_middle_pose.translation().x() = 2.0;
  Eigen::Isometry3d global_return_pose = pose;
  global_return_pose.translation().x() = 10.0;
  global_detector.process(0.0, image, points, pose);
  global_detector.process(1.0, image, points, global_middle_pose);
  const auto global_return = global_detector.process(
      2.0, image, points, global_return_pose);
  if (!global_return.accepted || !global_return.global_retrieval ||
      global_return.global_retrieval_candidates < 1 ||
      global_return.global_retrieval_descriptor_matches < 1 ||
      global_return.global_retrieval_votes < 1 ||
      global_return.global_retrieval_tables < 1 ||
      global_return.reference_id != 0)
  {
    std::cerr << "visual loop global-retrieval fallback regression: reason="
              << global_return.reason << " source="
              << global_return.global_retrieval << " candidates="
              << global_return.global_retrieval_candidates << " matches="
              << global_return.global_retrieval_descriptor_matches
              << " votes=" << global_return.global_retrieval_votes
              << " tables=" << global_return.global_retrieval_tables
              << " reference=" << global_return.reference_id << std::endl;
    return 12;
  }
  std::cout << "visual loop detector smoke test passed: matches="
            << third.descriptor_matches << " inliers=" << third.pnp_inliers
            << " quality=" << third.quality << std::endl;
  return 0;
}
