#include "hybrid_localization/semantic_pose_graph.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace
{

Eigen::Isometry3d pose(double x, double y, double z, double yaw)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  result.translation() = Eigen::Vector3d(x, y, z);
  return result;
}

hybrid_localization::SemanticGraphPointVector makeObservation(
    const Eigen::Isometry3d &world_from_body,
    const hybrid_localization::SemanticGraphPointVector &world)
{
  hybrid_localization::SemanticGraphPointVector local;
  const Eigen::Isometry3d body_from_world = world_from_body.inverse();
  for (const auto &world_point : world)
  {
    hybrid_localization::SemanticGraphPoint point = world_point;
    point.point = body_from_world * world_point.point;
    if (point.point.head<2>().norm() < 50.0) local.push_back(point);
  }
  return local;
}

}  // namespace

int main()
{
  hybrid_localization::SemanticPoseGraphOptions options;
  // Generic geometric loop proposals are intentionally opt-in in production.
  // This test exercises that separate path, so enable it explicitly.
  options.enable_xy_loops = true;
  options.enable_z_loops = true;
  options.enable_sequential_ground_z = true;
  options.keyframe_distance = 0.5;
  options.keyframe_interval_sec = 0.1;
  options.keyframe_yaw_deg = 2.0;
  options.submap_frames = 1;
  options.feature_min_points = 1;
  options.ground_min_points = 1;
  options.max_features_per_keyframe = 5000;
  options.loop_min_index_gap = 3;
  options.loop_search_radius = 8.0;
  options.loop_max_candidates = 3;
  options.descriptor_min_similarity = 0.05;
  options.coarse_xy_radius = 4.0;
  options.coarse_xy_step = 0.5;
  options.coarse_yaw_radius_deg = 8.0;
  options.coarse_yaw_step_deg = 1.0;
  options.coarse_min_inliers = 20;
  options.min_xy_inliers = 25;
  options.min_xy_inlier_ratio = 0.25;
  options.min_xy_spread = 4.0;
  options.min_xy_spread_ratio = 0.05;
  options.max_xy_rmse = 0.30;
  options.ransac_inlier_distance = 0.35;
  options.correspondence_distance = 0.8;
  options.min_z_inliers = 20;
  options.z_correspondence_distance = 0.8;
  options.z_inlier_residual_gate = 0.12;
  options.max_z_mad = 0.08;
  options.min_loops_for_xy_output = 1;
  options.enable_semantic_observation_factors = true;
  options.enable_visual_loop_factors = true;
  options.visual_loop_max_time_offset = 0.1;
  options.visual_loop_min_index_gap = 3;
  options.visual_loop_min_quality = 0.4;
  options.visual_loop_max_translation_disagreement = 3.0;
  options.visual_loop_max_rotation_disagreement_deg = 8.0;
  options.semantic_submap_observations = 1;
  options.semantic_observation_min_index_gap = 1;
  options.semantic_observation_max_index_gap = 1;
  options.semantic_observation_interval = 1;
  options.semantic_observation_min_features = 20;
  options.semantic_observation_min_inliers = 20;
  options.semantic_observation_min_z_inliers = 15;
  options.min_semantic_observation_factors_for_xy_output = 1;
  options.semantic_observation_search_radius = 25.0;
  options.semantic_observation_correspondence_distance = 0.9;
  options.semantic_observation_ransac_inlier_distance = 0.40;
  options.semantic_observation_min_inlier_ratio = 0.20;
  options.semantic_observation_min_spread = 3.0;
  options.semantic_observation_min_spread_ratio = 0.04;
  options.semantic_observation_max_rmse = 0.35;
  options.semantic_observation_max_xy_correction = 1.0;
  options.semantic_observation_max_yaw_correction_deg = 4.0;
  options.semantic_observation_max_z_correction = 0.5;

  hybrid_localization::SemanticGraphPointVector world;
  std::mt19937 generator(42U);
  std::uniform_real_distribution<double> xy(-28.0, 28.0);
  std::uniform_real_distribution<double> height(0.5, 7.0);
  for (int i = 0; i < 420; ++i)
  {
    hybrid_localization::SemanticGraphPoint point;
    point.point = Eigen::Vector3d(xy(generator), xy(generator), height(generator));
    point.label = 3U;
    world.push_back(point);
  }
  for (int x = -28; x <= 28; x += 2)
  {
    for (int y = -28; y <= 28; y += 2)
    {
      hybrid_localization::SemanticGraphPoint point;
      point.point = Eigen::Vector3d(x, y, 0.0);
      point.label = 1U;
      world.push_back(point);
    }
  }

  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> truth{
      pose(0.0, 0.0, 1.0, 0.0), pose(10.0, 0.0, 1.0, 0.0),
      pose(20.0, 0.0, 1.0, 0.0), pose(10.0, 0.0, 1.0, 0.0),
      pose(0.0, 0.0, 1.0, 0.0)};
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> raw{
      pose(0.0, 0.0, 1.0, 0.0), pose(10.4, 0.2, 1.15, 0.01),
      pose(20.9, 0.4, 1.35, 0.02), pose(11.4, 0.7, 1.55, 0.035),
      pose(2.0, 1.0, 1.80, 0.052)};

  hybrid_localization::SemanticPoseGraph graph(options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const double stamp = static_cast<double>(i);
    graph.addOdometrySample(stamp, raw[i]);
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!graph.addFrame(stamp, raw[i], geometry))
    {
      std::cerr << "failed to add synthetic keyframe " << i << '\n';
      return 1;
    }
    if (!graph.addSemanticObservation(stamp, raw[i], semantic))
    {
      std::cerr << "failed to associate semantic observation " << i << '\n';
      return 7;
    }
  }

  if (!graph.addVisualLoopConstraint(
          0.0, 4.0, truth.front().inverse() * truth.back(), 0.90))
  {
    std::cerr << "failed to add synthetic non-adjacent visual loop\n";
    return 10;
  }

  const hybrid_localization::SemanticPoseGraphStats stats = graph.stats();
  if (stats.xy_loop_factors < 1 || stats.z_loop_factors < 1)
  {
    std::cerr << "expected XY and Z loop factors, got xy=" << stats.xy_loop_factors
              << " z=" << stats.z_loop_factors
              << " reason=" << graph.lastDebug().reason << '\n';
    return 2;
  }
  if (stats.visual_loop_factors != 1 || stats.visual_loop_attempts != 1 ||
      stats.visual_loop_rejections != 0)
  {
    std::cerr << "visual loop factor regression: attempts="
              << stats.visual_loop_attempts << " rejected="
              << stats.visual_loop_rejections << " factors="
              << stats.visual_loop_factors << '\n';
    return 11;
  }
  if (stats.semantic_observation_factors < 1 ||
      stats.semantic_observation_xy_factors < 1)
  {
    std::cerr << "expected local semantic observation factors, got total="
              << stats.semantic_observation_factors << " xy="
              << stats.semantic_observation_xy_factors << " z="
              << stats.semantic_observation_z_factors << " reason="
              << graph.lastSemanticDebug().reason << '\n';
    return 5;
  }
  const auto semantic_map = graph.semanticMap(0.30, 20000);
  if (semantic_map.empty())
  {
    std::cerr << "expected optimized semantic map points\n";
    return 6;
  }
  const Eigen::Isometry3d corrected = graph.correctedPose(raw.back());
  const double raw_error = (raw.back().translation() - truth.back().translation()).norm();
  const double corrected_error =
      (corrected.translation() - truth.back().translation()).norm();
  if (!(corrected_error < raw_error))
  {
    std::cerr << "graph did not reduce endpoint error: raw=" << raw_error
              << " corrected=" << corrected_error << '\n';
    return 3;
  }
  const auto trajectory = graph.optimizedTrajectory();
  if (trajectory.size() != raw.size())
  {
    std::cerr << "dense trajectory sample count mismatch\n";
    return 4;
  }

  hybrid_localization::SemanticPoseGraphOptions batched_options = options;
  // Semantic factors must not depend on the separately configured generic
  // geometric loop path.
  batched_options.enable_xy_loops = false;
  batched_options.enable_z_loops = false;
  batched_options.semantic_submap_observations = 2;
  batched_options.semantic_observation_min_index_gap = 2;
  batched_options.semantic_observation_max_index_gap = 2;
  batched_options.loop_min_index_gap = 100;
  hybrid_localization::SemanticPoseGraph batched_graph(batched_options);
  for (int i = 0; i < 4; ++i)
  {
    const Eigen::Isometry3d batch_truth = pose(2.0 * i, 0.0, 1.0, 0.0);
    const Eigen::Isometry3d batch_raw = pose(2.0 * i + 0.12 * i,
                                             0.04 * i, 1.0 + 0.03 * i,
                                             0.003 * i);
    const double stamp = 10.0 + static_cast<double>(i);
    batched_graph.addOdometrySample(stamp, batch_raw);
    const auto semantic = makeObservation(batch_truth, world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!batched_graph.addFrame(stamp, batch_raw, geometry) ||
        !batched_graph.addSemanticObservation(stamp, batch_raw, semantic))
    {
      std::cerr << "failed to build batched semantic observation " << i << '\n';
      return 8;
    }
  }
  const auto batched_stats = batched_graph.stats();
  if (batched_stats.semantic_observations_received != 4 ||
      batched_stats.semantic_observations_associated != 4 ||
      batched_stats.semantic_keyframes != 2 ||
      batched_stats.semantic_observation_factors < 1 ||
      batched_stats.semantic_observation_xy_factors < 1)
  {
    std::cerr << "batched semantic submap regression: received="
              << batched_stats.semantic_observations_received << " associated="
              << batched_stats.semantic_observations_associated << " submaps="
              << batched_stats.semantic_keyframes << " factors="
              << batched_stats.semantic_observation_factors << " xy_factors="
              << batched_stats.semantic_observation_xy_factors << " reason="
              << batched_graph.lastSemanticDebug().reason << '\n';
    return 9;
  }
  std::cout << "semantic pose graph smoke test passed: loop_xy="
            << stats.xy_loop_factors << " loop_z=" << stats.z_loop_factors
            << " semantic_factors=" << stats.semantic_observation_factors
            << " visual_loop_factors=" << stats.visual_loop_factors
            << " semantic_map_points=" << semantic_map.size()
            << " raw_error=" << raw_error
            << " corrected_error=" << corrected_error << '\n';
  return 0;
}
