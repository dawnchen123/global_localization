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
  const hybrid_localization::SemanticPoseGraphOptions default_options;
  if (default_options.enable_semantic_observation_factors ||
      default_options.enable_semantic_observation_xy_factors ||
      default_options.enable_semantic_observation_z_factors)
  {
    std::cerr << "semantic feedback must be opt-in by default\n";
    return 20;
  }
  if (!default_options.visual_loop_require_lidar_geometry)
  {
    std::cerr << "visual loop geometry verification must be enabled by default\n";
    return 21;
  }

  hybrid_localization::SemanticPoseGraphOptions options;
  // Generic geometric loop proposals are intentionally opt-in in production.
  // This test exercises that separate path, so enable it explicitly.
  options.enable_xy_loops = true;
  options.enable_z_loops = true;
  options.enable_sequential_ground_z = true;
  // The visual-only and LiDAR-validated loop paths are exercised separately
  // below. Keep this synthetic ground-Z regression focused on its own gate.
  options.visual_loop_require_lidar_geometry = false;
  options.keyframe_distance = 0.5;
  options.keyframe_interval_sec = 0.1;
  options.keyframe_yaw_deg = 2.0;
  options.submap_frames = 1;
  options.feature_min_points = 1;
  options.ground_min_points = 1;
  options.max_features_per_keyframe = 5000;
  options.loop_min_index_gap = 3;
  options.loop_min_time_separation_sec = 0.0;
  options.loop_min_support = 1;
  options.loop_minimum_interval_sec = 0.0;
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
  options.enable_semantic_observation_xy_factors = true;
  options.enable_visual_loop_factors = true;
  options.visual_loop_max_time_offset = 0.1;
  options.visual_loop_min_index_gap = 3;
  options.visual_loop_min_time_separation_sec = 0.0;
  options.visual_loop_min_quality = 0.4;
  options.visual_loop_max_translation_disagreement = 3.0;
  options.visual_loop_max_rotation_disagreement_deg = 8.0;
  options.visual_loop_refine_z_with_ground = true;
  options.visual_loop_ground_z_candidate_residual_gate = 1.0;
  options.visual_loop_ground_z_inlier_residual_gate = 0.12;
  options.visual_loop_ground_z_min_inliers = 20;
  options.visual_loop_ground_z_max_mad = 0.08;
  options.visual_loop_ground_z_max_correction = 1.0;
  options.visual_loop_ground_z_sigma = 0.10;
  options.semantic_submap_observations = 1;
  options.semantic_observation_min_index_gap = 1;
  options.semantic_observation_max_index_gap = 1;
  options.semantic_observation_min_time_separation_sec = 0.0;
  options.semantic_observation_interval = 1;
  options.semantic_observation_minimum_interval_sec = 0.0;
  options.semantic_observation_max_factors = 0;
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
    if (!graph.hasKeyframeNear(stamp) ||
        std::abs(graph.latestKeyframeStamp() - stamp) > 1e-9)
    {
      std::cerr << "keyframe readiness regression at " << stamp << '\n';
      return 25;
    }
    if (!graph.addSemanticObservation(stamp, raw[i], semantic))
    {
      std::cerr << "failed to associate semantic observation " << i << '\n';
      return 7;
    }
  }

  const Eigen::Isometry3d biased_visual_loop = pose(0.0, 0.0, 0.60, 0.0);
  if (!graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.90))
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
  if (stats.visual_loop_ground_z_refinements != 1 ||
      std::abs(stats.last_visual_loop_ground_z_correction + 0.60) > 0.05 ||
      std::abs(stats.last_visual_loop_ground_z_applied_correction + 0.60) > 0.05 ||
      stats.last_visual_loop_ground_z_inliers < 20 ||
      !stats.last_visual_loop_ground_z_accepted ||
      !stats.last_visual_loop_z_constrained ||
      stats.visual_loop_z_without_ground_suppressed != 0)
  {
    std::cerr << "visual loop ground-Z refinement regression: refinements="
              << stats.visual_loop_ground_z_refinements << " correction="
              << stats.last_visual_loop_ground_z_correction << " inliers="
              << stats.last_visual_loop_ground_z_inliers << '\n';
    return 12;
  }

  // A LiDAR-verified visual loop is not inserted on its first proposal when
  // multi-frame support is enabled.  The adjacent proposal must agree on the
  // implied map correction, then the DCS robust kernel receives one factor.
  hybrid_localization::SemanticPoseGraphOptions support_options = options;
  support_options.enable_xy_loops = false;
  support_options.enable_z_loops = false;
  support_options.visual_loop_require_lidar_geometry = true;
  support_options.visual_loop_use_dcs = true;
  support_options.visual_loop_dcs_k = 1.0;
  support_options.visual_loop_min_support = 2;
  support_options.visual_loop_support_reference_neighborhood = 0;
  support_options.visual_loop_support_current_max_gap = 1;
  // The synthetic pass accumulates deliberately exaggerated raw drift between
  // the two observations, so retain finite but permissive consistency gates.
  support_options.visual_loop_support_max_correction_xy = 2.0;
  support_options.visual_loop_support_max_correction_yaw_deg = 5.0;
  support_options.visual_loop_support_max_correction_z = 0.80;
  hybrid_localization::SemanticPoseGraph support_graph(support_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!support_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add visual-loop support regression keyframe " << i << '\n';
      return 26;
    }
  }
  if (support_graph.addVisualLoopConstraint(0.0, 3.0, pose(10.0, 0.0, 0.0, 0.0), 0.90))
  {
    std::cerr << "visual loop was inserted before multiframe confirmation\n";
    return 27;
  }
  if (!support_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.90))
  {
    const hybrid_localization::SemanticPoseGraphStats failed_support_stats =
        support_graph.stats();
    std::cerr << "failed to insert multiframe-confirmed visual loop: reason="
              << failed_support_stats.last_visual_loop_reason << " support="
              << failed_support_stats.last_visual_loop_support << '\n';
    return 28;
  }
  const hybrid_localization::SemanticPoseGraphStats support_stats = support_graph.stats();
  if (support_stats.visual_loop_factors != 1 ||
      support_stats.visual_loop_support_waits < 1 ||
      support_stats.visual_loop_support_confirmations != 1 ||
      support_stats.visual_loop_rejections != 0)
  {
    std::cerr << "visual loop support/DCS regression: waits="
              << support_stats.visual_loop_support_waits << " confirmations="
              << support_stats.visual_loop_support_confirmations << " factors="
              << support_stats.visual_loop_factors << " rejections="
              << support_stats.visual_loop_rejections << '\n';
    return 29;
  }

  // A coherent large height residual must pass the hard ground gate, but a
  // configured trust-region step limits what one loop factor can inject.
  hybrid_localization::SemanticPoseGraphOptions clipped_z_options = options;
  clipped_z_options.enable_xy_loops = false;
  clipped_z_options.enable_z_loops = false;
  clipped_z_options.visual_loop_require_lidar_geometry = false;
  clipped_z_options.visual_loop_ground_z_max_step = 0.25;
  hybrid_localization::SemanticPoseGraph clipped_z_graph(clipped_z_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!clipped_z_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add clipped-Z regression keyframe " << i << '\n';
      return 22;
    }
  }
  if (!clipped_z_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.90))
  {
    std::cerr << "failed to add clipped-Z regression loop\n";
    return 23;
  }
  const hybrid_localization::SemanticPoseGraphStats clipped_z_stats = clipped_z_graph.stats();
  if (!clipped_z_stats.last_visual_loop_ground_z_accepted ||
      clipped_z_stats.visual_loop_ground_z_clipped_refinements != 1 ||
      std::abs(clipped_z_stats.last_visual_loop_ground_z_correction + 0.60) > 0.05 ||
      std::abs(clipped_z_stats.last_visual_loop_ground_z_applied_correction + 0.25) > 0.02)
  {
    std::cerr << "clipped ground-Z regression: raw="
              << clipped_z_stats.last_visual_loop_ground_z_correction
              << " applied=" << clipped_z_stats.last_visual_loop_ground_z_applied_correction
              << " clipped=" << clipped_z_stats.visual_loop_ground_z_clipped_refinements
              << '\n';
    return 24;
  }

  hybrid_localization::SemanticPoseGraphOptions visual_only_options = options;
  visual_only_options.enable_xy_loops = false;
  visual_only_options.enable_z_loops = false;
  visual_only_options.visual_loop_require_lidar_geometry = false;
  visual_only_options.visual_loop_refine_z_with_ground = false;
  visual_only_options.visual_loop_allow_pnp_z_without_ground = false;
  hybrid_localization::SemanticPoseGraph visual_only_graph(visual_only_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!visual_only_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add visual-only regression keyframe " << i << '\n';
      return 13;
    }
  }
  if (!visual_only_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.90))
  {
    std::cerr << "failed to add visual-only regression loop\n";
    return 14;
  }
  if (visual_only_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.90))
  {
    std::cerr << "visual loop factor limit regression\n";
    return 15;
  }
  const hybrid_localization::SemanticPoseGraphStats visual_only_stats =
      visual_only_graph.stats();
  if (visual_only_stats.last_visual_loop_ground_z_accepted ||
      visual_only_stats.last_visual_loop_z_constrained ||
      visual_only_stats.visual_loop_z_without_ground_suppressed != 1 ||
      visual_only_stats.visual_loop_factors != 1 ||
      visual_only_stats.visual_loop_factor_limit_rejections != 1 ||
      visual_only_stats.last_visual_loop_reason != "visual_loop_factor_limit")
  {
    std::cerr << "visual-only Z suppression regression: accepted="
              << visual_only_stats.last_visual_loop_ground_z_accepted
              << " constrained=" << visual_only_stats.last_visual_loop_z_constrained
              << " suppressed=" << visual_only_stats.visual_loop_z_without_ground_suppressed
              << '\n';
    return 16;
  }

  // A sub-threshold PnP quality is admissible only when an independent
  // LiDAR structural registration validates the same non-adjacent pair.
  hybrid_localization::SemanticPoseGraphOptions lidar_validated_options = options;
  lidar_validated_options.enable_xy_loops = false;
  lidar_validated_options.enable_z_loops = false;
  lidar_validated_options.visual_loop_require_lidar_geometry = true;
  lidar_validated_options.visual_loop_min_quality = 0.85;
  lidar_validated_options.visual_loop_min_quality_with_lidar_geometry = 0.55;
  lidar_validated_options.visual_loop_lidar_max_pnp_xy_disagreement = 0.50;
  lidar_validated_options.visual_loop_lidar_max_pnp_yaw_disagreement_deg = 2.0;
  hybrid_localization::SemanticPoseGraph lidar_validated_graph(lidar_validated_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!lidar_validated_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add LiDAR-validated regression keyframe " << i << '\n';
      return 17;
    }
  }
  if (!lidar_validated_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.60))
  {
    std::cerr << "failed to add LiDAR-validated visual loop\n";
    return 18;
  }
  const hybrid_localization::SemanticPoseGraphStats lidar_validated_stats =
      lidar_validated_graph.stats();
  if (lidar_validated_stats.visual_loop_lidar_geometry_validations != 1 ||
      lidar_validated_stats.visual_loop_lidar_geometry_rejections != 0 ||
      !lidar_validated_stats.last_visual_loop_lidar_accepted ||
      lidar_validated_stats.last_visual_loop_lidar_inliers < 25 ||
      lidar_validated_stats.visual_loop_factors != 1)
  {
    std::cerr << "LiDAR-validated visual loop regression: checks="
              << lidar_validated_stats.visual_loop_lidar_geometry_validations
              << " rejected=" << lidar_validated_stats.visual_loop_lidar_geometry_rejections
              << " inliers=" << lidar_validated_stats.last_visual_loop_lidar_inliers
              << " factors=" << lidar_validated_stats.visual_loop_factors << '\n';
    return 19;
  }

  // On a long return pass the raw frontend can be metres from the revisit.
  // The LiDAR verifier must then refine around an already gated PnP proposal,
  // rather than reject every structural model because it is too far from raw.
  hybrid_localization::SemanticPoseGraphOptions pnp_seeded_options =
      lidar_validated_options;
  pnp_seeded_options.visual_loop_lidar_use_pnp_seed = true;
  pnp_seeded_options.max_xy_correction = 0.20;
  pnp_seeded_options.max_yaw_correction_deg = 1.0;
  hybrid_localization::SemanticPoseGraph pnp_seeded_graph(pnp_seeded_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!pnp_seeded_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add PnP-seeded LiDAR regression keyframe " << i << '\n';
      return 36;
    }
  }
  if (!pnp_seeded_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.60))
  {
    std::cerr << "failed to add PnP-seeded LiDAR visual loop\n";
    return 37;
  }
  const hybrid_localization::SemanticPoseGraphStats pnp_seeded_stats =
      pnp_seeded_graph.stats();
  if (pnp_seeded_stats.visual_loop_lidar_seeded_validations != 1 ||
      !pnp_seeded_stats.last_visual_loop_lidar_seeded ||
      pnp_seeded_stats.visual_loop_lidar_geometry_rejections != 0 ||
      !pnp_seeded_stats.last_visual_loop_lidar_accepted ||
      pnp_seeded_stats.visual_loop_factors != 1)
  {
    std::cerr << "PnP-seeded LiDAR verification regression: seeded="
              << pnp_seeded_stats.visual_loop_lidar_seeded_validations
              << " rejected=" << pnp_seeded_stats.visual_loop_lidar_geometry_rejections
              << " factors=" << pnp_seeded_stats.visual_loop_factors << '\n';
    return 38;
  }

  // A small but spatially distributed ground overlap is allowed only by the
  // visual-loop-specific fallback, and only after the independent LiDAR XY
  // verification above. Force the normal high-count branch to reject so this
  // checks that the fallback cannot accidentally become a generic Z gate.
  hybrid_localization::SemanticPoseGraphOptions sparse_ground_options =
      lidar_validated_options;
  sparse_ground_options.visual_loop_ground_z_min_inliers = 1000;
  sparse_ground_options.visual_loop_ground_z_max_step = 0.25;
  sparse_ground_options.visual_loop_ground_z_sparse_min_inliers = 20;
  sparse_ground_options.visual_loop_ground_z_sparse_min_inlier_ratio = 0.05;
  sparse_ground_options.visual_loop_ground_z_sparse_min_spread = 4.0;
  sparse_ground_options.visual_loop_ground_z_sparse_min_spread_ratio = 0.0;
  sparse_ground_options.visual_loop_ground_z_sparse_max_mad = 0.03;
  sparse_ground_options.visual_loop_ground_z_sparse_min_lidar_xy_inliers = 100;
  sparse_ground_options.visual_loop_ground_z_sparse_max_lidar_xy_rmse = 0.30;
  sparse_ground_options.visual_loop_ground_z_sparse_min_lidar_xy_spread = 4.0;
  hybrid_localization::SemanticPoseGraph sparse_ground_graph(sparse_ground_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!sparse_ground_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add sparse-ground regression keyframe " << i << '\n';
      return 26;
    }
  }
  if (!sparse_ground_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.60))
  {
    std::cerr << "failed to add sparse-ground visual loop\n";
    return 27;
  }
  const hybrid_localization::SemanticPoseGraphStats sparse_ground_stats =
      sparse_ground_graph.stats();
  if (!sparse_ground_stats.last_visual_loop_ground_z_accepted ||
      !sparse_ground_stats.last_visual_loop_ground_z_sparse_accepted ||
      sparse_ground_stats.visual_loop_ground_z_sparse_refinements != 1 ||
      sparse_ground_stats.last_visual_loop_ground_z_inliers < 20 ||
      sparse_ground_stats.last_visual_loop_ground_z_spread < 4.0 ||
      std::abs(sparse_ground_stats.last_visual_loop_ground_z_applied_correction + 0.25) >
          0.02)
  {
    std::cerr << "sparse ground-Z fallback regression: accepted="
              << sparse_ground_stats.last_visual_loop_ground_z_sparse_accepted
              << " inliers=" << sparse_ground_stats.last_visual_loop_ground_z_inliers
              << " spread=" << sparse_ground_stats.last_visual_loop_ground_z_spread
              << " applied="
              << sparse_ground_stats.last_visual_loop_ground_z_applied_correction << '\n';
    return 28;
  }

  // Spatial coverage is not optional: the same high-quality visual/LiDAR
  // loop must remain Z-unconstrained when its accepted ground support is
  // artificially required to span an impossible area.
  hybrid_localization::SemanticPoseGraphOptions sparse_reject_options =
      sparse_ground_options;
  sparse_reject_options.visual_loop_ground_z_sparse_min_spread = 1000.0;
  hybrid_localization::SemanticPoseGraph sparse_reject_graph(sparse_reject_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!sparse_reject_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add sparse-ground rejection keyframe " << i << '\n';
      return 29;
    }
  }
  if (!sparse_reject_graph.addVisualLoopConstraint(0.0, 4.0, biased_visual_loop, 0.60))
  {
    std::cerr << "failed to add sparse-ground rejection loop\n";
    return 30;
  }
  const hybrid_localization::SemanticPoseGraphStats sparse_reject_stats =
      sparse_reject_graph.stats();
  if (sparse_reject_stats.last_visual_loop_ground_z_accepted ||
      sparse_reject_stats.last_visual_loop_ground_z_sparse_accepted ||
      sparse_reject_stats.visual_loop_ground_z_refinements != 0 ||
      sparse_reject_stats.last_visual_loop_z_constrained ||
      sparse_reject_stats.visual_loop_z_without_ground_suppressed != 1)
  {
    std::cerr << "sparse ground-Z coverage gate regression: accepted="
              << sparse_reject_stats.last_visual_loop_ground_z_accepted
              << " sparse=" << sparse_reject_stats.last_visual_loop_ground_z_sparse_accepted
              << " constrained=" << sparse_reject_stats.last_visual_loop_z_constrained
              << '\n';
    return 31;
  }

  // Sequential ground-Z factors are opt-in and must reject a narrow local
  // patch when the configured spatial coverage requirement is not met.
  hybrid_localization::SemanticPoseGraphOptions sequential_ground_options = options;
  sequential_ground_options.enable_xy_loops = false;
  sequential_ground_options.enable_z_loops = false;
  sequential_ground_options.enable_visual_loop_factors = false;
  sequential_ground_options.enable_semantic_observation_factors = false;
  sequential_ground_options.enable_semantic_observation_xy_factors = false;
  sequential_ground_options.enable_semantic_observation_z_factors = false;
  sequential_ground_options.sequential_ground_interval = 1;
  sequential_ground_options.sequential_ground_min_spread = 4.0;
  sequential_ground_options.sequential_ground_min_spread_ratio = 0.05;
  hybrid_localization::SemanticPoseGraph sequential_ground_graph(sequential_ground_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!sequential_ground_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add sequential ground regression keyframe " << i << '\n';
      return 32;
    }
  }
  if (sequential_ground_graph.stats().sequential_ground_factors < 1)
  {
    std::cerr << "sequential ground coverage regression: expected accepted factors\n";
    return 33;
  }
  hybrid_localization::SemanticPoseGraphOptions sequential_ground_reject_options =
      sequential_ground_options;
  sequential_ground_reject_options.sequential_ground_min_spread = 1000.0;
  hybrid_localization::SemanticPoseGraph sequential_ground_reject_graph(
      sequential_ground_reject_options);
  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    const auto semantic = makeObservation(truth[i], world);
    auto geometry = semantic;
    for (auto &point : geometry) point.label = 0U;
    if (!sequential_ground_reject_graph.addFrame(static_cast<double>(i), raw[i], geometry))
    {
      std::cerr << "failed to add sequential ground rejection keyframe " << i << '\n';
      return 34;
    }
  }
  if (sequential_ground_reject_graph.stats().sequential_ground_factors != 0)
  {
    std::cerr << "sequential ground spatial gate regression: factors="
              << sequential_ground_reject_graph.stats().sequential_ground_factors << '\n';
    return 35;
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
