#include "hybrid_localization/core.h"
#include "hybrid_localization/lidar_odometry.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  using namespace hybrid_localization;

  assert(convertSemanticLabel(0, "semantic_kitti") == 0U);
  assert(convertSemanticLabel(40, "semantic_kitti") == 1U);
  assert(convertSemanticLabel(2, "sam3") == 3U);
  assert(convertSemanticLabel(3, "sam3") == 4U);
  assert(convertSemanticLabel(5, "sam3") == 5U);
  assert(convertSemanticLabel(6, "internal") == 6U);

  Eigen::Matrix<double, 6, 1> xi;
  xi << 0.04, -0.02, 0.12, 1.5, -0.7, 0.3;
  const Eigen::Isometry3d transform = expSE3(xi);
  const Eigen::Matrix<double, 6, 1> recovered = logSE3(transform);
  assert((recovered - xi).norm() < 1e-6);

  BevGrid bev;
  bev.reset(40, 40, 0.5, 0.0, 0.0);
  BevPoint point;
  point.point = Eigen::Vector3d(1.0, -1.0, 0.2);
  point.label = 3U;
  point.confidence = 1.0F;
  bev.insert(point, -2.0, 3.0);
  bev.insert(point, -2.0, 3.0);
  bev.insert(point, -2.0, 3.0);
  assert(bev.occupiedAt(1.0, -1.0));

  PriorMap prior;
  prior.width = 80;
  prior.height = 80;
  prior.resolution = 0.5;
  prior.origin_x = -20.0;
  prior.origin_y = -20.0;
  prior.occupancy.assign(static_cast<std::size_t>(prior.width * prior.height), 0.0F);
  prior.labels.assign(prior.occupancy.size(), 0U);
  for (int i = -8; i <= 8; ++i)
  {
    for (int j = -2; j <= 2; ++j)
    {
      int ix = 0;
      int iy = 0;
      assert(prior.worldToCell(static_cast<double>(i), static_cast<double>(j), ix, iy));
      const std::size_t cell = static_cast<std::size_t>(iy) * static_cast<std::size_t>(prior.width) +
                               static_cast<std::size_t>(ix);
      prior.occupancy[cell] = 1.0F;
    }
  }
  prior.recomputeEdges();

  std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> local_points;
  for (int i = -8; i <= 8; ++i)
  {
    BevPoint local;
    local.point = Eigen::Vector3d(static_cast<double>(i), 0.0, 0.0);
    local_points.push_back(local);
  }
  MatcherOptions options;
  options.min_search_radius = 1.0;
  options.max_search_radius = 3.0;
  options.coarse_translation_step = 1.0;
  options.fine_translation_step = 0.25;
  options.yaw_search_deg = 10.0;
  options.min_inlier_ratio = 0.5;
  options.min_confidence = 0.0;
  options.min_score_gap = -1.0;
  PriorMatcher matcher(options);
  const MapMatchResult match = matcher.match(local_points, prior, Eigen::Isometry3d::Identity(),
                                             Eigen::Matrix3d::Identity() * 0.1);
  assert(match.valid);
  assert(match.inlier_ratio > 0.5);

  SlidingWindowOptimizer optimizer(4);
  PoseState first;
  first.pose = Eigen::Isometry3d::Identity();
  PoseState second;
  second.pose = planarTransform(1.0, 0.0, 0.0);
  const int first_index = optimizer.addState(first);
  const int second_index = optimizer.addState(second);
  optimizer.addRelativeFactor(first_index, second_index, planarTransform(1.0, 0.0, 0.0),
                              Eigen::Matrix<double, 6, 6>::Identity(), FactorType::LidarRegistration, 1.0);
  assert(optimizer.optimize(2, 1.0));
  assert(std::abs(optimizer.latestState().pose.translation().x() - 1.0) < 1e-4);

  Eigen::Isometry3d accumulated_pose = Eigen::Isometry3d::Identity();
  Eigen::Matrix<double, 6, 1> small_step;
  small_step << 0.001, -0.0005, 0.0008, 0.01, -0.005, 0.002;
  for (int iteration = 0; iteration < 10000; ++iteration)
  {
    const Eigen::Isometry3d updated = expSE3(small_step) * accumulated_pose;
    accumulated_pose = projectToSE3(updated);
  }
  assert((accumulated_pose.rotation().transpose() * accumulated_pose.rotation() -
          Eigen::Matrix3d::Identity()).norm() < 1e-10);
  assert(std::abs(accumulated_pose.rotation().determinant() - 1.0) < 1e-10);

  PointVector world_points;
  for (int ix = -30; ix <= 30; ++ix)
  {
    for (int iy = -20; iy <= 20; ++iy)
    {
      world_points.emplace_back(0.20 * ix, 0.20 * iy, 0.0);
    }
  }
  for (int ix = -30; ix <= 30; ++ix)
  {
    for (int iz = 0; iz <= 18; ++iz)
    {
      world_points.emplace_back(0.20 * ix, 4.0, 0.20 * iz);
    }
  }
  for (int iy = -20; iy <= 20; ++iy)
  {
    for (int iz = 0; iz <= 18; ++iz)
    {
      world_points.emplace_back(-6.0, 0.20 * iy, 0.20 * iz);
    }
  }
  for (int iz = 0; iz <= 20; ++iz)
  {
    world_points.emplace_back(2.0, -1.0, 0.15 * iz);
    world_points.emplace_back(2.2, -1.0, 0.15 * iz);
    world_points.emplace_back(2.0, -0.8, 0.15 * iz);
  }

  LidarOdometryOptions odometry_options;
  odometry_options.scan_voxel_size = 0.25;
  odometry_options.map_voxel_size = 0.30;
  odometry_options.max_correspondence_distance = 1.5;
  odometry_options.max_plane_distance = 0.60;
  odometry_options.min_scan_points = 100;
  odometry_options.min_correspondences = 80;
  odometry_options.min_inlier_ratio = 0.10;
  odometry_options.max_rmse = 0.45;
  // This synthetic test deliberately starts without IMU propagation and makes
  // a 0.72 m registration correction.  Production configs use much tighter
  // correction gates; set the test's intended scenario explicitly.
  odometry_options.max_lidar_correction_translation = 1.0;
  odometry_options.max_lidar_correction_rotation_deg = 10.0;
  odometry_options.degenerate_min_inlier_ratio = 0.10;
  odometry_options.degenerate_max_rmse = 0.45;
  odometry_options.imu_enabled = false;
  odometry_options.visual_enabled = true;
  odometry_options.visual_min_landmarks = 20;
  odometry_options.visual_min_residuals = 200;
  odometry_options.visual_max_rmse = 1.0;
  odometry_options.visual_fuse_correlated_states = true;
  odometry_options.visual_max_velocity_step = 0.05;
  LidarOdometry lidar_odometry(odometry_options);
  const LidarOdometryResult first_scan = lidar_odometry.processScan(world_points, 1.0);
  assert(first_scan.accepted);
  assert(lidar_odometry.mapPointCount() > 100);

  Eigen::Isometry3d expected_pose = planarTransform(0.65, -0.30, 0.06);
  expected_pose.translation().z() = 0.08;
  PointVector second_body_points;
  second_body_points.reserve(world_points.size());
  for (const Eigen::Vector3d &world_point : world_points)
  {
    second_body_points.push_back(expected_pose.inverse() * world_point);
  }
  const LidarOdometryResult second_scan = lidar_odometry.processScan(second_body_points, 1.1);
  assert(second_scan.accepted);
  assert(second_scan.observable_directions >= 3);
  assert(std::isfinite(second_scan.mean_normalized_residual));
  assert(std::isfinite(second_scan.measurement_condition));
  const Eigen::Matrix<double, 6, 1> registration_error =
      logSE3(expected_pose.inverse() * second_scan.pose);
  assert(registration_error.head<3>().norm() < 0.03);
  assert(registration_error.tail<3>().norm() < 0.10);

  // A single horizontal plane only observes roll, pitch, and height. The
  // scan update must expose that degeneracy rather than reporting a fully
  // constrained six-DoF registration.
  PointVector planar_points;
  for (int ix = -25; ix <= 25; ++ix)
  {
    for (int iy = -25; iy <= 25; ++iy)
    {
      planar_points.emplace_back(0.20 * ix, 0.20 * iy, 0.0);
    }
  }
  LidarOdometryOptions planar_options = odometry_options;
  planar_options.min_observable_directions = 3;
  planar_options.map_insertion_min_observable_directions = 3;
  planar_options.project_lidar_information_to_observable_subspace = true;
  planar_options.max_mean_normalized_residual = 5.0;
  planar_options.map_insertion_max_mean_normalized_residual = 5.0;
  LidarOdometry planar_odometry(planar_options);
  const LidarOdometryResult planar_initial =
      planar_odometry.processScan(planar_points, 1.5);
  assert(planar_initial.accepted);
  const LidarOdometryResult planar_scan =
      planar_odometry.processScan(planar_points, 1.6);
  assert(planar_scan.accepted);
  assert(planar_scan.degenerate);
  assert(planar_scan.observable_directions >= 3);
  assert(planar_scan.observable_directions < 6);
  assert(std::isfinite(planar_scan.mean_normalized_residual));
  const double initial_unobservable_variance =
      planar_initial.covariance(2, 2) +
      planar_initial.covariance(3, 3) +
      planar_initial.covariance(4, 4);
  const double updated_unobservable_variance =
      planar_scan.covariance(2, 2) +
      planar_scan.covariance(3, 3) +
      planar_scan.covariance(4, 4);
  assert(updated_unobservable_variance >=
         0.95 * initial_unobservable_variance);

  // Directional covariance, whitened Huber weighting, and point-wise NIS
  // gating must retain the coherent scan while rejecting individual plane
  // outliers instead of rejecting the whole frame.
  LidarOdometryOptions robust_options = odometry_options;
  robust_options.use_directional_lidar_covariance = true;
  robust_options.lidar_origin_in_body =
      Eigen::Vector3d(0.04, 0.02, 0.06);
  robust_options.lidar_normalized_huber_delta = 2.5;
  robust_options.lidar_innovation_gate = 0.05;
  robust_options.freeze_mature_voxels = true;
  robust_options.max_voxel_points = 8;
  LidarOdometry robust_odometry(robust_options);
  assert(robust_odometry.processScan(world_points, 1.7).accepted);
  PointVector mixed_scan = world_points;
  for (std::size_t index = 0; index < world_points.size(); index += 5U)
  {
    mixed_scan.push_back(world_points[index] +
        Eigen::Vector3d(0.31, -0.27, 0.36));
  }
  const LidarOdometryResult robust_scan =
      robust_odometry.processScan(mixed_scan, 1.8);
  assert(robust_scan.accepted);
  assert(robust_scan.innovation_rejections > 0);
  assert(robust_scan.mean_robust_weight > 0.0);
  assert(robust_scan.mean_robust_weight <= 1.0);
  assert(robust_scan.covariance.allFinite());

  // Motion-based map insertion must skip highly correlated stationary scans
  // but refresh after the configured maximum interval.
  LidarOdometryOptions keyframe_options = odometry_options;
  keyframe_options.map_insertion_min_translation = 0.50;
  keyframe_options.map_insertion_min_rotation_deg = 5.0;
  keyframe_options.map_insertion_max_interval = 1.0;
  LidarOdometry keyframe_odometry(keyframe_options);
  const LidarOdometryResult keyframe_initial =
      keyframe_odometry.processScan(world_points, 5.0);
  assert(keyframe_initial.accepted);
  assert(keyframe_initial.map_keyframe_selected);
  const LidarOdometryResult keyframe_skipped =
      keyframe_odometry.processScan(world_points, 5.1);
  assert(keyframe_skipped.accepted);
  assert(!keyframe_skipped.map_keyframe_selected);
  assert(!keyframe_skipped.map_updated);
  assert(keyframe_skipped.map_update_deferred);
  const LidarOdometryResult keyframe_refresh =
      keyframe_odometry.processScan(world_points, 6.1);
  assert(keyframe_refresh.accepted);
  assert(keyframe_refresh.map_keyframe_selected);
  assert(keyframe_refresh.map_updated);

  // Exercise the FAST-LIO-style sample KNN plane path used as a fallback in
  // sparse parts of the rolling voxel map.
  LidarOdometryOptions knn_options = odometry_options;
  knn_options.use_point_knn_plane = true;
  knn_options.point_knn_fallback = false;
  LidarOdometry knn_odometry(knn_options);
  assert(knn_odometry.processScan(world_points, 2.0).accepted);
  const LidarOdometryResult knn_scan = knn_odometry.processScan(second_body_points, 2.1);
  assert(knn_scan.accepted);
  const Eigen::Matrix<double, 6, 1> knn_registration_error =
      logSE3(expected_pose.inverse() * knn_scan.pose);
  assert(knn_registration_error.head<3>().norm() < 0.04);
  assert(knn_registration_error.tail<3>().norm() < 0.12);

  // Finite plane support and query-dependent extrapolation uncertainty must
  // retain a coherent scan in both the smooth voxel and KNN fallback paths.
  LidarOdometryOptions support_options = odometry_options;
  support_options.plane_support_radius_scale = 3.0;
  support_options.plane_extrapolation_uncertainty_scale = 1.0;
  support_options.plane_parameter_uncertainty_scale = 1.0;
  support_options.smooth_voxel_robust_refit = true;
  support_options.use_adaptive_subvoxel_plane = true;
  support_options.max_adaptive_subvoxels = 20000;
  support_options.point_knn_fallback = true;
  support_options.point_knn_fallback_max_queries = 200;
  LidarOdometry support_odometry(support_options);
  assert(support_odometry.processScan(world_points, 2.2).accepted);
  const LidarOdometryResult support_scan =
      support_odometry.processScan(second_body_points, 2.3);
  assert(support_scan.accepted);
  assert(support_scan.covariance.allFinite());

  // A wide-FOV scan can have a low global correspondence fraction even when
  // it has thousands of spatially distributed, low-residual constraints.
  // Strong support must be accepted without disabling the normal bad-scan
  // gates for weak or clustered observations.
  LidarOdometryOptions strong_support_options = odometry_options;
  strong_support_options.max_scan_points = 20000;
  strong_support_options.min_inlier_ratio = 0.80;
  strong_support_options.degenerate_min_inlier_ratio = 0.80;
  strong_support_options.strong_support_min_correspondences = 300;
  strong_support_options.strong_support_min_azimuth_sectors = 5;
  strong_support_options.strong_support_max_rmse = 0.45;
  LidarOdometry strong_support_odometry(strong_support_options);
  assert(strong_support_odometry.processScan(world_points, 3.0).accepted);
  PointVector wide_fov_second_points = second_body_points;
  wide_fov_second_points.reserve(second_body_points.size() + 8000U);
  for (int index = 0; index < 8000; ++index)
  {
    const int ix = index % 100;
    const int iy = (index / 100) % 40;
    const int iz = (index / 4000) % 2;
    wide_fov_second_points.emplace_back(60.0 + 0.40 * static_cast<double>(ix),
                                        -40.0 + 0.40 * static_cast<double>(iy),
                                        -2.0 + 0.80 * static_cast<double>(iz));
  }
  const LidarOdometryResult strong_support_scan =
      strong_support_odometry.processScan(wide_fov_second_points, 3.1);
  assert(strong_support_scan.inlier_ratio < strong_support_options.min_inlier_ratio);
  assert(strong_support_scan.strong_support);
  assert(strong_support_scan.accepted);

  // Registration may be delayed by the timestamp scheduler. Verify that a
  // physically plausible 0.7 s displacement uses the configured speed gate
  // rather than being rejected by the nominal per-scan displacement alone.
  LidarOdometryOptions delayed_options = odometry_options;
  delayed_options.max_translation_per_scan = 0.20;
  delayed_options.max_rotation_per_scan_deg = 2.0;
  delayed_options.max_translation_speed = 2.0;
  delayed_options.max_rotation_speed_deg = 20.0;
  delayed_options.max_lidar_correction_translation = 1.2;
  delayed_options.max_lidar_correction_rotation_deg = 10.0;
  LidarOdometry delayed_odometry(delayed_options);
  assert(delayed_odometry.processScan(world_points, 20.0).accepted);
  const LidarOdometryResult delayed_scan = delayed_odometry.processScan(
      second_body_points, 20.7);
  assert(delayed_scan.accepted);

  const Eigen::Vector3d visual_prediction =
      second_scan.pose.translation() + 0.1 * second_scan.velocity;
  const Eigen::Vector3d visual_target = visual_prediction +
      Eigen::Vector3d(0.01, -0.005, 0.002);
  const double visual_error_before =
      (visual_prediction - visual_target).norm();
  const VisualUpdateResult visual_update = lidar_odometry.processVisual(
      1.2, [&visual_target](const Eigen::Isometry3d &pose)
      {
        VisualPoseLinearization linearization;
        linearization.valid = true;
        linearization.landmarks = 40;
        linearization.residuals = 400;
        linearization.rmse = 0.2;
        linearization.mean_ncc = 0.95;
        linearization.hessian = Eigen::Matrix<double, 6, 6>::Identity() * 1000.0;
        linearization.gradient.tail<3>() =
            1000.0 * (pose.translation() - visual_target);
        linearization.reason = "synthetic_visual_measurement";
        return linearization;
      });
  assert(visual_update.propagated);
  assert(visual_update.accepted);
  assert((visual_update.pose.translation() - visual_target).norm() <
         visual_error_before);
  assert(visual_update.velocity_correction.allFinite());
  assert(visual_update.velocity_correction.norm() > 1e-8);
  assert(visual_update.velocity_correction.norm() <=
         odometry_options.visual_max_velocity_step + 1e-8);

  // A forward-facing direct image update must not manufacture roll, pitch, or
  // Z corrections when the fusion profile assigns those axes to LiDAR/IMU.
  LidarOdometryOptions constrained_visual_options = odometry_options;
  constrained_visual_options.visual_enabled = true;
  constrained_visual_options.visual_fuse_roll_pitch = false;
  constrained_visual_options.visual_fuse_yaw = true;
  constrained_visual_options.visual_fuse_translation_xy = true;
  constrained_visual_options.visual_fuse_translation_z = false;
  constrained_visual_options.visual_observability_eigen_ratio = 1e-3;
  constrained_visual_options.visual_min_observable_directions = 3;
  constrained_visual_options.visual_max_rotation_step_deg = 4.0;
  constrained_visual_options.visual_max_translation_step = 0.20;
  LidarOdometry constrained_visual_odometry(constrained_visual_options);
  assert(constrained_visual_odometry.processScan(world_points, 21.0).accepted);
  const Eigen::Vector3d constrained_visual_target(0.04, -0.03, 0.07);
  const VisualUpdateResult constrained_visual_update =
      constrained_visual_odometry.processVisual(
          21.1, [&constrained_visual_target](const Eigen::Isometry3d &pose)
          {
            VisualPoseLinearization linearization;
            linearization.valid = true;
            linearization.landmarks = 40;
            linearization.residuals = 400;
            linearization.rmse = 0.2;
            linearization.mean_ncc = 0.95;
            linearization.hessian = Eigen::Matrix<double, 6, 6>::Identity() * 1000.0;
            linearization.gradient(0) = 100.0;
            linearization.gradient(1) = -80.0;
            linearization.gradient(2) = 1000.0 *
                (std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0)) - 0.02);
            linearization.gradient.tail<3>() = 1000.0 *
                (pose.translation() - constrained_visual_target);
            linearization.reason = "synthetic_constrained_visual_measurement";
            return linearization;
          });
  assert(constrained_visual_update.accepted);
  assert(constrained_visual_update.observable_directions == 3);
  assert(constrained_visual_update.correction.translation().head<2>().norm() > 1e-3);
  assert(std::abs(constrained_visual_update.correction.translation().z()) < 1e-8);
  const Eigen::Matrix3d constrained_rotation =
      constrained_visual_update.correction.rotation();
  assert(std::abs(constrained_rotation(2, 0)) < 1e-8);
  assert(std::abs(constrained_rotation(2, 1)) < 1e-8);
  assert(std::abs(std::atan2(constrained_rotation(1, 0),
                             constrained_rotation(0, 0))) > 1e-3);

  // A two-direction direct alignment needs stricter image quality before it
  // can replace the normal three-direction observability requirement.
  LidarOdometryOptions two_mode_quality_options = constrained_visual_options;
  two_mode_quality_options.visual_min_observable_directions = 2;
  two_mode_quality_options.visual_two_mode_min_mean_ncc = 0.97;
  LidarOdometry two_mode_quality_odometry(two_mode_quality_options);
  assert(two_mode_quality_odometry.processScan(world_points, 21.5).accepted);
  const VisualUpdateResult rejected_two_mode_update =
      two_mode_quality_odometry.processVisual(
          21.6, [](const Eigen::Isometry3d &)
          {
            VisualPoseLinearization linearization;
            linearization.valid = true;
            linearization.landmarks = 80;
            linearization.residuals = 800;
            linearization.rmse = 0.20;
            linearization.mean_ncc = 0.95;
            linearization.hessian = Eigen::Matrix<double, 6, 6>::Zero();
            linearization.hessian(2, 2) = 1000.0;
            linearization.hessian(3, 3) = 1000.0;
            linearization.gradient(2) = 15.0;
            linearization.gradient(3) = -20.0;
            linearization.reason = "synthetic_two_mode_measurement";
            return linearization;
          });
  assert(!rejected_two_mode_update.accepted);
  assert(rejected_two_mode_update.observable_directions == 2);
  assert(rejected_two_mode_update.reason == "visual_two_mode_quality_gate");

  two_mode_quality_options.visual_two_mode_min_mean_ncc = 0.94;
  LidarOdometry accepted_two_mode_odometry(two_mode_quality_options);
  assert(accepted_two_mode_odometry.processScan(world_points, 21.7).accepted);
  const VisualUpdateResult accepted_two_mode_update =
      accepted_two_mode_odometry.processVisual(
          21.8, [](const Eigen::Isometry3d &)
          {
            VisualPoseLinearization linearization;
            linearization.valid = true;
            linearization.landmarks = 80;
            linearization.residuals = 800;
            linearization.rmse = 0.20;
            linearization.mean_ncc = 0.95;
            linearization.hessian = Eigen::Matrix<double, 6, 6>::Zero();
            linearization.hessian(2, 2) = 1000.0;
            linearization.hessian(3, 3) = 1000.0;
            linearization.gradient(2) = 15.0;
            linearization.gradient(3) = -20.0;
            linearization.reason = "synthetic_two_mode_measurement";
            return linearization;
          });
  assert(accepted_two_mode_update.accepted);
  assert(accepted_two_mode_update.observable_directions == 2);

  // A LiDAR-depth visual map can contribute altitude only when Z is an
  // independently conditioned photometric direction. The automatic path is
  // bounded over the entire ESKF update, not once per Gauss-Newton step.
  LidarOdometryOptions gated_z_options = constrained_visual_options;
  gated_z_options.visual_fuse_translation_z_when_observable = true;
  gated_z_options.visual_z_min_projection = 0.92;
  gated_z_options.visual_z_min_conditional_information_ratio = 0.20;
  gated_z_options.visual_max_z_step = 0.015;
  LidarOdometry gated_z_odometry(gated_z_options);
  assert(gated_z_odometry.processScan(world_points, 22.0).accepted);
  const Eigen::Vector3d gated_z_target(0.03, -0.02, 0.10);
  const VisualUpdateResult gated_z_update = gated_z_odometry.processVisual(
      22.1, [&gated_z_target](const Eigen::Isometry3d &pose)
      {
        VisualPoseLinearization linearization;
        linearization.valid = true;
        linearization.landmarks = 40;
        linearization.residuals = 400;
        linearization.rmse = 0.2;
        linearization.mean_ncc = 0.95;
        linearization.hessian = Eigen::Matrix<double, 6, 6>::Identity() * 1000.0;
        linearization.gradient.tail<3>() = 1000.0 *
            (pose.translation() - gated_z_target);
        linearization.reason = "synthetic_gated_z_visual_measurement";
        return linearization;
      });
  assert(gated_z_update.accepted);
  assert(gated_z_update.z_observable);
  assert(gated_z_update.z_fused);
  assert(gated_z_update.z_projection > 0.99);
  assert(gated_z_update.z_conditional_information_ratio > 0.99);
  assert(std::abs(gated_z_update.z_correction) > 0.005);
  assert(std::abs(gated_z_update.z_correction) <= 0.015 + 1e-8);

  // A single yaw/XY/Z coupled image direction must not masquerade as a
  // vertical measurement. The auto-Z gate re-projects the update onto the
  // remaining yaw/XY subspace and leaves altitude to LiDAR/IMU.
  LidarOdometry coupled_z_odometry(gated_z_options);
  assert(coupled_z_odometry.processScan(world_points, 23.0).accepted);
  const Eigen::Vector3d coupled_z_target(0.03, -0.02, 0.10);
  const VisualUpdateResult coupled_z_update = coupled_z_odometry.processVisual(
      23.1, [&coupled_z_target](const Eigen::Isometry3d &pose)
      {
        VisualPoseLinearization linearization;
        linearization.valid = true;
        linearization.landmarks = 40;
        linearization.residuals = 400;
        linearization.rmse = 0.2;
        linearization.mean_ncc = 0.95;
        linearization.hessian.setZero();
        linearization.hessian(2, 2) = 1000.0;
        linearization.hessian(3, 3) = 1000.0;
        linearization.hessian(4, 4) = 1000.0;
        linearization.hessian(3, 5) = 1000.0;
        linearization.hessian(5, 3) = 1000.0;
        linearization.hessian(5, 5) = 1000.0;
        linearization.gradient(2) = 1000.0 *
            std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
        linearization.gradient.tail<3>() = 1000.0 *
            (pose.translation() - coupled_z_target);
        linearization.reason = "synthetic_coupled_z_visual_measurement";
        return linearization;
      });
  assert(coupled_z_update.accepted);
  assert(!coupled_z_update.z_fused);
  assert(std::abs(coupled_z_update.z_correction) < 1e-8);

  PointVector unrelated_scan;
  unrelated_scan.reserve(world_points.size());
  for (const Eigen::Vector3d &world_point : world_points)
  {
    unrelated_scan.push_back(world_point + Eigen::Vector3d(100.0, 0.0, 0.0));
  }
  const Eigen::Isometry3d pose_before_rejection = lidar_odometry.pose();
  const std::size_t map_size_before_rejection = lidar_odometry.mapPointCount();
  const LidarOdometryResult rejected_scan = lidar_odometry.processScan(unrelated_scan, 1.3);
  assert(!rejected_scan.accepted);
  assert(!rejected_scan.map_updated);
  assert(lidar_odometry.mapPointCount() == map_size_before_rejection);
  assert((rejected_scan.pose.translation() - pose_before_rejection.translation()).norm() < 1.0);

  // A bounded KNN fallback must cap the expensive sample searches even when
  // the entire scan is outside the rolling local map.
  LidarOdometryOptions fallback_budget_options = odometry_options;
  fallback_budget_options.point_knn_fallback = true;
  fallback_budget_options.point_knn_fallback_max_queries = 7;
  LidarOdometry fallback_budget_odometry(fallback_budget_options);
  assert(fallback_budget_odometry.processScan(world_points, 4.0).accepted);
  const LidarOdometryResult fallback_budget_scan =
      fallback_budget_odometry.processScan(unrelated_scan, 4.1);
  assert(!fallback_budget_scan.accepted);
  assert(fallback_budget_scan.point_knn_fallback_queries == 7);
  assert(fallback_budget_scan.point_knn_fallback_matches == 0);

  LidarOdometryOptions inertial_options = odometry_options;
  inertial_options.imu_enabled = true;
  inertial_options.imu_init_duration = 1.0;
  inertial_options.imu_init_samples = 150;
  inertial_options.imu_init_require_stationary = true;
  inertial_options.imu_max_gap = 0.02;
  inertial_options.imu_init_use_mean_covariance = true;
  inertial_options.imu_init_gyro_bias_covariance_floor = 2e-8;
  inertial_options.imu_init_acceleration_bias_covariance = 3e-5;
  inertial_options.imu_init_gravity_covariance_floor = 4e-6;
  LidarOdometry inertial_odometry(inertial_options);
  const Eigen::Vector3d known_gyro_bias(0.010, -0.005, 0.002);
  for (int index = 0; index <= 1000; ++index)
  {
    ImuSample sample;
    sample.stamp = 10.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    inertial_odometry.addImuSample(sample);
  }
  assert(inertial_odometry.imuInitialized());
  assert((inertial_odometry.gyroBias() - known_gyro_bias).norm() < 1e-6);
  assert((inertial_odometry.gravity() - Eigen::Vector3d(0.0, 0.0, -9.81)).norm() < 1e-6);
  const Matrix18d &initialized_covariance = inertial_odometry.stateCovariance();
  assert((initialized_covariance.block<3, 3>(9, 9).diagonal() -
          Eigen::Vector3d::Constant(2e-8)).cwiseAbs().maxCoeff() < 1e-12);
  assert((initialized_covariance.block<3, 3>(12, 12).diagonal() -
          Eigen::Vector3d::Constant(3e-5)).cwiseAbs().maxCoeff() < 1e-12);
  assert(std::abs(initialized_covariance(15, 15) - 4e-6) < 1e-12);
  assert(std::abs(initialized_covariance(16, 16) - 4e-6) < 1e-12);
  assert(initialized_covariance(17, 17) < 1e-8);

  LidarOdometryResult inertial_result = inertial_odometry.processScan(world_points, 12.0);
  assert(inertial_result.accepted);
  for (int frame = 1; frame <= 20; ++frame)
  {
    inertial_result = inertial_odometry.processScan(
        world_points, 12.0 + 0.1 * static_cast<double>(frame));
    assert(inertial_result.accepted);
  }
  std::cout << "stationary_position_norm=" << inertial_result.pose.translation().norm()
            << " stationary_velocity_norm=" << inertial_result.velocity.norm() << "\n";
  assert(inertial_result.pose.translation().norm() < 0.05);
  assert(inertial_result.velocity.norm() < 0.02);
  assert(std::abs(inertial_result.gravity.norm() - 9.81) < 1e-6);

  // A registration outage must not turn into unbounded IMU-only Z drift.
  // The LiDAR scan is deliberately outside the local map while the IMU
  // reports a persistent vertical acceleration error.
  LidarOdometryOptions loss_options = inertial_options;
  loss_options.lidar_loss_hold_after_rejections = 3;
  loss_options.lidar_loss_freeze_after_rejections = 5;
  loss_options.lidar_loss_max_vertical_offset = 0.15;
  loss_options.lidar_loss_max_horizontal_speed = 1.0;
  loss_options.lidar_loss_max_horizontal_step = 0.10;
  loss_options.lidar_loss_velocity_decay = 0.95;
  loss_options.strong_support_min_correspondences = 100;
  loss_options.strong_support_min_azimuth_sectors = 1;
  loss_options.strong_support_max_rmse = 0.50;
  loss_options.recovery_map_insert_min_consecutive_strong_support = 2;
  LidarOdometry loss_odometry(loss_options);
  for (int index = 0; index <= 400; ++index)
  {
    ImuSample sample;
    sample.stamp = 30.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    loss_odometry.addImuSample(sample);
  }
  assert(loss_odometry.imuInitialized());
  LidarOdometryResult loss_result = loss_odometry.processScan(world_points, 32.0);
  assert(loss_result.accepted);
  const Eigen::Vector3d loss_reference_position = loss_result.pose.translation();
  for (int frame = 1; frame <= 8; ++frame)
  {
    for (int substep = 1; substep <= 20; ++substep)
    {
      ImuSample sample;
      sample.stamp = 32.0 + 0.1 * static_cast<double>(frame - 1) +
          0.005 * static_cast<double>(substep);
      sample.acceleration = Eigen::Vector3d(4.0, 0.0, 7.0);
      sample.angular_velocity = known_gyro_bias;
      loss_odometry.addImuSample(sample);
    }
    loss_result = loss_odometry.processScan(
        unrelated_scan, 32.0 + 0.1 * static_cast<double>(frame));
    assert(!loss_result.accepted);
  }
  std::cout << "loss_limited=" << (loss_result.loss_limited ? 1 : 0)
            << " frozen=" << (loss_result.loss_frozen ? 1 : 0)
            << " rejected=" << loss_result.consecutive_rejections
            << " position_offset=" <<
                (loss_result.pose.translation() - loss_reference_position).norm()
            << "\n";
  assert(loss_result.loss_limited);
  assert(loss_result.loss_frozen);
  assert(loss_result.consecutive_rejections >= 3);
  assert((loss_result.pose.translation() - loss_reference_position).norm() < 1e-9);
  assert(std::abs(loss_result.velocity.z()) < 1e-9);

  // Recovery matches can be geometrically plausible before the vehicle has
  // re-entered stable support. They may update the state, but must not seed a
  // new local map until enough consecutive strong scans confirm the recovery.
  for (int frame = 1; frame <= 2; ++frame)
  {
    for (int substep = 1; substep <= 20; ++substep)
    {
      ImuSample sample;
      sample.stamp = 32.8 + 0.1 * static_cast<double>(frame - 1) +
          0.005 * static_cast<double>(substep);
      sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
      sample.angular_velocity = known_gyro_bias;
      loss_odometry.addImuSample(sample);
    }
    loss_result = loss_odometry.processScan(
        world_points, 32.8 + 0.1 * static_cast<double>(frame));
    assert(loss_result.accepted);
    assert(loss_result.strong_support);
    if (frame == 1)
    {
      assert(!loss_result.map_updated);
      assert(loss_result.map_update_deferred);
    }
    else
    {
      assert(loss_result.map_updated);
    }
  }

  std::cout << "hybrid_localization_core_smoke_test: PASS\n";
  return 0;
}
