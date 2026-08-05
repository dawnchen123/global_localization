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
  assert(second_scan.converged);
  assert(second_scan.convergence_confirmations >= 2);
  assert(second_scan.final_linearization_valid);
  assert(second_scan.observable_directions >= 3);
  assert(std::isfinite(second_scan.mean_normalized_residual));
  assert(std::isfinite(second_scan.measurement_condition));
  const Eigen::Matrix<double, 6, 1> registration_error =
      logSE3(expected_pose.inverse() * second_scan.pose);
  assert(registration_error.head<3>().norm() < 0.03);
  assert(registration_error.tail<3>().norm() < 0.10);

  // Parallel association must be numerically identical to the serial path.
  // The fallback budget is selected before dispatch and the normal equation
  // is reduced in scan order, so thread scheduling cannot change acceptance,
  // convergence, or the pose written into the local map.
  LidarOdometryOptions serial_registration_options = odometry_options;
  serial_registration_options.registration_threads = 1;
  serial_registration_options.point_knn_fallback = true;
  serial_registration_options.point_knn_fallback_max_queries = 200;
  LidarOdometryOptions parallel_registration_options =
      serial_registration_options;
  parallel_registration_options.registration_threads = 2;
  LidarOdometry serial_registration_odometry(serial_registration_options);
  LidarOdometry parallel_registration_odometry(parallel_registration_options);
  const LidarOdometryResult serial_registration_initial =
      serial_registration_odometry.processScan(world_points, 1.4);
  const LidarOdometryResult parallel_registration_initial =
      parallel_registration_odometry.processScan(world_points, 1.4);
  assert(serial_registration_initial.accepted);
  assert(parallel_registration_initial.accepted);
  const LidarOdometryResult serial_registration_result =
      serial_registration_odometry.processScan(second_body_points, 1.5);
  const LidarOdometryResult parallel_registration_result =
      parallel_registration_odometry.processScan(second_body_points, 1.5);
  assert(serial_registration_result.accepted ==
         parallel_registration_result.accepted);
  assert(serial_registration_result.converged ==
         parallel_registration_result.converged);
  assert(serial_registration_result.correspondences ==
         parallel_registration_result.correspondences);
  assert(serial_registration_result.point_knn_fallback_queries ==
         parallel_registration_result.point_knn_fallback_queries);
  assert(serial_registration_result.point_knn_fallback_matches ==
         parallel_registration_result.point_knn_fallback_matches);
  assert((serial_registration_result.pose.matrix() -
          parallel_registration_result.pose.matrix()).norm() < 1e-12);
  assert((serial_registration_result.covariance -
          parallel_registration_result.covariance).norm() < 1e-12);

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

  // A reduced mature-plane gain must remain numerically stable after the
  // voxel reaches its bounded sufficient-statistics capacity.
  LidarOdometryOptions slow_map_options = odometry_options;
  slow_map_options.max_voxel_points = 8;
  slow_map_options.freeze_mature_voxels = false;
  slow_map_options.mature_voxel_update_gain = 0.10;
  LidarOdometry slow_map_odometry(slow_map_options);
  assert(slow_map_odometry.processScan(world_points, 1.9).accepted);
  assert(slow_map_odometry.processScan(world_points, 2.0).accepted);
  const LidarOdometryResult slow_map_scan =
      slow_map_odometry.processScan(world_points, 2.1);
  assert(slow_map_scan.accepted);
  assert(slow_map_scan.covariance.allFinite());

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

  // A scan that exhausts its update budget is evaluated once more at the
  // actual final pose. It may remain usable as a state update in a permissive
  // profile, but must not modify the persistent map until rematching confirms
  // convergence. A strict profile rejects the same marginal registration.
  LidarOdometryOptions unconfirmed_options = odometry_options;
  unconfirmed_options.max_iterations = 1;
  unconfirmed_options.convergence_confirmation_iterations = 2;
  unconfirmed_options.map_insertion_require_convergence = true;
  unconfirmed_options.require_convergence_for_acceptance = false;
  LidarOdometry unconfirmed_odometry(unconfirmed_options);
  assert(unconfirmed_odometry.processScan(world_points, 6.5).accepted);
  Eigen::Isometry3d unconfirmed_pose = planarTransform(0.25, -0.12, 0.04);
  PointVector unconfirmed_scan;
  unconfirmed_scan.reserve(world_points.size());
  for (const Eigen::Vector3d &world_point : world_points)
  {
    unconfirmed_scan.push_back(unconfirmed_pose.inverse() * world_point);
  }
  const LidarOdometryResult unconfirmed_result =
      unconfirmed_odometry.processScan(unconfirmed_scan, 6.6);
  assert(unconfirmed_result.accepted);
  assert(unconfirmed_result.final_linearization_valid);
  assert(!unconfirmed_result.converged);
  assert(!unconfirmed_result.map_updated);
  assert(unconfirmed_result.map_update_deferred);
  assert(unconfirmed_result.map_update_reason == "registration_not_converged");

  LidarOdometryOptions strict_convergence_options = unconfirmed_options;
  strict_convergence_options.require_convergence_for_acceptance = true;
  LidarOdometry strict_convergence_odometry(strict_convergence_options);
  assert(strict_convergence_odometry.processScan(world_points, 6.5).accepted);
  const LidarOdometryResult strict_convergence_result =
      strict_convergence_odometry.processScan(unconfirmed_scan, 6.6);
  assert(!strict_convergence_result.accepted);
  assert(strict_convergence_result.reject_reason ==
         "registration_not_converged");

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

  // Low-residual geometry alone must not override a precise stationary IMU
  // prediction with a contradictory rotation. The aggregate correction NIS
  // catches this wrong-but-locally-converged scan before it reaches the map.
  LidarOdometryOptions rotation_nis_options = inertial_options;
  rotation_nis_options.max_lidar_correction_rotation_deg = 10.0;
  rotation_nis_options.lidar_rotation_correction_nis_gate = 1e-8;
  rotation_nis_options.lidar_rotation_correction_std_floor_deg = 0.10;
  LidarOdometry rotation_nis_odometry(rotation_nis_options);
  for (int index = 0; index <= 440; ++index)
  {
    ImuSample sample;
    sample.stamp = 30.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    rotation_nis_odometry.addImuSample(sample);
  }
  assert(rotation_nis_odometry.imuInitialized());
  assert(rotation_nis_odometry.processScan(world_points, 32.0).accepted);
  Eigen::Isometry3d contradictory_rotation = Eigen::Isometry3d::Identity();
  contradictory_rotation.linear() = Eigen::AngleAxisd(
      0.08, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  PointVector contradictory_rotation_scan;
  contradictory_rotation_scan.reserve(world_points.size());
  for (const Eigen::Vector3d &world_point : world_points)
  {
    contradictory_rotation_scan.push_back(
        contradictory_rotation.inverse() * world_point);
  }
  const LidarOdometryResult rotation_nis_result =
      rotation_nis_odometry.processScan(contradictory_rotation_scan, 32.1);
  assert(!rotation_nis_result.accepted);
  assert(rotation_nis_result.reject_reason ==
         "rotation_correction_innovation");
  assert(rotation_nis_result.lidar_rotation_correction_nis >
         rotation_nis_options.lidar_rotation_correction_nis_gate);

  // Hidden-state bounds apply to the complete iterated scan. A contradictory
  // but geometrically valid observation may use all IEKF iterations without
  // multiplying the configured gyro-bias allowance by the iteration count.
  LidarOdometryOptions bounded_hidden_options = inertial_options;
  bounded_hidden_options.max_lidar_correction_rotation_deg = 10.0;
  bounded_hidden_options.max_lidar_gyro_bias_step = 1e-5;
  LidarOdometry bounded_hidden_odometry(bounded_hidden_options);
  for (int index = 0; index <= 440; ++index)
  {
    ImuSample sample;
    sample.stamp = 30.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    bounded_hidden_odometry.addImuSample(sample);
  }
  assert(bounded_hidden_odometry.processScan(world_points, 32.0).accepted);
  const Eigen::Vector3d gyro_bias_before_bounded_scan =
      bounded_hidden_odometry.gyroBias();
  const LidarOdometryResult bounded_hidden_result =
      bounded_hidden_odometry.processScan(contradictory_rotation_scan, 32.1);
  assert(bounded_hidden_result.accepted);
  assert((bounded_hidden_result.gyro_bias - gyro_bias_before_bounded_scan).norm() <=
         bounded_hidden_options.max_lidar_gyro_bias_step + 1e-9);

  // The time-window guard is evaluated before state injection and map write.
  LidarOdometryOptions persistent_yaw_options = bounded_hidden_options;
  persistent_yaw_options.lidar_yaw_correction_window_sec = 10.0;
  persistent_yaw_options.max_cumulative_lidar_yaw_correction_deg = 1.0;
  LidarOdometry persistent_yaw_odometry(persistent_yaw_options);
  for (int index = 0; index <= 440; ++index)
  {
    ImuSample sample;
    sample.stamp = 30.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    persistent_yaw_odometry.addImuSample(sample);
  }
  assert(persistent_yaw_odometry.processScan(world_points, 32.0).accepted);
  const LidarOdometryResult persistent_yaw_result =
      persistent_yaw_odometry.processScan(contradictory_rotation_scan, 32.1);
  assert(!persistent_yaw_result.accepted);
  assert(persistent_yaw_result.reject_reason ==
         "persistent_lidar_yaw_correction");
  assert(std::abs(persistent_yaw_result.cumulative_lidar_yaw_correction_deg) >
         persistent_yaw_options.max_cumulative_lidar_yaw_correction_deg);

  // Dataset profiles may preserve the well-supported non-yaw portion of the
  // same registration. The cumulative gravity-axis correction is limited,
  // its information is removed on the final relinearization, and the guarded
  // pose is never inserted into the persistent map.
  LidarOdometryOptions selective_yaw_options = persistent_yaw_options;
  selective_yaw_options.limit_cumulative_lidar_yaw_correction = true;
  selective_yaw_options.limited_lidar_yaw_information_scale = 0.0;
  selective_yaw_options.defer_map_when_lidar_yaw_limited = true;
  LidarOdometry selective_yaw_odometry(selective_yaw_options);
  for (int index = 0; index <= 440; ++index)
  {
    ImuSample sample;
    sample.stamp = 30.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    selective_yaw_odometry.addImuSample(sample);
  }
  assert(selective_yaw_odometry.processScan(world_points, 32.0).accepted);
  const LidarOdometryResult selective_yaw_result =
      selective_yaw_odometry.processScan(contradictory_rotation_scan, 32.1);
  assert(selective_yaw_result.accepted);
  assert(selective_yaw_result.lidar_yaw_correction_limited);
  assert(std::abs(selective_yaw_result.cumulative_lidar_yaw_correction_deg) <=
         selective_yaw_options.max_cumulative_lidar_yaw_correction_deg +
             1e-6);
  assert(selective_yaw_result.lidar_yaw_information_scale == 0.0);
  assert(!selective_yaw_result.map_updated);

  // A delayed AT128 frame during a sharp turn can legitimately exceed the
  // fixed per-scan rotation gate. Complete IMU propagation may expand only
  // the total-motion gate; the correction relative to IMU remains bounded
  // and is expanded modestly only when Ranger differential speed agrees.
  LidarOdometryOptions fixed_turn_options = inertial_options;
  fixed_turn_options.max_rotation_per_scan_deg = 8.0;
  fixed_turn_options.max_rotation_speed_deg = 40.0;
  fixed_turn_options.max_lidar_correction_rotation_deg = 2.8;
  fixed_turn_options.wheel_enabled = true;
  fixed_turn_options.wheel_yaw_rate_scale = -1.0 / 0.43;
  fixed_turn_options.wheel_yaw_rate_min_speed = 0.0;
  fixed_turn_options.wheel_yaw_rate_noise = 0.12;
  fixed_turn_options.wheel_yaw_rate_innovation_gate = 16.0;
  fixed_turn_options.wheel_yaw_rate_max_imu_difference = 0.20;
  fixed_turn_options.turn_aware_motion_gate_enabled = false;
  LidarOdometryOptions turn_aware_options = fixed_turn_options;
  turn_aware_options.turn_aware_motion_gate_enabled = true;
  turn_aware_options.turn_aware_rotation_margin_deg = 3.0;
  turn_aware_options.turn_aware_max_rotation_deg = 90.0;
  turn_aware_options.turn_aware_max_scan_dt = 1.25;
  turn_aware_options.turn_aware_min_yaw_rate = 0.20;
  turn_aware_options.turn_aware_lidar_correction_rotation_deg = 4.2;
  turn_aware_options.turn_aware_wheel_imu_max_yaw_rate_difference = 0.20;
  turn_aware_options.wheel_yaw_rate_relative_scale_uncertainty = 0.20;
  LidarOdometry fixed_turn_odometry(fixed_turn_options);
  LidarOdometry turn_aware_odometry(turn_aware_options);
  for (int index = 0; index <= 400; ++index)
  {
    ImuSample sample;
    sample.stamp = 20.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    fixed_turn_odometry.addImuSample(sample);
    turn_aware_odometry.addImuSample(sample);
  }
  assert(fixed_turn_odometry.imuInitialized());
  assert(turn_aware_odometry.imuInitialized());
  const LidarOdometryResult fixed_turn_initial =
      fixed_turn_odometry.processScan(world_points, 22.0);
  const LidarOdometryResult turn_aware_initial =
      turn_aware_odometry.processScan(world_points, 22.0);
  assert(fixed_turn_initial.accepted);
  assert(turn_aware_initial.accepted);
  for (int index = 1; index <= 100; ++index)
  {
    ImuSample sample;
    sample.stamp = 22.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity =
        known_gyro_bias + Eigen::Vector3d(0.0, 0.0, 1.0);
    fixed_turn_odometry.addImuSample(sample);
    turn_aware_odometry.addImuSample(sample);
  }
  WheelSample turn_wheel;
  turn_wheel.stamp = 22.5;
  turn_wheel.forward_speed = 0.0;
  turn_wheel.differential_speed =
      1.0 / fixed_turn_options.wheel_yaw_rate_scale;
  turn_wheel.differential_disagreement = 0.0;
  fixed_turn_odometry.addWheelSample(turn_wheel);
  turn_aware_odometry.addWheelSample(turn_wheel);
  Eigen::Isometry3d sharp_turn_pose = turn_aware_initial.pose;
  sharp_turn_pose.linear() =
      turn_aware_initial.pose.rotation() *
      Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  PointVector sharp_turn_scan;
  sharp_turn_scan.reserve(world_points.size());
  for (const Eigen::Vector3d &world_point : world_points)
  {
    sharp_turn_scan.push_back(sharp_turn_pose.inverse() * world_point);
  }
  const LidarOdometryResult fixed_turn_result =
      fixed_turn_odometry.processScan(sharp_turn_scan, 22.5);
  const LidarOdometryResult turn_aware_result =
      turn_aware_odometry.processScan(sharp_turn_scan, 22.5);
  assert(!fixed_turn_result.accepted);
  assert(fixed_turn_result.reject_reason == "implausible_scan_motion");
  assert(turn_aware_result.accepted);
  assert(turn_aware_result.turn_aware_gate_active);
  assert(turn_aware_result.expected_rotation_deg > 25.0);
  assert(turn_aware_result.rotation_motion_gate_deg >
         turn_aware_result.expected_rotation_deg);
  assert(turn_aware_result.used_wheel_yaw_rate);
  assert(!turn_aware_result.wheel_yaw_rate_rejected);
  assert(turn_aware_result.wheel_yaw_rate_effective_noise > 0.20);
  assert(std::abs(turn_aware_result.wheel_yaw_rate_residual) < 0.02);
  assert(turn_aware_result.lidar_correction_rotation_gate_deg >= 4.2);

  // A slipping or delayed wheel sample must not pull gyro bias toward a
  // contradictory yaw rate. IMU propagation still permits the physical turn,
  // but the independent correction gate stays tight.
  for (int index = 1; index <= 100; ++index)
  {
    ImuSample sample;
    sample.stamp = 22.5 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity =
        known_gyro_bias + Eigen::Vector3d(0.0, 0.0, 1.0);
    turn_aware_odometry.addImuSample(sample);
  }
  WheelSample slipping_wheel = turn_wheel;
  slipping_wheel.stamp = 23.0;
  slipping_wheel.differential_speed = 0.0;
  turn_aware_odometry.addWheelSample(slipping_wheel);
  Eigen::Isometry3d second_turn_pose = sharp_turn_pose;
  second_turn_pose.linear() =
      sharp_turn_pose.rotation() *
      Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  PointVector second_turn_scan;
  second_turn_scan.reserve(world_points.size());
  for (const Eigen::Vector3d &world_point : world_points)
  {
    second_turn_scan.push_back(second_turn_pose.inverse() * world_point);
  }
  const LidarOdometryResult slipping_wheel_result =
      turn_aware_odometry.processScan(second_turn_scan, 23.0);
  assert(slipping_wheel_result.accepted);
  assert(slipping_wheel_result.turn_aware_gate_active);
  assert(!slipping_wheel_result.used_wheel_yaw_rate);
  assert(slipping_wheel_result.wheel_yaw_rate_rejected);
  assert(std::abs(slipping_wheel_result.wheel_yaw_rate_residual) > 0.9);
  assert(std::abs(slipping_wheel_result.lidar_correction_rotation_gate_deg -
                  turn_aware_options.max_lidar_correction_rotation_deg) <
         1e-9);

  // A robust wheel/gyro bias window may anchor bg_z on straight motion, while
  // a valid high-rate turn remains available for gating without entering the
  // bias normal equation.
  LidarOdometryOptions robust_bias_options = inertial_options;
  robust_bias_options.wheel_enabled = true;
  robust_bias_options.wheel_yaw_rate_scale = -1.0 / 0.43;
  robust_bias_options.wheel_yaw_rate_min_speed = 0.0;
  robust_bias_options.wheel_yaw_rate_max_imu_difference = 0.20;
  robust_bias_options.wheel_yaw_bias_window_sec = 1.0;
  robust_bias_options.wheel_yaw_bias_min_samples = 3;
  robust_bias_options.wheel_yaw_bias_max_abs_rate = 0.10;
  robust_bias_options.wheel_yaw_bias_max_mad = 0.01;
  robust_bias_options.wheel_yaw_bias_noise_floor = 0.01;
  robust_bias_options.wheel_yaw_bias_calibrate_offset = true;
  robust_bias_options.lidar_acceleration_bias_require_stable_wheel_motion =
      true;
  LidarOdometry robust_bias_odometry(robust_bias_options);
  for (int index = 0; index <= 400; ++index)
  {
    ImuSample sample;
    sample.stamp = 40.0 + 0.005 * static_cast<double>(index);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity = known_gyro_bias;
    robust_bias_odometry.addImuSample(sample);
  }
  assert(robust_bias_odometry.imuInitialized());
  assert(robust_bias_odometry.processScan(world_points, 42.0).accepted);
  LidarOdometryResult robust_bias_result;
  constexpr double fixed_wheel_yaw_offset = -0.01;
  for (int frame = 1; frame <= 3; ++frame)
  {
    for (int substep = 1; substep <= 20; ++substep)
    {
      ImuSample sample;
      sample.stamp = 42.0 + 0.1 * static_cast<double>(frame - 1) +
          0.005 * static_cast<double>(substep);
      sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
      sample.angular_velocity = known_gyro_bias;
      robust_bias_odometry.addImuSample(sample);
    }
    WheelSample straight_wheel;
    straight_wheel.stamp = 42.0 + 0.1 * static_cast<double>(frame);
    straight_wheel.forward_speed = 1.0;
    straight_wheel.differential_speed =
        fixed_wheel_yaw_offset / robust_bias_options.wheel_yaw_rate_scale;
    straight_wheel.differential_disagreement = 0.0;
    robust_bias_odometry.addWheelSample(straight_wheel);
    robust_bias_result = robust_bias_odometry.processScan(
        world_points, straight_wheel.stamp);
    assert(robust_bias_result.accepted);
  }
  assert(robust_bias_result.used_wheel_yaw_rate);
  assert(robust_bias_result.wheel_yaw_bias_window_samples == 3);
  if (std::abs(robust_bias_result.wheel_yaw_bias_observation -
               known_gyro_bias.z()) >= 1e-6)
  {
    std::cerr << "wheel bias calibration mismatch: corrected="
              << robust_bias_result.wheel_yaw_bias_observation
              << " raw="
              << robust_bias_result.wheel_yaw_bias_raw_observation
              << " offset=" << robust_bias_result.wheel_yaw_bias_offset
              << " expected=" << known_gyro_bias.z() << '\n';
  }
  assert(std::abs(robust_bias_result.wheel_yaw_bias_observation -
                  known_gyro_bias.z()) < 1e-6);
  assert(std::abs(robust_bias_result.wheel_yaw_bias_raw_observation -
                  (known_gyro_bias.z() - fixed_wheel_yaw_offset)) < 1e-6);
  assert(std::abs(robust_bias_result.wheel_yaw_bias_offset +
                  fixed_wheel_yaw_offset) < 1e-6);
  assert(robust_bias_result.wheel_yaw_bias_offset_calibrated);
  assert(robust_bias_result.wheel_yaw_bias_mad < 1e-9);
  assert(robust_bias_result.acceleration_bias_update_allowed);

  for (int substep = 1; substep <= 20; ++substep)
  {
    ImuSample sample;
    sample.stamp = 42.3 + 0.005 * static_cast<double>(substep);
    sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    sample.angular_velocity =
        known_gyro_bias + Eigen::Vector3d(0.0, 0.0, 0.5);
    robust_bias_odometry.addImuSample(sample);
  }
  WheelSample curved_wheel;
  curved_wheel.stamp = 42.4;
  curved_wheel.forward_speed = 1.0;
  curved_wheel.differential_speed =
      0.5 / robust_bias_options.wheel_yaw_rate_scale;
  curved_wheel.differential_disagreement = 0.0;
  robust_bias_odometry.addWheelSample(curved_wheel);
  const LidarOdometryResult curved_bias_result =
      robust_bias_odometry.processScan(world_points, curved_wheel.stamp);
  assert(!curved_bias_result.used_wheel_yaw_rate);
  assert(curved_bias_result.wheel_yaw_bias_window_samples == 3);
  assert(!curved_bias_result.acceleration_bias_update_allowed);

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
  loss_options.wheel_enabled = true;
  loss_options.lidar_loss_use_wheel_dead_reckoning = true;
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

  // On a moving platform, freezing at the last accepted pose makes recovery
  // impossible because every subsequent scan is projected into stale map
  // coordinates. With synchronized wheel speed, keep the IMU turn and bounded
  // forward motion alive while still rejecting the scan and withholding map
  // insertion.
  const Eigen::Isometry3d wheel_loss_reference = loss_result.pose;
  for (int frame = 1; frame <= 8; ++frame)
  {
    for (int substep = 1; substep <= 20; ++substep)
    {
      ImuSample sample;
      sample.stamp = 33.0 + 0.1 * static_cast<double>(frame - 1) +
          0.005 * static_cast<double>(substep);
      sample.acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
      sample.angular_velocity =
          known_gyro_bias + Eigen::Vector3d(0.0, 0.0, 0.5);
      loss_odometry.addImuSample(sample);
    }
    WheelSample loss_wheel;
    loss_wheel.stamp = 33.0 + 0.1 * static_cast<double>(frame);
    loss_wheel.forward_speed = 1.0;
    loss_odometry.addWheelSample(loss_wheel);
    loss_result = loss_odometry.processScan(
        unrelated_scan, loss_wheel.stamp);
    assert(!loss_result.accepted);
  }
  const Eigen::Isometry3d wheel_loss_delta =
      wheel_loss_reference.inverse() * loss_result.pose;
  assert(loss_result.loss_limited);
  assert(!loss_result.loss_frozen);
  assert(loss_result.consecutive_rejections >= 5);
  assert(wheel_loss_delta.translation().head<2>().norm() > 0.30);
  assert(wheel_loss_delta.translation().head<2>().norm() < 0.90);
  assert(logSE3(wheel_loss_delta).head<3>().norm() * 180.0 /
         std::acos(-1.0) > 10.0);
  assert(!loss_result.map_updated);

  std::cout << "hybrid_localization_core_smoke_test: PASS\n";
  return 0;
}
