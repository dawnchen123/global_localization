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
  const Eigen::Matrix<double, 6, 1> registration_error =
      logSE3(expected_pose.inverse() * second_scan.pose);
  assert(registration_error.head<3>().norm() < 0.03);
  assert(registration_error.tail<3>().norm() < 0.10);

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

  const Eigen::Vector3d visual_target = second_scan.pose.translation() +
      Eigen::Vector3d(0.01, -0.005, 0.002);
  const double visual_error_before =
      (second_scan.pose.translation() - visual_target).norm();
  const VisualUpdateResult visual_update = lidar_odometry.processVisual(
      1.1, [&visual_target](const Eigen::Isometry3d &pose)
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

  PointVector unrelated_scan;
  unrelated_scan.reserve(world_points.size());
  for (const Eigen::Vector3d &world_point : world_points)
  {
    unrelated_scan.push_back(world_point + Eigen::Vector3d(100.0, 0.0, 0.0));
  }
  const Eigen::Isometry3d pose_before_rejection = lidar_odometry.pose();
  const std::size_t map_size_before_rejection = lidar_odometry.mapPointCount();
  const LidarOdometryResult rejected_scan = lidar_odometry.processScan(unrelated_scan, 1.2);
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

  std::cout << "hybrid_localization_core_smoke_test: PASS\n";
  return 0;
}
