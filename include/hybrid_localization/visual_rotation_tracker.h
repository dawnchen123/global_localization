#ifndef HYBRID_LOCALIZATION_VISUAL_ROTATION_TRACKER_H
#define HYBRID_LOCALIZATION_VISUAL_ROTATION_TRACKER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <opencv2/core.hpp>

#include <array>
#include <limits>
#include <string>
#include <vector>

namespace hybrid_localization
{

struct VisualRotationTrackerOptions
{
  double fx = 1064.8950;
  double fy = 1065.2546;
  double cx = 801.4049;
  double cy = 624.6878;
  std::array<double, 5> distortion{{-0.1516, 0.0942, 0.000169, -0.000142, -0.0229}};
  Eigen::Matrix3d body_from_camera_rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d body_from_camera_translation = Eigen::Vector3d::Zero();

  double image_scale = 0.5;
  double minimum_interval = 0.15;
  double maximum_reference_gap = 0.60;
  int maximum_features = 1400;
  int minimum_tracks = 90;
  int minimum_inliers = 65;
  double feature_quality = 0.01;
  double feature_minimum_distance = 10.0;
  double forward_backward_error = 1.2;
  double minimum_median_parallax = 1.2;
  double ransac_probability = 0.999;
  double ransac_threshold_pixels = 1.5;
  double minimum_inlier_ratio = 0.50;
  int grid_rows = 4;
  int grid_cols = 6;
  int minimum_occupied_cells = 8;
  double maximum_rotation_rate_deg = 80.0;
  double maximum_rotation_step_deg = 12.0;
  bool equalize_histogram = true;
  bool generate_debug_images = true;

  bool enable_lidar_pnp = false;
  bool require_lidar_pnp = true;
  int pnp_association_radius_pixels = 5;
  int pnp_minimum_correspondences = 45;
  int pnp_minimum_inliers = 40;
  int pnp_minimum_occupied_cells = 6;
  double pnp_minimum_inlier_ratio = 0.25;
  double pnp_ransac_reprojection_error = 2.0;
  int pnp_ransac_iterations = 600;
  double pnp_maximum_reprojection_rmse = 1.8;
  double pnp_maximum_local_depth_difference = 1.25;
  double pnp_maximum_local_depth_ratio = 0.08;
  double pnp_minimum_depth = 1.0;
  double pnp_maximum_depth = 80.0;
  double pnp_maximum_translation_speed = 15.0;
  double pnp_maximum_translation_step = 5.0;
};

using VisualLidarPointVector =
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

struct VisualRotationEstimate
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool observation_valid = false;
  bool motion_valid = false;
  bool metric_pose_valid = false;
  double stamp = 0.0;
  int segment = 0;
  int tracks = 0;
  int inliers = 0;
  int occupied_cells = 0;
  int pnp_correspondences = 0;
  int pnp_inliers = 0;
  int pnp_occupied_cells = 0;
  double inlier_ratio = 0.0;
  double median_parallax = 0.0;
  double rotation_deg = 0.0;
  double quality = 0.0;
  double pnp_inlier_ratio = 0.0;
  double pnp_reprojection_rmse = 0.0;
  double image_cloud_time_difference = std::numeric_limits<double>::quiet_NaN();
  Eigen::Matrix3d relative_body_rotation = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d visual_from_body_rotation = Eigen::Matrix3d::Identity();
  Eigen::Isometry3d relative_body_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d visual_from_body_pose = Eigen::Isometry3d::Identity();
  cv::Mat projection_debug_image;
  cv::Mat tracking_debug_image;
  cv::Mat pnp_debug_image;
  std::string reason = "not_processed";
};

class VisualRotationTracker
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit VisualRotationTracker(
      const VisualRotationTrackerOptions &options = VisualRotationTrackerOptions());

  void reset();
  VisualRotationEstimate process(double stamp, const cv::Mat &image);
  VisualRotationEstimate process(double stamp, const cv::Mat &image,
                                 const VisualLidarPointVector &body_points);

private:
  cv::Mat prepareImage(const cv::Mat &image) const;
  cv::Mat projectionDebugImage(const cv::Mat &gray,
                               const VisualLidarPointVector &body_points) const;
  VisualRotationEstimate initializeReference(double stamp, const cv::Mat &gray,
                                             const VisualLidarPointVector &body_points,
                                             const std::string &reason,
                                             bool new_segment);

  VisualRotationTrackerOptions options_;
  cv::Mat reference_gray_;
  VisualLidarPointVector reference_body_points_;
  double reference_stamp_ = 0.0;
  bool initialized_ = false;
  int segment_ = 0;
  Eigen::Matrix3d visual_from_body_rotation_ = Eigen::Matrix3d::Identity();
  Eigen::Isometry3d visual_from_body_pose_ = Eigen::Isometry3d::Identity();
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_VISUAL_ROTATION_TRACKER_H
