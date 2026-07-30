#ifndef HYBRID_LOCALIZATION_SPARSE_VISUAL_MAP_H
#define HYBRID_LOCALIZATION_SPARSE_VISUAL_MAP_H

#include "hybrid_localization/lidar_odometry.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace hybrid_localization
{

struct SparseVisualMapOptions
{
  bool enabled = false;
  double fx = 1064.8950;
  double fy = 1065.2546;
  double cx = 801.4049;
  double cy = 624.6878;
  std::array<double, 5> distortion{{-0.1516, 0.0942, 0.000169,
                                    -0.000142, -0.0229}};
  Eigen::Matrix3d body_from_camera_rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d body_from_camera_translation = Eigen::Vector3d::Zero();
  double image_scale = 0.5;
  int patch_half_size = 3;
  int grid_size_pixels = 24;
  int max_landmarks = 1400;
  int max_active_landmarks = 450;
  int max_lidar_candidates = 9000;
  int max_new_landmarks_per_frame = 180;
  int max_missed_frames = 20;
  int reference_refresh_observations = 8;
  double minimum_depth = 1.0;
  double maximum_depth = 80.0;
  double minimum_gradient = 12.0;
  double minimum_patch_stddev = 6.0;
  double minimum_ncc = 0.72;
  double photometric_huber_delta = 1.5;
  double photometric_noise = 1.0;
  double information_scale = 0.04;
  double landmark_voxel_size = 0.20;
  double local_map_radius = 65.0;
  std::array<float, 7> semantic_class_weights{{1.0F, 1.3F, 1.5F, 0.9F,
                                                0.75F, 0.0F, 0.6F}};
};

struct SparseVisualFrame
{
  double stamp = 0.0;
  cv::Mat gray;
  cv::Mat gradient_x;
  cv::Mat gradient_y;
  cv::Mat dynamic_mask;
  cv::Mat semantic_labels;

  bool valid() const
  {
    return std::isfinite(stamp) && !gray.empty() && gray.type() == CV_32F &&
           gradient_x.size() == gray.size() && gradient_y.size() == gray.size();
  }
};

struct SparseVisualMapStats
{
  std::uint64_t frames = 0U;
  std::uint64_t accepted_updates = 0U;
  std::uint64_t rejected_updates = 0U;
  std::uint64_t seeded_landmarks = 0U;
  std::uint64_t culled_landmarks = 0U;
  std::uint64_t dynamic_rejections = 0U;
  int landmarks = 0;
  int last_visible = 0;
  int last_inliers = 0;
  int last_residuals = 0;
  double last_rmse = std::numeric_limits<double>::infinity();
  double last_mean_ncc = 0.0;
  std::string last_reason = "not_processed";
};

class SparseVisualMap
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit SparseVisualMap(
      const SparseVisualMapOptions &options = SparseVisualMapOptions());

  void reset();
  SparseVisualFrame prepareFrame(double stamp, const cv::Mat &image,
                                 const cv::Mat &dynamic_mask = cv::Mat(),
                                 const cv::Mat &semantic_labels = cv::Mat()) const;
  void addLidarFrame(double stamp, const Eigen::Isometry3d &world_from_body,
                     const PointVector &body_points);
  void addLidarFrame(double stamp, const Eigen::Isometry3d &world_from_body,
                     const PointVector &body_points,
                     const std::vector<uint8_t> &labels,
                     const std::vector<float> &semantic_weights);
  VisualPoseLinearization linearize(const SparseVisualFrame &frame,
                                    const Eigen::Isometry3d &world_from_body);
  void commitFrame(const SparseVisualFrame &frame,
                   const Eigen::Isometry3d &world_from_body,
                   bool visual_update_accepted);

  std::size_t landmarkCount() const { return landmarks_.size(); }
  const SparseVisualMapStats &stats() const { return stats_; }
  const cv::Mat &debugImage() const { return debug_image_; }

private:
  struct Candidate
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    uint8_t label = 0U;
    float semantic_weight = 1.0F;
  };

  struct Landmark
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id = -1;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    std::vector<float> patch;
    uint8_t label = 0U;
    float semantic_weight = 1.0F;
    int observations = 0;
    int missed_frames = 0;
    double last_seen_stamp = 0.0;
  };

  using CandidateVector =
      std::vector<Candidate, Eigen::aligned_allocator<Candidate>>;
  using LandmarkVector =
      std::vector<Landmark, Eigen::aligned_allocator<Landmark>>;

  bool project(const Eigen::Vector3d &world_point,
               const Eigen::Isometry3d &world_from_body,
               cv::Point2f *pixel, Eigen::Vector3d *camera_point,
               Eigen::Vector3d *body_point = nullptr) const;
  bool normalizedPatch(const cv::Mat &image, const cv::Point2f &center,
                       std::vector<float> *patch, double *stddev = nullptr) const;
  static float bilinear(const cv::Mat &image, float x, float y);
  bool masked(const SparseVisualFrame &frame, const cv::Point2f &pixel) const;
  uint8_t semanticLabel(const SparseVisualFrame &frame,
                        const cv::Point2f &pixel) const;
  float semanticClassWeight(uint8_t label) const;
  void seedLandmarks(const SparseVisualFrame &frame,
                     const Eigen::Isometry3d &world_from_body);
  void cullLandmarks(const Eigen::Vector3d &position);

  SparseVisualMapOptions options_;
  CandidateVector candidates_;
  LandmarkVector landmarks_;
  int next_landmark_id_ = 0;
  std::vector<int> last_inlier_ids_;
  cv::Mat debug_image_;
  SparseVisualMapStats stats_;
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_SPARSE_VISUAL_MAP_H
