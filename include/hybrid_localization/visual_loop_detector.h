#ifndef HYBRID_LOCALIZATION_VISUAL_LOOP_DETECTOR_H
#define HYBRID_LOCALIZATION_VISUAL_LOOP_DETECTOR_H

#include "hybrid_localization/visual_rotation_tracker.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace hybrid_localization
{

struct VisualLoopDetectorOptions
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
  int maximum_features = 1800;
  int minimum_depth_features = 80;
  int maximum_database_size = 1000;
  double keyframe_distance = 0.75;
  double keyframe_interval_sec = 1.0;
  int minimum_index_gap = 25;
  double search_radius = 25.0;
  double maximum_yaw_difference_deg = 70.0;
  int maximum_retrieval_candidates = 40;
  int maximum_geometric_candidates = 5;
  double descriptor_ratio = 0.75;
  int minimum_descriptor_matches = 55;
  int depth_association_radius_pixels = 5;
  double minimum_depth = 1.0;
  double maximum_depth = 80.0;
  int pnp_iterations = 600;
  double pnp_reprojection_error = 2.5;
  int minimum_pnp_inliers = 40;
  double minimum_pnp_inlier_ratio = 0.35;
  int grid_rows = 4;
  int grid_cols = 6;
  int minimum_occupied_cells = 8;
  double maximum_reprojection_rmse = 2.0;
  double maximum_translation_disagreement = 3.0;
  double maximum_rotation_disagreement_deg = 10.0;
  double minimum_quality = 0.40;
};

struct VisualLoopResult
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool keyframe_created = false;
  bool candidate_found = false;
  bool accepted = false;
  int reference_id = -1;
  int current_id = -1;
  double reference_stamp = 0.0;
  double current_stamp = 0.0;
  int descriptor_matches = 0;
  int pnp_inliers = 0;
  int occupied_cells = 0;
  double descriptor_score = 0.0;
  double inlier_ratio = 0.0;
  double reprojection_rmse = std::numeric_limits<double>::infinity();
  double translation_disagreement = std::numeric_limits<double>::infinity();
  double rotation_disagreement_deg = std::numeric_limits<double>::infinity();
  double quality = 0.0;
  Eigen::Isometry3d reference_from_current = Eigen::Isometry3d::Identity();
  cv::Mat debug_image;
  std::string reason = "not_processed";
};

class VisualLoopDetector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit VisualLoopDetector(
      const VisualLoopDetectorOptions &options = VisualLoopDetectorOptions());

  void reset();
  VisualLoopResult process(double stamp, const cv::Mat &image,
                           const VisualLidarPointVector &body_points,
                           const Eigen::Isometry3d &world_from_body,
                           const cv::Mat &dynamic_mask = cv::Mat());
  std::size_t keyframeCount() const { return keyframes_.size(); }

private:
  struct Keyframe
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id = -1;
    double stamp = 0.0;
    Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
    cv::Mat gray;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    VisualLidarPointVector depth_points;
    std::vector<uint8_t> has_depth;
  };

  struct RetrievalCandidate
  {
    std::size_t database_index = 0U;
    int matches = 0;
    double score = 0.0;
    std::vector<cv::DMatch> correspondences;
  };

  cv::Mat prepareImage(const cv::Mat &image) const;
  cv::Mat featureMask(const cv::Mat &dynamic_mask,
                      const cv::Size &size) const;
  Keyframe buildKeyframe(double stamp, const cv::Mat &gray,
                         const VisualLidarPointVector &body_points,
                         const Eigen::Isometry3d &raw_pose,
                         const cv::Mat &mask);
  RetrievalCandidate retrieve(const Keyframe &reference,
                              const Keyframe &current,
                              std::size_t database_index) const;
  VisualLoopResult verify(const Keyframe &reference,
                          const Keyframe &current,
                          const RetrievalCandidate &candidate) const;
  void associateDepth(Keyframe *keyframe,
                      const VisualLidarPointVector &body_points) const;

  VisualLoopDetectorOptions options_;
  cv::Ptr<cv::ORB> orb_;
  std::deque<Keyframe, Eigen::aligned_allocator<Keyframe>> keyframes_;
  int next_id_ = 0;
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_VISUAL_LOOP_DETECTOR_H
