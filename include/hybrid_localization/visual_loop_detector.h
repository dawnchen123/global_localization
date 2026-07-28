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
#include <unordered_map>
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
  // Keep enough feature/depth keyframes for a complete long replay. Gray
  // images are retained separately only for recent debug rendering.
  int maximum_database_size = 1600;
  int debug_image_history_size = 96;
  // A loop keyframe is admitted only after both thresholds are met. This
  // bounds retained history for long runs while suppressing stationary views.
  double keyframe_distance = 0.75;
  double keyframe_interval_sec = 1.0;
  // Feature/depth keyframes are retained at the cadence above, while the
  // expensive database retrieval and PnP verification run at this lower rate.
  // This keeps the loop worker from starving the LiDAR frontend on long runs.
  double retrieval_interval_sec = 4.0;
  int minimum_index_gap = 25;
  // Index gaps vary with image cadence; keep a true revisit time gate as well
  // so short local PnP matches cannot consume the loop-factor budget.
  double minimum_time_separation_sec = 45.0;
  double search_radius = 25.0;
  double maximum_yaw_difference_deg = 70.0;
  int maximum_retrieval_candidates = 40;
  int maximum_geometric_candidates = 5;
  // The spatial gate above is efficient while raw odometry remains close to
  // the map, but it can hide a genuine revisit after long-term drift. The
  // optional fallback uses a multi-table ORB locality-sensitive inverted
  // index: high-response local features vote for old keyframes before the
  // normal descriptor match/PnP/LiDAR verification chain runs.
  bool enable_global_retrieval_fallback = false;
  int maximum_global_retrieval_candidates = 0;
  int global_retrieval_feature_count = 450;
  int global_retrieval_min_votes = 10;
  int global_retrieval_min_table_count = 2;
  // Reserve this many globally retrieved candidates for geometric verification
  // when they are available.  Zero retains pure score ordering.
  int minimum_global_geometric_candidates = 0;
  // Keep several independently verified proposals from one retrieval cycle.
  // The graph consumes at most one while a multi-frame support hypothesis is
  // active, and otherwise can fall back when the top-scoring reference is in
  // cooldown or fails an independent LiDAR gate.
  int maximum_verified_candidates = 1;
  // Candidate references closer than this temporal separation are treated as
  // one place. This prevents a dense run of adjacent images from becoming a
  // correlated set of graph alternatives.
  double candidate_reference_min_separation_sec = 0.0;
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
  // Counts for the current retrieval cycle.  They make it possible to tell
  // whether a long-return candidate reached the expensive PnP stage.
  int global_retrieval_candidates = 0;
  int global_retrieval_descriptor_matches = 0;
  bool global_retrieval = false;
  double descriptor_score = 0.0;
  int global_retrieval_votes = 0;
  int global_retrieval_tables = 0;
  double inlier_ratio = 0.0;
  double reprojection_rmse = std::numeric_limits<double>::infinity();
  double translation_disagreement = std::numeric_limits<double>::infinity();
  double rotation_disagreement_deg = std::numeric_limits<double>::infinity();
  double temporal_separation_sec = 0.0;
  double raw_xy_separation = 0.0;
  double raw_z_separation = 0.0;
  double quality = 0.0;
  Eigen::Isometry3d reference_from_current = Eigen::Isometry3d::Identity();
  cv::Mat debug_image;
  std::string reason = "not_processed";
};

using VisualLoopResultVector =
    std::vector<VisualLoopResult, Eigen::aligned_allocator<VisualLoopResult>>;

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
  // Results are ordered by verification quality and de-correlated by the
  // configured reference separation. process() still returns the best entry
  // for compatibility with existing callers.
  const VisualLoopResultVector &lastAcceptedCandidates() const
  {
    return last_accepted_candidates_;
  }

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
    // Only the strongest features enter the global inverted index. The full
    // descriptor matrix remains available for one-to-one geometric matching.
    std::vector<int> global_retrieval_feature_indices;
  };

  struct RetrievalCandidate
  {
    std::size_t database_index = 0U;
    int matches = 0;
    double score = 0.0;
    bool global_retrieval = false;
    int global_retrieval_votes = 0;
    int global_retrieval_tables = 0;
    std::vector<cv::DMatch> correspondences;
  };

  struct GlobalVoteCandidate
  {
    std::size_t database_index = 0U;
    int votes = 0;
    int tables = 0;
  };

  static constexpr int kGlobalRetrievalHashTables = 6;
  static constexpr int kGlobalRetrievalHashBits = 17;
  using GlobalVoteBins = std::vector<std::vector<int>>;
  using GlobalVoteIndex = std::array<GlobalVoteBins, kGlobalRetrievalHashTables>;

  cv::Mat prepareImage(const cv::Mat &image) const;
  cv::Mat featureMask(const cv::Mat &dynamic_mask,
                      const cv::Size &size) const;
  Keyframe buildKeyframe(double stamp, const cv::Mat &gray,
                         const VisualLidarPointVector &body_points,
                         const Eigen::Isometry3d &raw_pose,
                         const cv::Mat &mask);
  void selectGlobalRetrievalFeatures(Keyframe *keyframe) const;
  std::uint32_t globalRetrievalHash(const uint8_t *descriptor,
                                    int table) const;
  void addKeyframeToGlobalIndex(const Keyframe &keyframe);
  void rebuildGlobalIndex();
  std::vector<GlobalVoteCandidate> globalVoteCandidates(
      const Keyframe &current) const;
  std::size_t databaseIndexForKeyframeId(int id) const;
  RetrievalCandidate retrieve(const Keyframe &reference,
                              const Keyframe &current,
                              std::size_t database_index,
                              bool global_retrieval = false,
                              int global_retrieval_votes = 0,
                              int global_retrieval_tables = 0) const;
  VisualLoopResult verify(const Keyframe &reference,
                          const Keyframe &current,
                          const RetrievalCandidate &candidate) const;
  void associateDepth(Keyframe *keyframe,
                      const VisualLidarPointVector &body_points) const;
  void pruneDebugImages();

  VisualLoopDetectorOptions options_;
  cv::Ptr<cv::ORB> orb_;
  std::deque<Keyframe, Eigen::aligned_allocator<Keyframe>> keyframes_;
  GlobalVoteIndex global_vote_index_;
  int global_index_stale_keyframes_ = 0;
  VisualLoopResultVector last_accepted_candidates_;
  int next_id_ = 0;
  double last_retrieval_stamp_ = -std::numeric_limits<double>::infinity();
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_VISUAL_LOOP_DETECTOR_H
