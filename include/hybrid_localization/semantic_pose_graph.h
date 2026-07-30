#ifndef HYBRID_LOCALIZATION_SEMANTIC_POSE_GRAPH_H
#define HYBRID_LOCALIZATION_SEMANTIC_POSE_GRAPH_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hybrid_localization
{

struct SemanticGraphPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  uint8_t label = 0U;
  float confidence = 1.0F;
};

using SemanticGraphPointVector =
    std::vector<SemanticGraphPoint, Eigen::aligned_allocator<SemanticGraphPoint>>;

struct SemanticPoseGraphOptions
{
  bool enabled = true;
  // Generic geometric loop proposals are opt-in.  Semantic observations have
  // their own stricter multi-frame gates and remain enabled independently.
  bool enable_xy_loops = false;
  bool enable_z_loops = false;
  bool enable_sequential_ground_z = false;
  bool enable_wheel_factors = false;
  bool enable_visual_rotation_factors = false;
  bool enable_visual_translation_factors = false;
  bool enable_visual_loop_factors = false;
  bool enable_semantic_observation_factors = true;
  bool enable_semantic_observation_xy_factors = true;
  bool enable_semantic_observation_z_factors = true;
  // A vertical correspondence is meaningful only after the planar semantic
  // registration has established its horizontal association.
  bool semantic_observation_require_xy_for_z = true;
  bool use_semantics = true;

  double keyframe_distance = 1.0;
  double keyframe_yaw_deg = 8.0;
  double keyframe_interval_sec = 1.0;
  int max_keyframes = 2500;
  int submap_frames = 8;
  int max_points_per_frame = 7000;
  int max_features_per_keyframe = 3500;

  double feature_resolution = 0.60;
  int feature_min_points = 3;
  double structural_min_height_span = 0.55;
  double structural_min_z = -0.35;
  double ground_max_height = -0.20;
  double ground_max_height_span = 0.30;
  int ground_min_points = 4;

  int descriptor_rings = 20;
  int descriptor_sectors = 60;
  double descriptor_max_radius = 55.0;
  double descriptor_min_similarity = 0.22;
  double descriptor_min_score_gap = 0.015;
  double semantic_weight = 0.35;

  // Asynchronous semantic observations are associated to existing graph keys
  // by sensor timestamp, then grouped into non-overlapping multi-frame BEV
  // submaps. Each observation contributes to exactly one submap.
  int semantic_submap_observations = 3;
  int semantic_observation_min_index_gap = 4;
  int semantic_observation_max_index_gap = 30;
  // Limit repeated use of a semantic reference submap so one locally biased
  // projection cannot accumulate into many correlated graph factors. Zero
  // leaves reuse unlimited.
  int semantic_observation_max_reference_uses = 0;
  int semantic_observation_interval = 1;
  int semantic_observation_min_features = 80;
  int semantic_observation_min_inliers = 45;
  int semantic_observation_min_z_inliers = 30;
  int min_semantic_observation_factors_for_xy_output = 2;
  double semantic_observation_search_radius = 35.0;
  double semantic_observation_min_baseline = 0.0;
  double semantic_observation_max_time_offset = 0.75;
  double semantic_observation_correspondence_distance = 0.85;
  double semantic_observation_ransac_inlier_distance = 0.38;
  double semantic_observation_min_inlier_ratio = 0.35;
  double semantic_observation_min_spread = 5.0;
  double semantic_observation_min_spread_ratio = 0.08;
  double semantic_observation_max_rmse = 0.32;
  double semantic_observation_max_xy_correction = 0.50;
  double semantic_observation_max_yaw_correction_deg = 1.5;
  double semantic_observation_max_z_correction = 0.35;
  double semantic_observation_sigma_xy = 0.30;
  double semantic_observation_sigma_yaw_deg = 1.00;
  double semantic_observation_sigma_z = 0.25;
  double semantic_observation_huber_k = 1.345;

  int loop_min_index_gap = 30;
  int loop_max_candidates = 6;
  double loop_search_radius = 18.0;
  double loop_max_yaw_difference_deg = 35.0;
  double coarse_xy_radius = 5.0;
  double coarse_xy_step = 0.75;
  double coarse_yaw_radius_deg = 8.0;
  double coarse_yaw_step_deg = 2.0;
  int coarse_max_points = 900;
  int coarse_min_inliers = 70;

  double correspondence_distance = 1.20;
  double ransac_inlier_distance = 0.55;
  int ransac_iterations = 120;
  int min_xy_inliers = 90;
  double min_xy_inlier_ratio = 0.30;
  double min_xy_spread = 7.0;
  double min_xy_spread_ratio = 0.12;
  double max_xy_rmse = 0.42;
  double huber_delta = 0.35;
  int huber_iterations = 5;
  double max_xy_correction = 5.0;
  double max_yaw_correction_deg = 8.0;

  double z_correspondence_distance = 0.80;
  double z_candidate_residual_gate = 1.20;
  double z_inlier_residual_gate = 0.22;
  int min_z_inliers = 45;
  double max_z_mad = 0.12;
  double max_z_correction = 1.20;

  double graph_consistency_max_xy = 5.0;
  double graph_consistency_max_yaw_deg = 8.0;
  double graph_consistency_max_z = 1.5;
  int min_loops_for_xy_output = 2;

  double odom_sigma_roll_pitch = 0.010;
  double odom_sigma_yaw = 0.008;
  double odom_sigma_xy_base = 0.035;
  double odom_sigma_xy_per_meter = 0.010;
  double odom_sigma_z_base = 0.050;
  double loop_sigma_xy = 0.12;
  double loop_sigma_yaw_deg = 0.45;
  double loop_sigma_z = 0.08;
  double sequential_ground_sigma_z = 0.06;
  double loop_huber_k = 1.345;
  double sequential_ground_huber_k = 1.345;

  double wheel_speed_scale = 0.9865;
  double wheel_max_gap = 0.08;
  double wheel_sigma_base = 0.08;
  double wheel_sigma_per_meter = 0.025;
  double wheel_lateral_sigma = 0.15;
  double wheel_huber_k = 1.345;
  int wheel_min_samples = 5;

  double visual_max_time_offset = 0.15;
  double visual_min_quality = 0.30;
  double visual_max_angular_disagreement_deg = 4.0;
  double visual_sigma_roll_pitch_deg = 0.80;
  double visual_sigma_yaw_deg = 0.50;
  double visual_quality_sigma_scale = 1.5;
  double visual_huber_k = 1.345;
  double visual_max_translation_disagreement = 1.0;
  double visual_sigma_xy_base = 0.10;
  double visual_sigma_z_base = 0.18;
  double visual_sigma_translation_per_meter = 0.03;

  double visual_loop_max_time_offset = 0.65;
  int visual_loop_min_index_gap = 20;
  double visual_loop_min_quality = 0.40;
  double visual_loop_max_translation_disagreement = 3.0;
  double visual_loop_max_rotation_disagreement_deg = 10.0;
  double visual_loop_sigma_roll_pitch_deg = 2.0;
  double visual_loop_sigma_yaw_deg = 1.0;
  double visual_loop_sigma_xy = 0.25;
  double visual_loop_sigma_z = 0.35;
  double visual_loop_quality_sigma_scale = 1.5;
  double visual_loop_huber_k = 1.345;

  double isam_relinearize_threshold = 0.05;
  int isam_relinearize_skip = 1;
};

struct GraphTrajectorySample
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stamp = 0.0;
  Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d optimized_pose = Eigen::Isometry3d::Identity();
};

enum class DebugPairStage
{
  Candidate,
  Inlier,
  Outlier,
  Applied
};

struct GraphDebugPair
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d source_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_world = Eigen::Vector3d::Zero();
  DebugPairStage stage = DebugPairStage::Candidate;
  double residual = 0.0;
};

using GraphDebugPairVector =
    std::vector<GraphDebugPair, Eigen::aligned_allocator<GraphDebugPair>>;

struct SemanticLoopDebug
{
  bool valid = false;
  bool accepted = false;
  bool xy_accepted = false;
  bool z_accepted = false;
  int reference_id = -1;
  int current_id = -1;
  double descriptor_similarity = 0.0;
  double descriptor_score_gap = 0.0;
  double xy_rmse = 0.0;
  double z_median = 0.0;
  double z_mad = 0.0;
  std::string reason = "not_attempted";
  GraphDebugPairVector xy_pairs;
  GraphDebugPairVector z_pairs;
};

struct SemanticPoseGraphStats
{
  int keyframes = 0;
  int odometry_factors = 0;
  int sequential_ground_factors = 0;
  int wheel_factors = 0;
  int visual_rotation_factors = 0;
  int visual_translation_factors = 0;
  int visual_rotation_rejections = 0;
  int visual_loop_attempts = 0;
  int visual_loop_rejections = 0;
  int visual_loop_factors = 0;
  int loop_attempts = 0;
  int loop_rejections = 0;
  int loop_factors = 0;
  int xy_loop_factors = 0;
  int z_loop_factors = 0;
  int semantic_keyframes = 0;
  int semantic_observations_received = 0;
  int semantic_observations_associated = 0;
  int semantic_observation_attempts = 0;
  int semantic_observation_skips = 0;
  int semantic_observation_rejections = 0;
  int semantic_observation_reference_rejections = 0;
  int semantic_observation_factors = 0;
  int semantic_observation_xy_factors = 0;
  int semantic_observation_z_factors = 0;
  int semantic_observation_inliers = 0;
  int semantic_pending_observations = 0;
  int last_semantic_xy_candidates = 0;
  int last_semantic_xy_inliers = 0;
  int last_semantic_z_candidates = 0;
  int last_semantic_z_inliers = 0;
  double last_semantic_xy_inlier_ratio = 0.0;
  double last_semantic_xy_rmse = 0.0;
  double last_semantic_xy_spread = 0.0;
  double last_semantic_xy_spread_ratio = 0.0;
  double last_semantic_baseline = 0.0;
  double last_semantic_xy_correction = 0.0;
  double last_semantic_yaw_correction_deg = 0.0;
  double last_semantic_z_median = 0.0;
  double last_semantic_z_mad = 0.0;
  double last_optimization_ms = 0.0;
};

class SemanticPoseGraph
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit SemanticPoseGraph(const SemanticPoseGraphOptions &options = SemanticPoseGraphOptions());
  ~SemanticPoseGraph();

  SemanticPoseGraph(const SemanticPoseGraph &) = delete;
  SemanticPoseGraph &operator=(const SemanticPoseGraph &) = delete;

  void reset();
  void addWheelSample(double stamp, double forward_speed);
  void addVisualRotationSample(double stamp,
                               const Eigen::Matrix3d &visual_from_body_rotation,
                               int segment, double quality);
  void addVisualPoseSample(double stamp,
                           const Eigen::Isometry3d &visual_from_body_pose,
                           int segment, double quality);
  void addOdometrySample(double stamp, const Eigen::Isometry3d &raw_pose);
  bool addFrame(double stamp, const Eigen::Isometry3d &raw_pose,
                const SemanticGraphPointVector &local_points);
  bool addSemanticObservation(double stamp, const Eigen::Isometry3d &raw_pose,
                              const SemanticGraphPointVector &local_points);
  bool addVisualLoopConstraint(double reference_stamp, double current_stamp,
                               const Eigen::Isometry3d &reference_from_current,
                               double quality);

  bool initialized() const;
  Eigen::Isometry3d correctedPose(const Eigen::Isometry3d &raw_pose) const;
  std::vector<GraphTrajectorySample, Eigen::aligned_allocator<GraphTrajectorySample>>
  optimizedTrajectory() const;
  bool saveOptimizedTrajectory(const std::string &path) const;
  SemanticGraphPointVector semanticMap(double voxel_size = 0.30,
                                       int maximum_points = 120000) const;

  const SemanticLoopDebug &lastDebug() const;
  const SemanticLoopDebug &lastSemanticDebug() const;
  SemanticPoseGraphStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_SEMANTIC_POSE_GRAPH_H
