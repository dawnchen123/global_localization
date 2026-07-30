#ifndef HYBRID_LOCALIZATION_CORE_H
#define HYBRID_LOCALIZATION_CORE_H

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace hybrid_localization
{

struct PoseState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  double stamp = 0.0;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 15, 15> covariance = Eigen::Matrix<double, 15, 15>::Identity();
};

struct ImuPreintegration
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  double dt = 0.0;
  Eigen::Quaterniond delta_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d delta_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d delta_velocity = Eigen::Vector3d::Zero();

  void reset();
  void integrate(const Eigen::Vector3d &accel, const Eigen::Vector3d &gyro, double dt_sec);
};

Eigen::Isometry3d expSE3(const Eigen::Matrix<double, 6, 1> &xi);
Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Isometry3d &transform);
Eigen::Isometry3d projectToSE3(const Eigen::Isometry3d &transform);
double wrapAngle(double angle);
double yawOf(const Eigen::Isometry3d &pose);
Eigen::Isometry3d planarTransform(double x, double y, double yaw);
uint8_t convertSemanticLabel(int label, const std::string &mode);

enum class FactorType
{
  LidarRegistration,
  ImuPreintegration,
  Wheel,
  Visual,
  MediumMatch,
  MapMatch,
  Prior
};

struct Factor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FactorType type = FactorType::LidarRegistration;
  int first = -1;
  int second = -1;
  Eigen::Isometry3d measurement = Eigen::Isometry3d::Identity();
  Eigen::Vector3d wheel_measurement = Eigen::Vector3d::Zero();
  ImuPreintegration imu;
  Eigen::MatrixXd information;
  double huber_delta = 1.0;
  double confidence = 1.0;
  bool active = true;
};

class SlidingWindowOptimizer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit SlidingWindowOptimizer(std::size_t max_states = 12);

  void reset();
  int addState(const PoseState &state);
  void addRelativeFactor(int first, int second, const Eigen::Isometry3d &measurement,
                         const Eigen::Matrix<double, 6, 6> &information,
                         FactorType type, double huber_delta);
  void addWheelFactor(int first, int second, const Eigen::Vector3d &measurement,
                      const Eigen::Matrix3d &information, double huber_delta);
  void addImuFactor(int first, int second, const ImuPreintegration &measurement,
                    const Eigen::Matrix<double, 9, 9> &information, double huber_delta);
  void addAbsoluteFactor(int state_index, const Eigen::Isometry3d &measurement,
                         const Eigen::Matrix<double, 6, 6> &information,
                         FactorType type, double confidence, double huber_delta);

  bool optimize(int iterations, double max_step_norm);
  void marginalizeOldest();
  const std::vector<PoseState> &states() const { return states_; }
  std::vector<PoseState> &states() { return states_; }
  const std::vector<Factor> &factors() const { return factors_; }
  bool empty() const { return states_.empty(); }
  const PoseState &latestState() const { return states_.back(); }
  std::size_t size() const { return states_.size(); }

private:
  Eigen::VectorXd residual(const Factor &factor, const std::vector<PoseState> &states) const;
  Eigen::VectorXd residualForType(const Factor &factor, const PoseState &first,
                                  const PoseState &second) const;
  Eigen::VectorXd residualForAbsolute(const Factor &factor, const PoseState &state) const;
  Eigen::MatrixXd numericalJacobian(const Factor &factor, int state_index,
                                    const std::vector<PoseState> &states) const;
  static void boxPlus(PoseState &state, const Eigen::Matrix<double, 15, 1> &delta);
  static double robustWeight(double squared_norm, double huber_delta);

  std::size_t max_states_;
  std::vector<PoseState> states_;
  std::vector<Factor> factors_;
};

struct BevPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  uint8_t label = 0;
  float confidence = 1.0F;
  bool dynamic = false;
};

struct BevGrid
{
  int width = 0;
  int height = 0;
  double resolution = 0.25;
  double origin_x = 0.0;
  double origin_y = 0.0;
  std::vector<float> occupancy;
  std::vector<float> height_min;
  std::vector<float> height_max;
  std::vector<uint8_t> labels;
  std::vector<float> quality;

  void reset(int cells_x, int cells_y, double cell_resolution, double center_x, double center_y);
  bool worldToCell(double x, double y, int &ix, int &iy) const;
  Eigen::Vector2d cellCenter(int ix, int iy) const;
  void insert(const BevPoint &point, double ground_z, double max_height);
  bool occupiedAt(double x, double y) const;
};

struct PriorMap
{
  int width = 0;
  int height = 0;
  double resolution = 0.25;
  double origin_x = 0.0;
  double origin_y = 0.0;
  std::vector<float> occupancy;
  std::vector<uint8_t> labels;
  std::vector<float> edge;

  void clear();
  bool valid() const;
  bool worldToCell(double x, double y, int &ix, int &iy) const;
  Eigen::Vector2d cellCenter(int ix, int iy) const;
  float occupancyAt(double x, double y) const;
  uint8_t labelAt(double x, double y) const;
  float edgeAt(double x, double y) const;
  void recomputeEdges();
};

struct MatchPair
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d source = Eigen::Vector3d::Zero();
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  double residual = std::numeric_limits<double>::infinity();
  uint8_t source_label = 0;
  uint8_t target_label = 0;
  bool candidate = false;
  bool inlier = false;
  bool outlier = false;
  bool applied = false;
};

struct MapMatchResult
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  bool gate_passed = false;
  double best_score = -std::numeric_limits<double>::infinity();
  double second_score = -std::numeric_limits<double>::infinity();
  double confidence = 0.0;
  double inlier_ratio = 0.0;
  double search_radius = 0.0;
  Eigen::Isometry3d global_from_odom = Eigen::Isometry3d::Identity();
  std::vector<MatchPair, Eigen::aligned_allocator<MatchPair>> pairs;
  std::string reject_reason;
};

struct MatcherOptions
{
  double min_search_radius = 5.0;
  double max_search_radius = 60.0;
  double confidence_gamma = 9.21;
  double coarse_translation_step = 1.0;
  double fine_translation_step = 0.25;
  double coarse_yaw_step_deg = 10.0;
  double fine_yaw_step_deg = 2.0;
  double yaw_search_deg = 45.0;
  double match_distance = 1.0;
  double inlier_distance = 0.65;
  double min_inlier_ratio = 0.25;
  double min_confidence = 0.45;
  double min_score_gap = 0.04;
  double occupancy_weight = 1.0;
  double edge_weight = 0.35;
  double semantic_weight = 0.25;
  double dynamic_penalty = 0.75;
  int max_points = 2500;
  int max_candidates = 200000;
};

class PriorMatcher
{
public:
  explicit PriorMatcher(const MatcherOptions &options);
  MapMatchResult match(const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &local_points,
                       const PriorMap &prior, const Eigen::Isometry3d &predicted_global_from_odom,
                       const Eigen::Matrix3d &position_covariance) const;

private:
  double evaluate(const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points,
                  const PriorMap &prior, const Eigen::Isometry3d &global_from_odom) const;
  void buildPairs(const std::vector<BevPoint, Eigen::aligned_allocator<BevPoint>> &points,
                  const PriorMap &prior, const Eigen::Isometry3d &global_from_odom,
                  MapMatchResult &result) const;
  static std::vector<double> axisSamples(double center, double radius, double step);

  MatcherOptions options_;
};

}  // namespace hybrid_localization

#endif  // HYBRID_LOCALIZATION_CORE_H
