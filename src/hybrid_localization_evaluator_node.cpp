#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct PoseSample
{
  double stamp = 0.0;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
};

Eigen::Isometry3d fromMessage(const geometry_msgs::Pose &pose)
{
  Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  if (q.norm() < 1e-8) q = Eigen::Quaterniond::Identity();
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = q.normalized().toRotationMatrix();
  result.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return result;
}

class Evaluator
{
public:
  Evaluator()
      : nh_(), private_nh_("~")
  {
    private_nh_.param<std::string>("estimate_topic", estimate_topic_, "/hybrid/odometry");
    private_nh_.param<std::string>("ground_truth_topic", ground_truth_topic_, "/ground_truth/odom");
    private_nh_.param<std::string>("metrics_topic", metrics_topic_, "/hybrid/evaluation");
    private_nh_.param<std::string>("metrics_array_topic", metrics_array_topic_, "/hybrid/evaluation/values");
    private_nh_.param<std::string>("save_path", save_path_, "");
    private_nh_.param("max_association_dt", max_association_dt_, 0.05);
    private_nh_.param("align_initial_pose", align_initial_pose_, true);
    estimate_sub_ = nh_.subscribe(estimate_topic_, 200, &Evaluator::estimateCallback, this);
    truth_sub_ = nh_.subscribe(ground_truth_topic_, 200, &Evaluator::truthCallback, this);
    metrics_pub_ = nh_.advertise<std_msgs::String>(metrics_topic_, 2);
    metrics_array_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(metrics_array_topic_, 2);
  }

  ~Evaluator()
  {
    if (save_path_.empty()) return;
    std::ofstream output(save_path_.c_str());
    if (!output.is_open()) return;
    output << "index,estimate_x,estimate_y,estimate_z,truth_x,truth_y,truth_z,error_m\n";
    std::size_t index = 0;
    for (const auto &sample : matches_)
    {
      const Eigen::Vector3d estimate = (align_initial_pose_ ? align_ * sample.first : sample.first).translation();
      const Eigen::Vector3d truth = sample.second.translation();
      output << index++ << "," << estimate.x() << "," << estimate.y() << "," << estimate.z() << ","
             << truth.x() << "," << truth.y() << "," << truth.z() << ","
             << (estimate - truth).norm() << "\n";
    }
  }

  void spin() { ros::spin(); }

private:
  void truthCallback(const nav_msgs::OdometryConstPtr &message)
  {
    PoseSample sample;
    sample.stamp = message->header.stamp.toSec();
    sample.pose = fromMessage(message->pose.pose);
    truth_.push_back(sample);
    while (truth_.size() > 2000) truth_.erase(truth_.begin());
  }

  void estimateCallback(const nav_msgs::OdometryConstPtr &message)
  {
    if (truth_.empty()) return;
    PoseSample estimate;
    estimate.stamp = message->header.stamp.toSec();
    estimate.pose = fromMessage(message->pose.pose);
    const PoseSample *nearest = nullptr;
    double nearest_dt = std::numeric_limits<double>::infinity();
    for (const PoseSample &truth : truth_)
    {
      const double dt = std::abs(truth.stamp - estimate.stamp);
      if (dt < nearest_dt)
      {
        nearest_dt = dt;
        nearest = &truth;
      }
    }
    if (nearest == nullptr || nearest_dt > max_association_dt_) return;
    if (!aligned_)
    {
      align_ = nearest->pose * estimate.pose.inverse();
      aligned_ = true;
    }
    matches_.push_back(std::make_pair(estimate.pose, nearest->pose));
    while (matches_.size() > 20000) matches_.erase(matches_.begin());
    publishMetrics(estimate.stamp);
  }

  void publishMetrics(double stamp)
  {
    if (matches_.empty()) return;
    double squared_ate = 0.0;
    double squared_rpe = 0.0;
    double max_error = 0.0;
    for (std::size_t i = 0; i < matches_.size(); ++i)
    {
      const Eigen::Isometry3d estimate = align_initial_pose_ ? align_ * matches_[i].first : matches_[i].first;
      const Eigen::Vector3d error = estimate.translation() - matches_[i].second.translation();
      squared_ate += error.squaredNorm();
      max_error = std::max(max_error, error.norm());
      if (i > 0)
      {
        const Eigen::Isometry3d estimate_delta =
            (align_initial_pose_ ? align_ * matches_[i - 1].first : matches_[i - 1].first).inverse() * estimate;
        const Eigen::Isometry3d truth_delta = matches_[i - 1].second.inverse() * matches_[i].second;
        squared_rpe += (estimate_delta.translation() - truth_delta.translation()).squaredNorm();
      }
    }
    const double ate = std::sqrt(squared_ate / static_cast<double>(matches_.size()));
    const double rpe = matches_.size() > 1 ?
        std::sqrt(squared_rpe / static_cast<double>(matches_.size() - 1)) : 0.0;
    const double loop_error = matches_.size() > 1 ?
        (((align_initial_pose_ ? align_ * matches_.front().first : matches_.front().first).inverse() *
          (align_initial_pose_ ? align_ * matches_.back().first : matches_.back().first)).translation() -
         (matches_.front().second.inverse() * matches_.back().second).translation()).norm() : 0.0;
    std_msgs::String message;
    std::ostringstream text;
    text << "matches=" << matches_.size() << ";ate_rmse=" << ate << ";rpe_rmse=" << rpe
         << ";max_position_error=" << max_error << ";trajectory_span=" << loop_error;
    message.data = text.str();
    metrics_pub_.publish(message);
    std_msgs::Float64MultiArray values;
    values.data = {ate, rpe, max_error, loop_error, static_cast<double>(matches_.size()), stamp};
    metrics_array_pub_.publish(values);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber estimate_sub_;
  ros::Subscriber truth_sub_;
  ros::Publisher metrics_pub_;
  ros::Publisher metrics_array_pub_;
  std::string estimate_topic_;
  std::string ground_truth_topic_;
  std::string metrics_topic_;
  std::string metrics_array_topic_;
  std::string save_path_;
  double max_association_dt_ = 0.05;
  bool align_initial_pose_ = true;
  bool aligned_ = false;
  Eigen::Isometry3d align_ = Eigen::Isometry3d::Identity();
  std::vector<PoseSample> truth_;
  std::vector<std::pair<Eigen::Isometry3d, Eigen::Isometry3d>> matches_;
};

}  // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hybrid_localization_evaluator");
  Evaluator evaluator;
  evaluator.spin();
  return 0;
}
