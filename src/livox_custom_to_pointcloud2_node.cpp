#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <livox_ros_driver/CustomMsg.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

class LivoxCustomToPointCloud2Node {
public:
  LivoxCustomToPointCloud2Node() : nh_(), pnh_("~") {
    pnh_.param<std::string>("input_custom_topic", input_topic_, "/livox/mid360/points");
    pnh_.param<std::string>("output_cloud_topic", output_topic_, "/livox/mid360/points_xyzirt");
    pnh_.param("point_stride", point_stride_, 1);
    pnh_.param("min_range_m", min_range_m_, 0.3);
    pnh_.param("max_range_m", max_range_m_, 120.0);
    pnh_.param("drop_invalid", drop_invalid_, true);

    if (point_stride_ < 1) {
      point_stride_ = 1;
    }
    min_range2_ = min_range_m_ * min_range_m_;
    max_range2_ = max_range_m_ * max_range_m_;

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 2);
    sub_ = nh_.subscribe(input_topic_, 5, &LivoxCustomToPointCloud2Node::callback, this);
    diagnostic_timer_ = nh_.createWallTimer(
        ros::WallDuration(3.0), &LivoxCustomToPointCloud2Node::diagnosticTimerCb, this);
    ROS_INFO_STREAM("livox_custom_to_pointcloud2 started input=" << input_topic_
                    << " output=" << output_topic_
                    << " stride=" << point_stride_
                    << " range=[" << min_range_m_ << "," << max_range_m_ << "]");
  }

private:
  void callback(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
    ++input_msg_count_;
    last_input_wall_ = ros::WallTime::now();
    if (!msg || msg->points.empty()) {
      ROS_WARN_THROTTLE(2.0, "Received empty Livox CustomMsg on %s", input_topic_.c_str());
      return;
    }

    std::vector<size_t> keep;
    keep.reserve(msg->points.size() / static_cast<size_t>(point_stride_) + 1);
    for (size_t i = 0; i < msg->points.size(); i += static_cast<size_t>(point_stride_)) {
      const auto& p = msg->points[i];
      const double r2 = static_cast<double>(p.x) * p.x +
                        static_cast<double>(p.y) * p.y +
                        static_cast<double>(p.z) * p.z;
      if (drop_invalid_ && (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))) {
        continue;
      }
      if (r2 < min_range2_ || r2 > max_range2_) {
        continue;
      }
      keep.push_back(i);
    }

    sensor_msgs::PointCloud2 cloud;
    cloud.header = msg->header;
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = false;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
        6,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32,
        "line", 1, sensor_msgs::PointField::UINT16,
        "offset_time", 1, sensor_msgs::PointField::UINT32);
    modifier.resize(keep.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_i(cloud, "intensity");
    sensor_msgs::PointCloud2Iterator<uint16_t> iter_line(cloud, "line");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_offset(cloud, "offset_time");

    for (const size_t idx : keep) {
      const auto& p = msg->points[idx];
      *iter_x = p.x;
      *iter_y = p.y;
      *iter_z = p.z;
      *iter_i = static_cast<float>(p.reflectivity);
      *iter_line = static_cast<uint16_t>(p.line);
      *iter_offset = p.offset_time;
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_i;
      ++iter_line;
      ++iter_offset;
    }

    pub_.publish(cloud);
    ++output_msg_count_;
    last_output_wall_ = ros::WallTime::now();
    ROS_INFO_THROTTLE(2.0, "livox custom converted input=%zu output=%zu frame=%s",
                      msg->points.size(), keep.size(), cloud.header.frame_id.c_str());
  }

  void diagnosticTimerCb(const ros::WallTimerEvent&) {
    const ros::WallTime now = ros::WallTime::now();
    if (input_msg_count_ == 0) {
      ROS_WARN("livox_custom_to_pointcloud2 has not received Livox CustomMsg on %s. "
               "Check the Livox driver or rosbag topic and the launch arg livox_custom_topic.",
               input_topic_.c_str());
      return;
    }
    if ((now - last_input_wall_).toSec() > 3.0) {
      ROS_WARN("livox_custom_to_pointcloud2 input stalled: topic=%s last_age=%.1fs",
               input_topic_.c_str(), (now - last_input_wall_).toSec());
    }
    if (output_msg_count_ == 0) {
      ROS_WARN("livox_custom_to_pointcloud2 received %llu messages but has not published %s. "
               "Check range filters and empty input packets.",
               static_cast<unsigned long long>(input_msg_count_), output_topic_.c_str());
    } else if ((now - last_output_wall_).toSec() > 3.0) {
      ROS_WARN("livox_custom_to_pointcloud2 output stalled: topic=%s last_age=%.1fs",
               output_topic_.c_str(), (now - last_output_wall_).toSec());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::WallTimer diagnostic_timer_;
  std::string input_topic_;
  std::string output_topic_;
  int point_stride_ = 1;
  double min_range_m_ = 0.3;
  double max_range_m_ = 120.0;
  double min_range2_ = 0.09;
  double max_range2_ = 14400.0;
  bool drop_invalid_ = true;
  uint64_t input_msg_count_ = 0;
  uint64_t output_msg_count_ = 0;
  ros::WallTime last_input_wall_;
  ros::WallTime last_output_wall_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "livox_custom_to_pointcloud2");
  LivoxCustomToPointCloud2Node node;
  ros::spin();
  return 0;
}
