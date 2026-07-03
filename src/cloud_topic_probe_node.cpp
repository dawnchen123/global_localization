#include <ros/ros.h>
#include <ros/master.h>
#include <sensor_msgs/PointCloud2.h>
#include <boost/shared_ptr.hpp>
#include <set>
#include <string>
#include <vector>
#include <sstream>

class CloudTopicProbe
{
public:
  CloudTopicProbe(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh)
  {
    pnh_.param<double>("listen_sec", listen_sec_, 5.0);
    pnh_.param<bool>("subscribe_all", subscribe_all_, true);
    pnh_.param<std::string>("topic", single_topic_, std::string(""));

    if (!single_topic_.empty()) {
      ROS_INFO("Probing single PointCloud2 topic: %s", single_topic_.c_str());
      subs_.push_back(nh_.subscribe<sensor_msgs::PointCloud2>(single_topic_, 1,
        boost::bind(&CloudTopicProbe::cloudCb, this, _1, single_topic_)));
    } else {
      ros::master::V_TopicInfo topics;
      if (!ros::master::getTopics(topics)) {
        ROS_ERROR("Failed to query ROS master. Is roscore running?");
        return;
      }
      ROS_INFO("PointCloud2 topics on ROS master:");
      for (const auto& t : topics) {
        if (t.datatype == "sensor_msgs/PointCloud2") {
          ROS_INFO("  %s", t.name.c_str());
          if (subscribe_all_) {
            subs_.push_back(nh_.subscribe<sensor_msgs::PointCloud2>(t.name, 1,
              boost::bind(&CloudTopicProbe::cloudCb, this, _1, t.name)));
          }
        }
      }
      if (subs_.empty()) {
        ROS_WARN("No sensor_msgs/PointCloud2 topics found now. Start FAST-LIVO2 first, then rerun this node.");
      }
    }

    timer_ = nh_.createTimer(ros::Duration(listen_sec_), &CloudTopicProbe::timerCb, this, true);
  }

private:
  void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg, const std::string& topic)
  {
    if (printed_.count(topic)) return;
    printed_.insert(topic);

    bool has_x=false, has_y=false, has_z=false, has_i=false;
    std::ostringstream fields;
    for (const auto& f : msg->fields) {
      fields << f.name << " ";
      if (f.name == "x") has_x = true;
      if (f.name == "y") has_y = true;
      if (f.name == "z") has_z = true;
      if (f.name == "intensity" || f.name == "reflectivity") has_i = true;
    }
    const size_t npts = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    if (has_x && has_y && has_z && npts > 0 && !msg->data.empty()) {
      ROS_INFO("[OK_XYZ] %s  points=%zu  point_step=%u  frame=%s  fields=[%s] intensity_or_reflectivity=%s",
               topic.c_str(), npts, msg->point_step, msg->header.frame_id.c_str(), fields.str().c_str(), has_i ? "yes" : "no");
    } else {
      ROS_WARN("[BAD] %s  points=%zu width=%u height=%u data=%zu point_step=%u frame=%s fields=[%s]",
               topic.c_str(), npts, msg->width, msg->height, msg->data.size(), msg->point_step, msg->header.frame_id.c_str(), fields.str().c_str());
    }
  }

  void timerCb(const ros::TimerEvent&)
  {
    if (printed_.empty()) {
      ROS_WARN("No PointCloud2 message received in %.1f seconds. Check whether FAST-LIVO2 is publishing and whether rosbag is playing.", listen_sec_);
    }
    ROS_INFO("cloud_topic_probe_node finished. Use a topic marked [OK_XYZ] as fast_livo_cloud_topic.");
    ros::shutdown();
  }

  ros::NodeHandle nh_, pnh_;
  std::vector<ros::Subscriber> subs_;
  std::set<std::string> printed_;
  ros::Timer timer_;
  double listen_sec_ = 5.0;
  bool subscribe_all_ = true;
  std::string single_topic_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cloud_topic_probe_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  CloudTopicProbe node(nh, pnh);
  ros::spin();
  return 0;
}
