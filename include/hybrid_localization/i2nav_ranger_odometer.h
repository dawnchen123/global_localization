#ifndef HYBRID_LOCALIZATION_I2NAV_RANGER_ODOMETER_H
#define HYBRID_LOCALIZATION_I2NAV_RANGER_ODOMETER_H

#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>
#include <ros/serialization.h>
#include <std_msgs/Header.h>

#include <boost/shared_ptr.hpp>

#include <cstdint>

// Header-only compatibility definition for the message recorded by the
// i2Nav-Robot bags. This avoids a build dependency on the unavailable
// insprobe_msgs package while retaining its exact wire type and MD5.
namespace insprobe_msgs
{

struct RangerOdometer
{
  std_msgs::Header header;
  uint16_t week = 0U;
  double weeksec = 0.0;
  double unixtime = 0.0;
  double left_front_speed = 0.0;
  double right_front_speed = 0.0;
  double right_back_speed = 0.0;
  double left_back_speed = 0.0;
  double left_front_angle = 0.0;
  double right_front_angle = 0.0;
  double right_back_angle = 0.0;
  double left_back_angle = 0.0;

  using Ptr = boost::shared_ptr<RangerOdometer>;
  using ConstPtr = boost::shared_ptr<const RangerOdometer>;
};

using RangerOdometerPtr = RangerOdometer::Ptr;
using RangerOdometerConstPtr = RangerOdometer::ConstPtr;

}  // namespace insprobe_msgs

namespace ros
{
namespace message_traits
{

template<>
struct IsMessage<insprobe_msgs::RangerOdometer> : TrueType {};

template<>
struct IsMessage<const insprobe_msgs::RangerOdometer> : TrueType {};

template<>
struct IsFixedSize<insprobe_msgs::RangerOdometer> : FalseType {};

template<>
struct IsFixedSize<const insprobe_msgs::RangerOdometer> : FalseType {};

template<>
struct HasHeader<insprobe_msgs::RangerOdometer> : TrueType {};

template<>
struct HasHeader<const insprobe_msgs::RangerOdometer> : TrueType {};

template<>
struct MD5Sum<insprobe_msgs::RangerOdometer>
{
  static const char *value() { return "e661bb739a55d1297955b137b4eef03f"; }
  static const char *value(const insprobe_msgs::RangerOdometer &) { return value(); }
  static const uint64_t static_value1 = 0xe661bb739a55d129ULL;
  static const uint64_t static_value2 = 0x7955b137b4eef03fULL;
};

template<>
struct DataType<insprobe_msgs::RangerOdometer>
{
  static const char *value() { return "insprobe_msgs/RangerOdometer"; }
  static const char *value(const insprobe_msgs::RangerOdometer &) { return value(); }
};

template<>
struct Definition<insprobe_msgs::RangerOdometer>
{
  static const char *value()
  {
    return "std_msgs/Header header\n"
           "uint16 week\n"
           "float64 weeksec\n"
           "float64 unixtime\n"
           "float64 left_front_speed\n"
           "float64 right_front_speed\n"
           "float64 right_back_speed\n"
           "float64 left_back_speed\n"
           "float64 left_front_angle\n"
           "float64 right_front_angle\n"
           "float64 right_back_angle\n"
           "float64 left_back_angle\n";
  }
  static const char *value(const insprobe_msgs::RangerOdometer &) { return value(); }
};

}  // namespace message_traits

namespace serialization
{

template<>
struct Serializer<insprobe_msgs::RangerOdometer>
{
  template<typename Stream, typename T>
  inline static void allInOne(Stream &stream, T message)
  {
    stream.next(message.header);
    stream.next(message.week);
    stream.next(message.weeksec);
    stream.next(message.unixtime);
    stream.next(message.left_front_speed);
    stream.next(message.right_front_speed);
    stream.next(message.right_back_speed);
    stream.next(message.left_back_speed);
    stream.next(message.left_front_angle);
    stream.next(message.right_front_angle);
    stream.next(message.right_back_angle);
    stream.next(message.left_back_angle);
  }

  ROS_DECLARE_ALLINONE_SERIALIZER
};

}  // namespace serialization
}  // namespace ros

#endif  // HYBRID_LOCALIZATION_I2NAV_RANGER_ODOMETER_H
