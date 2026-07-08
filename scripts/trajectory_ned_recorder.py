#!/usr/bin/env python3
import math
import os
import threading

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def quat_norm(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12 or not math.isfinite(n):
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


class StreamWriter(object):
    def __init__(self, name, topic, path, recorder):
        self.name = name
        self.topic = topic
        self.path = path
        self.recorder = recorder
        self.count = 0
        self.skipped_rate = 0
        self.last_write_stamp = None
        self.file = None
        self.sub = None

    def open(self):
        mode = "a" if self.recorder.append else "w"
        self.file = open(self.path, mode, buffering=1)
        if self.recorder.write_header and (not self.recorder.append or os.path.getsize(self.path) == 0):
            self.file.write("# t\tpos_n\tpos_e\tpos_d\tqx\tqy\tqz\tqw\n")

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None

    def start(self):
        self.sub = rospy.Subscriber(self.topic, Odometry, self.cb, queue_size=200)

    def cb(self, msg):
        stamp = msg.header.stamp
        if stamp.to_sec() <= 0.0:
            stamp = rospy.Time.now()
        stamp_sec = stamp.to_sec()
        if self.recorder.max_rate_hz > 0.0 and self.last_write_stamp is not None:
            if stamp_sec - self.last_write_stamp < 1.0 / self.recorder.max_rate_hz:
                self.skipped_rate += 1
                return
        line = self.recorder.format_odom(msg, stamp_sec)
        with self.recorder.lock:
            if self.file is None:
                return
            self.file.write(line)
            self.count += 1
            self.last_write_stamp = stamp_sec
            if self.recorder.flush_every_n > 0 and self.count % self.recorder.flush_every_n == 0:
                self.file.flush()


class TrajectoryNEDRecorder(object):
    def __init__(self):
        self.fast_livo_odom_topic = rospy.get_param("~fast_livo_odom_topic", "/aft_mapped_to_init")
        self.fused_odom_topic = rospy.get_param("~fused_odom_topic", "/semantic_graph_corrected_odom")
        self.output_dir = rospy.get_param("~output_dir", "/tmp/trajectory_records")
        self.fast_livo_filename = rospy.get_param("~fast_livo_filename", "fast_livo2_local_ned.txt")
        self.fused_filename = rospy.get_param("~fused_filename", "fused_local_ned.txt")
        self.coordinate_mode = rospy.get_param("~coordinate_mode", "enu_to_ned").strip().lower()
        self.time_mode = rospy.get_param("~time_mode", "stamp").strip().lower()
        self.time_offset_sec = float(rospy.get_param("~time_offset_sec", 0.0))
        self.time_precision = int(rospy.get_param("~time_precision", 3))
        self.position_precision = int(rospy.get_param("~position_precision", 4))
        self.quaternion_precision = int(rospy.get_param("~quaternion_precision", 8))
        self.max_rate_hz = float(rospy.get_param("~max_rate_hz", 0.0))
        self.flush_every_n = int(rospy.get_param("~flush_every_n", 1))
        self.write_header = bool(rospy.get_param("~write_header", False))
        self.append = bool(rospy.get_param("~append", False))

        if self.coordinate_mode in ("ned", "local_ned", "ros_enu_to_ned"):
            self.coordinate_mode = "enu_to_ned"
        if self.coordinate_mode not in ("enu_to_ned", "as_is"):
            rospy.logwarn("Unknown coordinate_mode=%s, fallback to enu_to_ned", self.coordinate_mode)
            self.coordinate_mode = "enu_to_ned"
        if self.time_mode not in ("stamp", "relative"):
            rospy.logwarn("Unknown time_mode=%s, fallback to stamp", self.time_mode)
            self.time_mode = "stamp"

        os.makedirs(self.output_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.first_stamp = None
        self.status_pub = rospy.Publisher("~status", String, queue_size=1)

        # q_ned_enu represents R_ned_enu = [[0,1,0],[1,0,0],[0,0,-1]].
        s = math.sqrt(0.5)
        self.q_ned_enu = (s, s, 0.0, 0.0)

        self.fast_livo = StreamWriter(
            "fast_livo2",
            self.fast_livo_odom_topic,
            os.path.join(self.output_dir, self.fast_livo_filename),
            self,
        )
        self.fused = StreamWriter(
            "fused",
            self.fused_odom_topic,
            os.path.join(self.output_dir, self.fused_filename),
            self,
        )
        self.fast_livo.open()
        self.fused.open()
        self.fast_livo.start()
        self.fused.start()

        rospy.Timer(rospy.Duration(2.0), self.timer_cb)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "trajectory_ned_recorder started fast_livo=%s fused=%s output_dir=%s coordinate_mode=%s time_mode=%s",
            self.fast_livo_odom_topic,
            self.fused_odom_topic,
            self.output_dir,
            self.coordinate_mode,
            self.time_mode,
        )
        rospy.loginfo("writing FAST-LIVO2 trajectory: %s", self.fast_livo.path)
        rospy.loginfo("writing fused trajectory: %s", self.fused.path)

    def output_time(self, stamp_sec):
        if self.time_mode == "relative":
            if self.first_stamp is None:
                self.first_stamp = stamp_sec
            return stamp_sec - self.first_stamp + self.time_offset_sec
        return stamp_sec + self.time_offset_sec

    def convert_pose(self, p, q):
        if self.coordinate_mode == "as_is":
            return (p.x, p.y, p.z), quat_norm((q.x, q.y, q.z, q.w))
        pos = (p.y, p.x, -p.z)
        quat = quat_norm(quat_multiply(self.q_ned_enu, quat_norm((q.x, q.y, q.z, q.w))))
        return pos, quat

    def format_odom(self, msg, stamp_sec):
        t = self.output_time(stamp_sec)
        pos, quat = self.convert_pose(msg.pose.pose.position, msg.pose.pose.orientation)
        fmt = (
            "{t:." + str(self.time_precision) + "f}\t"
            "{x:." + str(self.position_precision) + "f}\t"
            "{y:." + str(self.position_precision) + "f}\t"
            "{z:." + str(self.position_precision) + "f}\t"
            "{qx:." + str(self.quaternion_precision) + "f}\t"
            "{qy:." + str(self.quaternion_precision) + "f}\t"
            "{qz:." + str(self.quaternion_precision) + "f}\t"
            "{qw:." + str(self.quaternion_precision) + "f}\n"
        )
        return fmt.format(
            t=t,
            x=pos[0],
            y=pos[1],
            z=pos[2],
            qx=quat[0],
            qy=quat[1],
            qz=quat[2],
            qw=quat[3],
        )

    def timer_cb(self, _event):
        msg = (
            "fast_livo2 count=%d skipped_rate=%d file=%s | "
            "fused count=%d skipped_rate=%d file=%s"
        ) % (
            self.fast_livo.count,
            self.fast_livo.skipped_rate,
            self.fast_livo.path,
            self.fused.count,
            self.fused.skipped_rate,
            self.fused.path,
        )
        self.status_pub.publish(String(data=msg))
        rospy.loginfo_throttle(10.0, "trajectory recorder: %s", msg)

    def shutdown(self):
        with self.lock:
            self.fast_livo.close()
            self.fused.close()


if __name__ == "__main__":
    rospy.init_node("trajectory_ned_recorder")
    TrajectoryNEDRecorder()
    rospy.spin()
