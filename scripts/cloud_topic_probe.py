#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Fallback Python probe. If your Python environment is polluted, use cloud_topic_probe_node instead.
import os
import sys
# Avoid local modules such as decimal.py shadowing Python stdlib when rosrun starts from a workspace.
sys.path = [p for p in sys.path if p not in ('', os.getcwd())]
# -*- coding: utf-8 -*-
"""Probe PointCloud2 topics and print width/height/fields for selecting fast_livo_cloud_topic."""
import rospy
import rosgraph
from sensor_msgs.msg import PointCloud2

class Probe:
    def __init__(self):
        self.timeout = float(rospy.get_param('~timeout', 8.0))
        self.prefix_filter = rospy.get_param('~prefix_filter', '')
        self.seen = {}
        master = rosgraph.Master('/cloud_topic_probe')
        topics = master.getPublishedTopics('/')
        self.cloud_topics = [t for t, typ in topics if typ == 'sensor_msgs/PointCloud2']
        if self.prefix_filter:
            self.cloud_topics = [t for t in self.cloud_topics if self.prefix_filter in t]
        rospy.loginfo('Found %d PointCloud2 topics:', len(self.cloud_topics))
        for t in self.cloud_topics:
            rospy.loginfo('  %s', t)
            rospy.Subscriber(t, PointCloud2, self.cb, callback_args=t, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(self.timeout), self.finish, oneshot=True)

    def cb(self, msg, topic):
        if topic in self.seen:
            return
        fields = [f.name for f in msg.fields]
        has_xyz = all(k in fields for k in ['x','y','z'])
        self.seen[topic] = (msg.width, msg.height, msg.point_step, len(msg.data), fields, has_xyz, msg.header.frame_id)
        status = 'OK_XYZ' if has_xyz and len(msg.data) > 0 else 'BAD_OR_EMPTY'
        rospy.logwarn('[%s] %s width=%d height=%d point_step=%d data=%d frame=%s fields=%s',
                      status, topic, msg.width, msg.height, msg.point_step, len(msg.data), msg.header.frame_id, fields)

    def finish(self, _):
        rospy.loginfo('==== Recommended fast_livo_cloud_topic candidates ====')
        for topic, info in self.seen.items():
            w,h,ps,ds,fields,has_xyz,frame = info
            if has_xyz and ds > 0:
                rospy.loginfo('  %s  width=%d height=%d frame=%s fields=%s', topic, w, h, frame, fields)
        rospy.signal_shutdown('probe finished')

if __name__ == '__main__':
    rospy.init_node('cloud_topic_probe')
    Probe()
    rospy.spin()