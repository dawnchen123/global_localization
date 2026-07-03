#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Noetic / Python3.8 side output publisher for the file-queue global
localization pipeline.

It watches output/result_<request_id>.json produced by an external matching service and
publishes accepted results as geometry_msgs/PoseWithCovarianceStamped on
/map_match_pose_external. It can also publish the debug image created by the
matching service.
"""
import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image


class EloftrQueueResultPublisher:
    def __init__(self):
        self.queue_dir = Path(rospy.get_param('~queue_dir', '/tmp/eloftr_queue'))
        self.output_topic = rospy.get_param('~output_topic', '/map_match_pose_external')
        self.debug_topic = rospy.get_param('~debug_topic', '/map_match/file_queue_debug')
        self.heatmap_topic = rospy.get_param('~heatmap_topic', '/map_match/similarity_heatmap')
        self.overlay_topic = rospy.get_param('~overlay_topic', '/map_match/sat_bev_overlay')
        self.local_semantic_topic = rospy.get_param('~local_semantic_topic', '/local_semantic_map/color')
        self.satellite_semantic_topic = rospy.get_param('~satellite_semantic_topic', '/satellite_semantic_map/color')
        self.poll_rate = float(rospy.get_param('~poll_rate', 2.0))
        self.publish_rejected = bool(rospy.get_param('~publish_rejected', False))
        self.frame_id = rospy.get_param('~frame_id', 'earth')
        self.move_processed = bool(rospy.get_param('~move_processed', True))

        self.output_dir = self.queue_dir / 'output'
        self.done_dir = self.queue_dir / 'done'
        for d in [self.output_dir, self.done_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.pub_pose = rospy.Publisher(self.output_topic, PoseWithCovarianceStamped, queue_size=5)
        self.pub_debug = rospy.Publisher(self.debug_topic, Image, queue_size=2)
        self.pub_heatmap = rospy.Publisher(self.heatmap_topic, Image, queue_size=2)
        self.pub_overlay = rospy.Publisher(self.overlay_topic, Image, queue_size=2)
        self.pub_local_semantic = rospy.Publisher(self.local_semantic_topic, Image, queue_size=2)
        self.pub_satellite_semantic = rospy.Publisher(self.satellite_semantic_topic, Image, queue_size=2)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(0.001, self.poll_rate)), self.timer_cb)
        rospy.loginfo('queue_result_publisher started. queue=%s output=%s', str(self.queue_dir), self.output_topic)

    def publish_image_path(self, path, publisher, label):
        if not path or not os.path.exists(path):
            return
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                return
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id
            publisher.publish(msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Failed to publish %s image: %s', label, str(exc))

    def publish_debug_if_available(self, result):
        self.publish_image_path(result.get('debug_path', ''), self.pub_debug, 'file-queue debug')
        self.publish_image_path(result.get('heatmap_path', ''), self.pub_heatmap, 'similarity heatmap')
        self.publish_image_path(result.get('overlay_path', ''), self.pub_overlay, 'satellite BEV overlay')
        self.publish_image_path(result.get('local_semantic_color_path', ''), self.pub_local_semantic, 'local semantic map')
        self.publish_image_path(result.get('satellite_semantic_color_path', ''), self.pub_satellite_semantic, 'satellite semantic map')

    def process_result(self, path):
        try:
            with open(path, 'r') as f:
                r = json.load(f)
        except Exception as exc:
            rospy.logwarn('Failed to read external map-match result %s: %s', path, str(exc))
            return
        accepted = bool(r.get('accepted', False))
        if not accepted:
            rospy.logwarn('External map-match result rejected: id=%s reason=%s matches=%s inliers=%s conf=%.3f ratio=%.3f',
                          r.get('request_id', ''), r.get('reason', ''), r.get('matches', 0), r.get('inliers', 0),
                          float(r.get('confidence', 0.0)), float(r.get('inlier_ratio', 0.0)))
            self.publish_debug_if_available(r)
            return
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.pose.pose.position.x = float(r['meas_e'])
        msg.pose.pose.position.y = float(r['meas_n'])
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        var = float(r.get('variance', 9.0))
        msg.pose.covariance[0] = var
        msg.pose.covariance[7] = var
        msg.pose.covariance[30] = float(r.get('confidence', 0.0))
        msg.pose.covariance[31] = float(r.get('inliers', 0))
        msg.pose.covariance[32] = float(r.get('inlier_ratio', 0.0))
        msg.pose.covariance[33] = float(r.get('scale', 1.0))
        msg.pose.covariance[34] = float(r.get('rotation_deg', 0.0))
        self.pub_pose.publish(msg)
        self.publish_debug_if_available(r)
        rospy.loginfo('Published file-queue map-match pose: id=%s e=%.2f n=%.2f conf=%.2f inliers=%d matches=%d',
                      r.get('request_id', ''), msg.pose.pose.position.x, msg.pose.pose.position.y,
                      float(r.get('confidence', 0.0)), int(r.get('inliers', 0)), int(r.get('matches', 0)))

    def timer_cb(self, _event):
        results = sorted(glob.glob(str(self.output_dir / 'result_*.json')))
        for p in results:
            self.process_result(p)
            if self.move_processed:
                try:
                    dst = self.done_dir / os.path.basename(p)
                    shutil.move(p, str(dst))
                except Exception:
                    try:
                        os.remove(p)
                    except Exception:
                        pass


if __name__ == '__main__':
    rospy.init_node('eloftr_queue_result_publisher')
    EloftrQueueResultPublisher()
    rospy.spin()
