#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Noetic / Python3.8 side file-queue exporter.

It exports the structured matching inputs used by the global localization service:
  - local BEV structure layers from FAST-LIVO2 point clouds
  - satellite/geographic crop around the current EKF state
  - optional local semantic label/confidence maps from local_geometric_semantic_mapper.py

Inputs by default:
  /local_bev/match_image        -> bev_<id>.png          (mono, structure)
  /satellite_map/match_image    -> sat_<id>.png          (mono, structure)
  /local_geometric_map/label    -> local_geom_<id>.png   (mono8 simplified geometry)
  /local_rgb_bev/image          -> local_sam3_<id>.png   (BGR image for optional SAM3)
  /local_semantic_map/label     -> local_label_<id>.png  (mono8 semantic labels)
  /local_semantic_map/confidence-> local_conf_<id>.png   (mono8 semantic confidence)
  /satellite_map/aligned_crop   -> sat_rgb_<id>.png      (BGR crop, fallback prior source)
  /map_match/input_meta         -> ready_<id>.json
"""
import json
import os
import time
import glob
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray


class EloftrQueueExporter:
    def __init__(self):
        self.bev_topic = rospy.get_param('~bev_topic', '/local_bev/match_image')
        self.sat_topic = rospy.get_param('~satellite_crop_topic', '/satellite_map/match_image')
        self.meta_topic = rospy.get_param('~meta_topic', '/map_match/input_meta')

        # Structured local semantics. If unavailable, bev_sem_path remains available as
        # a visual fallback for legacy semantic services.
        self.bev_semantic_topic = rospy.get_param('~bev_semantic_topic', '/local_bev/image')
        self.local_label_topic = rospy.get_param('~local_label_topic', '/local_semantic_map/label')
        self.local_confidence_topic = rospy.get_param('~local_confidence_topic', '/local_semantic_map/confidence')
        self.local_geometric_label_topic = rospy.get_param('~local_geometric_label_topic', '/local_geometric_map/label')
        self.local_geometric_confidence_topic = rospy.get_param('~local_geometric_confidence_topic', '/local_geometric_map/confidence')
        self.sam3_local_image_topic = rospy.get_param('~sam3_local_image_topic', '/local_rgb_bev/image')
        self.sat_rgb_topic = rospy.get_param('~satellite_rgb_topic', '/satellite_map/aligned_crop')

        self.queue_dir = Path(rospy.get_param('~queue_dir', '/tmp/eloftr_queue'))
        self.export_rate = float(rospy.get_param('~export_rate', 0.05))
        self.min_seconds_between_requests = float(rospy.get_param('~min_seconds_between_requests', 10.0))
        self.max_pending_requests = int(rospy.get_param('~max_pending_requests', 2))
        self.max_image_dim_to_save = int(rospy.get_param('~max_image_dim_to_save', 0))  # 0 = no resize
        self.overwrite_old_pending = bool(rospy.get_param('~overwrite_old_pending', True))
        self.require_new_meta = bool(rospy.get_param('~require_new_meta', False))
        self.export_semantic_sources = bool(rospy.get_param('~export_semantic_sources', True))

        self.input_dir = self.queue_dir / 'input'
        self.output_dir = self.queue_dir / 'output'
        self.done_dir = self.queue_dir / 'done'
        for d in [self.input_dir, self.output_dir, self.done_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.bev_img = None
        self.sat_img = None
        self.bev_sem_img = None
        self.local_label_img = None
        self.local_conf_img = None
        self.local_geom_img = None
        self.local_geom_conf_img = None
        self.sam3_local_img = None
        self.sat_rgb_img = None
        self.meta = None
        self.last_meta_stamp = None
        self.last_exported_meta_stamp = None
        self.last_export_wall = 0.0

        self.sub_bev = rospy.Subscriber(self.bev_topic, Image, self.bev_cb, queue_size=1)
        self.sub_sat = rospy.Subscriber(self.sat_topic, Image, self.sat_cb, queue_size=1)
        self.sub_meta = rospy.Subscriber(self.meta_topic, Float64MultiArray, self.meta_cb, queue_size=2)
        if self.export_semantic_sources:
            self.sub_bev_sem = rospy.Subscriber(self.bev_semantic_topic, Image, self.bev_sem_cb, queue_size=1)
            self.sub_local_label = rospy.Subscriber(self.local_label_topic, Image, self.local_label_cb, queue_size=1)
            self.sub_local_conf = rospy.Subscriber(self.local_confidence_topic, Image, self.local_conf_cb, queue_size=1)
            self.sub_local_geom = rospy.Subscriber(self.local_geometric_label_topic, Image, self.local_geom_cb, queue_size=1)
            self.sub_local_geom_conf = rospy.Subscriber(self.local_geometric_confidence_topic, Image, self.local_geom_conf_cb, queue_size=1)
            self.sub_sam3_local = rospy.Subscriber(self.sam3_local_image_topic, Image, self.sam3_local_cb, queue_size=1)
            self.sub_sat_rgb = rospy.Subscriber(self.sat_rgb_topic, Image, self.sat_rgb_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(0.001, self.export_rate)), self.timer_cb)
        rospy.loginfo('queue_exporter structured-prior started. bev=%s sat=%s local_label=%s sat_rgb=%s meta=%s queue=%s',
                      self.bev_topic, self.sat_topic, self.local_label_topic, self.sat_rgb_topic,
                      self.meta_topic, str(self.queue_dir))

    @staticmethod
    def to_mono_u8(img):
        if img is None:
            return None
        if len(img.shape) == 2:
            gray = img
        elif img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img[:, :, 0]
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    @staticmethod
    def to_bgr_u8(img):
        if img is None:
            return None
        if len(img.shape) == 2:
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            out = img
        elif img.shape[2] == 4:
            out = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            out = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if out.dtype != np.uint8:
            out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return out

    def maybe_resize(self, img):
        if self.max_image_dim_to_save <= 0:
            return img
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= self.max_image_dim_to_save:
            return img
        scale = float(self.max_image_dim_to_save) / float(m)
        nw = max(32, int(round(w * scale)))
        nh = max(32, int(round(h * scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def bev_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.bev_img = self.to_mono_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'BEV export conversion failed: %s', str(exc))

    def sat_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.sat_img = self.to_mono_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Satellite export conversion failed: %s', str(exc))

    def bev_sem_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.bev_sem_img = self.to_bgr_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'BEV semantic export conversion failed: %s', str(exc))

    def local_label_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.local_label_img = self.to_mono_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Local label export conversion failed: %s', str(exc))

    def local_conf_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.local_conf_img = self.to_mono_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Local confidence export conversion failed: %s', str(exc))

    def local_geom_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.local_geom_img = self.to_mono_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Local geometric label export conversion failed: %s', str(exc))

    def local_geom_conf_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.local_geom_conf_img = self.to_mono_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Local geometric confidence export conversion failed: %s', str(exc))

    def sam3_local_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.sam3_local_img = self.to_bgr_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'SAM3 local image export conversion failed: %s', str(exc))

    def sat_rgb_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.sat_rgb_img = self.to_bgr_u8(img).copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Satellite RGB export conversion failed: %s', str(exc))

    def meta_cb(self, msg):
        if len(msg.data) < 18:
            rospy.logwarn_throttle(2.0, 'Ignore input_meta with length %d < 18', len(msg.data))
            return
        self.meta = list(msg.data)
        self.last_meta_stamp = rospy.Time.now().to_sec()

    def prune_pending(self):
        ready = sorted(glob.glob(str(self.input_dir / 'ready_*.json')))
        if len(ready) <= self.max_pending_requests:
            return
        if not self.overwrite_old_pending:
            return
        for rp in ready[:-self.max_pending_requests]:
            try:
                with open(rp, 'r') as f:
                    req = json.load(f)
                for key in ['bev_path', 'sat_path', 'bev_sem_path', 'sat_rgb_path',
                            'local_label_path', 'local_confidence_path',
                            'local_geometric_label_path', 'local_geometric_confidence_path',
                            'local_sam3_path']:
                    p = req.get(key, '')
                    if p and os.path.exists(p):
                        os.remove(p)
                os.rename(rp, str(self.done_dir / ('dropped_' + os.path.basename(rp))))
            except Exception:
                try:
                    os.remove(rp)
                except Exception:
                    pass

    def atomic_write_json(self, path, data):
        tmp = str(path) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(path))

    @staticmethod
    def write_png_atomic(path, img):
        tmp = str(path) + '.tmp.png'
        cv2.imwrite(tmp, img)
        os.replace(tmp, str(path))

    def timer_cb(self, _event):
        now = time.time()
        if now - self.last_export_wall < self.min_seconds_between_requests:
            return
        if self.bev_img is None or self.sat_img is None or self.meta is None:
            rospy.logwarn_throttle(10.0, 'Queue exporter waiting inputs: bev=%s sat=%s meta=%s',
                                   self.bev_img is not None, self.sat_img is not None, self.meta is not None)
            return
        if self.export_semantic_sources and self.sat_rgb_img is None:
            rospy.logwarn_throttle(10.0, 'Queue exporter waiting satellite RGB crop')
            return
        if self.require_new_meta and self.last_meta_stamp == self.last_exported_meta_stamp:
            return
        self.prune_pending()
        if len(glob.glob(str(self.input_dir / 'ready_*.json'))) >= self.max_pending_requests and not self.overwrite_old_pending:
            rospy.logwarn_throttle(10.0, 'Queue exporter skip: too many pending requests')
            return

        bev = self.maybe_resize(self.bev_img.copy())
        sat = self.maybe_resize(self.sat_img.copy())
        if bev.size == 0 or sat.size == 0:
            return
        bev_sem = self.maybe_resize(self.bev_sem_img.copy()) if self.bev_sem_img is not None else None
        local_label = self.maybe_resize(self.local_label_img.copy()) if self.local_label_img is not None else None
        local_conf = self.maybe_resize(self.local_conf_img.copy()) if self.local_conf_img is not None else None
        local_geom = self.maybe_resize(self.local_geom_img.copy()) if self.local_geom_img is not None else None
        local_geom_conf = self.maybe_resize(self.local_geom_conf_img.copy()) if self.local_geom_conf_img is not None else None
        sam3_local = self.maybe_resize(self.sam3_local_img.copy()) if self.sam3_local_img is not None else None
        sat_rgb = self.maybe_resize(self.sat_rgb_img.copy()) if self.sat_rgb_img is not None else None

        rid = str(int(rospy.Time.now().to_nsec()))
        bev_path = self.input_dir / ('bev_%s.png' % rid)
        sat_path = self.input_dir / ('sat_%s.png' % rid)
        bev_sem_path = self.input_dir / ('bev_sem_%s.png' % rid)
        local_label_path = self.input_dir / ('local_label_%s.png' % rid)
        local_conf_path = self.input_dir / ('local_conf_%s.png' % rid)
        local_geom_path = self.input_dir / ('local_geom_%s.png' % rid)
        local_geom_conf_path = self.input_dir / ('local_geom_conf_%s.png' % rid)
        sam3_local_path = self.input_dir / ('local_sam3_%s.png' % rid)
        sat_rgb_path = self.input_dir / ('sat_rgb_%s.png' % rid)
        req_path = self.input_dir / ('ready_%s.json' % rid)
        try:
            self.write_png_atomic(bev_path, bev)
            self.write_png_atomic(sat_path, sat)
            req = {
                'request_id': rid,
                'stamp': rospy.Time.now().to_sec(),
                'bev_path': str(bev_path),
                'sat_path': str(sat_path),
                'meta': self.meta,
                'bev_shape': [int(bev.shape[0]), int(bev.shape[1])],
                'sat_shape': [int(sat.shape[0]), int(sat.shape[1])],
                'meta_layout': [
                    'abs_min_e','abs_max_e','abs_min_n','abs_max_n',
                    'bev_min_e','bev_max_e','bev_min_n','bev_max_n',
                    'robot_u_in_bev_px','robot_v_in_bev_px',
                    'bev_cols','bev_rows','sat_crop_cols','sat_crop_rows',
                    'state_e','state_n','state_yaw','sat_mpp'
                ]
            }
            if bev_sem is not None:
                self.write_png_atomic(bev_sem_path, bev_sem)
                req['bev_sem_path'] = str(bev_sem_path)
                req['bev_sem_shape'] = [int(bev_sem.shape[0]), int(bev_sem.shape[1])]
            if local_label is not None:
                self.write_png_atomic(local_label_path, local_label)
                req['local_label_path'] = str(local_label_path)
                req['local_label_shape'] = [int(local_label.shape[0]), int(local_label.shape[1])]
            if local_conf is not None:
                self.write_png_atomic(local_conf_path, local_conf)
                req['local_confidence_path'] = str(local_conf_path)
                req['local_confidence_shape'] = [int(local_conf.shape[0]), int(local_conf.shape[1])]
            if local_geom is not None:
                self.write_png_atomic(local_geom_path, local_geom)
                req['local_geometric_label_path'] = str(local_geom_path)
                req['local_geometric_label_shape'] = [int(local_geom.shape[0]), int(local_geom.shape[1])]
            if local_geom_conf is not None:
                self.write_png_atomic(local_geom_conf_path, local_geom_conf)
                req['local_geometric_confidence_path'] = str(local_geom_conf_path)
                req['local_geometric_confidence_shape'] = [int(local_geom_conf.shape[0]), int(local_geom_conf.shape[1])]
            if sam3_local is not None:
                self.write_png_atomic(sam3_local_path, sam3_local)
                req['local_sam3_path'] = str(sam3_local_path)
                req['local_sam3_shape'] = [int(sam3_local.shape[0]), int(sam3_local.shape[1])]
            if sat_rgb is not None:
                self.write_png_atomic(sat_rgb_path, sat_rgb)
                req['sat_rgb_path'] = str(sat_rgb_path)
                req['sat_rgb_shape'] = [int(sat_rgb.shape[0]), int(sat_rgb.shape[1])]
            self.atomic_write_json(req_path, req)
            self.last_export_wall = now
            self.last_exported_meta_stamp = self.last_meta_stamp
            rospy.loginfo('Exported request %s bev=%s sat=%s semantic=%s', rid, bev.shape, sat.shape, self.export_semantic_sources)
        except Exception as exc:
            rospy.logerr_throttle(5.0, 'Queue export failed: %s', str(exc))


if __name__ == '__main__':
    rospy.init_node('eloftr_queue_exporter')
    EloftrQueueExporter()
    rospy.spin()
