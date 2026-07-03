#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficient-LoFTR ROS bridge for BEV-to-satellite matching.

Inputs:
  /local_bev/match_image          sensor_msgs/Image, BEV edge/structure image
  /satellite_map/aligned_crop     sensor_msgs/Image, satellite crop aligned by current pose and BEV extent
  /map_match/input_meta           std_msgs/Float64MultiArray, georeference metadata from C++ node

Output:
  /map_match_pose_external        geometry_msgs/PoseWithCovarianceStamped
  /map_match/efficient_loftr_debug sensor_msgs/Image

Install Efficient-LoFTR separately, e.g. the official zju3dv repository, and set:
  _efficient_loftr_root:=/path/to/EfficientLoFTR
  _weights_path:=/path/to/weights/eloftr_outdoor.ckpt
"""
import os
import sys
import math
import threading
import gc

import rospy
import cv2
import numpy as np
# Compatibility patch for old onnx/kornia dependencies with NumPy >= 1.24.
# It must be executed before importing Efficient-LoFTR and its transitive dependencies.
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped

# Limit BLAS/OpenMP thread usage before importing torch. These limits avoid full-system freezes on CPU inference.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
try:
    import torch
except Exception:
    torch = None


class EfficientLoFTRBridge:
    def __init__(self):
        self.bev_topic = rospy.get_param('~bev_topic', '/local_bev/match_image')
        self.sat_topic = rospy.get_param('~satellite_crop_topic', '/satellite_map/match_image')
        self.meta_topic = rospy.get_param('~meta_topic', '/map_match/input_meta')
        self.output_topic = rospy.get_param('~output_topic', '/map_match_pose_external')
        self.debug_topic = rospy.get_param('~debug_topic', '/map_match/efficient_loftr_debug')
        self.root = rospy.get_param('~efficient_loftr_root', '')
        self.weights_path = rospy.get_param('~weights_path', '')
        self.device = rospy.get_param('~device', 'cuda')
        self.match_rate = float(rospy.get_param('~match_rate', 0.02))  # default: one match every 50s in safe mode
        # Backward-compatible aliases: older answers used max_image_size, while the node used max_image_dim.
        if rospy.has_param('~max_image_dim'):
            self.max_image_dim = int(rospy.get_param('~max_image_dim'))
        else:
            self.max_image_dim = int(rospy.get_param('~max_image_size', 320))
        self.torch_num_threads = int(rospy.get_param('~torch_num_threads', 2))
        self.publish_debug_image = bool(rospy.get_param('~publish_debug_image', False))
        self.drop_if_busy = bool(rospy.get_param('~drop_if_busy', True))
        self.min_seconds_between_matches = float(rospy.get_param('~min_seconds_between_matches', 30.0))
        self.min_matches = int(rospy.get_param('~min_matches', 30))
        self.min_inliers = int(rospy.get_param('~min_inliers', 20))
        self.min_confidence = float(rospy.get_param('~min_confidence', 0.35))
        self.min_inlier_ratio = float(rospy.get_param('~min_inlier_ratio', 0.20))
        self.ransac_thresh_px = float(rospy.get_param('~ransac_thresh_px', 4.0))
        self.max_rotation_deg = float(rospy.get_param('~max_rotation_deg', 25.0))
        self.min_scale = float(rospy.get_param('~min_scale', 0.70))
        self.max_scale = float(rospy.get_param('~max_scale', 1.30))
        self.match_noise_std = float(rospy.get_param('~match_noise_std', 1.5))
        self.use_opencv_fallback = bool(rospy.get_param('~use_opencv_fallback', False))
        if torch is not None:
            try:
                torch.set_num_threads(max(1, self.torch_num_threads))
                torch.set_num_interop_threads(1)
                rospy.loginfo('Torch CPU threads limited: num_threads=%d, interop=1', max(1, self.torch_num_threads))
            except Exception as exc:
                rospy.logwarn('Failed to limit torch threads: %s', str(exc))

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.bev_img = None
        self.sat_img = None
        self.meta = None
        self.busy = False
        self.last_match_wall_time = 0.0
        self.pub_pose = rospy.Publisher(self.output_topic, PoseWithCovarianceStamped, queue_size=5)
        self.pub_debug = rospy.Publisher(self.debug_topic, Image, queue_size=3)
        self.sub_bev = rospy.Subscriber(self.bev_topic, Image, self.bev_cb, queue_size=2)
        self.sub_sat = rospy.Subscriber(self.sat_topic, Image, self.sat_cb, queue_size=2)
        self.sub_meta = rospy.Subscriber(self.meta_topic, Float64MultiArray, self.meta_cb, queue_size=5)
        self.matcher = None
        self._load_matcher()
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(0.001, self.match_rate)), self.timer_cb)
        rospy.loginfo('efficient_loftr_bridge started. bev=%s sat=%s meta=%s out=%s',
                      self.bev_topic, self.sat_topic, self.meta_topic, self.output_topic)

    def _load_matcher(self):
        if not self.root or not self.weights_path:
            rospy.logwarn('Efficient-LoFTR root/weights not set. Set _efficient_loftr_root and _weights_path. Fallback=%s', self.use_opencv_fallback)
            return
        if torch is None:
            rospy.logerr('PyTorch is not available in this Python environment.')
            return
        if self.device == 'cuda' and not torch.cuda.is_available():
            rospy.logwarn('CUDA requested but unavailable. Falling back to CPU.')
            self.device = 'cpu'
        sys.path.insert(0, self.root)
        try:
            from copy import deepcopy
            from src.loftr import LoFTR, full_default_cfg, reparameter
            cfg = deepcopy(full_default_cfg)
            self.matcher = LoFTR(config=cfg)
            state = torch.load(self.weights_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.matcher.load_state_dict(state)
            self.matcher = reparameter(self.matcher)
            self.matcher = self.matcher.eval().to(self.device)
            rospy.loginfo('Efficient-LoFTR loaded from %s with weights %s on %s', self.root, self.weights_path, self.device)
        except Exception as exc:
            self.matcher = None
            rospy.logerr('Failed to load Efficient-LoFTR: %s', str(exc))

    def bev_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            with self.lock:
                self.bev_img = img.copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'BEV image conversion failed: %s', str(exc))

    def sat_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if img is None:
                return
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
            with self.lock:
                self.sat_img = gray.copy()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Satellite crop conversion failed: %s', str(exc))

    def meta_cb(self, msg):
        if len(msg.data) < 18:
            return
        with self.lock:
            self.meta = list(msg.data)

    @staticmethod
    def _resize_for_network(img, max_dim):
        h, w = img.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        new_w = max(32, int((w * scale) // 32 * 32))
        new_h = max(32, int((h * scale) // 32 * 32))
        if new_w != w or new_h != h:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = img
        sx = float(w) / float(new_w)
        sy = float(h) / float(new_h)
        return resized, sx, sy

    def _match_eloftr(self, img0, img1):
        if self.matcher is None:
            return None, None, None
        i0, sx0, sy0 = self._resize_for_network(img0, self.max_image_dim)
        i1, sx1, sy1 = self._resize_for_network(img1, self.max_image_dim)
        ten0 = torch.from_numpy(i0)[None][None].float().to(self.device) / 255.0
        ten1 = torch.from_numpy(i1)[None][None].float().to(self.device) / 255.0
        batch = {'image0': ten0, 'image1': ten1}
        try:
            with torch.inference_mode():
                self.matcher(batch)
            mk0 = batch['mkpts0_f'].detach().cpu().numpy()
            mk1 = batch['mkpts1_f'].detach().cpu().numpy()
            conf = batch['mconf'].detach().cpu().numpy()
        finally:
            # Release large tensors promptly; this matters for CPU mode and live ROS streams.
            try:
                del ten0, ten1
            except Exception:
                pass
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        if mk0.size == 0:
            return None, None, None
        mk0[:, 0] *= sx0; mk0[:, 1] *= sy0
        mk1[:, 0] *= sx1; mk1[:, 1] *= sy1
        return mk0.astype(np.float32), mk1.astype(np.float32), conf.astype(np.float32)

    def _match_opencv_fallback(self, img0, img1):
        orb = cv2.ORB_create(2000)
        k0, d0 = orb.detectAndCompute(img0, None)
        k1, d1 = orb.detectAndCompute(img1, None)
        if d0 is None or d1 is None or len(k0) < 10 or len(k1) < 10:
            return None, None, None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)
        knn = bf.knnMatch(d0, d1, k=2)
        pts0, pts1, conf = [], [], []
        for m in knn:
            if len(m) >= 2 and m[0].distance < 0.78 * m[1].distance:
                pts0.append(k0[m[0].queryIdx].pt)
                pts1.append(k1[m[0].trainIdx].pt)
                conf.append(1.0 - min(1.0, m[0].distance / 80.0))
        if len(pts0) == 0:
            return None, None, None
        return np.asarray(pts0, np.float32), np.asarray(pts1, np.float32), np.asarray(conf, np.float32)

    def timer_cb(self, _event):
        now_wall = rospy.Time.now().to_sec()
        if self.drop_if_busy and self.busy:
            rospy.logwarn_throttle(10.0, 'Skip LoFTR: previous inference is still running.')
            return
        if now_wall - self.last_match_wall_time < self.min_seconds_between_matches:
            return
        self.busy = True
        self.last_match_wall_time = now_wall
        try:
            with self.lock:
                if self.bev_img is None or self.sat_img is None or self.meta is None:
                    return
                bev = self.bev_img.copy()
                sat = self.sat_img.copy()
                meta = list(self.meta)
            if bev.size == 0 or sat.size == 0:
                return
            if sat.shape[:2] != bev.shape[:2]:
                sat = cv2.resize(sat, (bev.shape[1], bev.shape[0]), interpolation=cv2.INTER_LINEAR)
            bev_eq = cv2.equalizeHist(bev)
            sat_eq = cv2.equalizeHist(sat)

            if self.matcher is not None:
                pts0, pts1, conf = self._match_eloftr(bev_eq, sat_eq)
                src_name = 'Efficient-LoFTR'
            elif self.use_opencv_fallback:
                pts0, pts1, conf = self._match_opencv_fallback(bev_eq, sat_eq)
                src_name = 'OpenCV fallback'
            else:
                rospy.logwarn_throttle(5.0, 'Efficient-LoFTR is not loaded and fallback is disabled.')
                return
            if pts0 is None or len(pts0) < self.min_matches:
                rospy.logwarn_throttle(2.0, '%s matches too few: %d < %d', src_name, 0 if pts0 is None else len(pts0), self.min_matches)
                return
            A, mask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC,
                                                 ransacReprojThreshold=self.ransac_thresh_px,
                                                 maxIters=1500, confidence=0.995, refineIters=5)
            if A is None or mask is None:
                return
            mask = mask.reshape(-1).astype(bool)
            inliers = int(mask.sum())
            inlier_ratio = float(inliers) / max(1, len(pts0))
            mean_conf = float(np.mean(conf[mask])) if inliers > 0 else 0.0
            if inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio or mean_conf < self.min_confidence:
                rospy.logwarn_throttle(2.0, 'Reject %s: weak geometry conf=%.2f ratio=%.2f inliers=%d matches=%d',
                                       src_name, mean_conf, inlier_ratio, inliers, len(pts0))
                self._publish_debug(bev_eq, sat_eq, pts0, pts1, mask)
                return
            a = float(A[0,0]); b = float(A[1,0])
            scale = math.sqrt(a*a + b*b)
            rot = math.degrees(math.atan2(b, a))
            if scale < self.min_scale or scale > self.max_scale or abs(rot) > self.max_rotation_deg:
                rospy.logwarn_throttle(2.0, 'Reject %s: affine out of bounds scale=%.3f rot=%.1fdeg', src_name, scale, rot)
                self._publish_debug(bev_eq, sat_eq, pts0, pts1, mask)
                return

            abs_min_e, abs_max_e, abs_min_n, abs_max_n = meta[0], meta[1], meta[2], meta[3]
            robot_u, robot_v = meta[8], meta[9]
            p = np.asarray([robot_u, robot_v, 1.0], dtype=np.float64)
            q = A.dot(p)
            sat_u = float(np.clip(q[0], 0, bev.shape[1]-1))
            sat_v = float(np.clip(q[1], 0, bev.shape[0]-1))
            meas_e = abs_min_e + sat_u / max(1.0, bev.shape[1]) * (abs_max_e - abs_min_e)
            meas_n = abs_max_n - sat_v / max(1.0, bev.shape[0]) * (abs_max_n - abs_min_n)
            var = (self.match_noise_std / max(0.15, mean_conf)) ** 2

            msg = PoseWithCovarianceStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'earth'
            msg.pose.pose.position.x = meas_e
            msg.pose.pose.position.y = meas_n
            msg.pose.pose.orientation.w = 1.0
            msg.pose.covariance[0] = var
            msg.pose.covariance[7] = var
            msg.pose.covariance[30] = mean_conf
            msg.pose.covariance[31] = float(inliers)
            msg.pose.covariance[32] = inlier_ratio
            msg.pose.covariance[33] = scale
            msg.pose.covariance[34] = rot
            self.pub_pose.publish(msg)
            rospy.loginfo('%s pose published: e=%.2f n=%.2f conf=%.2f ratio=%.2f inliers=%d matches=%d scale=%.3f rot=%.1fdeg',
                          src_name, meas_e, meas_n, mean_conf, inlier_ratio, inliers, len(pts0), scale, rot)
            self._publish_debug(bev_eq, sat_eq, pts0, pts1, mask)
        except Exception as exc:
            rospy.logerr_throttle(5.0, 'LoFTR timer callback failed: %s', str(exc))
        finally:
            self.busy = False
            try:
                del bev, sat, bev_eq, sat_eq
            except Exception:
                pass
            if torch is not None and self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _publish_debug(self, img0, img1, pts0, pts1, mask):
        if not self.publish_debug_image:
            return
        try:
            h = max(img0.shape[0], img1.shape[0])
            w0, w1 = img0.shape[1], img1.shape[1]
            canvas = np.ones((h, w0+w1, 3), dtype=np.uint8) * 255
            canvas[:img0.shape[0], :w0] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
            canvas[:img1.shape[0], w0:w0+w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            idxs = np.where(mask)[0]
            if len(idxs) > 120:
                idxs = np.random.choice(idxs, 120, replace=False)
            for i in idxs:
                p0 = tuple(np.round(pts0[i]).astype(int))
                p1 = tuple(np.round(pts1[i] + np.array([w0, 0])).astype(int))
                color = (0, 180, 0)
                cv2.circle(canvas, p0, 2, color, -1)
                cv2.circle(canvas, p1, 2, color, -1)
                cv2.line(canvas, p0, p1, color, 1)
            out = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
            out.header.stamp = rospy.Time.now()
            self.pub_debug.publish(out)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'debug image failed: %s', str(exc))


if __name__ == '__main__':
    rospy.init_node('efficient_loftr_bridge')
    EfficientLoFTRBridge()
    rospy.spin()
