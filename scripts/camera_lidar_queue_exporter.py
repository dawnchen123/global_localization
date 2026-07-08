#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS-side queue exporter for SAM3 projected semantic mapping.

v25 changes:
  - supports raw sensor_msgs/Image and sensor_msgs/CompressedImage inputs
  - supports two LiDAR input types:
      1) sensor_msgs/PointCloud2 in LiDAR frame
      2) livox_ros_driver/CustomMsg raw Livox packets
  - keeps YAML intrinsics/extrinsics for datasets without /camera_info
  - explicitly distinguishes raw LiDAR points from FAST-LIVO2 /cloud_registered.
  - supports cloud_in_map_frame=true: queue stores /cloud_registered points and
    the projector will transform them back to the LiDAR frame before image projection.

Recommended for the unified-frame i2Nav AT128 pipeline:
  /avt_camera/left/image/compressed + /cloud_registered + /aft_mapped_to_init
  with cloud_in_map_frame=true.
"""

import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from tf.transformations import quaternion_matrix

try:
    from livox_ros_driver.msg import CustomMsg as LivoxCustomMsg
except Exception:
    LivoxCustomMsg = None

try:
    from livox_ros_driver2.msg import CustomMsg as LivoxCustomMsg2
except Exception:
    LivoxCustomMsg2 = None


class CameraLidarQueueExporter(object):
    def __init__(self):
        self.queue_dir = Path(rospy.get_param("~queue_dir", "/tmp/sam3_projected_semantic_queue"))
        self.input_dir = self.queue_dir / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)

        self.image_topic = rospy.get_param("~image_topic", "/avt_camera/left/image/compressed")
        self.image_input_type = rospy.get_param("~image_input_type", "compressed").strip().lower()
        if self.image_input_type not in ["image", "compressed"]:
            raise RuntimeError("Unsupported ~image_input_type=%s. Use image or compressed." % self.image_input_type)
        self.use_camera_info = bool(rospy.get_param("~use_camera_info", False))
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/hesai/at128/points")
        self.odom_topic = rospy.get_param("~odom_topic", "/aft_mapped_to_init")

        # v22: pointcloud2_lidar or livox_custom.
        # pointcloud2_lidar: use a sensor_msgs/PointCloud2 whose points are in the LiDAR frame.
        # livox_custom: legacy livox_ros_driver/CustomMsg raw packets.
        self.lidar_input_type = rospy.get_param("~lidar_input_type", "pointcloud2_lidar").strip().lower()
        if self.lidar_input_type not in ["pointcloud2_lidar", "livox_custom"]:
            raise RuntimeError("Unsupported ~lidar_input_type=%s. Use pointcloud2_lidar or livox_custom." % self.lidar_input_type)
        self.cloud_in_map_frame = bool(rospy.get_param("~cloud_in_map_frame", False))
        if self.cloud_in_map_frame and self.lidar_input_type == "livox_custom":
            raise RuntimeError("~cloud_in_map_frame=true requires sensor_msgs/PointCloud2 input, not livox_custom")

        self.export_rate = float(rospy.get_param("~export_rate", 1.0))
        self.max_pending_requests = int(rospy.get_param("~max_pending_requests", 8))
        self.approx_sync_slop = float(rospy.get_param("~approx_sync_slop", 0.08))
        # v23: Most real bags do not have image, raw LiDAR scan and FAST-LIVO2 odometry
        # timestamps aligned closely enough for a 3-way ApproximateTimeSynchronizer.
        # Default to synchronizing image + LiDAR only and using the latest buffered odometry.
        self.use_latest_odom = bool(rospy.get_param("~use_latest_odom", True))
        self.max_odom_age_sec = float(rospy.get_param("~max_odom_age_sec", 1.0))
        self.odom_match_mode = rospy.get_param("~odom_match_mode", "receipt").strip().lower()
        if self.odom_match_mode not in ["receipt", "latest", "stamp"]:
            raise RuntimeError("Unsupported ~odom_match_mode=%s. Use receipt, latest, or stamp." % self.odom_match_mode)
        self.odom_buffer_size = max(1, int(rospy.get_param("~odom_buffer_size", 500)))
        self.max_odom_receipt_age_sec = float(rospy.get_param("~max_odom_receipt_age_sec", 0.50))
        self.defer_odom_lookup_sec = float(rospy.get_param("~defer_odom_lookup_sec", 0.05))
        self.debug_sync = bool(rospy.get_param("~debug_sync", True))
        self.point_stride = max(1, int(rospy.get_param("~point_stride", 2)))
        self.max_export_points = int(rospy.get_param("~max_export_points", 180000))
        self.min_range_m = float(rospy.get_param("~min_range_m", 0.5))
        self.max_range_m = float(rospy.get_param("~max_range_m", 220.0))
        self.z_min_lidar = float(rospy.get_param("~z_min_lidar", -8.0))
        self.z_max_lidar = float(rospy.get_param("~z_max_lidar", 25.0))

        # Intrinsic fallback for bags without CameraInfo.
        self.cam_width = int(rospy.get_param("~cam_width", 0))
        self.cam_height = int(rospy.get_param("~cam_height", 0))
        self.image_scale = float(rospy.get_param("~image_scale", rospy.get_param("~scale", 1.0)))
        self.cam_fx = float(rospy.get_param("~cam_fx", 0.0))
        self.cam_fy = float(rospy.get_param("~cam_fy", 0.0))
        self.cam_cx = float(rospy.get_param("~cam_cx", 0.0))
        self.cam_cy = float(rospy.get_param("~cam_cy", 0.0))
        self.cam_d0 = float(rospy.get_param("~cam_d0", 0.0))
        self.cam_d1 = float(rospy.get_param("~cam_d1", 0.0))
        self.cam_d2 = float(rospy.get_param("~cam_d2", 0.0))
        self.cam_d3 = float(rospy.get_param("~cam_d3", 0.0))
        self.auto_resize_intrinsics = bool(rospy.get_param("~auto_resize_intrinsics_to_image", True))

        self.bridge = CvBridge()
        self.last_export_time = 0.0
        self.exported = 0

        image_cls = CompressedImage if self.image_input_type == "compressed" else Image
        image_sub = Subscriber(self.image_topic, image_cls)

        # v23: odom is buffered separately by default. This prevents zero export when
        # /aft_mapped_to_init has timestamps that do not align with raw camera/Livox stamps.
        self.latest_odom = None
        self.latest_odom_wall = None
        self.odom_buffer = deque(maxlen=self.odom_buffer_size)
        self.last_image_stamp = None
        self.last_cloud_stamp = None
        self.last_odom_stamp = None
        self.image_seen = 0
        self.cloud_seen = 0
        self.odom_seen = 0
        self.sync_seen = 0
        self.odom_sub_plain = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=200)

        if self.lidar_input_type == "livox_custom":
            lidar_cls = LivoxCustomMsg if LivoxCustomMsg is not None else LivoxCustomMsg2
            if lidar_cls is None:
                raise RuntimeError(
                    "~lidar_input_type=livox_custom but neither livox_ros_driver.msg.CustomMsg "
                    "nor livox_ros_driver2.msg.CustomMsg can be imported. Source the Livox driver workspace first."
                )
            cloud_sub = Subscriber(self.lidar_topic, lidar_cls)
        else:
            cloud_sub = Subscriber(self.lidar_topic, PointCloud2)

        # Lightweight topic counters for diagnosing why no queue files are generated.
        rospy.Subscriber(self.image_topic, image_cls, self.image_count_cb, queue_size=1)
        if self.lidar_input_type == "livox_custom":
            rospy.Subscriber(self.lidar_topic, lidar_cls, self.cloud_count_cb, queue_size=1)
        else:
            rospy.Subscriber(self.lidar_topic, PointCloud2, self.cloud_count_cb, queue_size=1)

        if self.use_latest_odom:
            if self.use_camera_info:
                info_sub = Subscriber(self.camera_info_topic, CameraInfo)
                self.sync = ApproximateTimeSynchronizer(
                    [image_sub, info_sub, cloud_sub],
                    queue_size=60,
                    slop=self.approx_sync_slop,
                    allow_headerless=False,
                )
                self.sync.registerCallback(self.callback_with_info_latest_odom)
            else:
                self.sync = ApproximateTimeSynchronizer(
                    [image_sub, cloud_sub],
                    queue_size=60,
                    slop=self.approx_sync_slop,
                    allow_headerless=False,
                )
                self.sync.registerCallback(self.callback_without_info_latest_odom)
        else:
            odom_sub = Subscriber(self.odom_topic, Odometry)
            if self.use_camera_info:
                info_sub = Subscriber(self.camera_info_topic, CameraInfo)
                self.sync = ApproximateTimeSynchronizer(
                    [image_sub, info_sub, cloud_sub, odom_sub],
                    queue_size=60,
                    slop=self.approx_sync_slop,
                    allow_headerless=False,
                )
                self.sync.registerCallback(self.callback_with_info)
            else:
                self.sync = ApproximateTimeSynchronizer(
                    [image_sub, cloud_sub, odom_sub],
                    queue_size=60,
                    slop=self.approx_sync_slop,
                    allow_headerless=False,
                )
                self.sync.registerCallback(self.callback_without_info)

        rospy.Timer(rospy.Duration(5.0), self.debug_timer_cb)

        rospy.loginfo("v25 camera_lidar_queue_exporter started")
        rospy.loginfo("image=%s image_input_type=%s use_camera_info=%s lidar=%s lidar_input_type=%s cloud_in_map=%s odom=%s queue=%s",
                      self.image_topic, self.image_input_type, str(self.use_camera_info),
                      self.lidar_topic, self.lidar_input_type, str(self.cloud_in_map_frame),
                      self.odom_topic, str(self.queue_dir))
        rospy.loginfo("sync mode: use_latest_odom=%s approx_sync_slop=%.3f max_odom_age=%.3f debug_sync=%s",
                      str(self.use_latest_odom), self.approx_sync_slop, self.max_odom_age_sec, str(self.debug_sync))
        rospy.loginfo("odom match: mode=%s buffer=%d max_receipt_age=%.3f defer_lookup=%.3f",
                      self.odom_match_mode, self.odom_buffer_size,
                      self.max_odom_receipt_age_sec, self.defer_odom_lookup_sec)
        rospy.loginfo("queue backpressure: export_rate=%.3fHz max_pending_requests=%d",
                      self.export_rate, self.max_pending_requests)
        if not self.use_camera_info:
            rospy.loginfo("using YAML intrinsics: width=%d height=%d scale=%.3f fx=%.3f fy=%.3f cx=%.3f cy=%.3f D=[%.5f %.5f %.6f %.6f]",
                          self.cam_width, self.cam_height, self.image_scale,
                          self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy,
                          self.cam_d0, self.cam_d1, self.cam_d2, self.cam_d3)

    def stamp_to_sec(self, stamp):
        try:
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        except Exception:
            return 0.0

    def image_count_cb(self, msg):
        self.image_seen += 1
        self.last_image_stamp = msg.header.stamp

    def cloud_count_cb(self, msg):
        self.cloud_seen += 1
        self.last_cloud_stamp = msg.header.stamp

    def odom_cb(self, msg):
        wall = time.time()
        self.latest_odom = msg
        self.latest_odom_wall = wall
        self.odom_buffer.append((wall, msg))
        self.odom_seen += 1
        self.last_odom_stamp = msg.header.stamp

    def can_export_now(self):
        now = time.time()
        if self.export_rate > 0.0 and now - self.last_export_time < 1.0 / self.export_rate:
            return False
        if self.max_pending_requests > 0:
            pending = len(list(self.input_dir.glob("ready_*.json")))
            if pending >= self.max_pending_requests:
                rospy.logwarn_throttle(
                    5.0,
                    "SAM3 queue backpressure active: pending ready requests=%d >= max_pending_requests=%d. "
                    "Exporter will skip frames until sam3_image_mask_service catches up.",
                    pending, self.max_pending_requests)
                return False
        return True

    def get_latest_odom_for(self, ref_stamp, ref_wall=None):
        if self.latest_odom is None:
            rospy.logwarn_throttle(3.0, "No odometry received yet on %s", self.odom_topic)
            return None, {}

        if ref_wall is None:
            ref_wall = time.time()

        selected = self.latest_odom
        selected_wall = self.latest_odom_wall
        stamp_dt = abs(self.stamp_to_sec(ref_stamp) - self.stamp_to_sec(selected.header.stamp))

        if self.odom_match_mode == "stamp" and self.odom_buffer:
            selected_wall, selected = min(
                self.odom_buffer,
                key=lambda item: abs(self.stamp_to_sec(ref_stamp) - self.stamp_to_sec(item[1].header.stamp)))
            stamp_dt = abs(self.stamp_to_sec(ref_stamp) - self.stamp_to_sec(selected.header.stamp))
        elif self.odom_match_mode == "receipt" and self.odom_buffer:
            latest_allowed = ref_wall + max(0.0, self.defer_odom_lookup_sec)
            candidates = [item for item in self.odom_buffer if item[0] <= latest_allowed]
            if candidates:
                selected_wall, selected = candidates[-1]
            stamp_dt = abs(self.stamp_to_sec(ref_stamp) - self.stamp_to_sec(selected.header.stamp))

        receipt_age = None if selected_wall is None else ref_wall - selected_wall
        diag = {
            "odom_match_mode": self.odom_match_mode,
            "odom_stamp_dt": float(stamp_dt),
            "odom_receipt_age_sec": None if receipt_age is None else float(receipt_age),
            "odom_header_stamp": self.stamp_to_sec(selected.header.stamp),
        }

        # With FAST-LIVO2 publish/use_sensor_time enabled, stamp mode is the
        # safest projection path: image, cloud, and odom all refer to sensor time.
        # Receipt mode is kept only as a fallback for older wall-time odometry.
        if self.odom_match_mode == "stamp" and self.max_odom_age_sec > 0.0 and stamp_dt > self.max_odom_age_sec:
            rospy.logwarn_throttle(3.0,
                "Latest odom too old for image/lidar pair: age=%.3fs > %.3fs. "
                "Increase ~max_odom_age_sec or play bag with /clock correctly.",
                stamp_dt, self.max_odom_age_sec)
            return None, diag

        if (receipt_age is not None and self.max_odom_receipt_age_sec > 0.0 and
                abs(receipt_age) > self.max_odom_receipt_age_sec):
            rospy.logwarn_throttle(
                2.0,
                "Selected odom receipt age is large: mode=%s receipt_age=%.3fs stamp_dt=%.3fs. "
                "This can scramble map-frame SAM3 semantic points.",
                self.odom_match_mode, receipt_age, stamp_dt)

        return selected, diag

    def debug_timer_cb(self, event):
        if not self.debug_sync:
            return
        def ss(st):
            return "none" if st is None else "%.6f" % self.stamp_to_sec(st)
        rospy.loginfo("sync debug: image_seen=%d cloud_seen=%d odom_seen=%d sync_pairs=%d exported=%d stamps image=%s cloud=%s odom=%s slop=%.3f use_latest_odom=%s",
                      self.image_seen, self.cloud_seen, self.odom_seen, self.sync_seen, self.exported,
                      ss(self.last_image_stamp), ss(self.last_cloud_stamp), ss(self.last_odom_stamp),
                      self.approx_sync_slop, str(self.use_latest_odom))

    def callback_with_info_latest_odom(self, image_msg, camera_info_msg, cloud_msg):
        self.sync_seen += 1
        if not self.can_export_now():
            return
        ref_wall = time.time()
        if self.defer_odom_lookup_sec > 0.0:
            time.sleep(self.defer_odom_lookup_sec)
        odom_msg, odom_diag = self.get_latest_odom_for(image_msg.header.stamp, ref_wall)
        if odom_msg is None:
            return
        self.callback_with_info(image_msg, camera_info_msg, cloud_msg, odom_msg, odom_diag=odom_diag)

    def callback_without_info_latest_odom(self, image_msg, cloud_msg):
        self.sync_seen += 1
        if not self.can_export_now():
            return
        ref_wall = time.time()
        if self.defer_odom_lookup_sec > 0.0:
            time.sleep(self.defer_odom_lookup_sec)
        odom_msg, odom_diag = self.get_latest_odom_for(image_msg.header.stamp, ref_wall)
        if odom_msg is None:
            return
        self.callback_without_info(image_msg, cloud_msg, odom_msg, odom_diag=odom_diag)

    @staticmethod
    def odom_to_matrix(odom_msg):
        q = odom_msg.pose.pose.orientation
        t = odom_msg.pose.pose.position
        T = quaternion_matrix([q.x, q.y, q.z, q.w]).astype(np.float64)
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    def yaml_intrinsics_for_image(self, image_width, image_height):
        if self.cam_fx <= 0.0 or self.cam_fy <= 0.0:
            raise RuntimeError("cam_fx/cam_fy must be set when use_camera_info=false")

        fx = self.cam_fx * self.image_scale
        fy = self.cam_fy * self.image_scale
        cx = self.cam_cx * self.image_scale
        cy = self.cam_cy * self.image_scale
        expected_w = int(round(self.cam_width * self.image_scale)) if self.cam_width > 0 else image_width
        expected_h = int(round(self.cam_height * self.image_scale)) if self.cam_height > 0 else image_height

        if self.auto_resize_intrinsics and expected_w > 0 and expected_h > 0:
            sx = float(image_width) / float(expected_w)
            sy = float(image_height) / float(expected_h)
            if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
                rospy.logwarn_throttle(5.0,
                    "image size %dx%d differs from expected %dx%d after scale %.3f; adjusting K by sx=%.4f sy=%.4f",
                    image_width, image_height, expected_w, expected_h, self.image_scale, sx, sy)
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy

        K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.asarray([self.cam_d0, self.cam_d1, self.cam_d2, self.cam_d3], dtype=np.float64)
        return K, D

    def filter_xyz_list(self, iterable, apply_lidar_filters=True):
        pts = []
        idx = 0
        for x, y, z in iterable:
            if idx % self.point_stride != 0:
                idx += 1
                continue
            idx += 1
            x, y, z = float(x), float(y), float(z)
            if not np.isfinite(x + y + z):
                continue
            if apply_lidar_filters:
                r = (x * x + y * y + z * z) ** 0.5
                if r < self.min_range_m or r > self.max_range_m:
                    continue
                if z < self.z_min_lidar or z > self.z_max_lidar:
                    continue
            pts.append((x, y, z))
            if len(pts) >= self.max_export_points:
                break
        if not pts:
            return None
        return np.asarray(pts, dtype=np.float32)

    def pointcloud2_to_xyz(self, cloud_msg):
        fields = [f.name for f in cloud_msg.fields]
        if not all(k in fields for k in ["x", "y", "z"]):
            rospy.logwarn_throttle(5.0, "PointCloud2 lacks x/y/z fields. fields=%s", fields)
            return None
        return self.filter_xyz_list(
            pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True),
            apply_lidar_filters=not self.cloud_in_map_frame)

    def livox_custom_to_xyz(self, msg):
        # livox_ros_driver/CustomMsg points are CustomPoint: offset_time, x, y, z, reflectivity, tag, line.
        if not hasattr(msg, "points"):
            rospy.logwarn_throttle(5.0, "Livox CustomMsg has no points field")
            return None
        return self.filter_xyz_list(((p.x, p.y, p.z) for p in msg.points), apply_lidar_filters=True)

    def cloud_to_xyz(self, cloud_msg):
        if self.lidar_input_type == "livox_custom":
            return self.livox_custom_to_xyz(cloud_msg)
        return self.pointcloud2_to_xyz(cloud_msg)

    def callback_with_info(self, image_msg, camera_info_msg, cloud_msg, odom_msg, odom_diag=None):
        K = np.asarray(camera_info_msg.K, dtype=np.float64).reshape(3, 3)
        D = np.asarray(camera_info_msg.D, dtype=np.float64) if camera_info_msg.D else np.zeros((0,), dtype=np.float64)
        self.handle_sample(image_msg, cloud_msg, odom_msg, K, D,
                           int(camera_info_msg.width), int(camera_info_msg.height),
                           camera_info_msg.header.frame_id, "camera_info", odom_diag=odom_diag)

    def callback_without_info(self, image_msg, cloud_msg, odom_msg, odom_diag=None):
        image_bgr = self.image_to_bgr(image_msg)
        if image_bgr is None:
            return
        h, w = image_bgr.shape[:2]
        K, D = self.yaml_intrinsics_for_image(w, h)
        self.handle_sample(image_msg, cloud_msg, odom_msg, K, D, w, h,
                           image_msg.header.frame_id, "yaml", image_bgr=image_bgr,
                           odom_diag=odom_diag)

    def image_to_bgr(self, image_msg):
        try:
            if self.image_input_type == "compressed":
                arr = np.frombuffer(image_msg.data, dtype=np.uint8)
                image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError("cv2.imdecode returned None")
                return image_bgr
            return self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("Failed to convert image from %s: %s", self.image_input_type, e)
            return None

    def handle_sample(self, image_msg, cloud_msg, odom_msg, K, D, image_width, image_height,
                      camera_frame_id, intrinsic_source, image_bgr=None, odom_diag=None):
        now = time.time()
        if self.export_rate > 0.0 and now - self.last_export_time < 1.0 / self.export_rate:
            return
        if self.max_pending_requests > 0:
            pending = len(list(self.input_dir.glob("ready_*.json")))
            if pending >= self.max_pending_requests:
                rospy.logwarn_throttle(
                    5.0,
                    "SAM3 queue backpressure active: pending ready requests=%d >= max_pending_requests=%d. "
                    "Exporter will skip frames until sam3_image_mask_service catches up.",
                    pending, self.max_pending_requests)
                return
        self.last_export_time = now

        if image_bgr is None:
            image_bgr = self.image_to_bgr(image_msg)
            if image_bgr is None:
                return

        xyz = self.cloud_to_xyz(cloud_msg)
        if xyz is None or xyz.shape[0] < 100:
            rospy.logwarn_throttle(3.0, "Too few valid LiDAR points for semantic queue")
            return

        T_map_body = self.odom_to_matrix(odom_msg)

        stamp = image_msg.header.stamp.to_nsec()
        req_id = str(stamp)
        image_path = self.input_dir / ("image_%s.png" % req_id)
        cloud_path = self.input_dir / ("cloud_%s.npz" % req_id)
        meta_tmp = self.input_dir / ("ready_%s.json.tmp" % req_id)
        meta_path = self.input_dir / ("ready_%s.json" % req_id)

        cv2.imwrite(str(image_path), image_bgr)
        np.savez_compressed(
            str(cloud_path),
            xyz=xyz,
            K=K,
            D=D,
            lidar_input_type=np.asarray(self.lidar_input_type),
            cloud_in_map_frame=np.asarray(self.cloud_in_map_frame),
        )

        meta = {
            "id": req_id,
            "stamp": {"secs": image_msg.header.stamp.secs, "nsecs": image_msg.header.stamp.nsecs},
            "image_path": str(image_path),
            "cloud_path": str(cloud_path),
            "camera_width": int(image_width),
            "camera_height": int(image_height),
            "image_width": int(image_bgr.shape[1]),
            "image_height": int(image_bgr.shape[0]),
            "camera_frame_id": camera_frame_id,
            "intrinsic_source": intrinsic_source,
            "K": K.reshape(-1).tolist(),
            "D": D.reshape(-1).tolist(),
            "lidar_frame_id": getattr(cloud_msg.header, "frame_id", ""),
            "lidar_input_type": self.lidar_input_type,
            "cloud_in_map_frame": self.cloud_in_map_frame,
            "odom_frame_id": odom_msg.header.frame_id,
            "odom_child_frame_id": odom_msg.child_frame_id,
            "odom_diag": odom_diag or {},
            "image_cloud_stamp_dt": abs(self.stamp_to_sec(image_msg.header.stamp) - self.stamp_to_sec(cloud_msg.header.stamp)),
            "T_map_body": T_map_body.reshape(-1).tolist(),
        }

        with open(str(meta_tmp), "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(str(meta_tmp), str(meta_path))

        self.exported += 1
        diag = odom_diag or {}
        rospy.loginfo("exported SAM3 semantic request id=%s points=%d image=%dx%d Kfx=%.2f Kfy=%.2f lidar_type=%s cloud_in_map=%s frame=%s odom_mode=%s odom_receipt_age=%s odom_stamp_dt=%.3f total=%d",
                      req_id, xyz.shape[0], image_bgr.shape[1], image_bgr.shape[0],
                      K[0, 0], K[1, 1], self.lidar_input_type,
                      str(self.cloud_in_map_frame),
                      getattr(cloud_msg.header, "frame_id", ""),
                      diag.get("odom_match_mode", "direct"),
                      "none" if diag.get("odom_receipt_age_sec", None) is None else "%.3f" % diag.get("odom_receipt_age_sec"),
                      float(diag.get("odom_stamp_dt", 0.0)),
                      self.exported)


if __name__ == "__main__":
    rospy.init_node("camera_lidar_queue_exporter")
    node = CameraLidarQueueExporter()
    rospy.spin()
