#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS-side queue exporter for SAM3 projected semantic mapping.

v32 changes:
  - supports raw sensor_msgs/Image and sensor_msgs/CompressedImage inputs
  - supports two LiDAR input types:
      1) sensor_msgs/PointCloud2 in LiDAR frame
      2) livox_ros_driver/CustomMsg raw Livox packets
  - keeps YAML intrinsics/extrinsics for datasets without /camera_info
  - explicitly distinguishes raw LiDAR points from FAST-LIVO2 /cloud_registered.
  - supports cloud_in_map_frame=true: queue stores /cloud_registered points and
    the projector will transform them back to the LiDAR frame before image projection.
  - applies the export-rate gate before waiting for LIO odometry.  This keeps
    delayed-pose buffering at the requested semantic rate instead of retaining
    every raw 10 Hz camera/LiDAR pair while the frontend catches up.

Recommended for the unified-frame i2Nav AT128 pipeline:
  /avt_camera/left/image/compressed + /cloud_registered + /aft_mapped_to_init
  with cloud_in_map_frame=true.

The ROS subscriber callbacks intentionally only enqueue synchronized message
objects.  JPEG decoding, PointCloud2 iteration, NumPy compression, and disk
I/O run in one foreground loop.  Calling OpenCV from concurrent rospy and
message_filters callback threads was the source of native ``SIGSEGV`` failures
on long AT128 replays.
"""

import faulthandler
import json
import os
import threading
import sys
import time
import uuid
from collections import deque
from pathlib import Path


def prefer_ros_binary_modules():
    """Use ROS Noetic's coherent distro NumPy/OpenCV binary pair."""
    system_dist = "/usr/lib/python3/dist-packages"
    if not os.path.isdir(system_dist):
        return
    home_local = os.path.join(os.path.expanduser("~"), ".local", "lib", "python")
    filtered = []
    for entry in sys.path:
        normalized = os.path.abspath(entry or ".")
        if normalized.startswith(home_local) or normalized.startswith("/usr/local/lib/python"):
            continue
        filtered.append(entry)
    sys.path[:] = [entry for entry in filtered if entry != system_dist]
    sys.path.insert(0, system_dist)


prefer_ros_binary_modules()

import cv2
import numpy as np
import rospy
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
        self.session_path = self.queue_dir / "queue_session.json"
        self.stale_dir = self.queue_dir / "stale_sessions"
        self.stale_dir.mkdir(parents=True, exist_ok=True)
        self.queue_session_id = self.start_queue_session()
        self.request_sequence = 0
        self.opencv_num_threads = max(
            1, int(rospy.get_param("~opencv_num_threads", 1)))
        cv2.setNumThreads(self.opencv_num_threads)
        cv2.ocl.setUseOpenCL(False)

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
        self.odom_buffer_size = max(1, int(rospy.get_param("~odom_buffer_size", 2000)))
        self.max_odom_receipt_age_sec = float(rospy.get_param("~max_odom_receipt_age_sec", 0.50))
        self.defer_odom_lookup_sec = float(rospy.get_param("~defer_odom_lookup_sec", 0.05))
        self.max_pending_sync_pairs = max(1, int(rospy.get_param("~max_pending_sync_pairs", 60)))
        self.max_pending_sync_wait_sec = float(rospy.get_param("~max_pending_sync_wait_sec", 0.0))
        self.max_raw_sync_pairs = max(
            1, int(rospy.get_param("~max_raw_sync_pairs", min(16, self.max_pending_sync_pairs))))
        self.process_batch_size = max(1, int(rospy.get_param("~process_batch_size", 1)))
        self.process_rate_hz = max(5.0, float(rospy.get_param("~process_rate_hz", 40.0)))
        self.debug_sync = bool(rospy.get_param("~debug_sync", True))
        self.point_stride = max(1, int(rospy.get_param("~point_stride", 2)))
        self.max_export_points = int(rospy.get_param("~max_export_points", 180000))
        self.queue_image_format = str(
            rospy.get_param("~queue_image_format", "jpg")).strip().lower()
        if self.queue_image_format == "jpeg":
            self.queue_image_format = "jpg"
        if self.queue_image_format not in ("jpg", "png"):
            raise RuntimeError(
                "Unsupported ~queue_image_format=%s. Use jpg or png." %
                self.queue_image_format)
        self.queue_jpeg_quality = max(
            1, min(100, int(rospy.get_param("~queue_jpeg_quality", 95))))
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
        self.cam_d4 = float(rospy.get_param("~cam_d4", 0.0))
        self.auto_resize_intrinsics = bool(rospy.get_param("~auto_resize_intrinsics_to_image", True))

        # Keep candidate throttling separate from completed queue requests.
        # During a bag replay the frontend can trail sensor time by seconds.
        # If the interval gate is tied to successful exports, every raw pair is
        # retained while waiting for odometry and the bounded pending queue
        # discards most of the trajectory before a pose arrives.
        self.last_candidate_stamp = None
        self.last_export_stamp = None
        self.exported = 0

        image_cls = CompressedImage if self.image_input_type == "compressed" else Image
        image_sub = Subscriber(self.image_topic, image_cls)

        # v23: odom is buffered separately by default. This prevents zero export when
        # /aft_mapped_to_init has timestamps that do not align with raw camera/Livox stamps.
        self.latest_odom = None
        self.latest_odom_wall = None
        self.odom_buffer = deque(maxlen=self.odom_buffer_size)
        self.raw_sync_pairs = deque()
        self.pending_sync_pairs = deque()
        self.state_lock = threading.Lock()
        self.pending_sync_drops = 0
        self.pending_capacity_drops = 0
        self.pending_timeout_drops = 0
        self.pending_unmatchable_drops = 0
        self.raw_capacity_drops = 0
        self.last_image_stamp = None
        self.last_cloud_stamp = None
        self.last_odom_stamp = None
        self.image_seen = 0
        self.cloud_seen = 0
        self.odom_seen = 0
        self.sync_seen = 0
        self.odom_sub_plain = None
        self.last_debug_wall = time.time()

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

        # Count messages on the same message_filters subscribers. A second pair
        # of high-bandwidth ROS subscribers doubles AT128 deserialization cost.
        image_sub.registerCallback(self.image_count_cb)
        cloud_sub.registerCallback(self.cloud_count_cb)

        if self.use_latest_odom:
            self.odom_sub_plain = rospy.Subscriber(
                self.odom_topic, Odometry, self.odom_cb, queue_size=200)
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
            odom_sub.registerCallback(self.odom_cb)
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

        rospy.loginfo("v32 camera_lidar_queue_exporter started session=%s", self.queue_session_id)
        rospy.loginfo("image=%s image_input_type=%s use_camera_info=%s lidar=%s lidar_input_type=%s cloud_in_map=%s odom=%s queue=%s",
                      self.image_topic, self.image_input_type, str(self.use_camera_info),
                      self.lidar_topic, self.lidar_input_type, str(self.cloud_in_map_frame),
                      self.odom_topic, str(self.queue_dir))
        rospy.loginfo("sync mode: use_latest_odom=%s approx_sync_slop=%.3f max_odom_age=%.3f debug_sync=%s",
                      str(self.use_latest_odom), self.approx_sync_slop, self.max_odom_age_sec, str(self.debug_sync))
        rospy.loginfo("odom match: mode=%s buffer=%d max_receipt_age=%.3f defer_lookup=%.3f "
                      "pending_pairs=%d pending_wall_timeout=%.3f",
                      self.odom_match_mode, self.odom_buffer_size,
                      self.max_odom_receipt_age_sec, self.defer_odom_lookup_sec,
                      self.max_pending_sync_pairs, self.max_pending_sync_wait_sec)
        rospy.loginfo("queue backpressure: export_rate=%.3fHz max_pending_requests=%d "
                      "raw_pairs=%d process_batch=%d process_rate=%.1fHz image_format=%s "
                      "jpeg_quality=%d opencv_threads=%d",
                      self.export_rate, self.max_pending_requests,
                      self.max_raw_sync_pairs, self.process_batch_size, self.process_rate_hz,
                      self.queue_image_format, self.queue_jpeg_quality,
                      self.opencv_num_threads)
        if not self.use_camera_info:
            rospy.loginfo("using YAML intrinsics: width=%d height=%d scale=%.3f fx=%.3f fy=%.3f cx=%.3f cy=%.3f D=[%.5f %.5f %.6f %.6f %.5f]",
                          self.cam_width, self.cam_height, self.image_scale,
                          self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy,
                          self.cam_d0, self.cam_d1, self.cam_d2, self.cam_d3,
                          self.cam_d4)

    @staticmethod
    def _sanitize_session_component(value):
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))

    def _write_json_atomically(self, path, payload):
        temporary = path.with_name("%s.tmp.%s" % (path.name, uuid.uuid4().hex))
        with open(str(temporary), "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.replace(str(temporary), str(path))

    def _existing_queue_session_id(self):
        try:
            with open(str(self.session_path), "r") as stream:
                value = json.load(stream).get("session_id", "")
            return str(value)
        except Exception:
            return ""

    def archive_stale_queue_controls(self, previous_session_id):
        """Preserve old control files without letting them block a new replay."""
        label = self._sanitize_session_component(previous_session_id or "no_session")
        archive_root = self.stale_dir / ("%d_%s" % (time.time_ns(), label))
        patterns = {
            "input": ("ready_*.json",),
            "segmented": ("ready_*.json",),
            "output": ("result_*.json", "meta_*.json"),
            "done": ("ready_*.json", "result_*.json", "meta_*.json"),
            "failed_sam3": ("ready_*.json",),
            "failed_mapper": ("result_*.json",),
        }
        moved = 0
        for relative, globs in patterns.items():
            source = self.queue_dir / relative
            if not source.exists():
                continue
            destination = archive_root / relative
            for glob_pattern in globs:
                for path in source.glob(glob_pattern):
                    destination.mkdir(parents=True, exist_ok=True)
                    try:
                        os.replace(str(path), str(destination / path.name))
                        moved += 1
                    except OSError:
                        # A service can be completing an old request while a new
                        # launch starts.  Session checks downstream still reject it.
                        pass
        if moved:
            rospy.loginfo("archived %d stale SAM3 queue control files under %s",
                          moved, str(archive_root))

    def start_queue_session(self):
        previous_session_id = self._existing_queue_session_id()
        self.archive_stale_queue_controls(previous_session_id)
        session_id = "%d_%s" % (time.time_ns(), uuid.uuid4().hex)
        self._write_json_atomically(self.session_path, {
            "session_id": session_id,
            "created_wall_time": time.time(),
            "producer": "camera_lidar_queue_exporter",
        })
        return session_id

    def stamp_to_sec(self, stamp):
        try:
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        except Exception:
            return 0.0

    def image_count_cb(self, msg):
        with self.state_lock:
            self.image_seen += 1
            self.last_image_stamp = msg.header.stamp

    def cloud_count_cb(self, msg):
        with self.state_lock:
            self.cloud_seen += 1
            self.last_cloud_stamp = msg.header.stamp

    def odom_cb(self, msg):
        wall = time.time()
        with self.state_lock:
            self.latest_odom = msg
            self.latest_odom_wall = wall
            self.odom_buffer.append((wall, msg))
            self.odom_seen += 1
            self.last_odom_stamp = msg.header.stamp

    def interval_due(self, sample_stamp, previous_stamp):
        if self.export_rate <= 0.0 or previous_stamp is None:
            return True
        # Support a new bag replay that rewinds the simulated clock.
        if sample_stamp + 1e-6 < previous_stamp:
            return True
        return sample_stamp - previous_stamp >= 1.0 / self.export_rate

    def can_queue_sample(self, stamp):
        sample_stamp = self.stamp_to_sec(stamp)
        if not self.interval_due(sample_stamp, self.last_candidate_stamp):
            return False
        if self.max_pending_requests > 0:
            pending = self.current_session_ready_count()
            if pending >= self.max_pending_requests:
                rospy.logwarn_throttle(
                    5.0,
                    "SAM3 queue backpressure active: pending ready requests=%d >= max_pending_requests=%d. "
                    "Exporter will skip frames until sam3_image_mask_service catches up.",
                    pending, self.max_pending_requests)
                return False
        return True

    def current_session_ready_count(self):
        count = 0
        for path in self.input_dir.glob("ready_*.json"):
            try:
                with open(str(path), "r") as stream:
                    if json.load(stream).get("queue_session_id", "") == self.queue_session_id:
                        count += 1
            except Exception:
                continue
        return count

    def get_latest_odom_for(self, ref_stamp, ref_wall=None):
        with self.state_lock:
            latest_odom = self.latest_odom
            latest_odom_wall = self.latest_odom_wall
            odom_buffer = list(self.odom_buffer)
        if latest_odom is None:
            rospy.logwarn_throttle(3.0, "No odometry received yet on %s", self.odom_topic)
            return None, {}

        if ref_wall is None:
            ref_wall = time.time()

        selected = latest_odom
        selected_wall = latest_odom_wall
        stamp_dt = abs(self.stamp_to_sec(ref_stamp) - self.stamp_to_sec(selected.header.stamp))

        if self.odom_match_mode == "stamp" and odom_buffer:
            selected_wall, selected = min(
                odom_buffer,
                key=lambda item: abs(self.stamp_to_sec(ref_stamp) - self.stamp_to_sec(item[1].header.stamp)))
            stamp_dt = abs(self.stamp_to_sec(ref_stamp) - self.stamp_to_sec(selected.header.stamp))
        elif self.odom_match_mode == "receipt" and odom_buffer:
            latest_allowed = ref_wall + max(0.0, self.defer_odom_lookup_sec)
            candidates = [item for item in odom_buffer if item[0] <= latest_allowed]
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

        if (self.odom_match_mode != "stamp" and
                receipt_age is not None and self.max_odom_receipt_age_sec > 0.0 and
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
        with self.state_lock:
            raw_pairs = len(self.raw_sync_pairs)
            pending_pairs = len(self.pending_sync_pairs)
            pending_drops = self.pending_sync_drops
            capacity_drops = self.pending_capacity_drops
            timeout_drops = self.pending_timeout_drops
            unmatchable_drops = self.pending_unmatchable_drops
            oldest_pending_stamp = (
                self.stamp_to_sec(self.pending_sync_pairs[0]["image_stamp"])
                if self.pending_sync_pairs else None)
            image_seen = self.image_seen
            cloud_seen = self.cloud_seen
            odom_seen = self.odom_seen
            sync_seen = self.sync_seen
            raw_drops = self.raw_capacity_drops
            candidate_stamp = self.last_candidate_stamp
            image_stamp = None if self.last_image_stamp is None else self.stamp_to_sec(self.last_image_stamp)
            cloud_stamp = None if self.last_cloud_stamp is None else self.stamp_to_sec(self.last_cloud_stamp)
            odom_stamp = None if self.last_odom_stamp is None else self.stamp_to_sec(self.last_odom_stamp)
        sensor_lag = None if image_stamp is None or odom_stamp is None else image_stamp - odom_stamp
        rospy.loginfo("sync debug: image_seen=%d cloud_seen=%d odom_seen=%d sync_pairs=%d candidates_until=%s exported=%d "
                      "raw_pairs=%d raw_drops=%d pending_pairs=%d pending_drops=%d capacity_drops=%d timeout_drops=%d "
                      "unmatchable_drops=%d sensor_lag=%s oldest_pending_stamp=%s "
                      "stamps image=%s cloud=%s odom=%s slop=%.3f use_latest_odom=%s",
                      image_seen, cloud_seen, odom_seen, sync_seen,
                      "none" if candidate_stamp is None else "%.6f" % candidate_stamp,
                      self.exported,
                      raw_pairs, raw_drops, pending_pairs, pending_drops, capacity_drops, timeout_drops,
                      unmatchable_drops,
                      "none" if sensor_lag is None else "%.3f" % sensor_lag,
            "none" if oldest_pending_stamp is None else "%.6f" % oldest_pending_stamp,
            "none" if image_stamp is None else "%.6f" % image_stamp,
            "none" if cloud_stamp is None else "%.6f" % cloud_stamp,
            "none" if odom_stamp is None else "%.6f" % odom_stamp,
            self.approx_sync_slop, str(self.use_latest_odom))

    def queue_sync_pair(self, image_msg, cloud_msg, camera_info_msg=None,
                        direct_odom_msg=None, direct_odom_diag=None):
        """Queue raw ROS messages; never decode native data in a callback thread."""
        pair = {
            "image_msg": image_msg,
            "cloud_msg": cloud_msg,
            "camera_info_msg": camera_info_msg,
            "direct_odom_msg": direct_odom_msg,
            "direct_odom_diag": direct_odom_diag,
            "received_wall": time.time(),
        }
        with self.state_lock:
            self.sync_seen += 1
            self.raw_sync_pairs.append(pair)
            while len(self.raw_sync_pairs) > self.max_raw_sync_pairs:
                self.raw_sync_pairs.popleft()
                self.raw_capacity_drops += 1

    def process_raw_sync_pairs(self):
        for _ in range(self.process_batch_size):
            with self.state_lock:
                if not self.raw_sync_pairs:
                    return
                raw_pair = self.raw_sync_pairs.popleft()
            image_msg = raw_pair["image_msg"]
            if not self.can_queue_sample(image_msg.header.stamp):
                continue
            try:
                pair = self.prepare_sample(
                    image_msg, raw_pair["cloud_msg"], raw_pair["camera_info_msg"])
            except Exception as error:
                rospy.logerr_throttle(2.0, "failed preparing SAM3 synchronized sample: %s", error)
                continue
            if pair is None:
                continue
            # Advance the semantic candidate clock before waiting for a pose.
            # The pending queue therefore contains at most export_rate samples,
            # even if the frontend is several seconds behind sensor time.
            self.last_candidate_stamp = self.stamp_to_sec(pair["image_stamp"])
            direct_odom_msg = raw_pair["direct_odom_msg"]
            try:
                if direct_odom_msg is not None:
                    self.handle_prepared_sample(
                        pair, direct_odom_msg, raw_pair["direct_odom_diag"])
                    continue
                pair["queued_wall"] = raw_pair["received_wall"]
                with self.state_lock:
                    self.pending_sync_pairs.append(pair)
                    while len(self.pending_sync_pairs) > self.max_pending_sync_pairs:
                        self.pending_sync_pairs.popleft()
                        self.pending_sync_drops += 1
                        self.pending_capacity_drops += 1
            except Exception as error:
                rospy.logerr_throttle(2.0, "failed queuing SAM3 synchronized sample: %s", error)

    def drain_pending_sync_pairs(self):
        while not rospy.is_shutdown():
            with self.state_lock:
                if not self.pending_sync_pairs:
                    return
                pair = self.pending_sync_pairs[0]
                latest_odom = self.latest_odom
            queued_wall = pair["queued_wall"]
            waited = time.time() - queued_wall
            if latest_odom is None:
                if self.max_pending_sync_wait_sec <= 0.0 or waited <= self.max_pending_sync_wait_sec:
                    return
                with self.state_lock:
                    if self.pending_sync_pairs and self.pending_sync_pairs[0] is pair:
                        self.pending_sync_pairs.popleft()
                        self.pending_sync_drops += 1
                        self.pending_timeout_drops += 1
                continue

            ref_stamp = self.stamp_to_sec(pair["image_stamp"])
            latest_stamp = self.stamp_to_sec(latest_odom.header.stamp)
            if self.odom_match_mode == "stamp" and latest_stamp + 1e-9 < ref_stamp:
                return

            odom_msg, odom_diag = self.get_latest_odom_for(pair["image_stamp"], queued_wall)
            if odom_msg is None:
                advanced = self.odom_match_mode == "stamp" and (
                    latest_stamp > ref_stamp + max(0.0, self.max_odom_age_sec))
                timed_out = self.odom_match_mode != "stamp" and (
                    self.max_pending_sync_wait_sec > 0.0 and
                    waited > self.max_pending_sync_wait_sec)
                if not advanced and not timed_out:
                    return
                with self.state_lock:
                    if self.pending_sync_pairs and self.pending_sync_pairs[0] is pair:
                        self.pending_sync_pairs.popleft()
                        self.pending_sync_drops += 1
                        if advanced:
                            self.pending_unmatchable_drops += 1
                        else:
                            self.pending_timeout_drops += 1
                continue

            with self.state_lock:
                if not self.pending_sync_pairs or self.pending_sync_pairs[0] is not pair:
                    continue
                self.pending_sync_pairs.popleft()
            try:
                self.handle_prepared_sample(pair, odom_msg, odom_diag=odom_diag)
            except Exception as error:
                rospy.logerr_throttle(2.0, "failed exporting SAM3 synchronized sample: %s", error)

    def callback_with_info_latest_odom(self, image_msg, camera_info_msg, cloud_msg):
        self.queue_sync_pair(image_msg, cloud_msg, camera_info_msg)

    def callback_without_info_latest_odom(self, image_msg, cloud_msg):
        self.queue_sync_pair(image_msg, cloud_msg)

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
        D = np.asarray([self.cam_d0, self.cam_d1, self.cam_d2, self.cam_d3,
                        self.cam_d4], dtype=np.float64)
        return K, D

    def filter_xyz_list(self, iterable, apply_lidar_filters=True):
        pts = []
        idx = 0
        malformed = 0
        for values in iterable:
            if values is None or len(values) < 3:
                malformed += 1
                continue
            x, y, z = values[:3]
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
        if malformed:
            rospy.logwarn_throttle(
                5.0, "ignored %d malformed LiDAR point records", malformed)
        if not pts:
            return None
        return np.asarray(pts, dtype=np.float32)

    def filter_xyz_array(self, xyz, apply_lidar_filters=True):
        """Apply the semantic-export sampling and range gates vectorially."""
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if self.point_stride > 1:
            xyz = xyz[::self.point_stride]
        if xyz.size == 0:
            return None
        valid = np.isfinite(xyz).all(axis=1)
        if apply_lidar_filters:
            squared_range = np.einsum("ij,ij->i", xyz, xyz)
            valid &= squared_range >= self.min_range_m * self.min_range_m
            valid &= squared_range <= self.max_range_m * self.max_range_m
            valid &= xyz[:, 2] >= self.z_min_lidar
            valid &= xyz[:, 2] <= self.z_max_lidar
        xyz = xyz[valid]
        if self.max_export_points > 0 and xyz.shape[0] > self.max_export_points:
            xyz = xyz[:self.max_export_points]
        if xyz.shape[0] == 0:
            return None
        return np.ascontiguousarray(xyz, dtype=np.float32)

    def pointcloud2_to_xyz(self, cloud_msg):
        """Decode x/y/z directly from the PointCloud2 byte buffer.

        AT128 packets use a compact 26-byte record.  The ROS Python
        ``read_points`` generator repeatedly calls struct unpacking for every
        point and has shown native crashes under long high-rate replays.  A
        bounded NumPy structured view uses the declared offsets and supports
        padded rows without invoking that generator.
        """
        fields = {field.name: field for field in cloud_msg.fields}
        if not all(name in fields for name in ("x", "y", "z")):
            rospy.logwarn_throttle(5.0, "PointCloud2 lacks x/y/z fields. fields=%s",
                                  list(fields.keys()))
            return None
        datatype_codes = {
            1: "i1",  # INT8
            2: "u1",  # UINT8
            3: "i2",  # INT16
            4: "u2",  # UINT16
            5: "i4",  # INT32
            6: "u4",  # UINT32
            7: "f4",  # FLOAT32
            8: "f8",  # FLOAT64
        }
        selected = [fields["x"], fields["y"], fields["z"]]
        point_step = int(cloud_msg.point_step)
        if point_step <= 0:
            raise RuntimeError("PointCloud2 has invalid point_step=%d" % point_step)
        endian = ">" if bool(cloud_msg.is_bigendian) else "<"
        formats = []
        offsets = []
        for field in selected:
            code = datatype_codes.get(int(field.datatype))
            if code is None or int(field.count) < 1:
                raise RuntimeError("unsupported PointCloud2 xyz field datatype/count: %s/%s" %
                                   (str(field.datatype), str(field.count)))
            dtype = np.dtype(endian + code)
            offset = int(field.offset)
            if offset < 0 or offset + dtype.itemsize > point_step:
                raise RuntimeError("PointCloud2 xyz field offset exceeds point_step")
            formats.append(dtype)
            offsets.append(offset)
        record_dtype = np.dtype({
            "names": ("x", "y", "z"),
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,
        })
        width = max(0, int(cloud_msg.width))
        height = max(1, int(cloud_msg.height))
        row_step = int(cloud_msg.row_step)
        required_row_step = width * point_step
        data_size = len(cloud_msg.data)
        if width <= 0 or data_size < required_row_step:
            return None
        if row_step >= required_row_step and data_size >= row_step * height:
            records = np.ndarray(
                shape=(height, width), dtype=record_dtype, buffer=cloud_msg.data,
                strides=(row_step, point_step))
        else:
            total = min(width * height, data_size // point_step)
            if total <= 0:
                return None
            records = np.ndarray(shape=(total,), dtype=record_dtype, buffer=cloud_msg.data)
        xyz = np.empty((records.size, 3), dtype=np.float32)
        xyz[:, 0] = np.asarray(records["x"]).reshape(-1)
        xyz[:, 1] = np.asarray(records["y"]).reshape(-1)
        xyz[:, 2] = np.asarray(records["z"]).reshape(-1)
        return self.filter_xyz_array(
            xyz, apply_lidar_filters=not self.cloud_in_map_frame)

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

    def prepare_sample(self, image_msg, cloud_msg, camera_info_msg=None):
        """Decode/copy synchronized sensor data while ROS owns the messages."""
        image_bgr = self.image_to_bgr(image_msg)
        if image_bgr is None:
            return None
        image_bgr = np.ascontiguousarray(image_bgr)

        try:
            xyz = self.cloud_to_xyz(cloud_msg)
        except Exception as e:
            rospy.logwarn_throttle(
                3.0, "Failed to decode synchronized LiDAR frame: %s", e)
            return None
        if xyz is None or xyz.shape[0] < 100:
            rospy.logwarn_throttle(
                3.0, "Too few valid LiDAR points for semantic queue")
            return None
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)

        height, width = image_bgr.shape[:2]
        if camera_info_msg is None:
            K, D = self.yaml_intrinsics_for_image(width, height)
            camera_width = width
            camera_height = height
            camera_frame_id = image_msg.header.frame_id
            intrinsic_source = "yaml"
        else:
            K = np.asarray(camera_info_msg.K, dtype=np.float64).reshape(3, 3)
            D = (np.asarray(camera_info_msg.D, dtype=np.float64)
                 if camera_info_msg.D else np.zeros((0,), dtype=np.float64))
            camera_width = int(camera_info_msg.width)
            camera_height = int(camera_info_msg.height)
            camera_frame_id = camera_info_msg.header.frame_id
            intrinsic_source = "camera_info"

        image_stamp = rospy.Time(
            int(image_msg.header.stamp.secs),
            int(image_msg.header.stamp.nsecs))
        cloud_stamp = rospy.Time(
            int(cloud_msg.header.stamp.secs),
            int(cloud_msg.header.stamp.nsecs))
        return {
            "image_stamp": image_stamp,
            "cloud_stamp": cloud_stamp,
            "camera_frame_id": str(camera_frame_id),
            "lidar_frame_id": str(getattr(cloud_msg.header, "frame_id", "")),
            "intrinsic_source": intrinsic_source,
            "camera_width": camera_width,
            "camera_height": camera_height,
            "image_bgr": image_bgr,
            "xyz": xyz,
            "K": np.ascontiguousarray(K, dtype=np.float64),
            "D": np.ascontiguousarray(D, dtype=np.float64),
        }

    def callback_with_info(self, image_msg, camera_info_msg, cloud_msg, odom_msg, odom_diag=None):
        self.queue_sync_pair(image_msg, cloud_msg, camera_info_msg, odom_msg, odom_diag)

    def callback_without_info(self, image_msg, cloud_msg, odom_msg, odom_diag=None):
        self.queue_sync_pair(image_msg, cloud_msg, None, odom_msg, odom_diag)

    def image_to_bgr(self, image_msg):
        try:
            if self.image_input_type == "compressed":
                arr = np.frombuffer(image_msg.data, dtype=np.uint8)
                image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError("cv2.imdecode returned None")
                return image_bgr
            return self.raw_image_to_bgr(image_msg)
        except Exception as e:
            rospy.logwarn("Failed to convert image from %s: %s", self.image_input_type, e)
            return None

    @staticmethod
    def raw_image_to_bgr(image_msg):
        """Convert common ROS Image encodings without loading cv_bridge.

        A pip OpenCV can be newer than the OpenCV linked into ROS Noetic's
        cv_bridge. Loading both native libraries in one Python process is
        ABI-unsafe, so the small conversion needed here is explicit.
        """
        encoding = image_msg.encoding.strip().lower()
        formats = {
            "mono8": (np.uint8, 1),
            "8uc1": (np.uint8, 1),
            "bgr8": (np.uint8, 3),
            "rgb8": (np.uint8, 3),
            "bgra8": (np.uint8, 4),
            "rgba8": (np.uint8, 4),
            "mono16": (np.uint16, 1),
            "16uc1": (np.uint16, 1),
        }
        if encoding not in formats:
            raise RuntimeError("unsupported raw image encoding: %s" % image_msg.encoding)

        dtype, channels = formats[encoding]
        dtype = np.dtype(dtype)
        if dtype.itemsize > 1:
            message_big_endian = bool(image_msg.is_bigendian)
            native_big_endian = not np.little_endian
            if message_big_endian != native_big_endian:
                dtype = dtype.newbyteorder(">" if message_big_endian else "<")

        row_values = int(image_msg.step) // dtype.itemsize
        required_values = int(image_msg.width) * channels
        if image_msg.height <= 0 or image_msg.width <= 0 or row_values < required_values:
            raise RuntimeError(
                "invalid raw image dimensions/step: %dx%d step=%d encoding=%s" %
                (image_msg.width, image_msg.height, image_msg.step, image_msg.encoding))

        expected_values = int(image_msg.height) * row_values
        raw = np.frombuffer(image_msg.data, dtype=dtype, count=expected_values)
        if raw.size != expected_values:
            raise RuntimeError("raw image data is shorter than height*step")
        pixels = raw.reshape(int(image_msg.height), row_values)[:, :required_values]
        if channels > 1:
            pixels = pixels.reshape(int(image_msg.height), int(image_msg.width), channels)
        else:
            pixels = pixels.reshape(int(image_msg.height), int(image_msg.width))
        pixels = np.ascontiguousarray(pixels)

        if encoding in ("mono16", "16uc1"):
            pixels = (pixels.astype(np.uint16) >> 8).astype(np.uint8)
            return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        if encoding in ("mono8", "8uc1"):
            return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        if encoding == "rgb8":
            return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
        return pixels

    def handle_prepared_sample(self, sample, odom_msg, odom_diag=None):
        image_stamp = sample["image_stamp"]
        cloud_stamp = sample["cloud_stamp"]
        image_bgr = sample["image_bgr"]
        xyz = sample["xyz"]
        K = sample["K"]
        D = sample["D"]
        sample_stamp = self.stamp_to_sec(image_stamp)
        if not self.interval_due(sample_stamp, self.last_export_stamp):
            return
        T_map_body = self.odom_to_matrix(odom_msg)

        stamp = image_stamp.to_nsec()
        self.request_sequence += 1
        req_id = "%s_%s_%06d" % (
            str(stamp), self.queue_session_id.split("_")[-1][:12], self.request_sequence)
        image_path = self.input_dir / (
            "image_%s.%s" % (req_id, self.queue_image_format))
        image_tmp = self.input_dir / (
            "image_%s.tmp.%s" % (req_id, self.queue_image_format))
        cloud_path = self.input_dir / ("cloud_%s.npz" % req_id)
        cloud_tmp = self.input_dir / ("cloud_%s.tmp.npz" % req_id)
        meta_tmp = self.input_dir / ("ready_%s.json.tmp" % req_id)
        meta_path = self.input_dir / ("ready_%s.json" % req_id)

        image_write_params = []
        if self.queue_image_format == "jpg":
            image_write_params = [
                int(cv2.IMWRITE_JPEG_QUALITY), self.queue_jpeg_quality]
        if not cv2.imwrite(str(image_tmp), image_bgr, image_write_params):
            raise RuntimeError("failed to write queue image: %s" % image_tmp)
        os.replace(str(image_tmp), str(image_path))
        np.savez_compressed(
            str(cloud_tmp),
            xyz=xyz,
            K=K,
            D=D,
            lidar_input_type=np.asarray(self.lidar_input_type),
            cloud_in_map_frame=np.asarray(self.cloud_in_map_frame),
        )
        os.replace(str(cloud_tmp), str(cloud_path))

        meta = {
            "id": req_id,
            "queue_session_id": self.queue_session_id,
            "created_wall_time": time.time(),
            "stamp": {"secs": image_stamp.secs, "nsecs": image_stamp.nsecs},
            "image_path": str(image_path),
            "cloud_path": str(cloud_path),
            "camera_width": int(sample["camera_width"]),
            "camera_height": int(sample["camera_height"]),
            "image_width": int(image_bgr.shape[1]),
            "image_height": int(image_bgr.shape[0]),
            "camera_frame_id": sample["camera_frame_id"],
            "intrinsic_source": sample["intrinsic_source"],
            "K": K.reshape(-1).tolist(),
            "D": D.reshape(-1).tolist(),
            "lidar_frame_id": sample["lidar_frame_id"],
            "lidar_input_type": self.lidar_input_type,
            "cloud_in_map_frame": self.cloud_in_map_frame,
            "odom_frame_id": odom_msg.header.frame_id,
            "odom_child_frame_id": odom_msg.child_frame_id,
            "odom_diag": odom_diag or {},
            "image_cloud_stamp_dt": abs(
                self.stamp_to_sec(image_stamp) -
                self.stamp_to_sec(cloud_stamp)),
            "T_map_body": T_map_body.reshape(-1).tolist(),
        }

        with open(str(meta_tmp), "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(str(meta_tmp), str(meta_path))

        self.last_export_stamp = sample_stamp
        self.exported += 1
        diag = odom_diag or {}
        rospy.loginfo("exported SAM3 semantic request id=%s points=%d image=%dx%d Kfx=%.2f Kfy=%.2f lidar_type=%s cloud_in_map=%s frame=%s odom_mode=%s odom_receipt_age=%s odom_stamp_dt=%.3f total=%d",
                      req_id, xyz.shape[0], image_bgr.shape[1], image_bgr.shape[0],
                      K[0, 0], K[1, 1], self.lidar_input_type,
                      str(self.cloud_in_map_frame),
                      sample["lidar_frame_id"],
                      diag.get("odom_match_mode", "direct"),
                      "none" if diag.get("odom_receipt_age_sec", None) is None else "%.3f" % diag.get("odom_receipt_age_sec"),
                      float(diag.get("odom_stamp_dt", 0.0)),
                      self.exported)

    def run(self):
        """Run all OpenCV/NumPy/file operations in one deterministic thread."""
        sleep_sec = 1.0 / self.process_rate_hz
        while not rospy.is_shutdown():
            try:
                self.process_raw_sync_pairs()
                self.drain_pending_sync_pairs()
                now = time.time()
                if self.debug_sync and now - self.last_debug_wall >= 5.0:
                    self.debug_timer_cb(None)
                    self.last_debug_wall = now
            except Exception as error:
                rospy.logerr_throttle(2.0, "SAM3 exporter foreground loop failed: %s", error)
            time.sleep(sleep_sec)


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    rospy.init_node("camera_lidar_queue_exporter")
    rospy.loginfo("SAM3 exporter runtime: numpy=%s (%s) cv2=%s (%s)",
                  np.__version__, np.__file__, cv2.__version__, cv2.__file__)
    node = CameraLidarQueueExporter()
    node.run()
