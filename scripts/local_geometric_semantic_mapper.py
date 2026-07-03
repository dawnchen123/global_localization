#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Geometric Semantic Mapper for FAST-LIVO2 global localization.

Purpose
-------
Build a local BEV semantic label map from accumulated /cloud_registered point clouds.
This node is designed for the thesis Chapter 3 pipeline:

    accumulated LiDAR local submap -> geometric semantic BEV map
    semantic map matching against satellite semantic map

It does NOT claim learning-based semantic segmentation. It generates reliable geometric
semantic labels from local point-cloud structure, and marks ambiguous cells as unknown.
This is preferable to forcing every cell into a wrong semantic class.

Published topics
----------------
/local_semantic_map/label       sensor_msgs/Image mono8
/local_semantic_map/color       sensor_msgs/Image bgr8
/local_semantic_map/confidence  sensor_msgs/Image mono8
/local_semantic_map/debug       sensor_msgs/Image bgr8
/local_semantic_map/stats       std_msgs/String JSON

Label convention
----------------
0 unknown / unobserved
1 open_ground / drivable/open low planar area
2 structure / building wall / tall rigid structure
3 vegetation / rough tall irregular structure
4 obstacle / low or medium isolated obstacle

Simplified geometric convention for matching
--------------------------------------------
0 unknown / ignored
1 road_area / traversable planar area
2 building_structure
5 road_edge / curb / road boundary
6 lane_line candidate

Notes
-----
- If /cloud_registered is already in FAST-LIVO2 map frame camera_init, set ~cloud_in_map_frame=true.
- If your point cloud is raw LiDAR frame, set ~cloud_in_map_frame=false and provide an odom/pose transform.
  This script currently supports only direct cloud-in-map mode robustly; raw mode is intentionally conservative.
"""

import json
import math
import struct
import threading
from collections import deque

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String


LABEL_UNKNOWN = 0
LABEL_OPEN = 1
LABEL_STRUCTURE = 2
LABEL_VEGETATION = 3
LABEL_OBSTACLE = 4
LABEL_ROAD_EDGE = 5
LABEL_LANE_LINE = 6

# BGR colors for rqt_image_view / OpenCV
COLORS = {
    LABEL_UNKNOWN: (190, 190, 190),   # gray
    LABEL_OPEN: (210, 210, 210),      # light gray/open area
    LABEL_STRUCTURE: (0, 0, 255),     # red
    LABEL_VEGETATION: (0, 180, 0),    # green
    LABEL_OBSTACLE: (0, 165, 255),    # orange
}

GEOM_COLORS = {
    LABEL_UNKNOWN: (30, 30, 30),
    LABEL_OPEN: (180, 180, 180),       # road/traversable area
    LABEL_STRUCTURE: (0, 0, 230),      # building/rigid structure
    LABEL_ROAD_EDGE: (255, 255, 255),  # road boundary
    LABEL_LANE_LINE: (255, 255, 0),    # lane/paint candidate
}


def safe_param(name, default):
    return rospy.get_param(name, default)


def voxel_downsample(points, voxel):
    if points.size == 0 or voxel <= 0:
        return points
    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def remove_small_components(mask, min_area):
    if min_area <= 1:
        return mask
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    out = np.zeros_like(mask_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out.astype(bool)


def smooth_float_height_map(src, k):
    if src.size == 0 or k <= 3:
        return src.astype(np.float32)
    k = int(k)
    if k % 2 == 0:
        k += 1
    src32 = src.astype(np.float32)
    finite = np.isfinite(src32)
    if not np.any(finite):
        return np.zeros_like(src32)
    lo = float(np.percentile(src32[finite], 2.0))
    hi = float(np.percentile(src32[finite], 98.0))
    if hi <= lo + 1e-6:
        return cv2.GaussianBlur(src32, (k, k), 0.0)
    norm = np.clip((src32 - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    try:
        u8 = cv2.medianBlur(u8, k)
        out = u8.astype(np.float32) / 255.0 * (hi - lo) + lo
    except cv2.error:
        out = src32
    return cv2.GaussianBlur(out.astype(np.float32), (k, k), 0.0)


def decode_packed_rgb(value):
    """Decode ROS PointCloud2 rgb/rgba fields that may arrive as float32 or uint32."""
    if value is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        if isinstance(value, float):
            if not math.isfinite(value):
                return 0.0, 0.0, 0.0, 0.0
            packed = struct.unpack("I", struct.pack("f", value))[0]
        else:
            packed = int(value)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0
    r = float((packed >> 16) & 255)
    g = float((packed >> 8) & 255)
    b = float(packed & 255)
    if r == 0.0 and g == 0.0 and b == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return r, g, b, 1.0


def normalize_rgb_channel(value):
    try:
        v = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if 0.0 <= v <= 1.0:
        v *= 255.0
    return float(np.clip(v, 0.0, 255.0))


class LocalGeometricSemanticMapper:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.cloud_buffer = deque()
        self.latest_odom = None
        self.latest_stamp = None

        # Topics
        self.cloud_topic = safe_param("~cloud_topic", "/cloud_registered")
        self.odom_topic = safe_param("~odom_topic", "/aft_mapped_to_init")
        self.cloud_in_map_frame = bool(safe_param("~cloud_in_map_frame", True))
        self.map_frame = safe_param("~map_frame", "camera_init")
        self.robot_frame = safe_param("~robot_frame", "base_link")

        # Accumulation
        self.window_sec = float(safe_param("~cloud_accumulation_window", 60.0))
        self.max_frames = int(safe_param("~cloud_accumulation_max_frames", 600))
        self.input_stride = int(safe_param("~cloud_accumulation_stride", 1))
        self.voxel_size = float(safe_param("~cloud_accumulation_voxel_size", 0.08))
        self.max_points_per_frame = int(safe_param("~max_points_per_frame", 80000))
        self.frame_count = 0

        # BEV grid
        self.resolution = float(safe_param("~bev_resolution", 0.20))
        self.map_size_m = float(safe_param("~bev_size_m", 160.0))
        self.max_range_m = float(safe_param("~max_point_range", 220.0))
        self.z_min_filter = float(safe_param("~z_min", -5.0))
        self.z_max_filter = float(safe_param("~z_max", 8.0))
        self.center_mode = safe_param("~center_mode", "odom")  # odom | cloud_median | origin
        self.fixed_center_x = float(safe_param("~fixed_center_x", 0.0))
        self.fixed_center_y = float(safe_param("~fixed_center_y", 0.0))

        # Classification thresholds
        self.min_points_per_cell = int(safe_param("~min_points_per_cell", 3))
        self.ground_percentile = float(safe_param("~ground_percentile", 5.0))
        self.ground_margin = float(safe_param("~ground_margin", 0.35))
        self.local_ground_smooth_cells = int(safe_param("~local_ground_smooth_cells", 21))
        self.local_ground_seed_max_above_global = float(safe_param("~local_ground_seed_max_above_global", 0.80))
        self.height_band_filter_enabled = bool(safe_param("~height_band_filter_enabled", True))
        self.height_band_min_above_ground = float(safe_param("~height_band_min_above_ground", -0.20))
        self.height_band_max_above_ground = float(safe_param("~height_band_max_above_ground", 1.50))
        self.open_height_range_max = float(safe_param("~open_height_range_max", 0.28))
        self.open_z_std_max = float(safe_param("~open_z_std_max", 0.12))
        self.structure_height_min = float(safe_param("~structure_height_min", 1.30))
        self.structure_z_std_max = float(safe_param("~structure_z_std_max", 0.95))
        self.vegetation_height_min = float(safe_param("~vegetation_height_min", 1.00))
        self.vegetation_z_std_min = float(safe_param("~vegetation_z_std_min", 0.45))
        self.vegetation_height_range_min = float(safe_param("~vegetation_height_range_min", 1.20))
        self.obstacle_height_min = float(safe_param("~obstacle_height_min", 0.25))
        self.obstacle_height_max = float(safe_param("~obstacle_height_max", 1.30))
        self.use_color_features = bool(safe_param("~use_color_features", True))
        self.use_intensity_features = bool(safe_param("~use_intensity_features", True))
        self.vegetation_green_excess_min = float(safe_param("~vegetation_green_excess_min", 0.08))
        self.vegetation_saturation_min = float(safe_param("~vegetation_saturation_min", 0.18))
        self.vegetation_color_texture_min = float(safe_param("~vegetation_color_texture_min", 0.05))
        self.structure_green_excess_max = float(safe_param("~structure_green_excess_max", 0.04))
        self.structure_color_texture_max = float(safe_param("~structure_color_texture_max", 0.16))
        self.road_saturation_max = float(safe_param("~road_saturation_max", 0.30))
        self.road_green_excess_max = float(safe_param("~road_green_excess_max", 0.05))
        self.publish_simplified_geometric = bool(safe_param("~publish_simplified_geometric", True))
        self.road_edge_dilate_iter = int(safe_param("~road_edge_dilate_iter", 1))
        self.lane_line_intensity_min = float(safe_param("~lane_line_intensity_min", 0.75))
        self.lane_line_value_min = float(safe_param("~lane_line_value_min", 0.72))
        self.lane_line_saturation_max = float(safe_param("~lane_line_saturation_max", 0.35))
        self.simplified_road_height_max = float(safe_param("~simplified_road_height_max", 0.70))
        self.simplified_road_height_range_max = float(safe_param("~simplified_road_height_range_max", 0.55))
        self.simplified_road_z_std_max = float(safe_param("~simplified_road_z_std_max", 0.25))
        self.simplified_building_height_min = float(safe_param("~simplified_building_height_min", 1.45))
        self.simplified_building_height_range_min = float(safe_param("~simplified_building_height_range_min", 0.90))
        self.simplified_building_z_std_max = float(safe_param("~simplified_building_z_std_max", 0.70))
        self.simplified_building_green_excess_max = float(safe_param("~simplified_building_green_excess_max", 0.01))
        self.simplified_building_color_texture_max = float(safe_param("~simplified_building_color_texture_max", 0.22))
        self.simplified_building_min_area = int(safe_param("~simplified_building_min_area", 24))
        self.simplified_road_min_area = int(safe_param("~simplified_road_min_area", 35))

        # Morphology and quality gates
        self.min_component_area_structure = int(safe_param("~min_component_area_structure", 6))
        self.min_component_area_vegetation = int(safe_param("~min_component_area_vegetation", 8))
        self.min_component_area_open = int(safe_param("~min_component_area_open", 20))
        self.dilate_observed = int(safe_param("~dilate_observed", 1))
        self.publish_rate = float(safe_param("~publish_rate", 1.0))
        self.min_total_points = int(safe_param("~min_total_points", 2000))

        # Publishers
        self.pub_label = rospy.Publisher("/local_semantic_map/label", Image, queue_size=1)
        self.pub_color = rospy.Publisher("/local_semantic_map/color", Image, queue_size=1)
        self.pub_conf = rospy.Publisher("/local_semantic_map/confidence", Image, queue_size=1)
        self.pub_debug = rospy.Publisher("/local_semantic_map/debug", Image, queue_size=1)
        self.pub_stats = rospy.Publisher("/local_semantic_map/stats", String, queue_size=1)
        self.pub_geom_label = rospy.Publisher("/local_geometric_map/label", Image, queue_size=1)
        self.pub_geom_color = rospy.Publisher("/local_geometric_map/color", Image, queue_size=1)
        self.pub_geom_conf = rospy.Publisher("/local_geometric_map/confidence", Image, queue_size=1)
        self.pub_rgb_bev = rospy.Publisher("/local_rgb_bev/image", Image, queue_size=1)

        self.sub_cloud = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_cb, queue_size=2, buff_size=2**26)
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 0.05)), self.timer_cb)

        rospy.loginfo("local_geometric_semantic_mapper started cloud=%s odom=%s res=%.2fm size=%.1fm window=%.1fs",
                      self.cloud_topic, self.odom_topic, self.resolution, self.map_size_m, self.window_sec)

    def odom_cb(self, msg):
        with self.lock:
            self.latest_odom = msg

    def cloud_cb(self, msg):
        self.frame_count += 1
        if self.input_stride > 1 and (self.frame_count % self.input_stride) != 0:
            return

        # Convert point cloud. Use geometry plus optional intensity/rgb fields when available.
        try:
            available = set([f.name for f in msg.fields])
            if not all(name in available for name in ["x", "y", "z"]):
                rospy.logwarn_throttle(
                    5.0,
                    "Skip cloud without x/y/z fields on %s: width=%d height=%d fields=%s",
                    self.cloud_topic, msg.width, msg.height, ",".join(sorted(list(available)))
                )
                return
            field_names = ["x", "y", "z"]
            has_intensity = self.use_intensity_features and "intensity" in available
            has_rgb = self.use_color_features and "rgb" in available
            has_rgba = self.use_color_features and (not has_rgb) and "rgba" in available
            has_direct_rgb = self.use_color_features and all(x in available for x in ["r", "g", "b"])
            if has_intensity:
                field_names.append("intensity")
            if has_rgb:
                field_names.append("rgb")
            elif has_rgba:
                field_names.append("rgba")
            elif has_direct_rgb:
                field_names.extend(["r", "g", "b"])
            rospy.loginfo_throttle(
                10.0,
                "semantic mapper cloud fields=%s use_intensity=%s use_color=%s",
                ",".join(sorted(list(available))), str(has_intensity), str(has_rgb or has_rgba or has_direct_rgb)
            )
            idx = {name: i for i, name in enumerate(field_names)}
            pts = []
            for p in pc2.read_points(msg, field_names=tuple(field_names), skip_nans=True):
                intensity = float("nan")
                if has_intensity:
                    try:
                        intensity = float(p[idx["intensity"]])
                    except Exception:
                        intensity = float("nan")
                r, g, b, color_valid = 0.0, 0.0, 0.0, 0.0
                if has_rgb:
                    r, g, b, color_valid = decode_packed_rgb(p[idx["rgb"]])
                elif has_rgba:
                    r, g, b, color_valid = decode_packed_rgb(p[idx["rgba"]])
                elif has_direct_rgb:
                    r = normalize_rgb_channel(p[idx["r"]])
                    g = normalize_rgb_channel(p[idx["g"]])
                    b = normalize_rgb_channel(p[idx["b"]])
                    color_valid = 1.0 if (r + g + b) > 1.0 else 0.0
                pts.append((p[0], p[1], p[2], intensity, r, g, b, color_valid))
                if len(pts) >= self.max_points_per_frame:
                    break
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Failed reading cloud fields: %s", str(e))
            return

        if not pts:
            rospy.logwarn_throttle(5.0, "Empty cloud on %s", self.cloud_topic)
            return

        arr = np.asarray(pts, dtype=np.float32)
        # Only x/y/z are mandatory. Optional intensity/color channels may be NaN
        # when the input cloud does not provide those fields.
        finite = np.isfinite(arr[:, :3]).all(axis=1)
        arr = arr[finite]
        if arr.size == 0:
            return

        # Filter z and range around current odom or origin for stability.
        arr = arr[(arr[:, 2] >= self.z_min_filter) & (arr[:, 2] <= self.z_max_filter)]
        if arr.size == 0:
            return

        if self.latest_odom is not None and self.center_mode == "odom":
            ox = self.latest_odom.pose.pose.position.x
            oy = self.latest_odom.pose.pose.position.y
        else:
            ox = 0.0
            oy = 0.0
        r = np.hypot(arr[:, 0] - ox, arr[:, 1] - oy)
        arr = arr[r <= self.max_range_m]
        if arr.size == 0:
            return

        arr = voxel_downsample(arr, self.voxel_size)
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()

        with self.lock:
            self.cloud_buffer.append((stamp, arr))
            self.latest_stamp = stamp
            self.trim_buffer_locked(stamp)

    def trim_buffer_locked(self, now_stamp):
        # Time window
        cutoff = now_stamp.to_sec() - self.window_sec
        while self.cloud_buffer and self.cloud_buffer[0][0].to_sec() < cutoff:
            self.cloud_buffer.popleft()
        # Max frames
        while len(self.cloud_buffer) > self.max_frames:
            self.cloud_buffer.popleft()

    def get_center_and_points(self):
        with self.lock:
            if not self.cloud_buffer:
                return None, None, None
            clouds = [x[1] for x in self.cloud_buffer if x[1].size > 0]
            stamp = self.latest_stamp or rospy.Time.now()
            odom = self.latest_odom

        if not clouds:
            return None, None, None
        pts = np.vstack(clouds)
        if pts.shape[0] < self.min_total_points:
            rospy.logwarn_throttle(5.0, "Too few accumulated points: %d < %d", pts.shape[0], self.min_total_points)

        if self.center_mode == "odom" and odom is not None:
            cx = odom.pose.pose.position.x
            cy = odom.pose.pose.position.y
        elif self.center_mode == "cloud_median":
            cx = float(np.median(pts[:, 0]))
            cy = float(np.median(pts[:, 1]))
        else:
            cx = self.fixed_center_x
            cy = self.fixed_center_y
        return (cx, cy), pts, stamp

    def build_grid_features(self, pts, cx, cy):
        W = int(round(self.map_size_m / self.resolution))
        H = W
        x0 = cx - self.map_size_m / 2.0
        y0 = cy - self.map_size_m / 2.0

        col = np.floor((pts[:, 0] - x0) / self.resolution).astype(np.int32)
        row_from_bottom = np.floor((pts[:, 1] - y0) / self.resolution).astype(np.int32)
        row = H - 1 - row_from_bottom
        valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        col = col[valid]
        row = row[valid]
        pts_valid = pts[valid]
        z = pts_valid[:, 2]
        has_extra = pts_valid.shape[1] >= 8
        intensity = pts_valid[:, 3] if has_extra else np.full_like(z, np.nan, dtype=np.float32)
        rgb = pts_valid[:, 4:7] if has_extra else np.zeros((z.shape[0], 3), dtype=np.float32)
        color_valid_pt = (pts_valid[:, 7] > 0.5) if has_extra else np.zeros(z.shape, dtype=bool)
        intensity_valid_pt = np.isfinite(intensity)

        count = np.zeros((H, W), dtype=np.float32)
        zsum = np.zeros((H, W), dtype=np.float32)
        zsum2 = np.zeros((H, W), dtype=np.float32)
        zmin = np.full((H, W), np.inf, dtype=np.float32)
        zmax = np.full((H, W), -np.inf, dtype=np.float32)
        isum = np.zeros((H, W), dtype=np.float32)
        isum2 = np.zeros((H, W), dtype=np.float32)
        icount = np.zeros((H, W), dtype=np.float32)
        rsum = np.zeros((H, W), dtype=np.float32)
        gsum = np.zeros((H, W), dtype=np.float32)
        bsum = np.zeros((H, W), dtype=np.float32)
        rsum2 = np.zeros((H, W), dtype=np.float32)
        gsum2 = np.zeros((H, W), dtype=np.float32)
        bsum2 = np.zeros((H, W), dtype=np.float32)
        ccount = np.zeros((H, W), dtype=np.float32)

        np.add.at(count, (row, col), 1.0)
        np.add.at(zsum, (row, col), z)
        np.add.at(zsum2, (row, col), z * z)
        np.minimum.at(zmin, (row, col), z)
        np.maximum.at(zmax, (row, col), z)
        if np.any(intensity_valid_pt):
            ri = row[intensity_valid_pt]
            ci = col[intensity_valid_pt]
            iv = intensity[intensity_valid_pt].astype(np.float32)
            np.add.at(isum, (ri, ci), iv)
            np.add.at(isum2, (ri, ci), iv * iv)
            np.add.at(icount, (ri, ci), 1.0)
        if np.any(color_valid_pt):
            rc = row[color_valid_pt]
            cc = col[color_valid_pt]
            rgb_norm = np.clip(rgb[color_valid_pt].astype(np.float32) / 255.0, 0.0, 1.0)
            rv = rgb_norm[:, 0]
            gv = rgb_norm[:, 1]
            bv = rgb_norm[:, 2]
            np.add.at(rsum, (rc, cc), rv)
            np.add.at(gsum, (rc, cc), gv)
            np.add.at(bsum, (rc, cc), bv)
            np.add.at(rsum2, (rc, cc), rv * rv)
            np.add.at(gsum2, (rc, cc), gv * gv)
            np.add.at(bsum2, (rc, cc), bv * bv)
            np.add.at(ccount, (rc, cc), 1.0)

        observed = count >= self.min_points_per_cell
        zmin[~observed] = 0.0
        zmax[~observed] = 0.0
        mean = np.zeros_like(count)
        var = np.zeros_like(count)
        mean[observed] = zsum[observed] / np.maximum(count[observed], 1.0)
        var[observed] = zsum2[observed] / np.maximum(count[observed], 1.0) - mean[observed] ** 2
        var = np.maximum(var, 0.0)
        zstd = np.sqrt(var)
        hrange = np.zeros_like(count)
        hrange[observed] = zmax[observed] - zmin[observed]

        # Global ground level estimate from low percentile of observed z minima.
        if np.any(observed):
            ground_z = float(np.percentile(zmin[observed], self.ground_percentile))
        else:
            ground_z = 0.0
        hmax_rel = zmax - ground_z
        hmin_rel = zmin - ground_z
        local_ground = zmin.copy()
        elevated_ground_seed = observed & ((zmin - ground_z) > self.local_ground_seed_max_above_global)
        local_ground[elevated_ground_seed] = ground_z
        local_ground[~observed] = ground_z
        k = max(3, int(self.local_ground_smooth_cells))
        if k % 2 == 0:
            k += 1
        if k > 3:
            local_ground = smooth_float_height_map(local_ground, k)
        hmax_local = zmax - local_ground
        hmin_local = zmin - local_ground
        hmax_local[~observed] = 0.0
        hmin_local[~observed] = 0.0

        raw_observed_cells = int(np.count_nonzero(observed))
        raw_points_in_grid = int(z.shape[0])
        height_band_points = raw_points_in_grid
        height_band_applied = False
        if self.height_band_filter_enabled and raw_points_in_grid > 0:
            point_ground = local_ground[row, col]
            point_h = z - point_ground
            band = (point_h >= self.height_band_min_above_ground) & \
                (point_h <= self.height_band_max_above_ground)
            if np.any(band):
                height_band_applied = True
                height_band_points = int(np.count_nonzero(band))
                rb = row[band]
                cb = col[band]
                zb = z[band]

                count = np.zeros((H, W), dtype=np.float32)
                zsum = np.zeros((H, W), dtype=np.float32)
                zsum2 = np.zeros((H, W), dtype=np.float32)
                zmin = np.full((H, W), np.inf, dtype=np.float32)
                zmax = np.full((H, W), -np.inf, dtype=np.float32)
                isum = np.zeros((H, W), dtype=np.float32)
                isum2 = np.zeros((H, W), dtype=np.float32)
                icount = np.zeros((H, W), dtype=np.float32)
                rsum = np.zeros((H, W), dtype=np.float32)
                gsum = np.zeros((H, W), dtype=np.float32)
                bsum = np.zeros((H, W), dtype=np.float32)
                rsum2 = np.zeros((H, W), dtype=np.float32)
                gsum2 = np.zeros((H, W), dtype=np.float32)
                bsum2 = np.zeros((H, W), dtype=np.float32)
                ccount = np.zeros((H, W), dtype=np.float32)

                np.add.at(count, (rb, cb), 1.0)
                np.add.at(zsum, (rb, cb), zb)
                np.add.at(zsum2, (rb, cb), zb * zb)
                np.minimum.at(zmin, (rb, cb), zb)
                np.maximum.at(zmax, (rb, cb), zb)

                intensity_valid_band = intensity_valid_pt & band
                if np.any(intensity_valid_band):
                    ri = row[intensity_valid_band]
                    ci = col[intensity_valid_band]
                    iv = intensity[intensity_valid_band].astype(np.float32)
                    np.add.at(isum, (ri, ci), iv)
                    np.add.at(isum2, (ri, ci), iv * iv)
                    np.add.at(icount, (ri, ci), 1.0)

                color_valid_band = color_valid_pt & band
                if np.any(color_valid_band):
                    rc = row[color_valid_band]
                    cc = col[color_valid_band]
                    rgb_norm = np.clip(rgb[color_valid_band].astype(np.float32) / 255.0, 0.0, 1.0)
                    rv = rgb_norm[:, 0]
                    gv = rgb_norm[:, 1]
                    bv = rgb_norm[:, 2]
                    np.add.at(rsum, (rc, cc), rv)
                    np.add.at(gsum, (rc, cc), gv)
                    np.add.at(bsum, (rc, cc), bv)
                    np.add.at(rsum2, (rc, cc), rv * rv)
                    np.add.at(gsum2, (rc, cc), gv * gv)
                    np.add.at(bsum2, (rc, cc), bv * bv)
                    np.add.at(ccount, (rc, cc), 1.0)

                observed = count >= self.min_points_per_cell
                zmin[~observed] = 0.0
                zmax[~observed] = 0.0
                mean = np.zeros_like(count)
                var = np.zeros_like(count)
                mean[observed] = zsum[observed] / np.maximum(count[observed], 1.0)
                var[observed] = zsum2[observed] / np.maximum(count[observed], 1.0) - mean[observed] ** 2
                var = np.maximum(var, 0.0)
                zstd = np.sqrt(var)
                hrange = np.zeros_like(count)
                hrange[observed] = zmax[observed] - zmin[observed]
                hmax_rel = zmax - ground_z
                hmin_rel = zmin - ground_z
                hmax_local = zmax - local_ground
                hmin_local = zmin - local_ground
                hmax_local[~observed] = 0.0
                hmin_local[~observed] = 0.0
            else:
                height_band_applied = True
                height_band_points = 0
                count = np.zeros((H, W), dtype=np.float32)
                zsum = np.zeros((H, W), dtype=np.float32)
                zsum2 = np.zeros((H, W), dtype=np.float32)
                zmin = np.zeros((H, W), dtype=np.float32)
                zmax = np.zeros((H, W), dtype=np.float32)
                isum = np.zeros((H, W), dtype=np.float32)
                isum2 = np.zeros((H, W), dtype=np.float32)
                icount = np.zeros((H, W), dtype=np.float32)
                rsum = np.zeros((H, W), dtype=np.float32)
                gsum = np.zeros((H, W), dtype=np.float32)
                bsum = np.zeros((H, W), dtype=np.float32)
                rsum2 = np.zeros((H, W), dtype=np.float32)
                gsum2 = np.zeros((H, W), dtype=np.float32)
                bsum2 = np.zeros((H, W), dtype=np.float32)
                ccount = np.zeros((H, W), dtype=np.float32)
                observed = count >= self.min_points_per_cell
                mean = np.zeros_like(count)
                zstd = np.zeros_like(count)
                hrange = np.zeros_like(count)
                hmax_rel = np.zeros_like(count)
                hmin_rel = np.zeros_like(count)
                hmax_local = np.zeros_like(count)
                hmin_local = np.zeros_like(count)

        # Density normalized for confidence; log makes it robust.
        density_norm = np.clip(np.log1p(count) / np.log1p(max(20.0, np.percentile(count[observed], 95) if np.any(observed) else 20.0)), 0.0, 1.0)

        # Texture/edge of height map, helpful for vegetation/building boundary.
        zmax_filled = zmax.copy()
        if np.any(observed):
            zmax_filled[~observed] = ground_z
        gx = cv2.Sobel(zmax_filled, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(zmax_filled, cv2.CV_32F, 0, 1, ksize=3)
        height_grad = np.sqrt(gx * gx + gy * gy)
        height_grad = np.clip(height_grad / 5.0, 0.0, 1.0)

        intensity_mean = np.zeros_like(count)
        intensity_std = np.zeros_like(count)
        has_intensity_cell = icount > 0
        if np.any(has_intensity_cell):
            intensity_mean[has_intensity_cell] = isum[has_intensity_cell] / np.maximum(icount[has_intensity_cell], 1.0)
            ivar = isum2[has_intensity_cell] / np.maximum(icount[has_intensity_cell], 1.0) - intensity_mean[has_intensity_cell] ** 2
            intensity_std[has_intensity_cell] = np.sqrt(np.maximum(ivar, 0.0))
        intensity_norm = np.zeros_like(count)
        if np.any(has_intensity_cell):
            vals = intensity_mean[has_intensity_cell]
            lo = float(np.percentile(vals, 5.0))
            hi = float(np.percentile(vals, 95.0))
            if hi > lo + 1e-6:
                intensity_norm[has_intensity_cell] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)

        color_valid = ccount > 0
        red_mean = np.zeros_like(count)
        green_mean = np.zeros_like(count)
        blue_mean = np.zeros_like(count)
        color_texture = np.zeros_like(count)
        if np.any(color_valid):
            red_mean[color_valid] = rsum[color_valid] / np.maximum(ccount[color_valid], 1.0)
            green_mean[color_valid] = gsum[color_valid] / np.maximum(ccount[color_valid], 1.0)
            blue_mean[color_valid] = bsum[color_valid] / np.maximum(ccount[color_valid], 1.0)
            rvar = rsum2[color_valid] / np.maximum(ccount[color_valid], 1.0) - red_mean[color_valid] ** 2
            gvar = gsum2[color_valid] / np.maximum(ccount[color_valid], 1.0) - green_mean[color_valid] ** 2
            bvar = bsum2[color_valid] / np.maximum(ccount[color_valid], 1.0) - blue_mean[color_valid] ** 2
            color_texture[color_valid] = np.sqrt(np.maximum(rvar + gvar + bvar, 0.0) / 3.0)
        max_rgb = np.maximum(np.maximum(red_mean, green_mean), blue_mean)
        min_rgb = np.minimum(np.minimum(red_mean, green_mean), blue_mean)
        saturation = np.zeros_like(count)
        saturation[color_valid] = (max_rgb[color_valid] - min_rgb[color_valid]) / np.maximum(max_rgb[color_valid], 1e-3)
        value = max_rgb
        green_excess = green_mean - 0.5 * (red_mean + blue_mean)
        green_dominance = (green_mean - red_mean) + (green_mean - blue_mean)

        return {
            "H": H,
            "W": W,
            "x0": x0,
            "y0": y0,
            "ground_z": ground_z,
            "count": count,
            "observed": observed,
            "zmin": zmin,
            "zmax": zmax,
            "mean": mean,
            "zstd": zstd,
            "hrange": hrange,
            "hmax_rel": hmax_rel,
            "hmin_rel": hmin_rel,
            "local_ground": local_ground,
            "hmax_local": hmax_local,
            "hmin_local": hmin_local,
            "density_norm": density_norm,
            "height_grad": height_grad,
            "raw_points_in_grid": raw_points_in_grid,
            "raw_observed_cells": raw_observed_cells,
            "height_band_filter_enabled": self.height_band_filter_enabled,
            "height_band_applied": height_band_applied,
            "height_band_points": height_band_points,
            "height_band_min_above_ground": self.height_band_min_above_ground,
            "height_band_max_above_ground": self.height_band_max_above_ground,
            "has_intensity": has_intensity_cell,
            "intensity_mean": intensity_mean,
            "intensity_norm": intensity_norm,
            "intensity_std": intensity_std,
            "color_valid": color_valid,
            "red_mean": red_mean,
            "green_mean": green_mean,
            "blue_mean": blue_mean,
            "green_excess": green_excess,
            "green_dominance": green_dominance,
            "saturation": saturation,
            "value": value,
            "color_texture": color_texture,
        }

    def classify(self, f):
        observed = f["observed"]
        count = f["count"]
        hr = f["hrange"]
        zs = f["zstd"]
        hmax = f.get("hmax_local", f["hmax_rel"])
        dens = f["density_norm"]
        grad = f["height_grad"]
        color_valid = f["color_valid"]
        global_color_available = bool(np.any(color_valid))
        green_excess = f["green_excess"]
        green_dom = f["green_dominance"]
        sat = f["saturation"]
        val = f["value"]
        color_tex = f["color_texture"]
        has_intensity = f["has_intensity"]
        inten = f["intensity_norm"]
        inten_std = f["intensity_std"]

        label = np.zeros((f["H"], f["W"]), dtype=np.uint8)
        conf = np.zeros_like(label, dtype=np.uint8)

        # Valid observed mask, optionally dilated for visualization and matching stability.
        obs = observed.copy()
        if self.dilate_observed > 0:
            kernel = np.ones((2 * self.dilate_observed + 1, 2 * self.dilate_observed + 1), np.uint8)
            obs = cv2.dilate(obs.astype(np.uint8), kernel, iterations=1).astype(bool)

        green_color_strong = color_valid & (green_excess >= self.vegetation_green_excess_min) & \
            (green_dom > 0.05) & (sat >= self.vegetation_saturation_min)
        green_color = color_valid & (green_excess >= self.vegetation_green_excess_min * 0.55) & \
            (green_dom > 0.01) & (sat >= max(0.10, self.vegetation_saturation_min * 0.60))
        neutral_or_dark = (~color_valid) | (sat <= self.road_saturation_max) | (val <= 0.35)
        non_green_color = (~color_valid) | ((green_excess <= self.structure_green_excess_max) & (~green_color))
        rough_vertical = (zs >= self.vegetation_z_std_min) | (hr >= self.vegetation_height_range_min) | (grad >= 0.45)
        rigid_vertical = (hmax >= self.structure_height_min) & (hr >= 0.80) & \
            (zs <= self.structure_z_std_max) & (color_tex <= self.structure_color_texture_max)
        intensity_edge = has_intensity & (inten_std >= np.percentile(inten_std[has_intensity], 70.0) if np.any(has_intensity) else False)

        # Class candidates. Color is used as evidence, not as the only criterion, because
        # FAST-LIVO2 colored clouds may be sparse or partly over/under-exposed.
        open_mask = observed & (hr <= self.open_height_range_max) & (zs <= self.open_z_std_max) & \
            (hmax <= self.ground_margin) & (neutral_or_dark | (green_excess <= self.road_green_excess_max))

        veg_mask = observed & (
            ((hmax >= self.vegetation_height_min) & rough_vertical) |
            ((hmax >= self.vegetation_height_min * 0.65) & green_color & ((hr >= 0.35) | (zs >= 0.12) | (grad >= 0.20)))
        ) & (
            green_color_strong |
            green_color |
            ((~global_color_available) & (zs >= self.vegetation_z_std_min + 0.12)) |
            ((~color_valid) & global_color_available & (zs >= self.vegetation_z_std_min + 0.08) & (hr >= self.vegetation_height_range_min + 0.05)) |
            (green_color & (color_tex >= self.vegetation_color_texture_min * 0.6))
        ) & (dens <= 0.98)

        structure_mask = observed & rigid_vertical & non_green_color
        structure_mask |= observed & (hmax >= self.structure_height_min + 0.4) & (zs <= self.structure_z_std_max * 0.8) & non_green_color
        # Road curbs, poles and facades can have high intensity/edge evidence. Do not let
        # colorless high-gradient cells become vegetation without green evidence.
        structure_mask |= observed & (hmax >= self.structure_height_min) & intensity_edge & non_green_color & (zs <= self.structure_z_std_max)
        structure_mask = structure_mask & (~veg_mask | ((dens > 0.80) & (zs < self.vegetation_z_std_min + 0.10) & non_green_color))

        obstacle_mask = observed & (hmax >= self.obstacle_height_min) & (hmax <= self.obstacle_height_max) & (~open_mask)
        obstacle_mask |= observed & (hmax < self.vegetation_height_min) & (grad >= 0.35) & non_green_color & (~open_mask)
        obstacle_mask = obstacle_mask & (~structure_mask) & (~veg_mask)

        # Morphological cleanup per class.
        open_mask = remove_small_components(open_mask, self.min_component_area_open)
        structure_mask = remove_small_components(structure_mask, self.min_component_area_structure)
        veg_mask = remove_small_components(veg_mask, self.min_component_area_vegetation)

        # Priority: structure > vegetation > obstacle > open. Unknown otherwise.
        label[open_mask] = LABEL_OPEN
        label[obstacle_mask] = LABEL_OBSTACLE
        label[veg_mask] = LABEL_VEGETATION
        label[structure_mask] = LABEL_STRUCTURE

        # Confidence. Ambiguous or low-density cells get low confidence.
        c = np.zeros_like(f["count"], dtype=np.float32)
        color_bonus_open = np.where(color_valid, np.clip(1.0 - sat / max(self.road_saturation_max, 1e-3), 0, 1), 0.6)
        color_bonus_struct = np.where(color_valid, np.clip(1.0 - np.maximum(green_excess, 0) / 0.25, 0, 1), 0.65)
        color_bonus_veg = np.where(color_valid, np.clip((green_excess - self.vegetation_green_excess_min) / 0.25, 0, 1), 0.45)
        c_open = np.clip(1.0 - hr / max(self.open_height_range_max, 1e-3), 0, 1) * dens * (0.6 + 0.4 * color_bonus_open)
        c_struct = np.clip((hmax - self.structure_height_min) / 2.0, 0, 1) * np.clip(1.2 - zs, 0, 1) * (0.4 + 0.6 * dens) * (0.6 + 0.4 * color_bonus_struct)
        c_veg = np.clip((zs - self.vegetation_z_std_min + 0.15) / 0.8, 0, 1) * np.clip(hmax / 3.0, 0, 1) * (0.3 + 0.7 * dens) * (0.5 + 0.5 * color_bonus_veg)
        c_obs = np.clip(hmax / max(self.obstacle_height_max, 1e-3), 0, 1) * dens
        c[label == LABEL_OPEN] = c_open[label == LABEL_OPEN]
        c[label == LABEL_STRUCTURE] = c_struct[label == LABEL_STRUCTURE]
        c[label == LABEL_VEGETATION] = c_veg[label == LABEL_VEGETATION]
        c[label == LABEL_OBSTACLE] = c_obs[label == LABEL_OBSTACLE]
        conf = np.clip(c * 255.0, 0, 255).astype(np.uint8)

        # Unknown observed cells with very weak evidence remain unknown but visible in debug.
        return label, conf

    def simplify_for_matching(self, label, conf, f):
        """Drop vegetation and keep stable geometry for global matching."""
        observed = f["observed"]
        hmax = f.get("hmax_local", f["hmax_rel"])
        hr = f["hrange"]
        zs = f["zstd"]
        grad = f["height_grad"]
        dens = f["density_norm"]
        has_intensity = f["has_intensity"]
        inten = f["intensity_norm"]
        color_valid = f["color_valid"]
        green_excess = f["green_excess"]
        green_dom = f["green_dominance"]
        sat = f["saturation"]
        val = f["value"]
        color_tex = f["color_texture"]
        global_color_available = bool(np.any(color_valid))
        no_global_color = not global_color_available

        green_like = color_valid & \
            (green_excess >= self.vegetation_green_excess_min * 0.7) & \
            (green_dom > 0.03) & \
            (sat >= max(0.12, self.vegetation_saturation_min * 0.65))
        tree_color_evidence = green_like | (color_valid & (sat >= 0.25) & (green_excess > -0.02))
        if no_global_color:
            tree_color_evidence = np.ones_like(observed, dtype=bool)
        tree_like = green_like | (
            observed &
            (hmax >= self.vegetation_height_min) &
            (hr >= self.vegetation_height_range_min * 0.85) &
            (zs >= self.vegetation_z_std_min) &
            tree_color_evidence
        )

        road_color_ok = (~color_valid) | \
            (sat <= max(self.road_saturation_max, 0.38)) | \
            (val <= 0.48) | \
            (green_excess <= self.road_green_excess_max + 0.03)
        flat_low = observed & \
            (hmax <= self.simplified_road_height_max) & \
            (hr <= self.simplified_road_height_range_max) & \
            (zs <= self.simplified_road_z_std_max)
        road = ((label == LABEL_OPEN) | (flat_low & road_color_ok)) & (~tree_like)

        road_u8 = road.astype(np.uint8) * 255
        if np.count_nonzero(road_u8) > 0:
            kernel = np.ones((3, 3), np.uint8)
            road_u8 = cv2.morphologyEx(road_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            road_u8 = cv2.morphologyEx(road_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        road = remove_small_components(road_u8 > 0, self.simplified_road_min_area)

        non_green = np.ones_like(observed, dtype=bool)
        if global_color_available:
            non_green = (~color_valid) | (green_excess <= self.simplified_building_green_excess_max)
        texture_ok = (~color_valid) | \
            (color_tex <= self.simplified_building_color_texture_max) | \
            (sat <= self.road_saturation_max)
        rigid = observed & \
            (hmax >= self.simplified_building_height_min) & \
            (hr >= self.simplified_building_height_range_min) & \
            (zs <= self.simplified_building_z_std_max) & \
            (dens >= 0.10) & non_green & texture_ok & (~tree_like)
        tall_rigid = observed & \
            (hmax >= self.simplified_building_height_min + 0.65) & \
            (zs <= self.simplified_building_z_std_max + 0.18) & \
            (dens >= 0.14) & non_green & texture_ok & (~tree_like)
        building = (rigid | tall_rigid | ((label == LABEL_STRUCTURE) & rigid)) & (~road)
        building = remove_small_components(building, self.simplified_building_min_area)
        if np.count_nonzero(building) > 0:
            building_u8 = building.astype(np.uint8) * 255
            building_u8 = cv2.morphologyEx(building_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            building = remove_small_components(building_u8 > 0, self.simplified_building_min_area)

        # Boundaries of traversable area plus low-height high-gradient curb-like cells.
        edge = np.zeros(label.shape, dtype=bool)
        road_u8 = road.astype(np.uint8) * 255
        if np.count_nonzero(road_u8) > 0:
            edge |= cv2.Canny(road_u8, 40, 120) > 0
        curb_like = observed & (hmax >= self.obstacle_height_min) & (hmax <= self.obstacle_height_max) & \
            (grad >= 0.25) & (hr <= max(0.8, self.obstacle_height_max)) & (~tree_like) & (~building)
        near_road = cv2.dilate(road.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        edge |= curb_like & near_road
        if self.road_edge_dilate_iter > 0:
            edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=self.road_edge_dilate_iter).astype(bool)
        edge &= ~building

        # Lane lines are only reliable when intensity or RGB exists; otherwise leave empty.
        lane = np.zeros(label.shape, dtype=bool)
        if np.any(has_intensity):
            lane |= road & (inten >= self.lane_line_intensity_min) & (grad < 0.20)
        if np.any(color_valid):
            lane |= road & color_valid & (val >= self.lane_line_value_min) & (sat <= self.lane_line_saturation_max)
        lane = remove_small_components(lane, 3)

        geom = np.zeros_like(label, dtype=np.uint8)
        geom[road] = LABEL_OPEN
        geom[building] = LABEL_STRUCTURE
        geom[edge] = LABEL_ROAD_EDGE
        geom[lane] = LABEL_LANE_LINE

        gconf = np.zeros_like(conf, dtype=np.uint8)
        road_score = np.clip(1.0 - hmax / max(self.simplified_road_height_max, 1e-3), 0.0, 1.0) * \
            np.clip(1.0 - hr / max(self.simplified_road_height_range_max, 1e-3), 0.0, 1.0) * \
            np.clip(1.0 - zs / max(self.simplified_road_z_std_max, 1e-3), 0.0, 1.0) * \
            (0.45 + 0.55 * dens)
        building_score = np.clip((hmax - self.simplified_building_height_min) / 2.0, 0.0, 1.0) * \
            np.clip(1.0 - zs / max(self.simplified_building_z_std_max + 0.25, 1e-3), 0.0, 1.0) * \
            (0.45 + 0.55 * dens)
        gconf[road] = np.maximum(gconf[road], np.clip(road_score[road] * 220.0, 40, 220).astype(np.uint8))
        gconf[building] = np.maximum(gconf[building], np.clip(building_score[building] * 230.0, 45, 230).astype(np.uint8))
        gconf[edge] = 220
        gconf[lane] = 230
        return geom, gconf

    def colorize_geometric(self, label):
        color = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for k, v in GEOM_COLORS.items():
            color[label == k] = v
        return color

    def make_rgb_bev(self, f):
        rgb = np.zeros((f["H"], f["W"], 3), dtype=np.uint8)
        color_valid = f["color_valid"]
        if not np.any(color_valid):
            return rgb
        # OpenCV/RViz image encoding is BGR.
        rgb[..., 0] = np.clip(f["blue_mean"] * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(f["green_mean"] * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(f["red_mean"] * 255.0, 0, 255).astype(np.uint8)
        rgb[~color_valid] = (0, 0, 0)
        return rgb

    def colorize(self, label):
        color = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for k, v in COLORS.items():
            color[label == k] = v
        return color

    def make_debug(self, label, conf, f):
        color = self.colorize(label)
        # Blend confidence as brightness.
        conf_f = (conf.astype(np.float32) / 255.0)[..., None]
        debug = (color.astype(np.float32) * (0.45 + 0.55 * conf_f)).astype(np.uint8)
        # Mark observed but unknown cells dark gray.
        obs_unknown = f["observed"] & (label == LABEL_UNKNOWN)
        debug[obs_unknown] = (80, 80, 80)
        # Draw center cross.
        H, W = label.shape
        cv2.drawMarker(debug, (W // 2, H // 2), (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        return debug

    def timer_cb(self, _event):
        center, pts, stamp = self.get_center_and_points()
        if pts is None:
            return
        cx, cy = center
        if pts.shape[0] < 10:
            return

        f = self.build_grid_features(pts, cx, cy)
        label, conf = self.classify(f)
        color = self.colorize(label)
        debug = self.make_debug(label, conf, f)
        geom_label, geom_conf = self.simplify_for_matching(label, conf, f)
        geom_color = self.colorize_geometric(geom_label)
        rgb_bev = self.make_rgb_bev(f)

        header_stamp = stamp if stamp is not None else rospy.Time.now()
        frame_id = self.map_frame

        label_msg = self.bridge.cv2_to_imgmsg(label, encoding="mono8")
        label_msg.header.stamp = header_stamp
        label_msg.header.frame_id = frame_id
        color_msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
        color_msg.header.stamp = header_stamp
        color_msg.header.frame_id = frame_id
        conf_msg = self.bridge.cv2_to_imgmsg(conf, encoding="mono8")
        conf_msg.header.stamp = header_stamp
        conf_msg.header.frame_id = frame_id
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header.stamp = header_stamp
        debug_msg.header.frame_id = frame_id

        self.pub_label.publish(label_msg)
        self.pub_color.publish(color_msg)
        self.pub_conf.publish(conf_msg)
        self.pub_debug.publish(debug_msg)
        if self.publish_simplified_geometric:
            geom_label_msg = self.bridge.cv2_to_imgmsg(geom_label, encoding="mono8")
            geom_label_msg.header.stamp = header_stamp
            geom_label_msg.header.frame_id = frame_id
            geom_color_msg = self.bridge.cv2_to_imgmsg(geom_color, encoding="bgr8")
            geom_color_msg.header.stamp = header_stamp
            geom_color_msg.header.frame_id = frame_id
            geom_conf_msg = self.bridge.cv2_to_imgmsg(geom_conf, encoding="mono8")
            geom_conf_msg.header.stamp = header_stamp
            geom_conf_msg.header.frame_id = frame_id
            self.pub_geom_label.publish(geom_label_msg)
            self.pub_geom_color.publish(geom_color_msg)
            self.pub_geom_conf.publish(geom_conf_msg)
        if np.any(rgb_bev):
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_bev, encoding="bgr8")
            rgb_msg.header.stamp = header_stamp
            rgb_msg.header.frame_id = frame_id
            self.pub_rgb_bev.publish(rgb_msg)

        total = label.size
        stats = {
            "stamp": header_stamp.to_sec(),
            "center_x": cx,
            "center_y": cy,
            "resolution": self.resolution,
            "size_m": self.map_size_m,
            "ground_z": f["ground_z"],
            "points": int(pts.shape[0]),
            "raw_points_in_grid": int(f.get("raw_points_in_grid", pts.shape[0])),
            "height_band_points": int(f.get("height_band_points", pts.shape[0])),
            "height_band_applied": bool(f.get("height_band_applied", False)),
            "height_band_min_above_ground": float(f.get("height_band_min_above_ground", 0.0)),
            "height_band_max_above_ground": float(f.get("height_band_max_above_ground", 0.0)),
            "observed_cells": int(np.count_nonzero(f["observed"])),
            "raw_observed_cells": int(f.get("raw_observed_cells", np.count_nonzero(f["observed"]))),
            "unknown": int(np.count_nonzero(label == LABEL_UNKNOWN)),
            "open": int(np.count_nonzero(label == LABEL_OPEN)),
            "structure": int(np.count_nonzero(label == LABEL_STRUCTURE)),
            "vegetation": int(np.count_nonzero(label == LABEL_VEGETATION)),
            "obstacle": int(np.count_nonzero(label == LABEL_OBSTACLE)),
            "geom_road": int(np.count_nonzero(geom_label == LABEL_OPEN)),
            "geom_building": int(np.count_nonzero(geom_label == LABEL_STRUCTURE)),
            "geom_road_edge": int(np.count_nonzero(geom_label == LABEL_ROAD_EDGE)),
            "geom_lane": int(np.count_nonzero(geom_label == LABEL_LANE_LINE)),
            "foreground_ratio": float(np.count_nonzero(label > 0) / float(total)),
            "mean_confidence": float(np.mean(conf[label > 0]) if np.any(label > 0) else 0.0),
        }
        self.pub_stats.publish(String(data=json.dumps(stats, ensure_ascii=False)))

        rospy.loginfo_throttle(5.0,
            "local semantic map points=%d band=%d fg=%.3f open=%d struct=%d veg=%d obs=%d geom road=%d bld=%d edge=%d lane=%d conf=%.1f",
            stats["points"], stats["height_band_points"], stats["foreground_ratio"], stats["open"], stats["structure"],
            stats["vegetation"], stats["obstacle"], stats["geom_road"], stats["geom_building"],
            stats["geom_road_edge"], stats["geom_lane"], stats["mean_confidence"])


def main():
    rospy.init_node("local_geometric_semantic_mapper")
    LocalGeometricSemanticMapper()
    rospy.spin()


if __name__ == "__main__":
    main()
