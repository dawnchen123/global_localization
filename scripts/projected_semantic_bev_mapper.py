#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS-side semantic BEV mapper.

Reads SAM3 label results from file queue. For each result:
  camera label map + saved LiDAR cloud + saved camera intrinsics + odometry
  -> project LiDAR points into camera label map
  -> assign semantic labels to 3D points
  -> transform points to FAST-LIVO2 map frame
  -> accumulate global or bounded-window semantic points
  -> publish local semantic BEV map.
"""

import faulthandler
import json
import os
import struct
import sys
import time
from pathlib import Path
from collections import deque


def prefer_ros_binary_modules():
    """Use the distro NumPy/OpenCV pair with the ROS Noetic interpreter.

    User-local pip wheels can shadow the Ubuntu packages with a different
    NumPy C ABI. The allocator mismatch only surfaced after dozens of
    projected frames, so choose the coherent distro pair before importing
    either extension module.
    """
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
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String


LABEL_UNKNOWN = 0
LABEL_ROAD = 1
LABEL_BUILDING = 2
LABEL_TREE = 3
LABEL_GRASS = 4
LABEL_DYNAMIC = 5
SAM3_LABELS = [LABEL_ROAD, LABEL_BUILDING, LABEL_TREE, LABEL_GRASS, LABEL_DYNAMIC]
BEV_LABELS = [LABEL_ROAD, LABEL_BUILDING, LABEL_TREE, LABEL_GRASS]

# semantic_voxel_mapper_node internal labels. Keep this mapper's local BEV in
# SAM3 labels, but publish the optional 3D semantic cloud in internal labels
# when it is fused with RangeNet++.
INTERNAL_LABEL_UNKNOWN = 0
INTERNAL_LABEL_ROAD = 1
INTERNAL_LABEL_SIDEWALK = 2
INTERNAL_LABEL_BUILDING = 3
INTERNAL_LABEL_VEGETATION = 4
INTERNAL_LABEL_DYNAMIC = 5
INTERNAL_LABEL_OTHER = 6

SAM3_TO_INTERNAL_LABEL = {
    LABEL_UNKNOWN: INTERNAL_LABEL_UNKNOWN,
    LABEL_ROAD: INTERNAL_LABEL_ROAD,
    LABEL_BUILDING: INTERNAL_LABEL_BUILDING,
    LABEL_TREE: INTERNAL_LABEL_VEGETATION,
    LABEL_GRASS: INTERNAL_LABEL_VEGETATION,
    LABEL_DYNAMIC: INTERNAL_LABEL_DYNAMIC,
}

COLOR_BGR = {
    LABEL_UNKNOWN: (190, 190, 190),
    LABEL_ROAD: (80, 80, 80),
    LABEL_BUILDING: (0, 0, 255),
    LABEL_TREE: (0, 170, 0),
    LABEL_GRASS: (0, 220, 120),
    LABEL_DYNAMIC: (0, 255, 255),
}

INTERNAL_COLOR_BGR = {
    INTERNAL_LABEL_UNKNOWN: (190, 190, 190),
    INTERNAL_LABEL_ROAD: (80, 80, 80),
    INTERNAL_LABEL_SIDEWALK: (80, 180, 80),
    INTERNAL_LABEL_BUILDING: (0, 0, 255),
    INTERNAL_LABEL_VEGETATION: (0, 170, 0),
    INTERNAL_LABEL_DYNAMIC: (0, 255, 255),
    INTERNAL_LABEL_OTHER: (120, 120, 120),
}


def rgb_float_from_bgr(bgr):
    b, g, r = [int(x) & 255 for x in bgr]
    rgb_u32 = (r << 16) | (g << 8) | b
    return struct.unpack("f", struct.pack("I", rgb_u32))[0]


LABEL_RGB_FLOAT = {
    lab: rgb_float_from_bgr(bgr) for lab, bgr in COLOR_BGR.items()
}

INTERNAL_LABEL_RGB_FLOAT = {
    lab: rgb_float_from_bgr(bgr) for lab, bgr in INTERNAL_COLOR_BGR.items()
}


def as_T(value, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 16:
        raise ValueError("%s must have 16 numbers" % name)
    return arr.reshape(4, 4)


def colorize_label(label):
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k, c in COLOR_BGR.items():
        out[label == k] = c
    return out


def numpy_to_image(array, encoding, stamp, frame_id):
    """Create a ROS Image without mixing pip and cv_bridge OpenCV ABIs."""
    image = np.ascontiguousarray(array, dtype=np.uint8)
    if encoding == "mono8":
        if image.ndim != 2:
            raise ValueError("mono8 image must be HxW")
        channels = 1
    elif encoding == "bgr8":
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr8 image must be HxWx3")
        channels = 3
    else:
        raise ValueError("unsupported ROS image encoding: %s" % encoding)

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(msg.width * channels)
    msg.data = image.tobytes()
    return msg


def unique_integer_rows(keys):
    """Return unique integer rows and inverse ids using only a plain sort.

    The system NumPy build used by ROS Noetic has shown native faults in both
    ``unique(..., axis=0)`` and ufunc ``.at`` scatter operations under long
    replay.  Keep grouping on contiguous int64 arrays and construct the
    inverse map from a stable one-dimensional sort instead.
    """
    rows = np.ascontiguousarray(keys, dtype=np.int64)
    if rows.ndim != 2:
        raise ValueError("integer row keys must be a 2D array")
    if rows.shape[0] == 0:
        return rows.copy(), np.empty((0,), dtype=np.int64)

    mins = rows.min(axis=0)
    maxs = rows.max(axis=0)
    spans = [
        int(maxs[col]) - int(mins[col]) + 1
        for col in range(rows.shape[1])
    ]
    max_int64 = int(np.iinfo(np.int64).max)
    cardinality = 1
    encodable = True
    for span in spans:
        if span <= 0 or cardinality > max_int64 // span:
            encodable = False
            break
        cardinality *= span

    if encodable:
        codes = np.zeros((rows.shape[0],), dtype=np.int64)
        stride = 1
        for col in range(rows.shape[1] - 1, -1, -1):
            codes += (rows[:, col] - mins[col]) * stride
            stride *= spans[col]
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        starts_mask = np.empty((sorted_codes.shape[0],), dtype=bool)
        starts_mask[0] = True
        starts_mask[1:] = sorted_codes[1:] != sorted_codes[:-1]
        starts = np.flatnonzero(starts_mask)
        inverse = np.empty((rows.shape[0],), dtype=np.int64)
        inverse[order] = np.cumsum(starts_mask, dtype=np.int64) - 1
        return np.ascontiguousarray(rows[order[starts]]), inverse

    # Coordinates spanning the full int64 domain are not expected for a
    # metric map, but retain a collision-free fallback for malformed input.
    lookup = {}
    unique_rows = []
    inverse = np.empty((rows.shape[0],), dtype=np.int64)
    for i, row in enumerate(rows):
        key = tuple(int(value) for value in row)
        index = lookup.get(key)
        if index is None:
            index = len(unique_rows)
            lookup[key] = index
            unique_rows.append(key)
        inverse[i] = index
    return np.asarray(unique_rows, dtype=np.int64), inverse


def grouped_layout(group_ids, group_count):
    """Return a stable sorted layout for non-negative dense group ids."""
    ids = np.ascontiguousarray(group_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty
    if group_count <= 0 or ids.min() < 0 or ids.max() >= group_count:
        raise ValueError("group ids are outside the declared dense range")
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts_mask = np.empty((sorted_ids.shape[0],), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_ids[1:] != sorted_ids[:-1]
    starts = np.flatnonzero(starts_mask)
    return order, starts, sorted_ids[starts]


def grouped_min_max_count(group_ids, values, group_count):
    """Compute dense per-group min/max/count without ufunc scatter updates."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    order, starts, present = grouped_layout(group_ids, group_count)
    minimum = np.full((group_count,), np.inf, dtype=np.float32)
    maximum = np.full((group_count,), -np.inf, dtype=np.float32)
    count = np.zeros((group_count,), dtype=np.int32)
    if order.size == 0:
        return minimum, maximum, count
    sorted_values = values[order]
    minimum[present] = np.minimum.reduceat(sorted_values, starts)
    maximum[present] = np.maximum.reduceat(sorted_values, starts)
    ends = np.empty((starts.shape[0] + 1,), dtype=np.int64)
    ends[:-1] = starts
    ends[-1] = sorted_values.shape[0]
    count[present] = np.diff(ends).astype(np.int32)
    return minimum, maximum, count


def grouped_minimum(group_ids, values, group_count, fill_value):
    """Compute a dense minimum image/vector without ``np.minimum.at``."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    order, starts, present = grouped_layout(group_ids, group_count)
    out = np.full((group_count,), fill_value, dtype=np.float32)
    if order.size:
        out[present] = np.minimum.reduceat(values[order], starts)
    return out


def spatially_select_cloud_indices(points, score, max_points, base_voxel):
    """Bound a cloud while retaining its spatial coverage.

    Selecting globally by vote count favors repeatedly observed start-area
    voxels. Group by a coarser global grid instead, then retain the strongest
    representative only inside each cell. The output therefore continues to
    show newly mapped regions after the point limit is reached.
    """
    count = int(points.shape[0])
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64), float(base_voxel)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    if score.shape[0] != count:
        raise ValueError("semantic cloud score length does not match points")

    # Most outdoor semantic points lie on surfaces, so the square-root scale
    # is a good first estimate for the coarser grid. A few retries cover dense
    # vegetation and facade volumes without repeatedly sorting the cloud.
    selection_voxel = max(1e-3, float(base_voxel)) * max(
        1.0, np.sqrt(float(count) / float(max_points)))
    group_ids = None
    group_count = count
    for _ in range(3):
        keys = np.floor(points / selection_voxel).astype(np.int64)
        _, group_ids = unique_integer_rows(keys)
        group_count = int(group_ids.max()) + 1 if group_ids.size else 0
        if group_count <= max_points:
            break
        selection_voxel *= 1.35

    # ``lexsort`` orders by group first and score second, so the first member
    # of each group is its highest-confidence representative. It avoids the
    # unstable global score truncation that kept only the frequently revisited
    # origin region.
    order = np.lexsort((-score, group_ids))
    sorted_groups = group_ids[order]
    starts = np.empty((sorted_groups.shape[0],), dtype=bool)
    starts[0] = True
    starts[1:] = sorted_groups[1:] != sorted_groups[:-1]
    selected = order[np.flatnonzero(starts)]
    if selected.shape[0] > max_points:
        # This only occurs for unusually volumetric clouds after the bounded
        # retries above. Sampling ordered spatial groups still preserves map
        # extent, unlike a global confidence ranking.
        keep = np.linspace(0, selected.shape[0] - 1, max_points).astype(np.int64)
        selected = selected[keep]
    return selected.astype(np.int64, copy=False), float(selection_voxel)


class ProjectedSemanticBEVMapper(object):
    def __init__(self):
        self.queue_dir = Path(rospy.get_param("~queue_dir", "/tmp/sam3_projected_semantic_queue"))
        self.output_dir = self.queue_dir / "output"
        self.done_dir = self.queue_dir / "done"
        self.failed_dir = self.queue_dir / "failed_mapper"
        self.session_path = self.queue_dir / "queue_session.json"
        self.stale_dir = self.queue_dir / "stale_sessions"
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self.stale_dir.mkdir(parents=True, exist_ok=True)
        self.opencv_num_threads = max(
            1, int(rospy.get_param("~opencv_num_threads", 1)))
        cv2.setNumThreads(self.opencv_num_threads)
        cv2.ocl.setUseOpenCL(False)

        self.T_cam_lidar = as_T(rospy.get_param("~T_cam_lidar"), "T_cam_lidar")
        self.T_body_lidar = as_T(rospy.get_param("~T_body_lidar"), "T_body_lidar")
        self.use_distortion_projection = bool(rospy.get_param("~use_distortion_projection", True))
        self.min_project_depth_m = float(rospy.get_param("~min_project_depth_m", 0.2))
        self.enable_projection_depth_filter = bool(rospy.get_param("~enable_projection_depth_filter", True))
        self.projection_depth_tolerance_m = float(rospy.get_param("~projection_depth_tolerance_m", 0.80))

        # SuMa++-style semantic refinement.  Do not trust a single projected
        # image label directly: require local mask support, penalize projected
        # points on range discontinuities, and check the label against simple
        # LiDAR geometry before fusing it into the map.
        self.enable_semantic_support_filter = bool(rospy.get_param("~enable_semantic_support_filter", True))
        self.semantic_support_kernel_px = int(rospy.get_param("~semantic_support_kernel_px", 5))
        self.min_semantic_support_ratio = float(rospy.get_param("~min_semantic_support_ratio", 0.28))
        self.enable_range_edge_weight = bool(rospy.get_param("~enable_range_edge_weight", True))
        self.range_edge_kernel_px = int(rospy.get_param("~range_edge_kernel_px", 5))
        self.range_edge_soft_m = float(rospy.get_param("~range_edge_soft_m", 0.8))
        self.range_edge_hard_m = float(rospy.get_param("~range_edge_hard_m", 2.5))
        self.enable_lidar_geometry_refinement = bool(rospy.get_param("~enable_lidar_geometry_refinement", True))
        self.geometry_grid_resolution = float(rospy.get_param("~geometry_grid_resolution", 0.60))
        self.road_max_ground_height_m = float(rospy.get_param("~road_max_ground_height_m", 0.35))
        self.road_reject_ground_height_m = float(rospy.get_param("~road_reject_ground_height_m", 1.10))
        self.low_vegetation_max_ground_height_m = float(rospy.get_param("~low_vegetation_max_ground_height_m", 0.75))
        self.building_min_ground_height_m = float(rospy.get_param("~building_min_ground_height_m", 0.45))
        self.tree_min_ground_height_m = float(rospy.get_param("~tree_min_ground_height_m", 0.40))
        self.structure_min_z_span_m = float(rospy.get_param("~structure_min_z_span_m", 0.80))
        self.flat_cell_max_z_span_m = float(rospy.get_param("~flat_cell_max_z_span_m", 0.45))
        self.min_geometry_score = float(rospy.get_param("~min_geometry_score", 0.18))

        self.bev_resolution = float(rospy.get_param("~bev_resolution", 0.20))
        self.bev_size_m = float(rospy.get_param("~bev_size_m", 140.0))
        self.accumulation_window_sec = float(rospy.get_param("~accumulation_window_sec", 90.0))
        self.max_semantic_points = int(rospy.get_param("~max_semantic_points", 2500000))
        self.min_votes_per_cell = int(rospy.get_param("~min_votes_per_cell", 1))
        self.publish_rate = float(rospy.get_param("~publish_rate", 0.5))
        self.min_region_area_px = int(rospy.get_param("~min_region_area_px", 8))
        self.morph_open_kernel = int(rospy.get_param("~morph_open_kernel", 1))
        self.morph_close_kernel = int(rospy.get_param("~morph_close_kernel", 3))

        # v26: dense local semantic BEV rendering.  A single LiDAR scan only
        # produces sparse semantic points; for map matching we need local
        # semantic regions/boundaries.  These options expand semantic points
        # into local support regions and optionally fill road rays from the
        # robot center to road-labeled points.
        self.enable_dense_bev = bool(rospy.get_param("~enable_dense_bev", True))
        self.road_splat_radius_px = int(rospy.get_param("~road_splat_radius_px", 4))
        self.building_splat_radius_px = int(rospy.get_param("~building_splat_radius_px", 3))
        self.tree_splat_radius_px = int(rospy.get_param("~tree_splat_radius_px", 4))
        self.grass_splat_radius_px = int(rospy.get_param("~grass_splat_radius_px", 4))
        self.enable_road_ray_fill = bool(rospy.get_param("~enable_road_ray_fill", True))
        self.road_ray_max_range_m = float(rospy.get_param("~road_ray_max_range_m", 35.0))
        self.road_ray_weight = float(rospy.get_param("~road_ray_weight", 0.35))
        self.road_ray_max_points = int(rospy.get_param("~road_ray_max_points", 2500))
        self.center_mode = rospy.get_param("~bev_center_mode", "latest_pose")

        # 3D semantic point-cloud map. This is the primary output for semantic
        # SLAM debugging: raw projected labels are fused in voxels, conflicts are
        # resolved by weighted majority vote, and only confident voxels are shown.
        self.publish_semantic_cloud = bool(rospy.get_param("~publish_semantic_cloud", True))
        self.semantic_cloud_topic = rospy.get_param("~semantic_cloud_topic", "/semantic_cloud_map")
        self.semantic_cloud_frame = rospy.get_param("~semantic_cloud_frame", "camera_init")
        self.publish_projected_points = bool(rospy.get_param("~publish_projected_points", True))
        self.projected_points_topic = rospy.get_param("~projected_points_topic", "/sam3/projected_semantic_points")
        self.publish_lidar_frame_projected_points = bool(rospy.get_param("~publish_lidar_frame_projected_points", False))
        self.lidar_frame_projected_points_topic = rospy.get_param(
            "~lidar_frame_projected_points_topic", "/sam3/projected_semantic_points_lidar")
        self.semantic_cloud_output_label_mode = rospy.get_param("~semantic_cloud_output_label_mode", "sam3").strip().lower()
        if self.semantic_cloud_output_label_mode not in ("sam3", "internal"):
            rospy.logwarn("unknown semantic_cloud_output_label_mode=%s, fallback to sam3",
                          self.semantic_cloud_output_label_mode)
            self.semantic_cloud_output_label_mode = "sam3"
        self.semantic_cloud_voxel_size = float(rospy.get_param("~semantic_cloud_voxel_size", 0.15))
        self.semantic_cloud_min_votes = float(rospy.get_param("~semantic_cloud_min_votes", 1.0))
        self.semantic_cloud_min_confidence = float(rospy.get_param("~semantic_cloud_min_confidence", 0.55))
        self.semantic_cloud_max_points = int(rospy.get_param("~semantic_cloud_max_points", 600000))
        self.semantic_cloud_confidence_vote_scale = float(rospy.get_param("~semantic_cloud_confidence_vote_scale", 3.0))
        self.semantic_cloud_save_path = rospy.get_param("~semantic_cloud_save_path", "")
        self.semantic_cloud_save_every_sec = float(rospy.get_param("~semantic_cloud_save_every_sec", 0.0))
        self.last_semantic_cloud_save = 0.0
        self.require_queue_session = bool(rospy.get_param("~require_queue_session", True))
        self.poll_period_sec = 1.0 / max(
            1.0, float(rospy.get_param("~queue_poll_hz", 5.0)))

        label_weights = rospy.get_param("~label_weights", {})
        self.label_weights = {
            LABEL_ROAD: float(label_weights.get("road", 1.4)),
            LABEL_BUILDING: float(label_weights.get("building", 1.5)),
            LABEL_TREE: float(label_weights.get("tree", 1.0)),
            LABEL_GRASS: float(label_weights.get("grass", 0.8)),
            LABEL_DYNAMIC: float(label_weights.get("dynamic", 0.8)),
        }

        self.label_pub = rospy.Publisher(
            rospy.get_param("~label_topic", "/local_semantic_map/label"),
            Image, queue_size=1, latch=True)
        self.camera_label_pub = rospy.Publisher(
            rospy.get_param("~camera_label_topic", "/sam3/camera/label"),
            Image, queue_size=2)
        self.camera_dynamic_mask_pub = rospy.Publisher(
            rospy.get_param("~camera_dynamic_mask_topic", "/sam3/camera/dynamic_mask"),
            Image, queue_size=2)
        self.color_pub = rospy.Publisher(
            rospy.get_param("~color_topic", "/local_semantic_map/color"),
            Image, queue_size=1, latch=True)
        self.conf_pub = rospy.Publisher(
            rospy.get_param("~confidence_topic", "/local_semantic_map/confidence"),
            Image, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher(
            rospy.get_param("~debug_topic", "/local_semantic_map/debug"),
            Image, queue_size=1, latch=True)
        self.proj_debug_pub = rospy.Publisher(rospy.get_param("~projection_debug_topic", "/local_semantic_map/projection_debug"), Image, queue_size=1)
        self.proj_stats_pub = rospy.Publisher(rospy.get_param("~projection_stats_topic", "/local_semantic_map/projection_stats"), String, queue_size=1)
        self.publish_projection_debug = bool(rospy.get_param("~publish_projection_debug", True))
        self.projection_debug_max_points = int(rospy.get_param("~projection_debug_max_points", 12000))
        self.stats_pub = rospy.Publisher(
            rospy.get_param("~stats_topic", "/local_semantic_map/stats"),
            String, queue_size=1, latch=True)
        self.semantic_cloud_pub = rospy.Publisher(
            self.semantic_cloud_topic, PointCloud2, queue_size=1, latch=True)
        self.semantic_cloud_stats_pub = rospy.Publisher(
            rospy.get_param(
                "~semantic_cloud_stats_topic", "/semantic_cloud_map/stats"),
            String, queue_size=1, latch=True)
        self.projected_points_pub = rospy.Publisher(self.projected_points_topic, PointCloud2, queue_size=1)
        self.lidar_frame_projected_points_pub = rospy.Publisher(self.lidar_frame_projected_points_topic, PointCloud2, queue_size=1)

        self.semantic_batches = deque()  # {time, pose, pts, labels, weights, request_id}
        self.latest_T_map_body = np.eye(4)
        self.processed = set()
        self.last_pub = 0.0
        self.last_queue_wait_log = 0.0
        self.map_dirty = False
        self.queue_session_id = ""

        self.refresh_queue_session(force=True)
        self.reconcile_completed_metadata()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("v37 projected_semantic_bev_mapper started session=%s queue=%s projected_points=%s semantic_cloud=%s voxel=%.2fm output_label_mode=%s lidar_debug=%s opencv_threads=%d",
                      self.queue_session_id or "waiting",
                      str(self.queue_dir), self.projected_points_topic, self.semantic_cloud_topic,
                      self.semantic_cloud_voxel_size, self.semantic_cloud_output_label_mode,
                      self.lidar_frame_projected_points_topic if self.publish_lidar_frame_projected_points else "disabled",
                      self.opencv_num_threads)

    @staticmethod
    def _sanitize_session_component(value):
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))

    def load_queue_session_id(self):
        try:
            with open(str(self.session_path), "r") as stream:
                return str(json.load(stream).get("session_id", ""))
        except Exception:
            return ""

    def publish_empty_semantic_cloud(self):
        if not self.publish_semantic_cloud:
            return
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.uint8)
        empty_confidence = np.empty((0,), dtype=np.float32)
        self.semantic_cloud_pub.publish(self.semantic_points_to_msg(
            empty_points, empty_labels, empty_confidence, rospy.Time.now()))

    def refresh_queue_session(self, force=False):
        active_session = self.load_queue_session_id()
        if self.require_queue_session and not active_session:
            return False
        if not force and active_session == self.queue_session_id:
            return bool(active_session) or not self.require_queue_session
        previous_session = self.queue_session_id
        self.queue_session_id = active_session
        if previous_session and previous_session != active_session:
            self.semantic_batches.clear()
            self.processed.clear()
            self.map_dirty = False
            self.publish_empty_semantic_cloud()
            rospy.loginfo("SAM3 mapper switched queue session old=%s new=%s; cleared old map state",
                          previous_session, active_session or "waiting")
        elif active_session:
            rospy.loginfo("SAM3 mapper using queue session=%s", active_session)
        return bool(active_session) or not self.require_queue_session

    def archive_stale_result(self, result_path, result_session_id, reason):
        label = self._sanitize_session_component(result_session_id or "no_session")
        destination = self.stale_dir / label / "output"
        destination.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(str(result_path), str(destination / result_path.name))
            rospy.logwarn("ignored stale SAM3 result=%s session=%s active=%s reason=%s",
                          result_path.name, result_session_id or "missing",
                          self.queue_session_id or "waiting", reason)
        except OSError:
            pass

    def current_session_control_count(self, directory, pattern):
        count = 0
        for path in directory.glob(pattern):
            try:
                with open(str(path), "r") as stream:
                    if json.load(stream).get("queue_session_id", "") == self.queue_session_id:
                        count += 1
            except Exception:
                continue
        return count

    def reconcile_completed_metadata(self):
        segmented_dir = self.queue_dir / "segmented"
        if not segmented_dir.exists():
            return
        moved = 0
        for ready_path in segmented_dir.glob("ready_*.json"):
            try:
                with open(str(ready_path), "r") as stream:
                    if self.require_queue_session and (
                            json.load(stream).get("queue_session_id", "") != self.queue_session_id):
                        continue
            except Exception:
                continue
            req_id = ready_path.stem[len("ready_"):]
            if not (self.done_dir / ("result_%s.json" % req_id)).exists():
                continue
            try:
                os.replace(str(ready_path), str(self.done_dir / ready_path.name))
                moved += 1
            except Exception:
                pass
        if moved:
            rospy.loginfo("reconciled %d completed SAM3 metadata files into %s",
                          moved, str(self.done_dir))

    def read_result(self, result_path):
        with open(str(result_path), "r") as f:
            result = json.load(f)
        req_id = result.get("id")
        if not req_id:
            raise RuntimeError("result has no id")
        return req_id, result

    def publish_projection_debug_image(self, req_id, meta, label_img, ui_all, vi_all, front_mask, inside_mask, sem_mask_inside, labels_inside, counts, status_text):
        if not self.publish_projection_debug:
            return
        try:
            image_path = Path(meta.get("image_path", ""))
            if image_path.exists():
                canvas = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            else:
                # fallback: colorize label map when original image is unavailable
                canvas = colorize_label(label_img)
            if canvas is None:
                canvas = colorize_label(label_img)

            h, w = canvas.shape[:2]
            overlay = canvas.copy()
            label_color = {
                LABEL_ROAD: (80, 80, 80),
                LABEL_BUILDING: (0, 0, 255),
                LABEL_TREE: (0, 170, 0),
                LABEL_GRASS: (0, 220, 120),
                LABEL_DYNAMIC: (0, 255, 255),
            }

            # Draw sampled projected points. Unknown points are blue; semantic points use class color.
            if ui_all is not None and vi_all is not None and inside_mask is not None:
                idx_inside = np.where(inside_mask)[0]
                if idx_inside.size > 0:
                    if idx_inside.size > self.projection_debug_max_points:
                        step = int(np.ceil(float(idx_inside.size) / float(self.projection_debug_max_points)))
                        idx_inside = idx_inside[::step]
                    # Need labels corresponding to inside points. Build a fast dictionary index into inside ordering.
                    # We draw unknown first in cyan/blue.
                    for idx in idx_inside:
                        u = int(ui_all[idx]); v = int(vi_all[idx])
                        if 0 <= u < w and 0 <= v < h:
                            cv2.circle(overlay, (u, v), 1, (255, 120, 0), -1)

            # Draw semantic points larger, using label colors.
            if ui_all is not None and vi_all is not None and inside_mask is not None and labels_inside is not None:
                idx_inside = np.where(inside_mask)[0]
                if idx_inside.size > 0:
                    labs = np.asarray(labels_inside)
                    if labs.shape[0] == idx_inside.shape[0]:
                        sem_pos = np.where(labs > 0)[0]
                        if sem_pos.size > self.projection_debug_max_points:
                            step = int(np.ceil(float(sem_pos.size) / float(self.projection_debug_max_points)))
                            sem_pos = sem_pos[::step]
                        for j in sem_pos:
                            idx = idx_inside[j]
                            lab = int(labs[j])
                            u = int(ui_all[idx]); v = int(vi_all[idx])
                            if 0 <= u < w and 0 <= v < h:
                                cv2.circle(overlay, (u, v), 2, label_color.get(lab, (255,255,255)), -1)

            out = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0.0)

            lines = [
                "id=%s" % req_id,
                status_text,
                "counts=%s" % str(counts),
            ]
            y = 24
            for line in lines:
                cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y += 24

            msg = numpy_to_image(
                out, "bgr8", rospy.Time.now(), meta.get("camera_frame_id", "camera"))
            self.proj_debug_pub.publish(msg)
        except Exception as e:
            rospy.logwarn("failed to publish projection debug image: %s", e)

    def publish_projection_stats(self, req_id, raw_pts, front_pts, inside_pts, semantic_pts, counts, status):
        try:
            d = {
                "id": str(req_id),
                "status": str(status),
                "raw_pts": int(raw_pts),
                "front_pts": int(front_pts),
                "inside_img_pts": int(inside_pts),
                "semantic_pts": int(semantic_pts),
                "counts": {str(k): int(v) for k, v in counts.items()},
            }
            self.proj_stats_pub.publish(String(data=json.dumps(d)))
        except Exception:
            pass

    @staticmethod
    def odd_kernel(value):
        k = max(1, int(value))
        if k % 2 == 0:
            k += 1
        return k

    def semantic_label_support(self, label_img, vi, ui):
        labels = label_img[vi, ui].astype(np.uint8)
        support = np.ones(labels.shape, dtype=np.float32)
        if not self.enable_semantic_support_filter:
            return labels, support

        k = self.odd_kernel(self.semantic_support_kernel_px)
        class_labels = np.asarray(SAM3_LABELS, dtype=np.uint8)
        support_stack = []
        for lab in class_labels:
            mask = (label_img == lab).astype(np.float32)
            support_stack.append(cv2.blur(mask, (k, k)))
        support_stack = np.stack(support_stack, axis=0)
        support_at = support_stack[:, vi, ui].T
        best_idx = np.argmax(support_at, axis=1)
        best_support = support_at[np.arange(support_at.shape[0]), best_idx].astype(np.float32)
        best_labels = class_labels[best_idx].astype(np.uint8)

        labels = np.where(best_support >= self.min_semantic_support_ratio, best_labels, LABEL_UNKNOWN).astype(np.uint8)
        return labels, best_support

    def range_edge_scores(self, depth_min, vi, ui):
        if not self.enable_range_edge_weight:
            return np.ones((ui.shape[0],), dtype=np.float32)
        valid_depth = np.isfinite(depth_min)
        if not valid_depth.any():
            return np.ones((ui.shape[0],), dtype=np.float32)
        k = self.odd_kernel(self.range_edge_kernel_px)
        large = np.float32(1.0e6)
        near = np.where(valid_depth, depth_min, large).astype(np.float32)
        far = np.where(valid_depth, depth_min, 0.0).astype(np.float32)
        local_min = cv2.erode(near, np.ones((k, k), np.uint8), iterations=1)
        local_max = cv2.dilate(far, np.ones((k, k), np.uint8), iterations=1)
        span = local_max[vi, ui] - local_min[vi, ui]
        span = np.maximum(span.astype(np.float32), 0.0)
        soft = max(1e-3, self.range_edge_soft_m)
        hard = max(soft + 1e-3, self.range_edge_hard_m)
        score = 1.0 - np.clip((span - soft) / (hard - soft), 0.0, 1.0) * 0.75
        return score.astype(np.float32)

    def build_lidar_grid_stats(self, pts_map):
        if not self.enable_lidar_geometry_refinement:
            return None
        pts = np.asarray(pts_map, dtype=np.float32)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if pts.shape[0] < 50:
            return None
        res = max(0.10, self.geometry_grid_resolution)
        keys = np.floor(pts[:, :2] / res).astype(np.int64)
        unique_keys, inverse = unique_integer_rows(keys)
        ncell = unique_keys.shape[0]
        z = pts[:, 2].astype(np.float32)
        zmin, zmax, count = grouped_min_max_count(inverse, z, ncell)
        lookup = {
            (int(unique_keys[i, 0]), int(unique_keys[i, 1])): i
            for i in range(ncell)
        }
        return {
            "res": res,
            "lookup": lookup,
            "zmin": zmin,
            "zmax": zmax,
            "count": count,
        }

    def semantic_geometry_scores(self, pts_map, labels, grid_stats):
        n = pts_map.shape[0]
        score = np.ones((n,), dtype=np.float32)
        keep = np.ones((n,), dtype=bool)
        info = {
            "geometry_known": 0,
            "geometry_rejected": 0,
            "road_high_rejected": 0,
        }
        if (not self.enable_lidar_geometry_refinement) or grid_stats is None or n == 0:
            return score, keep, info

        res = grid_stats["res"]
        qkeys = np.floor(pts_map[:, :2] / res).astype(np.int64)
        ground = np.full((n,), np.nan, dtype=np.float32)
        zspan = np.full((n,), np.nan, dtype=np.float32)
        cell_count = np.zeros((n,), dtype=np.int32)
        lookup = grid_stats["lookup"]
        zmin = grid_stats["zmin"]
        zmax = grid_stats["zmax"]
        count = grid_stats["count"]
        indices = np.fromiter(
            (lookup.get((int(key[0]), int(key[1])), -1) for key in qkeys),
            dtype=np.int64, count=n)
        found = (indices >= 0) & (indices < zmin.shape[0])
        found_indices = indices[found]
        ground[found] = zmin[found_indices]
        zspan[found] = zmax[found_indices] - zmin[found_indices]
        cell_count[found] = count[found_indices]

        known = np.isfinite(ground)
        info["geometry_known"] = int(known.sum())
        if not known.any():
            return score, keep, info
        dz = pts_map[:, 2].astype(np.float32) - ground
        span = np.where(np.isfinite(zspan), zspan, 0.0).astype(np.float32)

        road = known & (labels == LABEL_ROAD)
        if road.any():
            high = road & (dz > self.road_max_ground_height_m)
            denom = max(1e-3, self.road_reject_ground_height_m - self.road_max_ground_height_m)
            road_soft = 1.0 - np.clip((dz - self.road_max_ground_height_m) / denom, 0.0, 1.0) * 0.85
            score[road] *= road_soft[road]
            rough_road = road & (span > self.flat_cell_max_z_span_m) & (dz > 0.15)
            score[rough_road] *= 0.45
            reject = road & (dz > self.road_reject_ground_height_m)
            keep[reject] = False
            info["road_high_rejected"] = int(reject.sum())

        grass = known & (labels == LABEL_GRASS)
        if grass.any():
            high = grass & (dz > self.low_vegetation_max_ground_height_m)
            score[high] *= 0.35
            rough = grass & (span > self.flat_cell_max_z_span_m * 1.8)
            score[rough] *= 0.65

        building = known & (labels == LABEL_BUILDING)
        if building.any():
            low = building & (dz < self.building_min_ground_height_m)
            score[low] *= 0.40
            weak_vertical = building & (span < self.structure_min_z_span_m)
            score[weak_vertical] *= 0.55

        tree = known & (labels == LABEL_TREE)
        if tree.any():
            low = tree & (dz < self.tree_min_ground_height_m)
            score[low] *= 0.50
            weak_roughness = tree & (span < self.structure_min_z_span_m * 0.7)
            score[weak_roughness] *= 0.75

        weak = score < self.min_geometry_score
        keep[weak] = False
        info["geometry_rejected"] = int((~keep).sum())
        return score.astype(np.float32), keep, info

    def project_and_accumulate(self, result_path):
        req_id, result = self.read_result(result_path)
        result_session_id = str(result.get("queue_session_id", ""))
        if self.require_queue_session and result_session_id != self.queue_session_id:
            self.archive_stale_result(result_path, result_session_id, "result_session_mismatch")
            return
        if req_id in self.processed:
            return

        meta_path = Path(result.get("meta_path", ""))
        if not meta_path.exists():
            # v25 robust fallbacks. The segmentation service may move ready_<id>.json
            # from input/ to segmented/ before the mapper consumes result_<id>.json.
            candidates = [
                self.output_dir / ("meta_%s.json" % req_id),
                self.queue_dir / "input" / ("ready_%s.json" % req_id),
                self.queue_dir / "segmented" / ("ready_%s.json" % req_id),
                self.queue_dir / "done" / ("ready_%s.json" % req_id),
            ]
            for c in candidates:
                if c.exists():
                    meta_path = c
                    break
        if not meta_path.exists():
            raise RuntimeError("missing meta for id=%s; looked in output/meta, input, segmented and done" % req_id)

        with open(str(meta_path), "r") as f:
            meta = json.load(f)
        meta_session_id = str(meta.get("queue_session_id", ""))
        if self.require_queue_session and meta_session_id != self.queue_session_id:
            self.archive_stale_result(result_path, meta_session_id, "metadata_session_mismatch")
            return

        label_path = Path(result.get("label_path", ""))
        if not label_path.exists():
            label_path = self.output_dir / ("label_%s.png" % req_id)
        if not label_path.exists():
            raise RuntimeError("missing label image for id=%s" % req_id)

        cloud_path = Path(meta["cloud_path"])
        if not cloud_path.exists():
            raise RuntimeError("missing cloud npz for id=%s" % req_id)

        label_img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label_img is None:
            raise RuntimeError("failed reading label image %s" % label_path)
        if label_img.ndim == 3:
            label_img = label_img[:, :, 0]
        label_img = label_img.astype(np.uint8)

        source_stamp = meta.get("stamp", {}) or {}
        camera_stamp = rospy.Time(
            int(source_stamp.get("secs", 0)), int(source_stamp.get("nsecs", 0)))
        if camera_stamp == rospy.Time():
            try:
                camera_stamp = rospy.Time.from_sec(float(req_id) * 1.0e-9)
            except Exception:
                camera_stamp = rospy.Time.now()
        camera_frame = meta.get("camera_frame_id", "camera")
        self.camera_label_pub.publish(numpy_to_image(
            label_img, "mono8", camera_stamp, camera_frame))
        dynamic_mask = (label_img == LABEL_DYNAMIC).astype(np.uint8) * 255
        self.camera_dynamic_mask_pub.publish(numpy_to_image(
            dynamic_mask, "mono8", camera_stamp, camera_frame))

        instance_path = Path(result.get("instance_path", ""))
        if not instance_path.exists():
            instance_path = self.output_dir / ("instance_%s.png" % req_id)
        instance_img = None
        if instance_path.exists():
            instance_img = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
            if instance_img is not None and instance_img.ndim == 3:
                instance_img = instance_img[:, :, 0]
            if instance_img is not None and instance_img.shape[:2] != label_img.shape[:2]:
                rospy.logwarn("instance image shape mismatch for id=%s, ignoring instance ids", req_id)
                instance_img = None

        data = np.load(str(cloud_path))
        xyz = data["xyz"].astype(np.float64)
        K = data["K"].astype(np.float64)
        cloud_in_map_frame = bool(meta.get("cloud_in_map_frame", False))
        if not cloud_in_map_frame and "cloud_in_map_frame" in data.files:
            try:
                cloud_in_map_frame = bool(data["cloud_in_map_frame"])
            except Exception:
                cloud_in_map_frame = False
        odom_diag = meta.get("odom_diag", {}) or {}
        receipt_age = odom_diag.get("odom_receipt_age_sec", None)
        odom_mode = str(odom_diag.get("odom_match_mode", ""))
        odom_stamp_dt = float(odom_diag.get("odom_stamp_dt", 0.0))
        if (odom_mode != "stamp" and receipt_age is not None and
                abs(float(receipt_age)) > 0.50):
            rospy.logwarn_throttle(
                2.0,
                "SAM3 projected mapper is using an old odom for id=%s: receipt_age=%.3fs stamp_dt=%.3fs. "
                "Map-frame semantic points can look rotated/scattered.",
                req_id, float(receipt_age), odom_stamp_dt)
        T_map_body = np.asarray(meta["T_map_body"], dtype=np.float64).reshape(4, 4)
        self.latest_T_map_body = T_map_body.copy()
        T_map_lidar = T_map_body.dot(self.T_body_lidar)

        # RangeNet/SAM3 can now consume FAST-LIVO2 /cloud_registered. Those points
        # are already in map/camera_init, so project by first converting them back
        # to the current LiDAR frame; publish/fuse them in their original map frame.
        n = xyz.shape[0]
        xyz_h = np.ones((n, 4), dtype=np.float64)
        xyz_h[:, :3] = xyz
        if cloud_in_map_frame:
            T_lidar_map = np.linalg.inv(T_map_lidar)
            xyz_lidar = (T_lidar_map.dot(xyz_h.T)).T[:, :3]
            xyz_lidar_h = np.ones((n, 4), dtype=np.float64)
            xyz_lidar_h[:, :3] = xyz_lidar
            pts_map_all = xyz.astype(np.float32)
        else:
            xyz_lidar = xyz
            xyz_lidar_h = xyz_h
            pts_map_all = (T_map_lidar.dot(xyz_h.T)).T[:, :3].astype(np.float32)

        # p_cam = T_cam_lidar * p_lidar
        p_cam = (self.T_cam_lidar.dot(xyz_lidar_h.T)).T[:, :3]
        z = p_cam[:, 2]
        valid = z > self.min_project_depth_m
        if valid.sum() < 50:
            raise RuntimeError("too few points in front of camera. Check T_cam_lidar direction/order.")

        p_cam_v = p_cam[valid]

        h, w = label_img.shape[:2]
        D = data["D"].astype(np.float64) if "D" in data.files else np.zeros((0,), dtype=np.float64)
        if self.use_distortion_projection and D.size >= 4:
            # p_cam_v are already in camera frame, so project with rvec=tvec=0.
            img_pts, _ = cv2.projectPoints(
                p_cam_v.reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                K,
                D.reshape(-1, 1),
            )
            img_pts = img_pts.reshape(-1, 2)
            u = img_pts[:, 0]
            v = img_pts[:, 1]
        else:
            u = K[0, 0] * p_cam_v[:, 0] / p_cam_v[:, 2] + K[0, 2]
            v = K[1, 1] * p_cam_v[:, 1] / p_cam_v[:, 2] + K[1, 2]

        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        labels_inside = None
        depth_ok_inside = None
        support_inside = None
        depth_score_inside = None
        range_score_inside = None
        instance_inside = None
        counts = {}
        status_text = "raw=%d front=%d inside=%d sem=0" % (n, int(valid.sum()), int(inside.sum()))
        if inside.sum() > 0:
            ui_inside = ui[inside]
            vi_inside = vi[inside]
            labels_inside, support_inside = self.semantic_label_support(label_img, vi_inside, ui_inside)
            if instance_img is not None:
                instance_inside = instance_img[vi_inside, ui_inside].astype(np.uint32)
            else:
                instance_inside = np.zeros(labels_inside.shape, dtype=np.uint32)

            need_depth_image = self.enable_projection_depth_filter or self.enable_range_edge_weight
            if need_depth_image:
                z_inside = p_cam_v[:, 2][inside]
                flat_pixels = (
                    vi_inside.astype(np.int64) * int(w) +
                    ui_inside.astype(np.int64))
                depth_min = grouped_minimum(
                    flat_pixels, z_inside, int(h * w), np.inf).reshape(h, w)
                depth_ref = depth_min[vi_inside, ui_inside]
                if self.enable_projection_depth_filter:
                    depth_delta = z_inside - depth_ref
                    depth_ok_inside = z_inside <= (depth_ref + self.projection_depth_tolerance_m)
                    tol = max(1e-3, self.projection_depth_tolerance_m)
                    depth_score_inside = 1.0 - np.clip(depth_delta / tol, 0.0, 1.0).astype(np.float32) * 0.60
                else:
                    depth_ok_inside = np.ones(labels_inside.shape, dtype=bool)
                    depth_score_inside = np.ones(labels_inside.shape, dtype=np.float32)
                range_score_inside = self.range_edge_scores(depth_min, vi_inside, ui_inside)
            else:
                depth_ok_inside = np.ones(labels_inside.shape, dtype=bool)
                depth_score_inside = np.ones(labels_inside.shape, dtype=np.float32)
                range_score_inside = np.ones(labels_inside.shape, dtype=np.float32)
            for kk in np.unique(labels_inside):
                if int(kk) > 0:
                    counts[int(kk)] = int(((labels_inside == kk) & depth_ok_inside).sum())
            status_text = "raw=%d front=%d inside=%d depth=%d sem=%d" % (
                n, int(valid.sum()), int(inside.sum()), int(depth_ok_inside.sum()),
                int(((labels_inside > 0) & depth_ok_inside).sum()))
        self.publish_projection_debug_image(req_id, meta, label_img, ui, vi, valid, inside, None, labels_inside, counts, status_text)
        self.publish_projection_stats(
            req_id, n, int(valid.sum()), int(inside.sum()),
            int(((labels_inside > 0) & depth_ok_inside).sum()) if labels_inside is not None and depth_ok_inside is not None else 0,
            counts, "projected")

        if inside.sum() < 50:
            raise RuntimeError("too few projected points inside image. Check K/D, image_scale and T_cam_lidar.")

        valid_orig_idx = np.where(valid)[0]
        inside_orig_idx = valid_orig_idx[inside]
        labels = labels_inside
        sem_mask = (labels > 0) & depth_ok_inside
        if sem_mask.sum() < 30:
            raise RuntimeError("too few semantic LiDAR points")

        sem_orig_idx = inside_orig_idx[sem_mask]
        labels = labels[sem_mask].astype(np.uint8)
        instance_ids = instance_inside[sem_mask].astype(np.uint32) if instance_inside is not None else np.zeros(labels.shape, dtype=np.uint32)
        support_sem = support_inside[sem_mask].astype(np.float32) if support_inside is not None else np.ones(labels.shape, dtype=np.float32)
        depth_score_sem = depth_score_inside[sem_mask].astype(np.float32) if depth_score_inside is not None else np.ones(labels.shape, dtype=np.float32)
        range_score_sem = range_score_inside[sem_mask].astype(np.float32) if range_score_inside is not None else np.ones(labels.shape, dtype=np.float32)

        pts_map = pts_map_all[sem_orig_idx]
        pts_lidar = xyz_lidar[sem_orig_idx].astype(np.float32)

        grid_stats = self.build_lidar_grid_stats(pts_map_all)
        geom_score, geom_keep, geom_info = self.semantic_geometry_scores(pts_map, labels, grid_stats)
        if geom_keep.sum() < 30:
            raise RuntimeError("too few semantic LiDAR points after geometry refinement")
        labels = labels[geom_keep]
        instance_ids = instance_ids[geom_keep]
        pts_map = pts_map[geom_keep]
        pts_lidar = pts_lidar[geom_keep]
        support_sem = support_sem[geom_keep]
        depth_score_sem = depth_score_sem[geom_keep]
        range_score_sem = range_score_sem[geom_keep]
        geom_score = geom_score[geom_keep]

        weights = np.clip(support_sem, 0.05, 1.0) * \
            np.clip(depth_score_sem, 0.05, 1.0) * \
            np.clip(range_score_sem, 0.05, 1.0) * \
            np.clip(geom_score, 0.05, 1.0)
        weights = weights.astype(np.float32)
        for lab, wt in self.label_weights.items():
            weights[labels == lab] *= wt

        stamp_sec = meta.get("stamp", {}).get("secs", None)
        stamp_nsec = meta.get("stamp", {}).get("nsecs", 0)
        if stamp_sec is None:
            t = time.time()
        else:
            t = float(stamp_sec) + float(stamp_nsec) * 1e-9

        if self.publish_projected_points and self.projected_points_pub.get_num_connections() >= 0:
            msg_stamp = rospy.Time.from_sec(t) if t > 0.0 else rospy.Time.now()
            confidence = np.clip(weights, 0.05, 1.0).astype(np.float32)
            msg = self.semantic_points_to_msg(pts_map, labels, confidence, msg_stamp, instance_ids)
            self.projected_points_pub.publish(msg)
            if self.publish_lidar_frame_projected_points:
                lidar_frame = "current_lidar" if cloud_in_map_frame else meta.get("lidar_frame_id", "")
                if not lidar_frame:
                    lidar_frame = "lidar"
                lidar_msg = self.semantic_points_to_msg(
                    pts_lidar, labels, confidence, msg_stamp, instance_ids, frame_id=lidar_frame)
                self.lidar_frame_projected_points_pub.publish(lidar_msg)

        self.semantic_batches.append({
            "time": t,
            "pose": T_map_body[:3, 3].astype(np.float64).copy(),
            "pts": pts_map,
            "labels": labels,
            "weights": weights,
            "request_id": req_id,
        })
        self.trim_batches(t)
        self.map_dirty = True
        self.processed.add(req_id)

        try:
            os.replace(str(result_path), str(self.done_dir / result_path.name))
        except Exception:
            pass
        segmented_meta = self.queue_dir / "segmented" / ("ready_%s.json" % req_id)
        if segmented_meta.exists():
            try:
                os.replace(str(segmented_meta),
                           str(self.done_dir / segmented_meta.name))
            except Exception:
                pass

        counts = {int(k): int((labels == k).sum()) for k in np.unique(labels)}
        rospy.loginfo("semantic projected id=%s stamp=%.6f pose=[%.3f %.3f %.3f] "
                      "raw_pts=%d in_img=%d sem_pts=%d refined=%d cloud_frame=%s counts=%s geom=%s",
                      req_id, t, T_map_body[0, 3], T_map_body[1, 3],
                      T_map_body[2, 3],
                      n, int(inside.sum()), int(sem_mask.sum()), int(labels.shape[0]),
                      "map" if cloud_in_map_frame else "lidar", counts, geom_info)

    def trim_batches(self, current_t):
        if self.accumulation_window_sec > 0.0:
            while (self.semantic_batches and
                   current_t - self.semantic_batches[0]["time"] > self.accumulation_window_sec):
                self.semantic_batches.popleft()
        # Keep global max point count bounded.
        total = sum(b["pts"].shape[0] for b in self.semantic_batches)
        while self.semantic_batches and total > self.max_semantic_points:
            b = self.semantic_batches.popleft()
            total -= b["pts"].shape[0]

    def semantic_batch_diagnostics(self):
        if not self.semantic_batches:
            return {
                "oldest_stamp": None,
                "latest_stamp": None,
                "stamp_span_sec": 0.0,
                "latest_request_id": "",
                "latest_pose_xyz": [],
                "pose_extent_m": 0.0,
                "pose_path_length_m": 0.0,
            }
        times = np.asarray([b["time"] for b in self.semantic_batches],
                           dtype=np.float64)
        poses = np.asarray([
            b.get("pose", np.zeros((3,), dtype=np.float64))
            for b in self.semantic_batches
        ], dtype=np.float64).reshape(-1, 3)
        pose_min = poses.min(axis=0)
        pose_max = poses.max(axis=0)
        path_length = 0.0
        if poses.shape[0] > 1:
            path_length = float(
                np.linalg.norm(np.diff(poses, axis=0), axis=1).sum())
        return {
            "oldest_stamp": float(times[0]),
            "latest_stamp": float(times[-1]),
            "stamp_span_sec": float(max(0.0, times[-1] - times[0])),
            "latest_request_id": str(
                self.semantic_batches[-1].get("request_id", "")),
            "latest_pose_xyz": [float(v) for v in poses[-1]],
            "pose_bounds_min_xyz": [float(v) for v in pose_min],
            "pose_bounds_max_xyz": [float(v) for v in pose_max],
            "pose_extent_m": float(np.linalg.norm(pose_max - pose_min)),
            "pose_path_length_m": path_length,
        }

    def build_semantic_cloud(self):
        if not self.semantic_batches:
            return None
        pts = np.concatenate([b["pts"] for b in self.semantic_batches], axis=0).astype(np.float32)
        labels = np.concatenate([b["labels"] for b in self.semantic_batches], axis=0).astype(np.uint8)
        weights = np.concatenate([b["weights"] for b in self.semantic_batches], axis=0).astype(np.float32)
        valid = np.isfinite(pts).all(axis=1) & (labels > 0) & (labels <= LABEL_DYNAMIC) & (weights > 0)
        pts = pts[valid]
        labels = labels[valid]
        weights = weights[valid]
        if pts.shape[0] < 30:
            return None

        voxel = max(0.03, self.semantic_cloud_voxel_size)
        keys = np.floor(pts / voxel).astype(np.int64)
        unique_keys, inverse = unique_integer_rows(keys)
        nvox = unique_keys.shape[0]
        vote = np.zeros((LABEL_DYNAMIC + 1, nvox), dtype=np.float32)
        weights64 = weights.astype(np.float64)
        sum_w = np.bincount(inverse, weights=weights64, minlength=nvox)
        sum_xyz = np.empty((nvox, 3), dtype=np.float64)
        for axis in range(3):
            sum_xyz[:, axis] = np.bincount(
                inverse, weights=pts[:, axis].astype(np.float64) * weights64,
                minlength=nvox)
        for lab in SAM3_LABELS:
            m = labels == lab
            if m.any():
                vote[lab] = np.bincount(
                    inverse[m], weights=weights64[m], minlength=nvox).astype(
                        np.float32, copy=False)

        best_label = np.argmax(vote, axis=0).astype(np.uint8)
        best_vote = vote[best_label, np.arange(nvox)]
        total_vote = vote.sum(axis=0)
        sorted_vote = np.sort(vote, axis=0)
        second_vote = sorted_vote[-2]
        confidence = np.zeros((nvox,), dtype=np.float32)
        nz = total_vote > 1e-6
        purity = np.zeros((nvox,), dtype=np.float32)
        margin = np.zeros((nvox,), dtype=np.float32)
        vote_mag = np.zeros((nvox,), dtype=np.float32)
        purity[nz] = best_vote[nz] / np.maximum(total_vote[nz], 1e-6)
        margin[nz] = (best_vote[nz] - second_vote[nz]) / np.maximum(total_vote[nz], 1e-6)
        scale = max(1e-3, self.semantic_cloud_confidence_vote_scale)
        vote_mag[nz] = 1.0 - np.exp(-total_vote[nz] / scale)
        confidence[nz] = np.clip((0.65 * purity[nz] + 0.35 * margin[nz]) * (0.45 + 0.55 * vote_mag[nz]), 0.0, 1.0)
        keep = (best_label > 0) & \
            (total_vote >= self.semantic_cloud_min_votes) & \
            (confidence >= self.semantic_cloud_min_confidence) & \
            (sum_w > 1e-6)
        if not np.any(keep):
            return None

        pts_out = (sum_xyz[keep] / np.maximum(sum_w[keep, None], 1e-6)).astype(np.float32)
        labels_out = best_label[keep].astype(np.uint8)
        conf_out = confidence[keep].astype(np.float32)
        votes_out = total_vote[keep].astype(np.float32)

        pre_limit_points = int(pts_out.shape[0])
        selection_voxel = voxel
        if self.semantic_cloud_max_points > 0 and pts_out.shape[0] > self.semantic_cloud_max_points:
            score = votes_out * conf_out
            idx, selection_voxel = spatially_select_cloud_indices(
                pts_out, score, self.semantic_cloud_max_points, voxel)
            pts_out = pts_out[idx]
            labels_out = labels_out[idx]
            conf_out = conf_out[idx]
            votes_out = votes_out[idx]

        counts = {int(lab): int((labels_out == lab).sum()) for lab in SAM3_LABELS}
        return {
            "pts": pts_out,
            "labels": labels_out,
            "confidence": conf_out,
            "votes": votes_out,
            "raw_points": int(pts.shape[0]),
            "voxels": int(nvox),
            "pre_limit_points": pre_limit_points,
            "selection_voxel": float(selection_voxel),
            "counts": counts,
        }

    def semantic_points_to_msg(self, pts, labels, confidence, stamp, instance_ids=None, frame_id=None):
        if self.semantic_cloud_output_label_mode == "internal":
            labels_msg = np.zeros_like(labels, dtype=np.uint32)
            for sam3_label, internal_label in SAM3_TO_INTERNAL_LABEL.items():
                labels_msg[labels == sam3_label] = internal_label
            color_table = INTERNAL_LABEL_RGB_FLOAT
        else:
            labels_msg = labels.astype(np.uint32)
            color_table = LABEL_RGB_FLOAT
        n = pts.shape[0]
        include_instance = instance_ids is not None
        if include_instance:
            instance_ids = np.asarray(instance_ids, dtype=np.uint32).reshape(-1)
            if instance_ids.shape[0] != n:
                instance_ids = np.zeros((n,), dtype=np.uint32)
        arr = np.zeros(n, dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("rgb", "<f4"),
            ("label", "<u4"),
            ("confidence", "<f4"),
        ] + ([("instance_id", "<u4")] if include_instance else []))
        arr["x"] = pts[:, 0]
        arr["y"] = pts[:, 1]
        arr["z"] = pts[:, 2]
        rgb = np.zeros((n,), dtype=np.float32)
        for lab, rgb_float in color_table.items():
            rgb[labels_msg == lab] = rgb_float
        arr["rgb"] = rgb
        arr["label"] = labels_msg
        arr["confidence"] = confidence.astype(np.float32)
        if include_instance:
            arr["instance_id"] = instance_ids

        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=frame_id or self.semantic_cloud_frame)
        msg.height = 1
        msg.width = n
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
            PointField("label", 16, PointField.UINT32, 1),
            PointField("confidence", 20, PointField.FLOAT32, 1),
        ]
        if include_instance:
            fields.append(PointField("instance_id", 24, PointField.UINT32, 1))
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = arr.dtype.itemsize
        msg.row_step = msg.point_step * n
        msg.is_dense = True
        msg.data = arr.tobytes()
        return msg

    def semantic_cloud_to_msg(self, cloud, stamp):
        return self.semantic_points_to_msg(
            cloud["pts"], cloud["labels"], cloud["confidence"], stamp, None)

    def save_semantic_cloud_ply(self, cloud):
        path = self.semantic_cloud_save_path
        if not path:
            return
        now = time.time()
        if self.semantic_cloud_save_every_sec > 0.0 and now - self.last_semantic_cloud_save < self.semantic_cloud_save_every_sec:
            return
        self.last_semantic_cloud_save = now
        pts = cloud["pts"]
        labels = cloud["labels"]
        conf = cloud["confidence"]
        n = pts.shape[0]
        out = np.zeros(n, dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("label", "u1"), ("confidence", "<f4"),
        ])
        out["x"] = pts[:, 0]
        out["y"] = pts[:, 1]
        out["z"] = pts[:, 2]
        for lab, bgr in COLOR_BGR.items():
            m = labels == lab
            out["red"][m] = bgr[2]
            out["green"][m] = bgr[1]
            out["blue"][m] = bgr[0]
        out["label"] = labels
        out["confidence"] = conf
        tmp = path + ".tmp"
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex %d\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property uchar label\n"
            "property float confidence\n"
            "end_header\n"
        ) % n
        with open(tmp, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(out.tobytes())
        os.replace(tmp, path)

    def publish_semantic_cloud_map(self, stamp):
        if not self.publish_semantic_cloud:
            return None
        cloud = self.build_semantic_cloud()
        if cloud is None:
            return None
        batch_diag = self.semantic_batch_diagnostics()
        cloud_stamp = stamp
        latest_stamp = batch_diag.get("latest_stamp")
        if latest_stamp is not None and latest_stamp > 0.0:
            cloud_stamp = rospy.Time.from_sec(latest_stamp)
        msg = self.semantic_cloud_to_msg(cloud, cloud_stamp)
        self.semantic_cloud_pub.publish(msg)
        try:
            self.save_semantic_cloud_ply(cloud)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "failed saving semantic cloud ply: %s", e)
        stats = {
            "queue_session_id": self.queue_session_id,
            "points": int(cloud["pts"].shape[0]),
            "raw_projected_points": int(cloud["raw_points"]),
            "voxels": int(cloud["voxels"]),
            "pre_limit_points": int(cloud["pre_limit_points"]),
            "selection_voxel": float(cloud["selection_voxel"]),
            "voxel_size": float(self.semantic_cloud_voxel_size),
            "min_votes": float(self.semantic_cloud_min_votes),
            "min_confidence": float(self.semantic_cloud_min_confidence),
            "confidence_vote_scale": float(self.semantic_cloud_confidence_vote_scale),
            "counts": {str(k): int(v) for k, v in cloud["counts"].items()},
            "batches": int(len(self.semantic_batches)),
        }
        stats.update(batch_diag)
        self.semantic_cloud_stats_pub.publish(String(data=json.dumps(stats)))
        rospy.loginfo_throttle(
            5.0,
            "semantic cloud points=%d pre_limit=%d raw=%d voxels=%d select_voxel=%.2f batches=%d latest_stamp=%s "
            "latest_pose=%s path=%.2fm road=%d building=%d tree=%d grass=%d dynamic=%d",
            stats["points"], stats["pre_limit_points"], stats["raw_projected_points"], stats["voxels"],
            stats["selection_voxel"],
            stats["batches"],
            "none" if stats["latest_stamp"] is None else
            "%.6f" % stats["latest_stamp"],
            stats["latest_pose_xyz"], stats["pose_path_length_m"],
            cloud["counts"].get(LABEL_ROAD, 0), cloud["counts"].get(LABEL_BUILDING, 0),
            cloud["counts"].get(LABEL_TREE, 0), cloud["counts"].get(LABEL_GRASS, 0),
            cloud["counts"].get(LABEL_DYNAMIC, 0),
        )
        return cloud

    def _dilate_vote_map(self, vote_map, radius_px):
        if radius_px <= 0:
            return vote_map
        ksize = int(radius_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        # cv2.dilate on float32 performs local maximum spreading, which is
        # suitable for turning sparse LiDAR semantic points into BEV support.
        return cv2.dilate(vote_map.astype(np.float32), kernel, iterations=1)

    def _fill_road_rays(self, vote, ix, iy, labs, wts, size_px, center_ix, center_iy, res):
        if not self.enable_road_ray_fill:
            return
        road_mask = labs == LABEL_ROAD
        if not road_mask.any():
            return
        r_ix = ix[road_mask]
        r_iy = iy[road_mask]
        r_wt = wts[road_mask]
        dx = (r_ix.astype(np.float32) - float(center_ix)) * res
        dy = (r_iy.astype(np.float32) - float(center_iy)) * res
        rr = np.sqrt(dx * dx + dy * dy)
        keep = rr < self.road_ray_max_range_m
        r_ix, r_iy, r_wt = r_ix[keep], r_iy[keep], r_wt[keep]
        if r_ix.size == 0:
            return
        # Subsample rays to keep CPU cost bounded.
        if r_ix.size > self.road_ray_max_points:
            step = int(np.ceil(float(r_ix.size) / float(self.road_ray_max_points)))
            r_ix, r_iy, r_wt = r_ix[::step], r_iy[::step], r_wt[::step]
        ray_layer = np.zeros((size_px, size_px), dtype=np.float32)
        for x, y, wt in zip(r_ix, r_iy, r_wt):
            cv2.line(ray_layer, (int(center_ix), int(center_iy)), (int(x), int(y)),
                     float(max(0.05, wt) * self.road_ray_weight), 1, cv2.LINE_AA)
        vote[LABEL_ROAD] = np.maximum(vote[LABEL_ROAD], ray_layer)

    def build_bev(self):
        if not self.semantic_batches:
            return None
        pts = np.concatenate([b["pts"] for b in self.semantic_batches], axis=0)
        labels = np.concatenate([b["labels"] for b in self.semantic_batches], axis=0)
        weights = np.concatenate([b["weights"] for b in self.semantic_batches], axis=0)
        if pts.shape[0] < 30:
            return None

        if self.center_mode == "point_centroid" and pts.shape[0] > 0:
            center = np.median(pts[:, :2], axis=0)
        else:
            center = self.latest_T_map_body[:2, 3]
        half = self.bev_size_m * 0.5
        res = self.bev_resolution
        size_px = int(round(self.bev_size_m / res))
        size_px = max(64, size_px)

        dx = pts[:, 0] - center[0]
        dy = pts[:, 1] - center[1]
        in_roi = (dx >= -half) & (dx < half) & (dy >= -half) & (dy < half)
        if in_roi.sum() < 30:
            rospy.logwarn_throttle(5.0, "semantic points outside BEV ROI: in_roi=%d total=%d center=(%.2f %.2f). Check T_map_body/T_body_lidar or reduce bev_size/center_mode.",
                                   int(in_roi.sum()), int(pts.shape[0]), float(center[0]), float(center[1]))
            return None
        dx = dx[in_roi]
        dy = dy[in_roi]
        labs = labels[in_roi]
        wts = weights[in_roi]

        ix = np.floor((dx + half) / res).astype(np.int32)
        iy = np.floor((half - dy) / res).astype(np.int32)
        valid = (ix >= 0) & (ix < size_px) & (iy >= 0) & (iy < size_px)
        ix, iy, labs, wts = ix[valid], iy[valid], labs[valid], wts[valid]
        if ix.size < 30:
            return None

        vote_sparse = np.zeros((LABEL_DYNAMIC + 1, size_px, size_px), dtype=np.float32)
        flat_pixels = iy.astype(np.int64) * int(size_px) + ix.astype(np.int64)
        for lab in BEV_LABELS:
            m = labs == lab
            if m.any():
                vote_sparse[lab] = np.bincount(
                    flat_pixels[m], weights=wts[m].astype(np.float64),
                    minlength=size_px * size_px).reshape(size_px, size_px).astype(
                        np.float32, copy=False)

        vote = vote_sparse.copy()
        if self.enable_dense_bev:
            # Expand each semantic class differently. Road/grass need wider support
            # because LiDAR ground returns are sparse; building/tree are boundary-like.
            radii = {
                LABEL_ROAD: self.road_splat_radius_px,
                LABEL_BUILDING: self.building_splat_radius_px,
                LABEL_TREE: self.tree_splat_radius_px,
                LABEL_GRASS: self.grass_splat_radius_px,
            }
            for lab, rad in radii.items():
                vote[lab] = np.maximum(vote[lab], self._dilate_vote_map(vote_sparse[lab], rad))

            # Fill drivable rays from the robot center to road-labeled LiDAR hits;
            # this converts sparse road points into a local drivable-area semantic layer.
            center_ix = size_px // 2
            center_iy = size_px // 2
            self._fill_road_rays(vote, ix, iy, labs, wts, size_px, center_ix, center_iy, res)

        total_vote = vote.sum(axis=0)
        label = np.argmax(vote, axis=0).astype(np.uint8)
        label[total_vote < self.min_votes_per_cell] = LABEL_UNKNOWN

        # Morphological clean by class. Keep this conservative in debug mode;
        # aggressive opening removes sparse but valid scan-line semantics.
        cleaned = np.zeros_like(label)
        for lab in BEV_LABELS:
            mask = (label == lab).astype(np.uint8) * 255
            if self.morph_open_kernel > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_open_kernel, self.morph_open_kernel))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            if self.morph_close_kernel > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_close_kernel, self.morph_close_kernel))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            num, cc, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_region_area_px:
                    cleaned[cc == i] = lab
        label = cleaned

        max_vote = np.max(vote, axis=0)
        conf = np.zeros((size_px, size_px), dtype=np.uint8)
        nonzero = total_vote > 0
        ratio = np.zeros_like(total_vote)
        ratio[nonzero] = max_vote[nonzero] / np.maximum(total_vote[nonzero], 1e-3)
        # Also include vote magnitude so very weak dilated pixels are not overconfident.
        mag = np.zeros_like(total_vote)
        mag[nonzero] = np.clip(total_vote[nonzero] / 3.0, 0.0, 1.0)
        conf[nonzero] = np.clip((0.65 * ratio[nonzero] + 0.35 * mag[nonzero]) * 255.0, 0, 255).astype(np.uint8)
        conf[label == LABEL_UNKNOWN] = 0

        color = colorize_label(label)
        sparse_color = colorize_label(np.argmax(vote_sparse, axis=0).astype(np.uint8))
        sparse_color[vote_sparse.sum(axis=0) <= 0] = COLOR_BGR[LABEL_UNKNOWN]
        extra = {
            "in_roi_points": int(ix.size),
            "sparse_fg_ratio": float((vote_sparse.sum(axis=0) > 0).mean()),
        }
        return label, color, conf, pts.shape[0], sparse_color, extra

    def publish_bev(self):
        out = self.build_bev()
        if out is None:
            return False
        if len(out) == 4:
            label, color, conf, total_pts = out
            sparse_color = None
            extra = {}
        else:
            label, color, conf, total_pts, sparse_color, extra = out
        stamp = rospy.Time.now()
        label_msg = numpy_to_image(label, "mono8", stamp, self.semantic_cloud_frame)
        color_msg = numpy_to_image(color, "bgr8", stamp, self.semantic_cloud_frame)
        conf_msg = numpy_to_image(conf, "mono8", stamp, self.semantic_cloud_frame)
        self.label_pub.publish(label_msg)
        self.color_pub.publish(color_msg)
        self.conf_pub.publish(conf_msg)

        # Debug: concatenate sparse semantic points, dense semantic map and confidence heatmap.
        heat = cv2.applyColorMap(conf, cv2.COLORMAP_JET)
        if sparse_color is not None:
            debug = np.hstack([sparse_color, color, heat])
        else:
            debug = np.hstack([color, heat])
        debug_msg = numpy_to_image(debug, "bgr8", stamp, self.semantic_cloud_frame)
        self.debug_pub.publish(debug_msg)

        stats = {
            "total_accumulated_points": int(total_pts),
            "fg_ratio": float((label > 0).mean()),
            "road_px": int((label == LABEL_ROAD).sum()),
            "building_px": int((label == LABEL_BUILDING).sum()),
            "tree_px": int((label == LABEL_TREE).sum()),
            "grass_px": int((label == LABEL_GRASS).sum()),
            "mean_confidence": float(conf[label > 0].mean()) if (label > 0).any() else 0.0,
            "batches": len(self.semantic_batches),
            "in_roi_points": int(extra.get("in_roi_points", 0)),
            "sparse_fg_ratio": float(extra.get("sparse_fg_ratio", 0.0)),
            "dense_bev": bool(self.enable_dense_bev),
            "road_ray_fill": bool(self.enable_road_ray_fill),
        }
        stats.update(self.semantic_batch_diagnostics())
        self.stats_pub.publish(String(data=json.dumps(stats)))
        rospy.loginfo_throttle(5.0, "projected semantic map pts=%d fg=%.3f road=%d building=%d tree=%d grass=%d conf=%.1f batches=%d",
                               stats["total_accumulated_points"], stats["fg_ratio"], stats["road_px"],
                               stats["building_px"], stats["tree_px"], stats["grass_px"],
                               stats["mean_confidence"], stats["batches"])
        self.publish_semantic_cloud_map(stamp)
        return True

    def timer_cb(self, event):
        if not self.refresh_queue_session():
            rospy.logwarn_throttle(
                5.0, "semantic mapper waiting for camera_lidar_queue_exporter session marker: %s",
                str(self.session_path))
            return
        results = sorted(self.output_dir.glob("result_*.json"))
        if not results:
            now_log = time.time()
            if now_log - self.last_queue_wait_log > 5.0:
                pending = self.current_session_control_count(
                    self.queue_dir / "input", "ready_*.json")
                segmented = self.current_session_control_count(
                    self.queue_dir / "segmented", "ready_*.json")
                failed = self.current_session_control_count(
                    self.queue_dir / "failed_sam3", "ready_*.json")
                if pending > 0 or segmented > 0:
                    rospy.logwarn(
                        "semantic mapper has no completed SAM3 result right now: output/result_*.json=0 "
                        "input_ready=%d segmented=%d failed_sam3=%d. SAM3 is slower than the exporter or is still "
                        "processing backlog; this is normal while the queue drains.",
                        pending, segmented, failed)
                elif not self.processed:
                    rospy.logwarn(
                        "semantic mapper waiting for SAM3 results: output/result_*.json=0 "
                        "input_ready=%d segmented=%d failed_sam3=%d. Start camera_lidar_queue_exporter and "
                        "sam3_image_mask_service.py in the sam3 environment.",
                        pending, segmented, failed)
                self.last_queue_wait_log = now_log
        for rp in results[:5]:
            try:
                self.project_and_accumulate(rp)
            except Exception as e:
                rospy.logwarn("failed processing semantic result %s: %s", str(rp), e)
                try:
                    os.replace(str(rp), str(self.failed_dir / rp.name))
                except Exception:
                    pass
        now = time.time()
        publish_due = (
            self.publish_rate <= 0.0 or
            now - self.last_pub >= 1.0 / self.publish_rate)
        if self.map_dirty and publish_due:
            self.last_pub = now
            if not self.publish_bev():
                self.publish_semantic_cloud_map(rospy.Time.now())
            self.map_dirty = False

    def run(self):
        """Poll the queue on the process foreground thread.

        OpenCV's image decode/projection operations are no longer invoked from
        a background thread.  This avoids the native crashes observed after
        several dozen asynchronous SAM3 results on AT128 bags.
        """
        while not rospy.is_shutdown():
            try:
                self.timer_cb(None)
            except Exception as error:
                rospy.logerr_throttle(
                    2.0, "projected semantic mapper foreground loop failed: %s", error)
            time.sleep(self.poll_period_sec)

    def shutdown(self):
        pass


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    rospy.init_node("projected_semantic_bev_mapper")
    rospy.loginfo("semantic mapper runtime: numpy=%s (%s) cv2=%s (%s)",
                  np.__version__, np.__file__, cv2.__version__, cv2.__file__)
    node = ProjectedSemanticBEVMapper()
    node.run()
