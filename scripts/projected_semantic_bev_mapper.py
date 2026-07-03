#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS-side semantic BEV mapper.

Reads SAM3 label results from file queue. For each result:
  camera label map + saved LiDAR cloud + saved camera intrinsics + odometry
  -> project LiDAR points into camera label map
  -> assign semantic labels to 3D points
  -> transform points to FAST-LIVO2 map frame
  -> accumulate sliding-window semantic points
  -> publish local semantic BEV map.
"""

import json
import os
import struct
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String


LABEL_UNKNOWN = 0
LABEL_ROAD = 1
LABEL_BUILDING = 2
LABEL_TREE = 3
LABEL_GRASS = 4

COLOR_BGR = {
    LABEL_UNKNOWN: (190, 190, 190),
    LABEL_ROAD: (80, 80, 80),
    LABEL_BUILDING: (0, 0, 255),
    LABEL_TREE: (0, 170, 0),
    LABEL_GRASS: (0, 220, 120),
}


def rgb_float_from_bgr(bgr):
    b, g, r = [int(x) & 255 for x in bgr]
    rgb_u32 = (r << 16) | (g << 8) | b
    return struct.unpack("f", struct.pack("I", rgb_u32))[0]


LABEL_RGB_FLOAT = {
    lab: rgb_float_from_bgr(bgr) for lab, bgr in COLOR_BGR.items()
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


class ProjectedSemanticBEVMapper(object):
    def __init__(self):
        self.queue_dir = Path(rospy.get_param("~queue_dir", "/tmp/sam3_projected_semantic_queue"))
        self.output_dir = self.queue_dir / "output"
        self.done_dir = self.queue_dir / "done"
        self.failed_dir = self.queue_dir / "failed_mapper"
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

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
        self.semantic_cloud_voxel_size = float(rospy.get_param("~semantic_cloud_voxel_size", 0.15))
        self.semantic_cloud_min_votes = float(rospy.get_param("~semantic_cloud_min_votes", 1.0))
        self.semantic_cloud_min_confidence = float(rospy.get_param("~semantic_cloud_min_confidence", 0.55))
        self.semantic_cloud_max_points = int(rospy.get_param("~semantic_cloud_max_points", 600000))
        self.semantic_cloud_confidence_vote_scale = float(rospy.get_param("~semantic_cloud_confidence_vote_scale", 3.0))
        self.semantic_cloud_save_path = rospy.get_param("~semantic_cloud_save_path", "")
        self.semantic_cloud_save_every_sec = float(rospy.get_param("~semantic_cloud_save_every_sec", 0.0))
        self.last_semantic_cloud_save = 0.0

        label_weights = rospy.get_param("~label_weights", {})
        self.label_weights = {
            LABEL_ROAD: float(label_weights.get("road", 1.4)),
            LABEL_BUILDING: float(label_weights.get("building", 1.5)),
            LABEL_TREE: float(label_weights.get("tree", 1.0)),
            LABEL_GRASS: float(label_weights.get("grass", 0.8)),
        }

        self.bridge = CvBridge()
        self.label_pub = rospy.Publisher(rospy.get_param("~label_topic", "/local_semantic_map/label"), Image, queue_size=1)
        self.color_pub = rospy.Publisher(rospy.get_param("~color_topic", "/local_semantic_map/color"), Image, queue_size=1)
        self.conf_pub = rospy.Publisher(rospy.get_param("~confidence_topic", "/local_semantic_map/confidence"), Image, queue_size=1)
        self.debug_pub = rospy.Publisher(rospy.get_param("~debug_topic", "/local_semantic_map/debug"), Image, queue_size=1)
        self.proj_debug_pub = rospy.Publisher(rospy.get_param("~projection_debug_topic", "/local_semantic_map/projection_debug"), Image, queue_size=1)
        self.proj_stats_pub = rospy.Publisher(rospy.get_param("~projection_stats_topic", "/local_semantic_map/projection_stats"), String, queue_size=1)
        self.publish_projection_debug = bool(rospy.get_param("~publish_projection_debug", True))
        self.projection_debug_max_points = int(rospy.get_param("~projection_debug_max_points", 12000))
        self.stats_pub = rospy.Publisher(rospy.get_param("~stats_topic", "/local_semantic_map/stats"), String, queue_size=1)
        self.semantic_cloud_pub = rospy.Publisher(self.semantic_cloud_topic, PointCloud2, queue_size=1)
        self.semantic_cloud_stats_pub = rospy.Publisher(rospy.get_param("~semantic_cloud_stats_topic", "/semantic_cloud_map/stats"), String, queue_size=1)

        self.semantic_batches = deque()  # list of dict: {time, pts_map, labels, weights}
        self.latest_T_map_body = np.eye(4)
        self.processed = set()
        self.last_pub = 0.0
        self.last_queue_wait_log = 0.0

        rospy.Timer(rospy.Duration(0.2), self.timer_cb)
        rospy.loginfo("v27 projected_semantic_bev_mapper started queue=%s semantic_cloud=%s voxel=%.2fm",
                      str(self.queue_dir), self.semantic_cloud_topic, self.semantic_cloud_voxel_size)

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

            msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = meta.get("camera_frame_id", "camera")
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
        class_labels = np.asarray([LABEL_ROAD, LABEL_BUILDING, LABEL_TREE, LABEL_GRASS], dtype=np.uint8)
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
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        ncell = unique_keys.shape[0]
        zmin = np.full((ncell,), np.inf, dtype=np.float32)
        zmax = np.full((ncell,), -np.inf, dtype=np.float32)
        count = np.zeros((ncell,), dtype=np.int32)
        z = pts[:, 2].astype(np.float32)
        np.minimum.at(zmin, inverse, z)
        np.maximum.at(zmax, inverse, z)
        np.add.at(count, inverse, 1)
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
        for i, key in enumerate(qkeys):
            idx = lookup.get((int(key[0]), int(key[1])))
            if idx is None:
                continue
            ground[i] = zmin[idx]
            zspan[i] = zmax[idx] - zmin[idx]
            cell_count[i] = count[idx]

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

        data = np.load(str(cloud_path))
        xyz = data["xyz"].astype(np.float64)
        K = data["K"].astype(np.float64)
        T_map_body = np.asarray(meta["T_map_body"], dtype=np.float64).reshape(4, 4)
        self.latest_T_map_body = T_map_body.copy()
        T_map_lidar = T_map_body.dot(self.T_body_lidar)

        # p_cam = T_cam_lidar * p_lidar
        n = xyz.shape[0]
        xyz_h = np.ones((n, 4), dtype=np.float64)
        xyz_h[:, :3] = xyz
        p_cam = (self.T_cam_lidar.dot(xyz_h.T)).T[:, :3]
        z = p_cam[:, 2]
        valid = z > self.min_project_depth_m
        if valid.sum() < 50:
            raise RuntimeError("too few points in front of camera. Check T_cam_lidar direction/order.")

        p_cam_v = p_cam[valid]
        xyz_h_v = xyz_h[valid]

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
        counts = {}
        status_text = "raw=%d front=%d inside=%d sem=0" % (n, int(valid.sum()), int(inside.sum()))
        if inside.sum() > 0:
            ui_inside = ui[inside]
            vi_inside = vi[inside]
            labels_inside, support_inside = self.semantic_label_support(label_img, vi_inside, ui_inside)

            need_depth_image = self.enable_projection_depth_filter or self.enable_range_edge_weight
            if need_depth_image:
                z_inside = p_cam_v[:, 2][inside]
                depth_min = np.full((h, w), np.inf, dtype=np.float32)
                np.minimum.at(depth_min, (vi_inside, ui_inside), z_inside.astype(np.float32))
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
        support_sem = support_inside[sem_mask].astype(np.float32) if support_inside is not None else np.ones(labels.shape, dtype=np.float32)
        depth_score_sem = depth_score_inside[sem_mask].astype(np.float32) if depth_score_inside is not None else np.ones(labels.shape, dtype=np.float32)
        range_score_sem = range_score_inside[sem_mask].astype(np.float32) if range_score_inside is not None else np.ones(labels.shape, dtype=np.float32)

        pts_map_all = (T_map_lidar.dot(xyz_h.T)).T[:, :3].astype(np.float32)
        pts_map = pts_map_all[sem_orig_idx]

        grid_stats = self.build_lidar_grid_stats(pts_map_all)
        geom_score, geom_keep, geom_info = self.semantic_geometry_scores(pts_map, labels, grid_stats)
        if geom_keep.sum() < 30:
            raise RuntimeError("too few semantic LiDAR points after geometry refinement")
        labels = labels[geom_keep]
        pts_map = pts_map[geom_keep]
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

        self.semantic_batches.append({"time": t, "pts": pts_map, "labels": labels, "weights": weights})
        self.trim_batches(t)
        self.processed.add(req_id)

        try:
            os.replace(str(result_path), str(self.done_dir / result_path.name))
        except Exception:
            pass

        counts = {int(k): int((labels == k).sum()) for k in np.unique(labels)}
        rospy.loginfo("semantic projected id=%s raw_pts=%d in_img=%d sem_pts=%d refined=%d counts=%s geom=%s",
                      req_id, n, int(inside.sum()), int(sem_mask.sum()), int(labels.shape[0]), counts, geom_info)

    def trim_batches(self, current_t):
        while self.semantic_batches and current_t - self.semantic_batches[0]["time"] > self.accumulation_window_sec:
            self.semantic_batches.popleft()
        # Keep global max point count bounded.
        total = sum(b["pts"].shape[0] for b in self.semantic_batches)
        while self.semantic_batches and total > self.max_semantic_points:
            b = self.semantic_batches.popleft()
            total -= b["pts"].shape[0]

    def build_semantic_cloud(self):
        if not self.semantic_batches:
            return None
        pts = np.concatenate([b["pts"] for b in self.semantic_batches], axis=0).astype(np.float32)
        labels = np.concatenate([b["labels"] for b in self.semantic_batches], axis=0).astype(np.uint8)
        weights = np.concatenate([b["weights"] for b in self.semantic_batches], axis=0).astype(np.float32)
        valid = np.isfinite(pts).all(axis=1) & (labels > 0) & (labels <= LABEL_GRASS) & (weights > 0)
        pts = pts[valid]
        labels = labels[valid]
        weights = weights[valid]
        if pts.shape[0] < 30:
            return None

        voxel = max(0.03, self.semantic_cloud_voxel_size)
        keys = np.floor(pts / voxel).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        nvox = unique_keys.shape[0]
        vote = np.zeros((LABEL_GRASS + 1, nvox), dtype=np.float32)
        sum_xyz = np.zeros((nvox, 3), dtype=np.float64)
        sum_w = np.zeros((nvox,), dtype=np.float64)

        np.add.at(sum_xyz, inverse, pts.astype(np.float64) * weights[:, None].astype(np.float64))
        np.add.at(sum_w, inverse, weights.astype(np.float64))
        for lab in [LABEL_ROAD, LABEL_BUILDING, LABEL_TREE, LABEL_GRASS]:
            m = labels == lab
            if m.any():
                np.add.at(vote[lab], inverse[m], weights[m])

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

        if self.semantic_cloud_max_points > 0 and pts_out.shape[0] > self.semantic_cloud_max_points:
            score = votes_out * conf_out
            idx = np.argpartition(score, -self.semantic_cloud_max_points)[-self.semantic_cloud_max_points:]
            order = np.argsort(score[idx])[::-1]
            idx = idx[order]
            pts_out = pts_out[idx]
            labels_out = labels_out[idx]
            conf_out = conf_out[idx]
            votes_out = votes_out[idx]

        counts = {int(lab): int((labels_out == lab).sum()) for lab in [LABEL_ROAD, LABEL_BUILDING, LABEL_TREE, LABEL_GRASS]}
        return {
            "pts": pts_out,
            "labels": labels_out,
            "confidence": conf_out,
            "votes": votes_out,
            "raw_points": int(pts.shape[0]),
            "voxels": int(nvox),
            "counts": counts,
        }

    def semantic_cloud_to_msg(self, cloud, stamp):
        pts = cloud["pts"]
        labels = cloud["labels"]
        confidence = cloud["confidence"]
        n = pts.shape[0]
        arr = np.zeros(n, dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("rgb", "<f4"),
            ("label", "<u4"),
            ("confidence", "<f4"),
        ])
        arr["x"] = pts[:, 0]
        arr["y"] = pts[:, 1]
        arr["z"] = pts[:, 2]
        rgb = np.zeros((n,), dtype=np.float32)
        for lab, rgb_float in LABEL_RGB_FLOAT.items():
            rgb[labels == lab] = rgb_float
        arr["rgb"] = rgb
        arr["label"] = labels.astype(np.uint32)
        arr["confidence"] = confidence.astype(np.float32)

        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=self.semantic_cloud_frame)
        msg.height = 1
        msg.width = n
        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
            PointField("label", 16, PointField.UINT32, 1),
            PointField("confidence", 20, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = arr.dtype.itemsize
        msg.row_step = msg.point_step * n
        msg.is_dense = True
        msg.data = arr.tobytes()
        return msg

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
        msg = self.semantic_cloud_to_msg(cloud, stamp)
        self.semantic_cloud_pub.publish(msg)
        try:
            self.save_semantic_cloud_ply(cloud)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "failed saving semantic cloud ply: %s", e)
        stats = {
            "points": int(cloud["pts"].shape[0]),
            "raw_projected_points": int(cloud["raw_points"]),
            "voxels": int(cloud["voxels"]),
            "voxel_size": float(self.semantic_cloud_voxel_size),
            "min_votes": float(self.semantic_cloud_min_votes),
            "min_confidence": float(self.semantic_cloud_min_confidence),
            "confidence_vote_scale": float(self.semantic_cloud_confidence_vote_scale),
            "counts": {str(k): int(v) for k, v in cloud["counts"].items()},
            "batches": int(len(self.semantic_batches)),
        }
        self.semantic_cloud_stats_pub.publish(String(data=json.dumps(stats)))
        rospy.loginfo_throttle(
            5.0,
            "semantic cloud points=%d raw=%d voxels=%d road=%d building=%d tree=%d grass=%d",
            stats["points"], stats["raw_projected_points"], stats["voxels"],
            cloud["counts"].get(LABEL_ROAD, 0), cloud["counts"].get(LABEL_BUILDING, 0),
            cloud["counts"].get(LABEL_TREE, 0), cloud["counts"].get(LABEL_GRASS, 0),
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

        vote_sparse = np.zeros((5, size_px, size_px), dtype=np.float32)
        for lab in [LABEL_ROAD, LABEL_BUILDING, LABEL_TREE, LABEL_GRASS]:
            m = labs == lab
            if m.any():
                np.add.at(vote_sparse[lab], (iy[m], ix[m]), wts[m])

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
        for lab in [LABEL_ROAD, LABEL_BUILDING, LABEL_TREE, LABEL_GRASS]:
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
        label_msg = self.bridge.cv2_to_imgmsg(label, encoding="mono8")
        color_msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
        conf_msg = self.bridge.cv2_to_imgmsg(conf, encoding="mono8")
        label_msg.header.stamp = stamp
        color_msg.header.stamp = stamp
        conf_msg.header.stamp = stamp
        label_msg.header.frame_id = "camera_init"
        color_msg.header.frame_id = "camera_init"
        conf_msg.header.frame_id = "camera_init"
        self.label_pub.publish(label_msg)
        self.color_pub.publish(color_msg)
        self.conf_pub.publish(conf_msg)

        # Debug: concatenate sparse semantic points, dense semantic map and confidence heatmap.
        heat = cv2.applyColorMap(conf, cv2.COLORMAP_JET)
        if sparse_color is not None:
            debug = np.hstack([sparse_color, color, heat])
        else:
            debug = np.hstack([color, heat])
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header.stamp = stamp
        debug_msg.header.frame_id = "camera_init"
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
        self.stats_pub.publish(String(data=json.dumps(stats)))
        rospy.loginfo_throttle(5.0, "projected semantic map pts=%d fg=%.3f road=%d building=%d tree=%d grass=%d conf=%.1f batches=%d",
                               stats["total_accumulated_points"], stats["fg_ratio"], stats["road_px"],
                               stats["building_px"], stats["tree_px"], stats["grass_px"],
                               stats["mean_confidence"], stats["batches"])
        self.publish_semantic_cloud_map(stamp)
        return True

    def timer_cb(self, event):
        results = sorted(self.output_dir.glob("result_*.json"))
        if not results:
            now_log = time.time()
            if now_log - self.last_queue_wait_log > 5.0:
                pending = len(list((self.queue_dir / "input").glob("ready_*.json")))
                segmented = len(list((self.queue_dir / "segmented").glob("ready_*.json")))
                failed = len(list((self.queue_dir / "failed_sam3").glob("ready_*.json")))
                rospy.logwarn(
                    "semantic mapper waiting for SAM3 results: output/result_*.json=0 input_ready=%d segmented=%d failed_sam3=%d. "
                    "Start sam3_image_mask_service.py in the sam3 environment.",
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
        if self.publish_rate <= 0.0 or now - self.last_pub >= 1.0 / self.publish_rate:
            self.last_pub = now
            if not self.publish_bev():
                self.publish_semantic_cloud_map(rospy.Time.now())


if __name__ == "__main__":
    rospy.init_node("projected_semantic_bev_mapper")
    node = ProjectedSemanticBEVMapper()
    rospy.spin()
