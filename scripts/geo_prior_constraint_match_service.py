#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured local BEV vs multi-source geographic-prior constrained matcher.

This service replaces direct local-BEV-to-satellite-RGB feature matching.  It
reads queue requests exported by eloftr_queue_exporter.py, consumes the local
semantic BEV labels, builds or loads geographic-prior layers, searches low-rate
dx/dy/yaw candidates, and writes result_*.json for eloftr_queue_result_publisher.py.

Prior inputs are optional and can be mixed:
  --prior_yaml           georeferenced prior mosaic metadata, same simple format
                         as satellite_mosaic.yml: image_path, center_lat,
                         center_lon, meters_per_pixel
  --road_mask_path       full or crop-aligned binary road/drivable prior
  --road_network_mask_path full or crop-aligned road-network centerline prior
  --road_surface_mask_path full or crop-aligned road-surface prior
  --building_mask_path   full or crop-aligned binary building/CAD/BIM footprint prior
  --traversable_mask_path full or crop-aligned traversable/open-area prior
  --vegetation_mask_path full or crop-aligned vegetation prior
  --entrance_csv_path    CSV with either east,north or lat,lon entrance points

If no explicit prior masks are supplied, the service derives weak road/building/
vegetation priors from sat_rgb_path and sat_path so the pipeline can still run
for diagnosis.  For the final method, provide real OSM/CAD/BIM/entrance priors.
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys
import time

if sys.version_info[0] < 3:
    sys.stderr.write("geo_prior_constraint_match_service.py requires Python 3. Use python3, not python.\n")
    sys.exit(2)

from pathlib import Path

import cv2
import numpy as np

try:
    from sam3_semantic_match_service import (
        Sam3ConceptSegmenter,
        LABEL_ROAD as SAM_LABEL_ROAD,
        LABEL_BUILDING as SAM_LABEL_BUILDING,
    )
except Exception:
    Sam3ConceptSegmenter = None
    SAM_LABEL_ROAD = 2
    SAM_LABEL_BUILDING = 3


LOCAL_UNKNOWN = 0
LOCAL_OPEN = 1
LOCAL_STRUCTURE = 2
LOCAL_VEGETATION = 3
LOCAL_OBSTACLE = 4
LOCAL_ROAD_EDGE = 5
LOCAL_LANE_LINE = 6

PRIOR_UNKNOWN = 0
PRIOR_ROAD = 1
PRIOR_BUILDING = 2
PRIOR_TRAVERSABLE = 3
PRIOR_VEGETATION = 4
PRIOR_ENTRANCE = 5
PRIOR_ROAD_EDGE = 6
PRIOR_LANE_LINE = 7

LOCAL_COLORS = {
    LOCAL_UNKNOWN: (30, 30, 30),
    LOCAL_OPEN: (210, 210, 210),
    LOCAL_STRUCTURE: (0, 0, 230),
    LOCAL_VEGETATION: (40, 180, 40),
    LOCAL_OBSTACLE: (0, 160, 255),
    LOCAL_ROAD_EDGE: (255, 255, 255),
    LOCAL_LANE_LINE: (255, 255, 0),
}

PRIOR_COLORS = {
    PRIOR_UNKNOWN: (20, 20, 20),
    PRIOR_ROAD: (80, 80, 80),
    PRIOR_BUILDING: (30, 30, 230),
    PRIOR_TRAVERSABLE: (190, 190, 190),
    PRIOR_VEGETATION: (40, 180, 40),
    PRIOR_ENTRANCE: (255, 0, 255),
    PRIOR_ROAD_EDGE: (255, 255, 255),
    PRIOR_LANE_LINE: (255, 255, 0),
}


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json_atomic(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


def read_simple_yaml(path):
    out = {}
    if not path:
        return out
    try:
        with open(path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                val = val.strip().strip("\"'")
                if not key.strip() or key.strip().startswith("#"):
                    continue
                out[key.strip()] = val
    except OSError:
        return {}
    return out


def meta_dict(req):
    meta = req.get("meta", [])
    names = req.get("meta_layout", [])
    return {names[i]: float(meta[i]) for i in range(min(len(meta), len(names)))}


def ensure_gray_u8(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def colorize(label, colors):
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for lab, col in colors.items():
        out[label == lab] = col
    return out


def normalize_binary(img, thresh=127):
    g = ensure_gray_u8(img)
    if g is None:
        return None
    return (g > thresh).astype(np.uint8)


def latlon_to_enu(lat, lon, origin_lat, origin_lon):
    earth_radius = 6378137.0
    lat0 = math.radians(origin_lat)
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    return earth_radius * math.cos(lat0) * dlon, earth_radius * dlat


def semantic_boundary(label):
    known = label > 0
    b = np.zeros(label.shape, dtype=np.uint8)
    b[:, 1:] |= ((label[:, 1:] != label[:, :-1]) & known[:, 1:] & known[:, :-1]).astype(np.uint8)
    b[1:, :] |= ((label[1:, :] != label[:-1, :]) & known[1:, :] & known[:-1, :]).astype(np.uint8)
    for lab in [LOCAL_STRUCTURE, LOCAL_OBSTACLE, LOCAL_OPEN, LOCAL_ROAD_EDGE, LOCAL_LANE_LINE]:
        m = (label == lab).astype(np.uint8) * 255
        if m.sum() > 0:
            b |= (cv2.Canny(m, 50, 150) > 0).astype(np.uint8)
    return b


def split_prompts(text, default_items):
    if not text:
        return default_items
    return [x.strip() for x in text.split("|") if x.strip()] or default_items


def cleanup_binary_mask(mask, min_area=24, close_iter=1):
    mask = (mask > 0).astype(np.uint8)
    if close_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if min_area <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def union_binary(*masks):
    out = None
    for mask in masks:
        if mask is None:
            continue
        m = (mask > 0).astype(np.uint8)
        out = m.copy() if out is None else np.maximum(out, m)
    return out


class PriorMap:
    def __init__(self, args, meta):
        self.args = args
        self.meta = meta
        self.prior_geo = read_simple_yaml(args.prior_yaml)
        self.origin_lat = float(self.prior_geo.get("origin_lat", self.prior_geo.get("center_lat", 0.0)) or 0.0)
        self.origin_lon = float(self.prior_geo.get("origin_lon", self.prior_geo.get("center_lon", 0.0)) or 0.0)
        self.center_lat = float(self.prior_geo.get("center_lat", 0.0) or 0.0)
        self.center_lon = float(self.prior_geo.get("center_lon", 0.0) or 0.0)
        self.mpp = float(self.prior_geo.get("meters_per_pixel", 0.0) or 0.0)
        self.reference_image_path = self.prior_geo.get("image_path", "")

    def _meters_to_px(self, meters, shape):
        if meters <= 0:
            return 0
        mpp_x = (self.meta["abs_max_e"] - self.meta["abs_min_e"]) / float(max(1, shape[1]))
        mpp_y = (self.meta["abs_max_n"] - self.meta["abs_min_n"]) / float(max(1, shape[0]))
        return max(1, int(round(float(meters) / max(1e-3, 0.5 * (mpp_x + mpp_y)))))

    def _dilate_meters(self, mask, meters):
        if mask is None or meters <= 0:
            return mask
        radius = self._meters_to_px(meters, mask.shape)
        if radius <= 0:
            return mask
        k = 2 * radius + 1
        return cv2.dilate((mask > 0).astype(np.uint8), np.ones((k, k), np.uint8), iterations=1)

    def _load_mask_crop(self, path, out_shape):
        if not path:
            return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("[WARN] prior mask is unreadable: %s" % path)
            return None
        mask = normalize_binary(img, self.args.mask_threshold)
        if mask is None:
            print("[WARN] prior mask cannot be normalized: %s" % path)
            return None
        crop = self._crop_georeferenced(mask, out_shape)
        if crop is not None:
            return crop
        return cv2.resize(mask, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)

    def _crop_georeferenced(self, img, out_shape):
        if self.mpp <= 1e-6 or self.center_lat == 0.0 and self.center_lon == 0.0:
            return None
        center_e, center_n = latlon_to_enu(self.center_lat, self.center_lon, self.origin_lat, self.origin_lon)
        abs_min_e = self.meta["abs_min_e"]
        abs_max_e = self.meta["abs_max_e"]
        abs_min_n = self.meta["abs_min_n"]
        abs_max_n = self.meta["abs_max_n"]
        u0 = img.shape[1] / 2.0 + (abs_min_e - center_e) / self.mpp
        u1 = img.shape[1] / 2.0 + (abs_max_e - center_e) / self.mpp
        v0 = img.shape[0] / 2.0 - (abs_max_n - center_n) / self.mpp
        v1 = img.shape[0] / 2.0 - (abs_min_n - center_n) / self.mpp
        x0 = max(0, int(math.floor(min(u0, u1))))
        x1 = min(img.shape[1], int(math.ceil(max(u0, u1))))
        y0 = max(0, int(math.floor(min(v0, v1))))
        y1 = min(img.shape[0], int(math.ceil(max(v0, v1))))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = img[y0:y1, x0:x1]
        return cv2.resize(crop, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)

    def _fallback_from_satellite(self, req, out_shape):
        sat_rgb = cv2.imread(req.get("sat_rgb_path", ""), cv2.IMREAD_COLOR)
        sat_structure = cv2.imread(req.get("sat_path", ""), cv2.IMREAD_GRAYSCALE)
        if sat_rgb is None:
            return np.zeros(out_shape, dtype=np.uint8)
        sat_rgb = cv2.resize(sat_rgb, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sat_rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        b, g, r = cv2.split(sat_rgb)
        green = (g.astype(np.int16) > r.astype(np.int16) + 15) & (g.astype(np.int16) > b.astype(np.int16) + 10)
        veg = (((h >= 25) & (h <= 95) & (s > 35) & (v > 30)) | green)
        low_sat = (s < 45) & (v > 45) & (v < 210)
        bright = (v > 120) & (s < 85) & (~veg)
        road = low_sat & (~bright) & (~veg)
        label = np.zeros(out_shape, dtype=np.uint8)
        label[road] = PRIOR_ROAD
        label[bright] = PRIOR_BUILDING
        label[veg] = PRIOR_VEGETATION
        if sat_structure is not None:
            st = cv2.resize(sat_structure, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_AREA)
            edges = st > 90
            near_building = cv2.dilate((label == PRIOR_BUILDING).astype(np.uint8), np.ones((5, 5), np.uint8), 1) > 0
            label[edges & near_building] = PRIOR_BUILDING
        return label

    def _add_entrances(self, label):
        if not self.args.entrance_csv_path:
            return label
        sat_mpp_x = (self.meta["abs_max_e"] - self.meta["abs_min_e"]) / float(max(1, label.shape[1]))
        sat_mpp_y = (self.meta["abs_max_n"] - self.meta["abs_min_n"]) / float(max(1, label.shape[0]))
        radius_px = max(1, int(round(self.args.entrance_radius_m / max(1e-3, 0.5 * (sat_mpp_x + sat_mpp_y)))))
        try:
            with open(self.args.entrance_csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "east" in row and "north" in row:
                        e = float(row["east"])
                        n = float(row["north"])
                    elif "lat" in row and "lon" in row:
                        e, n = latlon_to_enu(float(row["lat"]), float(row["lon"]), self.origin_lat, self.origin_lon)
                    else:
                        continue
                    u = int(round((e - self.meta["abs_min_e"]) / sat_mpp_x))
                    v = int(round((self.meta["abs_max_n"] - n) / sat_mpp_y))
                    if 0 <= u < label.shape[1] and 0 <= v < label.shape[0]:
                        cv2.circle(label, (u, v), radius_px, PRIOR_ENTRANCE, -1)
        except OSError:
            pass
        return label

    def build_label(self, req, out_shape):
        road = self._load_mask_crop(self.args.road_mask_path, out_shape)
        road_network = self._load_mask_crop(self.args.road_network_mask_path, out_shape)
        road_surface = self._load_mask_crop(self.args.road_surface_mask_path, out_shape)
        building = self._load_mask_crop(self.args.building_mask_path, out_shape)
        traversable = self._load_mask_crop(self.args.traversable_mask_path, out_shape)
        vegetation = self._load_mask_crop(self.args.vegetation_mask_path, out_shape) if self.args.keep_vegetation_prior else None
        lane = self._load_mask_crop(self.args.lane_mask_path, out_shape)

        road_network_surface = self._dilate_meters(road_network, self.args.road_network_buffer_m)
        road = union_binary(road, road_surface, road_network_surface)
        lane_like = union_binary(lane, road_network)

        if all(x is None for x in [road, road_network, road_surface, building, traversable, vegetation, lane_like]) and not self.args.entrance_csv_path:
            label = self._fallback_from_satellite(req, out_shape)
        else:
            label = np.zeros(out_shape, dtype=np.uint8)
            if traversable is not None:
                label[traversable > 0] = PRIOR_TRAVERSABLE
            if road is not None:
                label[road > 0] = PRIOR_ROAD
            if vegetation is not None:
                label[vegetation > 0] = PRIOR_VEGETATION
            if building is not None:
                label[building > 0] = PRIOR_BUILDING
        if lane_like is not None:
            label[lane_like > 0] = PRIOR_LANE_LINE
        label = self._add_entrances(self._add_geometric_boundaries(label, road, building, lane_like))
        if not self.args.keep_vegetation_prior:
            label[label == PRIOR_VEGETATION] = PRIOR_UNKNOWN
        return label

    def _add_geometric_boundaries(self, label, road_mask=None, building_mask=None, lane_mask=None):
        road_like = ((label == PRIOR_ROAD) | (label == PRIOR_TRAVERSABLE)).astype(np.uint8) * 255
        if road_mask is not None:
            road_like |= (road_mask > 0).astype(np.uint8) * 255
        building_like = (label == PRIOR_BUILDING).astype(np.uint8) * 255
        if building_mask is not None:
            building_like |= (building_mask > 0).astype(np.uint8) * 255
        road_edge = np.zeros(label.shape, dtype=bool)
        if np.count_nonzero(road_like) > 0:
            road_edge |= cv2.Canny(road_like, 50, 150) > 0
        if np.count_nonzero(building_like) > 0:
            road_near_building = cv2.dilate(road_like, np.ones((5, 5), np.uint8), 1) > 0
            b_edge = cv2.Canny(building_like, 50, 150) > 0
            road_edge |= b_edge & road_near_building
        if self.args.prior_edge_dilate_iter > 0:
            road_edge = cv2.dilate(road_edge.astype(np.uint8), np.ones((3, 3), np.uint8), self.args.prior_edge_dilate_iter).astype(bool)
        label[road_edge & (label != PRIOR_BUILDING) & (label != PRIOR_LANE_LINE)] = PRIOR_ROAD_EDGE
        if lane_mask is not None:
            label[lane_mask > 0] = PRIOR_LANE_LINE
        return label


class GeoPriorConstraintMatchService:
    def __init__(self, args):
        self.args = args
        self.queue_dir = Path(args.queue_dir)
        self.input_dir = self.queue_dir / "input"
        self.output_dir = self.queue_dir / "output"
        self.done_dir = self.queue_dir / "done"
        for d in [self.input_dir, self.output_dir, self.done_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.sam3 = None
        if args.use_sam3_local or args.use_sam3_sat:
            if Sam3ConceptSegmenter is None:
                if args.require_sam3:
                    raise RuntimeError("sam3_semantic_match_service.py wrapper is not importable")
                print("[WARN] SAM3 wrapper is not importable; SAM3 refinement disabled")
            else:
                self.sam3 = Sam3ConceptSegmenter(args)
                if not self.sam3.enabled and args.require_sam3:
                    raise RuntimeError("SAM3 requested but failed to load")
        print("[INFO] geo-prior constrained matcher ready queue=%s" % self.queue_dir)

    def sam3_prompt_map(self):
        return {
            "road": split_prompts(
                self.args.sam3_road_prompts,
                ["road", "street", "paved road", "drive lane", "parking lot", "sidewalk"],
            ),
            "building": split_prompts(
                self.args.sam3_building_prompts,
                ["building", "roof", "house", "wall", "structure"],
            ),
        }

    def sam3_segment_road_building(self, bgr):
        if self.sam3 is None or not self.sam3.enabled or bgr is None:
            return None, None
        label, _ = self.sam3.segment(
            bgr,
            self.sam3_prompt_map(),
            self.args.sam3_score_thresh,
            self.args.sam3_mask_thresh,
            self.args.sam3_max_instances,
        )
        road = cleanup_binary_mask(label == SAM_LABEL_ROAD, self.args.sam3_min_mask_area, 1)
        building = cleanup_binary_mask(label == SAM_LABEL_BUILDING, self.args.sam3_min_mask_area, 1)
        return road, building

    def apply_sam3_local_geometry(self, req, local_label):
        if not self.args.use_sam3_local:
            return local_label
        local_path = req.get("local_sam3_path") or req.get("bev_sem_path") or req.get("bev_path", "")
        img = cv2.imread(local_path, cv2.IMREAD_COLOR)
        if img is None:
            print("[WARN] SAM3 local requested but local image is unreadable: %s" % local_path)
            return local_label
        if img.shape[:2] != local_label.shape:
            img = cv2.resize(img, (local_label.shape[1], local_label.shape[0]), interpolation=cv2.INTER_AREA)
        road, building = self.sam3_segment_road_building(img)
        if road is None:
            return local_label
        sam_label = np.zeros_like(local_label, dtype=np.uint8)
        sam_label[road > 0] = LOCAL_OPEN
        sam_label[building > 0] = LOCAL_STRUCTURE
        edge = cv2.Canny((road > 0).astype(np.uint8) * 255, 50, 150) > 0
        if self.args.prior_edge_dilate_iter > 0:
            edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8), self.args.prior_edge_dilate_iter).astype(bool)
        sam_label[edge] = LOCAL_ROAD_EDGE
        if self.args.sam3_local_mode == "replace":
            return sam_label
        merged = local_label.copy()
        # Drop vegetation/noisy obstacle evidence where SAM3 sees stable road/building.
        merged[(sam_label > 0)] = sam_label[(sam_label > 0)]
        return merged

    def apply_sam3_sat_prior(self, req, prior_label, prior_builder):
        if not self.args.use_sam3_sat:
            return prior_label
        sat_path = req.get("sat_rgb_path", "")
        img = cv2.imread(sat_path, cv2.IMREAD_COLOR)
        if img is None:
            print("[WARN] SAM3 satellite requested but sat_rgb is unreadable: %s" % sat_path)
            return prior_label
        if img.shape[:2] != prior_label.shape:
            img = cv2.resize(img, (prior_label.shape[1], prior_label.shape[0]), interpolation=cv2.INTER_AREA)
        road, building = self.sam3_segment_road_building(img)
        if road is None:
            return prior_label
        road_support = ((prior_label == PRIOR_ROAD) | (prior_label == PRIOR_TRAVERSABLE) |
                        (prior_label == PRIOR_ROAD_EDGE) | (prior_label == PRIOR_LANE_LINE)).astype(np.uint8)
        if np.count_nonzero(road_support) > 0 and any([
            self.args.road_mask_path,
            self.args.road_network_mask_path,
            self.args.road_surface_mask_path,
            self.args.traversable_mask_path,
            self.args.lane_mask_path,
        ]):
            road_gate = prior_builder._dilate_meters(road_support, self.args.sam3_prior_gate_m)
            road = ((road > 0) & (road_gate > 0)).astype(np.uint8)
        building_support = (prior_label == PRIOR_BUILDING).astype(np.uint8)
        if np.count_nonzero(building_support) > 0 and self.args.building_mask_path:
            building_gate = prior_builder._dilate_meters(building_support, self.args.sam3_prior_gate_m)
            building = ((building > 0) & (building_gate > 0)).astype(np.uint8)
        out = prior_label.copy()
        if self.args.sam3_sat_mode == "replace":
            out[:] = PRIOR_UNKNOWN
            out[prior_label == PRIOR_LANE_LINE] = PRIOR_LANE_LINE
            out[prior_label == PRIOR_ROAD_EDGE] = PRIOR_ROAD_EDGE
        out[road > 0] = PRIOR_ROAD
        out[building > 0] = PRIOR_BUILDING
        if not self.args.keep_vegetation_prior:
            out[out == PRIOR_VEGETATION] = PRIOR_UNKNOWN
        return prior_builder._add_geometric_boundaries(out, road_mask=road, building_mask=building)

    def load_local_label(self, req):
        geom_path = req.get("local_geometric_label_path", "")
        geom_label = cv2.imread(geom_path, cv2.IMREAD_GRAYSCALE) if geom_path else None
        if geom_label is not None:
            return geom_label.astype(np.uint8)
        path = req.get("local_label_path", "")
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
        if label is not None:
            return label.astype(np.uint8)
        # Legacy fallback: infer labels from visual BEV color only for diagnostics.
        src = cv2.imread(req.get("bev_sem_path", req.get("bev_path", "")), cv2.IMREAD_COLOR)
        if src is None:
            raise RuntimeError("no local_label_path or readable BEV semantic source")
        b, g, r = cv2.split(src)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        label = np.zeros(gray.shape, dtype=np.uint8)
        label[(gray > 90) & (gray < 248)] = LOCAL_OPEN
        label[(r.astype(np.int16) > g.astype(np.int16) + 20) & (r.astype(np.int16) > b.astype(np.int16) + 20)] = LOCAL_STRUCTURE
        label[(g.astype(np.int16) > r.astype(np.int16) + 20) & (g.astype(np.int16) > b.astype(np.int16) + 15)] = LOCAL_VEGETATION
        return label

    def build_points(self, local_label, local_conf, meta):
        h, w = local_label.shape
        valid = (local_label > 0) & (local_label != LOCAL_VEGETATION)
        if local_conf is not None:
            local_conf = cv2.resize(local_conf, (w, h), interpolation=cv2.INTER_AREA)
            valid &= local_conf >= self.args.min_local_confidence
        n_valid = int(valid.sum())
        fg_ratio = n_valid / float(max(1, h * w))
        if n_valid < self.args.min_local_valid_pixels or fg_ratio < self.args.min_local_foreground_ratio:
            return None, {"n_valid": n_valid, "fg_ratio": fg_ratio}
        boundary = semantic_boundary(local_label) > 0
        rng = np.random.RandomState(42)
        road_edge = valid & (local_label == LOCAL_ROAD_EDGE)
        lane_line = valid & (local_label == LOCAL_LANE_LINE)
        structure_boundary = valid & (local_label == LOCAL_STRUCTURE) & boundary
        road_boundary = valid & (local_label == LOCAL_OPEN) & boundary
        obstacle_boundary = valid & (local_label == LOCAL_OBSTACLE) & boundary
        important = road_edge | lane_line | structure_boundary | road_boundary | obstacle_boundary
        interior = valid & (~important)
        use = important | (interior & (rng.rand(h, w) < self.args.interior_sample_ratio))
        if int(use.sum()) < min(n_valid, self.args.min_local_valid_pixels):
            use |= valid & boundary
        if int(use.sum()) < max(200, self.args.min_local_valid_pixels // 2):
            use |= valid & (rng.rand(h, w) < min(0.12, self.args.interior_sample_ratio * 3.0 + 0.02))
        ys, xs = np.where(use)
        if len(xs) > self.args.max_points:
            sample_priority = np.ones(len(xs), dtype=np.float64)
            labs_for_sample = local_label[ys, xs]
            sample_priority[labs_for_sample == LOCAL_OPEN] = 0.45
            sample_priority[labs_for_sample == LOCAL_STRUCTURE] = 1.45
            sample_priority[labs_for_sample == LOCAL_ROAD_EDGE] = 2.40
            sample_priority[labs_for_sample == LOCAL_LANE_LINE] = 2.80
            sample_priority[labs_for_sample == LOCAL_OBSTACLE] = 0.80
            sample_priority = sample_priority / np.maximum(np.sum(sample_priority), 1e-9)
            idx = rng.choice(len(xs), size=self.args.max_points, replace=False, p=sample_priority)
            xs, ys = xs[idx], ys[idx]
        local_mpp_x = (meta["bev_max_e"] - meta["bev_min_e"]) / float(max(1, w))
        local_mpp_y = (meta["bev_max_n"] - meta["bev_min_n"]) / float(max(1, h))
        e = meta["bev_min_e"] + (xs.astype(np.float32) + 0.5) * local_mpp_x
        n = meta["bev_max_n"] - (ys.astype(np.float32) + 0.5) * local_mpp_y
        labs = local_label[ys, xs]
        weight = np.ones(len(xs), dtype=np.float32)
        if local_conf is not None:
            weight = np.clip(local_conf[ys, xs].astype(np.float32) / 255.0, 0.05, 1.0)
        class_weight = np.ones(len(xs), dtype=np.float32)
        class_weight[labs == LOCAL_OPEN] = 0.55
        class_weight[labs == LOCAL_STRUCTURE] = 1.35
        class_weight[labs == LOCAL_ROAD_EDGE] = 2.10
        class_weight[labs == LOCAL_LANE_LINE] = 2.40
        class_weight[labs == LOCAL_OBSTACLE] = 0.70
        boundary_weight = np.where(boundary[ys, xs], 1.20, 1.0).astype(np.float32)
        weight *= class_weight * boundary_weight
        return {
            "e": e, "n": n, "lab": labs, "weight": weight,
            "x": xs, "y": ys, "n_valid": n_valid, "fg_ratio": fg_ratio,
        }, {}

    def compatibility(self, local_lab, prior_lab, dt_building, dt_road, dt_trav, dt_entrance, idx):
        l = local_lab
        p = prior_lab
        score = np.zeros(l.shape, dtype=np.float32)
        score[(l == LOCAL_OPEN) & ((p == PRIOR_ROAD) | (p == PRIOR_TRAVERSABLE) | (p == PRIOR_ROAD_EDGE) | (p == PRIOR_LANE_LINE))] = 1.0
        score[(l == LOCAL_OPEN) & (p == PRIOR_BUILDING)] = 0.05
        score[(l == LOCAL_OPEN) & (p == PRIOR_VEGETATION)] = 0.0
        score[(l == LOCAL_STRUCTURE) & (p == PRIOR_BUILDING)] = 1.0
        score[(l == LOCAL_STRUCTURE) & (p == PRIOR_ENTRANCE)] = 0.9
        score[(l == LOCAL_STRUCTURE) & ((p == PRIOR_ROAD) | (p == PRIOR_TRAVERSABLE))] = 0.05
        score[(l == LOCAL_ROAD_EDGE) & ((p == PRIOR_ROAD_EDGE) | (p == PRIOR_LANE_LINE))] = 1.2
        score[(l == LOCAL_ROAD_EDGE) & ((p == PRIOR_ROAD) | (p == PRIOR_TRAVERSABLE))] = 0.45
        score[(l == LOCAL_ROAD_EDGE) & (p == PRIOR_BUILDING)] = 0.25
        score[(l == LOCAL_LANE_LINE) & (p == PRIOR_LANE_LINE)] = 1.3
        score[(l == LOCAL_LANE_LINE) & ((p == PRIOR_ROAD) | (p == PRIOR_ROAD_EDGE))] = 0.55
        score[(l == LOCAL_OBSTACLE) & ((p == PRIOR_BUILDING) | (p == PRIOR_ENTRANCE))] = 0.75
        score[(l == LOCAL_OBSTACLE) & (p == PRIOR_VEGETATION)] = 0.0
        score[(l == LOCAL_OBSTACLE) & ((p == PRIOR_ROAD) | (p == PRIOR_TRAVERSABLE))] = 0.15

        near_building = np.exp(-np.clip(dt_building[idx], 0, self.args.dt_clip_px) / self.args.dt_sigma_px)
        near_road = np.exp(-np.clip(dt_road[idx], 0, self.args.dt_clip_px) / self.args.dt_sigma_px)
        near_trav = np.exp(-np.clip(dt_trav[idx], 0, self.args.dt_clip_px) / self.args.dt_sigma_px)
        near_entrance = np.exp(-np.clip(dt_entrance[idx], 0, self.args.dt_clip_px) / self.args.dt_sigma_px)
        near_road_edge = np.exp(-np.clip(self._dt_lookup("road_edge", idx), 0, self.args.dt_clip_px) / self.args.dt_sigma_px)
        near_lane = np.exp(-np.clip(self._dt_lookup("lane", idx), 0, self.args.dt_clip_px) / self.args.dt_sigma_px)
        score += self.args.structure_distance_weight * near_building * (l == LOCAL_STRUCTURE)
        score += self.args.open_distance_weight * np.maximum(near_road, near_trav) * (l == LOCAL_OPEN)
        score += self.args.entrance_weight * near_entrance * ((l == LOCAL_STRUCTURE) | (l == LOCAL_OBSTACLE))
        score += self.args.road_edge_weight * near_road_edge * (l == LOCAL_ROAD_EDGE)
        score += self.args.lane_line_weight * near_lane * (l == LOCAL_LANE_LINE)
        return np.clip(score, 0.0, 1.5)

    def _dt_lookup(self, name, idx):
        return self._current_dts.get(name, self._current_dts["road"])[idx]

    @staticmethod
    def distance_to(label, target_values):
        mask = np.zeros(label.shape, dtype=np.uint8)
        for v in target_values:
            mask |= (label == v).astype(np.uint8)
        inv = ((mask == 0).astype(np.uint8) * 255)
        return cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    def score_candidate(self, pts, prior_label, dts, meta, dx, dy, yaw_deg):
        se = float(meta.get("state_e", 0.0))
        sn = float(meta.get("state_n", 0.0))
        c = math.cos(math.radians(yaw_deg))
        s = math.sin(math.radians(yaw_deg))
        # pts["e"]/pts["n"] are local north-up offsets from the robot, not
        # global ENU coordinates. Candidate dx/dy are global ENU corrections.
        x = pts["e"]
        y = pts["n"]
        e2 = se + c * x - s * y + dx
        n2 = sn + s * x + c * y + dy
        prior_mpp_x = (meta["abs_max_e"] - meta["abs_min_e"]) / float(max(1, prior_label.shape[1]))
        prior_mpp_y = (meta["abs_max_n"] - meta["abs_min_n"]) / float(max(1, prior_label.shape[0]))
        u = np.round((e2 - meta["abs_min_e"]) / prior_mpp_x).astype(np.int32)
        v = np.round((meta["abs_max_n"] - n2) / prior_mpp_y).astype(np.int32)
        inside = (u >= 0) & (u < prior_label.shape[1]) & (v >= 0) & (v < prior_label.shape[0])
        coverage = float(np.mean(inside)) if len(inside) else 0.0
        if coverage < self.args.min_coverage:
            return -1.0, coverage, 0.0
        uu = u[inside]
        vv = v[inside]
        flat_idx = (vv, uu)
        comp = self.compatibility(
            pts["lab"][inside], prior_label[vv, uu],
            dts["building"], dts["road"], dts["traversable"], dts["entrance"], flat_idx,
        )
        weights = pts["weight"][inside]
        semantic_score = float(np.sum(comp * weights) / max(1e-6, np.sum(weights)))
        score = semantic_score * min(1.0, coverage / max(1e-3, self.args.target_coverage))
        return score, coverage, semantic_score

    def match(self, local_label, local_conf, prior_label, meta):
        pts, info = self.build_points(local_label, local_conf, meta)
        if pts is None:
            fail = {"reason": "local_semantic_too_sparse"}
            fail.update(info)
            return None, fail
        dts = {
            "building": self.distance_to(prior_label, [PRIOR_BUILDING]),
            "road": self.distance_to(prior_label, [PRIOR_ROAD, PRIOR_ROAD_EDGE, PRIOR_LANE_LINE]),
            "traversable": self.distance_to(prior_label, [PRIOR_TRAVERSABLE, PRIOR_ROAD, PRIOR_ROAD_EDGE, PRIOR_LANE_LINE]),
            "entrance": self.distance_to(prior_label, [PRIOR_ENTRANCE]),
            "road_edge": self.distance_to(prior_label, [PRIOR_ROAD_EDGE, PRIOR_LANE_LINE]),
            "lane": self.distance_to(prior_label, [PRIOR_LANE_LINE]),
        }
        self._current_dts = dts
        dxs = np.arange(-self.args.search_radius_m, self.args.search_radius_m + 1e-6, self.args.search_step_m)
        dys = np.arange(-self.args.search_radius_m, self.args.search_radius_m + 1e-6, self.args.search_step_m)
        yaws = [float(x) for x in self.args.yaw_degs.split(",") if x.strip()]
        heat = np.zeros(prior_label.shape, dtype=np.float32)
        best = None
        scores = []
        prior_mpp_x = (meta["abs_max_e"] - meta["abs_min_e"]) / float(max(1, prior_label.shape[1]))
        prior_mpp_y = (meta["abs_max_n"] - meta["abs_min_n"]) / float(max(1, prior_label.shape[0]))
        for yaw in yaws:
            for dx in dxs:
                for dy in dys:
                    sc, cov, sem_sc = self.score_candidate(pts, prior_label, dts, meta, dx, dy, yaw)
                    if sc < 0:
                        continue
                    scores.append(sc)
                    ce = meta.get("state_e", 0.0) + dx
                    cn = meta.get("state_n", 0.0) + dy
                    uu = int(round((ce - meta["abs_min_e"]) / prior_mpp_x))
                    vv = int(round((meta["abs_max_n"] - cn) / prior_mpp_y))
                    if 0 <= uu < heat.shape[1] and 0 <= vv < heat.shape[0]:
                        heat[vv, uu] = max(heat[vv, uu], sc)
                    if best is None or sc > best["score"]:
                        best = {
                            "score": float(sc), "dx": float(dx), "dy": float(dy), "yaw": float(yaw),
                            "coverage": float(cov), "semantic_score": float(sem_sc),
                            "n_points": int(len(pts["e"])), "n_valid": int(pts["n_valid"]),
                            "fg_ratio": float(pts["fg_ratio"]),
                        }
        if best is None:
            fail = {"reason": "no_candidate_valid"}
            fail.update(info)
            return None, fail
        ordered = sorted(scores, reverse=True)
        second = ordered[1] if len(ordered) > 1 else 0.0
        best["peak_delta"] = float(best["score"] - second)
        best["peak_ratio"] = float(best["score"] / max(1e-6, second))
        best["heat"] = heat
        if best["score"] < self.args.min_score:
            return best, {"reason": "score_low"}
        if best["coverage"] < self.args.min_coverage:
            return best, {"reason": "coverage_low"}
        if best["peak_delta"] < self.args.peak_delta_min or best["peak_ratio"] < self.args.peak_ratio_min:
            return best, {"reason": "peak_not_unique"}
        return best, {"reason": "ok"}

    def render_overlay(self, req, local_label, prior_label, meta, best):
        sat = cv2.imread(req.get("sat_rgb_path", ""), cv2.IMREAD_COLOR)
        if sat is None:
            sat = np.full((prior_label.shape[0], prior_label.shape[1], 3), 80, dtype=np.uint8)
        sat = cv2.resize(sat, (prior_label.shape[1], prior_label.shape[0]), interpolation=cv2.INTER_AREA)
        prior_color = colorize(prior_label, PRIOR_COLORS)
        base = cv2.addWeighted(sat, 0.55, prior_color, 0.45, 0)
        heat = best.get("heat", np.zeros(prior_label.shape, np.float32)) if best else np.zeros(prior_label.shape, np.float32)
        if heat.max() > heat.min():
            h8 = ((heat - heat.min()) / (heat.max() - heat.min()) * 255).astype(np.uint8)
            h8 = cv2.GaussianBlur(h8, (0, 0), 5)
            base = cv2.addWeighted(base, 0.75, cv2.applyColorMap(h8, cv2.COLORMAP_JET), 0.25, 0)
        if best is not None:
            pts, _ = self.build_points(local_label, None, meta)
            if pts is not None:
                se = float(meta.get("state_e", 0.0))
                sn = float(meta.get("state_n", 0.0))
                c = math.cos(math.radians(best.get("yaw", 0.0)))
                s = math.sin(math.radians(best.get("yaw", 0.0)))
                x = pts["e"]
                y = pts["n"]
                e2 = se + c * x - s * y + best["dx"]
                n2 = sn + s * x + c * y + best["dy"]
                mpp_x = (meta["abs_max_e"] - meta["abs_min_e"]) / float(max(1, prior_label.shape[1]))
                mpp_y = (meta["abs_max_n"] - meta["abs_min_n"]) / float(max(1, prior_label.shape[0]))
                u = np.round((e2 - meta["abs_min_e"]) / mpp_x).astype(np.int32)
                v = np.round((meta["abs_max_n"] - n2) / mpp_y).astype(np.int32)
                inside = (u >= 0) & (u < base.shape[1]) & (v >= 0) & (v < base.shape[0])
                for uu, vv, lab in zip(u[inside][::2], v[inside][::2], pts["lab"][inside][::2]):
                    cv2.circle(base, (int(uu), int(vv)), 1, LOCAL_COLORS.get(int(lab), (255, 255, 255)), -1)
                cu = int(round((se + best["dx"] - meta["abs_min_e"]) / mpp_x))
                cv = int(round((meta["abs_max_n"] - (sn + best["dy"])) / mpp_y))
                cv2.drawMarker(base, (cu, cv), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
                txt = "score=%.3f dx=%.1f dy=%.1f yaw=%.1f cov=%.2f pk=%.3f/%.2f" % (
                    best.get("score", 0), best.get("dx", 0), best.get("dy", 0), best.get("yaw", 0),
                    best.get("coverage", 0), best.get("peak_delta", 0), best.get("peak_ratio", 0))
                cv2.putText(base, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(base, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        heat_bgr = cv2.applyColorMap((np.clip(heat, 0, 1.5) / 1.5 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return base, heat_bgr

    def write_result(self, req, local_label, prior_label, best, fail, accepted):
        rid = req.get("request_id", str(int(time.time() * 1e9)))
        meta = meta_dict(req)
        overlay, heat_bgr = self.render_overlay(req, local_label, prior_label, meta, best)
        local_color_path = self.output_dir / ("local_semantic_%s.png" % rid)
        prior_color_path = self.output_dir / ("geo_prior_%s.png" % rid)
        local_label_path = self.output_dir / ("local_label_%s.png" % rid)
        prior_label_path = self.output_dir / ("geo_prior_label_%s.png" % rid)
        overlay_path = self.output_dir / ("constraint_overlay_%s.png" % rid)
        heatmap_path = self.output_dir / ("constraint_heatmap_%s.png" % rid)
        cv2.imwrite(str(local_color_path), colorize(local_label, LOCAL_COLORS))
        cv2.imwrite(str(prior_color_path), colorize(prior_label, PRIOR_COLORS))
        cv2.imwrite(str(local_label_path), local_label.astype(np.uint8))
        cv2.imwrite(str(prior_label_path), prior_label.astype(np.uint8))
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(heatmap_path), heat_bgr)
        if best is None:
            best = {"score": 0, "dx": 0, "dy": 0, "yaw": 0, "coverage": 0,
                    "semantic_score": 0, "peak_delta": 0, "peak_ratio": 0, "n_points": 0}
        confidence = float(min(1.0, best.get("score", 0.0)))
        result = {
            "request_id": rid,
            "accepted": bool(accepted),
            "reason": fail.get("reason", "ok") if fail else "ok",
            "meas_e": float(meta.get("state_e", 0.0) + best.get("dx", 0.0)),
            "meas_n": float(meta.get("state_n", 0.0) + best.get("dy", 0.0)),
            "confidence": confidence,
            "semantic_score": float(best.get("semantic_score", 0.0)),
            "coverage": float(best.get("coverage", 0.0)),
            "inliers": int(max(0, round(best.get("coverage", 0.0) * 80))),
            "matches": int(best.get("n_points", 0)),
            "inlier_ratio": float(best.get("coverage", 0.0)),
            "scale": 1.0,
            "rotation_deg": float(best.get("yaw", 0.0)),
            "dx": float(best.get("dx", 0.0)),
            "dy": float(best.get("dy", 0.0)),
            "yaw_deg": float(best.get("yaw", 0.0)),
            "peak_delta": float(best.get("peak_delta", 0.0)),
            "peak_ratio": float(best.get("peak_ratio", 0.0)),
            "variance": float(self.args.output_variance / max(0.25, confidence)),
            "debug_path": str(overlay_path),
            "overlay_path": str(overlay_path),
            "heatmap_path": str(heatmap_path),
            "local_semantic_color_path": str(local_color_path),
            "satellite_semantic_color_path": str(prior_color_path),
            "local_label_path": str(local_label_path),
            "geo_prior_label_path": str(prior_label_path),
        }
        write_json_atomic(self.output_dir / ("result_%s.json" % rid), result)
        print("[INFO] request %s accepted=%s reason=%s score=%.3f dx=%.2f dy=%.2f yaw=%.1f cov=%.2f" % (
            rid, accepted, result["reason"], best.get("score", 0.0), best.get("dx", 0.0),
            best.get("dy", 0.0), best.get("yaw", 0.0), best.get("coverage", 0.0)))

    def process_one(self, ready_path):
        req = read_json(ready_path)
        meta = meta_dict(req)
        local_label = self.load_local_label(req)
        local_label = self.apply_sam3_local_geometry(req, local_label)
        local_conf = cv2.imread(req.get("local_geometric_confidence_path", ""), cv2.IMREAD_GRAYSCALE)
        if local_conf is None:
            local_conf = cv2.imread(req.get("local_confidence_path", ""), cv2.IMREAD_GRAYSCALE)
        if self.args.use_sam3_local and self.args.sam3_local_mode == "replace":
            local_conf = ((local_label > 0).astype(np.uint8) * 255)
        out_shape = (int(meta.get("sat_crop_rows", 0)) or local_label.shape[0],
                     int(meta.get("sat_crop_cols", 0)) or local_label.shape[1])
        prior_builder = PriorMap(self.args, meta)
        prior_label = prior_builder.build_label(req, out_shape)
        prior_label = self.apply_sam3_sat_prior(req, prior_label, prior_builder)
        best, fail = self.match(local_label, local_conf, prior_label, meta)
        accepted = bool(best is not None and fail.get("reason") == "ok")
        self.write_result(req, local_label, prior_label, best, fail, accepted)
        if os.path.exists(str(ready_path)):
            try:
                shutil.move(str(ready_path), str(self.done_dir / os.path.basename(ready_path)))
            except Exception:
                try:
                    os.remove(str(ready_path))
                except OSError:
                    pass

    def spin(self):
        print("[INFO] waiting requests in %s" % self.input_dir)
        while True:
            ready_files = sorted(glob.glob(str(self.input_dir / "ready_*.json")))
            if not ready_files:
                time.sleep(self.args.poll_sec)
                continue
            try:
                self.process_one(ready_files[-1])
            except Exception as exc:
                print("[ERROR] failed processing %s: %s" % (ready_files[-1], str(exc)))
                failed_dir = self.queue_dir / "failed"
                failed_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(ready_files[-1], str(failed_dir / os.path.basename(ready_files[-1])))
                except Exception:
                    try:
                        os.remove(ready_files[-1])
                    except Exception:
                        pass
                time.sleep(1.0)


def build_argparser():
    p = argparse.ArgumentParser(description="Local semantic BEV and geographic-prior constrained matcher.")
    p.add_argument("--queue_dir", default="/tmp/eloftr_queue")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sam3_root", default="", help="Path to facebookresearch/sam3 repo or empty if installed in env")
    p.add_argument("--sam3_checkpoint", default="")
    p.add_argument("--sam3_dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--require_sam3", action="store_true")
    p.add_argument("--use_sam3_local", action="store_true", help="Use SAM3 on local RGB BEV to extract road/building geometry.")
    p.add_argument("--use_sam3_sat", action="store_true", help="Use SAM3 on satellite crop to extract road/building geometry.")
    p.add_argument("--sam3_local_mode", choices=["merge", "replace"], default="replace")
    p.add_argument("--sam3_sat_mode", choices=["merge", "replace"], default="merge")
    p.add_argument("--sam3_road_prompts", default="road|street|paved road|drive lane|parking lot|sidewalk")
    p.add_argument("--sam3_building_prompts", default="building|roof|house|wall|structure")
    p.add_argument("--sam3_score_thresh", type=float, default=0.25)
    p.add_argument("--sam3_mask_thresh", type=float, default=0.0)
    p.add_argument("--sam3_max_instances", type=int, default=20)
    p.add_argument("--sam3_min_mask_area", type=int, default=64)
    p.add_argument("--sam3_prior_gate_m", type=float, default=6.0, help="When explicit priors exist, keep SAM3 satellite road/building only near those priors.")
    p.add_argument("--prior_yaml", default="", help="Georeferenced prior mosaic YAML.")
    p.add_argument("--road_mask_path", default="")
    p.add_argument("--road_network_mask_path", default="", help="Binary road-network/centerline mask aligned with prior_yaml or current crop.")
    p.add_argument("--road_surface_mask_path", default="", help="Binary road-surface mask aligned with prior_yaml or current crop.")
    p.add_argument("--building_mask_path", default="")
    p.add_argument("--traversable_mask_path", default="")
    p.add_argument("--vegetation_mask_path", default="")
    p.add_argument("--lane_mask_path", default="")
    p.add_argument("--road_network_buffer_m", type=float, default=3.5, help="Buffer road-network centerlines into road-surface support when no surface mask is available.")
    p.add_argument("--keep_vegetation_prior", action="store_true", help="Keep vegetation prior labels. Default drops vegetation so matching uses only road/building geometry.")
    p.add_argument("--entrance_csv_path", default="")
    p.add_argument("--mask_threshold", type=int, default=127)
    p.add_argument("--entrance_radius_m", type=float, default=2.0)
    p.add_argument("--search_radius_m", type=float, default=35.0)
    p.add_argument("--search_step_m", type=float, default=2.0)
    p.add_argument("--yaw_degs", default="-10,-5,0,5,10")
    p.add_argument("--min_score", type=float, default=0.46)
    p.add_argument("--min_coverage", type=float, default=0.20)
    p.add_argument("--target_coverage", type=float, default=0.45)
    p.add_argument("--peak_delta_min", type=float, default=0.020)
    p.add_argument("--peak_ratio_min", type=float, default=1.020)
    p.add_argument("--min_local_valid_pixels", type=int, default=800)
    p.add_argument("--min_local_foreground_ratio", type=float, default=0.008)
    p.add_argument("--min_local_confidence", type=int, default=15)
    p.add_argument("--max_points", type=int, default=12000)
    p.add_argument("--interior_sample_ratio", type=float, default=0.02)
    p.add_argument("--dt_sigma_px", type=float, default=6.0)
    p.add_argument("--dt_clip_px", type=float, default=25.0)
    p.add_argument("--structure_distance_weight", type=float, default=0.25)
    p.add_argument("--open_distance_weight", type=float, default=0.08)
    p.add_argument("--entrance_weight", type=float, default=0.15)
    p.add_argument("--road_edge_weight", type=float, default=0.55)
    p.add_argument("--lane_line_weight", type=float, default=0.65)
    p.add_argument("--prior_edge_dilate_iter", type=int, default=1)
    p.add_argument("--output_variance", type=float, default=9.0)
    p.add_argument("--poll_sec", type=float, default=0.5)
    return p


if __name__ == "__main__":
    GeoPriorConstraintMatchService(build_argparser().parse_args()).spin()
