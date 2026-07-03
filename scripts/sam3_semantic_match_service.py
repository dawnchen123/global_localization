#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v16 SAM3 semantic-map matching service.

Run this script OUTSIDE ROS Noetic, in a Python 3.12+/PyTorch/SAM3 conda env.
It reads requests exported by eloftr_queue_exporter.py and writes result_*.json
that eloftr_queue_result_publisher.py can publish as /map_match_pose_external.

Pipeline:
  queue/input/ready_<id>.json
    - bev_sem_<id>.png     local geometry/BEV source image
    - sat_rgb_<id>.png     satellite RGB crop
    - optional bev_<id>.png / sat_<id>.png structure images
    - meta with ENU crop bounds
  -> SAM3 concept segmentation for satellite and optionally local image
  -> local/satellite semantic label maps
  -> dx/dy/yaw semantic consistency search in ENU coordinates
  -> queue/output/result_<id>.json + heatmap/overlay/semantic color images

Notes:
  * SAM3 API is loaded lazily and isolated from ROS Python3.8.
  * For local BEV images, heuristic geometry semantics are usually more stable
    than text-prompt SAM3, unless the local source is a real RGB camera image.
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np

# NumPy compatibility for older dependencies that may still reference aliases.
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

LABEL_UNKNOWN = 0
LABEL_OPEN = 1
LABEL_ROAD = 2
LABEL_BUILDING = 3
LABEL_VEGETATION = 4
LABEL_OBSTACLE = 5
LABEL_WATER = 6

SEM_COLORS = {
    LABEL_UNKNOWN: (0, 0, 0),
    LABEL_OPEN: (210, 210, 210),
    LABEL_ROAD: (80, 80, 80),
    LABEL_BUILDING: (30, 30, 230),
    LABEL_VEGETATION: (40, 180, 40),
    LABEL_OBSTACLE: (0, 160, 255),
    LABEL_WATER: (230, 120, 20),
}

DEFAULT_SAT_PROMPTS = {
    'road': ['road', 'street', 'paved road', 'parking lot'],
    'building': ['building', 'roof', 'house', 'structure'],
    'vegetation': ['tree', 'vegetation', 'grass', 'green area'],
    'open': ['open ground', 'open area', 'square', 'bare ground'],
    'water': ['water', 'river', 'pond'],
}

DEFAULT_LOCAL_PROMPTS = {
    'road': ['road', 'ground', 'drivable area', 'open ground'],
    'building': ['building', 'wall', 'structure'],
    'vegetation': ['tree', 'vegetation', 'bush'],
    'obstacle': ['vehicle', 'pole', 'obstacle', 'barrier'],
}

NAME_TO_LABEL = {
    'open': LABEL_OPEN,
    'road': LABEL_ROAD,
    'building': LABEL_BUILDING,
    'vegetation': LABEL_VEGETATION,
    'obstacle': LABEL_OBSTACLE,
    'water': LABEL_WATER,
}


def parse_prompt_map(text, default_map):
    """Parse class=prompt1|prompt2,class2=... format."""
    if not text:
        return default_map
    out = {}
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            continue
        k, v = chunk.split('=', 1)
        prompts = [p.strip() for p in v.split('|') if p.strip()]
        if prompts:
            out[k.strip()] = prompts
    return out or default_map


def ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def colorize_labels(label):
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k, c in SEM_COLORS.items():
        out[label == k] = c
    return out


def largest_dim_resize(img, max_dim, interpolation=cv2.INTER_AREA):
    if max_dim <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img, 1.0
    scale = float(max_dim) / float(m)
    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=interpolation), scale


class Sam3ConceptSegmenter:
    def __init__(self, args):
        self.args = args
        self.enabled = False
        self.model = None
        self.processor = None
        self.torch = None
        self.dtype = None
        self.use_autocast = False
        if args.sam3_root:
            sys.path.insert(0, args.sam3_root)
        self._load_sam3()

    def _resolve_dtype(self, torch):
        name = str(getattr(self.args, 'sam3_dtype', 'bf16')).lower()
        if self.args.device.startswith('cpu'):
            return torch.float32, False
        if name in ['fp32', 'float32', '32']:
            return torch.float32, False
        if name in ['fp16', 'float16', 'half', '16']:
            return torch.float16, True
        if name in ['bf16', 'bfloat16']:
            return torch.bfloat16, True
        # auto: prefer bf16 on CUDA because SAM3/Blackwell pipelines often use bf16.
        if name == 'auto':
            return torch.bfloat16, True
        return torch.bfloat16, True

    def _autocast_context(self):
        if self.torch is None or not self.use_autocast or not self.args.device.startswith('cuda'):
            return nullcontext()
        return self.torch.autocast(device_type='cuda', dtype=self.dtype)

    def _cast_nested(self, obj, dtype):
        """Recursively cast floating tensors in SAM3 state to the selected dtype.
        SAM3 processor caches image features in `state`; on some torch/SAM3 builds these
        tensors keep a different dtype from the prompt/text branch, causing:
          mat1 and mat2 must have the same dtype, but got Float and BFloat16.
        """
        torch = self.torch
        if torch is None:
            return obj
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point():
                return obj.to(device=self.args.device, dtype=dtype)
            return obj.to(device=self.args.device)
        if isinstance(obj, dict):
            return {k: self._cast_nested(v, dtype) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._cast_nested(v, dtype) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._cast_nested(v, dtype) for v in obj)
        return obj

    def _force_module_dtype(self, dtype):
        """Force model/processor owned modules to one dtype/device."""
        torch = self.torch
        if torch is None:
            return
        for obj in [self.model, self.processor]:
            if obj is None:
                continue
            # direct module
            if hasattr(obj, 'to'):
                try:
                    obj.to(device=self.args.device, dtype=dtype)
                except Exception:
                    pass
            # common nested module holders
            for name in ['model', 'predictor', 'image_predictor', 'text_encoder', 'image_encoder', 'sam_model']:
                try:
                    sub = getattr(obj, name, None)
                    if sub is not None and hasattr(sub, 'to'):
                        sub.to(device=self.args.device, dtype=dtype)
                except Exception:
                    pass

    def _load_sam3(self):
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self.torch = torch
            # Official examples use build_sam3_image_model() with HF/cache handling.
            # Some forks expose checkpoint_path; use it when supported.
            try:
                if self.args.sam3_checkpoint:
                    model = build_sam3_image_model(checkpoint_path=self.args.sam3_checkpoint)
                else:
                    model = build_sam3_image_model()
            except TypeError:
                model = build_sam3_image_model()
            self.dtype, self.use_autocast = self._resolve_dtype(torch)
            # Keep every parameter/buffer in one dtype.  This avoids errors like:
            #   mat1 and mat2 must have the same dtype, but got BFloat16 and Float
            # The official SAM3 stack may create bf16 activations on recent CUDA GPUs;
            # therefore we explicitly cast the whole model to the selected dtype.
            if self.args.device.startswith('cuda'):
                model = model.to(device=self.args.device, dtype=self.dtype)
            else:
                model = model.to(device=self.args.device, dtype=torch.float32)
            model.eval()
            self.model = model
            self.processor = Sam3Processor(model)
            # Some SAM3 builds keep submodules in mixed dtypes after processor wrapping.
            # Force again after processor construction.
            self._force_module_dtype(self.dtype if self.args.device.startswith('cuda') else torch.float32)
            self.enabled = True
            print('[INFO] SAM3 loaded. python=%s torch=%s cuda=%s device=%s dtype=%s autocast=%s' % (
                sys.executable, torch.__version__, torch.cuda.is_available(), self.args.device,
                str(self.dtype), str(self.use_autocast)))
        except Exception as exc:
            if self.args.require_sam3:
                raise
            print('[WARN] SAM3 not loaded, fallback modes may still work: %s' % str(exc))
            self.enabled = False

    @staticmethod
    def _to_numpy_mask(m):
        try:
            import torch
            if isinstance(m, torch.Tensor):
                m = m.detach().float().cpu().numpy()
        except Exception:
            pass
        m = np.asarray(m)
        # Common shapes: [N,H,W], [N,1,H,W], [H,W]
        if m.ndim == 4:
            m = m[:, 0]
        return m

    @staticmethod
    def _to_numpy_scores(scores, n):
        try:
            import torch
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().float().cpu().numpy()
        except Exception:
            pass
        if scores is None:
            return np.ones((n,), dtype=np.float32)
        scores = np.asarray(scores).reshape(-1)
        if len(scores) < n:
            scores = np.pad(scores, (0, n-len(scores)), constant_values=1.0)
        return scores[:n].astype(np.float32)

    def segment(self, bgr_img, prompt_map, score_thresh=0.25, mask_thresh=0.0, max_instances_per_prompt=20):
        if not self.enabled:
            raise RuntimeError('SAM3 is not loaded')
        from PIL import Image
        img_rgb = cv2.cvtColor(ensure_bgr(bgr_img), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        h, w = img_rgb.shape[:2]
        label = np.zeros((h, w), dtype=np.uint8)
        conf = np.zeros((h, w), dtype=np.float32)
        # Reset image state for each input image.
        # Run SAM3 in a consistent dtype context.
        with self.torch.inference_mode():
            with self._autocast_context():
                state = self.processor.set_image(pil)
            # Keep cached image state in same dtype as prompt branch.
            target_dtype = self.dtype if self.args.device.startswith('cuda') else self.torch.float32
            state = self._cast_nested(state, target_dtype)
        for cls_name, prompts in prompt_map.items():
            lab = NAME_TO_LABEL.get(cls_name, LABEL_UNKNOWN)
            if lab == LABEL_UNKNOWN:
                continue
            for prompt in prompts:
                try:
                    with self.torch.inference_mode():
                        # Re-cast state right before each prompt because SAM3 may mutate it.
                        target_dtype = self.dtype if self.args.device.startswith('cuda') else self.torch.float32
                        state_for_prompt = self._cast_nested(state, target_dtype)
                        with self._autocast_context():
                            out = self.processor.set_text_prompt(state=state_for_prompt, prompt=prompt)
                    masks = self._to_numpy_mask(out.get('masks', []))
                    if masks.size == 0:
                        continue
                    if masks.ndim == 2:
                        masks = masks[None]
                    scores = self._to_numpy_scores(out.get('scores', None), masks.shape[0])
                    order = np.argsort(-scores)[:max_instances_per_prompt]
                    for idx in order:
                        sc = float(scores[idx])
                        if sc < score_thresh:
                            continue
                        m = masks[idx]
                        if m.shape != (h, w):
                            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                        # Some SAM3 outputs may be bool, logits or probabilities.
                        if m.dtype == np.bool_:
                            mm = m
                        else:
                            # If values span 0..1 use 0.5 threshold; if logits use mask_thresh.
                            thr = 0.5 if m.max() <= 1.0 and m.min() >= 0.0 else mask_thresh
                            mm = m > thr
                        update = mm & (sc >= conf)
                        label[update] = lab
                        conf[update] = sc
                except Exception as exc:
                    print('[WARN] SAM3 prompt failed prompt=%s err=%s' % (prompt, str(exc)))
        # Simple cleanup per label.
        kernel = np.ones((3, 3), np.uint8)
        for lab in [LABEL_OPEN, LABEL_ROAD, LABEL_BUILDING, LABEL_VEGETATION, LABEL_OBSTACLE, LABEL_WATER]:
            m = (label == lab).astype(np.uint8) * 255
            if m.sum() == 0:
                continue
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
            label[(m > 0) & (label == lab)] = lab
        return label, conf


def heuristic_local_semantic(bgr_or_gray, structure_gray=None):
    img = ensure_bgr(bgr_or_gray)
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = np.zeros(gray.shape, dtype=np.uint8)
    open_mask = (gray > 90) & (gray < 248) & (s < 80)
    red_like = (r.astype(np.int16) > g.astype(np.int16) + 20) & (r.astype(np.int16) > b.astype(np.int16) + 20)
    dark_structure = (gray < 100) & (s > 30)
    vegetation = (g.astype(np.int16) > r.astype(np.int16) + 20) & (g.astype(np.int16) > b.astype(np.int16) + 15) & (s > 40)
    if structure_gray is not None:
        sg = structure_gray if len(structure_gray.shape) == 2 else cv2.cvtColor(structure_gray, cv2.COLOR_BGR2GRAY)
        sg = cv2.resize(sg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        dark_structure |= sg > 90
        open_mask |= ((sg > 25) & ~dark_structure)
    label[open_mask] = LABEL_OPEN
    label[dark_structure | red_like] = LABEL_OBSTACLE
    label[vegetation] = LABEL_VEGETATION
    return label


def heuristic_satellite_semantic(bgr, structure_gray=None):
    img = ensure_bgr(bgr)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = np.zeros(gray.shape, dtype=np.uint8)
    green_dom = (g.astype(np.int16) > r.astype(np.int16) + 15) & (g.astype(np.int16) > b.astype(np.int16) + 10)
    veg = (((h >= 25) & (h <= 95) & (s > 35) & (v > 30)) | green_dom)
    low_sat = (s < 45) & (v > 45) & (v < 210)
    bright_roof = (v > 120) & (s < 85) & (~veg)
    road = low_sat & (~bright_roof) & (~veg)
    label[road] = LABEL_ROAD
    label[bright_roof] = LABEL_BUILDING
    label[veg] = LABEL_VEGETATION
    if structure_gray is not None:
        sg = structure_gray if len(structure_gray.shape) == 2 else cv2.cvtColor(structure_gray, cv2.COLOR_BGR2GRAY)
        sg = cv2.resize(sg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        edges = sg > 90
        dil_b = cv2.dilate((label == LABEL_BUILDING).astype(np.uint8), np.ones((5, 5), np.uint8), 1) > 0
        label[edges & dil_b] = LABEL_BUILDING
    return label


def semantic_boundary(label):
    known = label > 0
    b = np.zeros(label.shape, dtype=np.uint8)
    b[:, 1:] |= ((label[:, 1:] != label[:, :-1]) & known[:, 1:] & known[:, :-1]).astype(np.uint8)
    b[1:, :] |= ((label[1:, :] != label[:-1, :]) & known[1:, :] & known[:-1, :]).astype(np.uint8)
    for lab in [LABEL_BUILDING, LABEL_VEGETATION, LABEL_OBSTACLE, LABEL_ROAD]:
        m = (label == lab).astype(np.uint8) * 255
        if m.sum() > 0:
            b |= (cv2.Canny(m, 50, 150) > 0).astype(np.uint8)
    return b


def compatibility(local, sat):
    score = np.zeros(local.shape, dtype=np.float32)
    l, s = local, sat
    score[(l == LABEL_OPEN) & ((s == LABEL_OPEN) | (s == LABEL_ROAD))] = 1.0
    score[(l == LABEL_OPEN) & (s == LABEL_BUILDING)] = 0.10
    score[(l == LABEL_OPEN) & (s == LABEL_VEGETATION)] = 0.20
    score[(l == LABEL_ROAD) & (s == LABEL_ROAD)] = 1.0
    score[(l == LABEL_ROAD) & (s == LABEL_OPEN)] = 0.75
    score[(l == LABEL_ROAD) & ((s == LABEL_BUILDING) | (s == LABEL_VEGETATION))] = 0.05
    score[(l == LABEL_BUILDING) & (s == LABEL_BUILDING)] = 1.0
    score[(l == LABEL_BUILDING) & (s == LABEL_VEGETATION)] = 0.20
    score[(l == LABEL_BUILDING) & (s == LABEL_ROAD)] = 0.05
    score[(l == LABEL_OBSTACLE) & (s == LABEL_BUILDING)] = 0.85
    score[(l == LABEL_OBSTACLE) & (s == LABEL_VEGETATION)] = 0.65
    score[(l == LABEL_OBSTACLE) & (s == LABEL_ROAD)] = 0.10
    score[(l == LABEL_VEGETATION) & (s == LABEL_VEGETATION)] = 1.0
    score[(l == LABEL_VEGETATION) & (s == LABEL_BUILDING)] = 0.25
    score[(l == LABEL_VEGETATION) & (s == LABEL_ROAD)] = 0.10
    return score


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json_atomic(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


def meta_dict(req):
    meta = req.get('meta', [])
    names = req.get('meta_layout', [])
    return {names[i]: float(meta[i]) for i in range(min(len(meta), len(names)))}


class Sam3SemanticMatchService:
    def __init__(self, args):
        self.args = args
        self.queue_dir = Path(args.queue_dir)
        self.input_dir = self.queue_dir / 'input'
        self.output_dir = self.queue_dir / 'output'
        self.done_dir = self.queue_dir / 'done'
        for d in [self.input_dir, self.output_dir, self.done_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.sat_prompt_map = parse_prompt_map(args.sat_prompts, DEFAULT_SAT_PROMPTS)
        self.local_prompt_map = parse_prompt_map(args.local_prompts, DEFAULT_LOCAL_PROMPTS)
        self.sam3 = None
        if args.sat_backend == 'sam3' or args.local_backend == 'sam3':
            self.sam3 = Sam3ConceptSegmenter(args)
        print('[INFO] v18 SAM3 semantic match service ready. queue=%s local=%s sat=%s' % (
            self.queue_dir, args.local_backend, args.sat_backend))

    def segment_sat(self, req, sat_rgb, sat_structure):
        if self.args.sat_backend == 'sam3':
            return self.sam3.segment(sat_rgb, self.sat_prompt_map, self.args.sam3_score_thresh,
                                     self.args.sam3_mask_thresh, self.args.sam3_max_instances)[0]
        return heuristic_satellite_semantic(sat_rgb, sat_structure)

    def segment_local(self, req, bev_sem, bev_structure):
        if self.args.local_backend == 'sam3':
            return self.sam3.segment(bev_sem, self.local_prompt_map, self.args.sam3_score_thresh,
                                     self.args.sam3_mask_thresh, self.args.sam3_max_instances)[0]
        return heuristic_local_semantic(bev_sem, bev_structure)

    def build_points(self, local_label, meta):
        h, w = local_label.shape
        valid = local_label > 0
        n_valid = int(valid.sum())
        fg_ratio = n_valid / float(max(1, h*w))
        if n_valid < self.args.min_local_valid_pixels or fg_ratio < self.args.min_local_foreground_ratio:
            return None, {'n_valid': n_valid, 'fg_ratio': fg_ratio}
        boundary = semantic_boundary(local_label) > 0
        use = valid & (boundary | (np.random.RandomState(0).rand(h, w) < self.args.interior_sample_ratio))
        ys, xs = np.where(use)
        if len(xs) > self.args.max_points:
            rng = np.random.RandomState(123)
            idx = rng.choice(len(xs), size=self.args.max_points, replace=False)
            xs, ys = xs[idx], ys[idx]
        local_mpp_x = (meta['bev_max_e'] - meta['bev_min_e']) / float(max(1, w))
        local_mpp_y = (meta['bev_max_n'] - meta['bev_min_n']) / float(max(1, h))
        e = meta['bev_min_e'] + (xs.astype(np.float32) + 0.5) * local_mpp_x
        n = meta['bev_max_n'] - (ys.astype(np.float32) + 0.5) * local_mpp_y
        lab = local_label[ys, xs]
        return {'e': e, 'n': n, 'lab': lab, 'x': xs, 'y': ys, 'n_valid': n_valid, 'fg_ratio': fg_ratio}, {}

    def score_candidate(self, pts, sat_label, sat_dt, meta, dx, dy, yaw_deg):
        se = float(meta.get('state_e', 0.0))
        sn = float(meta.get('state_n', 0.0))
        c = math.cos(math.radians(yaw_deg))
        s = math.sin(math.radians(yaw_deg))
        x = pts['e'] - se
        y = pts['n'] - sn
        e2 = se + c*x - s*y + dx
        n2 = sn + s*x + c*y + dy
        sat_mpp_x = (meta['abs_max_e'] - meta['abs_min_e']) / float(max(1, sat_label.shape[1]))
        sat_mpp_y = (meta['abs_max_n'] - meta['abs_min_n']) / float(max(1, sat_label.shape[0]))
        u = np.round((e2 - meta['abs_min_e']) / sat_mpp_x).astype(np.int32)
        v = np.round((meta['abs_max_n'] - n2) / sat_mpp_y).astype(np.int32)
        inside = (u >= 0) & (u < sat_label.shape[1]) & (v >= 0) & (v < sat_label.shape[0])
        coverage = float(np.mean(inside)) if len(inside) else 0.0
        if coverage < self.args.min_coverage:
            return -1.0, coverage, 999.0, 0.0
        uu = u[inside]
        vv = v[inside]
        ll = pts['lab'][inside]
        ss = sat_label[vv, uu]
        comp = compatibility(ll, ss)
        semantic_score = float(np.mean(comp)) if comp.size else 0.0
        # Boundary distance term. Use only local boundary-ish points implicitly sampled.
        dt = sat_dt[vv, uu].astype(np.float32)
        mean_dt = float(np.mean(np.clip(dt, 0, self.args.dt_clip_px))) if dt.size else self.args.dt_clip_px
        boundary_score = math.exp(-mean_dt / max(1e-3, self.args.dt_sigma_px))
        score = self.args.semantic_weight * semantic_score + self.args.boundary_weight * boundary_score
        score *= min(1.0, coverage / max(1e-3, self.args.target_coverage))
        return score, coverage, mean_dt, semantic_score

    def match(self, local_label, sat_label, meta):
        pts, info = self.build_points(local_label, meta)
        if pts is None:
            return None, {'reason': 'local_semantic_too_sparse', **info}
        sat_boundary = semantic_boundary(sat_label)
        sat_edge_inv = ((sat_boundary == 0).astype(np.uint8) * 255)
        sat_dt = cv2.distanceTransform(sat_edge_inv, cv2.DIST_L2, 3)
        dxs = np.arange(-self.args.search_radius_m, self.args.search_radius_m + 1e-6, self.args.search_step_m)
        dys = np.arange(-self.args.search_radius_m, self.args.search_radius_m + 1e-6, self.args.search_step_m)
        yaws = [float(x) for x in self.args.yaw_degs.split(',') if x.strip()]
        heat = np.zeros(sat_label.shape, dtype=np.float32)
        best = None
        all_scores = []
        for yaw in yaws:
            for dx in dxs:
                for dy in dys:
                    sc, cov, mean_dt, sem_sc = self.score_candidate(pts, sat_label, sat_dt, meta, dx, dy, yaw)
                    if sc < 0:
                        continue
                    all_scores.append(sc)
                    se = meta.get('state_e', 0.0) + dx
                    sn = meta.get('state_n', 0.0) + dy
                    sat_mpp_x = (meta['abs_max_e'] - meta['abs_min_e']) / float(max(1, sat_label.shape[1]))
                    sat_mpp_y = (meta['abs_max_n'] - meta['abs_min_n']) / float(max(1, sat_label.shape[0]))
                    uu = int(round((se - meta['abs_min_e']) / sat_mpp_x))
                    vv = int(round((meta['abs_max_n'] - sn) / sat_mpp_y))
                    if 0 <= uu < heat.shape[1] and 0 <= vv < heat.shape[0]:
                        heat[vv, uu] = max(heat[vv, uu], sc)
                    if best is None or sc > best['score']:
                        best = {'score': sc, 'dx': float(dx), 'dy': float(dy), 'yaw': float(yaw),
                                'coverage': cov, 'mean_dt': mean_dt, 'semantic_score': sem_sc,
                                'n_points': int(len(pts['e'])), 'n_valid': int(pts['n_valid']),
                                'fg_ratio': float(pts['fg_ratio'])}
        if best is None:
            return None, {'reason': 'no_candidate_valid', **info}
        scores_sorted = sorted(all_scores, reverse=True)
        second = scores_sorted[1] if len(scores_sorted) > 1 else 0.0
        peak_delta = best['score'] - second
        peak_ratio = best['score'] / max(1e-6, second)
        best['peak_delta'] = float(peak_delta)
        best['peak_ratio'] = float(peak_ratio)
        best['heat'] = heat
        if best['score'] < self.args.min_score:
            return best, {'reason': 'score_low'}
        if best['coverage'] < self.args.min_coverage:
            return best, {'reason': 'coverage_low'}
        if peak_delta < self.args.peak_delta_min or peak_ratio < self.args.peak_ratio_min:
            return best, {'reason': 'peak_not_unique'}
        return best, {'reason': 'ok'}

    def render_overlay(self, sat_rgb, sat_label, local_label, meta, best, heat):
        sat = ensure_bgr(sat_rgb)
        sat = cv2.resize(sat, (sat_label.shape[1], sat_label.shape[0]), interpolation=cv2.INTER_AREA)
        sem_color = colorize_labels(sat_label)
        base = cv2.addWeighted(sat, 0.65, sem_color, 0.35, 0)
        if heat.max() > heat.min():
            h8 = ((heat - heat.min()) / (heat.max() - heat.min()) * 255).astype(np.uint8)
            h8 = cv2.GaussianBlur(h8, (0, 0), 5)
            hm = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
            base = cv2.addWeighted(base, 0.72, hm, 0.28, 0)
        # draw local semantic points transformed by best.
        if best is not None:
            pts, _ = self.build_points(local_label, meta)
            if pts is not None:
                se = float(meta.get('state_e', 0.0))
                sn = float(meta.get('state_n', 0.0))
                c = math.cos(math.radians(best.get('yaw', 0.0)))
                s = math.sin(math.radians(best.get('yaw', 0.0)))
                x = pts['e'] - se
                y = pts['n'] - sn
                e2 = se + c*x - s*y + best['dx']
                n2 = sn + s*x + c*y + best['dy']
                sat_mpp_x = (meta['abs_max_e'] - meta['abs_min_e']) / float(max(1, sat_label.shape[1]))
                sat_mpp_y = (meta['abs_max_n'] - meta['abs_min_n']) / float(max(1, sat_label.shape[0]))
                u = np.round((e2 - meta['abs_min_e']) / sat_mpp_x).astype(np.int32)
                v = np.round((meta['abs_max_n'] - n2) / sat_mpp_y).astype(np.int32)
                inside = (u >= 0) & (u < base.shape[1]) & (v >= 0) & (v < base.shape[0])
                for uu, vv, lab in zip(u[inside][::2], v[inside][::2], pts['lab'][inside][::2]):
                    cv2.circle(base, (int(uu), int(vv)), 1, SEM_COLORS.get(int(lab), (255,255,255)), -1)
                ce = se + best['dx']; cn = sn + best['dy']
                cu = int(round((ce - meta['abs_min_e']) / sat_mpp_x))
                cv = int(round((meta['abs_max_n'] - cn) / sat_mpp_y))
                cv2.drawMarker(base, (cu, cv), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
                txt = 'score=%.3f dx=%.1f dy=%.1f yaw=%.1f cov=%.2f pk=%.3f/%.2f' % (
                    best.get('score',0), best.get('dx',0), best.get('dy',0), best.get('yaw',0),
                    best.get('coverage',0), best.get('peak_delta',0), best.get('peak_ratio',0))
                cv2.putText(base, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(base, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        heat_bgr = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return base, heat_bgr

    def write_result(self, req, local_label, sat_label, best, fail_info, accepted):
        rid = req.get('request_id', str(int(time.time()*1e9)))
        meta = meta_dict(req)
        heat = best.get('heat', np.zeros(sat_label.shape, np.float32)) if best else np.zeros(sat_label.shape, np.float32)
        sat_rgb = cv2.imread(req.get('sat_rgb_path',''), cv2.IMREAD_COLOR)
        if sat_rgb is None:
            sat_rgb = colorize_labels(sat_label)
        overlay, heat_bgr = self.render_overlay(sat_rgb, sat_label, local_label, meta, best, heat)
        local_color_path = self.output_dir / ('local_semantic_%s.png' % rid)
        sat_color_path = self.output_dir / ('sat_semantic_%s.png' % rid)
        local_label_path = self.output_dir / ('local_label_%s.png' % rid)
        sat_label_path = self.output_dir / ('sat_label_%s.png' % rid)
        overlay_path = self.output_dir / ('semantic_overlay_%s.png' % rid)
        heatmap_path = self.output_dir / ('semantic_heatmap_%s.png' % rid)
        cv2.imwrite(str(local_color_path), colorize_labels(local_label))
        cv2.imwrite(str(sat_color_path), colorize_labels(sat_label))
        cv2.imwrite(str(local_label_path), local_label.astype(np.uint8))
        cv2.imwrite(str(sat_label_path), sat_label.astype(np.uint8))
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(heatmap_path), heat_bgr)
        reason = fail_info.get('reason', 'ok') if fail_info else 'ok'
        if best is None:
            best = {'score': 0, 'dx': 0, 'dy': 0, 'yaw': 0, 'coverage': 0, 'mean_dt': 999,
                    'semantic_score': 0, 'peak_delta': 0, 'peak_ratio': 0}
        result = {
            'request_id': rid,
            'accepted': bool(accepted),
            'reason': reason,
            'meas_e': float(meta.get('state_e', 0.0) + best.get('dx', 0.0)),
            'meas_n': float(meta.get('state_n', 0.0) + best.get('dy', 0.0)),
            'confidence': float(best.get('score', 0.0)),
            'semantic_score': float(best.get('semantic_score', 0.0)),
            'coverage': float(best.get('coverage', 0.0)),
            'mean_dt': float(best.get('mean_dt', 999.0)),
            'inliers': int(max(0, round(best.get('coverage', 0.0) * 50))),
            'matches': int(best.get('n_points', 0)),
            'inlier_ratio': float(best.get('coverage', 0.0)),
            'scale': 1.0,
            'rotation_deg': float(best.get('yaw', 0.0)),
            'dx': float(best.get('dx', 0.0)),
            'dy': float(best.get('dy', 0.0)),
            'yaw_deg': float(best.get('yaw', 0.0)),
            'peak_delta': float(best.get('peak_delta', 0.0)),
            'peak_ratio': float(best.get('peak_ratio', 0.0)),
            'variance': float(self.args.output_variance),
            'debug_path': str(overlay_path),
            'overlay_path': str(overlay_path),
            'heatmap_path': str(heatmap_path),
            'local_semantic_color_path': str(local_color_path),
            'satellite_semantic_color_path': str(sat_color_path),
            'local_label_path': str(local_label_path),
            'satellite_label_path': str(sat_label_path),
        }
        write_json_atomic(self.output_dir / ('result_%s.json' % rid), result)
        print('[INFO] request %s accepted=%s reason=%s score=%.3f dx=%.2f dy=%.2f yaw=%.1f cov=%.2f' % (
            rid, accepted, reason, best.get('score', 0.0), best.get('dx', 0.0), best.get('dy', 0.0),
            best.get('yaw', 0.0), best.get('coverage', 0.0)))

    def process_one(self, ready_path):
        req = read_json(ready_path)
        rid = req.get('request_id', os.path.basename(ready_path))
        bev_sem_path = req.get('bev_sem_path') or req.get('bev_path')
        sat_rgb_path = req.get('sat_rgb_path') or req.get('sat_path')
        bev_structure_path = req.get('bev_path', '')
        sat_structure_path = req.get('sat_path', '')
        local_src = cv2.imread(bev_sem_path, cv2.IMREAD_COLOR)
        sat_rgb = cv2.imread(sat_rgb_path, cv2.IMREAD_COLOR)
        bev_structure = cv2.imread(bev_structure_path, cv2.IMREAD_GRAYSCALE) if bev_structure_path else None
        sat_structure = cv2.imread(sat_structure_path, cv2.IMREAD_GRAYSCALE) if sat_structure_path else None
        if local_src is None or sat_rgb is None:
            raise RuntimeError('failed to read local or satellite source image')
        local_src, _ = largest_dim_resize(local_src, self.args.max_local_dim)
        sat_rgb, _ = largest_dim_resize(sat_rgb, self.args.max_sat_dim)
        if bev_structure is not None:
            bev_structure = cv2.resize(bev_structure, (local_src.shape[1], local_src.shape[0]), interpolation=cv2.INTER_AREA)
        if sat_structure is not None:
            sat_structure = cv2.resize(sat_structure, (sat_rgb.shape[1], sat_rgb.shape[0]), interpolation=cv2.INTER_AREA)
        local_label = self.segment_local(req, local_src, bev_structure)
        sat_label = self.segment_sat(req, sat_rgb, sat_structure)
        best, fail = self.match(local_label, sat_label, meta_dict(req))
        accepted = bool(best is not None and fail.get('reason') == 'ok')
        self.write_result(req, local_label, sat_label, best, fail, accepted)
        # Move request and its images to done directory to prevent repeated processing.
        try:
            shutil.move(str(ready_path), str(self.done_dir / os.path.basename(ready_path)))
        except Exception:
            os.remove(ready_path)

    def spin(self):
        print('[INFO] waiting requests in %s' % self.input_dir)
        while True:
            ready_files = sorted(glob.glob(str(self.input_dir / 'ready_*.json')))
            if not ready_files:
                time.sleep(self.args.poll_sec)
                continue
            # Process the newest one first; stale older files are less useful for online matching.
            p = ready_files[-1]
            try:
                self.process_one(p)
            except Exception as exc:
                print('[ERROR] failed processing %s: %s' % (p, str(exc)))
                # Prevent infinite retry of the same bad request.  Move it to failed/.
                failed_dir = self.queue_dir / 'failed'
                failed_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(p), str(failed_dir / os.path.basename(p)))
                except Exception:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                time.sleep(1.0)


def build_argparser():
    p = argparse.ArgumentParser(description='SAM3 semantic segmentation + semantic map matching file service.')
    p.add_argument('--queue_dir', default='/tmp/eloftr_queue')
    p.add_argument('--device', default='cuda')
    p.add_argument('--sam3_root', default='', help='Path to facebookresearch/sam3 repo or empty if installed in env')
    p.add_argument('--sam3_checkpoint', default='', help='Optional SAM3 checkpoint path; official API may use HF cache instead')
    p.add_argument('--sam3_dtype', default='bf16', choices=['auto','bf16','bfloat16','fp16','float16','fp32','float32'],
                   help='SAM3 model/inference dtype. Use bf16 on RTX 5090; use fp32 if bf16 has compatibility issues.')
    p.add_argument('--require_sam3', action='store_true')
    p.add_argument('--sat_backend', choices=['sam3','heuristic'], default='sam3')
    p.add_argument('--local_backend', choices=['sam3','heuristic'], default='heuristic')
    p.add_argument('--sat_prompts', default='', help='class=prompt1|prompt2,... classes: road,building,vegetation,open,water')
    p.add_argument('--local_prompts', default='', help='class=prompt1|prompt2,... classes: road,building,vegetation,open,obstacle')
    p.add_argument('--sam3_score_thresh', type=float, default=0.25)
    p.add_argument('--sam3_mask_thresh', type=float, default=0.0)
    p.add_argument('--sam3_max_instances', type=int, default=20)
    p.add_argument('--search_radius_m', type=float, default=35.0)
    p.add_argument('--search_step_m', type=float, default=2.0)
    p.add_argument('--yaw_degs', default='-10,-5,0,5,10')
    p.add_argument('--min_score', type=float, default=0.42)
    p.add_argument('--min_coverage', type=float, default=0.20)
    p.add_argument('--target_coverage', type=float, default=0.45)
    p.add_argument('--peak_delta_min', type=float, default=0.025)
    p.add_argument('--peak_ratio_min', type=float, default=1.025)
    p.add_argument('--min_local_valid_pixels', type=int, default=800)
    p.add_argument('--min_local_foreground_ratio', type=float, default=0.008)
    p.add_argument('--max_points', type=int, default=12000)
    p.add_argument('--interior_sample_ratio', type=float, default=0.08)
    p.add_argument('--semantic_weight', type=float, default=0.78)
    p.add_argument('--boundary_weight', type=float, default=0.22)
    p.add_argument('--dt_sigma_px', type=float, default=6.0)
    p.add_argument('--dt_clip_px', type=float, default=25.0)
    p.add_argument('--max_local_dim', type=int, default=900)
    p.add_argument('--max_sat_dim', type=int, default=1500)
    p.add_argument('--output_variance', type=float, default=9.0)
    p.add_argument('--poll_sec', type=float, default=0.5)
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    svc = Sam3SemanticMatchService(args)
    svc.spin()
