#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone file-queue matcher service with coarse Chamfer/NCC search + optional Efficient-LoFTR refinement.

Run this script in the conda Python3.10/3.11 environment that has a PyTorch build
compatible with RTX 5090 / Blackwell. Do not launch it with rosrun.
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

import cv2
import numpy as np

# NumPy compatibility for older third-party deps.
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex

import torch


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


def enhance_structure(gray):
    gray = to_mono_u8(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return eq


def structure_edges(gray):
    eq = enhance_structure(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0.0)
    edges = cv2.Canny(blur, 45, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def resize_for_network(img, max_dim):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    new_w = max(32, int((w * scale) // 32 * 32))
    new_h = max(32, int((h * scale) // 32 * 32))
    if new_w != w or new_h != h:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img
    sx = float(w) / float(new_w)
    sy = float(h) / float(new_h)
    return resized, sx, sy


def ncc_score(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())



def masked_ncc_score(a, b, mask):
    """NCC computed only on BEV foreground support to avoid background-driven false high scores."""
    if mask is None:
        return ncc_score(a, b)
    mask = mask.astype(bool)
    if int(mask.sum()) < 50:
        return -1.0
    a = a.astype(np.float32)[mask]
    b = b.astype(np.float32)[mask]
    if a.size < 50 or b.size < 50:
        return -1.0
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def bev_foreground_mask(gray):
    """Extract BEV foreground while suppressing the large blank background.

    The BEV images in this project can be white-background/dark-structure or
    edge-like. This function combines dark occupancy, gradients and Canny edges.
    """
    g = to_mono_u8(gray)
    inv = 255 - g
    if float(inv.std()) < 2.0:
        return np.zeros_like(g, dtype=np.uint8)
    _, dark = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = structure_edges(g)
    grad_x = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    _, grad_bin = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(dark, edge)
    mask = cv2.bitwise_or(mask, grad_bin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def rotate_image_and_mask(img, mask, angle_deg, center):
    if abs(float(angle_deg)) < 1e-6:
        return img, mask, np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle_deg), 1.0)
    rimg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    rmask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rimg, rmask, M.astype(np.float32)


def parse_float_list(text, default=(0.0,)):
    if text is None:
        return list(default)
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    items = []
    for token in str(text).replace(';', ',').split(','):
        token = token.strip()
        if token:
            try:
                items.append(float(token))
            except Exception:
                pass
    return items if items else list(default)


def draw_debug_pair(img0, img1, pts0=None, pts1=None, mask=None, out_path=None, max_lines=160, title=''):
    h = max(img0.shape[0], img1.shape[0])
    w0, w1 = img0.shape[1], img1.shape[1]
    canvas = np.ones((h + 28, w0 + w1, 3), dtype=np.uint8) * 255
    canvas[28:28 + img0.shape[0], :w0] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    canvas[28:28 + img1.shape[0], w0:w0 + w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    if pts0 is not None and pts1 is not None and len(pts0) > 0:
        idxs = np.where(mask)[0] if mask is not None else np.arange(min(len(pts0), len(pts1)))
        if len(idxs) > max_lines:
            idxs = np.random.choice(idxs, max_lines, replace=False)
        for i in idxs:
            p0 = tuple(np.round(pts0[i]).astype(int) + np.array([0, 28]))
            p1 = tuple(np.round(pts1[i] + np.array([w0, 28])).astype(int))
            color = (0, 180, 0) if (mask is None or mask[i]) else (0, 0, 220)
            cv2.circle(canvas, p0, 2, color, -1)
            cv2.circle(canvas, p1, 2, color, -1)
            cv2.line(canvas, p0, p1, color, 1)
    if out_path:
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)
    return ''


def normalize_heatmap_u8(score_map):
    if score_map is None or score_map.size == 0:
        return None
    hm = score_map.astype(np.float32)
    mx = float(np.max(hm))
    mn = float(np.min(hm))
    if mx <= mn + 1e-9:
        return np.zeros_like(hm, dtype=np.uint8)
    hm = (hm - mn) / (mx - mn)
    return np.clip(hm * 255.0, 0, 255).astype(np.uint8)


def draw_heatmap_overlay_on_sat(sat_gray, score_map, out_path=None, alpha=0.45, title=''):
    sat_gray = to_mono_u8(sat_gray)
    base = cv2.cvtColor(sat_gray, cv2.COLOR_GRAY2BGR)
    if score_map is None or score_map.size == 0:
        canvas = base
    else:
        hm_u8 = normalize_heatmap_u8(score_map)
        if hm_u8.shape[:2] != sat_gray.shape[:2]:
            hm_u8 = cv2.resize(hm_u8, (sat_gray.shape[1], sat_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        # Suppress very low-score background so the satellite map remains visible.
        mask = hm_u8 > 8
        canvas = base.copy()
        canvas[mask] = cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0.0)[mask]
    if title:
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 900), 26), (255, 255, 255), -1)
        cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)
    return canvas


def draw_sat_bev_overlay(sat_gray, bev_gray, cand, affine=None, score_map=None, out_path=None, title=''):
    """Draw satellite background, optional similarity heatmap, and matched BEV edge overlay.

    Green pixels are transformed BEV structure; red rectangle is the selected candidate crop.
    If affine is available it maps BEV pixels into the candidate crop pixel coordinates.
    """
    sat_gray = to_mono_u8(sat_gray)
    bev_gray = to_mono_u8(bev_gray)
    base = cv2.cvtColor(sat_gray, cv2.COLOR_GRAY2BGR)
    if score_map is not None and score_map.size > 0:
        hm_u8 = normalize_heatmap_u8(score_map)
        if hm_u8.shape[:2] != sat_gray.shape[:2]:
            hm_u8 = cv2.resize(hm_u8, (sat_gray.shape[1], sat_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        mask = hm_u8 > 8
        heat = cv2.addWeighted(base, 0.65, color, 0.35, 0.0)
        base[mask] = heat[mask]

    x0, y0, w, h = int(cand.get('x0', 0)), int(cand.get('y0', 0)), int(cand.get('w', bev_gray.shape[1])), int(cand.get('h', bev_gray.shape[0]))
    x0 = max(0, min(x0, base.shape[1] - 1)); y0 = max(0, min(y0, base.shape[0] - 1))
    w = max(1, min(w, base.shape[1] - x0)); h = max(1, min(h, base.shape[0] - y0))

    bev_edge = structure_edges(bev_gray)
    if affine is not None:
        try:
            A = np.asarray(affine, dtype=np.float32).reshape(2, 3)
            bev_warp = cv2.warpAffine(bev_edge, A, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        except Exception:
            bev_warp = cv2.resize(bev_edge, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        bev_warp = cv2.resize(bev_edge, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = bev_warp > 0
    roi = base[y0:y0 + h, x0:x0 + w]
    # BEV overlay: green structure with slight thickening for visibility.
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    mask = mask_u8 > 0
    green = np.zeros_like(roi)
    green[:, :, 1] = 255
    roi[mask] = cv2.addWeighted(roi, 0.20, green, 0.80, 0.0)[mask]
    base[y0:y0 + h, x0:x0 + w] = roi
    cv2.rectangle(base, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 2)

    # Mark best estimated robot pixel if available.
    if 'sat_u' in cand and 'sat_v' in cand:
        su, sv = int(round(cand['sat_u'])), int(round(cand['sat_v']))
        if 0 <= su < base.shape[1] and 0 <= sv < base.shape[0]:
            cv2.drawMarker(base, (su, sv), (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

    if title:
        cv2.rectangle(base, (0, 0), (min(base.shape[1], 1100), 30), (255, 255, 255), -1)
        cv2.putText(base, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 220), 1, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(str(out_path), base)
        return str(out_path)
    return base


class EloftrFileService:
    def __init__(self, args):
        self.args = args
        self.queue_dir = Path(args.queue_dir)
        self.input_dir = self.queue_dir / 'input'
        self.output_dir = self.queue_dir / 'output'
        self.done_dir = self.queue_dir / 'done'
        for d in [self.input_dir, self.output_dir, self.done_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.device = args.device
        if self.device == 'cuda' and not torch.cuda.is_available():
            print('[WARN] CUDA requested but unavailable, fallback to CPU', flush=True)
            self.device = 'cpu'
        torch.set_num_threads(max(1, args.torch_num_threads))
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        self.matcher = None
        self.last_coarse_heatmap = None
        self.last_coarse_score_max = 0.0
        if not args.coarse_only:
            self.matcher = self.load_matcher()
        print('[INFO] file service ready. queue=%s device=%s coarse=%s coarse_only=%s max_image_dim=%d' %
              (self.queue_dir, self.device, args.coarse_search, args.coarse_only, args.max_image_dim), flush=True)

    def load_matcher(self):
        sys.path.insert(0, self.args.efficient_loftr_root)
        from copy import deepcopy
        from src.loftr import LoFTR, full_default_cfg, reparameter
        cfg = deepcopy(full_default_cfg)
        matcher = LoFTR(config=cfg)
        try:
            state = torch.load(self.args.weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(self.args.weights_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        matcher.load_state_dict(state)
        matcher = reparameter(matcher)
        matcher = matcher.eval().to(self.device)
        print('[INFO] Efficient-LoFTR loaded: root=%s weights=%s torch=%s cuda=%s gpu=%s' % (
            self.args.efficient_loftr_root, self.args.weights_path, torch.__version__, torch.version.cuda,
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'), flush=True)
        return matcher

    def match_loftr(self, img0, img1):
        i0, sx0, sy0 = resize_for_network(img0, self.args.max_image_dim)
        i1, sx1, sy1 = resize_for_network(img1, self.args.max_image_dim)
        ten0 = torch.from_numpy(i0)[None][None].float().to(self.device) / 255.0
        ten1 = torch.from_numpy(i1)[None][None].float().to(self.device) / 255.0
        batch = {'image0': ten0, 'image1': ten1}
        with torch.inference_mode():
            self.matcher(batch)
        mk0 = batch['mkpts0_f'].detach().cpu().numpy()
        mk1 = batch['mkpts1_f'].detach().cpu().numpy()
        conf = batch['mconf'].detach().cpu().numpy()
        del ten0, ten1, batch
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        if mk0.size == 0:
            return None, None, None
        mk0[:, 0] *= sx0; mk0[:, 1] *= sy0
        mk1[:, 0] *= sx1; mk1[:, 1] *= sy1
        return mk0.astype(np.float32), mk1.astype(np.float32), conf.astype(np.float32)

    def scale_bev_to_sat_resolution(self, bev, meta):
        """Resize BEV image to the same ground resolution as the satellite crop."""
        if meta is None or len(meta) < 18:
            return bev, meta, {'scaled': False, 'reason': 'bad_meta'}
        bev_min_e, bev_max_e, bev_min_n, bev_max_n = meta[4], meta[5], meta[6], meta[7]
        old_h, old_w = bev.shape[:2]
        sat_mpp = float(meta[17])
        if sat_mpp <= 1e-6:
            return bev, meta, {'scaled': False, 'reason': 'bad_sat_mpp'}
        bev_w_m = max(1e-6, bev_max_e - bev_min_e)
        bev_h_m = max(1e-6, bev_max_n - bev_min_n)
        target_w = max(32, int(round(bev_w_m / sat_mpp)))
        target_h = max(32, int(round(bev_h_m / sat_mpp)))
        max_dim = int(getattr(self.args, 'coarse_max_bev_dim', 900))
        if max(target_w, target_h) > max_dim:
            scale = float(max_dim) / float(max(target_w, target_h))
            target_w = max(32, int(round(target_w * scale)))
            target_h = max(32, int(round(target_h * scale)))
        if abs(target_w - old_w) < 2 and abs(target_h - old_h) < 2:
            return bev, meta, {'scaled': False, 'reason': 'already_close', 'target_w': target_w, 'target_h': target_h}
        interp = cv2.INTER_AREA if target_w < old_w or target_h < old_h else cv2.INTER_LINEAR
        resized = cv2.resize(bev, (target_w, target_h), interpolation=interp)
        new_meta = list(meta)
        new_meta[8] = meta[8] * float(target_w) / max(1.0, float(old_w))
        new_meta[9] = meta[9] * float(target_h) / max(1.0, float(old_h))
        new_meta[10] = float(target_w)
        new_meta[11] = float(target_h)
        return resized, new_meta, {'scaled': True, 'old_w': old_w, 'old_h': old_h,
                                   'target_w': target_w, 'target_h': target_h,
                                   'sat_mpp': sat_mpp, 'bev_w_m': bev_w_m, 'bev_h_m': bev_h_m}

    def coarse_candidates(self, bev, sat, meta):
        """Strict foreground Chamfer/NCC candidate search.

        v14 fixes the false-high-score issue by:
        1) scoring only BEV foreground/edge pixels instead of image background;
        2) requiring enough BEV foreground and enough satellite edge support;
        3) adding coverage and peak-uniqueness metrics;
        4) optionally searching a small yaw set around the current heading.
        """
        bh, bw = bev.shape[:2]
        sh, sw = sat.shape[:2]
        self.last_coarse_heatmap = np.zeros((sh, sw), dtype=np.float32)
        self.last_coarse_score_max = 0.0
        if sh < bh or sw < bw:
            return []

        sat_edge = structure_edges(sat)
        sat_edge_bin_full = sat_edge > 0
        sat_edge_density_full = float(sat_edge_bin_full.mean())
        if sat_edge_density_full < 1e-4:
            return []

        robot_u_bev, robot_v_bev = float(meta[8]), float(meta[9])
        yaw_degs = parse_float_list(getattr(self.args, 'coarse_yaw_degs', '0'), default=(0.0,))
        if not getattr(self.args, 'coarse_use_yaw_search', False):
            yaw_degs = [0.0]

        # Predicted robot pixel in full satellite crop, derived from current state and crop ENU bounds.
        abs_min_e, abs_max_e, abs_min_n, abs_max_n = meta[0], meta[1], meta[2], meta[3]
        state_e, state_n = meta[14], meta[15]
        pred_u = (state_e - abs_min_e) / max(1e-6, (abs_max_e - abs_min_e)) * sw
        pred_v = (abs_max_n - state_n) / max(1e-6, (abs_max_n - abs_min_n)) * sh

        radius = int(self.args.coarse_search_radius_px)
        step = max(1, int(self.args.coarse_search_step_px))
        candidates = []
        all_scores = []
        valid_grid_count = 0
        fg_reject_reason = ''

        for yaw in yaw_degs:
            bev_rot, mask_rot, M_rot = rotate_image_and_mask(
                bev, bev_foreground_mask(bev), yaw, (robot_u_bev, robot_v_bev))
            bev_edge = structure_edges(bev_rot)
            # Use edges as primary support, but constrain them inside a dilated foreground support.
            fg = mask_rot > 0
            edge_support = (bev_edge > 0) & cv2.dilate(mask_rot, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
            if int(edge_support.sum()) < int(self.args.coarse_min_bev_edge_points):
                fg_reject_reason = 'too_few_bev_edge_points'
                continue
            fg_ratio = float(fg.mean())
            if fg_ratio < float(self.args.coarse_min_foreground_ratio):
                fg_reject_reason = 'bev_foreground_ratio_low'
                continue
            blur_bev = cv2.GaussianBlur(bev_edge, (9, 9), 0)
            support_for_ncc = cv2.dilate(edge_support.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1) > 0

            for dy in range(-radius, radius + 1, step):
                for dx in range(-radius, radius + 1, step):
                    x0 = int(round(pred_u - robot_u_bev + dx))
                    y0 = int(round(pred_v - robot_v_bev + dy))
                    if x0 < 0 or y0 < 0 or x0 + bw > sw or y0 + bh > sh:
                        continue
                    valid_grid_count += 1
                    sat_crop_edge = sat_edge[y0:y0 + bh, x0:x0 + bw]
                    sat_bin = (sat_crop_edge > 0).astype(np.uint8)
                    sat_edge_points = int(sat_bin.sum())
                    if sat_edge_points < int(self.args.coarse_min_sat_edge_points):
                        continue
                    dt = cv2.distanceTransform(1 - sat_bin, cv2.DIST_L2, 3)
                    sample_dt = dt[edge_support]
                    if sample_dt.size < int(self.args.coarse_min_bev_edge_points):
                        continue
                    mean_dt = float(sample_dt.mean())
                    coverage = float(np.mean(sample_dt <= float(self.args.coarse_dt_coverage_thresh_px)))
                    if coverage < float(self.args.coarse_min_coverage):
                        # Keep it out of candidate list, but include a low score in heatmap stats.
                        score = 0.0
                    else:
                        chamfer = math.exp(-mean_dt / max(1e-6, self.args.coarse_chamfer_sigma_px))
                        ncc = masked_ncc_score(blur_bev, cv2.GaussianBlur(sat_crop_edge, (9, 9), 0), support_for_ncc)
                        ncc01 = (ncc + 1.0) * 0.5
                        # Multiplicative foreground-only score. This prevents blank background matches.
                        score = (float(chamfer) ** float(self.args.coarse_chamfer_weight)) * \
                                (float(coverage) ** float(self.args.coarse_coverage_weight)) * \
                                (0.35 + 0.65 * float(np.clip(ncc01, 0.0, 1.0)))
                        score = float(np.clip(score, 0.0, 1.0))
                    all_scores.append(score)

                    # Heatmap point is the robot location after placing the BEV at this candidate.
                    hu = int(round(x0 + robot_u_bev))
                    hv = int(round(y0 + robot_v_bev))
                    if 0 <= hu < sw and 0 <= hv < sh:
                        self.last_coarse_heatmap[hv, hu] = max(self.last_coarse_heatmap[hv, hu], score)

                    if score >= self.args.coarse_min_score:
                        candidates.append({'x0': x0, 'y0': y0, 'w': bw, 'h': bh, 'score': score,
                                           'mean_dt': float(mean_dt), 'ncc': float(ncc),
                                           'chamfer': float(math.exp(-mean_dt / max(1e-6, self.args.coarse_chamfer_sigma_px))),
                                           'coverage': float(coverage), 'sat_edge_points': int(sat_edge_points),
                                           'bev_edge_points': int(edge_support.sum()), 'bev_foreground_ratio': float(fg_ratio),
                                           'yaw_deg': float(yaw), 'rot_affine': M_rot.tolist(),
                                           'sat': sat[y0:y0 + bh, x0:x0 + bw], 'bev': bev_rot})

        if all_scores:
            self.last_coarse_score_max = float(max(all_scores))
            k = max(9, int(round(step * 2 + 1)))
            if k % 2 == 0:
                k += 1
            self.last_coarse_heatmap = cv2.GaussianBlur(self.last_coarse_heatmap, (k, k), max(1.0, step * 1.5))
            sorted_scores = sorted(all_scores, reverse=True)
            best = float(sorted_scores[0])
            second = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            p90 = float(np.percentile(np.asarray(all_scores, dtype=np.float32), 90.0)) if len(all_scores) >= 10 else second
            peak_delta = best - p90
            peak_ratio = best / max(1e-6, second)
            for c in candidates:
                c['coarse_peak_delta'] = float(peak_delta)
                c['coarse_peak_ratio'] = float(peak_ratio)
                c['coarse_score_second'] = float(second)
                c['coarse_score_p90'] = float(p90)
                c['coarse_valid_grid_count'] = int(valid_grid_count)
                c['coarse_reject_hint'] = fg_reject_reason
        candidates.sort(key=lambda c: c['score'], reverse=True)
        return candidates[:max(1, self.args.coarse_top_k)]

    def write_result(self, rid, result):
        result['request_id'] = rid
        tmp = self.output_dir / ('result_%s.json.tmp' % rid)
        out = self.output_dir / ('result_%s.json' % rid)
        with open(tmp, 'w') as f:
            json.dump(result, f, indent=2)
        os.replace(str(tmp), str(out))

    def candidate_to_pose(self, cand, qx, qy, sat_shape, meta):
        sat_h, sat_w = sat_shape[:2]
        abs_min_e, abs_max_e, abs_min_n, abs_max_n = meta[0], meta[1], meta[2], meta[3]
        sat_u = float(np.clip(cand['x0'] + qx, 0, sat_w - 1))
        sat_v = float(np.clip(cand['y0'] + qy, 0, sat_h - 1))
        meas_e = abs_min_e + sat_u / max(1.0, sat_w) * (abs_max_e - abs_min_e)
        meas_n = abs_max_n - sat_v / max(1.0, sat_h) * (abs_max_n - abs_min_n)
        return meas_e, meas_n, sat_u, sat_v

    def process_request(self, req_path):
        try:
            with open(req_path, 'r') as f:
                req = json.load(f)
        except Exception as exc:
            print('[WARN] bad request %s: %s' % (req_path, exc), flush=True)
            return
        rid = req.get('request_id', Path(req_path).stem.replace('ready_', ''))
        result = {'accepted': False, 'reason': '', 'matches': 0, 'inliers': 0,
                  'confidence': 0.0, 'inlier_ratio': 0.0, 'scale': 1.0, 'rotation_deg': 0.0}
        try:
            bev = to_mono_u8(cv2.imread(req['bev_path'], cv2.IMREAD_UNCHANGED))
            sat = to_mono_u8(cv2.imread(req['sat_path'], cv2.IMREAD_UNCHANGED))
            if bev is None or sat is None or bev.size == 0 or sat.size == 0:
                result['reason'] = 'empty_input_image'
                self.write_result(rid, result); return
            bev_eq = enhance_structure(bev)
            sat_eq = enhance_structure(sat)
            meta = req['meta']
            bev_eq, meta, scale_info = self.scale_bev_to_sat_resolution(bev_eq, meta)
            result['scale_info'] = scale_info
            if scale_info.get('scaled', False):
                print('[INFO] request %s scale BEV to satellite resolution: %dx%d -> %dx%d, sat_mpp=%.3f' %
                      (rid, scale_info.get('old_w', 0), scale_info.get('old_h', 0),
                       scale_info.get('target_w', 0), scale_info.get('target_h', 0),
                       scale_info.get('sat_mpp', 0.0)), flush=True)

            if self.args.coarse_search:
                candidates = self.coarse_candidates(bev_eq, sat_eq, meta)
            else:
                if sat_eq.shape[:2] != bev_eq.shape[:2]:
                    sat_candidate = cv2.resize(sat_eq, (bev_eq.shape[1], bev_eq.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    sat_candidate = sat_eq
                candidates = [{'x0': 0, 'y0': 0, 'w': bev_eq.shape[1], 'h': bev_eq.shape[0],
                               'score': 1.0, 'mean_dt': 0.0, 'ncc': 1.0, 'sat': sat_candidate}]
            result['coarse_candidates'] = len(candidates)
            if not candidates:
                result['reason'] = 'no_coarse_candidate'
                result['coarse_heatmap_max'] = float(self.last_coarse_score_max)
                if self.args.publish_debug:
                    hp = self.output_dir / ('heatmap_%s.png' % rid)
                    result['heatmap_path'] = draw_heatmap_overlay_on_sat(
                        sat_eq, self.last_coarse_heatmap, hp,
                        title='coarse heatmap: no valid candidate max=%.3f' % float(self.last_coarse_score_max))
                self.write_result(rid, result)
                print('[WARN] request %s no coarse candidate max_score=%.3f' % (rid, float(self.last_coarse_score_max)), flush=True)
                return

            # Coarse-only mode is useful for diagnosing whether geometry search is feasible before running LoFTR.
            if self.args.coarse_only:
                cand = candidates[0]
                robot_u, robot_v = meta[8], meta[9]
                meas_e, meas_n, sat_u, sat_v = self.candidate_to_pose(cand, robot_u, robot_v, sat_eq.shape, meta)
                confidence = float(cand['score'])
                peak_delta = float(cand.get('coarse_peak_delta', 0.0))
                peak_ratio = float(cand.get('coarse_peak_ratio', 1.0))
                coverage = float(cand.get('coverage', 0.0))
                fg_ratio = float(cand.get('bev_foreground_ratio', 0.0))
                coarse_ok = (confidence >= self.args.coarse_accept_score and
                             peak_delta >= self.args.coarse_peak_delta_min and
                             peak_ratio >= self.args.coarse_peak_ratio_min and
                             coverage >= self.args.coarse_min_coverage and
                             fg_ratio >= self.args.coarse_min_foreground_ratio)
                if confidence < self.args.coarse_accept_score:
                    coarse_reason = 'coarse_score_low'
                elif peak_delta < self.args.coarse_peak_delta_min or peak_ratio < self.args.coarse_peak_ratio_min:
                    coarse_reason = 'coarse_peak_not_unique'
                elif coverage < self.args.coarse_min_coverage:
                    coarse_reason = 'coarse_coverage_low'
                elif fg_ratio < self.args.coarse_min_foreground_ratio:
                    coarse_reason = 'bev_foreground_ratio_low'
                else:
                    coarse_reason = 'coarse_accepted'
                result.update({'accepted': coarse_ok,
                               'reason': coarse_reason,
                               'matches': 0, 'inliers': 0, 'confidence': confidence, 'inlier_ratio': confidence,
                               'meas_e': meas_e, 'meas_n': meas_n, 'variance': (self.args.match_noise_std / max(0.15, confidence)) ** 2,
                               'sat_u': sat_u, 'sat_v': sat_v,
                               'coarse_score': cand['score'], 'coarse_mean_dt': cand['mean_dt'], 'coarse_ncc': cand['ncc'],
                               'coarse_chamfer': cand.get('chamfer', 0.0), 'coarse_coverage': coverage,
                               'coarse_peak_delta': peak_delta, 'coarse_peak_ratio': peak_ratio,
                               'bev_edge_points': cand.get('bev_edge_points', 0),
                               'bev_foreground_ratio': fg_ratio, 'coarse_yaw_deg': cand.get('yaw_deg', 0.0),
                               'candidate_x0': cand['x0'], 'candidate_y0': cand['y0']})
                result['coarse_heatmap_max'] = float(self.last_coarse_score_max)
                if self.args.publish_debug:
                    dbg = self.output_dir / ('debug_%s.png' % rid)
                    result['debug_path'] = draw_debug_pair(cand.get('bev', bev_eq), cand['sat'], out_path=dbg,
                                                           title='coarse score=%.3f cov=%.2f peak=%.2f yaw=%.1f' % (cand['score'], cand.get('coverage',0.0), cand.get('coarse_peak_delta',0.0), cand.get('yaw_deg',0.0)))
                    hp = self.output_dir / ('heatmap_%s.png' % rid)
                    result['heatmap_path'] = draw_heatmap_overlay_on_sat(
                        sat_eq, self.last_coarse_heatmap, hp,
                        title='coarse heatmap best=%.3f peak=%.3f ratio=%.2f' % (cand['score'], cand.get('coarse_peak_delta',0.0), cand.get('coarse_peak_ratio',1.0)))
                    op = self.output_dir / ('overlay_%s.png' % rid)
                    cand_vis = dict(cand)
                    cand_vis['sat_u'] = sat_u
                    cand_vis['sat_v'] = sat_v
                    result['overlay_path'] = draw_sat_bev_overlay(
                        sat_eq, cand.get('bev', bev_eq), cand_vis, affine=None, score_map=self.last_coarse_heatmap, out_path=op,
                        title='BEV overlay | score=%.3f cov=%.2f peak=%.2f yaw=%.1f reason=%s' % (cand['score'], cand.get('coverage',0.0), cand.get('coarse_peak_delta',0.0), cand.get('yaw_deg',0.0), coarse_reason))
                self.write_result(rid, result)
                print('[INFO] request %s coarse-only %s score=%.3f e=%.2f n=%.2f' % (rid, result['reason'], confidence, meas_e, meas_n), flush=True)
                return

            best_result = None
            best_debug = None
            for ci, cand in enumerate(candidates):
                cand_bev = cand.get('bev', bev_eq)
                sat_crop = cand['sat']
                if sat_crop.shape[:2] != cand_bev.shape[:2]:
                    sat_crop = cv2.resize(sat_crop, (cand_bev.shape[1], cand_bev.shape[0]), interpolation=cv2.INTER_LINEAR)
                t0 = time.time()
                pts0, pts1, conf = self.match_loftr(cand_bev, sat_crop)
                infer_ms = (time.time() - t0) * 1000.0
                cand_result = {'accepted': False, 'reason': '', 'matches': 0, 'inliers': 0,
                               'confidence': 0.0, 'inlier_ratio': 0.0, 'scale': 1.0, 'rotation_deg': 0.0,
                               'candidate_index': ci, 'candidate_x0': cand['x0'], 'candidate_y0': cand['y0'],
                               'coarse_score': cand['score'], 'coarse_mean_dt': cand['mean_dt'], 'coarse_ncc': cand['ncc'],
                               'coarse_coverage': cand.get('coverage',0.0), 'coarse_peak_delta': cand.get('coarse_peak_delta',0.0),
                               'coarse_peak_ratio': cand.get('coarse_peak_ratio',1.0), 'coarse_yaw_deg': cand.get('yaw_deg',0.0),
                               'bev_edge_points': cand.get('bev_edge_points',0), 'bev_foreground_ratio': cand.get('bev_foreground_ratio',0.0),
                               'infer_ms': infer_ms}
                if pts0 is None or len(pts0) < self.args.min_matches:
                    cand_result.update({'reason': 'too_few_matches', 'matches': 0 if pts0 is None else int(len(pts0))})
                else:
                    A, mask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC,
                                                         ransacReprojThreshold=self.args.ransac_thresh_px,
                                                         maxIters=2000, confidence=0.995, refineIters=5)
                    if A is None or mask is None:
                        cand_result.update({'reason': 'affine_failed', 'matches': int(len(pts0))})
                    else:
                        mask = mask.reshape(-1).astype(bool)
                        inliers = int(mask.sum())
                        inlier_ratio = float(inliers) / max(1, len(pts0))
                        mean_conf = float(np.mean(conf[mask])) if inliers > 0 else 0.0
                        a = float(A[0, 0]); b = float(A[1, 0])
                        scale = math.sqrt(a*a + b*b)
                        rot = math.degrees(math.atan2(b, a))
                        robot_u, robot_v = meta[8], meta[9]
                        q = A.dot(np.asarray([robot_u, robot_v, 1.0], dtype=np.float64))
                        meas_e, meas_n, sat_u, sat_v = self.candidate_to_pose(cand, q[0], q[1], sat_eq.shape, meta)
                        cand_result.update({'matches': int(len(pts0)), 'inliers': inliers, 'confidence': mean_conf,
                                            'inlier_ratio': inlier_ratio, 'scale': scale, 'rotation_deg': rot,
                                            'meas_e': meas_e, 'meas_n': meas_n, 'sat_u': sat_u, 'sat_v': sat_v,
                                            'variance': (self.args.match_noise_std / max(0.15, mean_conf)) ** 2})
                        if cand.get('score', 0.0) < self.args.coarse_accept_score:
                            cand_result['reason'] = 'coarse_score_low'
                        elif cand.get('coarse_peak_delta', 0.0) < self.args.coarse_peak_delta_min or cand.get('coarse_peak_ratio', 1.0) < self.args.coarse_peak_ratio_min:
                            cand_result['reason'] = 'coarse_peak_not_unique'
                        elif cand.get('coverage', 0.0) < self.args.coarse_min_coverage:
                            cand_result['reason'] = 'coarse_coverage_low'
                        elif inliers < self.args.min_inliers or inlier_ratio < self.args.min_inlier_ratio or mean_conf < self.args.min_confidence:
                            cand_result['reason'] = 'weak_geometry'
                        elif scale < self.args.min_scale or scale > self.args.max_scale or abs(rot) > self.args.max_rotation_deg:
                            cand_result['reason'] = 'affine_out_of_bounds'
                        else:
                            cand_result['accepted'] = True
                            cand_result['reason'] = 'accepted'
                        if self.args.publish_debug:
                            dbg = self.output_dir / ('debug_%s_cand%d.png' % (rid, ci))
                            dbg_path = draw_debug_pair(cand_bev, sat_crop, pts0, pts1, mask, dbg,
                                                       title='cand=%d coarse=%.3f cov=%.2f yaw=%.1f conf=%.2f inliers=%d/%d' % (ci, cand['score'], cand.get('coverage',0.0), cand.get('yaw_deg',0.0), mean_conf, inliers, len(pts0)))
                            cand_result['debug_path'] = dbg_path
                            cand_result['affine'] = A.tolist()
                            hp = self.output_dir / ('heatmap_%s_cand%d.png' % (rid, ci))
                            cand_result['heatmap_path'] = draw_heatmap_overlay_on_sat(
                                sat_eq, self.last_coarse_heatmap, hp,
                                title='coarse heatmap best=%.3f peak=%.3f ratio=%.2f' % (cand['score'], cand.get('coarse_peak_delta',0.0), cand.get('coarse_peak_ratio',1.0)))
                            op = self.output_dir / ('overlay_%s_cand%d.png' % (rid, ci))
                            cand_vis = dict(cand)
                            cand_vis['sat_u'] = sat_u
                            cand_vis['sat_v'] = sat_v
                            cand_result['overlay_path'] = draw_sat_bev_overlay(
                                sat_eq, cand_bev, cand_vis, affine=A, score_map=self.last_coarse_heatmap, out_path=op,
                                title='BEV overlay | coarse=%.3f cov=%.2f yaw=%.1f conf=%.2f inliers=%d/%d e=%.2f n=%.2f' % (cand['score'], cand.get('coverage',0.0), cand.get('yaw_deg',0.0), mean_conf, inliers, len(pts0), meas_e, meas_n))
                            if best_debug is None:
                                best_debug = dbg_path
                if best_result is None:
                    best_result = cand_result
                else:
                    # Accepted candidates dominate; otherwise use a blend of inliers and coarse score.
                    key = (1 if cand_result.get('accepted') else 0,
                           cand_result.get('inliers', 0), cand_result.get('confidence', 0.0), cand_result.get('coarse_score', 0.0))
                    best_key = (1 if best_result.get('accepted') else 0,
                                best_result.get('inliers', 0), best_result.get('confidence', 0.0), best_result.get('coarse_score', 0.0))
                    if key > best_key:
                        best_result = cand_result
            if best_result is None:
                result['reason'] = 'no_candidate_processed'
                self.write_result(rid, result); return
            result.update(best_result)
            result['coarse_heatmap_max'] = float(self.last_coarse_score_max)
            if self.args.publish_debug and 'debug_path' not in result and best_debug:
                result['debug_path'] = best_debug
            if self.args.publish_debug and 'heatmap_path' not in result:
                hp = self.output_dir / ('heatmap_%s_best.png' % rid)
                result['heatmap_path'] = draw_heatmap_overlay_on_sat(
                    sat_eq, self.last_coarse_heatmap, hp,
                    title='coarse similarity heatmap max=%.3f' % float(self.last_coarse_score_max))
            if self.args.publish_debug and 'overlay_path' not in result and candidates:
                # Fall back to a coarse-only overlay for rejected candidates that did not reach affine estimation.
                cand0 = candidates[0]
                robot_u, robot_v = meta[8], meta[9]
                meas_e0, meas_n0, sat_u0, sat_v0 = self.candidate_to_pose(cand0, robot_u, robot_v, sat_eq.shape, meta)
                cand_vis = dict(cand0)
                cand_vis['sat_u'] = sat_u0
                cand_vis['sat_v'] = sat_v0
                op = self.output_dir / ('overlay_%s_best.png' % rid)
                result['overlay_path'] = draw_sat_bev_overlay(
                    sat_eq, cand0.get('bev', bev_eq), cand_vis, affine=None, score_map=self.last_coarse_heatmap, out_path=op,
                    title='BEV coarse overlay | score=%.3f cov=%.2f peak=%.2f yaw=%.1f e=%.2f n=%.2f reason=%s' % (cand0.get('score', 0.0), cand0.get('coverage',0.0), cand0.get('coarse_peak_delta',0.0), cand0.get('yaw_deg',0.0), meas_e0, meas_n0, result.get('reason', '')))
            self.write_result(rid, result)
            if result.get('accepted'):
                print('[INFO] request %s accepted: e=%.2f n=%.2f conf=%.2f ratio=%.2f inliers=%d matches=%d coarse=%.3f' % (
                    rid, result['meas_e'], result['meas_n'], result['confidence'], result['inlier_ratio'], result['inliers'], result['matches'], result.get('coarse_score', 0.0)), flush=True)
            else:
                print('[WARN] request %s rejected: %s conf=%.2f ratio=%.2f inliers=%d matches=%d coarse=%.3f candidates=%d' % (
                    rid, result.get('reason', ''), result.get('confidence', 0.0), result.get('inlier_ratio', 0.0), result.get('inliers', 0), result.get('matches', 0), result.get('coarse_score', 0.0), len(candidates)), flush=True)
        except Exception as exc:
            result['reason'] = 'exception:%s' % str(exc)
            self.write_result(rid, result)
            print('[ERROR] request %s failed: %s' % (rid, exc), flush=True)
        finally:
            try:
                shutil.move(str(req_path), str(self.done_dir / os.path.basename(req_path)))
            except Exception:
                try:
                    os.remove(req_path)
                except Exception:
                    pass

    def run(self):
        while True:
            ready = sorted(glob.glob(str(self.input_dir / 'ready_*.json')))
            if not ready:
                time.sleep(self.args.poll_interval)
                continue
            for req_path in ready:
                self.process_request(req_path)
            time.sleep(self.args.poll_interval)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--queue_dir', default='/tmp/eloftr_queue')
    p.add_argument('--efficient_loftr_root', required=False, default='')
    p.add_argument('--weights_path', required=False, default='')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--max_image_dim', type=int, default=640)
    p.add_argument('--torch_num_threads', type=int, default=2)
    p.add_argument('--poll_interval', type=float, default=0.2)
    p.add_argument('--min_matches', type=int, default=20)
    p.add_argument('--min_inliers', type=int, default=10)
    p.add_argument('--min_confidence', type=float, default=0.25)
    p.add_argument('--min_inlier_ratio', type=float, default=0.20)
    p.add_argument('--ransac_thresh_px', type=float, default=4.0)
    p.add_argument('--max_rotation_deg', type=float, default=25.0)
    p.add_argument('--min_scale', type=float, default=0.70)
    p.add_argument('--max_scale', type=float, default=1.30)
    p.add_argument('--match_noise_std', type=float, default=1.5)
    p.add_argument('--publish_debug', action='store_true')

    # v11 coarse search parameters.
    p.add_argument('--coarse_search', action='store_true', default=True)
    p.add_argument('--no_coarse_search', dest='coarse_search', action='store_false')
    p.add_argument('--coarse_only', action='store_true')
    p.add_argument('--coarse_search_radius_px', type=int, default=80)
    p.add_argument('--coarse_search_step_px', type=int, default=8)
    p.add_argument('--coarse_top_k', type=int, default=3)
    p.add_argument('--coarse_min_score', type=float, default=0.15)
    p.add_argument('--coarse_accept_score', type=float, default=0.50)
    p.add_argument('--coarse_chamfer_sigma_px', type=float, default=5.0)
    p.add_argument('--coarse_chamfer_weight', type=float, default=0.70)
    p.add_argument('--coarse_coverage_weight', type=float, default=0.80)
    p.add_argument('--coarse_dt_coverage_thresh_px', type=float, default=4.0)
    p.add_argument('--coarse_min_coverage', type=float, default=0.12)
    p.add_argument('--coarse_min_foreground_ratio', type=float, default=0.008)
    p.add_argument('--coarse_min_bev_edge_points', type=int, default=300)
    p.add_argument('--coarse_min_sat_edge_points', type=int, default=300)
    p.add_argument('--coarse_peak_delta_min', type=float, default=0.035)
    p.add_argument('--coarse_peak_ratio_min', type=float, default=1.03)
    p.add_argument('--coarse_use_yaw_search', action='store_true')
    p.add_argument('--coarse_yaw_degs', default='0')
    p.add_argument('--coarse_max_bev_dim', type=int, default=900)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    service = EloftrFileService(args)
    service.run()
