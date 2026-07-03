#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 image segmentation service running in a separate conda environment.

Input queue:
  input/ready_<id>.json
  input/image_<id>.png
  input/cloud_<id>.npz

Output queue:
  output/result_<id>.json
  output/label_<id>.png       mono8 label image: 0 unknown, 1 road, 2 building, 3 tree, 4 grass
  output/color_<id>.png       bgr color label image
  output/debug_<id>.png

This service only segments the camera image. Projection to LiDAR and BEV
accumulation is done by projected_semantic_bev_mapper.py in ROS Python.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# NumPy compatibility for older dependencies.
if "object" not in np.__dict__:
    np.object = object
if "bool" not in np.__dict__:
    np.bool = bool

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

PROMPTS = {
    LABEL_ROAD: ["road", "street", "pavement", "sidewalk", "drivable road", "asphalt road"],
    LABEL_BUILDING: ["building", "wall", "house", "facade", "construction", "roof"],
    LABEL_TREE: ["tree", "trees", "vegetation", "bush", "shrub"],
    LABEL_GRASS: ["grass", "lawn", "green ground", "grassland"],
}

PRIORITY = [LABEL_BUILDING, LABEL_ROAD, LABEL_TREE, LABEL_GRASS]


def normalize_proxy_env():
    """Make shell proxy variables acceptable to httpx/huggingface_hub."""
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        val = os.environ.get(key, "")
        if val.startswith("socks://"):
            fixed = "socks5://" + val[len("socks://"):]
            os.environ[key] = fixed
            print("[WARN] normalized %s from socks:// to socks5:// for httpx compatibility" % key)


def find_default_sam3_checkpoint(sam3_root):
    root = Path(sam3_root)
    candidates = [
        root / "checkpoints" / "sam3.pt",
        root / "checkpoints" / "sam3.1_multiplex.pt",
        root / "sam3.pt",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return ""


def colorize_label(label):
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k, c in COLOR_BGR.items():
        out[label == k] = c
    return out


def install_dtype_hooks(model):
    """Robust dtype hook for SAM3 mixed image/text branches."""
    try:
        import torch
    except Exception:
        return

    def recursive_cast(obj, dtype=None, device=None):
        if torch.is_tensor(obj):
            if obj.is_floating_point():
                if dtype is not None and obj.dtype != dtype:
                    obj = obj.to(dtype=dtype)
                if device is not None and obj.device.type != device:
                    obj = obj.to(device)
            return obj
        if isinstance(obj, dict):
            return {k: recursive_cast(v, dtype, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [recursive_cast(v, dtype, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(recursive_cast(v, dtype, device) for v in obj)
        return obj

    if model is None or not hasattr(model, "modules"):
        return
    if getattr(model, "_dtype_hooks_installed", False):
        return

    def make_hook(module):
        def hook(mod, args):
            try:
                ref = None
                for p in mod.parameters(recurse=False):
                    ref = p
                    break
                if ref is None:
                    for b in mod.buffers(recurse=False):
                        if torch.is_tensor(b) and b.is_floating_point():
                            ref = b
                            break
                if ref is None:
                    return args
                return recursive_cast(args, ref.dtype, ref.device.type)
            except Exception:
                return args
        return hook

    count = 0
    for m in model.modules():
        try:
            m.register_forward_pre_hook(make_hook(m))
            count += 1
        except Exception:
            pass
    model._dtype_hooks_installed = True
    print("[INFO] installed dtype hooks on SAM3 modules:", count)


class SAM3Wrapper(object):
    def __init__(self, args):
        self.args = args
        self.enabled = args.backend == "sam3"
        self.model = None
        self.processor = None
        self.torch = None
        self.autocast_dtype = None
        self.autocast_enabled = False
        if self.enabled:
            self.load_sam3()

    def load_sam3(self):
        sys.path.insert(0, self.args.sam3_root)
        normalize_proxy_env()
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        dtype = torch.float32
        if self.args.sam3_dtype == "bf16":
            dtype = torch.bfloat16
        elif self.args.sam3_dtype == "fp16":
            dtype = torch.float16
        self.autocast_dtype = dtype
        self.autocast_enabled = self.args.device.startswith("cuda") and self.args.sam3_dtype in ["bf16", "fp16"]

        ckpt = self.args.sam3_checkpoint or find_default_sam3_checkpoint(self.args.sam3_root)
        if ckpt:
            print("[INFO] using local SAM3 checkpoint: %s" % ckpt)
        else:
            print("[WARN] no local SAM3 checkpoint found under %s; SAM3 may try HuggingFace download" %
                  self.args.sam3_root)
        try:
            if ckpt:
                model = build_sam3_image_model(checkpoint_path=ckpt)
            else:
                model = build_sam3_image_model()
        except TypeError:
            model = build_sam3_image_model()

        model = model.to(device=self.args.device)
        # For SAM3, dtype conversion can be fragile. Keep model as loaded, use hooks to align tensors.
        if self.args.force_model_dtype:
            model = model.to(dtype=dtype)
        install_dtype_hooks(model)
        self.model = model
        self.processor = Sam3Processor(model)
        if hasattr(self.processor, "model"):
            install_dtype_hooks(self.processor.model)
        if hasattr(self.processor, "predictor") and hasattr(self.processor.predictor, "model"):
            install_dtype_hooks(self.processor.predictor.model)
        print("[INFO] SAM3 loaded python=%s torch=%s cuda=%s device=%s dtype=%s autocast=%s" %
              (sys.executable, torch.__version__, torch.cuda.is_available(), self.args.device, str(dtype), self.autocast_enabled))

    def _autocast_ctx(self):
        if self.torch is None or not self.autocast_enabled:
            class Dummy(object):
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            return Dummy()
        return self.torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    @staticmethod
    def extract_masks_scores(result):
        masks = None
        scores = None
        if result is None:
            return None, None
        if isinstance(result, dict):
            for key in ["masks", "pred_masks", "mask", "segmentation"]:
                if key in result:
                    masks = result[key]
                    break
            for key in ["scores", "pred_scores", "iou_scores", "confidence"]:
                if key in result:
                    scores = result[key]
                    break
        else:
            if hasattr(result, "masks"):
                masks = result.masks
            if hasattr(result, "scores"):
                scores = result.scores
        return masks, scores

    def prompt_masks(self, image_rgb, prompt):
        if not self.enabled:
            return []
        import torch
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image_rgb)
        with torch.no_grad():
            with self._autocast_ctx():
                state = self.processor.set_image(pil)
                # Try several SAM3 API variants.
                try:
                    result = self.processor.set_text_prompt(state=state, prompt=prompt)
                except TypeError:
                    try:
                        result = self.processor.set_text_prompt(prompt, state)
                    except TypeError:
                        result = self.processor.set_text_prompt(prompt=prompt)
        masks, scores = self.extract_masks_scores(result)
        if masks is None:
            return []
        if torch.is_tensor(masks):
            masks_np = masks.detach().float().cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None]
        if scores is None:
            scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
        else:
            if torch.is_tensor(scores):
                scores_np = scores.detach().float().cpu().numpy().reshape(-1)
            else:
                scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
        out = []
        for i in range(masks_np.shape[0]):
            score = float(scores_np[i]) if i < len(scores_np) else 1.0
            if score < self.args.min_mask_score:
                continue
            m = masks_np[i]
            if m.dtype != np.bool_:
                m = m > self.args.mask_threshold
            if int(m.sum()) < self.args.min_mask_area:
                continue
            out.append((m.astype(np.uint8), score))
        return out


def heuristic_segment(image_bgr):
    """Fallback segmentation for testing the projected mapping pipeline."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    label = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    vegetation = ((h > 35) & (h < 95) & (s > 35) & (v > 40))
    label[vegetation] = LABEL_TREE
    # gray bright or dark low saturation road/open
    road = (s < 55) & (gray > 45) & (gray < 210)
    label[road] = LABEL_ROAD
    # structures: high contrast edges / bright roofs / vertical-looking patches approximate
    edges = cv2.Canny(gray, 80, 160)
    dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    building = ((s < 80) & (gray > 110) & dil)
    label[building] = LABEL_BUILDING
    # grass lower priority if green and smoother
    grass = vegetation & (gray > 60)
    label[grass] = LABEL_GRASS
    label[building] = LABEL_BUILDING
    return label


class Sam3ImageMaskService(object):
    def __init__(self, args):
        self.args = args
        self.queue_dir = Path(args.queue_dir)
        self.input_dir = self.queue_dir / "input"
        self.output_dir = self.queue_dir / "output"
        self.failed_dir = self.queue_dir / "failed_sam3"
        self.done_dir = self.queue_dir / "segmented"
        for d in [self.input_dir, self.output_dir, self.failed_dir, self.done_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.sam3 = SAM3Wrapper(args)
        print("[INFO] v20 SAM3 image mask service ready queue=%s backend=%s" % (str(self.queue_dir), args.backend))

    def segment_image(self, image_bgr):
        if self.args.backend == "heuristic":
            return heuristic_segment(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        label = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.float32)

        for lab in PRIORITY:
            for prompt in PROMPTS[lab]:
                try:
                    masks = self.sam3.prompt_masks(image_rgb, prompt)
                except Exception as e:
                    print("[WARN] SAM3 prompt failed prompt=%s err=%s" % (prompt, e))
                    continue
                for m, score in masks:
                    if m.shape != label.shape:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    update = (m > 0) & (score >= score_map)
                    label[update] = lab
                    score_map[update] = score

        # Morphological cleanup per label.
        cleaned = np.zeros_like(label)
        for lab in PRIORITY:
            m = (label == lab).astype(np.uint8) * 255
            if m.sum() == 0:
                continue
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            num, cc, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= self.args.min_mask_area:
                    cleaned[cc == i] = lab
        return cleaned

    def process_request(self, meta_path):
        with open(str(meta_path), "r") as f:
            meta = json.load(f)
        req_id = meta["id"]
        image_path = Path(meta["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("cannot read image %s" % image_path)
        label = self.segment_image(image)
        color = colorize_label(label)
        debug = cv2.addWeighted(image, 0.55, color, 0.45, 0.0)

        label_path = self.output_dir / ("label_%s.png" % req_id)
        color_path = self.output_dir / ("color_%s.png" % req_id)
        debug_path = self.output_dir / ("debug_%s.png" % req_id)
        result_tmp = self.output_dir / ("result_%s.json.tmp" % req_id)
        result_path = self.output_dir / ("result_%s.json" % req_id)
        # v25: copy request meta to output. The mapper may run after this service
        # moves ready_<id>.json out of input/, so result must point to a stable
        # meta file rather than the original input path.
        meta_copy_tmp = self.output_dir / ("meta_%s.json.tmp" % req_id)
        meta_copy_path = self.output_dir / ("meta_%s.json" % req_id)
        with open(str(meta_copy_tmp), "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(str(meta_copy_tmp), str(meta_copy_path))

        cv2.imwrite(str(label_path), label)
        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(debug_path), debug)
        stats = {
            "id": req_id,
            "status": "ok",
            "meta_path": str(meta_copy_path),
            "original_meta_path": str(meta_path),
            "label_path": str(label_path),
            "color_path": str(color_path),
            "debug_path": str(debug_path),
            "road_px": int((label == LABEL_ROAD).sum()),
            "building_px": int((label == LABEL_BUILDING).sum()),
            "tree_px": int((label == LABEL_TREE).sum()),
            "grass_px": int((label == LABEL_GRASS).sum()),
            "unknown_px": int((label == LABEL_UNKNOWN).sum()),
        }
        with open(str(result_tmp), "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(str(result_tmp), str(result_path))
        try:
            os.replace(str(meta_path), str(self.done_dir / meta_path.name))
        except Exception:
            pass
        print("[INFO] segmented id=%s road=%d building=%d tree=%d grass=%d" %
              (req_id, stats["road_px"], stats["building_px"], stats["tree_px"], stats["grass_px"]))

    def run(self):
        while True:
            reqs = sorted(self.input_dir.glob("ready_*.json"))
            if not reqs:
                time.sleep(self.args.sleep_sec)
                continue
            for rp in reqs[: self.args.max_batch]:
                try:
                    self.process_request(rp)
                except Exception as e:
                    print("[ERROR] failed processing %s: %s" % (str(rp), e))
                    try:
                        os.replace(str(rp), str(self.failed_dir / rp.name))
                    except Exception:
                        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--queue_dir", default="/tmp/sam3_projected_semantic_queue")
    p.add_argument("--backend", choices=["sam3", "heuristic"], default="sam3")
    p.add_argument("--sam3_root", default="/home/dawn/software/sam3")
    p.add_argument("--sam3_checkpoint", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sam3_dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--force_model_dtype", action="store_true", help="Force model.to(dtype). Leave false if SAM3 has internal mixed dtype issues.")
    p.add_argument("--min_mask_score", type=float, default=0.20)
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--min_mask_area", type=int, default=120)
    p.add_argument("--sleep_sec", type=float, default=0.2)
    p.add_argument("--max_batch", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    service = Sam3ImageMaskService(args)
    service.run()
