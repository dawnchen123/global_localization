#!/usr/bin/env python3
import importlib
import os
import struct
import sys

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


SEMANTIC_KITTI_LEARNING_TO_RAW = {
    0: 0,
    1: 10,
    2: 11,
    3: 15,
    4: 18,
    5: 20,
    6: 30,
    7: 31,
    8: 32,
    9: 40,
    10: 44,
    11: 48,
    12: 49,
    13: 50,
    14: 51,
    15: 70,
    16: 71,
    17: 72,
    18: 80,
    19: 81,
    20: 99,
}

SEMANTIC_TO_INTERNAL = {
    40: 1,
    44: 1,
    60: 1,
    48: 2,
    49: 2,
    72: 2,
    50: 3,
    51: 3,
    52: 3,
    70: 4,
    71: 4,
    80: 4,
    81: 4,
    10: 5,
    11: 5,
    13: 5,
    15: 5,
    16: 5,
    18: 5,
    20: 5,
    30: 5,
    31: 5,
    32: 5,
    252: 5,
    253: 5,
    254: 5,
    255: 5,
    256: 5,
    257: 5,
    258: 5,
    259: 5,
    99: 6,
}

INTERNAL_TO_SEMANTIC = {
    0: 0,
    1: 40,
    2: 48,
    3: 50,
    4: 70,
    5: 10,
    6: 99,
}

INTERNAL_COLORS = {
    0: (90, 90, 90),
    1: (130, 130, 130),
    2: (60, 210, 60),
    3: (235, 25, 25),
    4: (20, 175, 40),
    5: (255, 135, 0),
    6: (70, 95, 230),
}


def rgb_float(r, g, b):
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


RGB_FLOAT_BY_INTERNAL = {
    label: rgb_float(*color) for label, color in INTERNAL_COLORS.items()
}


def map_by_dict(labels, mapping, default=0):
    labels = np.asarray(labels, dtype=np.uint32)
    out = np.full(labels.shape, int(default), dtype=np.uint32)
    for src, dst in mapping.items():
        out[labels == int(src)] = int(dst)
    return out


def semantic_kitti_to_internal(labels):
    return map_by_dict(labels, SEMANTIC_TO_INTERNAL, default=0)


def learning_to_semantic_kitti(labels):
    return map_by_dict(labels, SEMANTIC_KITTI_LEARNING_TO_RAW, default=0)


def internal_to_semantic_kitti(labels):
    return map_by_dict(labels, INTERNAL_TO_SEMANTIC, default=0)


def first_present_field(fields, candidates):
    names = set(fields)
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


class RangeNetPPInferenceBridge(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_cloud_topic", "/cloud_registered")
        self.output_topic = rospy.get_param("~output_cloud_topic", "/rangenet/semantic_points")
        self.backend_python_path = rospy.get_param("~backend_python_path", "")
        self.backend_module = rospy.get_param("~backend_module", "")
        self.backend_class = rospy.get_param("~backend_class", "RangeNetPPBackend")
        self.model_dir = rospy.get_param("~model_dir", "")
        self.model_config = rospy.get_param("~model_config", "")
        self.model_weights = rospy.get_param("~model_weights", "")
        self.device = rospy.get_param("~device", "cuda")
        self.backend_label_space = rospy.get_param("~backend_label_space", "semantic_kitti")
        self.output_label_mode = rospy.get_param("~output_label_mode", "semantic_kitti")
        self.confidence_default = float(rospy.get_param("~confidence_default", 1.0))
        self.point_stride = max(1, int(rospy.get_param("~point_stride", 1)))
        self.max_points = int(rospy.get_param("~max_points", 120000))
        self.min_range_m = float(rospy.get_param("~min_range_m", 0.3))
        self.max_range_m = float(rospy.get_param("~max_range_m", 120.0))
        self.require_intensity = bool(rospy.get_param("~require_intensity", False))
        self.default_intensity = float(rospy.get_param("~default_intensity", 0.0))
        self.intensity_fields = rospy.get_param("~intensity_field_candidates", ["intensity", "reflectivity"])
        self.publish_every_n = max(1, int(rospy.get_param("~publish_every_n", 1)))
        self.queue_size = int(rospy.get_param("~subscriber_queue_size", 1))
        self.backend_extra_args = rospy.get_param("~backend_extra_args", {})
        self._frame_count = 0

        self.backend = self._load_backend()
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.cloud_cb,
            queue_size=self.queue_size,
            buff_size=16 * 1024 * 1024,
        )
        rospy.loginfo(
            "rangenetpp inference bridge started input=%s output=%s backend=%s.%s "
            "backend_label_space=%s output_label_mode=%s",
            self.input_topic,
            self.output_topic,
            self.backend_module,
            self.backend_class,
            self.backend_label_space,
            self.output_label_mode,
        )

    def _load_backend(self):
        if not self.backend_module:
            rospy.logfatal(
                "RangeNet++ backend_module is empty. Set backend_python_path/backend_module/"
                "backend_class in rangenetpp_inference_bridge.yaml."
            )
            rospy.signal_shutdown("missing RangeNet++ backend")
            return None

        for path in str(self.backend_python_path).split(":"):
            path = os.path.expanduser(path.strip())
            if path and path not in sys.path:
                sys.path.insert(0, path)

        try:
            module = importlib.import_module(self.backend_module)
            backend_cls = getattr(module, self.backend_class)
            kwargs = dict(self.backend_extra_args) if isinstance(self.backend_extra_args, dict) else {}
            return backend_cls(
                model_dir=self.model_dir,
                model_config=self.model_config,
                model_weights=self.model_weights,
                device=self.device,
                **kwargs
            )
        except Exception as exc:
            rospy.logfatal("Failed to load RangeNet++ backend: %s", exc)
            rospy.signal_shutdown("failed to load RangeNet++ backend")
            return None

    def cloud_cb(self, msg):
        if self.backend is None:
            return

        self._frame_count += 1
        if (self._frame_count - 1) % self.publish_every_n != 0:
            return

        field_names = [f.name for f in msg.fields]
        if not {"x", "y", "z"}.issubset(set(field_names)):
            rospy.logwarn_throttle(2.0, "RangeNet++ input cloud lacks x/y/z fields: %s", ",".join(field_names))
            return

        intensity_field = first_present_field(field_names, self.intensity_fields)
        if self.require_intensity and not intensity_field:
            rospy.logwarn_throttle(2.0, "RangeNet++ input cloud has no intensity/reflectivity field")
            return

        read_fields = ["x", "y", "z"]
        if intensity_field:
            read_fields.append(intensity_field)

        points = []
        min_r2 = self.min_range_m * self.min_range_m
        max_r2 = self.max_range_m * self.max_range_m
        try:
            for p in pc2.read_points(msg, field_names=read_fields, skip_nans=True):
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])
                r2 = x * x + y * y + z * z
                if r2 < min_r2 or r2 > max_r2:
                    continue
                intensity = float(p[3]) if intensity_field else self.default_intensity
                points.append((x, y, z, intensity))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed reading RangeNet++ input cloud: %s", exc)
            return

        if not points:
            rospy.logwarn_throttle(2.0, "RangeNet++ input cloud has no valid filtered points")
            return

        pts = np.asarray(points, dtype=np.float32)
        if self.point_stride > 1:
            pts = pts[:: self.point_stride]
        if self.max_points > 0 and pts.shape[0] > self.max_points:
            idx = np.linspace(0, pts.shape[0] - 1, self.max_points, dtype=np.int64)
            pts = pts[idx]

        labels, confidence = self._infer(pts)
        if labels is None:
            return
        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] != pts.shape[0]:
            rospy.logerr_throttle(
                2.0,
                "RangeNet++ backend returned %d labels for %d points",
                int(labels.shape[0]),
                int(pts.shape[0]),
            )
            return

        if confidence is None:
            confidence = np.full(labels.shape, self.confidence_default, dtype=np.float32)
        else:
            confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
            if confidence.shape[0] != pts.shape[0]:
                confidence = np.full(labels.shape, self.confidence_default, dtype=np.float32)
        confidence = np.clip(confidence, 0.0, 1.0)

        output_labels, internal_labels = self._normalize_labels(labels)
        out_msg = self._make_cloud(msg.header, pts[:, :3], output_labels, internal_labels, confidence)
        self.pub.publish(out_msg)

        counts = np.bincount(internal_labels.astype(np.int64), minlength=7)
        rospy.loginfo_throttle(
            2.0,
            "rangenetpp inference published points=%d road=%d sidewalk=%d building=%d "
            "vegetation=%d dynamic=%d other=%d",
            int(pts.shape[0]),
            int(counts[1]),
            int(counts[2]),
            int(counts[3]),
            int(counts[4]),
            int(counts[5]),
            int(counts[6]),
        )

    def _infer(self, pts):
        try:
            result = self.backend.infer(pts)
        except Exception as exc:
            rospy.logerr_throttle(2.0, "RangeNet++ backend inference failed: %s", exc)
            return None, None

        if isinstance(result, tuple) or isinstance(result, list):
            labels = result[0] if len(result) > 0 else None
            confidence = result[1] if len(result) > 1 else None
            return labels, confidence

        if isinstance(result, dict):
            labels = None
            for key in ("labels", "label", "pred", "prediction"):
                if key in result:
                    labels = result[key]
                    break
            confidence = None
            for key in ("confidence", "conf", "prob", "probability"):
                if key in result:
                    confidence = result[key]
                    break
            return labels, confidence

        return result, None

    def _normalize_labels(self, labels):
        labels = np.asarray(labels, dtype=np.uint32)
        if self.backend_label_space == "semantic_kitti_learning":
            semantic_labels = learning_to_semantic_kitti(labels)
            internal_labels = semantic_kitti_to_internal(semantic_labels)
        elif self.backend_label_space == "internal":
            internal_labels = labels
            semantic_labels = internal_to_semantic_kitti(internal_labels)
        else:
            semantic_labels = labels
            internal_labels = semantic_kitti_to_internal(semantic_labels)

        if self.output_label_mode == "internal":
            output_labels = internal_labels
        else:
            output_labels = semantic_labels
        return output_labels.astype(np.uint32), internal_labels.astype(np.uint32)

    def _make_cloud(self, header, xyz, labels, internal_labels, confidence):
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
            PointField("label", 16, PointField.UINT32, 1),
            PointField("confidence", 20, PointField.FLOAT32, 1),
        ]
        rows = []
        for i in range(xyz.shape[0]):
            label_internal = int(internal_labels[i])
            rgb = RGB_FLOAT_BY_INTERNAL.get(label_internal, RGB_FLOAT_BY_INTERNAL[0])
            rows.append(
                (
                    float(xyz[i, 0]),
                    float(xyz[i, 1]),
                    float(xyz[i, 2]),
                    rgb,
                    int(labels[i]),
                    float(confidence[i]),
                )
            )
        return pc2.create_cloud(header, fields, rows)


def main():
    rospy.init_node("rangenetpp_inference_bridge")
    RangeNetPPInferenceBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
