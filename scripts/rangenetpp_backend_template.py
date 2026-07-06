#!/usr/bin/env python3
"""Template backend for rangenetpp_inference_bridge.py.

Copy this file to a directory outside the ROS package, for example:

  /home/dawn/software/rangenetpp_ros_backend/rangenetpp_backend.py

Then edit rangenetpp_backend.py so RangeNetPPBackend loads your real
RangeNet++ runtime and returns one label per input point.
"""


class RangeNetPPBackend(object):
    def __init__(self, model_dir, model_config, model_weights, device, **kwargs):
        self.model_dir = model_dir
        self.model_config = model_config
        self.model_weights = model_weights
        self.device = device
        self.extra = kwargs

        # Load your real RangeNet++ model here.
        #
        # Examples of what this wrapper usually does:
        #   - import the RangeNet++ Python/C++ binding
        #   - load arch_cfg.yaml and model weights
        #   - create the spherical range-image projector
        #   - keep GPU tensors/model resident between callbacks
        raise NotImplementedError(
            "Replace rangenetpp_backend_template.py with a real RangeNet++ backend"
        )

    def infer(self, points_xyzi):
        """Return labels for points_xyzi.

        Args:
            points_xyzi: Nx4 float32 numpy array containing x y z intensity.

        Returns:
            Either labels, (labels, confidence), or
            {"labels": labels, "confidence": confidence}.

        labels should normally be raw SemanticKITTI ids:
            road=40, sidewalk=48, building=50, vegetation=70, ...
        """
        raise NotImplementedError
