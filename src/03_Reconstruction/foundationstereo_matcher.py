"""FoundationStereo (Wen, Shao, Gu, Nam, Kautz, Birchfield and Fox, CVPR 2025) as a matcher for dense_stereo.py.

A large stereo network trained for zero-shot use on a million photorealistic synthetic pairs. Its
features come from a side-tuned Depth Anything V2 (DINOv2) monocular depth backbone plus a CNN, its
cost volume is filtered with long-range attention, and disparity is refined iteratively like
RAFT-Stereo.

Needs PyTorch, timm, open3d, pandas and huggingface_hub, kept out of the main venv in envs/ffs (the
name is historical; see external/README.md):

    envs/ffs/bin/python src/03_Reconstruction/dense_stereo.py --session Sep24 --scan mjpg_pyr_lights_2 \
        --matcher foundationstereo --net-scale 0.5

The network code and the checkpoints are in external/FoundationStereo. The left-right check, scaling
and output convention are shared with RAFT-Stereo (StereoNetMatcher in raft_matcher.py).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import PROJECT_ROOT  # noqa: E402
from raft_matcher import StereoNetMatcher  # noqa: E402

FS_DIR = os.path.join(PROJECT_ROOT, "external", "FoundationStereo")
# The published checkpoints: folder and ViT backbone.
#   vitl  23-51-11, the authors' best model for general use
#   vits  11-33-40, slightly less accurate, faster and lighter
MODELS = {"vitl": "23-51-11", "vits": "11-33-40"}


class FoundationStereoMatcher(StereoNetMatcher):
    """FoundationStereo disparity for a rectified grey pair, with the shared left-right check."""

    def __init__(self, model="vitl", iterations=32, scale=1.0, lr_tolerance=1.0, max_disp=416):
        import torch
        from omegaconf import OmegaConf
        sys.path.insert(0, FS_DIR)   # FoundationStereo imports its modules as core.*, Utils and depth_anything.*
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        if not torch.cuda.is_available():
            raise SystemExit("FoundationStereo needs a CUDA GPU and a CUDA build of PyTorch (envs/ffs)")
        folder = os.path.join(FS_DIR, "pretrained_models", MODELS[model])
        # As the authors' run_demo.py: the checkpoint's training config, ViT-L when it does not say.
        cfg = OmegaConf.load(os.path.join(folder, "cfg.yaml"))
        cfg["vit_size"] = cfg.get("vit_size", "vitl")
        cfg["valid_iters"], cfg["max_disp"] = iterations, max_disp
        network = FoundationStereo(cfg)
        checkpoint = torch.load(os.path.join(folder, "model_best_bp2.pth"), map_location="cpu", weights_only=False)
        network.load_state_dict(checkpoint["model"])
        self.model = network.cuda().eval()
        self.torch, self.InputPadder = torch, InputPadder
        self.iterations, self.scale, self.lr_tolerance = iterations, scale, lr_tolerance

    def _raw(self, left, right):
        """Disparity (positive, px at the input resolution) of `left` against `right`."""
        torch = self.torch
        pair = [torch.from_numpy(np.repeat(im[None], 3, 0)).float()[None].cuda() for im in (left, right)]
        padder = self.InputPadder(pair[0].shape, divis_by=32, force_square=False)
        pair = padder.pad(*pair)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            disparity = self.model.forward(*pair, iters=self.iterations, test_mode=True)
        return padder.unpad(disparity.float()).squeeze().cpu().numpy().clip(0, None)
