"""Fast-FoundationStereo (Wen, Dewan and Birchfield, CVPR 2026) as a matcher for dense_stereo.py.

A distilled, pruned student of NVIDIA's FoundationStereo, a stereo "foundation model" trained for
zero-shot use on unseen scenes (synthetic data plus 1.4M pseudo-labelled real pairs). It builds a
group-wise correlation volume, filters it with a searched 3D-conv design and refines disparity with
a small recurrent unit, like RAFT-Stereo but with far fewer iterations (8 by default). The authors
note it does best on images under ~1000 px wide, so --net-scale 0.5-0.75 suits this rig.

Needs PyTorch and timm, kept out of the main venv: run dense_stereo.py with envs/ffs (see
external/README.md for the setup):

    envs/ffs/bin/python src/03_Reconstruction/dense_stereo.py --session Sep24 --scan mjpg_pyr_lights_2 \
        --matcher ffs --net-iters 8 --net-scale 0.75

The network code and the research checkpoints are in external/Fast-FoundationStereo. The left-right
check, scaling and output convention are shared with RAFT-Stereo (StereoNetMatcher in raft_matcher.py).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import PROJECT_ROOT  # noqa: E402
from raft_matcher import StereoNetMatcher  # noqa: E402

FFS_DIR = os.path.join(PROJECT_ROOT, "external", "Fast-FoundationStereo")
# The research checkpoints, slowest and most accurate first (the authors' table; 15-44-51 is a later
# upload not in it). Each folder holds the whole pickled model plus its cfg.yaml.
MODELS = ("23-36-37", "20-26-39", "20-30-48", "15-44-51")


class FastFoundationStereoMatcher(StereoNetMatcher):
    """Fast-FoundationStereo disparity for a rectified grey pair, with the shared left-right check."""

    def __init__(self, model="23-36-37", iterations=8, scale=1.0, lr_tolerance=1.0, max_disp=416):
        import torch
        sys.path.insert(0, FFS_DIR)   # the pickled model refers to its classes as core.* and Utils
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        if not torch.cuda.is_available():
            raise SystemExit("Fast-FoundationStereo needs a CUDA GPU and a CUDA build of PyTorch (envs/ffs)")
        network = torch.load(os.path.join(FFS_DIR, "weights", model, "model_best_bp2_serialize.pth"),
                             map_location="cpu", weights_only=False)
        # The cost volume spans max_disp px at the network's input scale; this rig's ~250-350 px at
        # full size needs more than the demo's 192 unless the images are halved.
        network.args.valid_iters, network.args.max_disp = iterations, max_disp
        self.model = network.cuda().eval()
        self.torch, self.InputPadder, self.amp_dtype = torch, InputPadder, AMP_DTYPE
        self.iterations, self.scale, self.lr_tolerance = iterations, scale, lr_tolerance

    def _raw(self, left, right):
        """Disparity (positive, px at the input resolution) of `left` against `right`."""
        torch = self.torch
        pair = [torch.from_numpy(np.repeat(im[None], 3, 0)).float()[None].cuda() for im in (left, right)]
        padder = self.InputPadder(pair[0].shape, divis_by=32, force_square=False)
        pair = padder.pad(*pair)
        # 'pytorch1' builds the correlation volume in plain PyTorch rather than the Triton kernel.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.amp_dtype):
            disparity = self.model.forward(*pair, iters=self.iterations, test_mode=True,
                                           optimize_build_volume="pytorch1")
        return padder.unpad(disparity.float()).squeeze().cpu().numpy().clip(0, None)
