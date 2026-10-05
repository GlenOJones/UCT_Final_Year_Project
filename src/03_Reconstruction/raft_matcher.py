"""RAFT-Stereo (Lipson, Teed and Deng, 3DV 2021) as a drop-in replacement for SGBM in dense_stereo.py.

A learned stereo network: it builds a correlation volume between left and right image features and
refines a disparity field over many iterations with a recurrent unit. Unlike block matching it
fills weakly textured areas by spreading depth in from where the image does have structure, which
is what the black faces of the pyramid need.

Needs PyTorch, which is not in the main venv: run dense_stereo.py with the separate environment in
envs/raft (see external/README.md for how it was set up):

    envs/raft/bin/python src/03_Reconstruction/dense_stereo.py --session Sep24 --scan mjpg_pyr_lights_2 --matcher raft

The network and the Middlebury-trained weights live in external/RAFT-Stereo.
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import PROJECT_ROOT  # noqa: E402

RAFT_DIR = os.path.join(PROJECT_ROOT, "external", "RAFT-Stereo")
# Architecture of the published checkpoints (the defaults of RAFT-Stereo's demo.py).
MODEL_ARGS = dict(hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=4,
                  corr_radius=4, n_downsample=2, context_norm="batch", slow_fast_gru=False, n_gru_layers=3)

# The published weights: file and any architecture setting that differs from MODEL_ARGS.
#   middlebury  trained up to Middlebury: high-resolution indoor scenes
#   eth3d       trained up to ETH3D: grey, low-texture indoor/outdoor scenes
#   rvc         iRaftStereo_RVC (Robust Vision Challenge 2022): a mix of datasets, built to generalise
MODELS = {"middlebury": ("raftstereo-middlebury.pth", {}),
          "eth3d": ("raftstereo-eth3d.pth", {}),
          "rvc": ("iraftstereo_rvc.pth", {"context_norm": "instance"})}


class StereoNetMatcher:
    """What every learned matcher shares: resizing, the left-right consistency check, and the
    output convention (disparity in px at full resolution, -1 where there is no measurement).
    A subclass loads its network and implements _raw(left, right)."""

    scale = 1.0
    lr_tolerance = 1.0

    def _raw(self, left, right):
        raise NotImplementedError

    def disparity(self, left, right):
        """Disparity of the left image in px at full resolution. Pixels failing the left-right check
        are set to -1 (no measurement), like SGBM's unmatched pixels."""
        h, w = left.shape
        if self.scale != 1.0:
            size = (int(round(w * self.scale)), int(round(h * self.scale)))
            left_s, right_s = (cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in (left, right))
        else:
            left_s, right_s = left, right
        d_left = self._raw(left_s, right_s)
        if self.lr_tolerance > 0:
            # The right image's disparity, by running the network on the mirrored pair: mirrored,
            # the right image becomes a left image whose matches lie to its right.
            d_right = self._raw(right_s[:, ::-1].copy(), left_s[:, ::-1].copy())[:, ::-1]
            columns = np.arange(d_left.shape[1])[None, :] - d_left
            back = cv2.remap(d_right.astype(np.float32), columns.astype(np.float32),
                             np.repeat(np.arange(d_left.shape[0], dtype=np.float32)[:, None], d_left.shape[1], 1),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e4)
            consistent = np.abs(back - d_left) < self.lr_tolerance * self.scale
        else:
            consistent = np.ones_like(d_left, bool)
        d_left = np.where(consistent & (d_left > 0), d_left, -1.0)
        if self.scale != 1.0:
            valid = cv2.resize((d_left > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            d_left = np.where(valid, cv2.resize(d_left, (w, h), interpolation=cv2.INTER_NEAREST) / self.scale, -1.0)
        return d_left.astype(np.float32)


class RaftMatcher(StereoNetMatcher):
    """RAFT-Stereo: disparity for a rectified grey pair, with an optional left-right check."""

    def __init__(self, model="middlebury", iterations=32, scale=1.0, lr_tolerance=1.0, mixed_precision=True):
        import torch
        sys.path.insert(0, RAFT_DIR)
        sys.path.insert(0, os.path.join(RAFT_DIR, "core"))
        from raft_stereo import RAFTStereo
        from utils.utils import InputPadder

        if not torch.cuda.is_available():
            raise SystemExit("RAFT-Stereo needs a CUDA GPU and a CUDA build of PyTorch (envs/raft)")
        self.torch, self.InputPadder = torch, InputPadder
        weights, overrides = MODELS[model]
        network = torch.nn.DataParallel(RAFTStereo(SimpleNamespace(mixed_precision=mixed_precision,
                                                                    **{**MODEL_ARGS, **overrides})))
        network.load_state_dict(torch.load(os.path.join(RAFT_DIR, "models", weights), map_location="cuda"))
        self.model = network.module.cuda().eval()
        self.iterations, self.scale, self.lr_tolerance = iterations, scale, lr_tolerance

    def _raw(self, left, right):
        """Disparity (positive, px at the input resolution) of `left` against `right`."""
        torch = self.torch
        pair = [torch.from_numpy(np.repeat(im[None], 3, 0)).float()[None].cuda() for im in (left, right)]
        padder = self.InputPadder(pair[0].shape, divis_by=32)
        pair = padder.pad(*pair)
        with torch.no_grad():
            _, flow = self.model(*pair, iters=self.iterations, test_mode=True)
        # RAFT-Stereo returns the horizontal flow from left to right, i.e. minus the disparity.
        return -padder.unpad(flow).squeeze().float().cpu().numpy()
