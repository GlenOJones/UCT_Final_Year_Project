"""AquaStereo (Wei, Liang, You and Fu, ECCV 2026) as a matcher for dense_stereo.py.

A learned stereo network built for underwater images: trained on underwater stereo pairs rendered by
a depth-conditioned diffusion model (so the binocular geometry stays exact), with features from a
frozen DINOv2 encoder and an X3D video backbone, and self-distilled from a teacher trained in air.
It predicts disparity from one rectified pair, iteratively refined like RAFT-Stereo, from a cost
volume that covers up to 768 px of disparity (plenty for this rig's ~250-350 px at full size).

Needs PyTorch and timm, kept out of the main venv: run dense_stereo.py with envs/aquastereo (see
external/README.md for the setup):

    envs/aquastereo/bin/python src/03_Reconstruction/dense_stereo.py --session Sep24 --scan mjpg_pyr_lights_2 --matcher aquastereo

The network code and the checkpoints are in external/AquaStereo. The left-right check, scaling and
output convention are shared with RAFT-Stereo (StereoNetMatcher in raft_matcher.py).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import PROJECT_ROOT  # noqa: E402
from raft_matcher import StereoNetMatcher  # noqa: E402

AQUA_DIR = os.path.join(PROJECT_ROOT, "external", "AquaStereo")
CHECKPOINTS = {"vitb": "AquaStereo_vitb_best.pth", "vits": "AquaStereo_vits_best.pth"}

# The defaults of AquaStereo's evaluate_stereo.py; the checkpoint's own saved arguments override them.
MODEL_ARGS = dict(hidden_dims=[128] * 3, corr_levels=2, corr_radius=4, n_downsample=2, n_gru_layers=3,
                  max_disp=768, s_disp_range=48, m_disp_range=96, l_disp_range=192, s_disp_interval=1,
                  m_disp_interval=2, l_disp_interval=4, num_perception_frame=2)


def load_checkpoint(model, path):
    """Load an AquaStereo checkpoint into a bare (unwrapped) model, strictly.

    Mirrors read_checkpoint_metadata / load_checkpoint_for_eval in AquaStereo's evaluate_stereo.py,
    which cannot be imported here without its dataset loaders' dependencies (imageio,
    scikit-image, ...). Weight names lose their "module."/"model." wrappers, and the learned
    perception frames are resized to this model's shape as the original does. Returns the
    checkpoint's saved training arguments."""
    import torch
    import torch.nn.functional as F
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = dict(checkpoint.get("args") or {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
    own = model.state_dict()
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        if key in own:
            cleaned[key] = value
    for key, value in cleaned.items():
        if key.endswith("encoder.perception_frames") and value.shape != own[key].shape:
            _, c, t, h0, w0 = value.shape
            flat = value.permute(0, 2, 1, 3, 4).reshape(-1, c, h0, w0).float()
            flat = F.interpolate(flat, size=own[key].shape[-2:], mode="bilinear", align_corners=False)
            cleaned[key] = flat.reshape(1, t, c, *own[key].shape[-2:]).permute(0, 2, 1, 3, 4).to(own[key].dtype)
    model.load_state_dict(cleaned, strict=True)
    return saved_args


class AquaStereoMatcher(StereoNetMatcher):
    """AquaStereo disparity for a rectified grey pair, with the shared left-right check."""

    def __init__(self, model="vitb", iterations=32, scale=1.0, lr_tolerance=1.0, precision="float32"):
        import torch
        sys.path.insert(0, AQUA_DIR)   # AquaStereo imports its modules as core.* and dinov2.*
        from core.AquaStereo import AquaStereo
        from core.utils.utils import InputPadder

        if not torch.cuda.is_available():
            raise SystemExit("AquaStereo needs a CUDA GPU and a CUDA build of PyTorch (envs/aquastereo)")
        checkpoint = os.path.join(AQUA_DIR, "checkpoints", CHECKPOINTS[model])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved = dict(saved.get("args") or {}) if isinstance(saved, dict) else {}
        args = {**MODEL_ARGS, **{k: v for k, v in saved.items() if k in MODEL_ARGS}}
        # restore_ckpt tells the encoder to take DINOv2 and X3D from the checkpoint rather than
        # from separate pretraining files.
        args = SimpleNamespace(**args, vit_size=model, restore_ckpt=checkpoint,
                               mixed_precision=precision != "float32", precision_dtype=precision)
        network = AquaStereo(args)
        load_checkpoint(network, checkpoint)
        self.model = network.cuda().eval()
        self.torch, self.InputPadder = torch, InputPadder
        self.iterations, self.scale, self.lr_tolerance = iterations, scale, lr_tolerance

    def _raw(self, left, right):
        """Disparity (positive, px at the input resolution) of `left` against `right`."""
        torch = self.torch
        pair = [torch.from_numpy(np.repeat(im[None], 3, 0)).float()[None].cuda() for im in (left, right)]
        padder = self.InputPadder(pair[0].shape, divis_by=32)
        pair = padder.pad(*pair)
        with torch.no_grad():
            disparity = self.model(*pair, iters=self.iterations, test_mode=True)
        return padder.unpad(disparity.float()).squeeze().cpu().numpy()
