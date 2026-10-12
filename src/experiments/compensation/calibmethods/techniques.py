"""A technique is a camera model plus the rule for which training frames it may use. The three compared:

    bc        Brown-Conrady on every training frame (the baseline; Zhang's method)
    kb        Kannala-Brandt on every training frame (wide-angle model)
    bc_close  Brown-Conrady on the close-range training frames only, scored at every distance. If the
              compensated optic behaves as a single-viewpoint (pinhole) camera, a model fitted close
              should hold far away; refraction that the model cannot absorb shows up as error that
              grows with distance.
    pinax     the in-air calibration plus flat-interface refraction (models.Pinax); it uses no underwater
              training frames for the intrinsics, only for the stereo extrinsics like the others.

They are defined in config.json, so a band or a model can change without touching the code.
"""
from dataclasses import dataclass

from . import models


@dataclass(frozen=True)
class Technique:
    key: str
    label: str
    model: str                            # key into models.MODELS
    train_distance_mm: tuple = None       # (min, max) board distance of training frames; None = all
    options: tuple = ()                   # extra model settings, as (name, value) pairs

    def new_model(self):
        return models.MODELS[self.model](**dict(self.options))

    def training_frames(self, frames, groups):
        """The frames of the given recordings this technique may train on."""
        out = [f for f in frames if f.group in groups]
        if self.train_distance_mm is not None:
            lo, hi = self.train_distance_mm
            out = [f for f in out if lo <= f.distance_mm < hi]
        return out

    def training_pairs(self, pairs, groups):
        if self.train_distance_mm is None:
            return [p for p in pairs if p[0].group in groups]
        lo, hi = self.train_distance_mm
        return [p for p in pairs if p[0].group in groups and lo <= p[0].distance_mm < hi]


def from_config(entries, resolve_path=lambda p: p):
    """Techniques from config.json; `resolve_path` turns a project-relative path option into a usable one."""
    out = []
    for e in entries:
        options = dict(e.get("options", {}))
        if "air_calibration" in options:
            options["air_calibration"] = resolve_path(options["air_calibration"])
        out.append(Technique(e["key"], e["label"], e["model"],
                             tuple(e["train_distance_mm"]) if e.get("train_distance_mm") else None,
                             tuple(sorted(options.items()))))
    return out
