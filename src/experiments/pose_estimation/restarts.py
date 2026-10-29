"""Restarting a pose estimator later in a pass when the run from the first frame fails.

    src/venv/bin/python -m unittest discover -s src/experiments/pose_estimation/tests -v
"""
import json
import os
import shutil

import numpy as np


def frame_number(name):
    """0123.png / 0123.npz -> 123."""
    return int(os.path.splitext(os.path.basename(name))[0])


def assess(names, poses, diagnostics, criteria):
    """Whether a trajectory failed."""
    posed = [n for n in names if n in poses and not diagnostics.get(n, {}).get("lost", 0)]
    fraction = len(posed) / len(names) if names else 0.0
    centres = np.array([np.asarray(poses[n])[:3, 3] for n in posed]).reshape(-1, 3)
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1) if len(centres) > 1 else np.zeros(0)
    jumps = int((steps > criteria["max_jump_mm"]).sum())
    jump_fraction = jumps / len(steps) if len(steps) else 0.0
    reasons = []
    if fraction < criteria["min_posed_fraction"]:
        reasons.append(f"posed {len(posed)}/{len(names)} < {criteria['min_posed_fraction']:.0%}")
    if jump_fraction > criteria["max_jump_fraction"]:
        reasons.append(f"{jumps} jumps > {criteria['max_jump_mm']:g} mm ({jump_fraction:.0%} of steps)")
    return {"frames": len(names), "posed": len(posed), "posed_fraction": round(fraction, 3), "jumps": jumps,
            "jump_fraction": round(jump_fraction, 3), "failed": bool(reasons), "reasons": reasons}


def candidate_starts(n_frames, start_step, max_start_fraction):
    """Later start frames to try, in order: start_step, 2*start_step, ..."""
    last = int(n_frames * max_start_fraction)
    return list(range(start_step, last + 1, start_step))


def link_subset(source_dir, target_dir, names, subfolders, copy_files=()):
    """target_dir/<sub>/<name> -> source_dir/<sub>/<name> (symlinks) for every name and subfolder, plus copies of
    copy_files (e.g. depth_frames.json)."""
    if os.path.lexists(target_dir):
        shutil.rmtree(target_dir)
    for sub in subfolders:
        os.makedirs(os.path.join(target_dir, sub))
        for name in names:
            src = os.path.join(source_dir, sub, name)
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            os.symlink(os.path.abspath(src), os.path.join(target_dir, sub, name))
    for f in copy_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, f))
    return target_dir


def run_with_restarts(names, run_from, read, criteria):
    """Try the run from frame 0, then from later starts until one passes."""
    attempts = []
    for start in [0] + candidate_starts(len(names), criteria["start_step"], criteria["max_start_fraction"]):
        subset = [n for n in names if frame_number(n) >= start]
        path = run_from(start)
        poses, diagnostics = read(path)
        verdict = assess(subset, poses, diagnostics, criteria)
        attempts.append({"start": start, "path": path, **verdict})
        if not verdict["failed"]:
            return attempts, (attempts[-1] if start else None)
    return attempts, None


def write_attempts(path, estimator, attempts, kept):
    """The restart record beside the poses: what was tried and what was kept."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"estimator": estimator, "kept_start": kept["start"] if kept else None,
                   "attempts": attempts}, fh, indent=1, default=str)
