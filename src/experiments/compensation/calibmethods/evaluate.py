"""Fit every technique on one split's training recordings and score it on the held-out recordings."""
import cv2
import numpy as np
from scipy.spatial.distance import pdist

from . import data, metrics
from .models import triangulate

POINT_COLUMNS = ("technique", "camera", "frame", "u", "v", "du", "dv", "r", "depth_mm", "frame_distance_mm")
BAR_COLUMNS = ("technique", "pair", "depth_mm", "frame_distance_mm", "bias_mm", "rms_mm", "bias_pct", "n_bars")


def score_frames(model, frames, frame_index, image_size):
    """Residual rows for held-out frames; frames whose pose fails are counted and skipped."""
    rows, failed = [], 0
    for f in frames:
        try:
            rvec, tvec = model.pose(f.obj, f.img)
        except (RuntimeError, ValueError):
            failed += 1
            continue
        proj = model.project(f.obj, rvec, tvec)
        d = f.img - proj
        R = cv2.Rodrigues(np.asarray(rvec, float))[0]
        depth = (f.obj @ R.T + np.asarray(tvec, float))[:, 2]
        r = metrics.normalised_radius(f.img[:, 0], f.img[:, 1], model.principal_point, image_size)
        n = len(f.img)
        rows.append(np.column_stack([np.full(n, frame_index[f.name]), f.img, d, r, depth,
                                     np.full(n, f.distance_mm)]))
    return (np.concatenate(rows) if rows else np.empty((0, 8))), failed


def scale_bars(left, right, R, T, pairs, pair_index, band_mm):
    """One row per held-out pair: length error of the corner-to-corner bars with true length in band_mm."""
    lo, hi = band_mm
    rows = []
    for lf, rf in pairs:
        X = triangulate(left, right, R, T, lf.img, rf.img)
        true = pdist(lf.obj)
        sel = (true >= lo) & (true <= hi)
        if not sel.any():
            continue
        err = pdist(X)[sel] - true[sel]
        rows.append([pair_index[lf.name], float(X[:, 2].mean()), lf.distance_mm, float(err.mean()),
                     float(np.sqrt(np.mean(err ** 2))), float(100 * np.mean(err / true[sel])), int(sel.sum())])
    return np.array(rows, float).reshape(-1, 7)


def run_split(run, split, techniques, cfg, keep_models=False):
    """Everything one split contributes: {"params", "points", "bars", "notes"}, plus "models" (the fitted cameras and
    stereo extrinsics per technique) when keep_models."""
    image_size = run.image_size
    frame_index = {c: {f.name: k for k, f in enumerate(run.frames[c])} for c in data.CAMERAS}
    all_pairs = run.stereo_pairs()
    pair_index = {lf.name: k for k, (lf, _) in enumerate(all_pairs)}
    test_pairs = [p for p in all_pairs if p[0].group in split.test]

    params, point_rows, bar_rows, notes, kept = {}, [], [], [], {}
    for t_idx, tech in enumerate(techniques):
        fitted, entry = {}, {}
        for c_idx, camera in enumerate(data.CAMERAS):
            train = tech.training_frames(run.frames[camera], split.train)
            model = tech.new_model().fit(train, image_size, camera)
            fitted[camera] = model
            entry[camera] = {**model.to_dict(), "training_frames": [f.name for f in train]}

            test = [f for f in run.frames[camera] if f.group in split.test]
            rows, failed = score_frames(model, test, frame_index[camera], image_size)
            if failed:
                notes.append(f"split {split.index} {tech.key} {camera}: {failed} held-out frames could not be posed")
            point_rows.append(np.column_stack([np.full(len(rows), t_idx), np.full(len(rows), c_idx), rows]))

        train_pairs = tech.training_pairs(all_pairs, split.train)
        R, T, rms = fitted["left"].stereo(train_pairs, fitted["right"], image_size)
        if keep_models:
            kept[tech.key] = {"left": fitted["left"], "right": fitted["right"], "R": np.asarray(R), "T": np.ravel(T)}
        entry["stereo"] = {"R": R.tolist(), "T_mm": list(map(float, T)), "baseline_mm": float(np.linalg.norm(T)),
                           "rms_px": rms, "training_pairs": len(train_pairs),
                           "opencv_function": "cv2.stereoCalibrate" if tech.model == "brown_conrady"
                           else "in-air stereo calibration, unchanged" if tech.model == "air_direct"
                           else "stereo_lm (see models.py)"}
        bars = scale_bars(fitted["left"], fitted["right"], R, T, test_pairs, pair_index, cfg["scale_bar_length_mm"])
        bar_rows.append(np.column_stack([np.full(len(bars), t_idx), bars]))
        params[tech.key] = entry

    out = {"split": split.index, "train": split.train, "test": split.test, "params": params,
           "points": np.concatenate(point_rows) if point_rows else np.empty((0, 10)),
           "bars": np.concatenate(bar_rows) if bar_rows else np.empty((0, 8)), "notes": notes}
    if keep_models:
        out["models"] = kept
    return out
