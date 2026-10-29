"""Measurement of the separate WaterComparison target with every technique's stereo calibration."""
import csv
import os
import sys

import numpy as np

from .models import triangulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import single_viewpoint  # noqa: E402  (the detector: one definition, shared with the single-viewpoint test)


def detect(target_cfg, detection_cfg, project_path):
    """[(label_cm, {id: left 4x2 px}, {id: right 4x2 px})] for every pair in the target's pairs.csv."""
    frames, ids = project_path(target_cfg["frames"]), set(target_cfg["ids"])
    detector = single_viewpoint.make_detector(detection_cfg, target_cfg["border_bits"])
    with open(os.path.join(frames, "pairs.csv"), newline="") as fh:
        pairs = sorted(csv.DictReader(fh), key=lambda r: float(r["label"]))
    return [(float(p["label"]),
             single_viewpoint.detect_target(detector, detection_cfg, os.path.join(frames, "left", p["left_file"]), ids),
             single_viewpoint.detect_target(detector, detection_cfg, os.path.join(frames, "right", p["right_file"]), ids))
            for p in pairs]


def measure(fitted, detections, size_mm):
    """Rows {label_cm, id, distance_mm, edges_mm, error_mm} per tag seen in both cameras, for one technique."""
    rows = []
    for label, L, R in detections:
        for t in sorted(set(L) & set(R)):
            X = triangulate(fitted["left"], fitted["right"], fitted["R"], fitted["T"], L[t], R[t])
            edges = np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1)
            rows.append({"label_cm": label, "id": t, "distance_mm": float(np.linalg.norm(X.mean(axis=0))),
                         "edges_mm": edges.tolist(), "error_mm": float(np.median(edges) - size_mm)})
    return rows


def tag_stats(rows, size_mm):
    """Tags, median and mean tag error (mm), percent, RMS over all edges (mm) and mean distance of the rows."""
    if not rows:
        return {"n_tags": 0}
    e = np.array([r["error_mm"] for r in rows])
    edges = np.concatenate([r["edges_mm"] for r in rows]) - size_mm
    return {"n_tags": len(rows), "median_error_mm": float(np.median(e)), "mean_error_mm": float(e.mean()),
            "median_error_pct": float(100 * np.median(e) / size_mm), "edge_rms_mm": float(np.sqrt(np.mean(edges ** 2))),
            "distance_mm": float(np.mean([r["distance_mm"] for r in rows]))}


def summary(rows, size_mm):
    """{"all": stats, "<label>": stats ...} for one technique's rows."""
    labels = sorted({r["label_cm"] for r in rows})
    return {"all": tag_stats(rows, size_mm),
            **{f"{lab:g}": tag_stats([r for r in rows if r["label_cm"] == lab], size_mm) for lab in labels}}
