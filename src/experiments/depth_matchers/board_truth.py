"""Ground truth for depth on the board: around every tag, the plane of that tag's triangulated corners.

Each tag is printed flat, and the board around it is flat over a few cm. So for every tag found in both cameras
(tag_poses.triangulate_frame, the same detection and refinement the pipeline poses frames with), the plane of its
four stereo-triangulated corners is the truth over the tag and its white surround, out to SURROUND times the
tag's size: the scoring REGION. In it the disparity every pixel should have follows from the plane, and a
matcher's disparity is scored against that.

One plane per tag rather than one through all of them: the tags of a frame are not coplanar enough. Fitted
together they leave about 1 mm RMS (3DPrintFinal s30, Ice pyr1; one tag 3.9 mm off), which at 400 mm is a whole
pixel of disparity, the size of the tolerance. A single tag's four corners fit their own plane to 0.33 mm
(median; 90th percentile 0.58 mm, 139 tags of s30 and pyr1 pass3). That corner noise is the reference's own
error: about 0.3 px of disparity at 400 mm, less further away. The learned matchers' median error on s30 (0.46 px)
is close to it, so differences below about 0.3 px are not resolved.

Scoring in disparity (px) rather than depth (mm) makes one tolerance mean the same matching error at every
distance: 1 px is about 1.1 mm of depth at 400 mm and 3.5 mm at 750 mm on this rig.

Region pixels whose match would fall left of the right image's edge are left out: no matcher can find them.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "03_Reconstruction"))
import tag_poses  # noqa: E402

SURROUND = 1.6          # region per tag: the tag scaled about its centre by this (the white border around it)
MAX_PLANE_RMS_MM = 0.5  # a tag whose corners sit further than this from their plane is a bad detection: skipped
                        # (0.3 dropped over half the tags; 0.5 keeps all but the worst ~15%)
MAX_REPROJECTION = 0.8  # px, tag_poses.py's default for accepting a triangulated tag


def fit_plane(points):
    """(unit normal n, offset c) of the least-squares plane n.X = c, and the RMS distance of the points to it."""
    centre = points.mean(0)
    _, _, vt = np.linalg.svd(points - centre)
    normal = vt[2]
    if normal[2] > 0:   # face the camera
        normal = -normal
    distances = (points - centre) @ normal
    return normal, float(normal @ centre), float(np.sqrt(np.mean(distances ** 2)))


def plane_disparity(rectifier, normal, offset, u, v):
    """The disparity Q maps to the depth of plane n.X = c along the rays of rectified-left pixels (u, v)."""
    P1, Q = rectifier.rig.P1, rectifier.rig.Q
    f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    rays_z = normal[0] * (u - cx) / f + normal[1] * (v - cy) / f + normal[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = offset / rays_z
        return np.where(z > 0, (Q[2, 3] / z - Q[3, 3]) / Q[3, 2], np.nan)


def frame_truth(rectifier, detector, left_path, right_path):
    """Board ground truth of one frame, or None if no usable tag was found in both cameras.

    Returns {"region": bool mask (rectified left image), "disparity": float32 plane disparity in the region (0
    elsewhere), "tags": ids used, "plane_rms_mm": the worst tag's corner RMS}.
    """
    tags, _, _ = tag_poses.triangulate_frame(rectifier.rig, detector, left_path, right_path, MAX_REPROJECTION)
    P1, R1 = rectifier.rig.P1, rectifier.rig.R1
    w, h = rectifier.rig.image_size
    truth = np.zeros((h, w), np.float32)
    used, worst = [], 0.0
    for tag_id, X in sorted(tags.items()):
        X = (R1 @ X.T).T                                            # left camera -> rectified left frame
        normal, offset, rms = fit_plane(X)
        if rms > MAX_PLANE_RMS_MM:
            continue
        centre = X.mean(0)
        uv = (P1[:, :3] @ (centre + SURROUND * (X - centre)).T).T
        polygon = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(polygon, np.round(uv[:, :2] / uv[:, 2:]).astype(np.int32), 1)
        v, u = np.nonzero(polygon)
        d = plane_disparity(rectifier, normal, offset, u.astype(np.float64), v.astype(np.float64))
        # Leave out pixels whose match would lie off the right image: no matcher can find them.
        ok = np.isfinite(d) & (d > 0) & (d < u)
        truth[v[ok], u[ok]] = d[ok]
        used.append(tag_id)
        worst = max(worst, rms)
    if not used:
        return None
    return {"region": truth > 0, "disparity": truth, "tags": used, "plane_rms_mm": worst}


def score(disparity, truth, tolerance=1.0):
    """Counts for one frame: region pixels, pixels with depth there, and those within tolerance px of the plane.

    disparity: the matcher's (px, 0 where it kept no depth), as depth_frames.py saves it.
    """
    region = truth["region"]
    has = region & (disparity > 0)
    error = np.abs(disparity - truth["disparity"])
    good = has & (error <= tolerance)
    return {"region": int(region.sum()), "with_depth": int(has.sum()), "good": int(good.sum()),
            "abs_error_px": error[has]}


def summarise(counts, tolerance=1.0):
    """Precision, recall and F1 over frames (pixel-weighted), and the median and 90th percentile error (px)."""
    region = sum(c["region"] for c in counts)
    with_depth = sum(c["with_depth"] for c in counts)
    good = sum(c["good"] for c in counts)
    errors = np.concatenate([c["abs_error_px"] for c in counts]) if counts else np.empty(0)
    precision = good / with_depth if with_depth else 0.0
    recall = good / region if region else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"frames": len(counts), "tolerance_px": tolerance, "precision": precision, "recall": recall, "f1": f1,
            "median_error_px": float(np.median(errors)) if len(errors) else None,
            "p90_error_px": float(np.percentile(errors, 90)) if len(errors) else None}
