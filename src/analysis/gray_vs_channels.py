"""Compare grayscale sources for AprilGrid calibration: recorded luma Y against the R, G and B channels.

Question: does calibrating from the colour channels (R, G or B) detect more tags or give a better fit than
the recorded luma plane Y that extract_stereo_frames.py --gray saves?

Conditions, all derived from the same raw YUYV frames:
    Y       recorded luma plane, untouched                         reference (what Calibration.py uses)
    R, G, B channels of the colour frame (ffmpeg, BT.601)          tested
    Gray    cv2.cvtColor(colour, BGR2GRAY) = 0.299R + 0.587G + 0.114B   reference for a "computed gray"

Every Rigframes pair is traced back to its recording through pairs.csv (frame indices; the source column, or
for older files a match on the timestamps), the colour frame is decoded again, and the same detector and
board model as Calibration.py run on each condition. Corners are refined on that condition's own image.

Caveat: the camera records YUYV 4:2:2, so Cb/Cr have half the horizontal resolution of Y. R and B (and to a
lesser extent G) are rebuilt from interpolated chroma, so they carry less horizontal detail than Y.

Reported: detection counts over all views, then, on the views usable in EVERY condition (a paired
comparison), the mono calibration of each camera and how far each condition's corners sit from Y's.

Usage (project root):
    src/venv/bin/python src/analysis/gray_vs_channels.py data/Calibration/18_Sep/Rigframes
Outputs (default results/gray_vs_channels/): results.json, fig_gray_vs_channels.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preproccessing"))
import extract_stereo_frames as esf  # noqa: E402

CONDS = ["Y", "R", "G", "B", "Gray"]
MIN_TAGS = 10

# ---- detector and board: keep identical to src/01_Calibration/Calibration.py ----
TAG_SIZE = 35.36
PITCH = TAG_SIZE * 1.3
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.markerBorderBits = 2
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax, params.adaptiveThreshWinSizeStep = 3, 53, 5
params.errorCorrectionRate = 1.0
detector = cv2.aruco.ArucoDetector(dictionary, params)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
tag_corners = []
for tag_id in range(70):
    x, y = (tag_id % 10) * PITCH, (tag_id // 10) * PITCH
    tag_corners.append(np.array([[x + TAG_SIZE, y, 0], [x, y, 0], [x, y + TAG_SIZE, 0], [x + TAG_SIZE, y + TAG_SIZE, 0]],
                                np.float32))
board = cv2.aruco.Board(tag_corners, dictionary, np.arange(70))


def detect(gray):
    """(corners, ids) with cornerSubPix refinement on this image, or None if fewer than MIN_TAGS tags."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < MIN_TAGS:
        return None, 0 if ids is None else len(ids)
    corners = tuple(cv2.cornerSubPix(gray, c.reshape(4, 1, 2).copy(), (5, 5), (-1, -1), SUBPIX_CRITERIA)
                    .reshape(1, 4, 2) for c in corners)
    return (corners, ids), len(ids)


def conditions(bgr, y):
    b, g, r = cv2.split(bgr)
    return {"Y": y, "R": r, "G": g, "B": b, "Gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)}


# ---------------------------------------------------------------- locating the frames
def infer_source(rows, root):
    """Recording folder whose right timestamps contain the first pair's right_pts_ns at its right_frame."""
    r0 = rows[0]
    for cand in sorted(root.glob("*/right_timestamps.csv")):
        with open(cand, newline="") as fh:
            pts = [int(x["pts_ns"]) for x in csv.DictReader(fh)]
        if int(r0["right_frame"]) < len(pts) and pts[int(r0["right_frame"])] == int(r0["right_pts_ns"]):
            return cand.parent
    sys.exit(f"cannot find the source recording for {rows[0]['right_file']}; add a source column to its pairs.csv")


def load_pairs(frames_dir):
    """[(subfolder, source recording, row)] for every pair under frames_dir/*/pairs.csv."""
    out = []
    for pc in sorted(frames_dir.glob("*/pairs.csv")):
        with open(pc, newline="") as fh:
            rows = list(csv.DictReader(fh))
        src = Path(rows[0]["source"]) if rows[0].get("source") else infer_source(rows, frames_dir.parent)
        out += [(pc.parent.name, src, r) for r in rows]
    return out


def scan(pairs):
    """Detect in every condition. Returns det[cond][cam][view] = (corners, ids) and counts[cond][cam][view] = n_tags.

    View names are 'subfolder/filename.png', as in Calibration.py. Y is checked against the saved PNG so a wrong
    frame mapping cannot go unnoticed.
    """
    det = {c: {cam: {} for cam in esf.CAMS} for c in CONDS}
    counts = {c: {cam: {} for cam in esf.CAMS} for c in CONDS}
    by_src = {}
    for sub, src, r in pairs:
        by_src.setdefault(src, []).append((sub, r))
    for src, items in by_src.items():
        rec = esf.load_recording(src)
        print(f"{src.name}: {len(items)} pairs")
        for cam, frame_key, file_key in (("right", "right_frame", "right_file"), ("left", "left_frame", "left_file")):
            wanted = {int(r[frame_key]): (sub, r[file_key]) for sub, r in items}
            for i, bgr in esf.ffmpeg_frames(rec[cam], wanted):
                sub, fname = wanted[i]
                y = np.ascontiguousarray(esf.yuyv(rec[cam], i)[..., 0])
                saved = cv2.imread(str(Path(frames_dir_global) / sub / cam / fname), cv2.IMREAD_GRAYSCALE)
                if saved is None or not np.array_equal(saved, y):
                    sys.exit(f"{sub}/{fname}: saved PNG differs from luma of {src.name} frame {i}")
                for c, g in conditions(bgr, y).items():
                    d, n = detect(g)
                    counts[c][cam][f"{sub}/{fname}"] = n
                    if d is not None:
                        det[c][cam][f"{sub}/{fname}"] = d
    return det, counts


# ---------------------------------------------------------------- calibration
def views(cam_det, names):
    obj, img = [], []
    for n in names:
        o, p = board.matchImagePoints(*cam_det[n])
        obj.append(o), img.append(p)
    return obj, img


def calibrate(cam_det, names, size):
    obj, img = views(cam_det, names)
    rms, K, D, _, _, std, _, per_view = cv2.calibrateCameraExtended(obj, img, size, None, None)
    e = per_view.ravel()
    return dict(rms=float(rms), median_view_err=float(np.median(e)), fx=float(K[0, 0]), fy=float(K[1, 1]),
                cx=float(K[0, 2]), cy=float(K[1, 2]), k1=float(D.ravel()[0]), k2=float(D.ravel()[1]),
                fx_sd=float(std.ravel()[0]), n_views=len(names), n_tags=int(sum(len(cam_det[n][1]) for n in names)))


def displacement(cam_det_c, cam_det_y, names):
    """Distance (px) between a condition's corners and Y's for every tag corner seen in both, over the views."""
    d = []
    for n in names:
        (cc, ci), (yc, yi) = cam_det_c[n], cam_det_y[n]
        ys = {int(k): c.reshape(4, 2) for k, c in zip(yi.ravel(), yc)}
        for k, c in zip(ci.ravel(), cc):
            if int(k) in ys:
                d.append(np.linalg.norm(c.reshape(4, 2) - ys[int(k)], axis=1))
    return np.concatenate(d) if d else np.zeros(1)


# ---------------------------------------------------------------- main
def main():
    global frames_dir_global
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", type=Path, help="folder of extracted pairs, e.g. data/Calibration/18_Sep/Rigframes")
    ap.add_argument("--out", type=Path, default=Path("results/gray_vs_channels"))
    args = ap.parse_args()
    frames_dir_global = args.frames_dir
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.frames_dir)
    det, counts = scan(pairs)
    first = cv2.imread(str(next(args.frames_dir.glob("*/left/*.png"))), cv2.IMREAD_GRAYSCALE)
    size = first.shape[::-1]
    total = {cam: len(counts["Y"][cam]) for cam in esf.CAMS}

    res = {"detection": {}, "paired": {}, "displacement_vs_Y": {}, "n_views_total": total}
    print(f"\nDetection over all views (usable = {MIN_TAGS}+ tags)")
    print(f"{'':6s}" + "".join(f"{cam + ' usable':>14s}{cam + ' tags':>12s}" for cam in esf.CAMS))
    for c in CONDS:
        row = {cam: dict(usable=len(det[c][cam]), tags=int(sum(counts[c][cam].values()))) for cam in esf.CAMS}
        res["detection"][c] = row
        print(f"{c:6s}" + "".join(f"{row[cam]['usable']:>10d}/{total[cam]:<3d}{row[cam]['tags']:>12d}" for cam in esf.CAMS))

    for cam in esf.CAMS:
        names = sorted(set.intersection(*(set(det[c][cam]) for c in CONDS)))
        print(f"\n{cam.upper()}: {len(names)} views usable in every condition (paired comparison)")
        print(f"{'':6s}{'RMS px':>8s}{'median px':>11s}{'fx':>8s}{'fx sd':>7s}{'cx':>8s}{'cy':>8s}{'k1':>8s}"
              f"{'tags':>7s}{'corner shift vs Y (median / p95 px)':>38s}")
        res["paired"][cam], res["displacement_vs_Y"][cam] = {}, {}
        for c in CONDS:
            cal = calibrate(det[c][cam], names, size)
            d = displacement(det[c][cam], det["Y"][cam], names)
            cal["shift_median"], cal["shift_p95"] = float(np.median(d)), float(np.percentile(d, 95))
            res["paired"][cam][c] = cal
            print(f"{c:6s}{cal['rms']:8.3f}{cal['median_view_err']:11.3f}{cal['fx']:8.1f}{cal['fx_sd']:7.1f}"
                  f"{cal['cx']:8.1f}{cal['cy']:8.1f}{cal['k1']:8.4f}{cal['n_tags']:7d}"
                  f"{cal['shift_median']:>22.3f} / {cal['shift_p95']:.3f}")
    (args.out / "results.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    x, w = np.arange(len(CONDS)), 0.38
    for k, cam in enumerate(esf.CAMS):
        ax[0].bar(x + (k - .5) * w, [res["detection"][c][cam]["usable"] for c in CONDS], w, label=cam)
        ax[1].bar(x + (k - .5) * w, [res["paired"][cam][c]["rms"] for c in CONDS], w, label=cam)
        ax[2].bar(x + (k - .5) * w, [res["paired"][cam][c]["median_view_err"] for c in CONDS], w, label=cam)
    for a, t in zip(ax, ["usable images (10+ tags)", "paired RMS reprojection (px)", "paired median per-view error (px)"]):
        a.set_xticks(x, CONDS), a.set_title(t, fontsize=9, loc="left")
    ax[0].legend()
    fig.savefig(args.out / "fig_gray_vs_channels.png", dpi=160)
    print(f"\nwrote {args.out}/results.json and fig_gray_vs_channels.png")


if __name__ == "__main__":
    main()
