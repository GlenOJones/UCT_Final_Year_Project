"""Extract every stereo pair of a scan recording in which the whole scan target is in view.

    src/venv/bin/python src/preproccessing/select_scan_frames.py --session Oct1/Testing --scan pyr1
    src/venv/bin/python src/preproccessing/select_scan_frames.py --session Oct1/Testing --scan all --dry-run
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import project_paths as paths  # noqa: E402
import extract_stereo_frames as ex  # noqa: E402

# ====== SCAN TARGETS ======
# Per scan: reference tag ids, a head-on reference pair and the block's corners in that right image (px).
SCANS = {
    "cub2":  dict(tags=(2, 4, 5), ref_frame=2181, corners=[(995, 60), (1382, 88), (1262, 842), (925, 800)]),
    "hang1": dict(tags=(2, 3, 7), ref_frame=1573, corners=[(325, 405), (1020, 475), (940, 758), (270, 745)]),
    "pyr1":  dict(tags=(2, 3, 7), ref_frame=923,  corners=[(275, 570), (862, 558), (864, 830), (280, 858)]),
    "wave1": dict(tags=(2, 3, 7), ref_frame=2827, corners=[(364, 548), (1080, 564), (1070, 904), (370, 882)]),
}

MARGIN_PX = 15

# Same dictionary as tag_poses.py; default detector parameters, since only presence matters here.
DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11

# Worker state, set once per process by init_worker().
_rec = _det = None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Extract the stereo pairs that show all tags and both faces.")
    ap.add_argument("--session", required=True, help="recording session, e.g. Oct1/Testing")
    ap.add_argument("--scan", required=True, help=f"scan name, or 'all' for {', '.join(SCANS)}")
    ap.add_argument("--calibration", default=os.path.join(paths.RESULTS_DIR, "30Sep/calibration/30Sep_wide_stereo.yaml"),
                    help="stereo calibration YAML, used only to undistort (default: 30Sep wide)")
    ap.add_argument("--color", action="store_true", help="save colour instead of grayscale luma")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing frames/ folder")
    ap.add_argument("--dry-run", action="store_true", help="report the selection, write nothing")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    return ap.parse_args(argv)


def load_intrinsics(path):
    """{camera: (K, D)} from a Calibration.py YAML."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        sys.exit(f"cannot read calibration {path}")
    cams = {c: (fs.getNode(f"{c}_camera").getNode("camera_matrix").mat(),
                fs.getNode(f"{c}_camera").getNode("distortion_coefficients").mat()) for c in ex.CAMS}
    fs.release()
    return cams


def undistort(points, K, D):
    """Pixel points -> normalised undistorted image coordinates, (N, 2)."""
    return cv2.undistortPoints(np.asarray(points, np.float64).reshape(-1, 1, 2), K, D).reshape(-1, 2)


def valid_region(K, D, size, margin):
    """The image area inset by margin px, as a polygon in normalised undistorted coordinates."""
    w, h = size
    n = 50
    x0, y0, x1, y1 = margin, margin, w - 1 - margin, h - 1 - margin
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    border = np.concatenate([np.c_[xs, np.full(n, y0)], np.c_[np.full(n, x1), ys],
                             np.c_[xs[::-1], np.full(n, y1)], np.c_[np.full(n, x0), ys[::-1]]])
    return undistort(border, K, D).astype(np.float32)


def init_worker(folder):
    global _rec, _det
    cv2.setNumThreads(1)
    _rec = ex.load_recording(folder)
    _det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), cv2.aruco.DetectorParameters())


def detect_pair(pair):
    """(pair, {camera: {tag id: 4x2 corners px}}) for one (right, left, dt) pair."""
    found = {}
    for cam, i in (("right", pair[0]), ("left", pair[1])):
        corners, ids, _ = _det.detectMarkers(ex.jpeg(_rec[cam], i, cv2.IMREAD_GRAYSCALE))
        found[cam] = {} if ids is None else {int(t): c.reshape(4, 2) for t, c in zip(ids.ravel(), corners)}
    return pair, found


def stacked(tags, detections):
    return np.concatenate([detections[t] for t in tags])


def block_in_view(target, ref_tags_n, detections, intr, regions):
    """Why the pair fails, or None if every tag is found and the block is inside both images."""
    tags = target["tags"]
    for cam in ex.CAMS:
        missing = [t for t in tags if t not in detections[cam]]
        if missing:
            return "tags"
        K, D = intr[cam]
        H, _ = cv2.findHomography(ref_tags_n, undistort(stacked(tags, detections[cam]), K, D))
        if H is None:
            return "tags"
        block = cv2.perspectiveTransform(target["block_n"].reshape(-1, 1, 2), H).reshape(-1, 2)
        if any(cv2.pointPolygonTest(regions[cam], (float(x), float(y)), False) < 0 for x, y in block):
            return "block"
    return None


def select(scan, folder, args, intr):
    target = SCANS[scan]
    rec = ex.load_recording(folder)
    size = (rec["right"]["w"], rec["right"]["h"])
    regions = {cam: valid_region(*intr[cam], size, MARGIN_PX) for cam in ex.CAMS}
    pairs = ex.pair_up(rec, range(len(rec["right"]["pts"])), ex.MAX_DT_MS)

    with Pool(args.workers, initializer=init_worker, initargs=(folder,)) as pool:
        results = pool.map(detect_pair, pairs, chunksize=8)

    ref = next((d for p, d in results if p[0] == target["ref_frame"]), None)
    if ref is None or any(t not in ref["right"] for t in target["tags"]):
        sys.exit(f"{scan}: reference frame {target['ref_frame']} does not show every tag {target['tags']}")
    K, D = intr["right"]
    ref_tags_n = undistort(stacked(target["tags"], ref["right"]), K, D)
    target = dict(target, block_n=undistort(target["corners"], K, D))

    kept, why = [], {"tags": 0, "block": 0}
    for pair, detections in results:
        reason = block_in_view(target, ref_tags_n, detections, intr, regions)
        if reason:
            why[reason] += 1
        else:
            kept.append(pair)
    print(f"{scan}: {len(pairs)} pairs, kept {len(kept)}  "
          f"(missing a tag: {why['tags']}, block not fully in view: {why['block']})")
    return rec, kept


def main(argv=None):
    args = parse_args(argv)
    scans = list(SCANS) if args.scan == "all" else [args.scan]
    unknown = [s for s in scans if s not in SCANS]
    if unknown:
        sys.exit(f"no target layout for {unknown}; add it to SCANS")
    intr = load_intrinsics(args.calibration)

    for scan in scans:
        folder = Path(paths.recording_dir(args.session, scan))
        out = folder / "frames"
        if not args.dry_run and out.exists() and not args.overwrite:
            sys.exit(f"{paths.rel(out)} exists; pass --overwrite to replace it")
        rec, kept = select(scan, folder, args, intr)
        if not args.dry_run:
            ex.write_pairs(rec, kept, out, gray=not args.color)


if __name__ == "__main__":
    main()
