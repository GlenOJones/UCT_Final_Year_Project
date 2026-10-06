"""Score every object on a multi-object board against its own CAD model.

compare_to_cad.py takes the tallest object in a cloud, so a board carrying several objects is scored
one object at a time: the cloud is cropped to a box around each object (board frame, mm) and the crop
compared with that object's STL. The crops are saved, so the comparison can be re-run or opened in
CloudCompare on its own.

    src/venv/bin/python src/04_Comparison/score_objects.py --session Sep30 --scans s20 s30 merged --methods raft_s075 raft_s075_weighted

Every (scan, method) folder that has a cloud_clean.ply is scored. Writes, per object:
    results/<session>/<scan>/<method>/objects/<object>.ply          the crop
    results/<session>/<scan>/comparison/<method>/<object>/          compare_to_cad.py's outputs
and one table of every scan, method, object and fit: results/<session>/objects_summary.csv.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import project_paths as paths  # noqa: E402

STL_DIR = os.path.join(paths.DATA_DIR, "GroundTruth", "roughness_blocks_STL")

# Sep30 board: object centres in the board frame (mm) and their CAD models. Every scan's board frame
# comes from the same five tags, so these hold for every pass to within a few mm (the crop is
# generous). An object with no STL yet is listed with None and skipped. Identified from the COLMAP
# height map of s30: four apexes on crossed ridges at (-20, 120), a grid of egg-cup bumps at (-210, 10).
OBJECTS = {
    "PYRAMID_1": ((195.0, 0.0), "PYRAMID_1.stl"),
    "PYRAMID_2X2": ((-20.0, 120.0), "PYRAMID_2X2.stl"),
    "EGG_1PK": ((-20.0, -100.0), "EGG_1PK.stl"),
    "EGG_2PKS": ((-210.0, 10.0), None),
}
CROP_HALF = 85.0    # mm, half-width of the crop box: a 100 mm object plus room for frame differences
FITS = ("on_board", "on_local_board", "shape")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Crop each object of a multi-object board and compare it with its CAD model.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="method folders; missing ones are skipped")
    ap.add_argument("--objects", nargs="+", default=list(OBJECTS), choices=list(OBJECTS))
    ap.add_argument("--jobs", type=int, default=8, help="comparisons run at once (default 8)")
    return ap.parse_args(argv)


def crop(cloud_path, centre, out_path):
    cloud = o3d.io.read_point_cloud(cloud_path)
    xyz = np.asarray(cloud.points)
    inside = np.all(np.abs(xyz[:, :2] - np.asarray(centre)) < CROP_HALF, axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_point_cloud(out_path, cloud.select_by_index(np.flatnonzero(inside)))
    return int(inside.sum())


def compare(job):
    session, scan, method, name, crop_path, stl = job
    out_dir = os.path.join(paths.comparison_dir(session, scan, method), name)   # <out-dir>/<crop name>
    script = os.path.join(paths.PROJECT_ROOT, "src", "04_Comparison", "compare_to_cad.py")
    result = subprocess.run([sys.executable, script, crop_path, "--out-dir", os.path.dirname(out_dir),
                             "--reference", os.path.join(STL_DIR, stl), "--no-summary"], capture_output=True, text=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    if result.returncode != 0 or not os.path.exists(metrics_path):
        print(f"  {scan}/{method}/{name}: FAILED\n{result.stderr[-800:]}")
        return job, None
    print(f"  {scan}/{method}/{name}: done")
    with open(metrics_path) as f:
        return job, json.load(f)


def main():
    args = parse_args()
    jobs = []
    for scan in args.scans:
        for method in args.methods:
            cloud_path = os.path.join(paths.method_dir(args.session, scan, method), paths.CLOUD_CLEAN)
            if not os.path.exists(cloud_path):
                continue
            for name in args.objects:
                centre, stl = OBJECTS[name]
                if stl is None:
                    continue
                crop_path = os.path.join(paths.method_dir(args.session, scan, method), "objects", f"{name}.ply")
                n = crop(cloud_path, centre, crop_path)
                jobs.append((args.session, scan, method, name, crop_path, stl))
                print(f"{scan}/{method}/{name}: {n} points cropped")
    skipped = [n for n in args.objects if OBJECTS[n][1] is None]
    if skipped:
        print(f"no CAD model for {', '.join(skipped)}: not scored")

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(compare, jobs))

    summary = os.path.join(paths.RESULTS_DIR, args.session, "objects_summary.csv")
    rows = {}
    if os.path.exists(summary):
        with open(summary) as f:
            rows = {(r["scan"], r["method"], r["object"], r["fit"]): r for r in csv.DictReader(f)}
    created = datetime.datetime.now().isoformat(timespec="seconds")
    for (session, scan, method, name, _, stl), metrics in results:
        if metrics is None:
            continue
        for fit in FITS:
            m = metrics["fits"].get(fit)
            if m is None:
                continue
            rows[(scan, method, name, fit)] = {
                "created": created, "scan": scan, "method": method, "object": name, "reference": stl,
                "fit": fit, "points": m["points"], "coverage": round(m["coverage"], 3),
                "mean_signed_mm": round(m["mean_signed_mm"], 3), "rms_mm": round(m["rms_mm"], 3),
                "median_abs_mm": round(m["median_abs_mm"], 3), "p95_abs_mm": round(m["p95_abs_mm"], 3),
                "within_2mm": round(m["within_2mm"], 3),
                "tilt_deg": round(float(np.hypot(*m["transform_board_from_cad"]["rotation_deg_xyz"][:2])), 2),
                "lift_mm": round(m["transform_board_from_cad"]["translation_mm"][2], 2)}
    fields = ["created", "scan", "method", "object", "reference", "fit", "points", "coverage", "mean_signed_mm",
              "rms_mm", "median_abs_mm", "p95_abs_mm", "within_2mm", "tilt_deg", "lift_mm"]
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: (r["object"], r["fit"], r["scan"], r["method"])))
    print(f"\n{len(results)} comparisons -> {paths.rel(summary)}")


if __name__ == "__main__":
    main()
