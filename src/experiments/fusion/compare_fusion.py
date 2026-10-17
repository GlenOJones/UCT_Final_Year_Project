"""Compare the fusion methods' clouds: per-object CAD scores, a summary table and side-by-side figures.

Reads only what run_fusion.py saved: each method's board-frame cloud (<work>/<method>/cloud_board.ply), the
board fit (alignment.json) and the run record (run_fusion.json). Each object is cropped to a box of
+-crop_half_mm around its centre (config evaluation.objects) and scored by compare_to_cad.py against its STL,
exactly as score_objects.py does for Sep30. The clouds are scored as fused (no postprocess_cloud.py), so what
is compared is the fusion itself; compare_to_cad.py keeps the largest cluster above the board.

compare_to_cad.py's fits (see its docstring): on_board (absolute: x, y, yaw only, so any height or tilt error
of the reconstruction, or of the board fit, shows), on_local_board (the board levelled around the object
first: the object's own error), shape (best rigid fit: shape alone).

    src/venv/bin/python src/experiments/fusion/compare_fusion.py --config src/experiments/fusion/config.json

Writes to results/<out_dir>/<recording>/:
    summary.md, summary.csv      per method and object: points, coverage, error statistics, per fit
    overview.png                 top-down height map of each method's cloud, objects outlined
    objects.png                  each object (rows) as each method reconstructed it (columns)
    comparison/<method>/<object>/ compare_to_cad.py's outputs (metrics.json, error maps, distances.ply)
    objects/<method>/<object>.ply the crops
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
from run_fusion import recording_paths  # noqa: E402

COMPARE_SCRIPT = os.path.join("src", "04_Comparison", "compare_to_cad.py")
STL_DIR = os.path.join("data", "GroundTruth", "roughness_blocks_STL")
FITS = ("on_board", "on_local_board", "shape")
CELL_MM = 1.0   # height-map raster for the figures


def load_cloud(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


# ====== CAD SCORES ======
def crop(xyz_cloud, centre, half, out_path):
    cloud = o3d.io.read_point_cloud(xyz_cloud)
    xyz = np.asarray(cloud.points)
    inside = np.all(np.abs(xyz[:, :2] - np.asarray(centre)) < half, axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_point_cloud(out_path, cloud.select_by_index(np.flatnonzero(inside)))
    return int(inside.sum())


def score(job):
    """compare_to_cad.py on one crop; its metrics.json, or None if it failed."""
    method, name, crop_path, stl, out_dir, python = job
    result = subprocess.run([python, common.project_path(COMPARE_SCRIPT), crop_path, "--out-dir", out_dir,
                             "--reference", common.project_path(os.path.join(STL_DIR, stl)), "--no-summary"],
                            capture_output=True, text=True, cwd=common.PROJECT_ROOT)
    metrics_path = os.path.join(out_dir, name, "metrics.json")
    if result.returncode != 0 or not os.path.exists(metrics_path):
        print(f"  {method}/{name}: FAILED\n{result.stderr[-800:]}")
        return job, None
    with open(metrics_path) as fh:
        return job, json.load(fh)


def score_all(cfg, p):
    ev = cfg["evaluation"]
    jobs = []
    for method in cfg["methods"]:
        cloud = os.path.join(p["method"](method["name"]), "cloud_board.ply")
        for name, obj in ev["objects"].items():
            crop_path = os.path.join(p["out"], "objects", method["name"], f"{name}.ply")
            n = crop(cloud, obj["centre"], ev["crop_half_mm"], crop_path)
            print(f"{method['name']}/{name}: {n} points cropped")
            jobs.append((method["name"], name, crop_path, obj["stl"],
                         os.path.join(p["out"], "comparison", method["name"]), common.project_path(cfg["python"])))
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(score, jobs))


def write_tables(cfg, p, results, run_record):
    labels = {m["name"]: m["label"] for m in cfg["methods"]}
    rows = []
    for (method, name, _, stl, _, _), metrics in results:
        if metrics is None:
            continue
        for fit in FITS:
            m = metrics["fits"].get(fit)
            if m is None:
                continue
            rows.append({"method": method, "object": name, "reference": stl, "fit": fit, "points": m["points"],
                         "coverage": round(m["coverage"], 3), "mean_signed_mm": round(m["mean_signed_mm"], 3),
                         "rms_mm": round(m["rms_mm"], 3), "median_abs_mm": round(m["median_abs_mm"], 3),
                         "p95_abs_mm": round(m["p95_abs_mm"], 3), "within_2mm": round(m["within_2mm"], 3)})
    with open(os.path.join(p["out"], "summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    timing = {r["step"].split(": ")[-1]: r for r in run_record["runs"]}
    lines = [f"# Fusion comparison: {cfg['recording']['title']}", "",
             f"Made {datetime.datetime.now().isoformat(timespec='seconds')} by compare_fusion.py. "
             f"{len(run_record['frames'])} frames, COLMAP poses for all three methods, tags masked. "
             "Clouds scored as fused. on_local_board = the object's own error (board levelled around it); "
             "shape = best rigid fit.", "",
             "| Method | Points (whole cloud) | Fusion time (s) |", "|---|---|---|"]
    for method in cfg["methods"]:
        seconds = timing.get(method["name"], {}).get("seconds")
        total = sum(seconds.values()) if isinstance(seconds, dict) else seconds
        lines.append(f"| {method['label']} | {run_record['points'].get(method['name'], 0):,} | "
                     f"{'-' if total is None else f'{total:.0f}'} |")
    for fit in ("on_local_board", "shape"):
        lines += ["", f"## Fit: {fit}", "",
                  "| Object | Method | Points | Coverage | Median abs (mm) | RMS (mm) | p95 abs (mm) | Within 2 mm |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in cfg["evaluation"]["objects"]:
            for r in (r for r in rows if r["object"] == name and r["fit"] == fit):
                lines.append(f"| {name} | {labels[r['method']]} | {r['points']:,} | {r['coverage']:.0%} | "
                             f"{r['median_abs_mm']:.2f} | {r['rms_mm']:.2f} | {r['p95_abs_mm']:.2f} | {r['within_2mm']:.0%} |")
    path = os.path.join(p["out"], "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {common.rel(path)}")


# ====== FIGURES ======
def height_map(xyz, x_range, y_range):
    """Highest point per CELL_MM cell (NaN where empty), rows = y from top."""
    nx = int(np.ceil((x_range[1] - x_range[0]) / CELL_MM))
    ny = int(np.ceil((y_range[1] - y_range[0]) / CELL_MM))
    i = np.floor((xyz[:, 0] - x_range[0]) / CELL_MM).astype(int)
    j = np.floor((y_range[1] - xyz[:, 1]) / CELL_MM).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    grid = np.full(ny * nx, -np.inf)
    np.maximum.at(grid, j[ok] * nx + i[ok], xyz[ok, 2])
    grid[np.isinf(grid)] = np.nan
    return grid.reshape(ny, nx)


def figures(cfg, p):
    ev = cfg["evaluation"]
    low, high = ev["height_range_mm"]
    clouds = {}
    for m in cfg["methods"]:
        xyz = load_cloud(os.path.join(p["method"](m["name"]), "cloud_board.ply"))
        clouds[m["name"]] = xyz[(xyz[:, 2] >= low) & (xyz[:, 2] <= high)]
    centres = np.array([o["centre"] for o in ev["objects"].values()])
    half = ev["crop_half_mm"]
    x_range = (centres[:, 0].min() - 2 * half, centres[:, 0].max() + 2 * half)
    y_range = (centres[:, 1].min() - 2 * half, centres[:, 1].max() + 2 * half)
    extent = (*x_range, *y_range)

    fig, axes = plt.subplots(1, len(cfg["methods"]), figsize=(6 * len(cfg["methods"]), 5.2), constrained_layout=True)
    for ax, m in zip(np.atleast_1d(axes), cfg["methods"]):
        image = ax.imshow(height_map(clouds[m["name"]], x_range, y_range), extent=extent, cmap="turbo",
                          vmin=0, vmax=high, interpolation="nearest")
        for name, obj in ev["objects"].items():
            cx, cy = obj["centre"]
            ax.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec="k", lw=0.8, ls="--"))
            ax.text(cx, cy + half + 4, name, ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{m['label']}\n{len(clouds[m['name']]):,} points {low:g} to {high:g} mm above the board",
                     fontsize=10)
        ax.set_xlabel("board x (mm)")
        ax.set_ylabel("board y (mm)")
    fig.colorbar(image, ax=axes, label="height above board (mm)", shrink=0.8)
    fig.savefig(os.path.join(p["out"], "overview.png"), dpi=ev["figure_dpi"])
    plt.close(fig)

    names = list(ev["objects"])
    fig, axes = plt.subplots(len(names), len(cfg["methods"]), figsize=(3.2 * len(cfg["methods"]), 3.0 * len(names)),
                             constrained_layout=True, squeeze=False)
    for r, name in enumerate(names):
        cx, cy = ev["objects"][name]["centre"]
        xr, yr = (cx - half, cx + half), (cy - half, cy + half)
        for c, m in enumerate(cfg["methods"]):
            ax = axes[r, c]
            image = ax.imshow(height_map(clouds[m["name"]], xr, yr), extent=(*xr, *yr), cmap="turbo",
                              vmin=0, vmax=high, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(m["label"], fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.colorbar(image, ax=axes, label="height above board (mm)", shrink=0.6)
    fig.savefig(os.path.join(p["out"], "objects.png"), dpi=ev["figure_dpi"])
    plt.close(fig)
    print(f"figures -> {common.rel(p['out'])}/overview.png, objects.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    p = recording_paths(cfg)
    with open(os.path.join(p["out"], "run_fusion.json"), encoding="utf-8") as fh:
        run_record = json.load(fh)
    results = score_all(cfg, p)
    write_tables(cfg, p, results, run_record)
    figures(cfg, p)


if __name__ == "__main__":
    main()
