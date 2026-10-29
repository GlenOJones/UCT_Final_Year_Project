"""Compare the fusion methods' clouds with the CAD models: an accuracy table and per-object figures.

    src/venv/bin/python src/experiments/fusion/compare_fusion.py --config src/experiments/fusion/config.json
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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
for _stage in ("04_pose", "evaluation"):
    sys.path.insert(0, os.path.join(SRC, _stage))
import common  # noqa: E402
import compare_to_cad  # noqa: E402
import pose_files  # noqa: E402
from run_fusion import recording_paths  # noqa: E402

COMPARE_SCRIPT = os.path.join("src", "evaluation", "compare_to_cad.py")
STL_DIR = os.path.join("data", "GroundTruth", "roughness_blocks_STL")
FITS = ("on_board", "on_local_board", "shape")
# One colour per method, in config order, for the overlaid profiles.
PROFILE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DIRECTIONS = ("+x", "+y", "-x", "-y", "up")
DIRECTION_LABELS = {"+x": "+x", "+y": "+y", "-x": "\u2212x", "-y": "\u2212y", "up": "top"}
OBJECT_LABELS = {"PYRAMID_1": "Pyramid", "PYRAMID_2X2": "Pyramid 2\u00d72", "EGG_1PK": "Egg carton",
                 "EGG_2PKS": "Egg carton (2)"}


# ====== CAD SCORES ======
def crop(cloud_path, centre, half, out_path):
    cloud = o3d.io.read_point_cloud(cloud_path)
    xyz = np.asarray(cloud.points)
    inside = np.all(np.abs(xyz[:, :2] - np.asarray(centre)) < half, axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_point_cloud(out_path, cloud.select_by_index(np.flatnonzero(inside)))
    return int(inside.sum())


def score(job):
    """compare_to_cad.py on one crop; its metrics.json, or None if it failed."""
    method, name, crop_path, stl, out_dir, python, extra = job
    result = subprocess.run([python, common.project_path(COMPARE_SCRIPT), crop_path, "--out-dir", out_dir,
                             "--reference", common.project_path(os.path.join(STL_DIR, stl)), "--no-summary", *extra],
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
        if not os.path.exists(cloud):
            print(f"{method['name']}: no cloud yet ({common.rel(cloud)}); left blank in the table and figures")
            continue
        for name, obj in ev["objects"].items():
            crop_path = os.path.join(p["out"], "objects", method["name"], f"{name}.ply")
            n = crop(cloud, obj["centre"], ev["crop_half_mm"], crop_path)
            print(f"{method['name']}/{name}: {n} points cropped")
            # obj["square_yaw"]: the object sits square on the board (see compare_to_cad.py --square-yaw).
            extra = ["--square-yaw", str(obj["square_yaw"])] if "square_yaw" in obj else []
            jobs.append((method["name"], name, crop_path, obj["stl"],
                         os.path.join(p["out"], "comparison", method["name"]), common.project_path(cfg["python"]), extra))
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(score, jobs))


def saved_scores(cfg, p):
    """The (job, metrics) pairs of an earlier score_all, read back from the crops and metrics.json it left."""
    results = []
    for method in cfg["methods"]:
        for name, obj in cfg["evaluation"]["objects"].items():
            out_dir = os.path.join(p["out"], "comparison", method["name"])
            metrics_path = os.path.join(out_dir, name, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as fh:
                metrics = json.load(fh)
            crop_path = os.path.join(p["out"], "objects", method["name"], f"{name}.ply")
            results.append(((method["name"], name, crop_path, obj["stl"], out_dir, None, []), metrics))
    return results


# ====== T1: ACCURACY TABLE ======
def table_rows(cfg, results):
    rows = []
    for (method, name, _, stl, _, _, _), metrics in results:
        if metrics is None:
            continue
        for fit in FITS:
            m = metrics["fits"].get(fit)
            if m is None:
                continue
            rows.append({"method": method, "object": name, "reference": stl, "fit": fit, "points": m["points"],
                         "coverage": m["coverage"], "mean_signed_mm": m["mean_signed_mm"], "rms_mm": m["rms_mm"],
                         "median_abs_mm": m["median_abs_mm"], "p95_abs_mm": m["p95_abs_mm"],
                         "within_1mm": m["within_1mm"], "within_2mm": m["within_2mm"], "within_5mm": m["within_5mm"]})
    return rows


def write_tables(cfg, p, rows, run_record):
    ev = cfg["evaluation"]
    labels = {m["name"]: m["label"] for m in cfg["methods"]}
    with open(os.path.join(p["out"], "summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)

    fit = ev["report_fit"]
    # One row per method in config order; a method with no result yet keeps its row, blank (None).
    found = {(r["object"], r["method"]): r for r in rows if r["fit"] == fit}
    chosen = [(name, [found.get((name, m["name"]), {"method": m["name"], "blank": True}) for m in cfg["methods"]])
              for name in ev["report_objects"]]
    frames = len(run_record["frames"])
    caption = (f"Accuracy of each fusion method against the CAD model, {cfg['recording']['title']}, {frames} frames, "
               f"COLMAP poses. Fit: {fit.replace('_', ' ')}. Coverage is the share of the visible CAD surface with "
               "a scan point within 2~mm; distances are from every object point to the CAD surface.")
    tex = ["% Made by src/experiments/fusion/compare_fusion.py from compare_to_cad.py's metrics.json; do not edit.",
           r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:fusion_accuracy}",
           r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
           r"Object & Method & Coverage (\%) & Median $|d|$ (mm) & RMS (mm) & p95 $|d|$ (mm) & "
           r"$\leq$1~mm (\%) & $\leq$2~mm (\%) & $\leq$5~mm (\%) \\", r"\midrule"]
    md = [f"# Fusion accuracy: {cfg['recording']['title']}", "",
          f"Made {datetime.datetime.now().isoformat(timespec='seconds')} by compare_fusion.py. {frames} frames, "
          f"COLMAP poses, tags masked, clouds scored as fused. Fit: {fit}.", "",
          "| Object | Method | Coverage | Median abs (mm) | RMS (mm) | p95 abs (mm) | <=1 mm | <=2 mm | <=5 mm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, (name, object_rows) in enumerate(chosen):
        for i, r in enumerate(object_rows):
            first = rf"\multirow{{{len(object_rows)}}}{{*}}{{{name.replace('_', chr(92) + '_')}}}" if i == 0 else ""
            if r.get("blank"):
                tex.append(f"{first} & {labels[r['method']]}" + " & --" * 7 + r" \\")
                md.append(f"| {name} | {labels[r['method']]} |" + " - |" * 7)
                continue
            tex.append(f"{first} & {labels[r['method']]} & {100 * r['coverage']:.0f} & {r['median_abs_mm']:.2f} & "
                       f"{r['rms_mm']:.2f} & {r['p95_abs_mm']:.2f} & {100 * r['within_1mm']:.0f} & "
                       f"{100 * r['within_2mm']:.0f} & {100 * r['within_5mm']:.0f} \\\\")
            md.append(f"| {name} | {labels[r['method']]} | {r['coverage']:.0%} | {r['median_abs_mm']:.2f} | "
                      f"{r['rms_mm']:.2f} | {r['p95_abs_mm']:.2f} | {r['within_1mm']:.0%} | {r['within_2mm']:.0%} | "
                      f"{r['within_5mm']:.0%} |")
        if k < len(chosen) - 1:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    for filename, lines in (("table_accuracy.tex", tex), ("summary.md", md)):
        with open(os.path.join(p["out"], filename), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(md))
    print(f"\n-> {common.rel(p['out'])}/table_accuracy.tex, summary.md, summary.csv")


# ====== FIGURES: SIGNED DISTANCE FROM ABOVE AND PROFILES THROUGH THE APEX ======
def signed_distances(metrics, crop_path, stl, fit):
    """(object points, signed distance to the CAD, T_board_cad, Reference) for one crop and fit, recomputed with
    compare_to_cad.py's own functions and the placement and settings it saved, so they match its numbers."""
    settings = argparse.Namespace(**metrics["settings"])
    reference = compare_to_cad.Reference(common.project_path(os.path.join(STL_DIR, stl)))
    xyz = np.asarray(o3d.io.read_point_cloud(crop_path).points, np.float64)
    points = compare_to_cad.object_points(xyz, settings.min_height, reference.height)
    T = np.array(metrics["fits"][fit]["transform_board_from_cad"]["matrix"])
    _, _, signed, keep = compare_to_cad.evaluate(reference, points, T, settings)
    return points[keep], signed, T, reference


def figures(cfg, p, results):
    ev = cfg["evaluation"]
    fit, limit = ev["report_fit"], ev["error_limit_mm"]
    cmap = LinearSegmentedColormap.from_list("signed", compare_to_cad.DIVERGING)
    norm = Normalize(-limit, limit)
    by_key = {(job[0], job[1]): (job, metrics) for job, metrics in results}
    methods = cfg["methods"]
    for name in ev["report_objects"]:
        fig, axes = plt.subplots(len(methods), 2, figsize=(11, 4.0 * len(methods)), squeeze=False,
                                 gridspec_kw={"width_ratios": [1, 1.4]}, constrained_layout=True)
        legend_drawn = False
        for row, method in enumerate(methods):
            job, metrics = by_key.get((method["name"], name), (None, None))
            top, prof = axes[row]
            if metrics is None:
                # Keep the method's row, empty, so the figure's layout does not change when it arrives.
                for ax in (top, prof):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, "results pending", transform=ax.transAxes, ha="center", va="center",
                            color=compare_to_cad.MUTED, fontsize=11)
                top.set_title(f"{method['label']}", loc="left", fontsize=10)
                prof.set_title("Profiles through the apex (slabs of +/-3 mm)", loc="left", fontsize=10)
                continue
            points, signed, T, reference = signed_distances(metrics, job[2], job[3], fit)
            stats = metrics["fits"][fit]
            compare_to_cad.draw_signed_top(top, reference, points, signed, T, cmap, norm)
            compare_to_cad.draw_profiles(prof, reference, points, T)
            top.set_title(f"{method['label']}\nmedian |d| {stats['median_abs_mm']:.2f} mm, "
                          f"coverage {100 * stats['coverage']:.0f}%", loc="left", fontsize=10)
            prof.set_title("Profiles through the apex (slabs of +/-3 mm)", loc="left", fontsize=10)
            if not legend_drawn:
                prof.legend(frameon=False, fontsize=8, loc="upper right")
                legend_drawn = True
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=axes[:, 0], orientation="horizontal",
                           shrink=0.8, aspect=30)
        bar.set_label("signed distance to CAD (mm), positive = outside")
        path = os.path.join(p["out"], f"accuracy_{name}.png")
        fig.savefig(path, dpi=ev["figure_dpi"])
        plt.close(fig)
        print(f"-> {common.rel(path)}")


def profile_figures(cfg, p, results):
    """Per report object, every method's profile through the CAD apex on one plot: one panel along the CAD model's x and
    one along its y."""
    ev = cfg["evaluation"]
    fit = ev["report_fit"]
    by_key = {(job[0], job[1]): (job, metrics) for job, metrics in results}
    for name in ev["report_objects"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True, facecolor=compare_to_cad.SURFACE)
        reference = None
        for k, method in enumerate(cfg["methods"]):
            job, metrics = by_key.get((method["name"], name), (None, None))
            if metrics is None:
                continue
            points, _, T, reference = signed_distances(metrics, job[2], job[3], fit)
            local = compare_to_cad.transform(np.linalg.inv(T), points)
            apex = compare_to_cad.cad_apex(reference)
            colour = PROFILE_COLOURS[k]
            for ax, (axis, other) in zip(axes, ((0, 1), (1, 0))):
                slab = local[np.abs(local[:, other] - apex[other]) < 3.0]
                s, z = slab[:, axis] - apex[axis], slab[:, 2]
                ax.scatter(s, z, s=1, color=colour, alpha=0.08, linewidths=0, rasterized=True)
                bins = np.floor(s).astype(int)
                edges = np.arange(bins.min(), bins.max() + 1) if len(bins) else np.array([], int)
                medians = np.array([np.median(z[bins == b]) if np.sum(bins == b) >= 5 else np.nan for b in edges])
                ax.plot(edges + 0.5, medians, color=colour, lw=2, label=method["label"])
        if reference is None:
            plt.close(fig)
            continue
        apex = compare_to_cad.cad_apex(reference)
        for ax, (axis, other) in zip(axes, ((0, 1), (1, 0))):
            ax.plot(*compare_to_cad.cad_section(reference, apex, axis, other), color=compare_to_cad.INK, lw=1.4,
                    ls="--", label="CAD")
            ax.set_facecolor(compare_to_cad.SURFACE)
            ax.set_aspect("equal")
            ax.set_ylim(bottom=0)
            ax.set_xlabel("distance from apex (mm)")
            ax.set_ylabel("height (mm)")
            ax.grid(color=compare_to_cad.MUTED, alpha=0.2, lw=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        # The report caption names the panels (left: along the CAD model's x, right: along its y), so no titles.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=len(labels))
        path = os.path.join(p["out"], f"profiles_{name}.png")
        fig.savefig(path, dpi=ev["figure_dpi"])
        plt.close(fig)
        print(f"-> {common.rel(path)}")


def coverage_by_direction(reference, points, T, radius):
    """{board direction: (coverage, share of the visible CAD area)} for one fit."""
    local = compare_to_cad.transform(np.linalg.inv(T), points)
    o3d.utility.random.seed(0)
    samples = reference.mesh.sample_points_uniformly(max(int(reference.mesh.get_surface_area()), 1000))
    sample_xyz = np.asarray(samples.points)
    _, _, sample_tri = reference.closest(sample_xyz)
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local)))
    covered = np.array([tree.search_radius_vector_3d(q, radius)[0] > 0 for q in sample_xyz])
    directions = np.array([compare_to_cad.board_direction(T[:3, :3] @ n) for n in reference.normals[sample_tri]])
    return {d: (float(np.mean(covered[directions == d])), float(np.mean(directions == d)))
            for d in DIRECTIONS if np.any(directions == d)}


def coverage_figure(cfg, p, results):
    """Coverage per face direction: one panel per report object, one bar per method for each direction the object's
    faces point on the board (directions under 2% of its visible area left out)."""
    ev = cfg["evaluation"]
    fit = ev["report_fit"]
    by_key = {(job[0], job[1]): (job, metrics) for job, metrics in results}
    methods = cfg["methods"]
    names = ev["report_objects"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), sharey=True, squeeze=False,
                             constrained_layout=True, facecolor=compare_to_cad.SURFACE)
    for ax, name in zip(axes[0], names):
        found = {}
        for method in methods:
            job, metrics = by_key.get((method["name"], name), (None, None))
            if metrics is None:
                continue
            points, _, T, reference = signed_distances(metrics, job[2], job[3], fit)
            found[method["name"]] = coverage_by_direction(reference, points, T, metrics["settings"]["coverage_radius"])
        shown = [d for d in DIRECTIONS if max((c.get(d, (0, 0))[1] for c in found.values()), default=0) >= 0.02]
        width = 0.8 / len(methods)
        for k, method in enumerate(methods):
            if method["name"] not in found:
                continue
            values = [100 * found[method["name"]].get(d, (np.nan, 0))[0] for d in shown]
            ax.bar(np.arange(len(shown)) + (k - (len(methods) - 1) / 2) * width, values, width * 0.9,
                   color=PROFILE_COLOURS[k], label=method["label"])
        ax.set_xticks(np.arange(len(shown)), [DIRECTION_LABELS[d] for d in shown])
        ax.set_facecolor(compare_to_cad.SURFACE)
        ax.set_ylim(0, 100)
        ax.set_xlabel(OBJECT_LABELS.get(name, name))
        ax.grid(axis="y", color=compare_to_cad.MUTED, alpha=0.2, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("coverage (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=len(labels))
    path = os.path.join(p["out"], "coverage_by_face.png")
    fig.savefig(path, dpi=ev["figure_dpi"])
    plt.close(fig)
    print(f"-> {common.rel(path)}")


# ====== TOP-DOWN VIEW AND COVERAGE (scans without CAD: the ice) ======
def camera_side(p):
    """+1 if the cameras sit on the board frame's +z side (S30: looking down on the board), -1 if on the -z side, so
    'height' can always be measured towards the camera and each cell keeps its point nearest the camera."""
    _, reference, _ = pose_files.read_trajectory(p["reference"])
    centres_z = [np.linalg.inv(T)[2, 3] for T in reference.values()]
    return 1.0 if np.median(centres_z) >= 0 else -1.0


def height_map(xyz, x_range, y_range, cell):
    """Per cell, the height (towards the camera) of the point nearest the camera; NaN where empty."""
    nx = int(np.ceil((x_range[1] - x_range[0]) / cell))
    ny = int(np.ceil((y_range[1] - y_range[0]) / cell))
    i = np.floor((xyz[:, 0] - x_range[0]) / cell).astype(int)
    j = np.floor((y_range[1] - xyz[:, 1]) / cell).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    grid = np.full(ny * nx, -np.inf)
    np.maximum.at(grid, j[ok] * nx + i[ok], xyz[ok, 2])
    grid[np.isinf(grid)] = np.nan
    return grid.reshape(ny, nx)


def board_clouds(cfg, p, side):
    """{method name: board-frame points with z flipped to 'height towards the camera'} for the methods with a cloud."""
    clouds = {}
    for m in cfg["methods"]:
        path = os.path.join(p["method"](m["name"]), "cloud_board.ply")
        if os.path.exists(path):
            xyz = np.asarray(o3d.io.read_point_cloud(path).points)
            clouds[m["name"]] = xyz * np.array([1.0, 1.0, side])
    return clouds


def height_limits(ev, clouds, x_range, y_range):
    if ev.get("height_range_mm"):
        return ev["height_range_mm"]
    z = np.concatenate([c[(c[:, 0] >= x_range[0]) & (c[:, 0] <= x_range[1]) & (c[:, 1] >= y_range[0]) &
                          (c[:, 1] <= y_range[1]), 2] for c in clouds.values()])
    return np.percentile(z, [2, 98]) if len(z) else (0, 1)


def topdown_figure(cfg, p, clouds, x_range, y_range, path, title, region=None, grid_mm=50):
    ev = cfg["evaluation"]
    low, high = height_limits(ev, clouds, x_range, y_range)
    methods = cfg["methods"]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods), 5.4), squeeze=False, constrained_layout=True)
    image = None
    for ax, m in zip(axes[0], methods):
        if m["name"] not in clouds:
            ax.text(0.5, 0.5, "results pending", transform=ax.transAxes, ha="center", va="center",
                    color=compare_to_cad.MUTED, fontsize=11)
            ax.set_title(m["label"], loc="left", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        image = ax.imshow(height_map(clouds[m["name"]], x_range, y_range, ev["coverage_cell_mm"]),
                          extent=(*x_range, *y_range), cmap="turbo", vmin=low, vmax=high, interpolation="nearest")
        if region:
            x0, x1, y0, y1 = region
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="k", lw=1.2, ls="--"))
        # Grid lines every grid_mm, labels every 2 x grid_mm so they stay readable.
        for axis, (low_, high_) in ((ax.xaxis, x_range), (ax.yaxis, y_range)):
            axis.set_ticks(np.arange(np.ceil(low_ / grid_mm) * grid_mm, high_, grid_mm), minor=True)
            axis.set_ticks(np.arange(np.ceil(low_ / (2 * grid_mm)) * 2 * grid_mm, high_, 2 * grid_mm))
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", color="k", alpha=0.25, lw=0.5)
        ax.set_title(m["label"], loc="left", fontsize=10)
        ax.set_xlabel("board x (mm)")
        ax.set_ylabel("board y (mm)")
    if image is not None:
        fig.colorbar(image, ax=axes[0], label="height towards the camera (mm)", shrink=0.8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.savefig(path, dpi=ev["figure_dpi"])
    plt.close(fig)
    print(f"-> {common.rel(path)}")


def topdown_and_coverage(cfg, p):
    """topdown.png (every method over the whole scan, a 50 mm grid to read the ice region off); with evaluation.region
    set, also topdown_region.png and the top-down coverage table."""
    ev = cfg["evaluation"]
    side = camera_side(p)
    clouds = board_clouds(cfg, p, side)
    if not clouds:
        print("no clouds yet: no top-down figures")
        return
    union = np.concatenate(list(clouds.values()))
    x_range = tuple(np.percentile(union[:, 0], [0.5, 99.5]) + [-10, 10])
    y_range = tuple(np.percentile(union[:, 1], [0.5, 99.5]) + [-10, 10])
    region = ev.get("region")
    topdown_figure(cfg, p, clouds, x_range, y_range, os.path.join(p["out"], "topdown.png"),
                   f"{cfg['recording']['title']}: nearest point to the camera per {ev['coverage_cell_mm']:g} mm cell, "
                   "board frame" + ("" if region else " (pick the ice region from the grid)"), region)
    if not region:
        return
    x0, x1, y0, y1 = region
    topdown_figure(cfg, p, clouds, (x0, x1), (y0, y1), os.path.join(p["out"], "topdown_region.png"),
                   f"{cfg['recording']['title']}: ice region", grid_mm=20)
    cell = ev["coverage_cell_mm"]
    total = int(np.ceil((x1 - x0) / cell) * np.ceil((y1 - y0) / cell))
    rows = []
    for m in cfg["methods"]:
        if m["name"] not in clouds:
            rows.append({"method": m["name"], "label": m["label"], "blank": True})
            continue
        xyz = clouds[m["name"]]
        inside = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
        filled = np.isfinite(height_map(xyz[inside], (x0, x1), (y0, y1), cell)).sum()
        rows.append({"method": m["name"], "label": m["label"], "points": int(inside.sum()),
                     "cells_filled": int(filled), "cells_total": total, "coverage": filled / total})
    frames = len(json.load(open(os.path.join(p["out"], "run_fusion.json")))["frames"])
    with open(os.path.join(p["out"], "coverage.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "points", "cells_filled", "cells_total", "coverage"])
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in writer.fieldnames} for r in rows)
    tex = ["% Made by src/experiments/fusion/compare_fusion.py from run_fusion.py's clouds; do not edit.",
           r"\begin{table}[H]", r"\centering",
           rf"\caption{{Top-down coverage of the ice region, {cfg['recording']['title']}, {frames} frames, COLMAP poses: "
           rf"the share of the region's {cell:g}~mm cells with at least one point.}}",
           rf"\label{{tab:fusion_coverage_{cfg['recording']['name']}}}", r"\begin{tabular}{lrr}", r"\toprule",
           r"Method & Coverage (\%) & Points in region ($\times10^3$) \\", r"\midrule"]
    md = [f"# Top-down coverage: {cfg['recording']['title']}", "",
          f"Region x {x0:g}..{x1:g}, y {y0:g}..{y1:g} mm (board frame), {cell:g} mm cells, {frames} frames.", "",
          "| Method | Coverage | Points in region |", "|---|---|---|"]
    for r in rows:
        if r.get("blank"):
            tex.append(rf"{r['label']} & -- & -- \\")
            md.append(f"| {r['label']} | - | - |")
        else:
            tex.append(rf"{r['label']} & {100 * r['coverage']:.1f} & {r['points'] / 1000:.0f} \\")
            md.append(f"| {r['label']} | {r['coverage']:.1%} | {r['points']:,} |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    for filename, lines in (("table_coverage.tex", tex), ("coverage.md", md)):
        with open(os.path.join(p["out"], filename), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(md))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from the saved scores instead of scoring again")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    p = recording_paths(cfg)
    with open(os.path.join(p["out"], "run_fusion.json"), encoding="utf-8") as fh:
        run_record = json.load(fh)
    if cfg["evaluation"]["objects"]:
        if args.figures_only:
            results = saved_scores(cfg, p)
        else:
            results = score_all(cfg, p)
            write_tables(cfg, p, table_rows(cfg, results), run_record)
        figures(cfg, p, results)
        profile_figures(cfg, p, results)
        coverage_figure(cfg, p, results)
    else:
        topdown_and_coverage(cfg, p)


if __name__ == "__main__":
    main()
