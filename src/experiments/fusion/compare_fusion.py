"""Compare the fusion methods' clouds with the CAD models: an accuracy table and per-object figures.

Reads only what run_fusion.py saved: each method's board-frame cloud (<work>/<method>/cloud_board.ply) and the
run record (run_fusion.json). Each object is cropped to a box of +-crop_half_mm around its centre (config
evaluation.objects) and scored by compare_to_cad.py against its STL, exactly as score_objects.py does for
Sep30. The clouds are scored as fused (no postprocess_cloud.py), so what is compared is the fusion itself;
compare_to_cad.py keeps the cluster of points that reaches highest above the board.

compare_to_cad.py makes three fits (see its docstring). The table and figures use one, evaluation.report_fit
(on_local_board: the board levelled around the object first, so the object's own error is measured rather than
the board frame's); summary.csv has all three.

    src/venv/bin/python src/experiments/fusion/compare_fusion.py --config src/experiments/fusion/config.json

Writes to results/<out_dir>/<recording>/:
    table_accuracy.tex            T1: per report object and method, coverage and error statistics (LaTeX)
    summary.md, summary.csv       the same table as Markdown; every object, method and fit as CSV
    accuracy_<object>.png         per report object: one row per method, signed distance to the CAD from above
                                  and profiles through the apex (compare_to_cad.py's own panels)
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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(SRC, "04_Comparison"))
import common  # noqa: E402
import compare_to_cad  # noqa: E402
from run_fusion import recording_paths  # noqa: E402

COMPARE_SCRIPT = os.path.join("src", "04_Comparison", "compare_to_cad.py")
STL_DIR = os.path.join("data", "GroundTruth", "roughness_blocks_STL")
FITS = ("on_board", "on_local_board", "shape")


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
        if not os.path.exists(cloud):
            print(f"{method['name']}: no cloud yet ({common.rel(cloud)}); left blank in the table and figures")
            continue
        for name, obj in ev["objects"].items():
            crop_path = os.path.join(p["out"], "objects", method["name"], f"{name}.ply")
            n = crop(cloud, obj["centre"], ev["crop_half_mm"], crop_path)
            print(f"{method['name']}/{name}: {n} points cropped")
            jobs.append((method["name"], name, crop_path, obj["stl"],
                         os.path.join(p["out"], "comparison", method["name"]), common.project_path(cfg["python"])))
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(score, jobs))


# ====== T1: ACCURACY TABLE ======
def table_rows(cfg, results):
    rows = []
    for (method, name, _, stl, _, _), metrics in results:
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
           r"\begin{tabular}{llrrrrrrr}", r"\toprule",
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
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    p = recording_paths(cfg)
    with open(os.path.join(p["out"], "run_fusion.json"), encoding="utf-8") as fh:
        run_record = json.load(fh)
    results = score_all(cfg, p)
    write_tables(cfg, p, table_rows(cfg, results), run_record)
    figures(cfg, p, results)


if __name__ == "__main__":
    main()
