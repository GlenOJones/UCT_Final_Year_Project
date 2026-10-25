"""Analysis of the frame-count sweep: reads what run_frame_count.py saved and scores every subset.

Per step (every N'th frame) and depth source:
    poses        frames SfM posed, ATE / RPE against the AprilGrid tag poses (evaluate_poses.evaluate)
    coverage     share of the ice region's 1 mm cells (board frame, seen from the camera side) with a point
    cloud-to-cloud  against the same source's all-frames cloud, inside the region: accuracy = share of the subset's
                 points within each threshold of the reference; completeness = share of the reference's points
                 within it of the subset
                 (the subset's cloud first aligned onto the reference by a short rigid ICP, since its own SfM poses
                 place it a few mm from the all-frames run's; the unaligned completeness is kept too)
    cost         wall time and peak RAM / GPU memory of each stage. SfM and TSDF are measured per subset; depth is
                 made once on every frame, so a subset's depth time is the matcher's start-up (the run's wall time
                 less its frames' times) plus its own frames' times, and its peak memory is the full run's.

    src/venv/bin/python src/experiments/frame_count/analyse_frame_count.py [--config ...]

Writes, under results/<out_dir>/<recording>/:
    summary.csv / summary.md      every measure, one row per step and source
    stage_costs.md                time and peak compute per stage and step
    table_frame_count.tex         the report table (LaTeX)
    quality_vs_frames.png         registration, ATE, coverage and completeness against frames used
    pose_vs_frames.png            the SfM poses alone: frames posed, ATE median / p95 / max
    time_vs_frames.png            time per stage against frames used, one panel per depth source
    memory_by_stage.png           peak RAM and GPU memory of each stage against frames used
    oblique_by_frames.png         each subset's cloud from the same oblique view (oblique_view.py's renderer), for
                                  the depth sources in figures.oblique_sources
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from matplotlib import colormaps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(EXPERIMENTS, "fusion"))
sys.path.insert(0, os.path.join(EXPERIMENTS, "pose_estimation"))
sys.path.insert(0, os.path.join(os.path.dirname(EXPERIMENTS), "03_Reconstruction"))
import common  # noqa: E402
import compare_fusion  # noqa: E402
import oblique_view  # noqa: E402
import pose_files  # noqa: E402
from evaluate_poses import evaluate  # noqa: E402
from run_frame_count import paths  # noqa: E402

# The reference palette's categorical slots, in order (as compare_fusion.py's PROFILE_COLOURS).
SOURCE_COLOURS = {"foundationstereo": "#2a78d6", "sgbm_tuned": "#eb6834"}
STAGE_COLOURS = {"sfm": "#2a78d6", "depth": "#eb6834", "tsdf": "#1baf7a"}
STAGE_LABELS = {"sfm": "COLMAP SfM (poses)", "depth": "Depth", "tsdf": "TSDF fusion"}
MUTED = "#6b6a63"


# ====== READING THE RUNS ======
def load_steps(cfg, p):
    steps = []
    for n in cfg["steps"]:
        path = os.path.join(p["step_out"](n), "run.json")
        if os.path.exists(path):
            with open(path) as fh:
                steps.append(json.load(fh))
        else:
            print(f"step {n}: no run.json, left out")
    return steps


def depth_cost(cfg, p, source, names):
    """(seconds, peak RAM MB, peak GPU MiB) of making depth for these frames alone, from the every-frame run."""
    with open(os.path.join(p["out"], "depth_costs.json")) as fh:
        run = json.load(fh)[source]
    with open(os.path.join(p["depth"](source), "depth_frames.json")) as fh:
        per_frame = {f["name"]: f["frame_seconds"] for f in json.load(fh)["per_frame"]}
    startup = run["wall_s"] - sum(per_frame.values())
    return startup + sum(per_frame[n] for n in names if n in per_frame), run["peak_tree_rss_mb"], run["peak_gpu_mib"]


def region_cloud(path, region, side):
    if not os.path.exists(path):
        return None
    xyz = np.asarray(o3d.io.read_point_cloud(path).points) * np.array([1.0, 1.0, side])
    x0, x1, y0, y1 = region
    return xyz[(xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)]


def coverage(xyz, region, cell):
    x0, x1, y0, y1 = region
    return float(np.isfinite(compare_fusion.height_map(xyz, (x0, x1), (y0, y1), cell)).mean())


def align_to(a, b, max_distance):
    """a moved onto b by point-to-point ICP (rigid, from identity): each subset's cloud is placed in the board frame
    by its own SfM poses, which differ from the all-frames run's by a few mm, and that offset would otherwise be
    counted as missing surface. Returns (moved a, ICP fitness, translation applied in mm)."""
    pa, pb = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pa.points, pb.points = o3d.utility.Vector3dVector(a), o3d.utility.Vector3dVector(b)
    result = o3d.pipelines.registration.registration_icp(
        pa, pb, max_distance, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    T = result.transformation
    return a @ T[:3, :3].T + T[:3, 3], float(result.fitness), float(np.linalg.norm(T[:3, 3]))


def nearest(a, b):
    """Distance from every point of a to its nearest point of b (mm)."""
    pa, pb = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pa.points, pb.points = o3d.utility.Vector3dVector(a), o3d.utility.Vector3dVector(b)
    return np.asarray(pa.compute_point_cloud_distance(pb))


def clean_reference(reference, window, limit):
    """The tag poses without the frames whose camera centre lies more than limit mm from where its neighbours put
    it. Each tag-posed frame within window frames either side predicts the centre as its own centre plus the pass's
    median velocity per frame times the frame gap; the median of those predictions is compared with the frame's
    centre. The rig moves ~7-10 mm per frame along a near-straight pass, so the prediction holds at the ends of the
    pass too; a tag pose far from it (seen on the overhang: ~250 mm jumps between consecutive frames) is a wrong
    tag fit, not camera motion. Returns (cleaned reference, dropped frame names)."""
    names = sorted(reference)
    index = {n: int(os.path.splitext(n)[0]) for n in names}
    centre = {n: np.linalg.inv(reference[n])[:3, 3] for n in names}
    velocity = np.median([(centre[b] - centre[a]) / (index[b] - index[a]) for a, b in zip(names, names[1:])], axis=0)
    dropped = []
    for n in names:
        predictions = [centre[m] + velocity * (index[n] - index[m]) for m in names
                       if m != n and abs(index[m] - index[n]) <= window]
        if len(predictions) >= 3 and np.linalg.norm(centre[n] - np.median(predictions, axis=0)) > limit:
            dropped.append(n)
    return {n: T for n, T in reference.items() if n not in dropped}, dropped


# ====== SCORING ======
def score(cfg, p, steps):
    ev = cfg["evaluation"]
    region, cell, thresholds = ev["region"], ev["coverage_cell_mm"], ev["c2c_thresholds_mm"]
    side = compare_fusion.camera_side(p)
    _, reference_poses, _ = pose_files.read_trajectory(p["reference"])
    reference_poses, dropped = clean_reference(reference_poses, ev.get("reference_window", 5),
                                               ev.get("reference_jump_mm", 30))
    print(f"reference: {len(reference_poses)} tag poses kept, {len(dropped)} dropped as jumps: {dropped}")
    work = {n: p["step_work"](n) for n in cfg["steps"]}
    references = {s: region_cloud(os.path.join(work[1], f"tsdf_{s}", "cloud_board.ply"), region, side)
                  for s in cfg["depth"]}
    rows = []
    for record in steps:
        n = record["step"]
        trajectory = os.path.join(p["step_out"](n), "poses_colmap.yaml")
        base = {"step": n, "frames": record.get("frames"), "frames_posed": record.get("frames_posed", 0),
                "failed": record.get("failed", "")}
        names = []
        if os.path.exists(trajectory):
            _, poses, diagnostics = pose_files.read_trajectory(trajectory)
            names = sorted(diagnostics)
            result, _ = evaluate(reference_poses, poses, {}, names, cfg)
            base.update({"frames": len(names), "registration": result["registration"],
                         "ate_median_mm": (result.get("ate_mm") or {}).get("median"),
                         "ate_rmse_mm": (result.get("ate_mm") or {}).get("rmse"),
                         "ate_p95_mm": (result.get("ate_mm") or {}).get("p95"),
                         "ate_max_mm": (result.get("ate_mm") or {}).get("max"),
                         "rpe_median_percent": (result.get("rpe_percent") or {}).get("median")})
        sfm = record["stages"].get("sfm", {})
        base.update({"sfm_s": sfm.get("wall_s"), "sfm_ram_mb": sfm.get("peak_tree_rss_mb"),
                     "sfm_gpu_mib": sfm.get("peak_gpu_mib"),
                     **{f"sfm_{k}_s": v for k, v in record.get("sfm_seconds", {}).items()}})
        for source in cfg["depth"]:
            row = dict(base, source=source)
            depth_s, depth_ram, depth_gpu = depth_cost(cfg, p, source, names) if names else (None, None, None)
            tsdf = record["stages"].get(f"tsdf_{source}", {})
            row.update({"depth_s": depth_s, "depth_ram_mb": depth_ram, "depth_gpu_mib": depth_gpu,
                        "tsdf_s": tsdf.get("wall_s"), "tsdf_ram_mb": tsdf.get("peak_tree_rss_mb")})
            parts = [row["sfm_s"], depth_s, row["tsdf_s"]]
            row["total_s"] = sum(parts) if all(v is not None for v in parts) else None
            xyz = region_cloud(os.path.join(work[n], f"tsdf_{source}", "cloud_board.ply"), region, side)
            if xyz is not None and len(xyz):
                row.update({"points": len(xyz), "coverage": coverage(xyz, region, cell)})
                ref = references[source]
                if ref is not None and len(ref):
                    raw = nearest(ref, xyz)
                    aligned, fitness, shift = align_to(xyz, ref, ev["c2c_icp_mm"])
                    to_ref, from_ref = nearest(aligned, ref), nearest(ref, aligned)
                    row.update({"c2c_median_mm": float(np.median(to_ref)), "icp_fitness": fitness,
                                "icp_shift_mm": shift})
                    for t in thresholds:
                        row[f"accuracy_{t:g}mm"] = float((to_ref <= t).mean())
                        row[f"completeness_{t:g}mm"] = float((from_ref <= t).mean())
                        row[f"completeness_unaligned_{t:g}mm"] = float((raw <= t).mean())
            rows.append(row)
        print(f"step {n}: {base['frames_posed']}/{base['frames']} posed, ATE median "
              f"{base.get('ate_median_mm') or float('nan'):.2f} mm")
    return rows


# ====== TABLES ======
def fmt(value, spec, blank="--"):
    return blank if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, spec)


def write_tables(cfg, p, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], list(rows[0]).index(k) if k in rows[0] else 0, k))
    with open(os.path.join(p["out"], "summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    labels = {s: spec["label"] for s, spec in cfg["depth"].items()}
    md = [f"# Frame-count sweep: {cfg['recording']['title']}", "",
          "Accuracy and completeness are against the same source's all-frames cloud, inside the ice region.", "",
          "| Step | Frames | Posed | ATE median (mm) | Depth | Coverage | Completeness 2 mm | Accuracy 2 mm | "
          "Total time (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['step']} | {r['frames']} | {r['frames_posed']} | {fmt(r.get('ate_median_mm'), '.2f')} | "
                  f"{labels[r['source']]} | {fmt(r.get('coverage'), '.1%')} | {fmt(r.get('completeness_2mm'), '.1%')} | "
                  f"{fmt(r.get('accuracy_2mm'), '.1%')} | {fmt(r.get('total_s'), '.0f')} |")
    with open(os.path.join(p["out"], "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))

    costs = ["# Stage costs", "", "Wall time (s) and peak memory per stage. Depth: start-up plus the subset's own "
             "frames from the every-frame run; its peaks are that run's.", "",
             "| Step | Frames | SfM s (features / match / map) | SfM RAM MB | SfM GPU MiB | Depth | Depth s | "
             "Depth RAM MB | Depth GPU MiB | TSDF s | TSDF RAM MB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sub = " / ".join(fmt(r.get(f"sfm_{k}_s"), ".0f") for k in ("features", "match", "map"))
        costs.append(f"| {r['step']} | {r['frames']} | {fmt(r.get('sfm_s'), '.0f')} ({sub}) | "
                     f"{fmt(r.get('sfm_ram_mb'), '.0f')} | {fmt(r.get('sfm_gpu_mib'), '.0f')} | {labels[r['source']]} | "
                     f"{fmt(r.get('depth_s'), '.0f')} | {fmt(r.get('depth_ram_mb'), '.0f')} | "
                     f"{fmt(r.get('depth_gpu_mib'), '.0f')} | {fmt(r.get('tsdf_s'), '.0f')} | "
                     f"{fmt(r.get('tsdf_ram_mb'), '.0f')} |")
    with open(os.path.join(p["out"], "stage_costs.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(costs) + "\n")

    by = {(r["step"], r["source"]): r for r in rows}
    fs, sg = "foundationstereo", "sgbm_tuned"
    tex = ["% Made by src/experiments/frame_count/analyse_frame_count.py; do not edit.",
           r"\begin{table}[H]", r"\centering",
           rf"\caption{{Frame-count sweep, {cfg['recording']['title'].lower()}: COLMAP SfM re-run on every $N$th frame, "
           r"then TSDF fusion of FoundationStereo (FS) and tuned-SGBM depth. ATE against the AprilGrid poses; "
           r"coverage of the ice region's 1~mm cells; completeness = share of the all-frames cloud within 2~mm; "
           r"time = SfM + depth + TSDF.}",
           rf"\label{{tab:frame_count_{cfg['recording']['name']}}}", r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
           r" & & & & \multicolumn{2}{c}{Coverage (\%)} & \multicolumn{2}{c}{Complete. (\%)} & "
           r"\multicolumn{2}{c}{Time (s)} \\",
           r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
           r"$N$ & Frames & Posed & ATE (mm) & FS & SGBM & FS & SGBM & FS & SGBM \\", r"\midrule"]
    for n in cfg["steps"]:
        a, b = by.get((n, fs)), by.get((n, sg))
        if a is None:
            continue
        pct = lambda r, k: fmt(100 * r[k] if r and r.get(k) is not None else None, ".1f")  # noqa: E731
        tex.append(rf"{n} & {a['frames']} & {a['frames_posed']} & {fmt(a.get('ate_median_mm'), '.1f')} & "
                   rf"{pct(a, 'coverage')} & {pct(b, 'coverage')} & {pct(a, 'completeness_2mm')} & "
                   rf"{pct(b, 'completeness_2mm')} & {fmt(a.get('total_s'), '.0f')} & "
                   rf"{fmt(b.get('total_s') if b else None, '.0f')} \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(p["out"], "table_frame_count.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")


# ====== FIGURES ======
def style(ax):
    ax.grid(True, color="k", alpha=0.12, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def frames_axis(ax, frames):
    ax.set_xscale("log")
    ax.set_xticks(frames)
    ax.set_xticklabels([str(f) for f in frames])
    ax.minorticks_off()
    ax.set_xlabel("frames used")


def series(rows, source, key):
    pts = sorted((r["frames"], r[key]) for r in rows if r["source"] == source and r.get(key) is not None)
    return [x for x, _ in pts], [y for _, y in pts]


def quality_figure(cfg, p, rows):
    frames = sorted({r["frames"] for r in rows if r.get("frames")})
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    one = [r for r in rows if r["source"] == next(iter(cfg["depth"]))]
    x = [r["frames"] for r in one]
    axes[0].plot(x, [100 * r.get("registration", r["frames_posed"] / max(r["frames"] or 1, 1)) for r in one],
                 "-o", color=MUTED, lw=2, ms=6)
    axes[0].set_ylabel("frames posed by SfM (%)")
    axes[0].set_ylim(0, 105)
    axes[1].plot(*series(one, next(iter(cfg["depth"])), "ate_median_mm"), "-o", color=MUTED, lw=2, ms=6)
    axes[1].set_ylabel("pose ATE median (mm)")
    axes[1].set_ylim(bottom=0)
    for ax, key, label in ((axes[2], "coverage", "coverage of ice region (%)"),
                           (axes[3], "completeness_2mm", "completeness vs all frames, 2 mm (%)")):
        for source, spec in cfg["depth"].items():
            xs, ys = series(rows, source, key)
            ax.plot(xs, [100 * y for y in ys], "-o", color=SOURCE_COLOURS[source], lw=2, ms=6, label=spec["label"])
        ax.set_ylabel(label)
        ax.set_ylim(0, 102)
        ax.legend(fontsize=8, loc="lower right", frameon=False)
    for ax in axes:
        style(ax)
        frames_axis(ax, frames)
    axes[0].set_title("Poses (COLMAP SfM)", loc="left", fontsize=10)
    axes[2].set_title("Reconstruction", loc="left", fontsize=10)
    fig.suptitle(f"{cfg['recording']['title']}: quality against frames used", x=0.01, ha="left", fontsize=11)
    path = os.path.join(p["out"], "quality_vs_frames.png")
    fig.savefig(path, dpi=cfg["evaluation"]["figure_dpi"])
    plt.close(fig)
    print(f"-> {common.rel(path)}")


def pose_figure(cfg, p, rows):
    """The SfM poses alone: share of frames posed, and ATE (median, p95, max) against the tag poses."""
    one = sorted((r for r in rows if r["source"] == next(iter(cfg["depth"])) and r.get("frames")),
                 key=lambda r: r["frames"])
    frames = [r["frames"] for r in one]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(frames, [100 * r.get("registration", 0) for r in one], "-o", color=STAGE_COLOURS["sfm"], lw=2, ms=6)
    axes[0].set_ylabel("frames posed by COLMAP SfM (%)")
    axes[0].set_ylim(0, 105)
    for key, label, ls in (("ate_median_mm", "median", "-"), ("ate_p95_mm", "95th percentile", "--"),
                           ("ate_max_mm", "max", ":")):
        xs = [r["frames"] for r in one if r.get(key) is not None]
        axes[1].plot(xs, [r[key] for r in one if r.get(key) is not None], ls, marker="o", color=STAGE_COLOURS["sfm"],
                     lw=2, ms=5, label=label)
    axes[1].set_ylabel("camera position error, ATE (mm)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        style(ax)
        frames_axis(ax, frames)
    fig.suptitle(f"{cfg['recording']['title']}: COLMAP poses against frames used (reference: AprilGrid tags)",
                 x=0.01, ha="left", fontsize=11)
    path = os.path.join(p["out"], "pose_vs_frames.png")
    fig.savefig(path, dpi=cfg["evaluation"]["figure_dpi"])
    plt.close(fig)
    print(f"-> {common.rel(path)}")


def time_figure(cfg, p, rows):
    sources = list(cfg["depth"])
    fig, axes = plt.subplots(1, len(sources), figsize=(6.5 * len(sources), 3.8), sharey=True, constrained_layout=True)
    for ax, source in zip(np.atleast_1d(axes), sources):
        rs = sorted((r for r in rows if r["source"] == source and r.get("total_s") is not None), key=lambda r: r["frames"])
        labels = [str(r["frames"]) for r in rs]
        bottom = np.zeros(len(rs))
        for stage in ("sfm", "depth", "tsdf"):
            values = np.array([r[f"{stage}_s"] / 60 for r in rs])
            ax.bar(labels, values, bottom=bottom, color=STAGE_COLOURS[stage], width=0.7, label=STAGE_LABELS[stage],
                   edgecolor="white", linewidth=1)
            bottom += values
        for k, total in enumerate(bottom):
            ax.text(k, total, f"{total:.1f}", ha="center", va="bottom", fontsize=8, color="#1f1f1d")
        ax.set_title(f"{cfg['depth'][source]['label']} depth", loc="left", fontsize=10)
        ax.set_xlabel("frames used")
        style(ax)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("time per reconstruction (min)")
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{cfg['recording']['title']}: pipeline time by stage", x=0.01, ha="left", fontsize=11)
    path = os.path.join(p["out"], "time_vs_frames.png")
    fig.savefig(path, dpi=cfg["evaluation"]["figure_dpi"])
    plt.close(fig)
    print(f"-> {common.rel(path)}")


def memory_figure(cfg, p, rows):
    frames = sorted({r["frames"] for r in rows if r.get("frames")})
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    lines = [("SfM", "sfm", next(iter(cfg["depth"])), STAGE_COLOURS["sfm"], "-")]
    for source, spec in cfg["depth"].items():
        lines += [(f"Depth, {spec['label']}", "depth", source, STAGE_COLOURS["depth"],
                   "-" if source == "foundationstereo" else "--"),
                  (f"TSDF, {spec['label']}", "tsdf", source, STAGE_COLOURS["tsdf"],
                   "-" if source == "foundationstereo" else "--")]
    for ax, unit, ylabel, scale in ((axes[0], "ram_mb", "peak RAM (GB)", 1 / 1024),
                                    (axes[1], "gpu_mib", "peak GPU memory (GB)", 1 / 1024)):
        for label, stage, source, colour, ls in lines:
            xs, ys = series(rows, source, f"{stage}_{unit}")
            if not xs or (unit == "gpu_mib" and stage == "tsdf"):
                continue
            ax.plot(xs, [y * scale for y in ys], ls, marker="o", color=colour, lw=2, ms=5, label=label)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        style(ax)
        frames_axis(ax, frames)
        ax.legend(fontsize=8, frameon=False)
    axes[1].axhline(8, color=MUTED, lw=1, ls=":")
    axes[1].text(frames[0], 8, " GPU total (8 GB)", fontsize=8, color=MUTED, va="bottom")
    axes[1].set_ylim(0, 9)
    fig.suptitle(f"{cfg['recording']['title']}: peak memory by stage (process tree; GPU excludes other programs)",
                 x=0.01, ha="left", fontsize=11)
    path = os.path.join(p["out"], "memory_by_stage.png")
    fig.savefig(path, dpi=cfg["evaluation"]["figure_dpi"])
    plt.close(fig)
    print(f"-> {common.rel(path)}")


def oblique_images(cfg, p, steps, sources, azimuth=225.0, elevation=30.0, pixel_mm=0.6, splat=1):
    """({(step, source): rendered image or None}, (low, high)): every subset's cloud over the ice region from one
    oblique view (oblique_view.render), with one colour range and extent for all, both set from the first step's
    clouds (heights 10th-99.5th percentile, as oblique_view.py)."""
    ev = cfg["evaluation"]
    side = compare_fusion.camera_side(p)
    clouds = {(n, s): region_cloud(os.path.join(p["step_work"](n), f"tsdf_{s}", "cloud_board.ply"), ev["region"], side)
              for n in steps for s in sources}
    ref = np.concatenate([c for (n, _), c in clouds.items() if n == steps[0] and c is not None])
    low, high = np.percentile(ref[:, 2], (10, 99.5))
    _, extent = oblique_view.render(ref[::50], np.zeros((len(ref[::50]), 3)), azimuth, elevation, pixel_mm, 0)
    cmap = colormaps["turbo"]
    images = {}
    for key, xyz in clouds.items():
        xyz = xyz[xyz[:, 2] >= low - 0.2 * (high - low)] if xyz is not None else None
        if xyz is None or not len(xyz):
            images[key] = None
            continue
        colours = cmap(np.clip((xyz[:, 2] - low) / (high - low), 0, 1))[:, :3] * oblique_view.shading(xyz)[:, None]
        images[key], _ = oblique_view.render(xyz, colours, azimuth, elevation, pixel_mm, splat, extent)
    # One crop for all of them, to the rows and columns any image draws on (plus a small margin): the extent comes
    # from the first step's points before the far-below cut above, so it leaves white space; a shared crop keeps
    # every panel at the same scale.
    drawn = [np.any(im < 1, axis=2) for im in images.values() if im is not None]
    if drawn:
        mask = np.logical_or.reduce(drawn)
        r, c = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        pad = 5
        r0, r1 = max(r[0] - pad, 0), min(r[-1] + pad + 1, mask.shape[0])
        c0, c1 = max(c[0] - pad, 0), min(c[-1] + pad + 1, mask.shape[1])
        images = {k: im[r0:r1, c0:c1] if im is not None else None for k, im in images.items()}
    return images, (low, high)


def show_image(ax, image):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if image is None:
        ax.text(0.5, 0.5, "no reconstruction", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    else:
        ax.imshow(image, interpolation="lanczos")


def oblique_figure(cfg, p, rows):
    """Each subset's cloud from one oblique view, a row per depth source in figures.oblique_sources."""
    ev = cfg["evaluation"]
    sources = {s: cfg["depth"][s] for s in cfg.get("figures", {}).get("oblique_sources", cfg["depth"])}
    shown = cfg.get("figures", {}).get("oblique_steps", cfg["steps"])
    steps = [n for n in shown if any(r["step"] == n and r.get("points") for r in rows)]
    images, (low, high) = oblique_images(cfg, p, steps, sources)
    frames = {r["step"]: r["frames"] for r in rows}
    cmap = colormaps["turbo"]
    shape = next(im.shape for im in images.values() if im is not None)
    fig, axes = plt.subplots(len(sources), len(steps),
                             figsize=(3.2 * len(steps) + 1.2, 3.2 * shape[0] / shape[1] * len(sources) + 0.5),
                             squeeze=False, constrained_layout=True)
    for i, (source, spec) in enumerate(sources.items()):
        for j, n in enumerate(steps):
            ax = axes[i, j]
            show_image(ax, images[(n, source)])
            if i == 0:
                ax.set_title(f"{frames.get(n)} frames (every {n})", fontsize=10)
            if j == 0 and len(sources) > 1:
                ax.set_ylabel(spec["label"], fontsize=10)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(low, high), cmap), ax=axes, shrink=0.8,
                       pad=0.01)
    bar.set_label("height towards camera (mm)", fontsize=9)
    path = os.path.join(p["out"], "oblique_by_frames.png")
    fig.savefig(path, dpi=ev["figure_dpi"])
    plt.close(fig)
    print(f"-> {common.rel(path)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    p = paths(cfg)
    steps = load_steps(cfg, p)
    rows = score(cfg, p, steps)
    with open(os.path.join(p["out"], "scores.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    write_tables(cfg, p, rows)
    quality_figure(cfg, p, rows)
    pose_figure(cfg, p, rows)
    time_figure(cfg, p, rows)
    memory_figure(cfg, p, rows)
    oblique_figure(cfg, p, rows)


if __name__ == "__main__":
    main()
