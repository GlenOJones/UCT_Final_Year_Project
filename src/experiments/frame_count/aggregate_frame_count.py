"""The frame-count sweep over every ice object: reads each recording's scores.json (analyse_frame_count.py) and
averages over the recordings at each step N (every N'th frame).

The passes differ in length (131 to 231 frames), so a step keeps a different number of frames per object; the
average is taken per step, i.e. per frame spacing along the pass, and the frame counts are given as a range.
Time depends on the number of frames rather than on the spacing, so it is plotted against frames used, every
recording's runs pooled, with a straight-line fit giving seconds per frame.

    src/venv/bin/python src/experiments/frame_count/aggregate_frame_count.py [--configs ...]

Writes, under results/<out_dir>/all_ice/:
    summary.csv / summary.md      per step: mean and range over the recordings of every measure
    table_frame_count_all.tex     the report table (LaTeX)
    stage_costs.md / table_stage_costs.tex   per stage over every run of every object: time per frame (straight-
                                  line fit), peak RAM, peak GPU memory, mean CPU (cores busy) and GPU utilisation
    pose_vs_step.png              frames posed and ATE median against step, a line per object and their mean
    quality_vs_step.png           coverage and completeness against step, a line per object and their mean
    time_vs_frames.png            total time against frames used, every run, with a fit per depth source
    memory_by_stage.png           peak RAM and GPU memory per stage, the largest over the recordings
    oblique_grid.png              a row per object, a column per step: FoundationStereo's cloud from one oblique
                                  view; each row has its own colour range and extent (set from its all-frames cloud)
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fusion"))
import common  # noqa: E402
from oblique_view import place_row_labels  # noqa: E402
from analyse_frame_count import MUTED, SOURCE_COLOURS, STAGE_COLOURS, fmt, oblique_images, show_image, style  # noqa: E402
from run_frame_count import paths  # noqa: E402

CONFIGS = ("config.json", "config_cubes.json", "config_wave.json", "config_overhang.json")
# One colour per object, the reference palette's categorical slots in order.
OBJECT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#1f1f1d"


def load(configs):
    """[(cfg, rows)] for every config whose scores.json exists."""
    runs = []
    for path in configs:
        cfg = common.load_config(path)
        scores = os.path.join(paths(cfg)["out"], "scores.json")
        if not os.path.exists(scores):
            print(f"{cfg['recording']['name']}: no scores.json (run analyse_frame_count.py), left out")
            continue
        with open(scores) as fh:
            runs.append((cfg, json.load(fh)))
    return runs


def value(rows, step, source, key):
    for r in rows:
        if r["step"] == step and r["source"] == source and r.get(key) is not None:
            return r[key]
    return None


def per_step(runs, steps, source, key):
    """{step: [value per recording]} (recordings without a value left out)."""
    return {n: [v for _, rows in runs if (v := value(rows, n, source, key)) is not None] for n in steps}


# ====== TABLES ======
def write_tables(runs, steps, out):
    fs, sg = "foundationstereo", "sgbm_tuned"
    measures = [("frames", fs, "frames"), ("registration", fs, "posed share"), ("ate_median_mm", fs, "ATE median mm"),
                ("ate_max_mm", fs, "ATE max mm"), ("coverage", fs, "coverage FS"), ("coverage", sg, "coverage SGBM"),
                ("completeness_2mm", fs, "completeness 2 mm FS"), ("completeness_2mm", sg, "completeness 2 mm SGBM"),
                ("total_s", fs, "time FS s"), ("total_s", sg, "time SGBM s")]
    rows = []
    for n in steps:
        row = {"step": n, "recordings": len([1 for _, r in runs if value(r, n, fs, "frames")])}
        for key, source, label in measures:
            values = per_step(runs, [n], source, key)[n]
            row[f"{label} mean"] = float(np.mean(values)) if values else None
            row[f"{label} min"] = float(np.min(values)) if values else None
            row[f"{label} max"] = float(np.max(values)) if values else None
        rows.append(row)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = ", ".join(cfg["recording"]["title"] for cfg, _ in runs)
    md = [f"# Frame-count sweep, all ice objects ({len(runs)}: {names})", "",
          "Mean over the objects at each step, range in brackets.", "",
          "| Every N | Frames | Posed | ATE median (mm) | ATE max (mm) | Coverage FS / SGBM | Completeness 2 mm FS / SGBM "
          "| Time FS / SGBM (min) |", "|---|---|---|---|---|---|---|---|"]
    pct = lambda r, k: fmt(100 * r[k] if r[k] is not None else None, ".0f")  # noqa: E731
    for r in rows:
        md.append(f"| {r['step']} | {fmt(r['frames min'], '.0f')}-{fmt(r['frames max'], '.0f')} | "
                  f"{pct(r, 'posed share mean')}% (min {pct(r, 'posed share min')}%) | "
                  f"{fmt(r['ATE median mm mean'], '.1f')} ({fmt(r['ATE median mm min'], '.1f')}-"
                  f"{fmt(r['ATE median mm max'], '.1f')}) | {fmt(r['ATE max mm max'], '.1f')} | "
                  f"{pct(r, 'coverage FS mean')}% / {pct(r, 'coverage SGBM mean')}% | "
                  f"{pct(r, 'completeness 2 mm FS mean')}% / {pct(r, 'completeness 2 mm SGBM mean')}% | "
                  f"{fmt(r['time FS s mean'] / 60 if r['time FS s mean'] else None, '.1f')} / "
                  f"{fmt(r['time SGBM s mean'] / 60 if r['time SGBM s mean'] else None, '.1f')} |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))

    tex = ["% Made by src/experiments/frame_count/aggregate_frame_count.py; do not edit.",
           r"\begin{table}[H]", r"\centering",
           rf"\caption{{Frame-count sweep over the {len(runs)} shaded ice objects: mean over the objects at each frame "
           r"spacing (every $N$th frame), with the range of frame counts. Posed: share of frames COLMAP registered "
           r"(lowest over the objects); ATE: median camera position error against the AprilGrid poses; coverage: share "
           r"of the ice region's 1~mm cells with a point; completeness: share of the all-frames reconstruction within "
           r"2~mm; time: SfM + depth + TSDF.}",
           r"\label{tab:frame_count_all}", r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
           r" & & & & \multicolumn{2}{c}{Coverage (\%)} & \multicolumn{2}{c}{Complete. (\%)} & "
           r"\multicolumn{2}{c}{Time (min)} \\",
           r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
           r"$N$ & Frames & Posed (\%) & ATE (mm) & FS & SGBM & FS & SGBM & FS & SGBM \\", r"\midrule"]
    for r in rows:
        tex.append(rf"{r['step']} & {fmt(r['frames min'], '.0f')}--{fmt(r['frames max'], '.0f')} & "
                   rf"{pct(r, 'posed share min')} & {fmt(r['ATE median mm mean'], '.1f')} & "
                   rf"{pct(r, 'coverage FS mean')} & {pct(r, 'coverage SGBM mean')} & "
                   rf"{pct(r, 'completeness 2 mm FS mean')} & {pct(r, 'completeness 2 mm SGBM mean')} & "
                   rf"{fmt(r['time FS s mean'] / 60 if r['time FS s mean'] else None, '.1f')} & "
                   rf"{fmt(r['time SGBM s mean'] / 60 if r['time SGBM s mean'] else None, '.1f')} \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out, "table_frame_count_all.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")


def stage_records(cfg):
    """[(stage, frames, cost record)] of one recording: SfM and TSDF per step, depth once (every frame)."""
    p = paths(cfg)
    records = []
    with open(os.path.join(p["out"], "depth_costs.json")) as fh:
        for source, record in json.load(fh).items():
            with open(os.path.join(p["depth"](source), "depth_frames.json")) as dj:
                records.append((f"depth_{source}", len(json.load(dj)["per_frame"]), record))
    for n in cfg["steps"]:
        path = os.path.join(p["step_out"](n), "run.json")
        if os.path.exists(path):
            with open(path) as fh:
                run = json.load(fh)
            for stage, record in run["stages"].items():
                if record.get("returncode") == 0:
                    records.append((stage, run["frames"], record))
            for sub, seconds in run.get("sfm_seconds", {}).items():
                records.append((f"sfm_{sub}", run["frames"], {"wall_s": seconds}))
    return records


def write_stage_costs(runs, out):
    records = [r for cfg, _ in runs for r in stage_records(cfg)]
    stages = [("sfm", "COLMAP SfM (total)", "CPU + GPU"), ("sfm_features", "  SIFT features", "GPU"),
              ("sfm_match", "  matching", "GPU"), ("sfm_map", "  mapping (bundle adjustment)", "CPU"),
              ("depth_foundationstereo", "Depth, FoundationStereo", "GPU"), ("depth_sgbm_tuned", "Depth, SGBM (tuned)", "CPU"),
              ("tsdf_foundationstereo", "TSDF, FoundationStereo depth", "CPU"),
              ("tsdf_sgbm_tuned", "TSDF, SGBM depth", "CPU")]
    rows = []
    for key, label, device in stages:
        rs = [(f, r) for k, f, r in records if k == key]
        if not rs:
            continue
        frames = np.array([f for f, _ in rs], float)
        seconds = np.array([r["wall_s"] for _, r in rs])
        per_frame = np.polyfit(frames, seconds, 1)[0] if len(set(frames)) > 1 else float(np.median(seconds / frames))
        get = lambda name: [r[name] for _, r in rs if r.get(name) is not None]  # noqa: E731
        rows.append({"label": label, "device": device, "runs": len(rs), "s_per_frame": per_frame,
                     "ram_gb": (min(get("peak_tree_rss_mb")) / 1024, max(get("peak_tree_rss_mb")) / 1024)
                     if get("peak_tree_rss_mb") else None,
                     "gpu_gb": max(get("peak_gpu_mib")) / 1024 if get("peak_gpu_mib") else None,
                     "cores": np.mean(get("mean_cpu_percent")) / 100 if get("mean_cpu_percent") else None,
                     "gpu_util": np.mean(get("mean_gpu_util_percent")) if get("mean_gpu_util_percent") else None})
    md = ["# Stage costs, every run of every ice object", "",
          "Time per frame: slope of a straight-line fit of wall time against frames. RAM: peak over the stage's process "
          "tree (lowest to highest run). CPU: mean cores busy (100% = one core; the laptop has 28 threads).", "",
          "| Stage | Device | Runs | Time per frame (s) | Peak RAM (GB) | Peak GPU memory (GB) | Mean CPU (cores) | "
          "Mean GPU use (%) |", "|---|---|---|---|---|---|---|---|"]
    tex = ["% Made by src/experiments/frame_count/aggregate_frame_count.py; do not edit.",
           r"\begin{table}[H]", r"\centering",
           r"\caption{Cost of each pipeline stage over every run of the four ice objects (RTX 5060 Laptop GPU, 8~GB; "
           r"30~GB RAM; 28 CPU threads). Time per frame: slope of wall time against frames used. RAM: peak over the "
           r"stage's processes, smallest to largest run. CPU: mean number of cores busy.}",
           r"\label{tab:frame_count_stage_costs}", r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Stage & Device & Time/frame (s) & Peak RAM (GB) & Peak GPU (GB) & CPU (cores) \\", r"\midrule"]
    for r in rows:
        ram = f"{r['ram_gb'][0]:.1f}-{r['ram_gb'][1]:.1f}" if r["ram_gb"] else "-"
        md.append(f"| {r['label'].strip()} | {r['device']} | {r['runs']} | {r['s_per_frame']:.2f} | {ram} | "
                  f"{fmt(r['gpu_gb'], '.1f', '-')} | {fmt(r['cores'], '.1f', '-')} | {fmt(r['gpu_util'], '.0f', '-')} |")
        indent = r"\quad " if r["label"].startswith("  ") else ""
        tex.append(rf"{indent}{r['label'].strip()} & {r['device']} & {r['s_per_frame']:.2f} & {ram.replace('-', '--')} & "
                   rf"{fmt(r['gpu_gb'], '.1f')} & {fmt(r['cores'], '.1f')} \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out, "stage_costs.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out, "table_stage_costs.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")
    print("\n".join(md))


# ====== FIGURES ======
def step_axis(ax, steps):
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(n) for n in steps])
    ax.minorticks_off()
    ax.set_xlabel("every N'th frame used")


def object_lines(ax, runs, steps, source, key, scale=1.0, ls="-", mean_label="mean", labels=True):
    for k, (cfg, rows) in enumerate(runs):
        xs = [n for n in steps if value(rows, n, source, key) is not None]
        ax.plot(xs, [value(rows, n, source, key) * scale for n in xs], ls, marker="o", ms=4, lw=1.2, alpha=0.75,
                color=OBJECT_COLOURS[k % len(OBJECT_COLOURS)], label=cfg["recording"]["title"] if labels else None)
    means = per_step(runs, steps, source, key)
    xs = [n for n in steps if means[n]]
    ax.plot(xs, [np.mean(means[n]) * scale for n in xs], ls, marker="s", ms=6, lw=2.6, color=INK, label=mean_label)


def save(fig, out, name, dpi):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"-> {common.rel(path)}")


def pose_figure(runs, steps, out, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    object_lines(axes[0], runs, steps, "foundationstereo", "registration", 100)
    axes[0].set_ylabel("frames posed by COLMAP SfM (%)")
    axes[0].set_ylim(0, 105)
    object_lines(axes[1], runs, steps, "foundationstereo", "ate_median_mm", labels=False)
    axes[1].set_ylabel("camera position error, ATE median (mm)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        style(ax)
        step_axis(ax, steps)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("COLMAP poses against frame spacing, every ice object (reference: AprilGrid tags)", x=0.01, ha="left",
                 fontsize=11)
    save(fig, out, "pose_vs_step.png", dpi)


def quality_figure(runs, steps, out, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, label in ((axes[0], "coverage", "coverage of ice region (%)"),
                           (axes[1], "completeness_2mm", "completeness vs all frames, 2 mm (%)")):
        object_lines(ax, runs, steps, "foundationstereo", key, 100, mean_label="mean, FoundationStereo",
                     labels=ax is axes[0])
        means = per_step(runs, steps, "sgbm_tuned", key)
        xs = [n for n in steps if means[n]]
        ax.plot(xs, [100 * np.mean(means[n]) for n in xs], "--", marker="s", ms=6, lw=2.2, color=MUTED,
                label="mean, SGBM (tuned)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 102)
        style(ax)
        step_axis(ax, steps)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Reconstruction against frame spacing, every ice object (lines per object: FoundationStereo)",
                 x=0.01, ha="left", fontsize=11)
    save(fig, out, "quality_vs_step.png", dpi)


def time_figure(runs, out, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for source, colour in SOURCE_COLOURS.items():
        pts = np.array([(r["frames"], r["total_s"] / 60) for _, rows in runs for r in rows
                        if r["source"] == source and r.get("total_s") is not None])
        if not len(pts):
            continue
        slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
        label = runs[0][0]["depth"][source]["label"]
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color=colour, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{label}: {60 * slope:.1f} s per frame")
        x = np.linspace(0, pts[:, 0].max(), 50)
        ax.plot(x, slope * x + intercept, color=colour, lw=1.5, alpha=0.8)
    ax.set_xlabel("frames used")
    ax.set_ylabel("time per reconstruction (min)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Pipeline time against frames used, every ice object", x=0.01, ha="left", fontsize=11)
    save(fig, out, "time_vs_frames.png", dpi)


def memory_figure(runs, out, dpi):
    """Largest peak over the recordings, per stage, against frames used (every run plotted)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    series = [("SfM", "sfm", "foundationstereo", STAGE_COLOURS["sfm"], "o"),
              ("Depth, FoundationStereo", "depth", "foundationstereo", STAGE_COLOURS["depth"], "o"),
              ("Depth, SGBM (tuned)", "depth", "sgbm_tuned", STAGE_COLOURS["depth"], "^"),
              ("TSDF, FoundationStereo", "tsdf", "foundationstereo", STAGE_COLOURS["tsdf"], "o"),
              ("TSDF, SGBM (tuned)", "tsdf", "sgbm_tuned", STAGE_COLOURS["tsdf"], "^")]
    for ax, unit, ylabel in ((axes[0], "ram_mb", "peak RAM (GB)"), (axes[1], "gpu_mib", "peak GPU memory (GB)")):
        for label, stage, source, colour, marker in series:
            if unit == "gpu_mib" and stage == "tsdf":
                continue
            pts = [(r["frames"], r[f"{stage}_{unit}"] / 1024) for _, rows in runs for r in rows
                   if r["source"] == source and r.get(f"{stage}_{unit}") is not None]
            if pts:
                ax.scatter(*zip(*pts), s=26, marker=marker, color=colour, edgecolor="white", linewidth=0.8,
                           label=label, zorder=3)
        ax.set_xlabel("frames used")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        style(ax)
        ax.legend(fontsize=8, frameon=False, loc="center right")
    axes[1].axhline(8, color=MUTED, lw=1, ls=":")
    axes[1].text(2, 8, "GPU total (8 GB)", fontsize=8, color=MUTED, va="bottom")
    axes[1].set_ylim(0, 9)
    axes[1].legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle("Peak memory by stage, every ice object (process tree; GPU excludes other programs)", x=0.01,
                 ha="left", fontsize=11)
    save(fig, out, "memory_by_stage.png", dpi)


def oblique_grid(runs, steps, out, dpi, source="foundationstereo"):
    """A row per object, a column per step; each row scaled on its own (colour bar per row), as in each object's
    oblique_by_frames.png."""
    cmap = colormaps["turbo"]
    fig, axes = plt.subplots(len(runs), len(steps), figsize=(3.0 * len(steps) + 1.6, 2.7 * len(runs)), squeeze=False,
                             constrained_layout=True)
    for i, (cfg, rows) in enumerate(runs):
        images, (low, high) = oblique_images(cfg, paths(cfg), steps, {source: cfg["depth"][source]})
        frames = {r["step"]: r["frames"] for r in rows}
        for j, n in enumerate(steps):
            show_image(axes[i, j], images.get((n, source)))
            axes[i, j].set_title(f"{frames.get(n)} frames" + (f" (every {n})" if i == 0 else ""), fontsize=10)
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(low, high), cmap), ax=axes[i, :],
                           shrink=0.85, pad=0.01)
        bar.set_label("height towards\ncamera (mm)", fontsize=8)
        bar.ax.tick_params(labelsize=8)
    place_row_labels(fig, axes, [cfg["recording"]["title"].split(",")[0].replace("Ice ", "").capitalize()
                                 for cfg, _ in runs], 13)
    save(fig, out, "oblique_grid.png", dpi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=[os.path.join(HERE, c) for c in CONFIGS])
    args = ap.parse_args(argv)
    runs = load(args.configs)
    if not runs:
        raise SystemExit("no recording has scores yet")
    cfg = runs[0][0]
    steps = cfg["steps"]
    out = os.path.join(common.project_path(cfg["out_dir"]), "all_ice")
    os.makedirs(out, exist_ok=True)
    dpi = cfg["evaluation"]["figure_dpi"]
    write_tables(runs, steps, out)
    write_stage_costs(runs, out)
    pose_figure(runs, steps, out, dpi)
    quality_figure(runs, steps, out, dpi)
    time_figure(runs, out, dpi)
    memory_figure(runs, out, dpi)
    oblique_grid(runs, cfg.get("figures", {}).get("oblique_steps", steps), out, dpi)


if __name__ == "__main__":
    main()
