"""The pose comparison table: AprilGrid, COLMAP and ICP on each depth source, on every recording of the config.

Reads only what run_poses.py and detection_figures.py saved. Per method and recording:

  frames posed   frames it posed (ICP: not lost) / frames it was given (every frame_step'th pair)
  ATE            after ONE rigid fit onto the AprilGrid poses: camera-centre error RMSE and max (mm), orientation
                 RMSE (deg). Recordings without tags have no reference, so no ATE.
  RPE            error of the motion over every rpe_distance_mm of reference travel: median (% and deg)
  runtime        seconds per frame for the pose step, and separately for the depth maps ICP needs
  grip           median per frame of what the method holds on to (see detection_figures.py)

AprilGrid is the reference, so its ATE/RPE are zero by definition and are not shown; its runtime is tag_poses.py
over every frame of the recording (detection, triangulation, layout fit). The scoring itself is
evaluate_poses.evaluate, the same as experiment 4.

    src/venv/bin/python src/experiments/pose_estimation/compare_poses.py --config src/experiments/pose_estimation/config_comparison.json

Writes results/<out_dir>/: summary.md (table + the mode each method ran in), summary.csv, summary.tex; and per
recording <recording>/evaluation/: metrics.json, trajectories.png (experiment 4's two-panel figure), and as
single figures for the report:

  paths.png  the camera paths in a TRACK frame: along the pass (the paths' main direction), across it, and out
             of the plane of travel, with the across/out-of-plane axes exaggerated so the methods separate (at
             true scale they lie on top of each other). With tags: every method after one rigid fit onto
             AprilGrid, which is drawn too. Without tags (sun) there is no reference, so the figure shows only
             how the methods agree, not which is right: the smoothest path (smallest p95 frame-to-frame step)
             is the anchor and every other path is rigidly fitted onto it over the frames where the two agree
             (iteratively: frames within 3x the median distance, at least 15 mm), so a method that is badly
             wrong over part of the pass does not drag the fit. Frame-to-frame steps above 40 mm, impossible
             for the trolley (~6 mm a frame), are drawn as dotted jumps.
  path_2d.png  the same paths from above (along vs across track) at TRUE scale, start and end marked.
  ate.png    (tagged recordings only) ATE per frame along the reference travel.

and per cfg["combined_paths"] entry, results/<out_dir>/<file>: those recordings' top views side by side at one scale.
"""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "03_Reconstruction"))
import common  # noqa: E402
import pose_files  # noqa: E402
from evaluate_poses import evaluate, plot, pose_points, umeyama  # noqa: E402
from run_poses import frame_step, has_tags, recording_paths  # noqa: E402
from stereo_rig import find_frame_pairs, invert  # noqa: E402

# Validated categorical order (dataviz reference palette, slots 1-3); the reference is neutral ink.
COLOURS = {"colmap": "#2a78d6", "icp_fs": "#eb6834", "icp_sgbm": "#1baf7a", "icp_aqua": "#eda100"}
RESTART = "_restart"   # suffix of a restarted run (run_poses.restart_estimator): same colour and line as the method
FIGURE_REFERENCE = "AprilTag"   # the reference's name in the figures (the tables keep cfg["reference"]["label"])


def colour(name):
    return COLOURS.get(name[:-len(RESTART)] if name.endswith(RESTART) else name)


def restart_start(p, name):
    """Start frame of the kept restart of estimator `name`, or None if it has none."""
    record = os.path.join(p["out"], "poses", f"{name}_restarts.json")
    if not os.path.exists(os.path.join(p["out"], "poses", name + RESTART + ".yaml")) or not os.path.exists(record):
        return None
    with open(record, encoding="utf-8") as fh:
        return json.load(fh)["kept_start"]
REFERENCE_INK = "#52514e"

COLUMNS = [("posed", "Frames posed"), ("ate_rmse_mm", "ATE RMSE (mm)"), ("ate_max_mm", "ATE max (mm)"),
           ("ate_rmse_deg", "ATE RMSE (deg)"), ("rpe_median_%", "RPE median (%)"), ("rpe_median_deg", "RPE median (deg)"),
           ("s_per_frame", "s/frame (pose)"), ("depth_s_per_frame", "s/frame (depth)"), ("grip", "Grip (median/frame)")]
FORMATS = {"ate_rmse_mm": ".2f", "ate_max_mm": ".2f", "ate_rmse_deg": ".3f", "rpe_median_%": ".2f",
           "rpe_median_deg": ".3f", "s_per_frame": ".2f", "depth_s_per_frame": ".2f", "grip": ",.0f"}


def fmt(row, key):
    value = row.get(key)
    if key == "posed":
        return f"{row['posed']}/{row['frames']}"
    return "-" if value is None else format(value, FORMATS[key])


def per_frame(seconds, frames):
    return seconds / frames if seconds and frames else None


def score_recording(cfg, recording):
    p = recording_paths(cfg, recording)
    names = [n for n, _, _ in find_frame_pairs(common.project_path(recording["frames"]))][::frame_step(cfg, recording)]
    grip_path = os.path.join(p["out"], "detections", "grip.json")
    grip = {}
    if os.path.exists(grip_path):
        with open(grip_path, encoding="utf-8") as fh:
            grip = json.load(fh)["median"]
    else:
        print(f"  {recording['name']}: no {common.rel(grip_path)} (run detection_figures.py), grip left blank")

    base = {"recording": recording.get("title", recording["name"])}
    rows, results, traces = [], {}, {}
    reference = {}
    if has_tags(recording):
        metadata, reference, _ = pose_files.read_trajectory(p["reference"])
        posed = sum(n in reference for n in names)
        rows.append({**base, "method": cfg["reference"]["label"], "frames": len(names), "posed": posed,
                     "s_per_frame": per_frame(metadata.get("seconds"), metadata.get("frames")),
                     "grip": grip.get("aprilgrid"), "note": "reference"})
    else:
        rows.append({**base, "method": cfg["reference"]["label"], "frames": len(names), "posed": 0,
                     "grip": 0, "note": "no tag seen by both cameras in any frame"})

    runs = []
    for e in cfg["estimators"]:
        runs.append(e)
        start = restart_start(p, e["name"])
        if start is not None:   # the failed full run stays; its restart is a row of its own
            runs.append({**e, "name": e["name"] + RESTART, "label": f"{e['label']} (restarted at frame {start})"})
    for e in runs:
        path = p["poses"](e["name"])
        if not os.path.exists(path):
            print(f"  {recording['name']}: {e['name']}: no poses at {common.rel(path)}, skipped")
            continue
        metadata, estimate, diagnostics = pose_files.read_trajectory(path)
        result, trace = evaluate(reference, estimate, diagnostics, list(diagnostics), cfg)
        results[e["name"]] = result
        if trace is not None:
            traces[e["label"]] = trace
        frames = metadata.get("frames") or result["frames"]
        rows.append({**base, "method": e["label"], "frames": result["frames"], "posed": result["posed"],
                     "ate_rmse_mm": result.get("ate_mm", {}).get("rmse"), "ate_max_mm": result.get("ate_mm", {}).get("max"),
                     "ate_rmse_deg": result.get("ate_deg", {}).get("rmse"),
                     "rpe_median_%": result.get("rpe_percent", {}).get("median"),
                     "rpe_median_deg": result.get("rpe_deg", {}).get("median"),
                     "s_per_frame": per_frame(metadata.get("seconds"), frames),
                     "depth_s_per_frame": per_frame(metadata.get("depth_seconds"), frames),
                     "grip": grip.get(e["name"].replace(RESTART, "")),
                     "icp_points": grip.get(e["name"].replace(RESTART, "") + "_points") if e["kind"] == "icp" else None})

    out_dir = os.path.join(p["out"], "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    if traces:
        plot(os.path.join(out_dir, "trajectories.png"), traces, recording.get("title", recording["name"]),
             cfg["reference"]["label"])
    trajectory_figures(cfg, recording, out_dir)
    return rows


def posed_centres(estimate, diagnostics):
    """{frame: T_world_cam} of the frames posed and not lost, in frame order."""
    return {n: invert(estimate[n]) for n in diagnostics if n in estimate and not diagnostics[n].get("lost", 0)}


def p95_step(poses):
    """95th percentile of a path's frame-to-frame camera step (mm): how smooth it is."""
    centres = np.array([T[:3, 3] for T in poses.values()])
    return float(np.percentile(np.linalg.norm(np.diff(centres, axis=0), axis=1), 95))


def track_axes(centres):
    """(origin, 3x3 rows: along, across, out of plane) of a pass: the centres' principal directions, along
    pointing from the first centre to the last."""
    origin = centres.mean(0)
    axes = np.linalg.svd(centres - origin)[2]
    if (centres[-1] - centres[0]) @ axes[0] < 0:
        axes[0] = -axes[0]
    axes[2] = np.cross(axes[0], axes[1])
    return origin, axes


JUMP_MM = 40.0   # a frame-to-frame step the trolley cannot make (~6 mm a frame): drawn as a dotted jump


def aligned_paths(cfg, recording):
    """Every method's camera centres in one frame, as the path figures draw them (see the module docstring):
    {"aligned": {method: {frame: centre}}, "reference": centres or None, "reference_label", "how", "labels",
    "to_track": centres -> (along, across, out of plane) mm, "start": along-track offset of the first frame}."""
    p = recording_paths(cfg, recording)
    lever = cfg["evaluation"]["align_lever_mm"]
    paths = {}
    labels = {}
    for e in cfg["estimators"]:
        candidates = [(e["name"], e["label"])]
        start = restart_start(p, e["name"])
        if start is not None:
            candidates.append((e["name"] + RESTART, f"{e['label']} (restarted at frame {start})"))
        for name, label in candidates:
            if os.path.exists(p["poses"](name)):
                _, estimate, diagnostics = pose_files.read_trajectory(p["poses"](name))
                paths[name], labels[name] = posed_centres(estimate, diagnostics), label
    title = recording.get("title", recording["name"])
    noisy = False
    if has_tags(recording):
        _, reference, _ = pose_files.read_trajectory(p["reference"])
        target = {n: invert(T) for n, T in reference.items()}
        target_label = f"{FIGURE_REFERENCE} (reference)"
        how = f"after one rigid fit onto {cfg['reference']['label']}"
        if recording.get("reference_noisy"):   # few, small tags: a scattered reference, drawn as dots
            noisy = True
            target_label = f"{FIGURE_REFERENCE}, single tag estimate"
    elif recording.get("plot_reference"):
        _, reference, _ = pose_files.read_trajectory(os.path.join(p["out"], recording["plot_reference"]["file"]))
        target = {n: invert(T) for n, T in reference.items()}
        target_label, noisy = recording["plot_reference"]["label"], True
        how = "fitted onto the single-tag poses where they agree"
    else:
        anchor = min(paths, key=lambda name: p95_step(paths[name]))
        target, target_label = paths[anchor], None
        how = f"no reference; fitted onto {labels[anchor]} where they agree"
    aligned = {}
    for name, poses in paths.items():
        names = [n for n in poses if n in target]
        if len(names) < 3:
            continue
        keep = np.ones(len(names), bool)
        for _ in range(1 if has_tags(recording) else 10):   # with tags: the same single fit as the table's ATE
            source = np.vstack([pose_points(poses[n], lever) for n, k in zip(names, keep) if k])
            goal = np.vstack([pose_points(target[n], lever) for n, k in zip(names, keep) if k])
            _, R, t = umeyama(source, goal, with_scale=False)
            distance = np.array([np.linalg.norm(R @ poses[n][:3, 3] + t - target[n][:3, 3]) for n in names])
            keep = distance < max(3 * np.median(distance[keep]), 15.0)
        aligned[name] = {n: R @ poses[n][:3, 3] + t for n in poses}
    frame_centres = np.array([T[:3, 3] for T in target.values()])
    origin, axes = track_axes(frame_centres)
    to_track = lambda xyz: (np.asarray(xyz) - origin) @ axes.T  # noqa: E731
    start = to_track(frame_centres)[:, 0].min()

    return {"aligned": aligned, "reference": frame_centres if target_label else None, "reference_label": target_label,
            "reference_noisy": noisy,
            "how": how, "labels": labels, "to_track": to_track, "start": start, "title": title,
            "target": target}


def draw_top_view(ax, data, legend=True, mark=8, lw=1.2):
    """The paths of aligned_paths() from above at true scale, jumps dotted, start (o) and end (square) marked.
    mark: start/end marker size and lw: path width (points), smaller for the report-width figure."""
    to_track, start = data["to_track"], data["start"]
    if data["reference"] is not None:
        ref = to_track(data["reference"])
        if data["reference_noisy"]:   # a noisy partial reference: points, not a line to follow
            ax.plot(ref[:, 0] - start, ref[:, 1], ".", color=REFERENCE_INK, ms=mark / 2, alpha=0.6,
                    label=data["reference_label"])
        else:
            ax.plot(ref[:, 0] - start, ref[:, 1], color=REFERENCE_INK, lw=2.5 * lw, alpha=0.35,
                    label=data["reference_label"])
    for name, centres in data["aligned"].items():
        xyz = np.array(list(centres.values()))
        track = to_track(xyz)
        cuts = np.flatnonzero(np.linalg.norm(np.diff(xyz, axis=0), axis=1) > JUMP_MM) + 1
        jumps = len(cuts)
        label = data["labels"][name] + (f" ({jumps} jump{'s' if jumps > 1 else ''} > {JUMP_MM:.0f} mm, dotted)"
                                        if jumps else "")
        x, y = track[:, 0] - start, track[:, 1]
        ax.plot(x, y, color=colour(name), lw=lw / 2, ls=":", alpha=0.5)
        for k, (a, b) in enumerate(zip(np.r_[0, cuts], np.r_[cuts, len(x)])):
            ax.plot(x[a:b], y[a:b], color=colour(name), lw=lw, label=label if k == 0 else None)
        ax.plot(x[0], y[0], "o", ms=mark, mfc="none", mec=colour(name), mew=mark / 5)
        ax.plot(x[-1], y[-1], "s", ms=mark, mfc="none", mec=colour(name), mew=mark / 5)
    ax.plot([], [], "o", ms=mark, mfc="none", mec=REFERENCE_INK, label="start")
    ax.plot([], [], "s", ms=mark, mfc="none", mec=REFERENCE_INK, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="x (mm)", ylabel="y (mm)")
    ax.grid(alpha=0.3, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))


def trajectory_figures(cfg, recording, out_dir):
    """paths.png, path_2d.png (and ate.png with tags) from the saved trajectories; see the module docstring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = aligned_paths(cfg, recording)
    aligned, labels, how, title = data["aligned"], data["labels"], data["how"], data["title"]
    to_track, start, target_label = data["to_track"], data["start"], data["reference_label"]
    frame_centres, jump, target = data["reference"], JUMP_MM, data["target"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    if target_label:
        ref = to_track(frame_centres)
        for k in (0, 1):
            if data["reference_noisy"]:
                ax[k].plot(ref[:, 0] - start, ref[:, k + 1], ".", color=REFERENCE_INK, ms=4, alpha=0.6, label=target_label)
            else:
                ax[k].plot(ref[:, 0] - start, ref[:, k + 1], color=REFERENCE_INK, lw=3, alpha=0.35, label=target_label)
    for name, centres in aligned.items():
        track = to_track(np.array(list(centres.values())))
        steps = np.linalg.norm(np.diff(np.array(list(centres.values())), axis=0), axis=1)
        jumps = int((steps > jump).sum())
        label = labels[name] + (f" ({jumps} jump{'s' if jumps > 1 else ''} > {jump:.0f} mm, dotted)" if jumps else "")
        for k in (0, 1):
            x, y = track[:, 0] - start, track[:, k + 1]
            # Break the line at the jumps and draw them as faint dotted links.
            y_solid = y.copy().astype(float)
            ax[k].plot(x, y, color=colour(name), lw=0.6, ls=":", alpha=0.5)
            for a, b in zip(np.r_[0, np.flatnonzero(steps > jump) + 1], np.r_[np.flatnonzero(steps > jump) + 1, len(x)]):
                ax[k].plot(x[a:b], y_solid[a:b], color=colour(name), lw=1.2,
                           label=label if (k == 0 and a == 0) else None)
    ax[0].set_ylabel("across track (mm)")
    ax[1].set_ylabel("out of plane (mm)")
    ax[1].set_xlabel("along track (mm)")
    for a in ax:
        a.grid(alpha=0.3, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(f"{title}: camera paths ({how})\nacross / out-of-plane axes exaggerated", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "paths.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    draw_top_view(ax, data)
    ax.set_title(f"{title}: camera paths seen from above, true scale\n({how})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "path_2d.png"), dpi=150)
    plt.close(fig)

    if not has_tags(recording):
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    # Along-track position of a noisy reference jitters back and forth, so plot those against frame number.
    by_frame = data["reference_noisy"]
    track_of = dict(zip(target, to_track(frame_centres)[:, 0] - start))
    for name, centres in aligned.items():
        names = [n for n in centres if n in target]
        error = [np.linalg.norm(centres[n] - target[n][:3, 3]) for n in names]
        x = [int(os.path.splitext(n)[0]) for n in names] if by_frame else [track_of[n] for n in names]
        ax.plot(x, error, color=colour(name), lw=1.2, label=labels[name])
    xlabel = "frame" if by_frame else f"along track (mm, {cfg['reference']['label']})"
    ax.set(xlabel=xlabel, ylabel="ATE (mm)", ylim=(0, None))
    ax.grid(alpha=0.3, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{title}: camera position error per frame, after one rigid fit onto {cfg['reference']['label']}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ate.png"), dpi=150)
    plt.close(fig)


def combined_top_views(cfg):
    """Each cfg["combined_paths"] entry: its recordings' top views side by side in one figure at one true scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_name = {r["name"]: r for r in cfg["recordings"]}
    for entry in cfg.get("combined_paths", []):
        data = [aligned_paths(cfg, by_name[name]) for name in entry["recordings"]]
        name_of = {id(d): name for d, name in zip(data, entry["recordings"])}
        from matplotlib.lines import Line2D
        # The report's text width (A4 less 1.5 cm margins), fonts for print; the key goes underneath.
        plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                             "ytick.labelsize": 6.5, "axes.linewidth": 0.6})
        fig, axes = plt.subplots(1, len(data), figsize=(18 / 2.54, 2.6))
        axes = np.atleast_1d(axes)
        for ax, d in zip(axes, data):
            draw_top_view(ax, d, legend=False, mark=4.5, lw=0.9)
            title = entry.get("titles", {}).get(name_of[id(d)], d["title"])
            ax.set_title(title, pad=3)
        # One key for both panels: the methods, the reference, and the jump / start / end marks.
        handles = [Line2D([], [], color=REFERENCE_INK, lw=2.2, alpha=0.35, label=f"{FIGURE_REFERENCE} (reference)")]
        handles += [Line2D([], [], ls="", marker=".", ms=5, color=REFERENCE_INK, alpha=0.6, label=d["reference_label"])
                    for d in data if d["reference_noisy"]]
        handles += [Line2D([], [], color=colour(e["name"]), lw=0.9, label=e["label"]) for e in cfg["estimators"]]
        handles += [Line2D([], [], color=REFERENCE_INK, lw=0.6, ls=":", label=f"jump > {JUMP_MM:.0f} mm"),
                    Line2D([], [], ls="", marker="o", ms=4.5, mfc="none", mec=REFERENCE_INK, label="start"),
                    Line2D([], [], ls="", marker="s", ms=4.5, mfc="none", mec=REFERENCE_INK, label="end")]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5, frameon=False,
                   columnspacing=1.2, handlelength=1.8)
        # One true scale for all panels: the union of their data limits, equal aspect by resizing the boxes.
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        x = (min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes))
        y = (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))
        for ax in axes:
            ax.set_xlim(x)
            ax.set_ylim(y)
            ax.set_aspect("equal", adjustable="box")
        for ax in axes[1:]:
            ax.set_ylabel("")
        fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.0)
        path = os.path.join(common.project_path(cfg["out_dir"]), entry["file"])
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"-> {common.rel(path)}")


def write_summary(cfg, rows):
    out = common.project_path(cfg["out_dir"])
    os.makedirs(out, exist_ok=True)
    keys = ["recording", "method", "frames", *[k for k, _ in COLUMNS], "icp_points", "note"]
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    distance = cfg["evaluation"]["rpe_distance_mm"]
    steps = ", ".join(f"{r.get('title', r['name'])}: " + (f"every {frame_step(cfg, r)}th frame" if frame_step(cfg, r) > 1
                                                          else "every frame") for r in cfg["recordings"])
    lines = ["# Pose estimation comparison", "",
             f"Frames: {steps}. Tags masked from the markerless methods. ATE after one rigid fit onto the "
             f"{cfg['reference']['label']} poses; RPE over {distance} mm of travel. Grip: median per frame "
             "(AprilGrid: tag corners; COLMAP: SIFT points triangulated; ICP: depth points resisting a slide along the track).", "",
             "| Recording | Method | " + " | ".join(h for _, h in COLUMNS) + " |",
             "|" + "---|" * (len(COLUMNS) + 2)]
    for row in rows:
        lines.append(f"| {row['recording']} | {row['method']} | " + " | ".join(fmt(row, k) for k, _ in COLUMNS) + " |")
    lines += ["", "## Mode each method ran in", "",
              f"- **{cfg['reference']['label']}**: {cfg['reference']['mode']}"]
    lines += [f"- **{e['label']}**: {e['mode']}" for e in cfg["estimators"]]
    lines += ["", "## Per recording", ""]
    for r in cfg["recordings"]:
        notes = [v for k, v in r.items() if k.startswith("_")]
        extra = f" ICP overrides: `{' '.join(r['icp_argv'])}`." if r.get("icp_argv") else ""
        lines.append(f"- **{r.get('title', r['name'])}** (`{r['frames']}`): " + " ".join(notes) + extra)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    tex = ["\\begin{tabular}{ll" + "r" * len(COLUMNS) + "}", "\\toprule",
           "Recording & Method & " + " & ".join(h.replace("%", "\\%") for _, h in COLUMNS) + " \\\\", "\\midrule"]
    previous = None
    for row in rows:
        if previous is not None and row["recording"] != previous:
            tex.append("\\midrule")
        tex.append(f"{row['recording'] if row['recording'] != previous else ''} & {row['method']} & "
                   + " & ".join("--" if fmt(row, k) == "-" else fmt(row, k) for k, _ in COLUMNS)
                   + " \\\\")
        previous = row["recording"]
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out, "summary.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "git_commit": common.git("rev-parse", "HEAD"),
                   "git_uncommitted_src_changes": common.uncommitted_src()}, fh, indent=1)
    print("\n".join(lines))
    print(f"-> {common.rel(out)}/summary.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config_comparison.json"))
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    rows = [row for recording in cfg["recordings"] for row in score_recording(cfg, recording)]
    write_summary(cfg, rows)
    combined_top_views(cfg)
    return rows


if __name__ == "__main__":
    main()
