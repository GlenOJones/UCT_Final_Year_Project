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
recording <recording>/evaluation/{metrics.json, trajectories.png}.
"""
import argparse
import csv
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "03_Reconstruction"))
import common  # noqa: E402
import pose_files  # noqa: E402
from evaluate_poses import evaluate, plot  # noqa: E402
from run_poses import frame_step, has_tags, recording_paths  # noqa: E402
from stereo_rig import find_frame_pairs  # noqa: E402

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

    for e in cfg["estimators"]:
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
                     "grip": grip.get(e["name"]),
                     "icp_points": grip.get(e["name"] + "_points") if e["kind"] == "icp" else None})

    out_dir = os.path.join(p["out"], "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    if traces:
        plot(os.path.join(out_dir, "trajectories.png"), traces, recording.get("title", recording["name"]),
             cfg["reference"]["label"])
    return rows


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
    return rows


if __name__ == "__main__":
    main()
