"""The pipeline runs of the pose experiment: reference poses, tag masks, per-frame depth, then every estimator.

    results/<out_dir>/<recording>/tag_poses.yaml          reference: tag_poses.py on every frame
    work/<work_dir>/<recording>/tag_masks/                 make_tag_masks.py, every frame_step'th pair
    work/<work_dir>/<recording>/depth/<source>/            depth_frames.py (tags masked)
    work/<work_dir>/<recording>/colmap/                    COLMAP workspace
    results/<out_dir>/<recording>/poses/<estimator>.yaml   colmap_poses.py / icp_poses.py

    src/venv/bin/python src/experiments/pose_estimation/run_poses.py --config src/experiments/pose_estimation/config.json
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import restarts  # noqa: E402

RECON = os.path.join("src", "03_Reconstruction")


def recording_paths(cfg, recording):
    out = os.path.join(common.project_path(cfg["out_dir"]), recording["name"])
    work = os.path.join(common.project_path(cfg["work_dir"]), recording["name"])
    return {"out": out, "work": work,
            "reference": os.path.join(out, "tag_poses.yaml"),
            "masks": os.path.join(work, "tag_masks"),
            "depth": lambda source: os.path.join(work, "depth", source),
            "colmap": os.path.join(work, "colmap"),
            "poses": lambda name: os.path.join(out, "poses", name + ".yaml")}


def run(python, script, argv, output, skip_existing, runs, label):
    """One pipeline script; recorded for the manifest."""
    record = {"step": label, "script": script, "python": python, "argv": [common.rel(a) if os.path.isabs(a) else a for a in argv]}
    if skip_existing and os.path.exists(output):
        print(f"=== {label}: {common.rel(output)} exists, skipped")
        runs.append({**record, "skipped": True})
        return
    print(f"\n=== {label}", flush=True)
    start = time.time()
    result = subprocess.run([common.project_path(python), common.project_path(script), *argv], cwd=common.PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed ({result.returncode})")
    runs.append({**record, "seconds": round(time.time() - start, 1)})


def frame_step(cfg, recording):
    return recording.get("frame_step", cfg["frame_step"])


def has_tags(recording):
    return recording.get("tags", True)


def run_recording(cfg, recording, skip_existing):
    p = recording_paths(cfg, recording)
    py, step, runs = cfg["python"], str(frame_step(cfg, recording)), []
    frames, calibration = common.project_path(recording["frames"]), common.project_path(recording["calibration"])

    if has_tags(recording):
        run(py["default"], os.path.join(RECON, "tag_poses.py"),
            ["--frames-dir", frames, "--calibration", calibration, "--tag-size", str(recording["tag_size_mm"]),
             "--max-reprojection", str(recording.get("max_reprojection_px", cfg["reference"]["max_reprojection_px"])),
             "--out", p["reference"], *(["--enhance"] if recording.get("enhance_tags") else [])],
            p["reference"], skip_existing, runs, f"{recording['name']}: reference tag poses")

    masks = []
    if cfg["tag_masks"]["enabled"] and has_tags(recording):
        run(py["default"], os.path.join("src", "experiments", "pose_estimation", "make_tag_masks.py"),
            ["--frames-dir", frames, "--poses", p["reference"], "--out-dir", p["masks"], "--step", step,
             "--scale", str(cfg["tag_masks"]["scale"]), *(["--enhance"] if recording.get("enhance_tags") else [])],
            os.path.join(p["masks"], "masks.json"), skip_existing, runs, f"{recording['name']}: tag masks")
        masks = ["--masks", p["masks"]]

    for source in sorted({e["depth"] for e in cfg["estimators"] if e["kind"] == "icp"}):
        spec = cfg["depth"][source]
        run(py[spec["python"]], os.path.join(RECON, "depth_frames.py"),
            ["--frames-dir", frames, "--calibration", calibration, "--out-dir", p["depth"](source), "--step", step,
             "--depth-min", str(recording["depth_min_mm"]), "--depth-max", str(recording["depth_max_mm"]),
             *masks, *spec["argv"]],
            os.path.join(p["depth"](source), "depth_frames.json"), skip_existing, runs,
            f"{recording['name']}: depth {source}")

    for estimator in cfg["estimators"]:
        out = p["poses"](estimator["name"])
        if estimator["kind"] == "colmap":
            argv = ["--frames-dir", frames, "--calibration", calibration, "--workspace", p["colmap"], "--out", out,
                    "--step", step, *masks, *estimator["argv"]]
            script = os.path.join(RECON, "colmap_poses.py")
        elif estimator["kind"] == "icp":
            argv = ["--depth-dir", p["depth"](estimator["depth"]), "--out", out, *estimator["argv"],
                    *recording.get("icp_argv", [])]
            script = os.path.join(RECON, "icp_poses.py")
        else:
            raise SystemExit(f"unknown estimator kind {estimator['kind']!r}")
        run(py["default"], script, argv, out, skip_existing, runs, f"{recording['name']}: {estimator['name']}")
        if recording.get("restart_on_failure"):
            restart_estimator(cfg, recording, estimator, argv, script, skip_existing, runs)
    return runs


def restart_estimator(cfg, recording, estimator, argv, script, skip_existing, runs):
    """Judge the full run; if it failed, rerun from later starts on subsets of its inputs (restarts.py)."""
    sys.path.insert(0, os.path.join(common.PROJECT_ROOT, RECON))
    import pose_files
    p = recording_paths(cfg, recording)
    name, out = estimator["name"], p["poses"](estimator["name"])
    record_path = os.path.join(p["out"], "poses", f"{name}_restarts.json")
    if skip_existing and os.path.exists(record_path):
        print(f"=== {recording['name']}: {name} restarts: {common.rel(record_path)} exists, skipped")
        return
    _, _, full_diag = pose_files.read_trajectory(out)
    names = list(full_diag)

    def option(flag):   # the value after a flag in argv, which the restarted run replaces
        return argv[argv.index(flag) + 1]

    def run_from(start):
        if start == 0:
            return out
        subset = [n for n in names if restarts.frame_number(n) >= start]
        base = os.path.join(p["work"], "restarts", f"{name}_from_{start}")
        target = os.path.join(p["out"], "poses", "restarts", f"{name}_from_{start}.yaml")
        new = list(argv)
        if estimator["kind"] == "colmap":
            frames = restarts.link_subset(option("--frames-dir"), os.path.join(base, "frames"), subset, ["left", "right"])
            new[new.index("--frames-dir") + 1] = frames
            new[new.index("--workspace") + 1] = os.path.join(base, "colmap")
            new[new.index("--step") + 1] = "1"   # the subset already holds only the frames given
        else:
            depth = option("--depth-dir")
            npz = [os.path.splitext(n)[0] + ".npz" for n in subset]
            new[new.index("--depth-dir") + 1] = restarts.link_subset(depth, os.path.join(base, "depth"), npz,
                                                                     ["frames"], ["depth_frames.json"])
        new[new.index("--out") + 1] = target
        run(cfg["python"]["default"], script, new, target, skip_existing, runs,
            f"{recording['name']}: {name} restarted at frame {start}")
        return target

    def read(path):
        _, poses, diagnostics = pose_files.read_trajectory(path)
        return poses, diagnostics

    attempts, kept = restarts.run_with_restarts(names, run_from, read, cfg["restarts"])
    for a in attempts:
        state = "FAILED: " + "; ".join(a["reasons"]) if a["failed"] else "passed"
        print(f"  {name} from frame {a['start']}: posed {a['posed']}/{a['frames']}, {a['jumps']} jumps -> {state}")
    restarts.write_attempts(record_path, name, [{**a, "path": common.rel(a["path"])} for a in attempts], kept)
    restart_out = os.path.join(p["out"], "poses", f"{name}_restart.yaml")
    if kept:
        shutil.copy(kept["path"], restart_out)
        print(f"  {name}: kept the restart at frame {kept['start']} -> {common.rel(restart_out)}")
    elif os.path.exists(restart_out):
        os.remove(restart_out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    return [r for recording in cfg["recordings"] for r in run_recording(cfg, recording, args.skip_existing)]


if __name__ == "__main__":
    main()
