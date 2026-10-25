"""The pipeline runs of the frame-count sweep: the whole chosen pipeline on every N'th frame, for each N in steps.

    depth     depth_frames.py once on every frame, per depth source (FoundationStereo, tuned SGBM), tags masked.
              Each frame's depth depends on that frame alone, so each subset fuses its own frames from this run.
    per step N (every N'th frame pair):
      sfm     colmap_poses.py --step N in a fresh workspace: COLMAP SfM re-run on the subset alone, so the poses
              are what the subset itself gives (SfM is where sparse frames fail).
      tsdf    tsdf_fuse.py per depth source at that subset's poses (only its posed frames are fused).
      board   for evaluation only: the subset's trajectory fitted onto the AprilGrid tag poses (run_fusion.py's
              board_alignment), and each cloud moved into the board frame.

Every stage runs through monitor.py, one at a time, so its wall time, CPU, peak RAM and peak GPU memory are its own.

    src/venv/bin/python src/experiments/frame_count/run_frame_count.py [--config ...] [--skip-existing]

Writes, under work/<work_dir>/<recording>/:
    depth/<source>/                       depth_frames.py output, every frame
    step_<N>/colmap/                      COLMAP workspace
    step_<N>/tsdf_<source>/               tsdf_fuse.py output; cloud_board.ply in the board frame
    logs/                                 each stage's console output
and under results/<out_dir>/<recording>/: depth_costs.json, step_<N>/poses_colmap.yaml (the subset's trajectory),
step_<N>/run.json (frames, each stage's cost record, the board fit).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
SRC = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, HERE)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(EXPERIMENTS, "fusion"))
sys.path.insert(0, os.path.join(EXPERIMENTS, "pose_estimation"))
sys.path.insert(0, os.path.join(SRC, "03_Reconstruction"))
import common  # noqa: E402
import monitor  # noqa: E402
import pose_files  # noqa: E402
from run_fusion import board_alignment  # noqa: E402

RECON = os.path.join("src", "03_Reconstruction")


def paths(cfg):
    rec = cfg["recording"]
    out = os.path.join(common.project_path(cfg["out_dir"]), rec["name"])
    work = os.path.join(common.project_path(cfg["work_dir"]), rec["name"])
    return {"out": out, "work": work, "logs": os.path.join(work, "logs"),
            "depth": lambda source: os.path.join(work, "depth", source),
            "step_out": lambda n: os.path.join(out, f"step_{n}"),
            "step_work": lambda n: os.path.join(work, f"step_{n}"),
            "reference": common.project_path(rec["reference_poses"])}


def stage(cfg, p, name, python, script, argv, memory_max=None):
    """One monitored pipeline stage; its cost record (raises if the command failed)."""
    command = [common.project_path(cfg["python"][python]), common.project_path(script), *argv]
    print(f"\n=== {name}", flush=True)
    record = monitor.run(command, common.PROJECT_ROOT, memory_max or cfg["monitor"]["memory_max"],
                         cfg["monitor"]["interval_s"], os.path.join(p["logs"], f"{name}.log"))
    record["command"] = [common.rel(c) if os.path.isabs(c) else c for c in record["command"]]
    print(f"=== {name}: {record['wall_s']} s, peak RAM {record['peak_tree_rss_mb']:.0f} MB, "
          f"peak GPU {record['peak_gpu_mib']} MiB, exit {record['returncode']}", flush=True)
    return record


# ====== DEPTH (every frame, once) ======
def run_depth(cfg, p, skip_existing):
    rec = cfg["recording"]
    costs_path = os.path.join(p["out"], "depth_costs.json")
    costs = json.load(open(costs_path)) if os.path.exists(costs_path) else {}
    for source, spec in cfg["depth"].items():
        out = p["depth"](source)
        if skip_existing and source in costs and os.path.exists(os.path.join(out, "depth_frames.json")):
            print(f"=== depth {source}: exists, skipped")
            continue
        shutil.rmtree(out, ignore_errors=True)
        argv = ["--frames-dir", common.project_path(rec["frames"]), "--calibration", common.project_path(rec["calibration"]),
                "--out-dir", out, "--step", "1", "--depth-min", str(rec["depth_min_mm"]),
                "--depth-max", str(rec["depth_max_mm"]), "--masks", common.project_path(rec["tag_masks"]), *spec["argv"]]
        costs[source] = stage(cfg, p, f"depth_{source}", spec["python"], os.path.join(RECON, "depth_frames.py"), argv)
        if costs[source]["returncode"] != 0:
            raise SystemExit(f"depth {source} failed")
        with open(costs_path, "w", encoding="utf-8") as fh:
            json.dump(costs, fh, indent=1)
    return costs


# ====== ONE SUBSET ======
def to_board(source, target, T):
    cloud = o3d.io.read_point_cloud(source)
    cloud.transform(T)
    o3d.io.write_point_cloud(target, cloud)
    return len(cloud.points)


def run_step(cfg, p, n, skip_existing):
    rec = cfg["recording"]
    out, work = p["step_out"](n), p["step_work"](n)
    run_path = os.path.join(out, "run.json")
    if skip_existing and os.path.exists(run_path):
        print(f"\n######## step {n}: exists, skipped")
        return json.load(open(run_path))
    print(f"\n######## step {n}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    trajectory = os.path.join(out, "poses_colmap.yaml")
    if os.path.exists(trajectory):
        os.remove(trajectory)
    record = {"step": n, "stages": {}}

    record["stages"]["sfm"] = stage(
        cfg, p, f"step{n}_sfm", "default", os.path.join(RECON, "colmap_poses.py"),
        ["--frames-dir", common.project_path(rec["frames"]), "--calibration", common.project_path(rec["calibration"]),
         "--workspace", os.path.join(work, "colmap"), "--out", trajectory, "--step", str(n),
         "--masks", common.project_path(rec["tag_masks"]), *cfg["colmap"]["argv"]])
    if record["stages"]["sfm"]["returncode"] != 0 or not os.path.exists(trajectory):
        record["failed"] = "sfm"
    else:
        metadata, poses, diagnostics = pose_files.read_trajectory(trajectory)
        record.update({"frames": len(diagnostics) or int(metadata["frames"]), "frames_posed": len(poses),
                       "sfm_seconds": {k[len("seconds_"):]: float(v) for k, v in metadata.items()
                                       if k.startswith("seconds_")}})
        if len(poses) < 3:
            record["failed"] = "sfm: fewer than 3 frames posed"
    if "failed" not in record:
        for source in cfg["depth"]:
            folder = os.path.join(work, f"tsdf_{source}")
            record["stages"][f"tsdf_{source}"] = stage(
                cfg, p, f"step{n}_tsdf_{source}", "default", os.path.join(RECON, "tsdf_fuse.py"),
                ["--depth-dir", p["depth"](source), "--poses", trajectory, "--out-dir", folder, *cfg["tsdf"]["argv"]],
                cfg["tsdf"]["memory_max"])
            if record["stages"][f"tsdf_{source}"]["returncode"] != 0:
                record["failed"] = f"tsdf_{source} (killed by the {cfg['tsdf']['memory_max']} memory cap?)"
                print(f"WARNING: step {n} {record['failed']}", flush=True)
        print("\n=== board frame (evaluation only)")
        try:
            T, alignment = board_alignment(cfg, p, poses)
        except ValueError as error:   # too few frames shared with the tags to fit
            T, alignment = None, {"error": str(error)}
        record["alignment"] = alignment
        record["points"] = {}
        for source in cfg["depth"]:
            folder = os.path.join(work, f"tsdf_{source}")
            if T is not None and os.path.exists(os.path.join(folder, "cloud.ply")):
                record["points"][source] = to_board(os.path.join(folder, "cloud.ply"),
                                                    os.path.join(folder, "cloud_board.ply"), T)
    with open(run_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--steps", type=int, nargs="+", help="only these steps (default: the config's)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--skip-existing-depth", action="store_true",
                    help="keep the depth runs that exist but re-run the steps (e.g. one step whose TSDF failed)")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    p = paths(cfg)
    os.makedirs(p["out"], exist_ok=True)
    os.makedirs(p["logs"], exist_ok=True)
    run_depth(cfg, p, args.skip_existing or args.skip_existing_depth)
    return [run_step(cfg, p, n, args.skip_existing) for n in (args.steps or cfg["steps"])]


if __name__ == "__main__":
    main()
