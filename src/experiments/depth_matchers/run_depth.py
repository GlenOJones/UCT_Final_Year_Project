"""The pipeline runs of the depth experiment: depth_frames.py once per recording and matcher.

    work/<work_dir>/<recording>/<matcher>/depth_frames.json    settings, per-frame time, pixels and points
    work/<work_dir>/<recording>/<matcher>/frames/              points per frame (1 mm voxels, camera frame)
    work/<work_dir>/<recording>/<matcher>/disparity/           disparity per frame, uint16 PNG, 1/16 px

    src/venv/bin/python src/experiments/depth_matchers/run_depth.py --trial 3          # check the output first
    src/venv/bin/python src/experiments/depth_matchers/run_depth.py                    # every recording, matcher
    src/venv/bin/python src/experiments/depth_matchers/run_depth.py --recordings ice_pyramid_shade_pyr1_pass4 --matchers sgbm
    src/venv/bin/python src/experiments/depth_matchers/run_depth.py --frames data/3DRecon/Ice/Wave/Shade/wave1_frames_pass2_720mm \
        --depth-min 400 --depth-max 1400 --out-dir results/experiments/depth_wave --work-dir work/experiments/depth_wave
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common  # noqa: E402
import depth_config  # noqa: E402

DEPTH_FRAMES = os.path.join("src", "03_depth", "depth_frames.py")


def run_one(cfg, recording, matcher, skip_existing):
    """depth_frames.py for one recording and matcher."""
    spec = cfg["matchers"][matcher]
    out = depth_config.depth_dir(cfg, recording, matcher)
    argv = ["--frames-dir", common.project_path(recording["frames"]),
            "--calibration", common.project_path(recording["calibration"]), "--out-dir", out,
            "--step", str(recording.get("frame_step", cfg["frame_step"])), "--depth-min", str(recording["depth_min_mm"]),
            "--depth-max", str(recording["depth_max_mm"]), "--save-disparity", *spec["argv"]]
    if cfg["max_frames"]:
        argv += ["--max-frames", str(cfg["max_frames"])]
    python = cfg["python"][spec["python"]]
    label = f"{recording['name']}: {matcher}"
    record = {"step": label, "script": DEPTH_FRAMES, "python": python,
              "argv": [common.rel(a) if os.path.isabs(a) else a for a in argv]}
    if skip_existing and os.path.exists(os.path.join(out, "depth_frames.json")):
        print(f"=== {label}: {common.rel(out)} exists, skipped")
        return {**record, "skipped": True}
    print(f"\n=== {label}", flush=True)
    start = time.time()
    # FoundationStereo needs the expandable allocator to fit full frames in 8 GB.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    result = subprocess.run([common.project_path(python), common.project_path(DEPTH_FRAMES), *argv],
                            cwd=common.PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed ({result.returncode})")
    return {**record, "seconds": round(time.time() - start, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    depth_config.add_arguments(ap)
    ap.add_argument("--skip-existing", action="store_true", help="keep runs whose depth_frames.json exists")
    args = ap.parse_args(argv)
    cfg = depth_config.resolve(args, ap)
    return [run_one(cfg, recording, matcher, args.skip_existing)
            for recording in cfg["recordings"] for matcher in cfg["matchers"]]


if __name__ == "__main__":
    main()
