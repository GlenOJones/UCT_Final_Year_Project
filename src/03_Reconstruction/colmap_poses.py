"""Rig pose for every frame of a scan from the images alone: COLMAP structure-from-motion, no tags.

The SfM half of colmap_mvs.py, run on every --step'th frame pair whether or not it has a tag pose,
and written out as a trajectory instead of being aligned onto tag poses:

  workspace  link the frame pairs into <workspace>/images/{left,right}/ (and the masks, if given)
  features   SIFT per camera with the calibrated intrinsics (FULL_OPENCV)
  rig        left/right declared a rig with the calibrated extrinsics
  match      every camera combination within --window frames
  map        incremental SfM, intrinsics and rig FIXED, so the 150 mm baseline gives metric scale

The stages are colmap_mvs.py's own functions, so both scripts make the same model from the same frames.
The world frame is the left camera of the first registered frame. SfM may split a scan into several
models when the images stop overlapping well; only the largest is kept, and the frames it left out
are written without a pose.

    src/venv/bin/python src/03_Reconstruction/colmap_poses.py --frames-dir data/3DRecon/3DPrintFinal/30_frames_range \
        --calibration results/Calibration/calibration/FullCalibration_water_final_merge_stereo.yaml \
        --workspace work/3DPrintFinal/s30_range/colmap_poses --out results/3DPrintFinal/s30_range/poses/colmap_poses.yaml --step 5

--masks takes a folder with {left,right}/<frame name> images, 0 = no features there (e.g. the reference
tags, when testing what SfM would do without them).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_mvs  # noqa: E402
from pose_files import write_trajectory  # noqa: E402
from stereo_rig import CAMERAS, find_frame_pairs, invert, load_rig, rel  # noqa: E402

STAGES = ("workspace", "features", "rig", "match", "map", "poses")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-frame rig poses from COLMAP SfM, without tags.")
    ap.add_argument("--frames-dir", required=True, help="recording's frame pairs: left/ and right/")
    ap.add_argument("--calibration", required=True, help="stereo calibration YAML from Calibration.py")
    ap.add_argument("--workspace", required=True, help="COLMAP working folder (database, sparse model)")
    ap.add_argument("--out", required=True, help="trajectory YAML to write")
    ap.add_argument("--step", type=int, default=1, help="every Nth frame pair (default 1)")
    ap.add_argument("--window", type=int, default=12, help="frames either side matched against each other (default 12)")
    ap.add_argument("--masks", default=None, help="folder with {left,right}/<frame name> masks, 0 = ignore")
    ap.add_argument("--from-stage", default="workspace", choices=STAGES, help="resume from this stage")
    ap.add_argument("--gpu", default="0", help="GPU index for COLMAP, or -1 for CPU")
    return ap.parse_args(argv)


def link_masks(ws, masks, names):
    """COLMAP looks for <mask_path>/<image name>.png; link the given masks under those names."""
    folder = os.path.join(ws, "masks")
    for camera in CAMERAS:
        os.makedirs(os.path.join(folder, camera), exist_ok=True)
        for name in names:
            source = os.path.abspath(os.path.join(masks, camera, name))
            link = os.path.join(folder, camera, name + ".png")
            if os.path.exists(source) and not os.path.lexists(link):
                os.symlink(source, link)
    return folder


def frame_poses(ws, rig, names):
    """{frame name: T_cam_world of the LEFT camera} from the largest model. A frame whose left image did
    not register but whose right one did is posed through the calibrated extrinsics."""
    images = colmap_mvs.read_model_images(ws, colmap_mvs.best_model(ws))
    T_right_left = np.eye(4)
    T_right_left[:3, :3], T_right_left[:3, 3] = rig.R, rig.T.ravel()
    poses = {}
    for name in names:
        if f"left/{name}" in images:
            poses[name] = images[f"left/{name}"]
        elif f"right/{name}" in images:
            poses[name] = invert(T_right_left) @ images[f"right/{name}"]
    return poses


def main(argv=None):
    args = parse_args(argv)
    rig = load_rig(args.calibration)
    ws = os.path.abspath(args.workspace)
    os.makedirs(ws, exist_ok=True)
    names = [name for name, _, _ in find_frame_pairs(args.frames_dir)][::args.step]
    run = STAGES[STAGES.index(args.from_stage):]
    start = time.time()
    stage_seconds = {}   # wall time of each SfM stage run, for the frame-count sweep's timing breakdown

    def timed(stage, function, *stage_args, **stage_kwargs):
        print(f"== {stage}")
        stage_start = time.time()
        result = function(*stage_args, **stage_kwargs)
        stage_seconds[stage] = round(time.time() - stage_start, 1)
        return result

    if "workspace" in run:
        names = timed("workspace", colmap_mvs.stage_workspace, ws, args.frames_dir, names)
    mask_path = link_masks(ws, args.masks, names) if args.masks else None
    if "features" in run:
        timed("features", colmap_mvs.stage_features, ws, rig, names, args.gpu, mask_path=mask_path)
    has_rig = True
    if "rig" in run:
        has_rig = timed("rig", colmap_mvs.stage_rig, ws, rig)
    if "match" in run:
        timed("match", colmap_mvs.stage_match, ws, names, args.window, args.gpu)
    if "map" in run:
        timed("map", colmap_mvs.stage_map, ws, has_rig)
    seconds = time.time() - start

    poses = frame_poses(ws, rig, names)
    first = next((n for n in names if n in poses), None)
    if first is None:
        raise SystemExit("SfM registered no frame")
    T_world_first = invert(poses[first])   # world := left camera of the first registered frame
    frames = [{"name": n, "T_cam_world": poses[n] @ T_world_first if n in poses else None} for n in names]
    write_trajectory(args.out, "src/03_Reconstruction/colmap_poses.py",
                     f"left camera of frame {first}, the first frame SfM registered",
                     {"frames_dir": rel(args.frames_dir), "calibration": rel(args.calibration),
                      "workspace": rel(ws), "step": args.step, "window": args.window,
                      "masks": rel(args.masks) if args.masks else "", "frames": len(names),
                      "frames_posed": len(poses), "seconds": round(seconds, 1),
                      **{f"seconds_{stage}": value for stage, value in stage_seconds.items()}},
                     frames)
    print(f"SfM posed {len(poses)} of {len(names)} frame pairs ({seconds:.0f} s) -> {rel(args.out)}")


if __name__ == "__main__":
    main()
