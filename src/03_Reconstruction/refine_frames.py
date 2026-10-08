"""Re-align every frame of a dense_stereo.py run to the others, then fuse again.

Each frame is placed in the board frame by its own tag pose (tag_poses.py). Small errors in those
poses do not average out: they stack frames at slightly different heights and tilts, so a flat face
comes out as a slab mm thick, and the thicker the farther it is from the tags, because a rotation
error grows with the lever arm (Oct1/cub2: 4.5 mm thick on the tag block, 7-11 mm on the cube block
150-400 mm away). This script removes that error, as a multi-view refinement:

  1. fuse all frames (1 mm voxels, kept if seen from --min-views frames) into a reference model;
  2. align every frame to the reference with robust point-to-plane ICP, starting from its tag pose;
     a correction larger than --max-shift mm / --max-turn deg is refused and the frame keeps its pose;
  3. remove the corrections' common part, so the result stays in the tag-defined board frame;
  4. rebuild the reference from the corrected frames and repeat, with a tighter correspondence
     distance each round (--distances).

    src/venv/bin/python src/03_Reconstruction/refine_frames.py --session Oct1 --scan cub2_all --method raft_s075_frames

Needs a run made with dense_stereo.py --save-frames (<method>/frames/*.npz). Writes
<method>_refined/cloud.ply (same format as dense_stereo.py's, with "views") and corrections.json.
Run postprocess_cloud.py on it as on any method.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_stereo import VoxelGrid, write_ply  # noqa: E402
from merge_scans import motion  # noqa: E402
from stereo_rig import mean_rotation, rel, transform  # noqa: E402
import project_paths as paths  # noqa: E402

ICP_VOXEL = 2.0     # mm, frames are downsampled to this for ICP


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Re-align the frames of a dense_stereo.py --save-frames run and re-fuse.")
    paths.add_scan_arguments(ap, method="raft_s075_frames")
    ap.add_argument("--out", help="output method folder (default <method>_refined)")
    ap.add_argument("--distances", type=float, nargs="+", default=[5.0, 3.0, 2.0, 1.5],
                    help="mm, ICP correspondence distance per round; one round each (default 5 3 2 1.5)")
    ap.add_argument("--min-views", type=int, default=3, help="frames a voxel needs, for the reference and the output (default 3)")
    ap.add_argument("--max-shift", type=float, default=20.0, help="mm, largest correction accepted (default 20)")
    ap.add_argument("--max-turn", type=float, default=3.0, help="deg, largest correction accepted (default 3)")
    args = ap.parse_args(argv)
    paths.require_scan(args, ap)
    args.out = args.out or f"{args.method}_refined"
    return args


def load_frames(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        raise SystemExit(f"no frames in {rel(folder)}: run dense_stereo.py with --save-frames first")
    frames = []
    for f in files:
        data = np.load(f)
        frames.append({"name": os.path.basename(f)[:-4], "xyz": data["xyz"].astype(np.float64),
                       "grey": data["grey"].astype(np.float64)})
    return frames


def fuse(frames, corrections, min_views):
    grid = VoxelGrid(1.0)
    for k, (frame, T) in enumerate(zip(frames, corrections)):
        grid.add(transform(T, frame["xyz"]), frame["grey"])
        if (k + 1) % 20 == 0:
            grid.merge()
    return grid.points(min_views)


def remove_common(corrections):
    """Take out the mean correction, so refinement does not move the whole model off the tag frame."""
    R = mean_rotation(np.array([T[:3, :3] for T in corrections]))
    t = np.mean([T[:3, 3] for T in corrections], axis=0)
    common = np.eye(4)
    common[:3, :3], common[:3, 3] = R, t
    undo = np.linalg.inv(common)
    return [undo @ T for T in corrections], common


def main():
    args = parse_args()
    in_dir = paths.method_dir(args.session, args.scan, args.method)
    frames = load_frames(os.path.join(in_dir, "frames"))
    print(f"{len(frames)} frames from {rel(in_dir)}/frames")
    sources = []
    for frame in frames:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame["xyz"])).voxel_down_sample(ICP_VOXEL)
        sources.append(cloud)

    corrections = [np.eye(4) for _ in frames]
    history = []
    for round_index, distance in enumerate(args.distances):
        start = time.time()
        xyz, _, _ = fuse(frames, corrections, args.min_views)
        reference = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=distance / 2))
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        updated, refused, rmse = [], 0, []
        for source, T in zip(sources, corrections):
            result = o3d.pipelines.registration.registration_icp(source, reference, distance, T, estimation, criteria)
            shift, turn = motion(np.asarray(result.transformation))
            if shift > args.max_shift or turn > args.max_turn or result.fitness < 0.2:
                updated.append(T)
                refused += 1
            else:
                updated.append(np.asarray(result.transformation))
                rmse.append(result.inlier_rmse)
        corrections, common = remove_common(updated)
        sizes = np.array([motion(T) for T in corrections])
        history.append({"distance_mm": distance, "reference_points": len(xyz), "refused": refused,
                        "median_shift_mm": float(np.median(sizes[:, 0])), "p90_shift_mm": float(np.percentile(sizes[:, 0], 90)),
                        "median_turn_deg": float(np.median(sizes[:, 1])), "p90_turn_deg": float(np.percentile(sizes[:, 1], 90)),
                        "median_inlier_rmse_mm": float(np.median(rmse)) if rmse else None,
                        "common_removed_mm_deg": list(motion(common))})
        h = history[-1]
        print(f"round {round_index + 1} ({distance} mm): reference {len(xyz)} points; corrections median "
              f"{h['median_shift_mm']:.2f} mm / {h['median_turn_deg']:.3f} deg, p90 {h['p90_shift_mm']:.2f} mm / "
              f"{h['p90_turn_deg']:.3f} deg; inlier rmse {h['median_inlier_rmse_mm']:.2f} mm; {refused} refused; "
              f"{time.time() - start:.0f} s")

    xyz, grey, views = fuse(frames, corrections, args.min_views)
    out_dir = paths.method_dir(args.session, args.scan, args.out)
    os.makedirs(out_dir, exist_ok=True)
    write_ply(os.path.join(out_dir, paths.CLOUD), xyz, grey, views)
    with open(os.path.join(out_dir, "corrections.json"), "w") as f:
        json.dump({"source": rel(in_dir), "rounds": history,
                   "frames": {fr["name"]: T.tolist() for fr, T in zip(frames, corrections)}}, f, indent=1)
    print(f"\n{len(xyz)} points seen from >= {args.min_views} frames -> {rel(os.path.join(out_dir, paths.CLOUD))}")


if __name__ == "__main__":
    main()
