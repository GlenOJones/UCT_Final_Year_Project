"""Rig pose for every frame of a scan by registering each frame's depth onto the frames before it (ICP).

Reads depth_frames.py's per-frame points (left camera frame) and chains them: each frame is registered
onto what came before, and its pose is the previous pose times the registration. Two choices:

  --target f2f   frame-to-frame: onto the previous frame alone. Simple; every error carries forward.
  --target map   frame-to-local-map: onto the last --map-frames frames, already posed and merged, so
                 one bad frame pulls the chain less and the overlap is larger.

  --metric plane    point-to-plane ICP on geometry alone.
  --metric colored  Park et al.'s coloured ICP: geometry plus the grey level. Texture holds the
                    registration where the surface is flat and point-to-plane can slide.

Registration runs coarse-to-fine over --scales (voxel sizes, mm) from a guess: the previous pose moved by
the TRANSLATION of the last registered step, with no rotation. Repeating the rotation too was tried first and
fed itself on 3DPrintFinal s30: on a mostly flat scene ICP has almost no restoring force against a small
rotation-plus-slide, so it kept whatever the guess offered, the next guess repeated it, and the step's
rotation grew ~1.5x per frame (0.6 deg to 50 deg in 11 frames, fitness 0.99 throughout). A frame is marked
LOST when its final fitness (share of its points with a match within the search distance) is below
--min-fitness, or when the registration moved the rig further than a trolley can between two frames
(--max-step mm, --max-rotation deg). A lost frame keeps the previous frame's pose, is left out of the
local map, and is flagged so the evaluation counts it as a failure. Without this, one wild step is
repeated by the constant-velocity guess and the whole rest of the trajectory flies off.

Every frame also records how well the registration pinned the TRANSLATION: the smallest eigenvalue of
the translational block of ICP's information matrix over the largest. Near 0 means the surface left
the frame free to slide in some direction (a flat board, featureless ice); point-to-plane ICP then
drifts along it. This is the number to read before trusting ICP on a new surface.

    src/venv/bin/python src/03_Reconstruction/icp_poses.py --depth-dir work/3DPrintFinal/s30_range/depth/sgbm \
        --out results/3DPrintFinal/s30_range/poses/icp_sgbm_map_plane.yaml --target map --metric plane

The world frame is the left camera of the first frame.
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_files import write_trajectory  # noqa: E402
from stereo_rig import invert, rel  # noqa: E402

reg = o3d.pipelines.registration


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-frame rig poses by ICP on per-frame depth.")
    ap.add_argument("--depth-dir", required=True, help="depth_frames.py output folder (with frames/*.npz)")
    ap.add_argument("--out", required=True, help="trajectory YAML to write")
    ap.add_argument("--target", default="map", choices=("f2f", "map"),
                    help="register onto the previous frame (f2f) or a local map of recent frames (default map)")
    ap.add_argument("--metric", default="plane", choices=("plane", "colored"),
                    help="point-to-plane, or coloured ICP using the grey level too (default plane)")
    ap.add_argument("--scales", type=float, nargs="+", default=[4.0, 2.0],
                    help="voxel sizes, mm, coarse to fine (default 4 2)")
    ap.add_argument("--max-distance", type=float, default=2.5,
                    help="correspondence search distance as a multiple of the voxel size (default 2.5)")
    ap.add_argument("--iterations", type=int, default=30, help="ICP iterations per scale (default 30)")
    ap.add_argument("--map-frames", type=int, default=10, help="frames in the local map for --target map (default 10)")
    ap.add_argument("--min-fitness", type=float, default=0.3,
                    help="below this final fitness a frame is marked lost (default 0.3)")
    ap.add_argument("--max-step", type=float, default=20.0,
                    help="mm; a registration that moves the rig further than this from the previous frame is "
                         "marked lost (default 20)")
    ap.add_argument("--max-rotation", type=float, default=5.0,
                    help="deg; likewise for rotation (default 5)")
    return ap.parse_args(argv)


def load_frame(path, scales):
    """{voxel: point cloud in the camera frame, with normals facing the camera and grey as colour}."""
    data = np.load(path)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["xyz"].astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.repeat(data["grey"][:, None] / 255.0, 3, 1))
    levels = {}
    for voxel in scales:
        down = cloud.voxel_down_sample(voxel)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
        down.orient_normals_towards_camera_location(np.zeros(3))
        levels[voxel] = down
    return levels


def register(source, target, init, args):
    """Coarse-to-fine registration of source (camera frame) onto target (world): (T_world_cam, fitness,
    rmse, translation constraint ratio) at the finest scale."""
    T = init
    for voxel in args.scales:
        distance = args.max_distance * voxel
        criteria = reg.ICPConvergenceCriteria(max_iteration=args.iterations)
        if args.metric == "colored":
            result = reg.registration_colored_icp(source[voxel], target[voxel], distance, T,
                                                  reg.TransformationEstimationForColoredICP(), criteria)
        else:
            result = reg.registration_icp(source[voxel], target[voxel], distance, T,
                                          reg.TransformationEstimationPointToPlane(), criteria)
        T = result.transformation
    finest = args.scales[-1]
    return T, result.fitness, result.inlier_rmse, translation_constraint(source[finest], target[finest],
                                                                           args.max_distance * finest, T)


def translation_constraint(source, target, distance, T):
    """Smallest over largest eigenvalue of sum(n n^T) over the matched target normals: the translational
    block of the point-to-plane normal equations. Each match only pins translation along its normal, so a
    surface whose normals all lie near one plane or line leaves a direction free and the ratio near 0.
    (Open3D's get_information_matrix_from_point_clouds uses point-to-point terms, whose translational
    block is always a multiple of the identity, so it cannot show this.)"""
    moved = np.asarray(source.points) @ T[:3, :3].T + T[:3, 3]
    distances, index = cKDTree(np.asarray(target.points)).query(moved, distance_upper_bound=distance)
    normals = np.asarray(target.normals)[index[np.isfinite(distances)]]
    if len(normals) < 3:
        return 0.0
    eig = np.linalg.eigvalsh(normals.T @ normals)
    return float(max(eig[0], 0.0) / eig[-1]) if eig[-1] > 0 else 0.0


def to_world(levels, T_world_cam):
    moved = {}
    for voxel, cloud in levels.items():
        copy = o3d.geometry.PointCloud(cloud)
        copy.transform(T_world_cam)
        moved[voxel] = copy
    return moved


def merge(recent, scales):
    """The local map: the recent frames (already in the world frame) merged and thinned per scale."""
    merged = {}
    for voxel in scales:
        cloud = o3d.geometry.PointCloud()
        for frame in recent:
            cloud += frame[voxel]
        merged[voxel] = cloud.voxel_down_sample(voxel)
    return merged

def main(argv=None):
    args = parse_args(argv)
    paths = sorted(glob.glob(os.path.join(args.depth_dir, "frames", "*.npz")))
    if not paths:
        raise SystemExit(f"no frames in {rel(args.depth_dir)}/frames - run depth_frames.py first")
    with open(os.path.join(args.depth_dir, "depth_frames.json"), encoding="utf-8") as fh:
        depth_record = json.load(fh)
    names = [os.path.basename(p)[:-4] + ".png" for p in paths]
    print(f"{len(paths)} frames from {rel(args.depth_dir)}; target {args.target}, metric {args.metric}, "
          f"scales {args.scales} mm")

    start = time.time()
    recent = collections.deque(maxlen=args.map_frames if args.target == "map" else 1)
    poses, lost_flags, frames = [], [], []
    last_step = np.eye(4)   # last registered frame-to-frame motion (T_prev_cam)
    for k, path in enumerate(paths):
        levels = load_frame(path, args.scales)
        if k == 0:
            T, fitness, rmse, ratio, lost = np.eye(4), 1.0, 0.0, None, False
        else:
            # The last registered step's translation, without its rotation (see the docstring).
            glide = np.eye(4)
            glide[:3, 3] = last_step[:3, 3]
            guess = poses[-1] @ glide
            target = recent[0] if args.target == "f2f" else merge(recent, args.scales)
            try:
                T, fitness, rmse, ratio = register(levels, target, guess, args)
            except RuntimeError as error:   # coloured ICP raises when nothing is within the search distance
                print(f"  {names[k]}: registration failed ({str(error).splitlines()[0][-60:]})")
                T, fitness, rmse, ratio = guess, 0.0, 0.0, 0.0
            step = invert(poses[-1]) @ T
            angle = np.degrees(np.arccos(np.clip((np.trace(step[:3, :3]) - 1) / 2, -1, 1)))
            lost = (fitness < args.min_fitness or np.linalg.norm(step[:3, 3]) > args.max_step
                    or angle > args.max_rotation)
            if lost:
                T = guess
            else:
                last_step = step
        poses.append(T)
        lost_flags.append(lost)
        if not lost:
            recent.append(to_world(levels, T))
        frames.append({"name": names[k], "T_cam_world": invert(T), "fitness": fitness, "rmse_mm": rmse,
                       "translation_constraint": ratio, "lost": float(lost)})
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(paths)}  fitness {fitness:.2f}, rmse {rmse:.2f} mm, "
                  f"translation constraint {ratio if ratio is not None else float('nan'):.3f}, "
                  f"{time.time() - start:.0f} s", flush=True)
    seconds = time.time() - start

    lost = sum(f["lost"] for f in frames)
    write_trajectory(args.out, "src/03_Reconstruction/icp_poses.py", f"left camera of frame {names[0]}",
                     {"depth_dir": rel(args.depth_dir), "frames_dir": rel(depth_record["settings"]["frames_dir"]),
                      "calibration": rel(depth_record["settings"]["calibration"]), "matcher": depth_record["settings"]["matcher"],
                      "target": args.target, "metric": args.metric, "scales_mm": " ".join(f"{s:g}" for s in args.scales),
                      "max_distance_voxels": args.max_distance, "iterations": args.iterations,
                      "map_frames": args.map_frames, "min_fitness": args.min_fitness, "max_step_mm": args.max_step,
                      "max_rotation_deg": args.max_rotation, "frames": len(frames),
                      "frames_posed": int(len(frames) - lost), "seconds": round(seconds, 1),
                      "depth_seconds": depth_record.get("seconds") or 0.0},
                     frames)
    print(f"ICP posed {len(frames) - int(lost)} of {len(frames)} frames ({int(lost)} lost) in {seconds:.0f} s "
          f"-> {rel(args.out)}")


if __name__ == "__main__":
    main()
