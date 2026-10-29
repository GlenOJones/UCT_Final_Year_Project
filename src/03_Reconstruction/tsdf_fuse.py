"""Fuse posed depth frames into one surface with a TSDF (truncated signed distance function).

    src/venv/bin/python src/03_Reconstruction/tsdf_fuse.py \
        --depth-dir work/experiments/pose_comparison/ice_pyramid_shade_pyr1_pass4/depth/foundationstereo \
        --poses results/experiments/pose_comparison/ice_pyramid_shade_pyr1_pass4/tag_poses.yaml \
        --out-dir work/experiments/fusion/ice_shade/tags_tsdf
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_files  # noqa: E402
from stereo_rig import rel  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="TSDF fusion of posed depth frames.")
    ap.add_argument("--depth-dir", required=True, help="depth_frames.py --out-dir (frames/<frame>.npz)")
    ap.add_argument("--poses", required=True, help="trajectory: tag_poses.py, colmap_poses.py or icp_poses.py YAML")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voxel", type=float, default=1.0, help="mm, TSDF voxel size (default 1)")
    ap.add_argument("--trunc", type=float, default=4.0,
                    help="mm, truncation distance: how far either side of a measured surface a frame updates "
                         "voxels; must exceed the depth noise and the frames' disagreement (default 4)")
    ap.add_argument("--min-weight", type=float, default=3.0,
                    help="keep surface only where at least this many frames contributed, the TSDF counterpart "
                         "of dense_stereo.py's --min-views (default 3)")
    ap.add_argument("--pixel-mm", type=float, default=1.2,
                    help="mm, width of a rendered depth pixel at the frame's near depth; above the frames' "
                         "point spacing (depth_frames.py --voxel) so the rendering has no holes (default 1.2)")
    ap.add_argument("--depth-min", type=float, default=0.0, help="mm, ignore frame points nearer than this camera z")
    ap.add_argument("--depth-max", type=float, default=1e5, help="mm, ignore frame points further than this camera z")
    ap.add_argument("--mesh", action="store_true", help="also extract and write the triangle mesh")
    ap.add_argument("--block-resolution", type=int, default=4,
                    help="voxels along a block's edge; small blocks use less memory (default 4)")
    ap.add_argument("--block-count", type=int, default=20000,
                    help="initial number of blocks allocated (grows if needed; default 20000)")
    return ap.parse_args(argv)


def render_depth(xyz, grey, pixel_mm):
    """A depth image (mm, float32, 0 = empty), grey image and pinhole intrinsics (3x3) holding xyz, seen from the camera
    centre: nearest point per pixel, each pixel pixel_mm wide at the near depth."""
    z = xyz[:, 2]
    focal = np.percentile(z, 2) / pixel_mm
    u, v = focal * xyz[:, 0] / z, focal * xyz[:, 1] / z
    cx, cy = 0.5 - np.floor(u.min()), 0.5 - np.floor(v.min())
    col, row = np.floor(u + cx).astype(np.int64), np.floor(v + cy).astype(np.int64)
    width, height = int(col.max()) + 1, int(row.max()) + 1
    # Nearest point per pixel: sort by pixel, then depth, and keep each pixel's first.
    flat = row * width + col
    order = np.lexsort((z, flat))
    first = order[np.r_[True, np.diff(flat[order]) != 0]]
    depth = np.zeros(height * width, np.float32)
    image = np.zeros((height * width, 3), np.float32)
    depth[flat[first]] = z[first]
    image[flat[first]] = (grey[first] / 255.0)[:, None]
    K = np.array([[focal, 0, cx - 0.5], [0, focal, cy - 0.5], [0, 0, 1]])
    return depth.reshape(height, width), image.reshape(height, width, 3), K


def load_frames(args):
    """[(frame name, T_cam_world)] with a pose and a depth file, in frame order, and the names skipped."""
    _, poses, diagnostics = pose_files.read_trajectory(args.poses)
    frames_dir = os.path.join(args.depth_dir, "frames")
    names = sorted(diagnostics) if diagnostics else sorted(poses)
    used, no_pose, no_depth = [], [], []
    for name in names:
        if name not in poses:
            no_pose.append(name)
        elif not os.path.exists(os.path.join(frames_dir, os.path.splitext(name)[0] + ".npz")):
            no_depth.append(name)
        else:
            used.append((name, poses[name]))
    if not used:
        raise SystemExit(f"no frame has both a pose in {rel(args.poses)} and depth in {rel(frames_dir)}")
    return used, no_pose, no_depth


def main(argv=None):
    args = parse_args(argv)
    frames, no_pose, no_depth = load_frames(args)
    print(f"{len(frames)} frames to fuse ({len(no_pose)} without a pose, {len(no_depth)} without depth); "
          f"voxel {args.voxel} mm, trunc {args.trunc} mm, min weight {args.min_weight:g}")
    device = o3d.core.Device("CPU:0")
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3d.core.float32,) * 3,
        attr_channels=((1), (1), (3)), voxel_size=args.voxel, block_resolution=args.block_resolution,
        block_count=args.block_count, device=device)
    start = time.perf_counter()
    for k, (name, T_cam_world) in enumerate(frames):
        data = np.load(os.path.join(args.depth_dir, "frames", os.path.splitext(name)[0] + ".npz"))
        xyz, grey = data["xyz"].astype(np.float64), data["grey"].astype(np.float32)
        keep = (xyz[:, 2] >= args.depth_min) & (xyz[:, 2] <= args.depth_max)
        if keep.sum() < 100:
            continue
        depth, image, K = render_depth(xyz[keep], grey[keep], args.pixel_mm)
        depth_t = o3d.t.geometry.Image(o3d.core.Tensor(depth)).to(device)
        color_t = o3d.t.geometry.Image(o3d.core.Tensor(image)).to(device)
        K_t = o3d.core.Tensor(K, o3d.core.float64)
        T_t = o3d.core.Tensor(np.asarray(T_cam_world, np.float64))
        depth_max = float(depth.max()) + args.trunc
        blocks = grid.compute_unique_block_coordinates(depth_t, K_t, T_t, depth_scale=1.0, depth_max=depth_max,
                                                       trunc_voxel_multiplier=args.trunc / args.voxel)
        grid.integrate(blocks, depth_t, color_t, K_t, T_t, depth_scale=1.0, depth_max=depth_max,
                       trunc_voxel_multiplier=args.trunc / args.voxel)
        if (k + 1) % 20 == 0 or k + 1 == len(frames):
            print(f"  {k + 1}/{len(frames)}  {name}: {keep.sum()} points, {depth.shape[1]}x{depth.shape[0]} px, "
                  f"{time.perf_counter() - start:.0f} s", flush=True)
    seconds = time.perf_counter() - start

    os.makedirs(args.out_dir, exist_ok=True)
    cloud = grid.extract_point_cloud(weight_threshold=args.min_weight).to_legacy()
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "cloud.ply"), cloud)
    triangles = None
    voxels = grid.hashmap().size() * args.block_resolution ** 3
    if args.mesh and voxels * 16 >= 2 ** 31:
        print(f"--mesh skipped: {voxels:,} voxels is past Open3D's extraction limit (~134 M); use smaller "
              f"--block-resolution, a coarser --voxel or a tighter --depth-max")
    elif args.mesh:
        mesh = grid.extract_triangle_mesh(weight_threshold=args.min_weight).to_legacy()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(args.out_dir, "mesh.ply"), mesh)
        triangles = len(mesh.triangles)
    record = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
              "script": "src/03_Reconstruction/tsdf_fuse.py", "argv": sys.argv[1:] if argv is None else argv,
              "settings": {k: (rel(v) if k in ("depth_dir", "poses", "out_dir") else v) for k, v in vars(args).items()},
              "frames_fused": len(frames), "frames_without_pose": no_pose, "frames_without_depth": no_depth,
              "seconds": round(seconds, 1), "seconds_per_frame": round(seconds / len(frames), 3),
              "voxels": voxels, "points": len(cloud.points), "triangles": triangles}
    with open(os.path.join(args.out_dir, "tsdf_fuse.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    print(f"{len(frames)} frames in {seconds:.0f} s -> {len(cloud.points)} points"
          + (f", {triangles} triangles" if triangles is not None else "") + f" -> {rel(args.out_dir)}/")


if __name__ == "__main__":
    main()
