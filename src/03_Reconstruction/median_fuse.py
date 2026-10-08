"""Fuse saved frames into one surface by taking, per x-y cell, the MEDIAN of the frames' heights.

dense_stereo.py fuses by voxel union: every voxel enough frames hit is kept. That is right when the
frames agree, but on Oct1/cub2 they do not. Each frame is clean on its own (0.6 mm scatter on a stud
top), yet frames disagree on the stud heights by ~4 mm (p10-p90), and systematically with viewing
angle (frames from further off the normal put the studs higher): a view-dependent depth bias such as
the flat-port refraction a pinhole calibration leaves in. Rigid re-alignment (refine_frames.py) cannot
remove it, and a voxel union stacks every frame's version into a band several mm thick.

Here each frame contributes one height per cell (the median of its points in the cell), and the
output height is the median over the frames that saw the cell: one consensus surface, the middle of
the band, with the spread across frames kept as a per-cell quality measure. Every view looks at the
surface from one side (here: the board side), so a height field over the board's x-y holds it; walls
steeper than the cell size are represented only by the jump between neighbouring cells.

    src/venv/bin/python src/03_Reconstruction/median_fuse.py --session Oct1 --scan cub2_all \
        --frames raft_s075_frames raft_s075_frames_odd --corrections raft_s075_frames_refined --out raft_median

Writes <out>/cloud_clean.ply (one point per cell, coloured by grey), spread.ply (the same points
coloured by the frames' spread in that cell), mesh.ply (the grid triangulated, cells joined where
their heights differ by less than --max-step mm) and fuse.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import rel, transform  # noqa: E402
import project_paths as paths  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Median height-field fusion of saved frames.")
    paths.add_scan_arguments(ap)
    ap.add_argument("--frames", nargs="+", required=True, help="method folders whose frames/*.npz to use")
    ap.add_argument("--corrections", nargs="*", default=[],
                    help="method folders with refine_frames.py corrections.json to apply (frames not in them are used as saved)")
    ap.add_argument("--out", required=True, help="output method folder")
    ap.add_argument("--cell", type=float, default=1.0, help="mm, grid cell (default 1)")
    ap.add_argument("--min-frames", type=int, default=5, help="frames a cell needs (default 5)")
    ap.add_argument("--max-spread", type=float, default=None,
                    help="mm; drop cells whose frames disagree more than this (p10-p90; default keep all)")
    ap.add_argument("--max-step", type=float, default=4.0,
                    help="mm; neighbouring cells further apart in height are not joined in the mesh (default 4)")
    args = ap.parse_args(argv)
    paths.require_scan(args, ap)
    return args


def load(args):
    corrections = {}
    for method in args.corrections:
        with open(os.path.join(paths.method_dir(args.session, args.scan, method), "corrections.json")) as f:
            corrections.update({k: np.array(v) for k, v in json.load(f)["frames"].items()})
    frames, corrected = [], 0
    for method in args.frames:
        for path in sorted(glob.glob(os.path.join(paths.method_dir(args.session, args.scan, method), "frames", "*.npz"))):
            data = np.load(path)
            name = os.path.basename(path)[:-4]
            xyz = data["xyz"].astype(np.float64)
            if name in corrections:
                xyz = transform(corrections[name], xyz)
                corrected += 1
            frames.append((xyz, data["grey"].astype(np.float64)))
    print(f"{len(frames)} frames from {', '.join(args.frames)} ({corrected} with refined poses)")
    return frames


def frame_cells(xyz, grey, origin, shape, cell):
    """(flat cell index, median height, mean grey) for every cell one frame hits."""
    ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    inside = np.all((ij >= 0) & (ij < shape), axis=1)
    flat = ij[inside, 0] * shape[1] + ij[inside, 1]
    z, g = xyz[inside, 2], grey[inside]
    order = np.lexsort((z, flat))
    flat, z, g = flat[order], z[order], g[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    counts = np.diff(np.r_[starts, len(flat)])
    return flat[starts], z[starts + counts // 2], np.add.reduceat(g, starts) / counts


def grid_mesh(height, origin, cell, max_step):
    """Triangulate the height grid: two triangles per 2x2 block of filled cells, unless it spans a step."""
    filled = ~np.isnan(height)
    index = -np.ones(height.shape, np.int64)
    index[filled] = np.arange(filled.sum())
    i, j = np.nonzero(filled)
    vertices = np.c_[origin[0] + (i + 0.5) * cell, origin[1] + (j + 0.5) * cell, height[filled]]
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
    za, zb, zc, zd = height[:-1, :-1], height[1:, :-1], height[:-1, 1:], height[1:, 1:]
    triangles = []
    for (p, q, r), (zp, zq, zr) in (((a, b, d), (za, zb, zd)), ((a, d, c), (za, zd, zc))):
        ok = (p >= 0) & (q >= 0) & (r >= 0)
        with np.errstate(invalid="ignore"):
            ok &= (np.nanmax([zp, zq, zr], 0) - np.nanmin([zp, zq, zr], 0)) <= max_step
        triangles.append(np.c_[p[ok], q[ok], r[ok]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(np.vstack(triangles)))
    mesh.compute_vertex_normals()
    return mesh


def main():
    args = parse_args()
    frames = load(args)
    everything = np.vstack([f[0] for f in frames])
    origin = np.floor(everything[:, :2].min(0))
    shape = tuple((np.ceil((everything[:, :2].max(0) - origin) / args.cell) + 1).astype(int))
    del everything

    flat_all, z_all, g_all = [], [], []
    for xyz, grey in frames:
        flat, z, g = frame_cells(xyz, grey, origin, shape, args.cell)
        flat_all.append(flat), z_all.append(z), g_all.append(g)
    flat, z, g = np.concatenate(flat_all), np.concatenate(z_all), np.concatenate(g_all)
    order = np.lexsort((z, flat))
    flat, z, g = flat[order], z[order], g[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    counts = np.diff(np.r_[starts, len(flat)])
    keep = counts >= args.min_frames
    starts, counts = starts[keep], counts[keep]
    cells = flat[starts]
    median = z[starts + counts // 2]
    spread = z[starts + (counts * 9) // 10] - z[starts + counts // 10]
    grey = np.add.reduceat(g, np.r_[0, np.flatnonzero(np.diff(flat)) + 1])[keep] / counts
    if args.max_spread is not None:
        ok = spread <= args.max_spread
        cells, median, spread, grey, counts = cells[ok], median[ok], spread[ok], grey[ok], counts[ok]
    print(f"{len(cells)} cells seen by >= {args.min_frames} frames; frames per cell median {np.median(counts):.0f}; "
          f"spread across frames (p10-p90) median {np.median(spread):.2f} mm")

    i, j = np.divmod(cells, shape[1])
    xyz = np.c_[origin[0] + (i + 0.5) * args.cell, origin[1] + (j + 0.5) * args.cell, median]
    out_dir = paths.method_dir(args.session, args.scan, args.out)
    os.makedirs(out_dir, exist_ok=True)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(np.repeat(np.clip(grey, 0, 255)[:, None] / 255.0, 3, 1))
    o3d.io.write_point_cloud(os.path.join(out_dir, paths.CLOUD_CLEAN), cloud)
    import matplotlib
    shown = o3d.geometry.PointCloud(cloud)
    shown.colors = o3d.utility.Vector3dVector(matplotlib.colormaps["viridis"](np.clip(spread / 6.0, 0, 1))[:, :3])
    o3d.io.write_point_cloud(os.path.join(out_dir, "spread.ply"), shown)
    height = np.full(shape, np.nan)
    height[i, j] = median
    mesh = grid_mesh(height, origin, args.cell, args.max_step)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, paths.MESH), mesh)
    with open(os.path.join(out_dir, "fuse.json"), "w") as f:
        json.dump({"frames": args.frames, "corrections": args.corrections, "n_frames": len(frames), "cell_mm": args.cell,
                   "min_frames": args.min_frames, "max_spread_mm": args.max_spread, "cells": int(len(cells)),
                   "spread_median_mm": float(np.median(spread)), "frames_per_cell_median": float(np.median(counts))}, f, indent=2)
    print(f"-> {rel(out_dir)}/: cloud_clean.ply, spread.ply (0-6 mm, viridis), mesh.ply ({len(mesh.triangles)} triangles)")


if __name__ == "__main__":
    main()
