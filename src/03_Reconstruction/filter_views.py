"""Make stricter-voted versions of a fused cloud without re-running the matcher.

dense_stereo.py keeps a voxel if at least --min-views frames put a point in it (default 3), and
records that count in the PLY's "views" field. Raising the threshold afterwards is equivalent to
re-running with a higher --min-views (checked: raft_v5 matched a --min-views 5 run exactly), and
takes seconds instead of minutes.

    src/venv/bin/python src/03_Reconstruction/filter_views.py --session Sep24 --scan mjpg_pyr_lights_2_all --method raft_s075 15 18

writes <method>_v15/cloud.ply and <method>_v18/cloud.ply next to <method>/. Run postprocess_cloud.py
and compare_to_cad.py on them as on any method folder.

Stricter voting trades coverage for accuracy: on mjpg_pyr_lights_2_all (RAFT, 212 frames) at least
3 frames gave 96% coverage at 1.81 mm median error, at least 15 gave 82% at 1.00 mm, and at least 24
gave 60% at 0.72 mm (on_local_board fit). More frames per scan make stricter thresholds affordable.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_stereo  # noqa: E402
import project_paths as paths  # noqa: E402
from view_cloud import read_ply  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Filter a fused cloud by the number of frames that saw each point.")
    paths.add_scan_arguments(ap, method="sgbm")
    ap.add_argument("thresholds", type=int, nargs="+", help="minimum views to keep, one output per value")
    args = ap.parse_args(argv)
    paths.require_scan(args, ap)
    return args


def main():
    args = parse_args()
    source = os.path.join(paths.method_dir(args.session, args.scan, args.method), paths.CLOUD)
    xyz, views = read_ply(source)
    if views.max() <= 1:
        raise SystemExit(f"{paths.rel(source)} has no views field (only dense_stereo.py clouds do)")
    grey = np.asarray(o3d.io.read_point_cloud(source).colors)[:, 0] * 255
    for k in args.thresholds:
        keep = views >= k
        out_dir = paths.method_dir(args.session, args.scan, f"{args.method}_v{k}")
        os.makedirs(out_dir, exist_ok=True)
        dense_stereo.write_ply(os.path.join(out_dir, paths.CLOUD), xyz[keep], grey[keep], views[keep])
        print(f"{args.method}_v{k}: {keep.sum()} of {len(xyz)} points seen from >= {k} frames -> {paths.rel(out_dir)}/")


if __name__ == "__main__":
    main()
