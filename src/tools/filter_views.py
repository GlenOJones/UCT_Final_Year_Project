"""Make stricter-voted versions of a fused cloud without re-running the matcher.

    src/venv/bin/python src/tools/filter_views.py --session Sep24 --scan mjpg_pyr_lights_2_all --method raft_s075 15 18
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "03_depth"))
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
