"""Compare reconstructions of the same scan, from the clouds the pipeline saved.

All inputs are board-frame clouds (dense_stereo.py, colmap_mvs.py, or their _clean versions from
postprocess_cloud.py). For each one it reports, on the same crop box:
  - points on the OBJECT: in the box, more than --object-height above the board, within
    --object-radius of the object's centre;
  - board NOISE: RMS height of the points on the board around the tags. The board is z = 0 by
    construction, so this is the spread of a surface that should be flat (plus any real warp);
  - how far each cloud is from the first one (nearest-neighbour distance, median and p90).

    src/venv/bin/python src/03_Reconstruction/compare_clouds.py \
        results/reconstruction/mjpg_pyr2_cloud_clean.ply results/reconstruction/mjpg_pyr2_colmap_cloud_clean.ply
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess_cloud import read_tags  # noqa: E402
from stereo_rig import RESULTS_DIR, rel  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compare board-frame point clouds of one scan.")
    ap.add_argument("clouds", nargs="+", help="PLY files; the first is the reference for distances")
    ap.add_argument("--poses", default=os.path.join(RESULTS_DIR, "mjpg_pyr2_tag_poses.yaml"))
    ap.add_argument("--object-height", type=float, default=5.0,
                    help="mm above the board that counts as object (default 5)")
    ap.add_argument("--object-radius", type=float, default=100.0,
                    help="mm around the object's centre that counts as object (default 100)")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    tags = read_tags(args.poses)
    corners = np.vstack(list(tags.values()))
    # Board points used for the noise figure: within 30 mm of a tag's centre, i.e. on or right
    # around the tags, where every method has texture and the surface is known to be the board.
    tag_centres = np.array([c.mean(0) for c in tags.values()])

    clouds = {path: o3d.io.read_point_cloud(path) for path in args.clouds}
    reference_path = args.clouds[0]
    print(f"{'cloud':48s} {'points':>9s} {'object':>8s} {'board RMS':>10s} {'to ref med':>11s} {'p90':>7s}")
    for path, cloud in clouds.items():
        xyz = np.asarray(cloud.points)
        xy_near_tags = np.min(np.linalg.norm(xyz[:, None, :2] - tag_centres[None, :, :2], axis=2), axis=1) < 30
        board = xy_near_tags & (np.abs(xyz[:, 2]) < 10)
        raised = xyz[:, 2] > args.object_height
        centre = np.median(xyz[raised, :2], axis=0) if raised.any() else np.zeros(2)
        on_object = raised & (np.linalg.norm(xyz[:, :2] - centre, axis=1) < args.object_radius)
        if path == reference_path:
            distance = "-", "-"
        else:
            d = np.asarray(cloud.compute_point_cloud_distance(clouds[reference_path]))
            distance = f"{np.median(d):.2f}", f"{np.percentile(d, 90):.2f}"
        print(f"{rel(path)[-48:]:48s} {len(xyz):9d} {on_object.sum():8d} "
              f"{np.sqrt(np.mean(xyz[board, 2] ** 2)):9.2f}  {distance[0]:>10s} {distance[1]:>7s}")
        print(f"{'':48s} object centre ({centre[0]:.1f}, {centre[1]:.1f}), "
              f"max height {np.percentile(xyz[on_object, 2], 99.5) if on_object.any() else 0:.1f} mm")
    print(f"\ndistances are to {rel(reference_path)}, mm; board RMS is |z| of points near the tags, mm")
    print(f"box: tags span x {corners[:, 0].min():.0f}..{corners[:, 0].max():.0f}, "
          f"y {corners[:, 1].min():.0f}..{corners[:, 1].max():.0f}")


if __name__ == "__main__":
    main()
