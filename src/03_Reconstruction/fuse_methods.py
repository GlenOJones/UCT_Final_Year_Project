"""Fuse two reconstructions of the same scan: a sparse, reliable one (COLMAP) checks a dense, smooth
one (RAFT-Stereo) and vetoes it where they disagree.

On dark, low-texture prints the two fail differently. RAFT always returns a disparity, so where the
image has no detail it invents a smooth surface (Sep30: the 2x2 pyramid and the egg cartons came out
as flat slabs at ~30 mm). COLMAP's PatchMatch leaves those pixels empty, but what it keeps is right
(Sep30/s30 EGG_1PK: 1.9 mm RMS against RAFT's 7.0 mm, on 19% of the surface). The fusion:

  1. ALIGNS the dense cloud onto the sparse one (point-to-plane ICP, from the shared board frame);
  2. CHECKS every dense point against the sparse points within --radius mm of it in x, y: if at least
     --min-support of them are there, it is "checked", and it agrees if it lies within --agree mm of
     their median height;
  3. VETOES the dense points that disagree, and the unchecked dense points where most checked points
     within --veto mm disagree: a slab is wrong across the whole hole COLMAP leaves in it, not just at
     the hole's rim, while scattered board noise is not a reason to drop anything;
  4. KEEPS all sparse points, the dense points that agree, and the unchecked dense points far from
     any disagreement (where COLMAP has nothing to say and RAFT is not contradicted).

Everything runs in x, y with heights, as every view of these scans looks down on the board.

    src/venv/bin/python src/03_Reconstruction/fuse_methods.py --session Sep30 --scan s30 --sparse colmap --dense raft_s075

Reads <scan>/<sparse>/cloud_clean.ply and <scan>/<dense>/cloud_clean.ply; writes
<scan>/<out>/cloud_clean.ply (default <sparse>+<dense>), with the dense points coloured by verdict in
fusion_verdicts.ply beside it. Score it with score_objects.py like any method.
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_scans import motion, refine_icp  # noqa: E402
from stereo_rig import rel, transform  # noqa: E402
import project_paths as paths  # noqa: E402

VERDICT_COLOURS = {"agree": (0.20, 0.60, 0.25), "disagree": (0.80, 0.15, 0.15),
                   "vetoed": (0.95, 0.60, 0.10), "unchecked": (0.55, 0.55, 0.55)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Let a sparse reconstruction veto a dense one where they disagree.")
    paths.add_scan_arguments(ap)
    ap.add_argument("--sparse", default="colmap", help="the reliable, sparse method (default colmap)")
    ap.add_argument("--dense", default="raft_s075", help="the complete, smooth method (default raft_s075)")
    ap.add_argument("--out", help="output method folder (default <sparse>+<dense>)")
    ap.add_argument("--radius", type=float, default=3.0, help="mm, x-y radius a dense point is checked over (default 3)")
    ap.add_argument("--min-support", type=int, default=5,
                    help="sparse points needed within --radius for a check (default 5)")
    ap.add_argument("--agree", type=float, default=3.0,
                    help="mm, height difference within which the two agree (default 3)")
    ap.add_argument("--veto", type=float, default=10.0,
                    help="mm, neighbourhood over which disagreement is judged for unchecked points (default 10)")
    ap.add_argument("--veto-share", type=float, default=0.5,
                    help="unchecked dense points are dropped where more than this share of the checked "
                         "points within --veto disagree (default 0.5)")
    ap.add_argument("--no-icp", action="store_true", help="trust the shared board frame as it is")
    args = ap.parse_args(argv)
    paths.require_scan(args, ap)
    args.out = args.out or f"{args.sparse}+{args.dense}"
    return args


def load(session, scan, method):
    path = os.path.join(paths.method_dir(session, scan, method), paths.CLOUD_CLEAN)
    xyz = np.asarray(o3d.io.read_point_cloud(path).points, np.float64)
    print(f"{method}: {len(xyz)} points from {rel(path)}")
    return xyz


def verdicts(sparse, dense, args):
    """Per dense point: 'agree', 'disagree', 'vetoed' or 'unchecked'."""
    tree = cKDTree(sparse[:, :2])
    neighbours = tree.query_ball_point(dense[:, :2], args.radius, workers=-1)
    support = np.fromiter((len(n) for n in neighbours), int, len(dense))
    checked = support >= args.min_support
    reference = np.full(len(dense), np.nan)
    reference[checked] = [np.median(sparse[n, 2]) for n, c in zip(neighbours, checked) if c]
    agree = checked & (np.abs(dense[:, 2] - reference) <= args.agree)
    disagree = checked & ~agree
    result = np.where(agree, "agree", np.where(disagree, "disagree", "unchecked")).astype(object)
    # Veto where MOST checked points within --veto disagree. A single disagreeing point is no
    # evidence: on bare Sep30/s30 board the two differ with a 1.8 mm MAD, so 30% of board points fall
    # outside 3 mm by noise alone, and vetoing around each of those blanked the whole board.
    share = disagreement_share(dense[:, :2], checked, disagree, args.veto)
    result[~checked & (share > args.veto_share)] = "vetoed"
    return result


def disagreement_share(xy, checked, disagree, radius, cell=2.0):
    """Per point: the share of checked points within ~radius (on a cell-mm grid) that disagree."""
    from scipy.ndimage import uniform_filter
    origin = xy.min(0)
    ij = np.floor((xy - origin) / cell).astype(int)
    shape = tuple(ij.max(0) + 1)
    n_checked, n_disagree = np.zeros(shape), np.zeros(shape)
    np.add.at(n_checked, tuple(ij[checked].T), 1)
    np.add.at(n_disagree, tuple(ij[disagree].T), 1)
    size = 2 * int(round(radius / cell)) + 1
    n_checked, n_disagree = uniform_filter(n_checked, size), uniform_filter(n_disagree, size)
    share = np.where(n_checked > 0, n_disagree / np.maximum(n_checked, 1e-9), 0.0)
    return share[tuple(ij.T)]


def main():
    args = parse_args()
    sparse = load(args.session, args.scan, args.sparse)
    dense = load(args.session, args.scan, args.dense)

    T, icp = np.eye(4), None
    if not args.no_icp:
        T, fitness, rmse = refine_icp(dense, sparse, np.eye(4))
        dense = transform(T, dense)
        moved_mm, turned_deg = motion(T)
        icp = {"moved_mm": moved_mm, "turned_deg": turned_deg, "fitness": fitness, "rmse_mm": rmse}
        print(f"ICP of {args.dense} onto {args.sparse}: moved {moved_mm:.2f} mm / {turned_deg:.2f} deg "
              f"(fitness {fitness:.2f}, rmse {rmse:.2f} mm)")

    verdict = verdicts(sparse, dense, args)
    counts = {v: int(np.sum(verdict == v)) for v in VERDICT_COLOURS}
    print(f"\n{args.dense} points checked against {args.sparse} "
          f"(radius {args.radius} mm, >= {args.min_support} points, agree within {args.agree} mm, "
          f"veto where > {args.veto_share:.0%} disagree within {args.veto} mm):")
    for v, n in counts.items():
        print(f"  {v:9s} {n:9d}  {100 * n / len(dense):5.1f}%")

    keep = (verdict == "agree") | (verdict == "unchecked")
    fused = np.vstack([sparse, dense[keep]])
    out_dir = paths.method_dir(args.session, args.scan, args.out)
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, paths.CLOUD_CLEAN),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fused)))
    shown = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense))
    shown.colors = o3d.utility.Vector3dVector(np.array([VERDICT_COLOURS[v] for v in verdict]))
    o3d.io.write_point_cloud(os.path.join(out_dir, "fusion_verdicts.ply"), shown)
    with open(os.path.join(out_dir, "fusion.json"), "w") as f:
        json.dump({"sparse": args.sparse, "dense": args.dense, "radius_mm": args.radius,
                   "min_support": args.min_support, "agree_mm": args.agree, "veto_mm": args.veto,
                   "veto_share": args.veto_share,
                   "icp": icp, "T_sparse_dense": T.tolist(), "verdicts": counts,
                   "points": {"sparse": len(sparse), "dense_kept": int(keep.sum()), "fused": len(fused)}}, f, indent=2)
    print(f"\n{len(sparse)} {args.sparse} + {int(keep.sum())} {args.dense} = {len(fused)} points "
          f"-> {rel(os.path.join(out_dir, paths.CLOUD_CLEAN))}")
    print(f"dense points coloured by verdict (green agree, red disagree, orange vetoed, grey unchecked) "
          f"-> {rel(os.path.join(out_dir, 'fusion_verdicts.ply'))}")


if __name__ == "__main__":
    main()
