"""Merge reconstructions of the same board made from several passes (e.g. Sep30 s20 ... s60, the rig
20 ... 60 cm above the board) into one cloud.

Each pass has its own board frame: tag_poses.py builds it from that pass's estimate of the tag layout,
so two passes over an unmoved board differ by a small rigid motion (Sep30: about 1-2 deg of yaw). The
merge:

  1. ALIGNS every scan to the reference scan's frame: a rigid fit of its tag layout onto the
     reference layout (the corners of the tags both saw), then point-to-plane ICP of its cloud onto
     the reference cloud, which takes out the remaining mm-level frame error;
  2. FUSES them as a height field. Every pass looks down on the board, so each scan holds at most one
     surface per (x, y): per 1 mm cell, each scan's height is the median of its points there. Each
     scan's noise sigma is measured on the bare board (spread about local planes), and per cell:
       - weighted: the inverse-variance weighted mean of the scans' heights, after dropping heights
         more than --reject mm from the cell's weighted median. Near passes dominate where they have
         data, far passes fill the gaps they leave;
       - nearest: the height of the lowest-noise scan that has the cell (no averaging, hole filling);
       - concat: every aligned point of every scan, as a baseline.

    src/venv/bin/python src/03_Reconstruction/merge_scans.py --session Sep30 --scans s20 s30 s40 s50 s60

Reads results/<session>/<scan>/<method>/cloud_clean.ply and poses/tag_poses.yaml of every scan;
writes results/<session>/<out-scan>/<method>_{weighted,nearest,concat}/cloud_clean.ply, aligned/
(each scan in the reference frame), alignment.json, and poses/tag_poses.yaml (the reference scan's,
so later stages find the tags). Score them with compare_to_cad.py / score_objects.py like any scan.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import rel, rigid_fit, transform  # noqa: E402
from view_cloud import read_tags  # noqa: E402
import project_paths as paths  # noqa: E402

CELL = 1.0              # mm, height-field grid
ICP_VOXEL = 2.0         # mm, downsampling for ICP
ICP_DISTANCE = 4.0      # mm, max correspondence distance
BOARD_PATCH = 20.0      # mm, patches the board noise is measured over
BOARD_BAND = 4.0        # mm, |z| below which a point counts as bare board


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Align and fuse several passes over the same board.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--scans", nargs="+", required=True, help="scans to merge, e.g. s20 s30 s40")
    ap.add_argument("--reference", help="scan whose board frame the merge uses (default: the first)")
    ap.add_argument("--method", default="raft_s075", help="method folder in every scan (default raft_s075)")
    ap.add_argument("--out-scan", default="merged", help="scan folder for the result (default merged)")
    ap.add_argument("--reject", type=float, default=3.0,
                    help="mm; a scan's height this far from the cell's weighted median is dropped (default 3)")
    ap.add_argument("--no-icp", action="store_true", help="align by the tag layouts only")
    args = ap.parse_args(argv)
    args.reference = args.reference or args.scans[0]
    if args.reference not in args.scans:
        ap.error("--reference must be one of --scans")
    return args


# ====== ALIGNMENT ======
def layout_transform(session, scan, reference):
    """T taking scan's board frame into reference's, from the tags both layouts hold, and its RMS (mm)."""
    tags = read_tags(paths.poses_path(session, scan))
    ref_tags = read_tags(paths.poses_path(session, reference))
    shared = sorted(set(tags) & set(ref_tags))
    if len(shared) < 2:
        raise SystemExit(f"{scan} and {reference} share {len(shared)} tags; need at least 2 to align")
    return rigid_fit(np.vstack([tags[t] for t in shared]), np.vstack([ref_tags[t] for t in shared]))


def refine_icp(points, ref_points, T_init):
    """Point-to-plane ICP of points onto ref_points starting from T_init: (T, fitness, rmse)."""
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(ICP_VOXEL)
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_points)).voxel_down_sample(ICP_VOXEL)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * ICP_VOXEL, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, ICP_DISTANCE, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    return np.asarray(result.transformation), result.fitness, result.inlier_rmse


def motion(T):
    """Size of a rigid motion: (translation mm, rotation deg)."""
    angle = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(T[:3, 3])), float(angle)


# ====== FUSION ======
def board_noise(xyz):
    """Robust spread (mm) of bare-board points about planes fitted per BOARD_PATCH patch."""
    board = xyz[np.abs(xyz[:, 2]) < BOARD_BAND]
    keys = np.floor(board[:, :2] / BOARD_PATCH).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    residuals = []
    for patch in np.split(np.argsort(inverse), np.cumsum(np.bincount(inverse))[:-1]):
        if len(patch) < 50:
            continue
        p = board[patch]
        A = np.c_[p[:, :2], np.ones(len(p))]
        coef, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
        residuals.append(p[:, 2] - A @ coef)
    r = np.concatenate(residuals)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def height_field(xyz, origin, shape):
    """Per-cell median height of xyz on the grid (NaN where the scan has no point)."""
    ij = np.floor((xyz[:, :2] - origin) / CELL).astype(np.int64)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.lexsort((xyz[:, 2], flat))
    flat, z = flat[order], xyz[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    counts = np.diff(np.r_[starts, len(flat)])
    medians = z[starts + counts // 2]
    field = np.full(shape[0] * shape[1], np.nan)
    field[flat[starts]] = medians
    return field.reshape(shape)


def fuse(fields, sigmas, reject):
    """weighted and nearest height fields from the per-scan fields (stack, NaN = no data)."""
    weights = np.where(np.isnan(fields), 0.0, 1.0 / np.square(np.asarray(sigmas))[:, None, None])
    heights = np.nan_to_num(fields)
    # Weighted median per cell: the outlier test needs a centre one bad scan cannot drag.
    order = np.argsort(np.where(np.isnan(fields), np.inf, fields), axis=0)
    w_sorted = np.take_along_axis(weights, order, axis=0)
    z_sorted = np.take_along_axis(heights, order, axis=0)
    cumulative = np.cumsum(w_sorted, axis=0)
    total = cumulative[-1]
    median_index = np.argmax(cumulative >= total / 2, axis=0)
    median = np.take_along_axis(z_sorted, median_index[None], axis=0)[0]
    keep = (weights > 0) & (np.abs(heights - median) <= reject)
    w = np.where(keep, weights, 0.0)
    w_sum = w.sum(0)
    weighted = np.where(w_sum > 0, (w * heights).sum(0) / np.where(w_sum > 0, w_sum, 1), np.nan)
    # Nearest: scans ordered by noise; the first that has the cell wins.
    nearest = np.full(fields.shape[1:], np.nan)
    for k in np.argsort(sigmas)[::-1]:
        nearest = np.where(np.isnan(fields[k]), nearest, fields[k])
    return weighted, nearest, keep.sum(0)


def field_points(field, origin):
    i, j = np.nonzero(~np.isnan(field))
    return np.c_[origin[0] + (i + 0.5) * CELL, origin[1] + (j + 0.5) * CELL, field[i, j]]


def save_cloud(points, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, paths.CLOUD_CLEAN)
    o3d.io.write_point_cloud(path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    print(f"  {len(points):8d} points -> {rel(path)}")


def main():
    args = parse_args()
    out_dir = paths.scan_dir(args.session, args.out_scan)
    clouds = {}
    for scan in args.scans:
        path = os.path.join(paths.method_dir(args.session, scan, args.method), paths.CLOUD_CLEAN)
        clouds[scan] = np.asarray(o3d.io.read_point_cloud(path).points, np.float64)
        print(f"{scan}: {len(clouds[scan])} points from {rel(path)}")

    print(f"\nALIGNMENT onto {args.reference}'s board frame")
    aligned, report = {}, {}
    for scan in args.scans:
        T_tags, tag_rms = layout_transform(args.session, scan, args.reference)
        T, fitness, rmse = T_tags, None, None
        if scan != args.reference and not args.no_icp:
            T, fitness, rmse = refine_icp(clouds[scan], clouds[args.reference], T_tags)
        aligned[scan] = transform(T, clouds[scan])
        tags_t, tags_r = motion(T_tags)
        icp_t, icp_r = motion(T @ np.linalg.inv(T_tags))
        report[scan] = {"T_reference_scan": T.tolist(), "tag_layout_rms_mm": float(tag_rms),
                        "tag_motion_mm_deg": [tags_t, tags_r], "icp_correction_mm_deg": [icp_t, icp_r],
                        "icp_fitness": fitness, "icp_rmse_mm": rmse}
        icp_text = "" if fitness is None else \
            f", ICP then moved it {icp_t:.2f} mm / {icp_r:.2f} deg (fitness {fitness:.2f}, rmse {rmse:.2f} mm)"
        print(f"  {scan}: tag layouts agree to {tag_rms:.2f} mm RMS, {tags_t:.1f} mm / {tags_r:.2f} deg{icp_text}")

    print("\nBOARD NOISE (spread about local planes on bare board)")
    sigmas = []
    for scan in args.scans:
        sigma = board_noise(aligned[scan])
        report[scan]["board_noise_mm"] = sigma
        sigmas.append(sigma)
        print(f"  {scan}: {sigma:.2f} mm")

    everything = np.vstack(list(aligned.values()))
    origin = np.floor(everything[:, :2].min(0))
    shape = tuple((np.ceil((everything[:, :2].max(0) - origin) / CELL) + 1).astype(int))
    fields = np.stack([height_field(aligned[s], origin, shape) for s in args.scans])
    weighted, nearest, used = fuse(fields, sigmas, args.reject)

    print("\nCOVERAGE (cells of the 1 mm grid with a height)")
    for scan, field in zip(args.scans, fields):
        print(f"  {scan}: {np.count_nonzero(~np.isnan(field))}")
    print(f"  merged: {np.count_nonzero(~np.isnan(weighted))}, "
          f"from {np.mean(used[used > 0]):.2f} scans per cell on average")

    print("\nSAVING")
    for scan in args.scans:
        save_cloud(aligned[scan], os.path.join(out_dir, "aligned", scan))
    save_cloud(field_points(weighted, origin), paths.method_dir(args.session, args.out_scan, f"{args.method}_weighted"))
    save_cloud(field_points(nearest, origin), paths.method_dir(args.session, args.out_scan, f"{args.method}_nearest"))
    save_cloud(everything, paths.method_dir(args.session, args.out_scan, f"{args.method}_concat"))

    os.makedirs(os.path.dirname(paths.poses_path(args.session, args.out_scan)), exist_ok=True)
    shutil.copy(paths.poses_path(args.session, args.reference), paths.poses_path(args.session, args.out_scan))
    with open(os.path.join(out_dir, "alignment.json"), "w") as f:
        json.dump({"session": args.session, "scans": args.scans, "reference": args.reference,
                   "method": args.method, "cell_mm": CELL, "reject_mm": args.reject, "per_scan": report}, f, indent=2)
    print(f"  alignment and noise -> {rel(os.path.join(out_dir, 'alignment.json'))}")


if __name__ == "__main__":
    main()
