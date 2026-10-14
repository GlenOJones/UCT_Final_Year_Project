"""Per-frame depth of a scan in each frame's own LEFT CAMERA frame, without any poses.

dense_stereo.py needs a pose for every frame before it matches: the tag poses crop each frame to a box
around the board and set SGBM's search range. A scan with no tags (ice) has no poses until they are
estimated from the images themselves, and icp_poses.py estimates them from exactly this depth. So
here every --step'th frame pair is matched on its own, with the same rectification and matchers as
dense_stereo.py, and the only limits are camera-frame ones:

  - depth (camera z) between --depth-min and --depth-max, which also sets SGBM's disparity range;
  - optionally --masks: pixels to ignore, as images <masks>/left/<frame name> (0 = ignore), in the
    original (unrectified) left image. The pose experiment masks the reference tags this way, so the
    markerless estimators cannot lean on texture that ice will not have.

    src/venv/bin/python src/03_Reconstruction/depth_frames.py --frames-dir data/3DRecon/3DPrintFinal/30_frames_range \
        --calibration results/Calibration/calibration/FullCalibration_water_final_merge_stereo.yaml \
        --out-dir work/3DPrintFinal/s30_range/depth/sgbm --step 5 --depth-min 250 --depth-max 600
    envs/ffs/bin/python src/03_Reconstruction/depth_frames.py ... --matcher foundationstereo --net-scale 0.5 --lr-tolerance 3

Writes <out-dir>/frames/<frame>.npz (xyz: float32 points in the left camera frame, mm, one per --voxel
cube, the mean of the pixels in it; grey: uint8) and <out-dir>/depth_frames.json (settings, timing).
"""
import argparse
import datetime
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_stereo import Rectifier, add_matcher_arguments, make_network, match  # noqa: E402
from stereo_rig import find_frame_pairs, load_rig, rel, transform  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-frame camera-frame depth, without poses.")
    ap.add_argument("--frames-dir", required=True, help="recording's frame pairs: left/ and right/")
    ap.add_argument("--calibration", required=True, help="stereo calibration YAML from Calibration.py")
    ap.add_argument("--out-dir", required=True)
    add_matcher_arguments(ap)
    ap.add_argument("--step", type=int, default=1, help="every Nth frame pair (default 1)")
    ap.add_argument("--step-offset", type=int, default=0, help="start --step at this pair (default 0)")
    ap.add_argument("--depth-min", type=float, required=True, help="mm, nearest camera z kept")
    ap.add_argument("--depth-max", type=float, required=True, help="mm, furthest camera z kept")
    ap.add_argument("--voxel", type=float, default=1.0, help="mm, points kept per frame: one per cube (default 1)")
    ap.add_argument("--masks", default=None, help="folder with left/<frame name> masks, 0 = ignore (default none)")
    ap.add_argument("--skip-existing", action="store_true", help="keep frames already in <out-dir>/frames")
    return ap.parse_args(argv)


def disparity_range(rectifier, depth_min, depth_max):
    """(min disparity, number of disparities) for SGBM covering camera depths depth_min..depth_max."""
    d_min = rectifier.focal * rectifier.baseline / depth_max
    d_max = rectifier.focal * rectifier.baseline / depth_min
    minimum = max(int(np.floor(d_min)) - 8, 0)
    return minimum, int(np.ceil((d_max - minimum + 8) / 16) * 16)


def voxel_means(xyz, grey, voxel):
    """One point per voxel: the mean position and grey of the points in it."""
    _, inverse = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse)
    mean_xyz = np.stack([np.bincount(inverse, xyz[:, i]) for i in range(3)], 1) / counts[:, None]
    return mean_xyz, np.bincount(inverse, grey) / counts


def frame_points(rectifier, network, left_path, right_path, mask, args, search):
    """Left-camera-frame points and grey of one pair, within the depth limits and outside the mask."""
    left, right = rectifier.rectify("left", left_path), rectifier.rectify("right", right_path)
    minimum, count = search
    if network is None:
        disparity = match(left, right, minimum, count, args.block_size, args.uniqueness)
    else:
        disparity = network.disparity(left, right)
        disparity[disparity < minimum] = minimum - 1
    valid = disparity >= minimum
    if mask is not None:
        # The mask is drawn on the original image; carry it into the rectified one like the image itself.
        valid &= cv2.remap(mask, *rectifier.maps["left"], cv2.INTER_NEAREST) > 0
    xyz_rect = cv2.reprojectImageTo3D(disparity, rectifier.rig.Q)[valid].astype(np.float64)
    keep = (xyz_rect[:, 2] >= args.depth_min) & (xyz_rect[:, 2] <= args.depth_max)
    xyz = transform(rectifier.T_cam_rect, xyz_rect[keep])
    return xyz, left[valid][keep]


def main(argv=None):
    args = parse_args(argv)
    rig = load_rig(args.calibration)
    rectifier = Rectifier(rig)
    pairs = find_frame_pairs(args.frames_dir)[args.step_offset::args.step]
    search = disparity_range(rectifier, args.depth_min, args.depth_max)
    out_frames = os.path.join(args.out_dir, "frames")
    os.makedirs(out_frames, exist_ok=True)
    print(f"{len(pairs)} frame pairs from {rel(args.frames_dir)} (every {args.step}th); depth "
          f"{args.depth_min:.0f}..{args.depth_max:.0f} mm = disparity {search[0]}+{search[1]} px"
          + (f"; masks from {rel(args.masks)}" if args.masks else ""))
    network = make_network(args)

    start, made, points = time.time(), 0, []
    for k, (name, left_path, right_path) in enumerate(pairs):
        out_path = os.path.join(out_frames, os.path.splitext(name)[0] + ".npz")
        if args.skip_existing and os.path.exists(out_path):
            continue
        mask = None
        if args.masks:
            mask_path = os.path.join(args.masks, "left", name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
        xyz, grey = frame_points(rectifier, network, left_path, right_path, mask, args, search)
        xyz, grey = voxel_means(xyz, grey, args.voxel) if len(xyz) else (xyz, grey)
        np.savez(out_path, xyz=xyz.astype(np.float32), grey=np.clip(np.round(grey), 0, 255).astype(np.uint8))
        made += 1
        points.append(len(xyz))
        if made % 10 == 0:
            print(f"  {k + 1}/{len(pairs)}  {name}: {len(xyz)} points, {time.time() - start:.0f} s", flush=True)
    seconds = time.time() - start

    settings = {k: (rel(os.path.abspath(v)) if k in ("frames_dir", "calibration", "out_dir", "masks") and v else v)
                for k, v in vars(args).items()}
    record = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "script": "src/03_Reconstruction/depth_frames.py",
              "argv": sys.argv[1:] if argv is None else argv, "settings": settings, "frames": len(pairs),
              "frames_made": made, "seconds": round(seconds, 1),
              "seconds_per_frame": round(seconds / made, 3) if made else None,
              "points_per_frame_median": float(np.median(points)) if points else None,
              "disparity_search": list(search)}
    with open(os.path.join(args.out_dir, "depth_frames.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    print(f"{made} frames in {seconds:.0f} s -> {rel(out_frames)}/")


if __name__ == "__main__":
    main()
