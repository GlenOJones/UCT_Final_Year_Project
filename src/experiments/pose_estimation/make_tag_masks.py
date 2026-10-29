"""Masks that hide the reference tags from the markerless pose estimators.

    src/venv/bin/python src/experiments/pose_estimation/make_tag_masks.py --frames-dir data/3DRecon/3DPrintFinal/30_frames_range \
        --poses results/experiments/pose_estimation/3DPrintFinal_s30/tag_poses.yaml --out-dir .../tag_masks --step 5
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "03_Reconstruction"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), ".."))
from stereo_rig import CAMERAS, find_frame_pairs, load_rig, rel  # noqa: E402
import tag_poses  # noqa: E402
from tag_poses import detect, make_detector  # noqa: E402
import pose_files  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Masks hiding the reference tags.")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--poses", required=True, help="tag_poses.py YAML for this recording (layout + per-frame poses)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step", type=int, default=1, help="every Nth frame pair, as the estimators use (default 1)")
    ap.add_argument("--enhance", action="store_true",
                    help="detect as tag_poses.py --enhance does (contrast-boosted image), for backlit recordings")
    ap.add_argument("--scale", type=float, default=1.6,
                    help="mask side / tag side: covers the white paper square around each tag (default 1.6)")
    return ap.parse_args(argv)


def read_layout(path):
    """{tag id: 4x3 corners in the board frame} from tag_poses.py's YAML."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    ids = fs.getNode("tags").getNode("ids")
    corners = fs.getNode("tags").getNode("corners_mm")
    layout = {int(ids.at(i).real()): corners.at(i).mat() for i in range(ids.size())}
    calibration = fs.getNode("metadata").getNode("calibration").string()
    fs.release()
    return layout, calibration


def grow(quad, scale):
    centre = quad.mean(0)
    return centre + (quad - centre) * scale


def main(argv=None):
    args = parse_args(argv)
    layout, calibration = read_layout(args.poses)
    rig = load_rig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), calibration))
    _, poses, _ = pose_files.read_trajectory(args.poses)
    T_right_left = np.eye(4)
    T_right_left[:3, :3], T_right_left[:3, 3] = rig.R, rig.T.ravel()
    tag_poses.ENHANCE = args.enhance
    detector = make_detector(args.enhance)
    width, height = rig.image_size
    pairs = find_frame_pairs(args.frames_dir)[::args.step]
    for camera in CAMERAS:
        os.makedirs(os.path.join(args.out_dir, camera), exist_ok=True)

    stats = {"frames": len(pairs), "projected": 0, "detected_only": 0, "unmasked": 0, "masked_fraction": []}
    for name, left_path, right_path in pairs:
        T_cam_board = poses.get(name)
        if T_cam_board is not None:
            stats["projected"] += 1
        any_tag = False
        for camera, path in (("left", left_path), ("right", right_path)):
            mask = np.full((height, width), 255, np.uint8)
            quads = [grow(q, args.scale) for q in detect(detector, path).values()]
            if T_cam_board is not None:
                T = T_cam_board if camera == "left" else T_right_left @ T_cam_board
                for corners in layout.values():
                    board_quad = grow(corners, args.scale)
                    in_cam = board_quad @ T[:3, :3].T + T[:3, 3]
                    if np.all(in_cam[:, 2] > 0):
                        px, _ = cv2.projectPoints(board_quad, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], rig.K[camera], rig.D[camera])
                        quads.append(px.reshape(4, 2))
            for quad in quads:
                cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 0)
            any_tag |= bool(quads)
            stats["masked_fraction"].append(float((mask == 0).mean()))
            cv2.imwrite(os.path.join(args.out_dir, camera, name), mask)
        if T_cam_board is None:
            stats["detected_only" if any_tag else "unmasked"] += 1
    fractions = stats.pop("masked_fraction")
    stats["masked_fraction_median"] = float(np.median(fractions))
    with open(os.path.join(args.out_dir, "masks.json"), "w", encoding="utf-8") as fh:
        json.dump({**stats, "scale": args.scale, "poses": rel(args.poses)}, fh, indent=1)
    print(f"tag masks for {len(pairs)} frame pairs -> {rel(args.out_dir)}/: {stats['projected']} from the tag layout "
          f"(+ detections), {stats['detected_only']} from detections only, {stats['unmasked']} with no tag found "
          f"(left unmasked); median {100 * stats['masked_fraction_median']:.1f}% of each image masked")


if __name__ == "__main__":
    main()
