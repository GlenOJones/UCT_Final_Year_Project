"""Where in the image each markerless pose estimator gets its grip, for a few frames along the scan.

Left panel, COLMAP: every SIFT keypoint in the left image, split into those the final model triangulated
(used to pose the frame) and those it did not. Right panel, ICP: the frame's depth points (as icp_poses.py
sees them, 2 mm), coloured by how much each resists a slide along the direction of travel, |n . d| with n the
surface normal and d the pass's main direction of travel from the tag poses. A flat board parallel to the motion
scores 0 everywhere; object walls and brick edges facing along the track score near 1. Both panels show the
masked tags (hatched red) and the board outline from the tag layout (dashed), so the share of the grip that
comes from the board and from its surroundings can be read off.

    src/venv/bin/python src/experiments/pose_estimation/plot_detections.py --config src/experiments/pose_estimation/config.json

Writes results/<out_dir>/<recording>/evaluation/detections/<frame>.png and detections.json (per frame: keypoints,
triangulated, share on the board; ICP points and share with |n . d| > 0.5 on and off the board).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "03_Reconstruction"))
import common  # noqa: E402
import pose_files  # noqa: E402
from run_poses import recording_paths  # noqa: E402
from stereo_rig import invert, load_rig  # noqa: E402

BOARD_MARGIN_MM = 60.0   # the board outline: tag layout plus this, as dense_stereo.py's default crop


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--frames", type=int, default=4, help="frames to draw, spread evenly along the scan (default 4)")
    ap.add_argument("--depth", default="foundationstereo", help="depth source for the ICP panel (default foundationstereo)")
    return ap.parse_args(argv)


def colmap_keypoints(workspace):
    """{image name: (N, 3) array of x, y, triangulated (0/1)} from the model's text export."""
    with open(os.path.join(workspace, "best_model.txt"), encoding="utf-8") as fh:
        model = os.path.join(workspace, "sparse", fh.read().strip() + "_txt")
    with open(os.path.join(model, "images.txt"), encoding="utf-8") as fh:
        lines = [line for line in fh if not line.startswith("#")]
    out = {}
    for header, points in zip(lines[::2], lines[1::2]):
        values = np.array(points.split(), float).reshape(-1, 3)
        out[header.split()[9]] = np.c_[values[:, :2], values[:, 2] >= 0]
    return out


def board_outline(layout_path):
    fs = cv2.FileStorage(layout_path, cv2.FILE_STORAGE_READ)
    node = fs.getNode("tags").getNode("corners_mm")
    corners = np.vstack([node.at(i).mat() for i in range(node.size())])
    fs.release()
    lo, hi = corners[:, :2].min(0) - BOARD_MARGIN_MM, corners[:, :2].max(0) + BOARD_MARGIN_MM
    return np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]], float)


def project(points_cam, rig):
    px, _ = cv2.projectPoints(points_cam, np.zeros(3), np.zeros(3), rig.K["left"], rig.D["left"])
    return px.reshape(-1, 2)


def inside(px, polygon):
    path = polygon.astype(np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(path, (float(x), float(y)), False) >= 0 for x, y in px], bool)


def draw(path, image, mask, board_px, keypoints, icp_px, icp_score, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.4))
    for a in ax:
        a.imshow(image, cmap="gray", vmin=0, vmax=255)
        if mask is not None:
            a.contourf(mask == 0, levels=[0.5, 1], colors="none", hatches=["////"])
            a.contour(mask == 0, levels=[0.5], colors="tab:red", linewidths=1)
        if board_px is not None:
            loop = np.vstack([board_px, board_px[:1]])
            a.plot(loop[:, 0], loop[:, 1], "--", color="gold", lw=1.5)
        a.set_axis_off()
    used = keypoints[:, 2] > 0
    ax[0].scatter(keypoints[~used, 0], keypoints[~used, 1], s=2, c="tab:gray", alpha=0.6, label=f"detected, not used ({(~used).sum()})")
    ax[0].scatter(keypoints[used, 0], keypoints[used, 1], s=4, c="lime", label=f"triangulated, used ({used.sum()})")
    ax[0].legend(loc="lower right", fontsize=8, markerscale=3)
    ax[0].set_title("COLMAP: SIFT keypoints")
    sc = ax[1].scatter(icp_px[:, 0], icp_px[:, 1], s=0.6, c=icp_score, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax[1], fraction=0.035, label="|n · direction of travel|: resists the slide")
    ax[1].set_title("ICP: depth points (2 mm)")
    for a in ax:
        a.set_xlim(0, image.shape[1])
        a.set_ylim(image.shape[0], 0)
    fig.suptitle(title + "   (red hatch: masked tags; gold dashes: board)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run(cfg, recording, args):
    p = recording_paths(cfg, recording)
    rig = load_rig(common.project_path(recording["calibration"]))
    _, reference, _ = pose_files.read_trajectory(p["reference"])
    keypoints = colmap_keypoints(p["colmap"])
    outline = board_outline(p["reference"])
    depth_dir = p["depth"](args.depth)
    names = sorted(n for n in os.listdir(os.path.join(p["masks"], "left")) if n in reference)
    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, args.frames + 2)[1:-1]]
    # The pass's main direction of travel (board frame): the trolley's motion is close to one straight line,
    # while neighbouring frames' motion is too small and noisy where it stops and starts.
    centres = np.array([invert(reference[n])[:3, 3] for n in names])
    track = np.linalg.svd(centres - centres.mean(0))[2][0]
    out_dir = os.path.join(p["out"], "evaluation", "detections")
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = common.project_path(recording["frames"])
    stats = {}
    for name in picks:
        image = cv2.imread(os.path.join(frames_dir, "left", name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(p["masks"], "left", name), cv2.IMREAD_GRAYSCALE)
        T = reference[name]                                   # T_cam_board
        board_px = project(outline @ T[:3, :3].T + T[:3, 3], rig)
        kp = keypoints.get(f"left/{name}", np.zeros((0, 3)))
        direction = T[:3, :3] @ track   # direction of travel, in this camera's frame
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            np.load(os.path.join(depth_dir, "frames", name[:-4] + ".npz"))["xyz"].astype(float))).voxel_down_sample(2.0)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6.0, max_nn=30))
        xyz, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
        score = np.abs(normals @ direction)
        icp_px = project(xyz, rig)
        on_board_kp = inside(kp[:, :2], board_px) if len(kp) else np.zeros(0, bool)
        on_board_icp = inside(icp_px, board_px)
        used = kp[:, 2] > 0
        grip = score > 0.5
        stats[name] = {"keypoints": int(len(kp)), "triangulated": int(used.sum()),
                       "triangulated_on_board_%": round(100 * float(on_board_kp[used].mean()), 1) if used.any() else None,
                       "icp_points": int(len(xyz)), "icp_on_board_%": round(100 * float(on_board_icp.mean()), 1),
                       "icp_resisting_slide_%": round(100 * float(grip.mean()), 1),
                       "icp_resisting_slide_on_board_%": round(100 * float(on_board_icp[grip].mean()), 1) if grip.any() else None}
        draw(os.path.join(out_dir, name), image, mask, board_px, kp, icp_px, score,
             f"{recording['name']} frame {name}")
        print(f"  {name}: {stats[name]}")
    with open(os.path.join(out_dir, "detections.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    print(f"-> {common.rel(out_dir)}/")


def main(argv=None):
    args = parse_args(argv)
    cfg = common.load_config(args.config)
    for recording in cfg["recordings"]:
        run(cfg, recording, args)


if __name__ == "__main__":
    main()
