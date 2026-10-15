"""Where each pose method gets its grip: per-frame counts for the comparison table, and images for the report.

Grip, per frame and per method (what the method actually holds on to):

  AprilGrid   tag corners triangulated in that frame (4 per tag seen by both cameras, after the reprojection gate)
  COLMAP      SIFT keypoints in the left image that the final model triangulated
  ICP         depth points (at --icp_voxel_mm, ICP's finest scale) whose surface normal n resists a slide along
              the direction of travel d, |n . d| > slide_threshold. A flat board parallel to the motion scores 0
              everywhere; object walls and edges facing along the track score near 1. The total point count is
              kept too.

d is the pass's main direction of travel from the tag poses, or, on a recording without tags, from COLMAP's
trajectory (a direction needs no scale), turned into each frame's camera frame by that frame's rotation.

Images, for `frames` frames spread evenly along each pass, every method drawn on the same left image:
  AprilGrid   the tags it detected, corners marked (it sees the unmasked image)
  COLMAP      SIFT keypoints, triangulated (used) vs detected only
  ICP         depth points coloured by |n . d|
The markerless panels hatch the masked tags in red; the dashed gold line is the board outline from the tag layout.

    src/venv/bin/python src/experiments/pose_estimation/detection_figures.py --config src/experiments/pose_estimation/config_comparison.json

Writes results/<out_dir>/<recording>/detections/:
  grip.json                         per method: grip per frame, its median, and the frames drawn
  <frame>_<method>.png              one panel per method and frame, to arrange as wanted
  <frame>_all.png                   the methods side by side for that frame
  grid.png                          frames (rows) x methods (columns)
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
from plot_detections import board_outline, colmap_keypoints, project  # noqa: E402
from run_poses import frame_step, has_tags, recording_paths  # noqa: E402
from stereo_rig import find_frame_pairs, invert, load_rig  # noqa: E402
from tag_poses import detect, make_detector  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config_comparison.json"))
    ap.add_argument("--recordings", nargs="+", help="only these recordings (default all in the config)")
    ap.add_argument("--images-only", action="store_true", help="redraw the images from an existing grip.json's frames, "
                    "skipping the per-frame grip over every frame")
    return ap.parse_args(argv)


def tags_per_frame(path):
    """{frame name: [tag ids used in its pose]} from tag_poses.py's YAML."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    frames = fs.getNode("frames")
    out = {}
    for i in range(frames.size()):
        node = frames.at(i)
        ids = node.getNode("tags")
        out[node.getNode("name").string()] = [int(ids.at(k).real()) for k in range(ids.size())]
    fs.release()
    return out


def travel_directions(names, rotations_source):
    """{name: unit direction of travel in that frame's left camera}. rotations_source: {name: T_cam_world} of the
    frames some method posed; frames it did not pose borrow the nearest posed frame's rotation."""
    posed = [n for n in names if n in rotations_source]
    centres = np.array([invert(rotations_source[n])[:3, 3] for n in posed])
    track = np.linalg.svd(centres - centres.mean(0))[2][0]
    index = {n: i for i, n in enumerate(names)}
    out = {}
    for n in names:
        nearest = n if n in rotations_source else min(posed, key=lambda m: abs(index[m] - index[n]))
        out[n] = rotations_source[nearest][:3, :3] @ track
    return out


def icp_points(depth_dir, name, det):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.load(os.path.join(depth_dir, "frames", name[:-4] + ".npz"))["xyz"].astype(float))).voxel_down_sample(det["icp_voxel_mm"])
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=det["icp_normal_radius_mm"], max_nn=30))
    return np.asarray(cloud.points), np.asarray(cloud.normals)


def median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


# ====== DRAWING ======
def draw_panel(ax, image, panel, mask=None, board_px=None):
    """One method's panel on ax. panel: {"kind", "title", ...data}. Returns the ICP scatter (for a colour bar)."""
    ax.imshow(image)
    if mask is not None and panel["kind"] != "aprilgrid":
        ax.contourf(mask == 0, levels=[0.5, 1], colors="none", hatches=["////"])
        ax.contour(mask == 0, levels=[0.5], colors="tab:red", linewidths=1)
    if board_px is not None:
        loop = np.vstack([board_px, board_px[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "--", color="gold", lw=1.2)
    scatter = None
    if panel["kind"] == "aprilgrid":
        for tag_id, (quad, used) in panel["tags"].items():
            loop = np.vstack([quad, quad[:1]])
            colour = "lime" if used else "tab:orange"
            ax.plot(loop[:, 0], loop[:, 1], "-", color=colour, lw=1.2)
            ax.scatter(quad[:, 0], quad[:, 1], s=14, c=colour, edgecolors="k", linewidths=0.4, zorder=3)
            ax.text(*quad.mean(0), str(tag_id), color=colour, fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        if not panel["tags"]:
            ax.text(0.5, 0.5, "no tags detected", transform=ax.transAxes, ha="center", va="center", color="white",
                    fontsize=12, bbox=dict(facecolor="black", alpha=0.6, lw=0))
        ax.plot([], [], "-", color="lime", label="tag used in pose (both cameras)")
        ax.plot([], [], "-", color="tab:orange", label="detected in left only / rejected")
        ax.legend(loc="lower right", fontsize=7)
    elif panel["kind"] == "colmap":
        kp = panel["keypoints"]
        used = kp[:, 2] > 0
        ax.scatter(kp[~used, 0], kp[~used, 1], s=1.5, c="tab:gray", alpha=0.6, label=f"detected, not used ({(~used).sum()})")
        ax.scatter(kp[used, 0], kp[used, 1], s=3, c="lime", label=f"triangulated, used ({used.sum()})")
        ax.legend(loc="lower right", fontsize=7, markerscale=3)
    else:
        scatter = ax.scatter(panel["px"][:, 0], panel["px"][:, 1], s=0.4, c=panel["score"], cmap="magma", vmin=0, vmax=1)
    ax.set_title(panel["title"], fontsize=10)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_axis_off()
    return scatter


def save_single(path, image, panel, mask, board_px, dpi):
    import matplotlib.pyplot as plt
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7 * h / w + 0.4))
    scatter = draw_panel(ax, image, panel, mask, board_px)
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, fraction=0.035, label="|n · d|: resists a slide along the track")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_grid(path, rows, dpi, title):
    """rows: [(image, mask, board_px, [panels], row label)]."""
    import matplotlib.pyplot as plt
    h, w = rows[0][0].shape[:2]
    cols = len(rows[0][3])
    fig, axes = plt.subplots(len(rows), cols, figsize=(4.2 * cols, 4.2 * h / w * len(rows) + 0.8), squeeze=False)
    scatter = None
    for r, (image, mask, board_px, panels, label) in enumerate(rows):
        for c, panel in enumerate(panels):
            s = draw_panel(axes[r, c], image, panel, mask, board_px)
            scatter = scatter or s
        axes[r, 0].text(-0.03, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=9)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, fraction=0.015, label="ICP: |n · d|, resists a slide along the track")
    fig.suptitle(title + "   (red hatch: tags masked from the markerless methods; gold dashes: board)", fontsize=10)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ====== PER RECORDING ======
def run(cfg, recording, args):
    import matplotlib
    matplotlib.use("Agg")
    det = cfg["detections"]
    p = recording_paths(cfg, recording)
    rig = load_rig(common.project_path(recording["calibration"]))
    frames_dir = common.project_path(recording["frames"])
    names = [n for n, _, _ in find_frame_pairs(frames_dir)][::frame_step(cfg, recording)]
    tagged = has_tags(recording)
    reference = pose_files.read_trajectory(p["reference"])[1] if tagged else {}
    used_tags = tags_per_frame(p["reference"]) if tagged else {}
    outline = board_outline(p["reference"]) if tagged else None
    colmap = next(e for e in cfg["estimators"] if e["kind"] == "colmap")
    icps = [e for e in cfg["estimators"] if e["kind"] == "icp"]
    keypoints = colmap_keypoints(p["colmap"]) if os.path.exists(os.path.join(p["colmap"], "best_model.txt")) else {}
    direction_source = "tag poses"
    if not reference:   # no tags: COLMAP's directions, or the first ICP's if COLMAP posed nothing
        candidates = [p["poses"](e["name"]) for e in [colmap, *icps]]
        direction_source = next(os.path.basename(c)[:-5] for c in candidates if os.path.exists(c))
    source = reference or pose_files.read_trajectory(p["poses"](direction_source))[1]
    directions = travel_directions(names, source)
    out_dir = os.path.join(p["out"], "detections")
    os.makedirs(out_dir, exist_ok=True)
    threshold = det["slide_threshold"]
    label = cfg["reference"]["label"]

    grip_path = os.path.join(out_dir, "grip.json")
    if args.images_only and os.path.exists(grip_path):
        with open(grip_path, encoding="utf-8") as fh:
            grip = json.load(fh)
    else:
        print(f"{recording['name']}: grip over {len(names)} frames (direction of travel from {direction_source})", flush=True)
        per = {"aprilgrid": {}, colmap["name"]: {}, **{e["name"]: {} for e in icps},
               **{e["name"] + "_points": {} for e in icps}}
        for k, name in enumerate(names):
            per["aprilgrid"][name] = 4 * len(used_tags.get(name, [])) if name in reference else 0
            kp = keypoints.get(f"left/{name}")
            per[colmap["name"]][name] = int((kp[:, 2] > 0).sum()) if kp is not None else 0
            for e in icps:
                xyz, normals = icp_points(p["depth"](e["depth"]), name, det)
                per[e["name"]][name] = int((np.abs(normals @ directions[name]) > threshold).sum())
                per[e["name"] + "_points"][name] = int(len(xyz))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(names)}", flush=True)
        grip = {"direction_of_travel_from": direction_source, "slide_threshold": threshold,
                "icp_voxel_mm": det["icp_voxel_mm"], "frames": len(names),
                "median": {m: median(v.values()) for m, v in per.items()}, "per_frame": per}

    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, det["frames"] + 2)[1:-1]]
    grip["drawn"] = picks
    with open(grip_path, "w", encoding="utf-8") as fh:
        json.dump(grip, fh, indent=1)

    detector = make_detector()
    rows = []
    for name in picks:
        left = os.path.join(frames_dir, "left", name)
        image = cv2.cvtColor(cv2.imread(left, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask_path = os.path.join(p["masks"], "left", name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if tagged and os.path.exists(mask_path) else None
        board_px = None
        if name in reference:
            T = reference[name]
            board_px = project(outline @ T[:3, :3].T + T[:3, 3], rig)
        used = set(used_tags.get(name, []))
        tags = {t: (q, t in used) for t, q in detect(detector, left).items()}
        panels = [{"kind": "aprilgrid", "method": "aprilgrid", "tags": tags,
                   "title": f"{label}\n{4 * len(used)} tag corners used"}]
        kp = keypoints.get(f"left/{name}", np.zeros((0, 3)))
        panels.append({"kind": "colmap", "method": colmap["name"], "keypoints": kp,
                       "title": f"{colmap['label']}\n{int((kp[:, 2] > 0).sum()):,} SIFT points used"})
        for e in icps:
            xyz, normals = icp_points(p["depth"](e["depth"]), name, det)
            score = np.abs(normals @ directions[name])
            panels.append({"kind": "icp", "method": e["name"], "px": project(xyz, rig), "score": score,
                           "title": f"{e['label']}\n{int((score > threshold).sum()):,} of {len(xyz):,} points resist"})
        stem = name[:-4]
        for panel in panels:
            save_single(os.path.join(out_dir, f"{stem}_{panel['method']}.png"), image, panel, mask, board_px, det["image_dpi"])
        save_grid(os.path.join(out_dir, f"{stem}_all.png"), [(image, mask, board_px, panels, f"frame {stem}")],
                  det["image_dpi"], f"{recording.get('title', recording['name'])}, frame {stem}")
        rows.append((image, mask, board_px, panels, f"frame {stem}"))
        print(f"  drew {name}", flush=True)
    save_grid(os.path.join(out_dir, "grid.png"), rows, det["image_dpi"], recording.get("title", recording["name"]))
    print(f"  median grip: {grip['median']}\n-> {common.rel(out_dir)}/")


def main(argv=None):
    args = parse_args(argv)
    cfg = common.load_config(args.config)
    for recording in cfg["recordings"]:
        if not args.recordings or recording["name"] in args.recordings:
            run(cfg, recording, args)


if __name__ == "__main__":
    main()
