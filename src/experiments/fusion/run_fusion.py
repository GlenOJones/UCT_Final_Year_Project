"""The pipeline runs of the fusion comparison: full COLMAP, and TSDF on each depth source, at the same poses.

All three reconstructions start from the pose comparison's outputs for the recording (reproduce.py
pose_comparison): its COLMAP SfM model and the trajectory written from it, its tag masks, and its
per-frame FoundationStereo and tuned-SGBM depth. Nothing is re-matched or re-posed here, so the methods
differ only in how depth is made and fused:

  colmap     COLMAP's own dense pipeline on that SfM model: image_undistorter, PatchMatch stereo with
             geometric consistency, stereo_fusion (colmap_mvs.py's stage functions and defaults). COLMAP
             cannot take poses from outside, so the poses the TSDF runs use are the ones this model made:
             the same poses for all three. The tag masks are undistorted with the images and passed to
             stereo_fusion, so tag pixels are dropped as they were from the depth frames.
  tsdf_*     tsdf_fuse.py on one depth source at those poses, under a memory cap (systemd-run).

Every method uses the same frames: every frame_step'th posed frame. For evaluation only, every cloud is then
moved into the board frame by one rigid fit of the COLMAP trajectory onto the tag poses (camera centres plus
lever points along each camera axis, as evaluate_poses.py does), so the CAD models can be placed on it.

    src/venv/bin/python src/experiments/fusion/run_fusion.py --config src/experiments/fusion/config.json

Writes, under work/<work_dir>/<recording>/:
    poses_used.yaml               the COLMAP trajectory cut to the frames used
    colmap/                       COLMAP dense workspace (dense/fused.ply: SfM model frame)
    tsdf_<source>/                tsdf_fuse.py output (cloud.ply, mesh.ply: COLMAP world frame)
    <method>/cloud_board.ply      each method's cloud in the board frame (and mesh_board.ply for TSDF)
and under results/<out_dir>/<recording>/: alignment.json (the board fit and its residuals) and
run_fusion.json (frames, timings, point counts).

With skip_existing, a reconstruction whose output exists is not run again; the board-frame copies are
always remade (seconds).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
SRC = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(EXPERIMENTS, "pose_estimation"))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "03_Reconstruction"))
import colmap_mvs  # noqa: E402
import common  # noqa: E402
import pose_files  # noqa: E402
from evaluate_poses import pose_points, umeyama  # noqa: E402
from stereo_rig import CAMERAS, invert  # noqa: E402

TSDF_SCRIPT = os.path.join("src", "03_Reconstruction", "tsdf_fuse.py")


def recording_paths(cfg):
    rec = cfg["recording"]
    out = os.path.join(common.project_path(cfg["out_dir"]), rec["name"])
    work = os.path.join(common.project_path(cfg["work_dir"]), rec["name"])
    return {"out": out, "work": work,
            "poses": common.project_path(rec["poses"]),
            "poses_used": os.path.join(work, "poses_used.yaml"),
            "sfm": common.project_path(rec["sfm_workspace"]),
            "masks": common.project_path(rec["tag_masks"]),
            "reference": common.project_path(rec["reference_poses"]),
            "colmap": os.path.join(work, "colmap"),
            "method": lambda name: os.path.join(work, name)}


def check_inputs(p, cfg):
    needed = [p["poses"], os.path.join(p["sfm"], "best_model.txt"), p["masks"], p["reference"]]
    needed += [common.project_path(d) for d in cfg["recording"]["depth"].values()]
    missing = [common.rel(n) for n in needed if not os.path.exists(n)]
    if missing:
        raise SystemExit("missing inputs (made by reproduce.py pose_comparison):\n  " + "\n  ".join(missing))


# ====== FRAMES AND POSES ======
def frames_used(cfg, p):
    """The posed frames every method uses (every frame_step'th), and the full {name: T_cam_world}."""
    metadata, poses, diagnostics = pose_files.read_trajectory(p["poses"])
    names = [n for n in sorted(set(diagnostics) | set(poses)) if n in poses][::cfg["frame_step"]]
    pose_files.write_trajectory(
        p["poses_used"], "src/experiments/fusion/run_fusion.py", metadata["world_frame"],
        {"source": common.rel(p["poses"]), "frame_step": cfg["frame_step"], "frames": len(names)},
        [{"name": n, "T_cam_world": poses[n]} for n in names])
    print(f"{len(names)} of {len(poses)} posed frames used (every {cfg['frame_step']}) -> {common.rel(p['poses_used'])}")
    return names, poses


def board_alignment(cfg, p, poses):
    """T_board_world: one rigid fit of the COLMAP trajectory onto the tag poses, over every frame both posed."""
    _, reference, _ = pose_files.read_trajectory(p["reference"])
    names = sorted(set(reference) & set(poses))
    lever = cfg["evaluation"]["align_lever_mm"]
    source = np.vstack([pose_points(invert(poses[n]), lever) for n in names])
    target = np.vstack([pose_points(invert(reference[n]), lever) for n in names])
    _, R, t = umeyama(source, target, with_scale=False)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    centres = np.array([(T @ invert(poses[n]))[:3, 3] - invert(reference[n])[:3, 3] for n in names])
    error = np.linalg.norm(centres, axis=1)
    record = {"T_board_world": T.tolist(), "frames": len(names), "lever_mm": lever,
              "camera_centre_residual_mm": {"median": float(np.median(error)), "rmse": float(np.sqrt(np.mean(error ** 2))),
                                            "max": float(error.max())}}
    print(f"board fit over {len(names)} frames: camera centres within {np.median(error):.2f} mm median, "
          f"{error.max():.2f} mm max")
    return T, record


# ====== COLMAP ======
def link(source, target):
    if not os.path.lexists(target):
        os.symlink(source, target)


def undistort_masks(ws, masks, names, c):
    """The tag masks in the undistorted images' geometry, where stereo_fusion reads them: undistorted by
    COLMAP itself, from the same model and image list, linked as dense/masks/<camera>/<image>.png."""
    staged = os.path.join(ws, "mask_images")
    for camera in CAMERAS:
        os.makedirs(os.path.join(staged, camera), exist_ok=True)
        for name in names:
            link(os.path.join(masks, camera, name), os.path.join(staged, camera, name))
    undistorted = os.path.join(ws, "mask_dense")
    shutil.rmtree(undistorted, ignore_errors=True)
    colmap_mvs.colmap(ws, "image_undistorter", {
        "image_path": staged, "input_path": colmap_mvs.best_model(ws), "output_path": undistorted,
        "output_type": "COLMAP", "max_image_size": c["max_image_size"],
        "image_list_path": os.path.join(ws, "dense_images.txt")})
    folder = os.path.join(ws, "dense", "masks")
    for camera in CAMERAS:
        os.makedirs(os.path.join(folder, camera), exist_ok=True)
        for name in names:
            link(os.path.join(undistorted, "images", camera, name), os.path.join(folder, camera, name + ".png"))
    return folder


def run_colmap(cfg, p, names, skip_existing):
    """COLMAP dense on the SfM model the poses came from. The SfM workspace is only read: its images and
    model are linked into a workspace of this experiment's own."""
    ws, c = p["colmap"], cfg["colmap"]
    fused = os.path.join(ws, "dense", "fused.ply")
    if skip_existing and os.path.exists(fused):
        print(f"=== colmap: {common.rel(fused)} exists, skipped")
        return {"skipped": True}
    os.makedirs(ws, exist_ok=True)
    free_gb = shutil.disk_usage(ws).free / 1e9
    if free_gb < c["min_free_disk_gb"]:
        raise SystemExit(f"only {free_gb:.0f} GB free; COLMAP dense needs ~{c['min_free_disk_gb']} GB here")
    link(os.path.join(p["sfm"], "images"), os.path.join(ws, "images"))
    link(os.path.join(p["sfm"], "sparse"), os.path.join(ws, "sparse"))
    shutil.copy(os.path.join(p["sfm"], "best_model.txt"), os.path.join(ws, "best_model.txt"))
    settings = argparse.Namespace(**c)
    seconds = {}
    print("\n=== colmap", flush=True)
    start = time.time()
    # The names are already every frame_step'th, so every one of them gets a depth map.
    colmap_mvs.stage_undistort(ws, c["max_image_size"], c["source_images"], names, 1)
    if c["sources"] == "rig":
        colmap_mvs.write_source_lists(ws, c["source_images"])
    mask_path = undistort_masks(ws, p["masks"], names, c) if c["mask_fusion"] else None
    seconds["undistort"] = round(time.time() - start, 1)
    start = time.time()
    colmap_mvs.stage_patchmatch(ws, settings)
    seconds["patchmatch"] = round(time.time() - start, 1)
    start = time.time()
    colmap_mvs.stage_fuse(ws, settings, mask_path)
    seconds["fuse"] = round(time.time() - start, 1)
    return {"seconds": seconds, "images": 2 * len(names)}


def colmap_world_from_model(p, poses):
    """T_world_model taking COLMAP's model frame into the trajectory's world frame, from the left images'
    poses in the undistorted model; their spread checks that model and trajectory agree."""
    ws = p["colmap"]
    model = colmap_mvs.read_model_images(ws, os.path.join(ws, "dense", "sparse"))
    names = [n for n in sorted(poses) if f"left/{n}" in model]
    transforms = [invert(poses[n]) @ model[f"left/{n}"] for n in names]
    T = transforms[0]
    spread = max(np.linalg.norm(t[:3, 3] - T[:3, 3]) for t in transforms)
    if spread > 0.1:
        raise SystemExit(f"COLMAP model and {common.rel(p['poses'])} disagree by {spread:.2f} mm: not the same model?")
    return T


# ====== TSDF ======
def run_tsdf(cfg, p, method, skip_existing):
    out = p["method"](method["name"])
    record = {"script": TSDF_SCRIPT}
    if skip_existing and os.path.exists(os.path.join(out, "tsdf_fuse.json")):
        print(f"=== {method['name']}: {common.rel(out)} exists, skipped")
        return {**record, "skipped": True}
    argv = ["--depth-dir", common.project_path(cfg["recording"]["depth"][method["depth"]]),
            "--poses", p["poses_used"], "--out-dir", out, *cfg["tsdf"]["argv"]]
    command = [common.project_path(cfg["python"]), common.project_path(TSDF_SCRIPT), *argv]
    if shutil.which("systemd-run"):
        command = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cfg['tsdf']['memory_max']}",
                   "-p", "MemorySwapMax=0", *command]
    else:
        print("WARNING: no systemd-run, so TSDF runs without a memory cap")
    print(f"\n=== {method['name']}", flush=True)
    start = time.time()
    if subprocess.run(command, cwd=common.PROJECT_ROOT).returncode != 0:
        raise SystemExit(f"{method['name']} failed (killed by the {cfg['tsdf']['memory_max']} memory cap?)")
    return {**record, "argv": [common.rel(a) if os.path.isabs(a) else a for a in argv],
            "seconds": round(time.time() - start, 1)}


# ====== BOARD-FRAME COPIES ======
def export_board(cfg, p, poses, T_board_world):
    """Every method's cloud (and TSDF mesh) moved into the board frame: <work>/<method>/cloud_board.ply."""
    counts = {}
    for method in cfg["methods"]:
        folder = p["method"](method["name"])
        os.makedirs(folder, exist_ok=True)
        if method["kind"] == "colmap":
            transform = T_board_world @ colmap_world_from_model(p, poses)
            sources = {"cloud_board.ply": os.path.join(p["colmap"], "dense", "fused.ply")}
        else:
            transform = T_board_world
            sources = {"cloud_board.ply": os.path.join(folder, "cloud.ply"), "mesh_board.ply": os.path.join(folder, "mesh.ply")}
        for target, source in sources.items():
            if not os.path.exists(source):
                continue
            if target.startswith("mesh"):
                mesh = o3d.io.read_triangle_mesh(source)
                mesh.transform(transform)
                o3d.io.write_triangle_mesh(os.path.join(folder, target), mesh)
            else:
                cloud = o3d.io.read_point_cloud(source)
                cloud.transform(transform)
                o3d.io.write_point_cloud(os.path.join(folder, target), cloud)
                counts[method["name"]] = len(cloud.points)
        print(f"  {method['name']}: {counts.get(method['name'], 0)} points -> {common.rel(folder)}/cloud_board.ply")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    p = recording_paths(cfg)
    check_inputs(p, cfg)
    os.makedirs(p["out"], exist_ok=True)
    names, poses = frames_used(cfg, p)

    runs = []
    for method in cfg["methods"]:
        if method["kind"] == "colmap":
            record = run_colmap(cfg, p, names, args.skip_existing)
        else:
            record = run_tsdf(cfg, p, method, args.skip_existing)
        runs.append({"step": f"{cfg['recording']['name']}: {method['name']}", **record})

    print("\n=== board frame (evaluation only)")
    T_board_world, alignment = board_alignment(cfg, p, poses)
    with open(os.path.join(p["out"], "alignment.json"), "w", encoding="utf-8") as fh:
        json.dump(alignment, fh, indent=1)
    counts = export_board(cfg, p, poses, T_board_world)
    with open(os.path.join(p["out"], "run_fusion.json"), "w", encoding="utf-8") as fh:
        json.dump({"frames": names, "frame_step": cfg["frame_step"], "runs": runs, "points": counts}, fh, indent=1)
    return runs


if __name__ == "__main__":
    main()
