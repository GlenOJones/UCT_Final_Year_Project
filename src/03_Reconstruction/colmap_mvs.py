"""Dense reconstruction of a scan with COLMAP: rig-constrained SfM, then multi-view PatchMatch stereo.

This is the multi-view counterpart of dense_stereo.py. There, each frame pair is matched on its own
and the frames only meet when their points are voted on. Here every depth estimate compares a pixel
against many neighbouring views at once, which is what gives PatchMatch its edge on weak texture.

The stages, each a COLMAP command unless noted:
  workspace    link the frame pairs tag_poses.py could pose into <workspace>/images/{left,right}/
               (--all-frames links the rest too, for SfM to register from image features alone)
  features     SIFT, one run per camera so each gets its own calibrated intrinsics (FULL_OPENCV,
               which keeps k3; the OPENCV model would drop it)
  rig          declare left/right a rigid rig with the calibrated extrinsics (rig_configurator)
  match        SIFT matching on a pair list: each frame's left-right pair plus every camera
               combination within --window frames, which is all a 1-DOF trolley can overlap with
  map          incremental SfM with intrinsics and the rig held FIXED, so only the trolley poses and
               the points are estimated, and the 150 mm baseline carries metric scale into the model
  align        (Python) rigid transform taking the SfM camera poses onto the poses from
               tag_poses.py, which puts the model in the board frame. The best-fit SCALE is reported
               but not applied (see stage_align), and the residual checks both pose sources
  undistort    image_undistorter: pinhole images for dense matching, of every --dense-step'th frame
  patchmatch   patch_match_stereo with geometric consistency (needs CUDA)
  fuse         stereo_fusion into one cloud
  export       (Python) move the fused cloud into the board frame -> results/<session>/<scan>/colmap/

    src/venv/bin/python src/03_Reconstruction/colmap_mvs.py --session Sep24 --scan mjpg_pyr_lights_2
    src/venv/bin/python src/03_Reconstruction/colmap_mvs.py --session Sep24 --scan mjpg_pyr_lights_2 --from-stage patchmatch

The working folder (database, sparse model, depth maps: gigabytes) is work/<session>/<scan>/colmap/;
the clouds (cloud.ply, sparse.ply) go to results/<session>/<scan>/colmap/.

Written against COLMAP 4.3 built with CUDA: SIFT extraction, matching and PatchMatch run on the GPU.
Options that differ between COLMAP releases are only passed if `colmap <command> -h` lists them.

Run tag_poses.py first; its YAML names the frames and the calibration. The resulting cloud is in the
same board frame as dense_stereo.py's, so postprocess_cloud.py and view_cloud.py take either.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import CAMERAS, PROJECT_ROOT, find_frame_pairs, invert, load_rig, rel  # noqa: E402
import project_paths as paths  # noqa: E402

STAGES = ("workspace", "features", "rig", "match", "map", "align", "undistort", "patchmatch", "fuse", "export")

COLMAP = os.environ.get("COLMAP", "colmap")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="COLMAP SfM + PatchMatch MVS for a tag-posed stereo scan.")
    paths.add_scan_arguments(ap, method="colmap")
    ap.add_argument("--workspace", default=None,
                    help="COLMAP working folder (default work/<session>/<scan>/<method>)")
    ap.add_argument("--from-stage", default="workspace", choices=STAGES, help="resume from this stage")
    ap.add_argument("--to-stage", default="export", choices=STAGES, help="stop after this stage")
    ap.add_argument("--all-frames", action="store_true",
                    help="also use the frames with no tag pose (default: tag-posed frames only)")
    ap.add_argument("--dense-step", type=int, default=1,
                    help="depth maps for every Nth frame only; SfM still uses every frame. Neighbouring "
                         "frames are ~3.3 mm apart, so 3-10 loses little and is 3-10x faster (default 1)")
    ap.add_argument("--window", type=int, default=12,
                    help="frames either side that are matched against each other (default 12)")
    ap.add_argument("--max-image-size", type=int, default=1600,
                    help="px, longest side used for dense matching (default 1600 = full resolution)")
    ap.add_argument("--window-radius", type=int, default=7,
                    help="PatchMatch window half-size, px; larger helps weak texture (COLMAP default 5)")
    ap.add_argument("--align-scale", action="store_true",
                    help="let the board alignment scale the model to the tags (default: rigid)")
    # COLMAP's default of 3 deg removed most of each depth map here. The source views it picks are
    # mostly neighbouring trolley frames, 3.3 mm apart at ~450 mm (~0.4 deg each), so few pixels
    # reach 3 deg. On frame 0100 of Sep24/mjpg_pyr2: 3 deg kept 7% of pixels (two tags, the pyramid
    # outline); 1 deg kept 31% (all four tags, the board edge, the pyramid outline and ridges).
    ap.add_argument("--min-triangulation-angle", type=float, default=1.0,
                    help="deg; PatchMatch drops pixels no source view sees at this angle or more "
                         "(COLMAP default 3)")
    ap.add_argument("--sources", default="rig", choices=("rig", "auto"),
                    help="how PatchMatch picks the views each depth map is matched against: 'rig' = the "
                         "stereo partner plus the frames at the best triangulation angle (see "
                         "write_source_lists); 'auto' = COLMAP's choice by shared sparse points")
    ap.add_argument("--source-images", type=int, default=20,
                    help="neighbouring views each depth map is matched against (COLMAP default 20)")
    ap.add_argument("--min-fused-pixels", type=int, default=3,
                    help="depth maps that must agree for a fused point (COLMAP default 5)")
    ap.add_argument("--cache-gb", type=int, default=12,
                    help="GB of depth/normal maps PatchMatch and fusion keep in RAM (COLMAP default 32, "
                         "which is more than this 30 GB machine has)")
    ap.add_argument("--gpu", default="0", help="GPU index for COLMAP, or -1 for CPU where supported")
    args = ap.parse_args(argv)
    paths.require_scan(args, ap)
    return args


# ====== COLMAP PLUMBING ======
_help_cache = {}


def supports(command, option):
    """Whether this COLMAP build knows an option. COLMAP renames options between releases (rig
    support arrived in 3.12), so options that are only improvements are passed only if known."""
    if command not in _help_cache:
        result = subprocess.run([COLMAP, command, "-h"], capture_output=True, text=True)
        _help_cache[command] = result.stdout + result.stderr
    return f"--{option} " in _help_cache[command] or f"--{option}\n" in _help_cache[command]


def colmap(workspace, command, options, optional=None):
    """Run one COLMAP command, logging to <workspace>/logs/<command>.log. `optional` options are
    dropped, with a note, if this build does not have them."""
    args = [COLMAP, command]
    for key, value in options.items():
        args += [f"--{key}", str(value)]
    for key, value in (optional or {}).items():
        if supports(command, key):
            args += [f"--{key}", str(value)]
        else:
            print(f"  (this COLMAP has no --{key}; skipped)")
    os.makedirs(os.path.join(workspace, "logs"), exist_ok=True)
    log_path = os.path.join(workspace, "logs", f"{command}.log")
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(args)}\n")
        log.flush()
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            tail = log.readlines()[-15:]
        raise SystemExit(f"colmap {command} failed ({result.returncode}); end of {rel(log_path)}:\n" + "".join(tail))
    print(f"  colmap {command}: {time.time() - start:.0f} s")


# ====== INPUTS ======
def load_poses(path):
    """(frames_dir, calibration path, {frame name: T_cam_board}, {tag id: 4x3 corners}) from
    tag_poses.py's YAML."""
    if not os.path.exists(path):
        raise SystemExit(f"no poses at {rel(path)} - run src/03_Reconstruction/tag_poses.py first")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    meta = fs.getNode("metadata")
    frames_dir = os.path.join(PROJECT_ROOT, meta.getNode("frames_dir").string())
    calibration = os.path.join(PROJECT_ROOT, meta.getNode("calibration").string())
    corners_node = fs.getNode("tags").getNode("corners_mm")
    ids_node = fs.getNode("tags").getNode("ids")
    tags_corners = {int(ids_node.at(i).real()): corners_node.at(i).mat() for i in range(ids_node.size())}
    poses = {}
    frames = fs.getNode("frames")
    for i in range(frames.size()):
        node = frames.at(i)
        if not node.getNode("T_cam_board").isNone():
            poses[node.getNode("name").string()] = node.getNode("T_cam_board").mat()
    fs.release()
    return frames_dir, calibration, poses, tags_corners


# ====== STAGES ======
def stage_workspace(ws, frames_dir, names):
    """Symlink the chosen frame pairs into images/left and images/right. COLMAP names an image by
    its path under images/, so "left/0123.png" and "right/0123.png" are one frame of the rig."""
    pairs = [p for p in find_frame_pairs(frames_dir) if p[0] in set(names)]
    shutil.rmtree(os.path.join(ws, "images"), ignore_errors=True)
    for camera in CAMERAS:
        folder = os.path.join(ws, "images", camera)
        os.makedirs(folder, exist_ok=True)
    for name, left, right in pairs:
        for camera, source in (("left", left), ("right", right)):
            link = os.path.join(ws, "images", camera, name)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(source), link)
    print(f"  {len(pairs)} frame pairs linked")
    return [name for name, _, _ in pairs]


def full_opencv_params(rig, camera):
    """fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 for COLMAP's FULL_OPENCV model. k4..k6 are the
    rational denominator terms OpenCV's 5-coefficient model does not have, so they are zero."""
    K, D = rig.K[camera], rig.D[camera].ravel()
    k1, k2, p1, p2, k3 = (list(D) + [0.0] * 5)[:5]
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2], k1, k2, p1, p2, k3, 0.0, 0.0, 0.0]


def stage_features(ws, rig, names, gpu):
    """SIFT on each camera's images separately, so each camera is created with its own intrinsics."""
    database = os.path.join(ws, "database.db")
    if os.path.exists(database):
        os.remove(database)
    for camera in CAMERAS:
        image_list = os.path.join(ws, f"{camera}_images.txt")
        with open(image_list, "w", encoding="utf-8") as fh:
            fh.write("".join(f"{camera}/{n}\n" for n in names))
        colmap(ws, "feature_extractor", {
            "database_path": database,
            "image_path": os.path.join(ws, "images"),
            "image_list_path": image_list,
            "ImageReader.camera_model": "FULL_OPENCV",
            "ImageReader.single_camera": 1,
            "ImageReader.camera_params": ",".join(f"{v:.10g}" for v in full_opencv_params(rig, camera)),
        }, optional={"FeatureExtraction.use_gpu": 0 if gpu == "-1" else 1,
                     "FeatureExtraction.gpu_index": gpu,
                     # The board is low-texture, so take every feature there is (default 8192).
                     "SiftExtraction.max_num_features": 16384,
                     # Domain-size pooling makes descriptors more robust to the viewpoint change
                     # along the trolley. (Affine shape estimation would help too, but COLMAP only
                     # implements it on the CPU, which would forfeit GPU extraction.)
                     "SiftExtraction.domain_size_pooling": 1})


def rotation_to_quaternion(R):
    """(w, x, y, z) unit quaternion of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def stage_rig(ws, rig):
    """Declare the stereo pair a rig: left is the reference sensor, right sits at the calibrated
    (R, T) from it. COLMAP groups images into frames by the name after the prefix."""
    if not supports("rig_configurator", "rig_config_path"):
        print("  this COLMAP has no rig_configurator (needs 3.12+): cameras will be posed independently")
        return False
    config = [{"cameras": [
        {"image_prefix": "left/", "ref_sensor": True},
        {"image_prefix": "right/",
         "cam_from_rig_rotation": rotation_to_quaternion(rig.R).tolist(),
         "cam_from_rig_translation": rig.T.ravel().tolist()},
    ]}]
    path = os.path.join(ws, "rig_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    colmap(ws, "rig_configurator", {"database_path": os.path.join(ws, "database.db"), "rig_config_path": path})
    return True


def stage_match(ws, names, window, gpu):
    """Match the pairs a trolley scan can overlap: every camera combination within `window` frames."""
    pairs = set()
    for i, a in enumerate(names):
        for j in range(i, min(i + window + 1, len(names))):
            b = names[j]
            for ca in CAMERAS:
                for cb in CAMERAS:
                    first, second = f"{ca}/{a}", f"{cb}/{b}"
                    if first != second:
                        pairs.add(tuple(sorted((first, second))))
    path = os.path.join(ws, "pairs.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    print(f"  {len(pairs)} image pairs to match")
    colmap(ws, "matches_importer", {
        "database_path": os.path.join(ws, "database.db"),
        "match_list_path": path,
        "match_type": "pairs",
    }, optional={"FeatureMatching.use_gpu": 0 if gpu == "-1" else 1,
                 "FeatureMatching.gpu_index": gpu,
                 # Second matching pass restricted to the verified two-view geometry: more
                 # correct matches on repetitive or weak texture.
                 "FeatureMatching.guided_matching": 1})


def stage_map(ws, has_rig):
    """Incremental SfM with everything the calibration already knows held fixed."""
    sparse = os.path.join(ws, "sparse")
    shutil.rmtree(sparse, ignore_errors=True)
    os.makedirs(sparse)
    colmap(ws, "mapper", {
        "database_path": os.path.join(ws, "database.db"),
        "image_path": os.path.join(ws, "images"),
        "output_path": sparse,
        "Mapper.ba_refine_focal_length": 0,
        "Mapper.ba_refine_principal_point": 0,
        "Mapper.ba_refine_extra_params": 0,
    }, optional={"Mapper.ba_refine_sensor_from_rig": 0} if has_rig else {})
    models = [d for d in os.listdir(sparse) if os.path.isdir(os.path.join(sparse, d))]
    if not models:
        raise SystemExit("mapper produced no model")
    # Several models means SfM broke the scan into pieces; keep the one with the most images.
    sizes = {m: len(read_model_images(ws, os.path.join(sparse, m))) for m in models}
    best = max(sizes, key=sizes.get)
    print("  models (images registered): " + ", ".join(f"{m}: {n}" for m, n in sorted(sizes.items())))
    with open(os.path.join(ws, "best_model.txt"), "w", encoding="utf-8") as fh:
        fh.write(best + "\n")


def best_model(ws):
    with open(os.path.join(ws, "best_model.txt"), encoding="utf-8") as fh:
        return os.path.join(ws, "sparse", fh.read().strip())


def read_model_images(ws, model):
    """{image name: 4x4 T_cam_world} from a COLMAP model, via a text export of it."""
    text_dir = model + "_txt"
    os.makedirs(text_dir, exist_ok=True)
    colmap(ws, "model_converter", {"input_path": model, "output_path": text_dir, "output_type": "TXT"})
    poses = {}
    with open(os.path.join(text_dir, "images.txt"), encoding="utf-8") as fh:
        lines = [line for line in fh if not line.startswith("#")]
    for line in lines[::2]:   # each image is a pose line followed by a line of 2D points
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        R = quaternion_to_rotation(qw, qx, qy, qz)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, (tx, ty, tz)
        poses[parts[9]] = T
    return poses


def quaternion_to_rotation(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama(source, target, with_scale=True):
    """Similarity (scale, R, t) with target ~= scale * R @ source + t, least squares.
    With with_scale=False the scale is fixed at 1 and the fit is rigid."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    cs, ct = source - mu_s, target - mu_t
    U, S, Vt = np.linalg.svd(ct.T @ cs / len(source))
    flip = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ flip @ Vt
    scale = np.trace(np.diag(S) @ flip) / np.mean(np.sum(cs ** 2, 1)) if with_scale else 1.0
    return scale, R, mu_t - scale * R @ mu_s


# Each camera pose enters the alignment as its centre plus the points this far out along its three
# axes, so orientation counts as well as position. 450 mm is roughly the camera-to-board distance, so
# the fit is weighted towards agreeing where the board and object actually are.
ALIGN_LEVER_MM = 450.0


def pose_points(T_cam_world):
    """World positions of the camera centre and of ALIGN_LEVER_MM along each camera axis, (4, 3)."""
    local = np.vstack([np.zeros(3), np.eye(3) * ALIGN_LEVER_MM])
    T_world_cam = invert(T_cam_world)
    return local @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def tag_points(ws, S, tags_corners):
    """SfM points lying on the reference tags, in the board frame given by S. Returns the points
    and {tag id: boolean mask of its points}."""
    points = []
    with open(os.path.join(best_model(ws) + "_txt", "points3D.txt"), encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                points.append([float(v) for v in line.split()[1:4]])
    xyz = np.array(points) @ S[:3, :3].T + S[:3, 3]
    # Within the inner 70 mm of each 80 mm tag, and near the board, so edge points and the object
    # do not count.
    masks = {tag_id: np.all(np.abs(xyz[:, :2] - corners.mean(0)[:2]) < 35, axis=1) & (np.abs(xyz[:, 2]) < 60)
             for tag_id, corners in tags_corners.items()}
    return xyz, masks


def plane_fit(xyz):
    """z = a x + b y + c through the points: returns (a, b, c) and the tilt in degrees."""
    a, b, c = np.linalg.lstsq(np.c_[xyz[:, :2], np.ones(len(xyz))], xyz[:, 2], rcond=None)[0]
    return (a, b, c), float(np.degrees(np.arctan(np.hypot(a, b))))


def tag_plane_check(ws, S, tags_corners):
    """Height (mm) of the SfM points on each reference tag once in the board frame, and the tilt
    of a plane fitted through them. The tags ARE the board, so both should be ~0 if the alignment
    is right; this is the check that caught the roll error of fitting camera centres alone."""
    xyz, masks = tag_points(ws, S, tags_corners)
    heights = {t: float(np.median(xyz[m, 2])) if m.any() else float("nan") for t, m in masks.items()}
    return heights, plane_fit(xyz[np.any(list(masks.values()), axis=0)])[1]


def level_on_tags(ws, S, tags_corners):
    """Correction C so that, in C @ S, the plane through the SfM points on the tags is z = 0.

    The pose alignment leaves the board frame only as good as the tag poses, which carry ~1.5 mm of
    out-of-plane scatter in the layout; on Sep24/mjpg_pyr2 the SfM tag points came out 0.9-3.4 mm
    below the board and tilted 0.55 deg. SfM has hundreds of points on every tag, so its own plane
    through them is the better estimate of the board. The correction rotates about the centroid of
    those points and shifts along z only, so positions in the board plane are unchanged."""
    xyz, masks = tag_points(ws, S, tags_corners)
    on_tags = xyz[np.any(list(masks.values()), axis=0)]
    (a, b, c), _ = plane_fit(on_tags)
    normal = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
    axis = np.cross(normal, [0.0, 0.0, 1.0])
    angle = np.arcsin(np.linalg.norm(axis))
    R = cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0] if angle > 1e-12 else np.eye(3)
    pivot = on_tags.mean(0)
    pivot_on_plane = np.array([pivot[0], pivot[1], a * pivot[0] + b * pivot[1] + c])
    C = np.eye(4)
    C[:3, :3] = R
    C[:3, 3] = np.array([pivot[0], pivot[1], 0.0]) - R @ pivot_on_plane
    return C


def stage_align(ws, tag_poses, tags_corners, rig, with_scale, sparse_path):
    """Transform from the SfM world onto the board frame, fitted on the camera poses.

    Positions AND orientations are matched (see ALIGN_LEVER_MM). Fitting camera centres alone was
    tried first and is not enough on a 1-DOF trolley: the centres lie close to one straight line,
    which leaves the roll about that line almost free. On Sep24/mjpg_pyr2 it left the board tilted
    2.9 deg about x and 10 mm off z = 0 in the aligned model (tags at +1.7 to +21 mm).

    Rigid by default, because the SfM model's scale is the better one. On Sep24/mjpg_pyr2 the
    similarity fit wanted a scale of 1.0091, i.e. the tag-based frame is 0.9% larger than the SfM
    model - and the tags triangulated to 80.79 mm against 80 mm printed, which is 80.07 mm at SfM
    scale. The bundle adjustment, with the rig baseline fixed and features over the whole image, gets
    the scale right; the per-tag triangulation is inflated by the calibration's weak periphery.
    Scaling the model onto the tags would copy that error into the cloud.

    Both cameras of every tag-posed frame are used: the left from tag_poses.py directly, the right
    through the calibrated extrinsics. Images whose pose disagrees by more than 3x the median are
    dropped and the fit repeated, so a few bad tag poses cannot tilt the board frame.
    """
    sfm = read_model_images(ws, best_model(ws))
    T_right_left = np.eye(4)
    T_right_left[:3, :3], T_right_left[:3, 3] = rig.R, rig.T.ravel()
    source, target, labels = [], [], []
    for name, T_cam_board in tag_poses.items():
        for camera, T_cam_board_cam in (("left", T_cam_board), ("right", T_right_left @ T_cam_board)):
            key = f"{camera}/{name}"
            if key in sfm:
                source.append(pose_points(sfm[key]))
                target.append(pose_points(T_cam_board_cam))
                labels.append(key)
    source, target = np.array(source), np.array(target)   # (images, 4, 3)
    if len(source) < 3:
        raise SystemExit(f"only {len(source)} SfM images also have a tag pose: cannot align")

    def fit(keep, scaled):
        return umeyama(source[keep].reshape(-1, 3), target[keep].reshape(-1, 3), scaled)

    def residuals(scale, R, t):
        """Per image: RMS over its four pose points, and the centre alone."""
        error = np.linalg.norm(scale * source @ R.T + t - target, axis=2)
        return np.sqrt(np.mean(error ** 2, axis=1)), error[:, 0]

    keep = np.ones(len(source), bool)
    for _ in range(3):
        pose_error, _ = residuals(*fit(keep, with_scale))
        keep = pose_error < max(3 * np.median(pose_error), 1.0)
    scale, R, t = fit(keep, with_scale)
    pose_error, centre_error = residuals(scale, R, t)
    best_scale = fit(keep, True)[0]

    print(f"  SfM registered {len(sfm)} images; {len(source)} also have a tag pose, {keep.sum()} used")
    print(f"  best-fit scale SfM -> tags {best_scale:.5f} (1.0 = rig baseline and tags agree); "
          f"{'applied' if with_scale else 'NOT applied, fit is rigid'}")
    print(f"  pose residual (centre + axes at {ALIGN_LEVER_MM:.0f} mm): median {np.median(pose_error[keep]):.2f} mm, "
          f"p90 {np.percentile(pose_error[keep], 90):.2f} mm; centre alone median {np.median(centre_error[keep]):.2f} mm")

    S = np.eye(4)
    S[:3, :3], S[:3, 3] = scale * R, t
    heights, tilt = tag_plane_check(ws, S, tags_corners)
    print("  SfM points on the tags, height above board: "
          + ", ".join(f"{tag}: {h:+.2f}" for tag, h in heights.items()) + f" mm; plane tilt {tilt:.2f} deg")
    S = level_on_tags(ws, S, tags_corners) @ S
    heights, tilt = tag_plane_check(ws, S, tags_corners)
    print("  after levelling on the SfM tag points: "
          + ", ".join(f"{tag}: {h:+.2f}" for tag, h in heights.items()) + f" mm; plane tilt {tilt:.2f} deg")
    fs = cv2.FileStorage(os.path.join(ws, "board_from_sfm.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.writeComment("Similarity transform taking COLMAP world coordinates into the board frame (mm):")
    fs.writeComment("  X_board = board_from_sfm @ [X_sfm, 1]   (the 3x3 block includes the scale)")
    fs.write("created", datetime.datetime.now().isoformat(timespec="seconds"))
    fs.write("board_from_sfm", S)
    fs.write("scale_applied", float(scale))
    fs.writeComment("Scale a similarity fit would have used; tags / SfM, so above 1 = tag frame larger")
    fs.write("best_fit_scale", float(best_scale))
    fs.write("images_registered", len(sfm))
    fs.write("images_with_tag_pose", len(source))
    fs.write("images_used_for_fit", int(keep.sum()))
    fs.write("pose_residual_median_mm", float(np.median(pose_error[keep])))
    fs.write("pose_residual_p90_mm", float(np.percentile(pose_error[keep], 90)))
    fs.write("centre_residual_median_mm", float(np.median(centre_error[keep])))
    fs.writeComment("Check: SfM points on each tag should sit at z ~ 0 and their plane should not tilt")
    fs.write("tag_plane_tilt_deg", tilt)
    fs.startWriteStruct("tag_point_height_mm", cv2.FileNode_MAP)
    for tag, height in heights.items():
        fs.write(f"tag_{tag}", height)
    fs.endWriteStruct()
    fs.release()
    export_sparse(ws, S, sparse_path)


def export_sparse(ws, S, out_path):
    """The SfM points in the board frame, as sparse.ply next to the dense cloud:
    something to look at, and to check the alignment on, long before PatchMatch finishes."""
    import open3d as o3d
    sfm_ply = os.path.join(ws, "sparse_points_sfm.ply")
    colmap(ws, "model_converter", {"input_path": best_model(ws), "output_path": sfm_ply, "output_type": "PLY"})
    cloud = o3d.io.read_point_cloud(sfm_ply)
    cloud.transform(S)
    o3d.io.write_point_cloud(out_path, cloud)
    os.remove(sfm_ply)
    print(f"  {len(cloud.points)} sparse points -> {rel(out_path)}")


def stage_undistort(ws, max_image_size, source_images, names, dense_step):
    """Undistort the frames that will get depth maps. Only every dense_step'th frame goes into the
    dense workspace, so PatchMatch also picks its source views among those frames and never needs
    a depth map that was not computed."""
    dense = os.path.join(ws, "dense")
    shutil.rmtree(dense, ignore_errors=True)
    image_list = os.path.join(ws, "dense_images.txt")
    with open(image_list, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{camera}/{n}\n" for n in names[::dense_step] for camera in CAMERAS))
    print(f"  {len(names[::dense_step])} of {len(names)} frames get depth maps (--dense-step {dense_step})")
    colmap(ws, "image_undistorter", {
        "image_path": os.path.join(ws, "images"),
        "input_path": best_model(ws),
        "output_path": dense,
        "output_type": "COLMAP",
        "max_image_size": max_image_size,
        "image_list_path": image_list,
    }, optional={"num_patch_match_src_images": source_images})


# Source-view selection for --sources rig. COLMAP's own choice ranks views by shared sparse points,
# which on a trolley scan are the same camera's nearest frames: for left/0100 of Sep24/mjpg_pyr2 all
# 10 were left frames at 1.2-3.5 deg, and the stereo partner at 18.5 deg was not among them. Depth
# error scales with 1/angle, so the partner alone is ~10x more precise than those.
SOURCE_TARGET_ANGLE = 15.0       # deg; triangulation angle preferred for a source view
SOURCE_ANGLE_RANGE = (4.0, 40.0)  # deg; outside this a view adds little or matches poorly
SOURCE_MAX_OFF_AXIS = 35.0       # deg; the reference's target point must be this close to the view's axis
TARGET_DEPTH_MM = 450.0          # the board's distance, where the angles are measured


def write_source_lists(ws, n_sources):
    """Overwrite dense/stereo/patch-match.cfg with explicit source views for every image: its stereo
    partner first, then the views whose triangulation angle at a point TARGET_DEPTH_MM in front of the
    reference is closest to SOURCE_TARGET_ANGLE, among those that see that point near their centre."""
    poses = read_model_images(ws, os.path.join(ws, "dense", "sparse"))
    names = sorted(poses)
    centre = {n: invert(T)[:3, 3] for n, T in poses.items()}
    axis = {n: invert(T)[:3, 2] for n, T in poses.items()}
    lines, chosen_angles = [], []
    for ref in names:
        target = centre[ref] + TARGET_DEPTH_MM * axis[ref]
        partner = ("right/" if ref.startswith("left/") else "left/") + ref.split("/", 1)[1]
        scored = []
        for other in names:
            if other == ref:
                continue
            to_ref, to_other = centre[ref] - target, centre[other] - target
            angle = np.degrees(np.arccos(np.clip(to_ref @ to_other / np.linalg.norm(to_ref) / np.linalg.norm(to_other), -1, 1)))
            off_axis = np.degrees(np.arccos(np.clip(-to_other @ axis[other] / np.linalg.norm(to_other), -1, 1)))
            if SOURCE_ANGLE_RANGE[0] <= angle <= SOURCE_ANGLE_RANGE[1] and off_axis <= SOURCE_MAX_OFF_AXIS:
                scored.append((other != partner, abs(angle - SOURCE_TARGET_ANGLE), other, angle))
        picked = sorted(scored)[:n_sources]
        if not picked:
            continue
        chosen_angles += [a for *_, a in picked]
        lines.append(f"{ref}\n{', '.join(o for _, _, o, _ in picked)}\n")
    with open(os.path.join(ws, "dense", "stereo", "patch-match.cfg"), "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"  explicit source views for {len(lines)} of {len(names)} images; triangulation angles "
          f"median {np.median(chosen_angles):.1f} deg, range {min(chosen_angles):.1f}-{max(chosen_angles):.1f}")


def stage_patchmatch(ws, args):
    colmap(ws, "patch_match_stereo", {
        "workspace_path": os.path.join(ws, "dense"),
        "PatchMatchStereo.geom_consistency": 1,
        "PatchMatchStereo.gpu_index": args.gpu,
        "PatchMatchStereo.max_image_size": args.max_image_size,
        "PatchMatchStereo.window_radius": args.window_radius,
        "PatchMatchStereo.filter_min_triangulation_angle": args.min_triangulation_angle,
    }, optional={"PatchMatchStereo.cache_size": args.cache_gb})


def stage_fuse(ws, args):
    colmap(ws, "stereo_fusion", {
        "workspace_path": os.path.join(ws, "dense"),
        "input_type": "geometric",
        "output_path": os.path.join(ws, "dense", "fused.ply"),
        "StereoFusion.min_num_pixels": args.min_fused_pixels,
    }, optional={"StereoFusion.cache_size": args.cache_gb})


def stage_export(ws, out_path):
    """Fused cloud -> board frame, as a PLY next to the SGBM one."""
    import open3d as o3d
    fs = cv2.FileStorage(os.path.join(ws, "board_from_sfm.yaml"), cv2.FILE_STORAGE_READ)
    S = fs.getNode("board_from_sfm").mat()
    fs.release()
    cloud = o3d.io.read_point_cloud(os.path.join(ws, "dense", "fused.ply"))
    cloud.transform(S)
    if cloud.has_normals():
        # transform() applied the scale to the normals' rotation part too; renormalise them.
        cloud.normalize_normals()
    o3d.io.write_point_cloud(out_path, cloud)
    print(f"  {len(cloud.points)} points -> {rel(out_path)}")


def main():
    args = parse_args()
    if shutil.which(COLMAP) is None:
        raise SystemExit(f"'{COLMAP}' not found on PATH (set COLMAP=/path/to/colmap)")
    poses_file = paths.poses_path(args.session, args.scan)
    frames_dir, calibration, tag_poses, tags_corners = load_poses(poses_file)
    rig = load_rig(calibration)
    ws = args.workspace or paths.work_dir(args.session, args.scan, args.method)
    os.makedirs(ws, exist_ok=True)
    out_dir = paths.method_dir(args.session, args.scan, args.method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, paths.CLOUD)
    sparse_path = os.path.join(out_dir, paths.SPARSE)
    print(f"workspace {rel(ws)}; {len(tag_poses)} tag-posed frames from {rel(poses_file)}")

    run = STAGES[STAGES.index(args.from_stage):STAGES.index(args.to_stage) + 1]
    names = [n for n, _, _ in find_frame_pairs(frames_dir) if args.all_frames or n in tag_poses]
    print(f"{len(names)} frame pairs used ({'all frames' if args.all_frames else 'tag-posed frames only'})")
    has_rig = supports("rig_configurator", "rig_config_path")
    for stage in run:
        print(f"[{stage}]")
        if stage == "workspace":
            stage_workspace(ws, frames_dir, names)
        elif stage == "features":
            stage_features(ws, rig, names, args.gpu)
        elif stage == "rig":
            has_rig = stage_rig(ws, rig)
        elif stage == "match":
            stage_match(ws, names, args.window, args.gpu)
        elif stage == "map":
            stage_map(ws, has_rig)
        elif stage == "align":
            stage_align(ws, tag_poses, tags_corners, rig, args.align_scale, sparse_path)
        elif stage == "undistort":
            stage_undistort(ws, args.max_image_size, args.source_images, names, args.dense_step)
            if args.sources == "rig":
                write_source_lists(ws, args.source_images)
        elif stage == "patchmatch":
            stage_patchmatch(ws, args)
        elif stage == "fuse":
            stage_fuse(ws, args)
        elif stage == "export":
            stage_export(ws, out_path)


if __name__ == "__main__":
    main()
