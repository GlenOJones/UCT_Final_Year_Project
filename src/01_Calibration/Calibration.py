"""Stereo camera calibration from Kalibr AprilGrid (36h11) frames."""
import argparse
import contextlib
import datetime
import glob
import io
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_config import parameters, validate  # noqa: E402

# Paths are relative to the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====== BOARD GEOMETRY ======
# Board: kalibr_create_target_pdf --type apriltag --nx 10 --ny 7 --tsize 0.050 --tspace 0.3
# TAG_SIZE is measured on the print, not taken from the PDF.
TAG_SIZE =  35.355339           # mm, edge of the outer black square as printed
TAG_GAP = TAG_SIZE * 0.3    # mm, white gap between tags
PITCH = TAG_SIZE + TAG_GAP  # mm, centre to centre
N_TAGS = 70                 # 10 across x 7 up, ids 0..69

CAMERAS = ("left", "right")

# Corners are refined with cv2.cornerSubPix (detector refinement off), window 15% of the tag side.
SUBPIX_WIN_FRACTION = 0.15   # half-size of the search window, as a fraction of the tag side
SUBPIX_WIN_MIN = 2           # px, floor for very small tags
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)


# ====== IMAGE CHANNEL ======
# Channel the detector works on: gray, y, r, g, b, or merge (detect in each, keep every tag found).
CHANNELS = ("gray", "y", "r", "g", "b", "merge")
# A tag found in several channels takes its corners from the first of these.
MERGE_ORDER = ("gray", "g", "r", "b")
# px. A tag whose corners disagree by more than this between channels is dropped.
MERGE_TOLERANCE_PX = 1.0


# ====== DETECTOR PRESETS ======
# Tag family printed on the board: AprilTag 36h11.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_presets.json")


@dataclass
class Presets:
    """The validated contents of detector_presets.json."""
    path: str
    default: str
    board_params: dict
    presets: dict

    def settings(self, preset):
        """Detector parameters for one preset: board_params overlaid with the preset's own tuning."""
        if preset not in self.presets:
            raise SystemExit(f"unknown preset {preset!r}; {self.path} defines: {', '.join(self.presets)}")
        return {**self.board_params, **self.presets[preset]}

    def describe(self, preset):
        """One-line summary of a preset's settings, for logs and for the YAML metadata."""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.settings(preset).items()))

    def make_detector(self, preset):
        """ArucoDetector for this board, configured from one preset."""
        params = cv2.aruco.DetectorParameters()
        # Corners are refined later by refine_corners(), so the detector's own refinement stays off.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        for key, value in self.settings(preset).items():
            setattr(params, key, value)
        return cv2.aruco.ArucoDetector(dictionary, params)


def load_presets():
    """Read detector_presets.json, validate it, and drop the "_why"/"_comment" documentation keys."""
    with open(PRESETS_FILE, encoding="utf-8") as fh:
        config = validate(json.load(fh), PRESETS_FILE)
    return Presets(path=PRESETS_FILE,
                   default=config["default"],
                   board_params=parameters(config["board_params"]),
                   presets={name: parameters(body) for name, body in config["presets"].items()})


# ====== DETECTOR REGISTRY ======
# OpenCV's ArucoDetector, or UWARUco (Cejka et al. 2019).
UWARUCO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive", "UWARUco")


def uwaruco_module():
    """Import src/archive/UWARUco/uwaruco.py."""
    sys.path.insert(0, UWARUCO_DIR)
    import uwaruco
    return uwaruco


def aruco_backend():
    presets = load_presets()
    return (presets.default, presets.describe, presets.make_detector)


def uwaruco_backend():
    presets = uwaruco_module().load_presets()
    return (presets.default,
            presets.describe,
            lambda preset: presets.make_detector(dictionary, preset))


DETECTORS = {"aruco": aruco_backend, "uwaruco": uwaruco_backend}


# ====== RUN SETTINGS ======
def run_name(frames_dir, detector_name, preset, channel="gray"):
    """Runs are named by recording, detector, preset and channel, so a sweep over any of them does not overwrite itself
    and compare_calibrations.py can pick the whole set up with a glob."""
    recording = os.path.basename(os.path.normpath(frames_dir)).replace(" ", "_")
    # The default detector is unlabelled so older filenames stay the same.
    label = preset if detector_name == "aruco" else f"{detector_name}-{preset}"
    # Likewise the default channel, so gray runs keep the names they had before --channel.
    return f"{recording}_{label}" + ("" if channel == "gray" else f"_{channel}")


@dataclass
class Run:
    """Everything one calibration run needs, resolved once so no function reads a global."""
    frames_dir: str
    detector_name: str        # "aruco" or "uwaruco": which detector implementation ran
    preset: str
    out_dir: str
    min_tags: int
    min_common_tags: int
    save_detections: bool
    image_size: tuple         # (width, height)
    detector_settings: str    # merged settings of the preset, recorded in the YAML
    detector: cv2.aruco.ArucoDetector
    board: cv2.aruco.Board
    channel: str = "gray"     # which image the detector sees, one of CHANNELS
    reject: bool = True       # drop misplaced tags and badly fitting images before the final fit
    arguments: str = ""       # the command-line arguments of the run, recorded in the YAML

    @property
    def name(self):
        return run_name(self.frames_dir, self.detector_name, self.preset, self.channel)

    @property
    def yaml_path(self):
        return os.path.join(self.out_dir, self.name + "_stereo.yaml")

    @property
    def detections_dir(self):
        return os.path.join(self.out_dir, self.name + "_detections")

    @property
    def corners_path(self):
        return os.path.join(self.out_dir, self.name + "_corners.json")


def rel(path):
    """Project-relative form of a path, so what is recorded in the YAML stays portable."""
    return os.path.relpath(path, PROJECT_ROOT)


def parse_args(presets, argv=None):
    ap = argparse.ArgumentParser(description="Calibrate the stereo cameras from AprilGrid frames.")
    ap.add_argument("--frames-dir", default=os.path.join(PROJECT_ROOT, "data/Calibration/18_Sep/Rigframes"),
                    help="one folder per recording, each with left/ and right/ (default 18_Sep/Rigframes)")
    ap.add_argument("--detector", default="aruco", choices=list(DETECTORS),
                    help="which detector implementation to use: OpenCV's ArUco, or the UWARUco "
                         "re-implementation in src/archive/UWARUco (default aruco)")
    # Valid presets depend on --detector, so no choices= here.
    ap.add_argument("--preset", default=None,
                    help=f"detector preset; from detector_presets.json for --detector aruco "
                         f"(default {presets.default}), or from src/archive/UWARUco/uwaruco_presets.json "
                         f"for --detector uwaruco")
    ap.add_argument("--out-dir", default=None,
                    help="where the YAML and detection images go (default results/<session>/calibration, "
                         "the session being the folder --frames-dir is in, e.g. 18_Sep)")
    ap.add_argument("--channel", default="merge", choices=CHANNELS,
                    help="image the detector sees: gray (computed from colour), y (the camera's JPEG "
                         "luma, from <recording>/gray/), one colour channel r, g, b, or merge (detect in "
                         "gray, g, r and b and combine the tags) (default merge)")
    ap.add_argument("--min-tags", type=int, default=10,
                    help="tags an image needs to be used (default 10)")
    ap.add_argument("--min-common-tags", type=int, default=10,
                    help="tags both cameras must see in a pair for it to be used for extrinsics "
                         "(default 10)")
    ap.add_argument("--no-reject", action="store_true",
                    help="keep every detected tag and image: skip the outlier rejection (see REJECT_*)")
    ap.add_argument("--no-detection-images", action="store_true",
                    help="skip writing a copy of every frame with its detections drawn on")
    return ap.parse_args(argv)


def find_images(frames_dir, camera):
    """Every PNG of one camera across all recording folders under frames_dir, sorted."""
    return sorted(glob.glob(os.path.join(frames_dir, "*", camera, "*.png")))


def build_board():
    """Physical position (mm) of every tag corner, as the object-point model for calibration."""
    tag_corners = []
    for tag_id in range(N_TAGS):
        x, y = (tag_id % 10) * PITCH, (tag_id // 10) * PITCH   # id 0 bottom-left, ids run right then up
        # Kalibr prints tags rotated 180 deg: corners come as BR, BL, TL, TR. Object points use the same order.
        tag_corners.append(np.array([[x + TAG_SIZE, y, 0], [x, y, 0],
                                     [x, y + TAG_SIZE, 0], [x + TAG_SIZE, y + TAG_SIZE, 0]], np.float32))
    return cv2.aruco.Board(tag_corners, dictionary, np.arange(N_TAGS))


def setup(args, arguments=""):
    """Validate the inputs and resolve everything the run needs."""
    default_preset, describe, make_detector = DETECTORS[args.detector]()
    preset = args.preset or default_preset

    for camera in CAMERAS:
        if not find_images(args.frames_dir, camera):
            raise SystemExit(f"no {camera} PNGs under {args.frames_dir}/*/{camera}/ - "
                             f"check --frames-dir, or extract frames first with "
                             f"preproccessing/extract_stereo_frames.py")

    first_image = find_images(args.frames_dir, "left")[0]
    image_size = cv2.imread(first_image, cv2.IMREAD_COLOR).shape[1::-1]   # (width, height)
    print("image size:", image_size)
    print(f"detector: {args.detector}  preset: {preset}  ({describe(preset)})")

    # Outputs are filed by session (see src/project_paths.py): the folder the recording sits in.
    session = os.path.basename(os.path.dirname(os.path.normpath(args.frames_dir)))
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results", session, "calibration")
    return Run(frames_dir=args.frames_dir, detector_name=args.detector, preset=preset,
               out_dir=out_dir,
               min_tags=args.min_tags, min_common_tags=args.min_common_tags,
               save_detections=not args.no_detection_images,
               image_size=image_size, detector_settings=describe(preset),
               detector=make_detector(preset), board=build_board(), channel=args.channel,
               reject=not args.no_reject, arguments=arguments)


# ====== DETECTION ======
def load_image(path, channel="gray"):
    """Read a frame as (BGR image for drawing on, single-channel image for detection); see CHANNELS."""
    if channel == "y":
        folder, name = os.path.split(path)
        luma_path = os.path.join(os.path.dirname(folder), "gray", os.path.basename(folder), name)
        luma = cv2.imread(luma_path, cv2.IMREAD_GRAYSCALE)
        if luma is None:
            raise SystemExit(f"{luma_path}: no luma copy for --channel y; extract one with "
                             f"preproccessing/extract_stereo_frames.py --gray")
        return cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR), luma
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"{path}: could not read image")
    return image, channel_image(image, "gray" if channel == "merge" else channel)


def channel_image(image, channel):
    """One single-channel image made from a BGR frame: gray, r, g or b (see CHANNELS)."""
    if channel == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(image[..., "bgr".index(channel)])


def detect(run, image, refine_on=None):
    """(corners, ids, rejected) in one single-channel image, corners refined on refine_on (default: the same image)."""
    corners, ids, rejected = run.detector.detectMarkers(image)
    return refine_corners(image if refine_on is None else refine_on, corners), ids, rejected


def merge_detections(results):
    """Combine detections of the same frame from several channels, given in MERGE_ORDER."""
    found = {}                               # id -> [4x2 corners per channel that found it once]
    for corners, ids, _ in results:
        if ids is None:
            continue
        flat = [int(i) for i in ids.ravel()]
        for i, c in zip(flat, corners):
            if flat.count(i) == 1:
                found.setdefault(i, []).append(c.reshape(4, 2))
    keep, conflicts = {}, 0
    for i, seen in found.items():
        if 0 <= i < N_TAGS and any(np.max(np.linalg.norm(other - seen[0], axis=1)) > MERGE_TOLERANCE_PX
                                   for other in seen[1:]):
            conflicts += 1
            continue
        keep[i] = seen[0]
    if not keep:
        return (), None, conflicts
    ids = sorted(keep)
    return (tuple(keep[i].reshape(1, 4, 2).astype(np.float32) for i in ids),
            np.array(ids, np.int32).reshape(-1, 1), conflicts)


def subpix_window(tag_corners):
    """cornerSubPix half-window for one tag: SUBPIX_WIN_FRACTION of its mean side length in the image."""
    quad = tag_corners.reshape(4, 2)
    side = np.mean(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1))
    half = max(SUBPIX_WIN_MIN, int(round(SUBPIX_WIN_FRACTION * side)))
    return half, half


def refine_corners(gray, corners):
    """Sub-pixel refine every tag corner on the original image, with a window sized to its tag."""
    return tuple(cv2.cornerSubPix(gray, c.reshape(4, 1, 2).copy(), subpix_window(c), (-1, -1), SUBPIX_CRITERIA)
                 .reshape(1, 4, 2) for c in corners)


def save_detection_image(run, camera, name, image, corners, ids, rejected):
    """Save a copy of the image with detected tags (green, with id) and rejected candidates (red)."""
    vis = image.copy()
    if rejected:
        cv2.aruco.drawDetectedMarkers(vis, rejected, borderColor=(0, 0, 255))
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids, borderColor=(0, 255, 0))
    n_tags = 0 if ids is None else len(ids)
    missing = sorted(set(range(N_TAGS)) - set([] if ids is None else ids.ravel().tolist()))
    colour = (0, 255, 0) if n_tags >= run.min_tags else (0, 0, 255)
    lines = [f"{camera} {name}: {n_tags}/{N_TAGS} tags" + ("" if n_tags >= run.min_tags else "  SKIPPED"),
             f"{len(rejected)} rejected candidates (red)"]
    if 0 < len(missing) <= 20:
        lines.append("missing ids: " + " ".join(map(str, missing)))
    for k, text in enumerate(lines):
        cv2.putText(vis, text, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    out_path = os.path.join(run.detections_dir, camera, name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def detect_tags(run, camera, found=None):
    """Detect tags in every image of one camera."""
    image_files = find_images(run.frames_dir, camera)
    detections = {}
    for path in image_files:
        image, gray = load_image(path, run.channel)
        if gray.shape[::-1] != run.image_size:
            raise SystemExit(f"{path}: size {gray.shape[::-1]} differs from {run.image_size}")
        name = os.path.relpath(path, run.frames_dir).replace(os.sep + camera + os.sep, "/")   # "140/C140_Left_90.png"
        conflicts = None
        if run.channel == "merge":
            # Channels only find tags; every corner is refined on gray.
            results = [detect(run, channel_image(image, c), refine_on=gray) for c in MERGE_ORDER]
            corners, ids, conflicts = merge_detections(results)
            rejected = results[0][2]         # gray's, for the detection image
        else:
            corners, ids, rejected = detect(run, gray)
        if run.save_detections:
            save_detection_image(run, camera, name, image, corners, ids, rejected)

        n_tags = 0 if ids is None else len(ids)
        if found is not None:
            found[name] = {"ids": [] if ids is None else [int(i) for i in ids.ravel()],
                           "corners": [c.reshape(4, 2).tolist() for c in corners]}
            if conflicts is not None:
                found[name]["merge_conflicts"] = conflicts
        if n_tags < run.min_tags:
            print(f"  {camera}: skip {name}: {n_tags} tags")
            continue

        detections[name] = (corners, ids)

    print(f"{camera}: {len(detections)} of {len(image_files)} images usable")
    return detections


# ====== CAMERA CALIBRATION ======
def calibrate_camera(run, camera, detections):
    """Calibrate one camera from its detections."""
    names = sorted(detections)
    all_obj_points, all_img_points = [], []
    for name in names:
        corners, ids = detections[name]
        obj_points, img_points = run.board.matchImagePoints(corners, ids)
        all_obj_points.append(obj_points)
        all_img_points.append(img_points)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs, std_intrinsics, std_extrinsics, per_view_errors = \
        cv2.calibrateCameraExtended(all_obj_points, all_img_points, run.image_size, None, None)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"\n{camera.upper()} CAMERA")
    print(f"RMS reprojection error: {rms:.3f} px")
    print(f"focal length   fx = {fx:.1f} px   fy = {fy:.1f} px")
    print(f"principal point cx = {cx:.1f} px   cy = {cy:.1f} px   "
          f"(image centre {run.image_size[0] / 2:.0f}, {run.image_size[1] / 2:.0f})")
    print("distortion (k1, k2, p1, p2, k3):", ", ".join(f"{v:.4f}" for v in dist_coeffs.ravel()))

    # Error per image, worst first
    per_view_errors = per_view_errors.ravel()
    print("error per image:")
    for name, err in sorted(zip(names, per_view_errors), key=lambda t: -t[1]):
        flag = "  <-- high" if err > 2 * np.median(per_view_errors) else ""
        print(f"  {name}: {err:.3f} px{flag}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rms": rms,
        "std_intrinsics": std_intrinsics.ravel()[:9],   # 1-sigma of fx, fy, cx, cy, k1, k2, p1, p2, k3
        # 1-sigma of each image's board pose: rvec (rad, 3) then tvec (mm, 3), one row per image
        "std_extrinsics": std_extrinsics.reshape(-1, 6),
        "images": names,
        "per_image_errors": per_view_errors,
    }


# ====== OUTLIER REJECTION ======
# Each round refits, re-poses every image with RANSAC and keeps the tags and images that agree.
REJECT_TAG_PX = 2.0          # a tag is dropped when its worst corner is further than
REJECT_TAG_MEDIANS = 5.0     #   max(REJECT_TAG_PX, REJECT_TAG_MEDIANS x the median over kept tags)
REJECT_IMAGE_PX = 1.0        # an image is dropped when its kept tags' corner RMS is above
REJECT_IMAGE_MEDIANS = 3.0   #   max(REJECT_IMAGE_PX, REJECT_IMAGE_MEDIANS x the median over kept images)
REJECT_MAX_ROUNDS = 10


def tag_errors(run, corners, ids, K, D):
    """{tag id: (worst corner px, mean squared corner px)} for the board tags of one image, against the board posed
    robustly on them with intrinsics K, D."""
    flat = [int(i) for i in ids.ravel()]
    tags = {i: c.reshape(4, 2) for i, c in zip(flat, corners) if 0 <= i < N_TAGS and flat.count(i) == 1}
    if len(tags) < 4:
        return {}
    obj_all = run.board.getObjPoints()
    obj = np.concatenate([obj_all[i] for i in tags]).astype(np.float64)
    img = np.concatenate(list(tags.values())).astype(np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE,
                                                 reprojectionError=3 * REJECT_TAG_PX, iterationsCount=500)
    if not ok or inliers is None or len(inliers) < 8:
        return {i: (np.inf, np.inf) for i in tags}
    rvec, tvec = cv2.solvePnPRefineLM(obj[inliers.ravel()], img[inliers.ravel()], K, D, rvec, tvec)
    d = np.linalg.norm(cv2.projectPoints(obj, rvec, tvec, K, D)[0].reshape(-1, 2) - img, axis=1).reshape(-1, 4)
    return {i: (float(e.max()), float(np.mean(e ** 2))) for i, e in zip(tags, d)}


def keep_only(detections, keep):
    """Detections reduced to the kept tag ids of the kept images."""
    out = {}
    for name, ids_kept in keep.items():
        corners, ids = detections[name]
        sel = [k for k, i in enumerate(ids.ravel()) if int(i) in ids_kept]
        out[name] = (tuple(corners[k] for k in sel), ids[sel].reshape(-1, 1))
    return out


def reject_outliers(run, camera, detections):
    """Iteratively drop misplaced tags and badly fitting images (see REJECT_*)."""
    keep = {n: {int(i) for i in ids.ravel() if 0 <= i < N_TAGS} for n, (_, ids) in detections.items()}
    keep = {n: k for n, k in keep.items() if len(k) >= run.min_tags}
    for rounds in range(1, REJECT_MAX_ROUNDS + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            fit = calibrate_camera(run, camera, keep_only(detections, keep))
        K, D = fit["camera_matrix"], fit["dist_coeffs"]
        errors = {n: tag_errors(run, *detections[n], K, D) for n in detections}
        kept_worst = [errors[n][i][0] for n in keep for i in keep[n] if i in errors[n]]
        tag_limit = max(REJECT_TAG_PX, REJECT_TAG_MEDIANS * float(np.median(kept_worst)))
        new = {n: {i for i, (worst, _) in e.items() if worst <= tag_limit} for n, e in errors.items()}
        rms = {n: float(np.sqrt(np.mean([errors[n][i][1] for i in k]))) for n, k in new.items() if len(k) >= run.min_tags}
        image_limit = max(REJECT_IMAGE_PX, REJECT_IMAGE_MEDIANS * float(np.median(list(rms.values()))))
        new = {n: new[n] for n, r in rms.items() if r <= image_limit}
        if new == keep:
            break
        keep = new
    rejected_tags = sorted((n, i, errors[n][i][0]) for n in errors for i in errors[n]
                           if n in keep and i not in keep[n])
    rejected_images = sorted((n, rms.get(n, float("nan"))) for n in detections if n not in keep)
    print(f"{camera}: outlier rejection kept {len(keep)} of {len(detections)} images, dropped "
          f"{len(rejected_tags)} tags in kept images ({rounds} rounds; limits {tag_limit:.2f} px per tag, "
          f"{image_limit:.2f} px per image)")
    report = {"rounds": rounds, "tag_limit_px": tag_limit, "image_limit_px": image_limit,
              "rejected_tags": rejected_tags, "rejected_images": rejected_images,
              "kept_tags": sum(len(k) for k in keep.values())}
    return keep_only(detections, keep), report


# ====== STEREO EXTRINSICS ======
def pair_up(detections):
    """Match each left detection to its right one, as (left_name, right_name) pairs."""
    right = detections["right"]
    return [(name, name.replace("_Left_", "_Right_")) for name in sorted(detections["left"])
            if name.replace("_Left_", "_Right_") in right]


def common_tag_points(run, left, right):
    """Object and image points for the tags BOTH cameras saw in one pair, or None if too few."""
    left_corners = {int(i): c for i, c in zip(left[1].ravel(), left[0])}
    right_corners = {int(i): c for i, c in zip(right[1].ravel(), right[0])}
    shared = sorted(left_corners.keys() & right_corners.keys())
    if len(shared) < run.min_common_tags:
        return None

    ids = np.array(shared, np.int32).reshape(-1, 1)
    obj_points, left_points = run.board.matchImagePoints(tuple(left_corners[i] for i in shared), ids)
    _, right_points = run.board.matchImagePoints(tuple(right_corners[i] for i in shared), ids)
    return obj_points, left_points, right_points


def calibrate_stereo(run, detections, intrinsics):
    """Fit the rotation and translation from the left camera to the right."""
    pairs = pair_up(detections)
    obj_points, left_points, right_points, used = [], [], [], []
    for left_name, right_name in pairs:
        points = common_tag_points(run, detections["left"][left_name], detections["right"][right_name])
        if points is None:
            continue
        obj_points.append(points[0])
        left_points.append(points[1])
        right_points.append(points[2])
        used.append(left_name)

    print("\nSTEREO")
    if len(used) < 2:
        print(f"only {len(used)} usable pairs (need >= 2): extrinsics not computed. Both cameras "
              f"must see at least {run.min_common_tags} of the SAME tags in a pair.")
        return None

    left_cam, right_cam = intrinsics["left"], intrinsics["right"]
    result = cv2.stereoCalibrate(
        obj_points, left_points, right_points,
        left_cam["camera_matrix"], left_cam["dist_coeffs"],
        right_cam["camera_matrix"], right_cam["dist_coeffs"],
        run.image_size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    rms, R, T = result[0], result[5], result[6]

    baseline = float(np.linalg.norm(T))
    angles = np.degrees(cv2.Rodrigues(R)[0].ravel())
    print(f"{len(used)} of {len(pairs)} pairs usable")
    print(f"RMS reprojection error: {rms:.3f} px")
    print(f"baseline |T| = {baseline:.2f} mm   T = "
          + ", ".join(f"{v:.2f}" for v in T.ravel()) + " mm")
    print("rotation (deg about x, y, z): " + ", ".join(f"{a:.3f}" for a in angles))
    print("CHECK the baseline against a ruler: it carries the same scale error as TAG_SIZE.")

    # Rectification, so 03_Reconstruction does not recompute it.
    R1, R2, P1, P2, Q = cv2.stereoRectify(
        left_cam["camera_matrix"], left_cam["dist_coeffs"],
        right_cam["camera_matrix"], right_cam["dist_coeffs"],
        run.image_size, R, T, alpha=0)[:5]

    return {"R": R, "T": T, "E": result[7], "F": result[8], "rms": rms, "baseline_mm": baseline,
            "angles_deg": angles, "pairs": used, "n_pairs_total": len(pairs),
            "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q}


# ====== SAVE CALIBRATION (YAML) ======
def write_camera(fs, camera, result, image_size):
    """Write one camera's intrinsics, uncertainty and quality as a commented YAML section."""
    K, D = result["camera_matrix"], result["dist_coeffs"].ravel()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    std = result["std_intrinsics"]
    names = ["k1", "k2", "p1", "p2", "k3"]

    fs.startWriteStruct(f"{camera}_camera", cv2.FileNode_MAP)

    fs.writeComment("---- Intrinsics: pinhole model + lens distortion ----")
    fs.writeComment("camera_matrix K = [ fx   0  cx ]")
    fs.writeComment("                  [  0  fy  cy ]")
    fs.writeComment("                  [  0   0   1 ]")
    fs.writeComment(f"fx = {fx:.2f} px, fy = {fy:.2f} px (focal length); "
                    f"cx = {cx:.2f} px, cy = {cy:.2f} px (principal point)")
    fs.write("camera_matrix", K)

    fs.writeComment("distortion_coefficients = [k1, k2, p1, p2, k3] (OpenCV standard model)")
    fs.writeComment("k1, k2, k3: radial distortion (negative k1 = barrel); p1, p2: tangential (lens tilt)")
    fs.writeComment(", ".join(f"{n} = {v:.5f}" for n, v in zip(names, D)))
    fs.write("distortion_coefficients", result["dist_coeffs"])

    fs.writeComment("Readable copies of the values above")
    fs.write("focal_length_x_px", fx)
    fs.write("focal_length_y_px", fy)
    fs.write("principal_point_x_px", cx)
    fs.write("principal_point_y_px", cy)
    fs.write("field_of_view_horizontal_deg", np.degrees(2 * np.arctan(image_size[0] / (2 * fx))))
    fs.write("field_of_view_vertical_deg", np.degrees(2 * np.arctan(image_size[1] / (2 * fy))))

    fs.writeComment("---- Uncertainty: 1 standard deviation from the calibration fit ----")
    fs.writeComment(f"fx +- {std[0]:.2f} px, fy +- {std[1]:.2f} px, cx +- {std[2]:.2f} px, cy +- {std[3]:.2f} px")
    fs.startWriteStruct("uncertainty_1sigma", cv2.FileNode_MAP)
    for key, value in zip(["fx_px", "fy_px", "cx_px", "cy_px"] + names, std):
        fs.write(key, value)
    fs.endWriteStruct()

    # 1-sigma of every image's board pose.
    ext = result["std_extrinsics"]
    rot_deg = np.degrees(np.linalg.norm(ext[:, :3], axis=1))
    trans_mm = np.linalg.norm(ext[:, 3:], axis=1)
    fs.writeComment("Board pose (extrinsics) 1-sigma per image, from calibrateCameraExtended: |sigma rvec| in deg and")
    fs.writeComment(f"|sigma tvec| in mm, listed in the order of images. Median {np.median(rot_deg):.3f} deg, "
                    f"{np.median(trans_mm):.2f} mm; worst {rot_deg.max():.3f} deg, {trans_mm.max():.2f} mm")
    fs.startWriteStruct("pose_uncertainty_1sigma", cv2.FileNode_MAP)
    for key, values in (("rotation_deg", rot_deg), ("translation_mm", trans_mm)):
        fs.startWriteStruct(key, cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for v in values:
            fs.write("", float(v))
        fs.endWriteStruct()
    fs.endWriteStruct()

    fs.writeComment("---- Quality ----")
    fs.writeComment("RMS distance between detected corners and where the model puts them; below ~0.5 px is good")
    fs.writeComment(f"rms = {result['rms']:.3f} px over {len(result['images'])} images")
    fs.write("rms_reprojection_error_px", result["rms"])
    fs.write("images_used", len(result["images"]))
    fs.writeComment("Per-image RMS error (px), worst first:")
    for name, err in sorted(zip(result["images"], result["per_image_errors"]), key=lambda t: -t[1]):
        fs.writeComment(f"  {name}  {err:.3f}")
    fs.writeComment("Same data as lists: images[i] has error per_image_error_px[i]")
    fs.startWriteStruct("images", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
    for name in result["images"]:
        fs.write("", name)
    fs.endWriteStruct()
    fs.startWriteStruct("per_image_error_px", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
    for err in result["per_image_errors"]:
        fs.write("", err)
    fs.endWriteStruct()

    fs.endWriteStruct()

def write_stereo(fs, stereo, run):
    """Write the extrinsics and the rectification transforms derived from them."""
    fs.writeComment("---- Stereo extrinsics: where the right camera sits relative to the left ----")
    fs.startWriteStruct("stereo", cv2.FileNode_MAP)
    if stereo is None:
        fs.writeComment("Too few frame pairs where both cameras saw the same tags.")
        fs.write("status", "not computed")
        fs.write("min_common_tags", run.min_common_tags)
        fs.endWriteStruct()
        return

    fs.write("status", "computed")
    fs.writeComment("R and T map a point from the LEFT camera frame to the RIGHT:")
    fs.writeComment("  X_right = R @ X_left + T        (T in mm, same units as tag_size_mm)")
    fs.writeComment("Intrinsics were held fixed (CALIB_FIX_INTRINSIC) while these were fitted.")
    fs.writeComment("No uncertainty is given for R and T: cv2.stereoCalibrate (OpenCV 5.0) does not report one.")
    fs.write("rotation_matrix", stereo["R"])
    fs.write("translation_mm", stereo["T"])
    fs.writeComment(f"baseline |T| = {stereo['baseline_mm']:.2f} mm - CHECK THIS AGAINST A RULER.")
    fs.writeComment("It carries the same scale error as TAG_SIZE, so a wrong tag size shows up here")
    fs.writeComment("first, and then in every reconstructed distance.")
    fs.write("baseline_mm", stereo["baseline_mm"])
    fs.writeComment("Rotation as angles about x, y, z (Rodrigues), degrees: "
                    + ", ".join(f"{a:.3f}" for a in stereo["angles_deg"]))
    fs.startWriteStruct("rotation_angles_deg", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
    for angle in stereo["angles_deg"]:
        fs.write("", float(angle))
    fs.endWriteStruct()
    fs.write("essential_matrix", stereo["E"])
    fs.write("fundamental_matrix", stereo["F"])

    fs.writeComment("---- Quality ----")
    fs.writeComment(f"rms = {stereo['rms']:.3f} px over {len(stereo['pairs'])} pairs")
    fs.write("rms_reprojection_error_px", stereo["rms"])
    fs.write("pairs_used", len(stereo["pairs"]))
    fs.write("pairs_available", stereo["n_pairs_total"])
    fs.write("min_common_tags", run.min_common_tags)
    fs.writeComment("Frame pairs used, named by their left image:")
    fs.startWriteStruct("pairs", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
    for name in stereo["pairs"]:
        fs.write("", name)
    fs.endWriteStruct()

    fs.writeComment("---- Rectification (from stereoRectify, alpha=0) ----")
    fs.writeComment("R1/R2 rotate each camera onto a common plane; P1/P2 are the new projections.")
    fs.writeComment("Q turns disparity into 3D mm: cv2.reprojectImageTo3D(disparity, Q).")
    fs.writeComment("Build the remap tables with cv2.initUndistortRectifyMap(K, D, R1, P1, size, ...).")
    fs.startWriteStruct("rectification", cv2.FileNode_MAP)
    for key in ("R1", "R2", "P1", "P2", "Q"):
        fs.write(key, stereo[key])
    fs.endWriteStruct()

    fs.endWriteStruct()


def write_detection(fs, run, found):
    """Tags found per image and per tag id, for every image of each camera (usable or not)."""
    fs.writeComment("---- Detection: how many of the board's tags were found, in every image ----")
    fs.startWriteStruct("detection", cv2.FileNode_MAP)
    for camera in CAMERAS:
        names = sorted(found[camera])
        # Ids off the board are false detections; a board id seen twice in one image is a misread.
        ids = {n: found[camera][n]["ids"] for n in names}
        on_board = {n: sorted({i for i in ids[n] if 0 <= i < N_TAGS}) for n in names}
        counts = np.array([len(on_board[n]) for n in names])
        false = np.array([sum(not 0 <= i < N_TAGS for i in ids[n]) for n in names])
        duplicates = np.array([len(ids[n]) - false[k] - counts[k] for k, n in enumerate(names)])
        per_id = np.zeros(N_TAGS, int)
        for n in names:
            per_id[on_board[n]] += 1
        fs.writeComment(f"{camera}: {len(names)} images, median {np.median(counts):.0f} of {N_TAGS} tags, "
                        f"{np.sum(counts >= run.min_tags)} usable (>= {run.min_tags} tags), "
                        f"{np.sum(counts == 0)} with none, {false.sum()} false detections (ids not on the board), "
                        f"{duplicates.sum()} duplicates")
        fs.startWriteStruct(camera, cv2.FileNode_MAP)
        fs.startWriteStruct("images", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for n in names:
            fs.write("", n)
        fs.endWriteStruct()
        fs.writeComment("tags_found[i] is the number of distinct board tags detected in images[i]")
        fs.startWriteStruct("tags_found", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for c in counts:
            fs.write("", int(c))
        fs.endWriteStruct()
        fs.writeComment("false_found[i] is the number of detections in images[i] with an id not on the board")
        fs.startWriteStruct("false_found", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for c in false:
            fs.write("", int(c))
        fs.endWriteStruct()
        fs.writeComment("duplicate_found[i] is the number of extra detections of a board id already found in images[i]")
        fs.startWriteStruct("duplicate_found", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for c in duplicates:
            fs.write("", int(c))
        fs.endWriteStruct()
        fs.writeComment(f"images_per_tag_id[k] is the number of images in which tag id k was found (ids 0..{N_TAGS - 1})")
        fs.startWriteStruct("images_per_tag_id", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for c in per_id:
            fs.write("", int(c))
        fs.endWriteStruct()
        fs.endWriteStruct()
    fs.endWriteStruct()


def write_rejection(fs, rejection):
    """What the outlier rejection dropped, per camera: settings, then every dropped tag and image."""
    fs.writeComment("---- Outlier rejection: tags and images dropped before the final fit (see REJECT_* in Calibration.py) ----")
    fs.startWriteStruct("outlier_rejection", cv2.FileNode_MAP)
    fs.write("tag_limit_rule", f"worst corner > max({REJECT_TAG_PX} px, {REJECT_TAG_MEDIANS} x median)")
    fs.write("image_limit_rule", f"kept-tag corner RMS > max({REJECT_IMAGE_PX} px, {REJECT_IMAGE_MEDIANS} x median)")
    for camera in CAMERAS:
        r = rejection[camera]
        fs.writeComment(f"{camera}: {len(r['rejected_images'])} images and {len(r['rejected_tags'])} tags dropped, "
                        f"{r['kept_tags']} tags kept, {r['rounds']} rounds")
        fs.startWriteStruct(camera, cv2.FileNode_MAP)
        fs.write("tag_limit_px", r["tag_limit_px"])
        fs.write("image_limit_px", r["image_limit_px"])
        fs.write("kept_tags", r["kept_tags"])
        fs.writeComment("rejected_tags[k] = 'image|tag id|worst corner px'")
        fs.startWriteStruct("rejected_tags", cv2.FileNode_SEQ)
        for n, i, e in r["rejected_tags"]:
            fs.write("", f"{n}|{i}|{e:.2f}")
        fs.endWriteStruct()
        fs.writeComment("rejected_images[k] = 'image|corner RMS px of the tags that passed (nan: too few did)'")
        fs.startWriteStruct("rejected_images", cv2.FileNode_SEQ)
        for n, e in r["rejected_images"]:
            fs.write("", f"{n}|{e:.2f}")
        fs.endWriteStruct()
        fs.endWriteStruct()
    fs.endWriteStruct()


def write_yaml(run, intrinsics, stereo, found=None, rejection=None):
    """Write the whole calibration, with the comments that make it readable on its own."""
    os.makedirs(run.out_dir, exist_ok=True)
    fs = cv2.FileStorage(run.yaml_path, cv2.FILE_STORAGE_WRITE)

    fs.writeComment("=====================================================================")
    fs.writeComment("STEREO CAMERA CALIBRATION")
    fs.writeComment("Generated by src/01_Calibration/Calibration.py - re-run the script rather than editing by hand.")
    fs.writeComment("Units: px = pixels (image quantities), mm = millimetres (board sizes).")
    fs.writeComment("Numbers are stored at full precision; comments show rounded values for reading.")
    fs.writeComment("Load in Python:")
    fs.writeComment('  fs = cv2.FileStorage("' + rel(run.yaml_path) + '", cv2.FILE_STORAGE_READ)')
    fs.writeComment('  K_left = fs.getNode("left_camera").getNode("camera_matrix").mat()')
    fs.writeComment("=====================================================================")

    fs.startWriteStruct("metadata", cv2.FileNode_MAP)
    fs.write("created", datetime.datetime.now().isoformat(timespec="seconds"))
    fs.write("script", "src/01_Calibration/Calibration.py")
    fs.write("opencv_version", cv2.__version__)
    fs.write("frames_dir", rel(run.frames_dir))
    fs.writeComment("Command-line arguments of this run (paths project-relative). Anything not given was at its default.")
    fs.write("arguments", run.arguments)
    fs.endWriteStruct()

    fs.writeComment("Image size the calibration was made at; it is only valid at this resolution")
    fs.startWriteStruct("image", cv2.FileNode_MAP)
    fs.write("width_px", run.image_size[0])
    fs.write("height_px", run.image_size[1])
    fs.endWriteStruct()

    fs.writeComment("Calibration target: Kalibr AprilGrid (kalibr_create_target_pdf --type apriltag --nx 10 --ny 7)")
    fs.startWriteStruct("calibration_board", cv2.FileNode_MAP)
    fs.write("type", "Kalibr AprilGrid")
    fs.write("tag_family", "AprilTag 36h11")
    fs.write("tags_x", 10)
    fs.write("tags_y", 7)
    fs.writeComment(f"tag size {TAG_SIZE:.2f} mm (outer black square, as printed), gap {TAG_GAP:.2f} mm")
    fs.write("tag_size_mm", TAG_SIZE)
    fs.write("tag_gap_mm", TAG_GAP)
    fs.writeComment("Detection: the detector named below with the thresholding preset named below,")
    fs.writeComment("its own corner refinement off, then cv2.cornerSubPix, both on the image channel named")
    fs.writeComment("below. 'aruco' is cv2.aruco.ArucoDetector; 'uwaruco' is the re-implementation of")
    fs.writeComment("Cejka et al. 2019 (doi:10.3390/rs11040459) in src/archive/UWARUco.")
    fs.writeComment("channel: gray = BGR2GRAY of the colour frame, y = the camera's JPEG luma, r/g/b = one colour channel")
    fs.write("channel", run.channel)
    if run.channel == "merge":
        fs.writeComment(f"merge: detected in each of {', '.join(MERGE_ORDER)}; a tag found in several takes the first one's")
        fs.writeComment(f"corners, and is dropped if their corners disagree by more than {MERGE_TOLERANCE_PX:g} px")
        fs.write("merge_order", ",".join(MERGE_ORDER))
        fs.write("merge_tolerance_px", MERGE_TOLERANCE_PX)
    fs.write("detector", run.detector_name)
    fs.write("detector_preset", run.preset)
    fs.writeComment("Full detector settings for this preset, so the run can be reproduced from this file alone:")
    fs.write("detector_settings", run.detector_settings)
    fs.write("corner_refinement", "cornerSubPix")
    fs.writeComment(f"cornerSubPix half-window = {SUBPIX_WIN_FRACTION:g} x tag side in px (at least {SUBPIX_WIN_MIN} px)")
    fs.write("corner_window_fraction", SUBPIX_WIN_FRACTION)
    fs.write("corner_window_min_px", SUBPIX_WIN_MIN)
    fs.write("min_tags_per_image", run.min_tags)
    fs.writeComment("on: misplaced tags and badly fitting images were dropped before the final fit (see outlier_rejection)")
    fs.write("outlier_rejection", "on" if run.reject else "off")
    fs.endWriteStruct()

    for camera in CAMERAS:
        write_camera(fs, camera, intrinsics[camera], run.image_size)

    write_stereo(fs, stereo, run)

    if rejection is not None:
        write_rejection(fs, rejection)

    if found is not None:
        write_detection(fs, run, found)

    fs.release()


def write_corners(run, found):
    """Every detection of the run, refined corners included, as JSON beside the YAML."""
    with open(run.corners_path, "w", encoding="utf-8") as fh:
        json.dump({"calibration": os.path.basename(run.yaml_path), "frames_dir": rel(run.frames_dir),
                   "channel": run.channel, "detector": run.detector_name, "preset": run.preset,
                   "corner_order": "as reported by the detector; corners[k] belongs to ids[k]",
                   "cameras": found}, fh)


def calibrate(argv=None):
    """Run one calibration end to end."""
    presets = load_presets()
    argv = sys.argv[1:] if argv is None else list(argv)
    arguments = " ".join(rel(a) if os.path.isabs(a) else a for a in argv)
    run = setup(parse_args(presets, argv), arguments)
    found = {camera: {} for camera in CAMERAS}
    detections = {camera: detect_tags(run, camera, found[camera]) for camera in CAMERAS}
    rejection = None
    if run.reject:
        rejection = {}
        for camera in CAMERAS:
            detections[camera], rejection[camera] = reject_outliers(run, camera, detections[camera])
    intrinsics = {camera: calibrate_camera(run, camera, detections[camera]) for camera in CAMERAS}
    stereo = calibrate_stereo(run, detections, intrinsics)
    write_yaml(run, intrinsics, stereo, found, rejection)
    write_corners(run, found)
    print(f"\nsaved calibration to {rel(run.yaml_path)} and corners to {rel(run.corners_path)}")
    return run, intrinsics, stereo


def main():
    calibrate()


if __name__ == "__main__":
    main()
