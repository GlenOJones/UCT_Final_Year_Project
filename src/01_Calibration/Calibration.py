"""Stereo camera calibration from Kalibr AprilGrid (36h11) frames.

Detects the board in every extracted frame pair, refines the corners to sub-pixel accuracy, and fits
a pinhole-plus-distortion model for each camera. The result is written as a commented YAML file that
is readable both by a person and by cv2.FileStorage.

Paths resolve from this file, so it can be run from any working directory:
    src/venv/bin/python src/01_Calibration/Calibration.py
    src/venv/bin/python src/01_Calibration/Calibration.py --preset uwaruco --min-tags 8
    src/venv/bin/python src/01_Calibration/Calibration.py --detector uwaruco --preset scaled

Two detectors can be put in front of the calibration. --detector aruco (the default) is OpenCV's
ArucoDetector, tuned by a preset from detector_presets.json, which detector_config.py checks on
load. --detector uwaruco is the re-implementation of the underwater detector of Cejka et al. 2019
in src/UWARUco, tuned by a preset from src/UWARUco/uwaruco_presets.json. Either way the detector,
the preset name and its full settings are recorded in the output, so a run can be reproduced from
the YAML alone, and compare_calibrations.py reads those outputs to compare runs.

Each camera is calibrated on every image it could use, then the rotation and baseline between
them are fitted with the intrinsics held fixed, on the frame pairs where both cameras saw enough of
the same tags. The rectification transforms are saved alongside, so 03_Reconstruction can rectify
and triangulate from this file alone.

See docs/calibration_improvement_notes.md for notes on improving the quality of the result.
"""
import argparse
import datetime
import glob
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_config import parameters, validate  # noqa: E402

# Paths are relative to the project root, resolved from this file
# (src/01_Calibration/Calibration.py -> parents: 01_Calibration, src, project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====== BOARD GEOMETRY ======
# Board config used: kalibr_create_target_pdf --type apriltag --nx 10 --ny 7 --tsize 0.050 --tspace 0.3
# TAG_SIZE should be MEASURED on the print with callipers, not taken from the PDF: printers scale,
# and a 1% size error goes straight into the baseline and every reconstructed distance.
TAG_SIZE = 35.36            # mm, edge of the outer black square as printed
TAG_GAP = TAG_SIZE * 0.3    # mm, white gap between tags
PITCH = TAG_SIZE + TAG_GAP  # mm, centre to centre
N_TAGS = 70                 # 10 across x 7 up, ids 0..69

CAMERAS = ("left", "right")

# Corner refinement. On the 18 Sep underwater frames the built-in options were poor:
#   CORNER_REFINE_APRILTAG  accurate corners (~0.3 px) but drops ~75% of the tags
#   CORNER_REFINE_SUBPIX    keeps the tags but corners are off by ~1.5 px (OpenCV 5.0.0)
# Calling cv2.cornerSubPix directly keeps every tag and gives ~0.3 px corners, so the detector's own
# refinement is left off (forced in Presets.make_detector).
SUBPIX_WIN = (5, 5)   # half-size of the search window, px
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)


# ====== DETECTOR PRESETS ======
# Tag family printed on the board: AprilTag 36h11.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_presets.json")


@dataclass
class Presets:
    """The validated contents of detector_presets.json.

    Tuning lives in JSON so that changing it is a data change rather than a code change. Structure
    and key names are checked by detector_config.validate() when the file is read, so everything
    here can assume the presets are usable.
    """
    path: str
    default: str
    board_params: dict
    presets: dict

    def settings(self, preset):
        """Detector parameters for one preset: board_params overlaid with the preset's own tuning.

        board_params describe the printed target rather than the tuning, so merging them in under
        every preset means a preset cannot silently drop a setting the board requires.
        """
        if preset not in self.presets:
            raise SystemExit(f"unknown preset {preset!r}; {self.path} defines: {', '.join(self.presets)}")
        return {**self.board_params, **self.presets[preset]}

    def describe(self, preset):
        """One-line summary of a preset's settings, for logs and for the YAML metadata."""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.settings(preset).items()))

    def make_detector(self, preset):
        """ArucoDetector for this board, configured from one preset."""
        params = cv2.aruco.DetectorParameters()
        # Set before the preset is applied so that no preset can turn it back on: corners are
        # refined afterwards by refine_corners() on the original image, and enabling both would
        # refine twice, each pass moving the corners again.
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
# Two detectors can be put in front of the same calibration: OpenCV's ArucoDetector, and the
# re-implementation of Cejka et al.'s underwater ARUco in src/UWARUco. They are swapped here rather
# than compared in a script of their own, so that what gets compared is two real calibrations of the
# same frames and not two tag counts from a program that only resembles this one.
#
# Each entry answers three questions: which preset to use when none is named, what the settings were
# (recorded in the YAML), and how to build the detector. Anything with a detectMarkers(gray) that
# returns (corners, ids, rejected) fits here.
UWARUCO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UWARUco")


def uwaruco_module():
    """Import src/UWARUco/uwaruco.py. Imported on use so that ArUco runs do not depend on it."""
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

    @property
    def name(self):
        """Runs are named by recording, detector and preset, so a sweep over either does not
        overwrite itself and compare_calibrations.py can pick the whole set up with a glob."""
        recording = os.path.basename(os.path.normpath(self.frames_dir)).replace(" ", "_")
        # The default detector is left unlabelled so that the ArUco runs made before UWARUco existed
        # keep their filenames and stay comparable with new ones.
        label = self.preset if self.detector_name == "aruco" else f"{self.detector_name}-{self.preset}"
        return f"{recording}_{label}"

    @property
    def yaml_path(self):
        return os.path.join(self.out_dir, self.name + "_stereo.yaml")

    @property
    def detections_dir(self):
        return os.path.join(self.out_dir, self.name + "_detections")


def rel(path):
    """Project-relative form of a path, so what is recorded in the YAML stays portable."""
    return os.path.relpath(path, PROJECT_ROOT)


def parse_args(presets, argv=None):
    ap = argparse.ArgumentParser(description="Calibrate the stereo cameras from AprilGrid frames.")
    ap.add_argument("--frames-dir", default=os.path.join(PROJECT_ROOT, "data/Calibration/18_Sep/Rigframes"),
                    help="one folder per recording, each with left/ and right/ (default 18_Sep/Rigframes)")
    ap.add_argument("--detector", default="aruco", choices=list(DETECTORS),
                    help="which detector implementation to use: OpenCV's ArUco, or the UWARUco "
                         "re-implementation in src/UWARUco (default aruco)")
    # No choices= here: the valid presets depend on --detector, and each detector's own loader
    # already rejects an unknown name with a message listing the ones it does define.
    ap.add_argument("--preset", default=None,
                    help=f"detector preset; from detector_presets.json for --detector aruco "
                         f"(default {presets.default}), or from src/UWARUco/uwaruco_presets.json "
                         f"for --detector uwaruco")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results/calibration"),
                    help="where the YAML and detection images go")
    ap.add_argument("--min-tags", type=int, default=10,
                    help="tags an image needs to be used (default 10)")
    ap.add_argument("--min-common-tags", type=int, default=10,
                    help="tags both cameras must see in a pair for it to be used for extrinsics "
                         "(default 10)")
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
        # Kalibr prints tags rotated 180 deg, so OpenCV reports: bottom-right, bottom-left, top-left,
        # top-right. The object points below are in that same order; changing one without the other
        # silently corrupts the fit.
        tag_corners.append(np.array([[x + TAG_SIZE, y, 0], [x, y, 0],
                                     [x, y + TAG_SIZE, 0], [x + TAG_SIZE, y + TAG_SIZE, 0]], np.float32))
    return cv2.aruco.Board(tag_corners, dictionary, np.arange(N_TAGS))


def setup(args):
    """Validate the inputs and resolve everything the run needs. Returns a Run."""
    default_preset, describe, make_detector = DETECTORS[args.detector]()
    preset = args.preset or default_preset

    for camera in CAMERAS:
        if not find_images(args.frames_dir, camera):
            raise SystemExit(f"no {camera} PNGs under {args.frames_dir}/*/{camera}/ - "
                             f"check --frames-dir, or extract frames first with "
                             f"preproccessing/extract_stereo_frames.py")

    first_image = find_images(args.frames_dir, "left")[0]
    image_size = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE).shape[::-1]   # (width, height)
    print("image size:", image_size)
    print(f"detector: {args.detector}  preset: {preset}  ({describe(preset)})")

    return Run(frames_dir=args.frames_dir, detector_name=args.detector, preset=preset,
               out_dir=args.out_dir,
               min_tags=args.min_tags, min_common_tags=args.min_common_tags,
               save_detections=not args.no_detection_images,
               image_size=image_size, detector_settings=describe(preset),
               detector=make_detector(preset), board=build_board())


# ====== DETECTION ======
def refine_corners(gray, corners):
    """Sub-pixel refine every tag corner on the original image."""
    return tuple(cv2.cornerSubPix(gray, c.reshape(4, 1, 2).copy(), SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA)
                 .reshape(1, 4, 2) for c in corners)


def save_detection_image(run, camera, name, gray, corners, ids, rejected):
    """Save a copy of the image with detected tags (green, with id) and rejected candidates (red)."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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


def detect_tags(run, camera):
    """Detect tags in every image of one camera.
    Returns {"folder/filename": (corners, ids)} for the usable images only."""
    image_files = find_images(run.frames_dir, camera)
    detections = {}
    for path in image_files:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray.shape[::-1] != run.image_size:
            raise SystemExit(f"{path}: size {gray.shape[::-1]} differs from {run.image_size}")
        name = os.path.relpath(path, run.frames_dir).replace(os.sep + camera + os.sep, "/")   # "140/C140_Left_90.png"
        corners, ids, rejected = run.detector.detectMarkers(gray)
        corners = refine_corners(gray, corners)
        if run.save_detections:
            save_detection_image(run, camera, name, gray, corners, ids, rejected)

        n_tags = 0 if ids is None else len(ids)
        if n_tags < run.min_tags:
            print(f"  {camera}: skip {name}: {n_tags} tags")
            continue

        detections[name] = (corners, ids)

    print(f"{camera}: {len(detections)} of {len(image_files)} images usable")
    return detections


# ====== CAMERA CALIBRATION ======
def calibrate_camera(run, camera, detections):
    """Calibrate one camera from its detections.
    Returns a dict: camera_matrix, dist_coeffs, rms, std_intrinsics, images, per_image_errors."""
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
        "images": names,
        "per_image_errors": per_view_errors,
    }


# ====== STEREO EXTRINSICS ======
def pair_up(detections):
    """Match each left detection to its right one, as (left_name, right_name) pairs.

    extract_stereo_frames.py writes a timestamp-matched pair under the same label, so the two names
    differ only by Left/Right and the pairing is already in the filename. pairs.csv says the same
    thing, but reading it would tie this script to a file the frames do not have to carry.
    """
    right = detections["right"]
    return [(name, name.replace("_Left_", "_Right_")) for name in sorted(detections["left"])
            if name.replace("_Left_", "_Right_") in right]


def common_tag_points(run, left, right):
    """Object and image points for the tags BOTH cameras saw in one pair, or None if too few.

    stereoCalibrate needs one object point list that both image point lists line up with, so the
    tags each camera saw on its own are useless here and have to be dropped. Ids are sorted so the
    two cameras contribute their corners in the same order.
    """
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
    """Fit the rotation and translation from the left camera to the right. Returns a dict, or None.

    The intrinsics are held FIXED (CALIB_FIX_INTRINSIC). Each camera was just calibrated on every
    image it could use on its own, whereas only the pairs where both cameras saw the same tags can
    contribute here - usually far fewer. Refitting the intrinsics on that smaller set would replace
    a good estimate with a worse one, and a single bad pair could then corrupt both cameras.
    """
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

    # Rectification is a pure function of the intrinsics and extrinsics, so it costs nothing to
    # derive here and saves 03_Reconstruction from having to recompute it consistently.
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


def write_yaml(run, intrinsics, stereo):
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
    fs.writeComment("its own corner refinement off, then cv2.cornerSubPix on the original grayscale")
    fs.writeComment("image. 'aruco' is cv2.aruco.ArucoDetector; 'uwaruco' is the re-implementation of")
    fs.writeComment("Cejka et al. 2019 (doi:10.3390/rs11040459) in src/UWARUco.")
    fs.write("detector", run.detector_name)
    fs.write("detector_preset", run.preset)
    fs.writeComment("Full detector settings for this preset, so the run can be reproduced from this file alone:")
    fs.write("detector_settings", run.detector_settings)
    fs.write("corner_refinement", "cornerSubPix")
    fs.write("min_tags_per_image", run.min_tags)
    fs.endWriteStruct()

    for camera in CAMERAS:
        write_camera(fs, camera, intrinsics[camera], run.image_size)

    write_stereo(fs, stereo, run)

    fs.release()


def calibrate(argv=None):
    """Run one calibration end to end. Returns (run, intrinsics, stereo) for callers to use.

    The detections are kept rather than consumed per camera: the stereo fit needs the same tags
    again, and detecting them twice would be both slow and a chance for the two passes to disagree.
    """
    presets = load_presets()
    run = setup(parse_args(presets, argv))
    detections = {camera: detect_tags(run, camera) for camera in CAMERAS}
    intrinsics = {camera: calibrate_camera(run, camera, detections[camera]) for camera in CAMERAS}
    stereo = calibrate_stereo(run, detections, intrinsics)
    write_yaml(run, intrinsics, stereo)
    print(f"\nsaved calibration to {rel(run.yaml_path)}")
    return run, intrinsics, stereo


def main():
    calibrate()


if __name__ == "__main__":
    main()
