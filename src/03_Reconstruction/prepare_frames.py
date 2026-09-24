"""Turn extracted stereo frame pairs into a COLMAP workspace with known intrinsics.

Scores every frame for blur (variance of the Laplacian), drops the ones below a threshold, and
writes an image folder plus the COLMAP files needed to reconstruct with the intrinsics from
01_Calibration held FIXED rather than re-estimated from the scene.

Paths resolve from this file, so it can be run from any working directory:
    src/venv/bin/python src/03_Reconstruction/prepare_frames.py --dry-run
    src/venv/bin/python src/03_Reconstruction/prepare_frames.py --blur-threshold 15

Run it once with --dry-run to see the blur distribution for the recording, then pick a threshold.
There is no safe default: the score depends on scene contrast, and on the 18 Sep underwater frames
the value usually quoted for sharp photographs (100) rejects every single frame.

Left and right become two SEPARATE COLMAP cameras. They have genuinely different intrinsics
(fx 777.1 vs 767.9 on the 18 Sep set), so sharing one camera would force COLMAP to average them.
"""
import argparse
import csv
import glob
import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np

# Paths are relative to the project root, resolved from this file
# (src/03_Reconstruction/prepare_frames.py -> parents: 03_Reconstruction, src, project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAMERAS = ("left", "right")

# COLMAP camera models and the order of their PARAMS column.
# OPENCV has NO k3: OpenCV calibration fits k1, k2, p1, p2, k3, so the k3 term is DROPPED when this
# model is used. FULL_OPENCV keeps it (k4..k6 are written as zero, which OpenCV never fitted).
CAMERA_MODELS = {
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
}

# Thresholds worth knowing about, measured on the 104 left frames of 18_Sep/Rigframes:
#   min 7.1, p10 8.3, median 12.6, p90 39.8, max 64.4
# The frames calibration could actually use had a median of 17.3 against 11.4 for those it rejected,
# so the score does track usefulness on this footage - but only over a narrow range.
REPORT_PERCENTILES = (0, 10, 25, 50, 75, 90, 100)


@dataclass
class Camera:
    """One camera's COLMAP intrinsics, read from the calibration YAML."""
    name: str
    camera_id: int
    model: str
    width: int
    height: int
    params: list

    def params_string(self):
        """The value for COLMAP's --ImageReader.camera_params, which wants a bare comma list."""
        return ",".join(f"{v:.10g}" for v in self.params)

    def line(self):
        """One cameras.txt row: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"""
        return f"{self.camera_id} {self.model} {self.width} {self.height} {self.params_string()}"


@dataclass
class Frame:
    """One input frame and what was decided about it."""
    camera: str
    source: str      # absolute path of the input PNG
    name: str        # name inside the COLMAP image folder, e.g. "left/100_C100_Left_90.png"
    label: str       # identifies the stereo pair: same for a left frame and its right partner
    blur: float


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Prepare frame pairs as a COLMAP workspace.")
    ap.add_argument("--frames-dir", default=os.path.join(PROJECT_ROOT, "data/Calibration/18_Sep/Rigframes"),
                    help="one folder per recording, each with left/ and right/ (default 18_Sep/Rigframes)")
    ap.add_argument("--calibration", default=os.path.join(PROJECT_ROOT, "results/18_Sep/calibration/Rigframes_wide_stereo.yaml"),
                    help="calibration YAML from 01_Calibration/Calibration.py")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "work/18_Sep/Rigframes/prepare_frames"),
                    help="where the COLMAP workspace is written")
    ap.add_argument("--camera-model", default="OPENCV", choices=list(CAMERA_MODELS),
                    help="COLMAP camera model; OPENCV drops k3, FULL_OPENCV keeps it (default OPENCV)")

    blur = ap.add_mutually_exclusive_group()
    blur.add_argument("--blur-threshold", type=float,
                      help="drop frames whose variance of the Laplacian is below this; "
                           "with neither blur option given, every frame is kept")
    blur.add_argument("--blur-percentile", type=float,
                      help="drop this percent of the sharpest-to-blurriest range, per camera "
                           "(e.g. 25 drops the blurriest quarter). Adapts to the recording.")

    ap.add_argument("--require-pairs", action="store_true",
                    help="drop both frames of a pair when either one is too blurry "
                         "(default: drop only the blurry frame, since COLMAP does not need pairs)")
    ap.add_argument("--symlink", action="store_true",
                    help="link the images instead of copying them")
    ap.add_argument("--dry-run", action="store_true",
                    help="score the frames and report, but write nothing")
    return ap.parse_args(argv)


def rel(path):
    """Project-relative form of a path, for messages that stay portable."""
    return os.path.relpath(path, PROJECT_ROOT)


# ====== INTRINSICS ======
def load_cameras(calibration_path, model):
    """Read the two cameras out of the calibration YAML as COLMAP cameras.

    COLMAP camera ids are 1-based and must match the order the image lists are fed in, so left is
    always 1 and right always 2.
    """
    if not os.path.exists(calibration_path):
        raise SystemExit(f"no calibration at {rel(calibration_path)} - run "
                         f"src/01_Calibration/Calibration.py first, or pass --calibration")

    fs = cv2.FileStorage(calibration_path, cv2.FILE_STORAGE_READ)
    image_node = fs.getNode("image")
    width, height = int(image_node.getNode("width_px").real()), int(image_node.getNode("height_px").real())

    cameras = {}
    for camera_id, camera in enumerate(CAMERAS, start=1):
        node = fs.getNode(f"{camera}_camera")
        if node.isNone():
            raise SystemExit(f"{rel(calibration_path)}: no {camera}_camera section")
        K = node.getNode("camera_matrix").mat()
        D = node.getNode("distortion_coefficients").mat().ravel()
        k1, k2, p1, p2, k3 = (list(D) + [0.0] * 5)[:5]

        values = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                  "k1": k1, "k2": k2, "p1": p1, "p2": p2,
                  "k3": k3, "k4": 0.0, "k5": 0.0, "k6": 0.0}
        params = [float(values[key]) for key in CAMERA_MODELS[model]]
        cameras[camera] = Camera(camera, camera_id, model, width, height, params)

        print(f"{camera:5s} camera {camera_id}: {model} {width}x{height}  "
              f"fx={values['fx']:.1f} fy={values['fy']:.1f} "
              f"cx={values['cx']:.1f} cy={values['cy']:.1f}")
        if model == "OPENCV" and abs(k3) > 1e-9:
            print(f"  NOTE: k3 = {k3:.5f} is dropped by the OPENCV model. "
                  f"Use --camera-model FULL_OPENCV to keep it.")
    fs.release()
    return cameras


# ====== BLUR ======
def blur_score(gray):
    """Variance of the Laplacian: high on sharp edges, low when blur has smeared them out.

    Absolute values are NOT comparable between recordings: the score scales with scene contrast, so
    a dim underwater frame scores lower than a sharp one of a bright textured wall. Always pick the
    threshold from the distribution of the recording in hand (--dry-run).
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def collect_frames(frames_dir):
    """Score every frame of both cameras. Returns {camera: [Frame, ...]}.

    The name inside the COLMAP folder is flattened to <recording>_<file>, because COLMAP identifies
    an image by a single name and the recording folders would otherwise collide.
    """
    frames = {}
    for camera in CAMERAS:
        paths = sorted(glob.glob(os.path.join(frames_dir, "*", camera, "*.png")))
        if not paths:
            raise SystemExit(f"no {camera} PNGs under {rel(frames_dir)}/*/{camera}/ - "
                             f"check --frames-dir, or extract frames first with "
                             f"preproccessing/extract_stereo_frames.py")

        found = []
        for path in paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"could not read {path}")
            # Blur is measured on grey. Which grey does not matter here the way it does for corner
            # detection: this is a relative sharpness score, not a corner position.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            recording = os.path.basename(os.path.dirname(os.path.dirname(path)))
            filename = os.path.basename(path)
            found.append(Frame(camera=camera, source=path,
                               name=f"{camera}/{recording}_{filename}",
                               label=f"{recording}_{filename.replace('_Left_', '_').replace('_Right_', '_')}",
                               blur=blur_score(gray)))

        if len({f.name for f in found}) != len(found):
            raise SystemExit(f"{camera}: two input frames flatten to the same COLMAP name")
        frames[camera] = found
    return frames


def report_blur(frames):
    """Print the blur distribution and what a few candidate thresholds would keep."""
    print("\nBLUR (variance of the Laplacian)")
    everything = np.array([f.blur for camera in CAMERAS for f in frames[camera]])
    for camera in CAMERAS:
        scores = np.array([f.blur for f in frames[camera]])
        percentiles = "  ".join(f"p{p}={np.percentile(scores, p):.1f}" for p in REPORT_PERCENTILES)
        print(f"  {camera:5s} n={len(scores):3d}  {percentiles}")

    print("  a threshold would keep:")
    for percentile in (10, 25, 50, 75):
        threshold = float(np.percentile(everything, percentile))
        kept = {c: int((np.array([f.blur for f in frames[c]]) >= threshold).sum()) for c in CAMERAS}
        print(f"    {threshold:8.1f}  left {kept['left']:3d}/{len(frames['left']):<3d} "
              f"right {kept['right']:3d}/{len(frames['right']):<3d}  (drops the blurriest {percentile}%)")

    print("  worst 5 frames:")
    for frame in sorted((f for c in CAMERAS for f in frames[c]), key=lambda f: f.blur)[:5]:
        print(f"    {frame.blur:8.1f}  {frame.name}")


def select(frames, args):
    """Apply the blur rule. Returns {camera: [Frame, ...]} of the frames to write."""
    if args.blur_percentile is not None:
        everything = np.array([f.blur for camera in CAMERAS for f in frames[camera]])
        threshold = float(np.percentile(everything, args.blur_percentile))
        print(f"\n--blur-percentile {args.blur_percentile:g} -> threshold {threshold:.2f}")
    elif args.blur_threshold is not None:
        threshold = args.blur_threshold
    else:
        print("\nno blur threshold given, keeping every frame "
              "(run with --dry-run to choose one from the distribution above)")
        return {camera: list(frames[camera]) for camera in CAMERAS}

    kept = {camera: [f for f in frames[camera] if f.blur >= threshold] for camera in CAMERAS}

    if args.require_pairs:
        # COLMAP reconstructs from single images, so a frame whose partner is blurry is still
        # useful. Keeping pairs intact only matters if the extrinsics are to be imposed later.
        labels = {f.label for f in kept["left"]} & {f.label for f in kept["right"]}
        dropped = sum(len(kept[c]) - len([f for f in kept[c] if f.label in labels]) for c in CAMERAS)
        kept = {camera: [f for f in kept[camera] if f.label in labels] for camera in CAMERAS}
        print(f"--require-pairs dropped a further {dropped} frames whose partner was too blurry")

    for camera in CAMERAS:
        n_in, n_out = len(frames[camera]), len(kept[camera])
        print(f"  {camera:5s}: keeping {n_out} of {n_in} frames ({n_in - n_out} too blurry)")
    if not all(kept.values()):
        raise SystemExit("the threshold rejected every frame of a camera; lower it")
    return kept


# ====== COLMAP WORKSPACE ======
def write_images(out_dir, kept, symlink):
    """Copy (or link) the kept frames into <out-dir>/images/<camera>/."""
    image_dir = os.path.join(out_dir, "images")
    for camera in CAMERAS:
        os.makedirs(os.path.join(image_dir, camera), exist_ok=True)
    for camera in CAMERAS:
        for frame in kept[camera]:
            target = os.path.join(image_dir, frame.name)
            if os.path.lexists(target):
                os.remove(target)
            if symlink:
                os.symlink(frame.source, target)
            else:
                shutil.copy2(frame.source, target)
    return image_dir


def write_cameras(out_dir, cameras):
    """Write sparse/cameras.txt in COLMAP's text model format."""
    sparse_dir = os.path.join(out_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    path = os.path.join(sparse_dir, "cameras.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write(f"# Number of cameras: {len(cameras)}\n")
        fh.write("# Intrinsics calibrated by src/01_Calibration/Calibration.py; they are valid only\n")
        fh.write("# at this resolution. Hold them fixed in the mapper rather than re-estimating.\n")
        for camera in CAMERAS:
            fh.write(f"# {camera}\n{cameras[camera].line()}\n")
    return path


def write_image_lists(out_dir, kept):
    """One image list per camera, for feeding feature_extractor once per camera.

    COLMAP takes a single --ImageReader.camera_params for a whole feature_extractor run, so two
    cameras with different intrinsics mean two runs over two lists, into the same database.
    Names are relative to the image folder, which is what COLMAP expects in a list.
    """
    paths = {}
    for camera in CAMERAS:
        path = os.path.join(out_dir, f"{camera}_images.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(frame.name for frame in kept[camera]) + "\n")
        paths[camera] = path
    return paths


def write_blur_report(out_dir, frames, kept):
    """Every input frame with its score and whether it survived, so a threshold can be revisited."""
    survivors = {f.name for camera in CAMERAS for f in kept[camera]}
    path = os.path.join(out_dir, "blur_report.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["camera", "colmap_name", "pair_label", "laplacian_variance", "kept", "source"])
        for camera in CAMERAS:
            for frame in sorted(frames[camera], key=lambda f: f.blur):
                writer.writerow([camera, frame.name, frame.label, f"{frame.blur:.4f}",
                                 int(frame.name in survivors), rel(frame.source)])
    return path


def write_run_script(out_dir, cameras):
    """A runnable COLMAP pipeline with the calibrated intrinsics pinned.

    The three ba_refine flags are the point of this whole script: without them the mapper treats the
    intrinsics as a starting guess and refits them on the scene, which throws away the calibration.
    """
    path = os.path.join(out_dir, "run_colmap.sh")
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by src/03_Reconstruction/prepare_frames.py - re-run that rather than editing.",
        "# Reconstructs with the intrinsics from 01_Calibration held FIXED.",
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'DB="$HERE/database.db"',
        'IMAGES="$HERE/images"',
        'OUT="$HERE/sparse"',
        'mkdir -p "$OUT"',
        "",
        "# One feature_extractor run per camera: COLMAP applies a single --camera_params per run,",
        "# and left and right have different intrinsics.",
    ]
    for camera in CAMERAS:
        cam = cameras[camera]
        lines += [
            f'colmap feature_extractor \\',
            f'    --database_path "$DB" --image_path "$IMAGES" \\',
            f'    --image_list_path "$HERE/{camera}_images.txt" \\',
            f'    --ImageReader.camera_model {cam.model} \\',
            f'    --ImageReader.single_camera 1 \\',
            f'    --ImageReader.camera_params "{cam.params_string()}"',
            "",
        ]
    lines += [
        "colmap exhaustive_matcher --database_path \"$DB\"",
        "",
        "# ba_refine_* = 0 keeps the calibrated intrinsics: the mapper would otherwise re-estimate",
        "# them from the scene and silently discard the calibration.",
        'colmap mapper \\',
        '    --database_path "$DB" --image_path "$IMAGES" --output_path "$OUT" \\',
        '    --Mapper.ba_refine_focal_length 0 \\',
        '    --Mapper.ba_refine_principal_point 0 \\',
        '    --Mapper.ba_refine_extra_params 0',
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def main():
    args = parse_args()
    cameras = load_cameras(args.calibration, args.camera_model)
    frames = collect_frames(args.frames_dir)
    report_blur(frames)
    kept = select(frames, args)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    image_dir = write_images(args.out_dir, kept, args.symlink)
    write_cameras(args.out_dir, cameras)
    write_image_lists(args.out_dir, kept)
    write_blur_report(args.out_dir, frames, kept)
    script = write_run_script(args.out_dir, cameras)

    total = sum(len(kept[camera]) for camera in CAMERAS)
    print(f"\nwrote {total} images to {rel(image_dir)}")
    print(f"COLMAP workspace in {rel(args.out_dir)}:")
    print("  sparse/cameras.txt   calibrated intrinsics, COLMAP text model")
    print("  left_images.txt, right_images.txt   one list per camera")
    print("  blur_report.csv      every frame scored, kept or not")
    print(f"  run_colmap.sh        the pipeline with intrinsics pinned")
    print(f"\nrun it with:  {rel(script)}")


if __name__ == "__main__":
    main()
