"""Compare calibration runs by reading the YAML files Calibration.py writes.

WHY THIS IS NOT JUST "LOWEST RMS WINS"
Each preset calibrates from a DIFFERENT set of images, because a preset that detects more tags pulls
in harder frames - blurrier, further away, more oblique - that the others dropped. Those frames are
exactly the ones with the largest reprojection error, so a preset can look worse on RMS purely for
having attempted more. Comparing the headline RMS across presets therefore penalises the better
detector.

This script reports both:
  * all-images RMS, over whatever each run happened to use, and
  * common-subset RMS, over only the images EVERY run used, which is the like-for-like number.
The YAML stores per-image error keyed by image name, so the common subset is recoverable after the
fact. A preset that wins on coverage AND ties on the common subset is a genuine improvement; one
that wins on the common subset but uses far fewer images has simply thrown away the hard frames.

Run (from any directory):
    src/venv/bin/python src/01_Calibration/compare_calibrations.py
    src/venv/bin/python src/01_Calibration/compare_calibrations.py --run wide uwaruco opencv_default
    src/venv/bin/python src/01_Calibration/compare_calibrations.py --run wide uwaruco:scaled
    src/venv/bin/python src/01_Calibration/compare_calibrations.py 'results/calibration/*_stereo.yaml'

A run to make is named "preset" for the default ArUco detector or "detector:preset" for another,
so "uwaruco:scaled" calibrates with the UWARUco detector of src/UWARUco at its "scaled" preset.
Rows are labelled the same way, so an ArUco preset and a UWARUco preset of the same name stay
apart in the table.
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Calibration  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GLOB = os.path.join(PROJECT_ROOT, "results/calibration", "*_stereo.yaml")
CAMERAS = ("left", "right")


def read_run(path):
    """One calibration YAML -> {preset, frames_dir, per camera: intrinsics, uncertainty, per-image error}."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    board = fs.getNode("calibration_board")
    # "detector" was added when the UWARUco detector arrived; a YAML written before that is an
    # ArUco run, so an absent field means "aruco" rather than an unreadable file.
    detector = board.getNode("detector").string() or "aruco"
    preset = board.getNode("detector_preset").string() or "?"
    run = {"path": path,
           "detector": detector,
           "preset": preset if detector == "aruco" else f"{detector}:{preset}",
           "min_tags": int(board.getNode("min_tags_per_image").real() or 0),
           "frames_dir": fs.getNode("metadata").getNode("frames_dir").string(),
           "cameras": {}}

    for camera in CAMERAS:
        node = fs.getNode(f"{camera}_camera")
        if node.empty():
            continue
        images = node.getNode("images")
        errors = node.getNode("per_image_error_px")
        per_image = {images.at(i).string(): errors.at(i).real() for i in range(images.size())}
        K = node.getNode("camera_matrix").mat()
        std = node.getNode("uncertainty_1sigma")
        run["cameras"][camera] = {
            "rms": node.getNode("rms_reprojection_error_px").real(),
            "n_images": int(node.getNode("images_used").real()),
            "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
            "fx_std": std.getNode("fx_px").real(), "cx_std": std.getNode("cx_px").real(),
            "k1": node.getNode("distortion_coefficients").mat().ravel()[0],
            "per_image": per_image,
        }
    fs.release()
    return run


def rms(values):
    return float(np.sqrt(np.mean(np.square(list(values))))) if len(values) else float("nan")


def compare(runs, camera):
    """Print one camera's table. Returns the set of images common to every run."""
    common = set.intersection(*(set(r["cameras"][camera]["per_image"]) for r in runs))

    width = max(16, max(len(r["preset"]) for r in runs) + 2)
    print(f"\n{camera.upper()} CAMERA   ({len(common)} images common to all {len(runs)} runs)")
    header = (f"{'preset':{width}s}{'images':>8s}{'RMS all':>10s}{'RMS common':>12s}"
              f"{'fx':>10s}{'fx 1sig':>9s}{'cx':>9s}{'k1':>9s}")
    print(header)
    print("-" * len(header))

    for run in runs:
        c = run["cameras"][camera]
        common_rms = rms([c["per_image"][n] for n in common])
        print(f"{run['preset']:{width}s}{c['n_images']:>8d}{c['rms']:>10.3f}{common_rms:>12.3f}"
              f"{c['fx']:>10.1f}{c['fx_std']:>9.2f}{c['cx']:>9.1f}{c['k1']:>9.4f}")

    if len(runs) > 1:
        spread_fx = max(r["cameras"][camera]["fx"] for r in runs) - min(r["cameras"][camera]["fx"] for r in runs)
        worst_std = max(r["cameras"][camera]["fx_std"] for r in runs)
        verdict = "within" if spread_fx <= worst_std else "WIDER than"
        print(f"  fx spread across presets {spread_fx:.1f} px, {verdict} the largest 1-sigma "
              f"({worst_std:.2f} px) - {'the presets agree on the lens' if spread_fx <= worst_std else 'the choice of preset is moving the answer'}")
    return common


def coverage(runs, camera, common):
    """Which images each run used that the common subset does not: the frames the preset bought."""
    if len(runs) < 2:
        return
    width = max(16, max(len(r["preset"]) for r in runs) + 2)
    print(f"  images beyond the common subset:")
    for run in runs:
        extra = set(run["cameras"][camera]["per_image"]) - common
        if not extra:
            print(f"    {run['preset']:{width}s} none")
            continue
        errs = [run["cameras"][camera]["per_image"][n] for n in extra]
        print(f"    {run['preset']:{width}s} {len(extra):3d} extra, their RMS {rms(errs):6.3f} px "
              f"(vs {rms([run['cameras'][camera]['per_image'][n] for n in common]):.3f} px on the common set)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default=DEFAULT_GLOB,
                    help="glob of calibration YAMLs (default results/calibration/*_stereo.yaml)")
    ap.add_argument("--run", nargs="+", metavar="PRESET",
                    help="run Calibration.py for these presets first, then compare. A name is "
                         "\"preset\" for the default ArUco detector or \"detector:preset\" for "
                         "another, e.g. uwaruco:scaled")
    ap.add_argument("--frames-dir", help="passed through to Calibration.py when --run is used")
    args = ap.parse_args()

    if args.run:
        for name in args.run:
            detector, _, preset = name.rpartition(":")
            detector = detector or "aruco"
            if detector not in Calibration.DETECTORS:
                sys.exit(f"unknown detector {detector!r} in {name!r}; "
                         f"Calibration.py has: {', '.join(Calibration.DETECTORS)}")
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Calibration.py"),
                   "--detector", detector, "--preset", preset, "--no-detection-images"]
            if args.frames_dir:
                cmd += ["--frames-dir", args.frames_dir]
            print(f"\n=== running Calibration.py --detector {detector} --preset {preset} ===")
            if subprocess.run(cmd).returncode:
                sys.exit(f"Calibration.py failed for {name}")

    paths = sorted(glob.glob(args.pattern if os.path.isabs(args.pattern)
                             else os.path.join(PROJECT_ROOT, args.pattern)))
    runs = [r for r in (read_run(p) for p in paths) if r and r["cameras"]]
    if not runs:
        sys.exit(f"no readable calibration YAMLs matching {args.pattern}")

    # Runs over different footage share no image names, so a single table would have an empty common
    # subset and every like-for-like number would be nan. Group by footage and compare within a group.
    groups = {}
    for run in runs:
        groups.setdefault(run["frames_dir"], []).append(run)

    # Two runs of the same footage and preset are the same experiment, usually an older output file
    # left behind by a rename. Label them by filename so the table does not show two identical rows.
    for group in groups.values():
        seen = {}
        for run in group:
            seen.setdefault(run["preset"], []).append(run)
        for preset, same in seen.items():
            if len(same) > 1:
                for run in same:
                    run["preset"] = f"{preset} [{os.path.basename(run['path']).removesuffix('_stereo.yaml')}]"

    for frames_dir, group in sorted(groups.items()):
        print(f"\n{'=' * 84}\nFOOTAGE: {frames_dir}   ({len(group)} run{'s' if len(group) > 1 else ''}: "
              f"{', '.join(r['preset'] for r in group)})")
        if len(group) == 1:
            print("  only one run for this footage; nothing to compare against")
        for camera in CAMERAS:
            present = [r for r in group if camera in r["cameras"]]
            if not present:
                continue
            common = compare(present, camera)
            coverage(present, camera, common)

    print("\nRMS common is the like-for-like number. More images at the same common RMS is the win;\n"
          "a lower common RMS from far fewer images just means the hard frames were dropped.")


if __name__ == "__main__":
    main()
