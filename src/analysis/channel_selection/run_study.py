"""Run the channel selection study: one Calibration.py run per medium and channel in study.json.

Nothing is measured here. Each run is an ordinary calibration, made by calling Calibration.py's own
calibrate() with the study's settings plus --channel, so what analyse_study.py compares is exactly
what the pipeline produces. Outputs go to <out_dir>/<medium>/ (YAML + corners JSON per channel).

A manifest.json beside them records what the results came from: the git commit and whether src/ had
uncommitted changes, library versions, the study config, and a SHA-256 digest of every input folder.
If the manifest of two runs matches, their outputs should match too (only the timestamps differ).

Use (from the project root):
    src/venv/bin/python src/analysis/channel_selection/run_study.py
    src/venv/bin/python src/analysis/channel_selection/run_study.py --skip-existing   # resume
    src/venv/bin/python src/analysis/channel_selection/run_study.py --study src/analysis/channel_selection/study_opencv_default.json
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "01_Calibration"))
import Calibration  # noqa: E402

STUDY_FILE = os.path.join(HERE, "study.json")


def load_study(path=STUDY_FILE):
    """study.json with its paths made absolute."""
    with open(path, encoding="utf-8") as fh:
        study = json.load(fh)
    study["config_path"] = os.path.abspath(path)
    study["media"] = {m: os.path.join(PROJECT_ROOT, p) for m, p in study["media"].items()}
    study["test_target"]["frames"] = os.path.join(PROJECT_ROOT, study["test_target"]["frames"])
    study["out_dir"] = os.path.join(PROJECT_ROOT, study["out_dir"])
    return study


def add_study_argument(parser):
    parser.add_argument("--study", default=STUDY_FILE,
                        help="study config (default study.json); e.g. study_opencv_default.json for the same "
                             "comparison with the stock OpenCV thresholding")


def run_name(study, medium, channel):
    """The name Calibration.py gives this run, so its outputs can be found without running it."""
    cal = study["calibration"]
    return Calibration.run_name(study["media"][medium], cal["detector"], cal["preset"], channel)


def yaml_path(study, medium, channel):
    return os.path.join(study["out_dir"], medium, run_name(study, medium, channel) + "_stereo.yaml")


def corners_path(study, medium, channel):
    return os.path.join(study["out_dir"], medium, run_name(study, medium, channel) + "_corners.json")


# ====== PROVENANCE ======
def folder_digest(folder):
    """SHA-256 over every file under folder (relative path + contents), in sorted order."""
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(root, name)
            digest.update(os.path.relpath(path, folder).encode())
            with open(path, "rb") as fh:
                digest.update(hashlib.sha256(fh.read()).digest())
    return digest.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", PROJECT_ROOT, *args], capture_output=True, text=True).stdout.strip()


def write_manifest(study, runs):
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_uncommitted_src_changes": git("status", "--porcelain", "--", "src").splitlines(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "study_file": os.path.relpath(study["config_path"], PROJECT_ROOT),
        "study": json.load(open(study["config_path"], encoding="utf-8")),
        "input_digests": {os.path.relpath(p, PROJECT_ROOT): folder_digest(p)
                          for p in [*study["media"].values(), study["test_target"]["frames"]]},
        "runs": runs,
    }
    path = os.path.join(study["out_dir"], "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    if manifest["git_uncommitted_src_changes"]:
        print("WARNING: src/ has uncommitted changes, so the commit in manifest.json does not fully "
              "describe the code that made these results")
    print(f"wrote {os.path.relpath(path, PROJECT_ROOT)}")


# ====== RUNS ======
def calibration_argv(study, medium, channel):
    cal = study["calibration"]
    argv = ["--frames-dir", study["media"][medium],
            "--detector", cal["detector"], "--preset", cal["preset"], "--channel", channel,
            "--min-tags", str(cal["min_tags"]), "--min-common-tags", str(cal["min_common_tags"]),
            "--out-dir", os.path.join(study["out_dir"], medium)]
    return argv + ([] if cal["detection_images"] else ["--no-detection-images"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_study_argument(ap)
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep runs whose YAML and corners file already exist instead of redoing them")
    args = ap.parse_args()
    study = load_study(args.study)

    runs = []
    for medium in study["media"]:
        for channel in study["channels"]:
            argv = calibration_argv(study, medium, channel)
            runs.append({"medium": medium, "channel": channel,
                         "yaml": os.path.relpath(yaml_path(study, medium, channel), PROJECT_ROOT),
                         "argv": [os.path.relpath(a, PROJECT_ROOT) if os.path.isabs(a) else a for a in argv]})
            if args.skip_existing and all(os.path.exists(p) for p in (yaml_path(study, medium, channel),
                                                                     corners_path(study, medium, channel))):
                print(f"=== {medium} / {channel}: exists, skipped")
                continue
            print(f"\n=== {medium} / {channel} ===")
            Calibration.calibrate(argv)
    write_manifest(study, runs)


if __name__ == "__main__":
    main()
