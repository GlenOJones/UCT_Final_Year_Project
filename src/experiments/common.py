"""What every experiment shares: making pipeline runs, and recording what the results came from."""
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
SRC_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "02_calibration"))
import Calibration  # noqa: E402

# The settings block every pipeline run in an experiment must spell out in full.
CALIBRATION_KEYS = ("detector", "preset", "channel", "reject", "min_tags", "min_common_tags", "detection_images")


def project_path(path):
    """A config path (relative to the project root) made absolute."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ====== PIPELINE RUNS ======
def check_settings(settings, where):
    missing = [k for k in CALIBRATION_KEYS if k not in settings]
    if missing:
        raise SystemExit(f"{where}: calibration settings must state every pipeline setting, missing: "
                         f"{', '.join(missing)}. Experiments do not rely on Calibration.py's defaults.")


def calibration_argv(settings, frames_dir, out_dir):
    """Calibration.py arguments for one run, every setting explicit."""
    check_settings(settings, "calibration settings")
    argv = ["--frames-dir", frames_dir, "--out-dir", out_dir,
            "--detector", settings["detector"], "--preset", settings["preset"], "--channel", settings["channel"],
            "--min-tags", str(settings["min_tags"]), "--min-common-tags", str(settings["min_common_tags"])]
    argv += [] if settings["detection_images"] else ["--no-detection-images"]
    return argv + ([] if settings["reject"] else ["--no-reject"])


def calibration_paths(settings, frames_dir, out_dir):
    """(YAML, corners JSON) Calibration.py writes for this run, found without running it."""
    name = Calibration.run_name(frames_dir, settings["detector"], settings["preset"], settings["channel"])
    return os.path.join(out_dir, name + "_stereo.yaml"), os.path.join(out_dir, name + "_corners.json")


def calibrate(settings, frames_dir, out_dir, skip_existing=False):
    """Make one pipeline run (or keep it, with skip_existing)."""
    argv = calibration_argv(settings, frames_dir, out_dir)
    yaml_path, corners_path = calibration_paths(settings, frames_dir, out_dir)
    record = {"yaml": rel(yaml_path), "argv": [rel(a) if os.path.isabs(a) else a for a in argv]}
    if skip_existing and os.path.exists(yaml_path) and os.path.exists(corners_path):
        print(f"=== {rel(yaml_path)}: exists, skipped")
        return {**record, "skipped": True}
    print(f"\n=== {rel(yaml_path)} ===")
    Calibration.calibrate(argv)
    return record


# ====== PROVENANCE ======
def git(*args):
    return subprocess.run(["git", "-C", PROJECT_ROOT, *args], capture_output=True, text=True).stdout.strip()


def uncommitted_src():
    return git("status", "--porcelain", "--", "src").splitlines()


def require_clean(allow_dirty):
    """Refuse to make results from code that is not a commit, unless told to."""
    dirty = uncommitted_src()
    if not dirty:
        return
    listing = "\n  ".join(dirty[:20]) + ("\n  ..." if len(dirty) > 20 else "")
    if not allow_dirty:
        raise SystemExit(f"src/ has uncommitted changes, so the results could not be traced to a commit:\n  {listing}\n"
                         f"Commit first, or pass --allow-dirty for a trial run (the manifest will say so).")
    print(f"WARNING: src/ has uncommitted changes; the manifest records them:\n  {listing}")


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


def write_manifest(out_dir, experiment, configs, inputs, presets, runs, extra=None):
    """manifest.json in out_dir: what these results were made from."""
    all_presets = Calibration.load_presets()
    manifest = {
        "experiment": experiment,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([rel(os.path.abspath(sys.argv[0]))] + sys.argv[1:]),
        "git_commit": git("rev-parse", "HEAD"),
        "git_uncommitted_src_changes": uncommitted_src(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "configs": {rel(p): load_config(p) for p in configs},
        "detector_presets": {p: all_presets.describe(p) for p in sorted(set(presets))},
        "input_digests": {rel(p): folder_digest(p) for p in sorted(set(inputs))},
        "runs": runs,
        **(extra or {}),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {rel(path)}")
    return path
