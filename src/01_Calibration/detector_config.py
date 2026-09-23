"""Detector configuration for the Kalibr AprilGrid (36h11), loaded from detector_presets.json.

The presets live in JSON so that tuning is a data change, not a code change, and so that the preset
a comparison reports is exactly the one the calibration ran. Calibration.py records the preset name
in its output YAML, which is what compare_calibrations.py keys its table on.

Use:
    from detector_config import make_detector, PRESETS, DEFAULT_PRESET, dictionary
    detector = make_detector()            # the JSON "default"
    detector = make_detector("uwaruco")   # or any key of PRESETS
"""
import json
import os

import cv2

PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_presets.json")

# Tag family: AprilTag 36h11
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

with open(PRESETS_FILE, encoding="utf-8") as _fh:
    _config = json.load(_fh)

BOARD_PARAMS = _config["board_params"]
DEFAULT_PRESET = _config["default"]
# Keys starting with "_" are documentation (_why, _comment), not detector parameters.
PRESETS = {name: {k: v for k, v in body.items() if not k.startswith("_")}
           for name, body in _config["presets"].items()}

if DEFAULT_PRESET not in PRESETS:
    raise SystemExit(f"{PRESETS_FILE}: default {DEFAULT_PRESET!r} is not one of {', '.join(PRESETS)}")


def make_detector(preset=None):
    """ArucoDetector for this board, using one of the presets in detector_presets.json.

    board_params are merged in under the preset, so a preset cannot silently drop the settings the
    target requires. Overriding one is allowed but warned about, because getting markerBorderBits
    wrong halves the tags found without raising anything.
    """
    preset = DEFAULT_PRESET if preset is None else preset
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; {PRESETS_FILE} defines: {', '.join(PRESETS)}")

    settings = {**BOARD_PARAMS, **PRESETS[preset]}
    for key in PRESETS[preset].keys() & BOARD_PARAMS.keys():
        print(f"  WARNING: preset {preset!r} overrides board parameter {key} "
              f"({BOARD_PARAMS[key]} -> {PRESETS[preset][key]})")

    params = cv2.aruco.DetectorParameters()
    # The detector's own refinement stays off: corners are refined by the caller with cv2.cornerSubPix
    # on the original image. Enabling both would refine twice, each pass moving the corners again.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    for key, value in settings.items():
        if not hasattr(params, key):
            raise SystemExit(f"{PRESETS_FILE}: preset {preset!r} sets {key!r}, which is not a "
                             f"cv2.aruco.DetectorParameters attribute")
        setattr(params, key, value)
    return cv2.aruco.ArucoDetector(dictionary, params)


def describe(preset):
    """One-line summary of a preset's settings, for logs and YAML metadata."""
    return ", ".join(f"{k}={v}" for k, v in sorted({**BOARD_PARAMS, **PRESETS[preset]}.items()))
