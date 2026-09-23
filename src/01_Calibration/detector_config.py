"""Validation for detector_presets.json.

Calibration.py reads the JSON and builds its own detector from it; this module's only job is to say
whether what was read is usable. Keeping the checks here means a bad preset file fails once, at load,
with a message naming the offending key, instead of surfacing later as a detector that silently
finds half the tags.

Use:
    from detector_config import validate
    config = validate(json.load(open(PRESETS_FILE)), PRESETS_FILE)
"""
import cv2

# Keys starting with "_" are documentation (_why, _comment), not detector parameters.
DOC_PREFIX = "_"

# cv2.aruco.DetectorParameters takes numbers and flags; anything else fails on setattr.
VALID_TYPES = (bool, int, float)


def parameters(body):
    """The real detector parameters of a preset or of board_params, with documentation keys dropped."""
    return {k: v for k, v in body.items() if not k.startswith(DOC_PREFIX)}


def check_parameters(body, where, path):
    """Every key must name a cv2.aruco.DetectorParameters attribute and every value must be settable.

    A typo is otherwise silent: setting adaptiveThreshWinSizeMaxx on a plain Python object just adds
    an attribute OpenCV never reads, so the preset appears to do nothing.
    """
    defaults = cv2.aruco.DetectorParameters()
    for key, value in parameters(body).items():
        if not hasattr(defaults, key):
            raise SystemExit(f"{path}: {where} sets {key!r}, which is not a "
                             f"cv2.aruco.DetectorParameters attribute")
        if not isinstance(value, VALID_TYPES):
            raise SystemExit(f"{path}: {where} sets {key} to {value!r}; "
                             f"detector parameters must be a number or a boolean")


def validate(config, path):
    """Raise SystemExit unless config (the parsed detector_presets.json) is usable.

    Returns the config unchanged, so it can be used inline around json.load().
    """
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: top level must be a JSON object, not {type(config).__name__}")

    for key, kind in (("default", str), ("board_params", dict), ("presets", dict)):
        if key not in config:
            raise SystemExit(f"{path}: missing required key {key!r}")
        if not isinstance(config[key], kind):
            raise SystemExit(f"{path}: {key!r} must be a JSON "
                             f"{'string' if kind is str else 'object'}")

    check_parameters(config["board_params"], "board_params", path)

    presets = config["presets"]
    if not presets:
        raise SystemExit(f"{path}: 'presets' is empty, so there is nothing to calibrate with")
    for name, body in presets.items():
        if not isinstance(body, dict):
            raise SystemExit(f"{path}: preset {name!r} must be a JSON object")
        check_parameters(body, f"preset {name!r}", path)

    if config["default"] not in presets:
        raise SystemExit(f"{path}: default {config['default']!r} is not one of {', '.join(presets)}")

    # board_params describe the printed target rather than the tuning, so overriding one is allowed
    # but nearly always a mistake: markerBorderBits at its default halves the tags found, and nothing
    # raises when it does. Warn once here rather than on every use of the preset.
    for name, body in presets.items():
        for key in parameters(body).keys() & parameters(config["board_params"]).keys():
            print(f"  WARNING: {path}: preset {name!r} overrides board parameter {key} "
                  f"({config['board_params'][key]} -> {body[key]})")

    return config
