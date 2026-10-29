"""Run the channel comparison: one Calibration.py run per medium and channel in the config."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
from common import Calibration, PROJECT_ROOT  # noqa: E402,F401  (re-exported for the analysis scripts)

STUDY_FILE = os.path.join(HERE, "config.json")


def load_study(path=STUDY_FILE):
    """The config with its paths made absolute."""
    study = common.load_config(path)
    study["config_path"] = os.path.abspath(path)
    study["media"] = {m: common.project_path(p) for m, p in study["media"].items()}
    study["test_target"]["frames"] = common.project_path(study["test_target"]["frames"])
    study["out_dir"] = common.project_path(study["out_dir"])
    return study


def add_study_argument(parser):
    parser.add_argument("--study", default=STUDY_FILE,
                        help="experiment config (default config.json here); e.g. variants/wide.json")


def settings(study, channel):
    return {**study["calibration"], "channel": channel}


def out_dir(study, medium):
    return os.path.join(study["out_dir"], medium)


def yaml_path(study, medium, channel):
    return common.calibration_paths(settings(study, channel), study["media"][medium], out_dir(study, medium))[0]


def corners_path(study, medium, channel):
    return common.calibration_paths(settings(study, channel), study["media"][medium], out_dir(study, medium))[1]


def run(study, skip_existing=False):
    """Every calibration of the study, then the manifest."""
    runs = []
    for medium in study["media"]:
        for channel in study["channels"]:
            record = common.calibrate(settings(study, channel), study["media"][medium], out_dir(study, medium),
                                      skip_existing)
            runs.append({"medium": medium, "channel": channel, **record})
    common.write_manifest(study["out_dir"], "channels", [study["config_path"]],
                          [*study["media"].values(), study["test_target"]["frames"]],
                          [study["calibration"]["preset"]], runs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_study_argument(ap)
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep runs whose YAML and corners file already exist instead of redoing them")
    args = ap.parse_args(argv)
    run(load_study(args.study), args.skip_existing)


if __name__ == "__main__":
    main()
