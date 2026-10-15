"""The settings of the depth experiment, from config.json and the command line, shared by run_depth.py and
analyse_depth.py so the two always agree on which recordings, matchers and folders a run used.

Command-line options, all optional, override the config:

    --recordings NAME...      only these recordings of the config
    --frames DIR              a recording not in the config (with --depth-min/--depth-max; --calibration and
                              --name default to the first recording's calibration and the folder name)
    --matchers NAME...        only these matchers of the config
    --out-dir / --work-dir    other result / intermediate folders
    --trial N                 only the first N frames of each recording (after frame_step), in <out_dir>_trial
                              and <work_dir>_trial, to check the output before a full run
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

DEFAULT_CONFIG = os.path.join(HERE, "config.json")


def add_arguments(ap):
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--recordings", nargs="+", help="only these recordings of the config")
    ap.add_argument("--frames", help="frames folder (left/, right/) of a recording not in the config")
    ap.add_argument("--name", help="name for --frames (default: the folder name)")
    ap.add_argument("--calibration", help="stereo calibration for --frames (default: the config's first)")
    ap.add_argument("--depth-min", type=float, help="mm, nearest camera depth kept, for --frames")
    ap.add_argument("--depth-max", type=float, help="mm, furthest camera depth kept, for --frames")
    ap.add_argument("--matchers", nargs="+", help="only these matchers of the config")
    ap.add_argument("--out-dir", help="results folder (default: the config's out_dir)")
    ap.add_argument("--work-dir", help="intermediates folder (default: the config's work_dir)")
    ap.add_argument("--trial", type=int, metavar="N", help="first N frames of each recording only, in *_trial folders")


def resolve(args, ap):
    """The config as this run uses it: recordings, matchers and folders narrowed or replaced by the options."""
    cfg = copy.deepcopy(common.load_config(args.config))
    if args.frames:
        if args.depth_min is None or args.depth_max is None:
            ap.error("--frames needs --depth-min and --depth-max")
        cfg["recordings"] = [{"name": args.name or os.path.basename(os.path.normpath(args.frames)),
                              "frames": os.path.relpath(os.path.abspath(args.frames), common.PROJECT_ROOT),
                              "calibration": args.calibration or cfg["recordings"][0]["calibration"],
                              "depth_min_mm": args.depth_min, "depth_max_mm": args.depth_max}]
    elif args.recordings:
        unknown = set(args.recordings) - {r["name"] for r in cfg["recordings"]}
        if unknown:
            ap.error(f"not in the config: {', '.join(sorted(unknown))}")
        cfg["recordings"] = [r for r in cfg["recordings"] if r["name"] in args.recordings]
    if args.matchers:
        unknown = set(args.matchers) - set(cfg["matchers"])
        if unknown:
            ap.error(f"matchers not in the config: {', '.join(sorted(unknown))} (have {', '.join(cfg['matchers'])})")
        cfg["matchers"] = {k: v for k, v in cfg["matchers"].items() if k in args.matchers}
    cfg["out_dir"] = args.out_dir or cfg["out_dir"]
    cfg["work_dir"] = args.work_dir or cfg["work_dir"]
    cfg["max_frames"] = args.trial
    if args.trial:
        cfg["out_dir"] += "_trial"
        cfg["work_dir"] += "_trial"
    return cfg


def depth_dir(cfg, recording, matcher):
    """depth_frames.py's output folder for one recording and matcher."""
    return os.path.join(common.project_path(cfg["work_dir"]), recording["name"], matcher)


def result_dir(cfg, recording=None):
    out = common.project_path(cfg["out_dir"])
    return os.path.join(out, recording["name"]) if recording else out
