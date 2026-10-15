"""Tune dense_stereo.py's SGBM settings against the tag planes, on frames the depth comparison does not use.

Every candidate setting is run through the pipeline's own per-frame code (depth_frames.frame_points, the same
call depth_frames.py makes, with the same depth range and disparity search), and its disparity is scored in the
tag regions of each frame against the plane of that tag (board_truth.py):

    precision   of the region pixels the matcher gave depth, the share within tolerance_px of the plane
    recall      of all region pixels, the share with depth within tolerance_px
    F1          their harmonic mean; the objective is the mean F1 over the scenes in tune_config.json

The search is coordinate descent from the current defaults: each setting in turn is tried at every candidate
value with the others held, and the best is kept if it gains at least min_gain. Rounds repeat until one changes
nothing (or rounds runs out). Frames are matched in parallel (workers); the seconds per frame of the default and
the winner are then measured again one frame at a time, since parallel runs slow each other down.

    src/venv/bin/python src/experiments/depth_matchers/tune_sgbm.py
    src/venv/bin/python src/experiments/depth_matchers/tune_sgbm.py --resume      # keep evaluations already logged

Writes to <out_dir>/: evaluations.csv (every setting tried, per-scene scores), tuning.json (default, winner,
the search path, timing, and the depth_frames.py arguments of the winner for config.json).
"""
import argparse
import csv
import datetime
import json
import multiprocessing
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SRC, "03_Reconstruction"))
import board_truth  # noqa: E402
import common  # noqa: E402
import depth_frames  # noqa: E402
import tag_poses  # noqa: E402
from dense_stereo import Rectifier  # noqa: E402
from stereo_rig import find_frame_pairs, load_rig  # noqa: E402

# SGBM settings and their depth_frames.py / dense_stereo.py options.
OPTIONS = {"block_size": "--block-size", "uniqueness": "--uniqueness", "sgbm_mode": "--sgbm-mode", "p1": "--p1",
           "p2": "--p2", "speckle_window": "--speckle-window", "speckle_range": "--speckle-range",
           "disp12_max_diff": "--disp12-max-diff", "prefilter_cap": "--prefilter-cap", "sgbm_scale": "--sgbm-scale",
           "median_filter": "--median-filter"}


def sgbm_argv(settings):
    argv = ["--matcher", "sgbm"]
    for key, option in OPTIONS.items():
        argv += [option, str(settings[key])]
    return argv


def key_of(settings):
    return json.dumps({k: settings[k] for k in OPTIONS}, sort_keys=True)


def valid(settings):
    """Settings SGBM accepts and that make sense: odd block, P2 above P1."""
    return settings["block_size"] % 2 == 1 and settings["p2"] > settings["p1"]


# ====== GROUND TRUTH ======
def build_truth(cfg, rectifier):
    """[(set name, frame name, left path, right path, truth path)] for every tuning frame with a usable tag, the
    truth cached as .npz in the work folder (tag detection is the slow part and never changes)."""
    detector = tag_poses.make_detector()
    frames = []
    for spec in cfg["sets"]:
        folder = os.path.join(common.project_path(cfg["work_dir"]), "truth", spec["name"])
        os.makedirs(folder, exist_ok=True)
        pairs = find_frame_pairs(common.project_path(spec["frames"]))[spec["offset"]::spec["step"]]
        used = 0
        for name, left, right in pairs:
            path = os.path.join(folder, os.path.splitext(name)[0] + ".npz")
            if not os.path.exists(path):
                truth = board_truth.frame_truth(rectifier, detector, left, right)
                if truth is None:
                    np.savez_compressed(path, disparity=np.zeros(0, np.float32), tags=np.zeros(0, int))
                else:
                    np.savez_compressed(path, disparity=truth["disparity"], tags=np.array(truth["tags"]))
            if np.load(path)["disparity"].size:
                frames.append((spec["name"], name, left, right, path))
                used += 1
        print(f"  {spec['name']}: {used} of {len(pairs)} frames have a usable tag")
    return frames


# ====== EVALUATION (worker processes) ======
_worker = {}


def init_worker(calibration, sets):
    cv2.setNumThreads(1)
    _worker["rectifier"] = Rectifier(load_rig(calibration))
    _worker["sets"] = {s["name"]: s for s in sets}


def evaluate_frame(job):
    """One frame with one setting: (set name, region, with depth, good, sum and count of errors, error p50 sample)."""
    settings, (set_name, _, left, right, truth_path) = job
    spec = _worker["sets"][set_name]
    args = depth_frames.parse_args(["--frames-dir", "-", "--calibration", "-", "--out-dir", "-",
                                    "--depth-min", str(spec["depth_min_mm"]), "--depth-max", str(spec["depth_max_mm"]),
                                    *sgbm_argv(settings)])
    rectifier = _worker["rectifier"]
    search = depth_frames.disparity_range(rectifier, args.depth_min, args.depth_max)
    _, _, disparity, seconds = depth_frames.frame_points(rectifier, None, left, right, None, args, search)
    truth_disparity = np.load(truth_path)["disparity"]
    counts = board_truth.score(disparity, {"region": truth_disparity > 0, "disparity": truth_disparity})
    errors = counts.pop("abs_error_px")
    return set_name, counts, errors[::25], seconds   # every 25th error: enough for the percentiles


def evaluate(pool, settings, frames, cfg):
    """Scores of one setting over every tuning frame: per set, per scene and the objective."""
    results = pool.map(evaluate_frame, [(settings, f) for f in frames])
    per_set = {}
    for spec in cfg["sets"]:
        rows = [(c, e) for name, c, e, _ in results if name == spec["name"]]
        per_set[spec["name"]] = board_truth.summarise([{**c, "abs_error_px": e} for c, e in rows], cfg["tolerance_px"])
    scenes = {scene: float(np.mean([per_set[s]["f1"] for s in names])) for scene, names in cfg["scenes"].items()}
    return {"settings": {k: settings[k] for k in OPTIONS}, "objective": float(np.mean(list(scenes.values()))),
            "scenes": scenes, "sets": per_set,
            "parallel_seconds_per_frame": float(np.median([r[3] for r in results]))}


# ====== SEARCH ======
def write_log(path, evaluations, cfg):
    sets = [s["name"] for s in cfg["sets"]]
    fields = list(OPTIONS) + ["objective"] + [f"f1_{s}" for s in cfg["scenes"]] + \
        [f"{m}_{s}" for s in sets for m in ("precision", "recall", "median_error_px")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for e in evaluations.values():
            row = {**e["settings"], "objective": round(e["objective"], 4)}
            row.update({f"f1_{s}": round(v, 4) for s, v in e["scenes"].items()})
            for s in sets:
                for m in ("precision", "recall", "median_error_px"):
                    value = e["sets"][s][m]
                    row[f"{m}_{s}"] = None if value is None else round(value, 4)
            writer.writerow(row)


def search(pool, frames, cfg, evaluations, log_path):
    """Coordinate descent from cfg["start"]; returns (best evaluation, path of accepted changes)."""
    def score(settings):
        key = key_of(settings)
        if key not in evaluations:
            start = time.time()
            evaluations[key] = evaluate(pool, settings, frames, cfg)
            e = evaluations[key]
            print(f"    {json.dumps({k: settings[k] for k in OPTIONS if settings[k] != cfg['start'][k]}) or 'defaults'}: "
                  f"F1 {e['objective']:.4f} ({', '.join(f'{s} {v:.3f}' for s, v in e['scenes'].items())}) "
                  f"[{time.time() - start:.0f} s]", flush=True)
            write_log(log_path, evaluations, cfg)
        return evaluations[key]

    best = dict(cfg["start"])
    best_score = score(best)
    path = [{"round": 0, "change": "start", "objective": best_score["objective"]}]
    for round_index in range(1, cfg["rounds"] + 1):
        print(f"\n  round {round_index}: from F1 {best_score['objective']:.4f}")
        changed = False
        for name, values in cfg["candidates"].items():
            tried = []
            for value in values:
                candidate = {**best, name: value}
                if valid(candidate):
                    tried.append((score(candidate)["objective"], value))
            top, value = max(tried)
            if value != best[name] and top >= best_score["objective"] + cfg["min_gain"]:
                print(f"  -> {name} {best[name]} -> {value}: F1 {best_score['objective']:.4f} -> {top:.4f}")
                path.append({"round": round_index, "change": f"{name} {best[name]} -> {value}", "objective": top})
                best = {**best, name: value}
                best_score = score(best)
                changed = True
        if not changed:
            print("  no change this round: done")
            break
    return best_score, path


def timing(cfg, frames, settings_list):
    """Median seconds per frame of each setting, one frame at a time in this process."""
    rectifier = Rectifier(load_rig(common.project_path(cfg["calibration"])))
    sets = {s["name"]: s for s in cfg["sets"]}
    picked = frames[:: max(1, len(frames) // cfg["timing_frames"])][:cfg["timing_frames"]]
    out = []
    for settings in settings_list:
        seconds = []
        for set_name, _, left, right, _ in picked:
            spec = sets[set_name]
            args = depth_frames.parse_args(["--frames-dir", "-", "--calibration", "-", "--out-dir", "-",
                                            "--depth-min", str(spec["depth_min_mm"]), "--depth-max",
                                            str(spec["depth_max_mm"]), *sgbm_argv(settings)])
            search_range = depth_frames.disparity_range(rectifier, args.depth_min, args.depth_max)
            seconds.append(depth_frames.frame_points(rectifier, None, left, right, None, args, search_range)[3])
        out.append(float(np.median(seconds)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "tune_config.json"))
    ap.add_argument("--resume", action="store_true", help="keep the evaluations already in evaluations.json")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    out = common.project_path(cfg["out_dir"])
    os.makedirs(out, exist_ok=True)
    cache_path = os.path.join(common.project_path(cfg["work_dir"]), "evaluations.json")
    evaluations = {}
    if args.resume and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            evaluations = json.load(fh)
        print(f"resuming with {len(evaluations)} evaluations")

    calibration = common.project_path(cfg["calibration"])
    print("ground truth (tag planes):")
    frames = build_truth(cfg, Rectifier(load_rig(calibration)))
    log_path = os.path.join(out, "evaluations.csv")
    start = time.time()
    with multiprocessing.get_context("forkserver").Pool(cfg["workers"], init_worker, (calibration, cfg["sets"])) as pool:
        try:
            best, path = search(pool, frames, cfg, evaluations, log_path)
        finally:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(evaluations, fh)
    default = evaluations[key_of(cfg["start"])]
    print(f"\nsearch: {len(evaluations)} settings in {(time.time() - start) / 60:.0f} min; timing default and winner...")
    default_s, best_s = timing(cfg, frames, [cfg["start"], best["settings"]])

    record = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "config": common.rel(args.config),
              "frames": {s["name"]: sum(f[0] == s["name"] for f in frames) for s in cfg["sets"]},
              "default": {**default, "seconds_per_frame": default_s},
              "best": {**best, "seconds_per_frame": best_s}, "path": path,
              "evaluations": len(evaluations),
              "depth_frames_argv": sgbm_argv(best["settings"]) + ["--voxel", "1"]}
    with open(os.path.join(out, "tuning.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    for label, e, s in (("default", default, default_s), ("tuned", best, best_s)):
        print(f"{label:8s} F1 {e['objective']:.4f}  " + "  ".join(
            f"{n}: P {v['precision']:.3f} R {v['recall']:.3f} med {v['median_error_px']:.2f} px"
            for n, v in e["sets"].items()) + f"  {s:.2f} s/frame")
    print(f"winner's arguments: {' '.join(record['depth_frames_argv'])}\nresults in {common.rel(out)}/")


if __name__ == "__main__":
    main()
