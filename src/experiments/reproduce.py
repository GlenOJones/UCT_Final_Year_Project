"""Reproduce an experiment's results from the raw frames, end to end.

    src/venv/bin/python src/experiments/reproduce.py channels          # experiment 1: colour channel comparison
    src/venv/bin/python src/experiments/reproduce.py detector_sweep    # experiment 2: OpenCV parameter sweep
    src/venv/bin/python src/experiments/reproduce.py compensation      # experiment 3: underwater calibration methods
                                                                       # (+ the methods report PDF)
    src/venv/bin/python src/experiments/reproduce.py pose_estimation   # experiment 4: markerless pose estimation
    src/venv/bin/python src/experiments/reproduce.py depth_matchers    # experiment 5: depth map generation
    src/venv/bin/python src/experiments/reproduce.py pose_comparison   # experiment 4 on experiment 5's recordings
    src/venv/bin/python src/experiments/reproduce.py pose_ice          # pose on the ice passes (shade, sun pass 2)
    src/venv/bin/python src/experiments/reproduce.py fusion            # COLMAP vs TSDF (FS, SGBM) on S30
    src/venv/bin/python src/experiments/reproduce.py all

Each experiment's settings are in src/experiments/<experiment>/config.json and its results go to
results/experiments/<experiment>/, with a manifest.json saying what they were made from. The steps of each
experiment are the functions below, in order: the pipeline runs first (ordinary Calibration.py runs, every
setting stated in the config), then the analyses and figures, which only read what those runs saved.

The run refuses to start while src/ has uncommitted changes, because the results could then not be traced
to a commit. --allow-dirty overrides that for trial runs; the manifest records the changes.

Options:
    --config PATH      another config for a single experiment (e.g. channels/variants/wide.json)
    --skip-existing    keep pipeline runs (and the sweep's evaluation cache) that already exist
    --allow-dirty      run even though src/ has uncommitted changes
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _experiment in ("channels", "detector_sweep", "compensation", "pose_estimation", "depth_matchers", "fusion"):
    sys.path.insert(0, os.path.join(HERE, _experiment))
sys.path.insert(0, HERE)
import common  # noqa: E402
from common import Calibration  # noqa: E402


# ====== EXPERIMENT 1: COLOUR CHANNELS ======
def channels(config, skip_existing):
    """One calibration per medium and channel; detection, accuracy and corner-precision analysis; figures."""
    import analyse_study
    import plot_air_vs_water
    import plot_air_water_crops
    import plot_channel_overlap
    import run_study

    config = config or os.path.join(HERE, "channels", "config.json")
    study = run_study.load_study(config)
    run_study.run(study, skip_existing)
    analyse_study.main(["--study", config])
    figures = study.get("figures", {})
    for spec in figures.get("channel_overlap", []):
        argv = ["--study", config, "--name", spec["name"], "--channels", *spec["channels"],
                "--images", *spec["images"], "--width", str(spec["width"])]
        plot_channel_overlap.main(argv + (["--legend-right"] if spec.get("legend_right") else []))
    if figures.get("air_vs_water"):
        plot_air_vs_water.main(["--study", config])
    if figures.get("air_vs_water_crops"):
        plot_air_water_crops.main(["--study", config])


# ====== EXPERIMENT 2: DETECTOR PARAMETER SWEEP ======
def sweep_check(best, expected_name):
    """Compare the sweep's best settings with the preset detector_presets.json stores for it."""
    expected = Calibration.load_presets().presets[expected_name]
    differences = {k: {"sweep": best.get(k), "preset": expected.get(k)}
                   for k in sorted(set(best) | set(expected)) if best.get(k) != expected.get(k)}
    if differences:
        print(f"\nSWEEP DOES NOT MATCH preset {expected_name!r} in detector_presets.json:")
        for k, d in differences.items():
            print(f"  {k}: sweep {d['sweep']}, preset {d['preset']}")
    else:
        print(f"\nsweep result matches preset {expected_name!r} in detector_presets.json")
    return {"expected_preset": expected_name, "matches": not differences, "differences": differences, "best": best}


def detector_sweep(config, skip_existing):
    """The coordinate-descent search, then every preset calibrated in both media, then the accuracy check."""
    import accuracy_check
    import summarise
    import sweep_detector

    config = config or os.path.join(HERE, "detector_sweep", "config.json")
    cfg = common.load_config(config)
    out_dir, sweep = common.project_path(cfg["out_dir"]), cfg["sweep"]

    best, score, log = sweep_detector.main(
        ["--frames-dir", common.project_path(sweep["frames"]), "--out-dir", os.path.join(out_dir, "sweep"),
         "--rounds", str(sweep["rounds"]), "--min-tags", str(sweep["min_tags"]),
         "--max-bad-rate", str(sweep["max_bad_rate"])] + ([] if skip_existing else ["--fresh"]))
    check = sweep_check(best, sweep["expected_preset"])

    # Preset comparison (every preset, every medium, one channel) plus the accuracy setups (one medium).
    jobs = [(m, p, cfg["presets"]["channel"]) for m in cfg["media"] for p in cfg["presets"]["names"]]
    jobs += [(cfg["accuracy"]["medium"], p, c) for p, c in cfg["accuracy"]["setups"]]
    runs = []
    for medium, preset, channel in dict.fromkeys(jobs):
        settings = {**cfg["calibration"], "preset": preset, "channel": channel}
        record = common.calibrate(settings, common.project_path(cfg["media"][medium]),
                                  os.path.join(out_dir, "calibration", medium), skip_existing)
        runs.append({"medium": medium, "preset": preset, "channel": channel, **record})

    summarise.main(["--config", config])
    accuracy_check.main(["--config", config])
    inputs = [common.project_path(p) for p in cfg["media"].values()] + [common.project_path(cfg["accuracy"]["target"])]
    common.write_manifest(out_dir, "detector_sweep", [config], inputs, [p for _, p, _ in jobs], runs,
                          {"sweep": {**check, "score": score.__dict__, "log": common.rel(log)}})
    if not check["matches"]:
        print("\nNOTE: the sweep no longer lands on the stored preset (see above and manifest.json); the preset "
              "comparison still used the stored presets.")


# ====== EXPERIMENT 3: UNDERWATER CALIBRATION METHODS ======
def compensation(config, skip_existing):
    """The Calibration.py runs the comparison reads, then the held-out comparison for each config, then the
    methods report (PDF) for each config that asks for one."""
    import make_report
    import run_comparison

    configs = [config] if config else [os.path.join(HERE, "compensation", "config.json"),
                                       os.path.join(HERE, "compensation", "config_pinax_sensitivity.json"),
                                       os.path.join(HERE, "compensation", "config_full.json")]
    runs, seen, presets, inputs, out_dirs = [], set(), [], [], []
    for path in configs:
        cfg = common.load_config(path)
        block = cfg["pipeline_runs"]
        for spec in block["runs"]:
            settings = {**block["calibration"], "preset": spec["preset"], "channel": spec["channel"]}
            frames = common.project_path(spec["frames"])
            key = (frames, spec["preset"], spec["channel"])
            presets.append(spec["preset"])
            inputs.append(frames)
            if key in seen:
                continue
            seen.add(key)
            record = common.calibrate(settings, frames, os.path.join(common.project_path(block["out_dir"]),
                                                                     spec["medium"]), skip_existing)
            runs.append({"medium": spec["medium"], **record})
        out_dirs.append(common.project_path(cfg["output"]))
    for path in configs:
        run_comparison.main(["--config", path])
        if "report" in common.load_config(path):
            make_report.main(["--config", path])
    # One manifest for the experiment, beside the per-config output folders.
    common.write_manifest(os.path.commonpath(out_dirs) if len(out_dirs) > 1 else out_dirs[0], "compensation",
                          configs, inputs, presets, runs)


# ====== EXPERIMENT 4: MARKERLESS POSE ESTIMATION ======
def pose_estimation(config, skip_existing):
    """Reference tag poses, tag masks, per-frame depth and every markerless estimator (run_poses.py), then their
    scores against the tag poses (evaluate_poses.py)."""
    import evaluate_poses
    import run_poses

    config = config or os.path.join(HERE, "pose_estimation", "config.json")
    cfg = common.load_config(config)
    runs = run_poses.main(["--config", config] + (["--skip-existing"] if skip_existing else []))
    evaluate_poses.main(["--config", config])
    common.write_manifest(common.project_path(cfg["out_dir"]), "pose_estimation", [config],
                          [common.project_path(r["frames"]) for r in cfg["recordings"]], [], runs)


# ====== EXPERIMENT 5: DEPTH MAP GENERATION ======
def depth_matchers(config, skip_existing):
    """depth_frames.py for every recording and matcher (run_depth.py), then the table and figures (analyse_depth.py)."""
    import analyse_depth
    import run_depth

    config = config or os.path.join(HERE, "depth_matchers", "config.json")
    cfg = common.load_config(config)
    runs = run_depth.main(["--config", config] + (["--skip-existing"] if skip_existing else []))
    analyse_depth.main(["--config", config])
    common.write_manifest(common.project_path(cfg["out_dir"]), "depth_matchers", [config],
                          [common.project_path(r["frames"]) for r in cfg["recordings"]], [], runs)


def pose_comparison(config, skip_existing):
    """Experiment 4 on the depth-test recordings, one mode per method (pose_estimation/config_comparison.json):
    the pipeline runs (run_poses.py), grip and detection images (detection_figures.py), then the table
    (compare_poses.py)."""
    import compare_poses
    import detection_figures
    import run_poses

    config = config or os.path.join(HERE, "pose_estimation", "config_comparison.json")
    cfg = common.load_config(config)
    runs = run_poses.main(["--config", config] + (["--skip-existing"] if skip_existing else []))
    detection_figures.main(["--config", config])
    compare_poses.main(["--config", config])
    common.write_manifest(common.project_path(cfg["out_dir"]), "pose_comparison", [config],
                          [common.project_path(r["frames"]) for r in cfg["recordings"]], [], runs)


def pose_ice(config, skip_existing):
    """Pose estimation on the two ice passes (pose_estimation/config_ice.json): AprilGrid, COLMAP, ICP on
    FoundationStereo and on AquaStereo depth, with restarts of failed runs on the sun pass. The restart rule's
    unit tests run first, so a broken rule stops the run before any compute."""
    import unittest
    import compare_poses
    import detection_figures
    import run_poses

    tests = unittest.defaultTestLoader.discover(os.path.join(HERE, "pose_estimation", "tests"))
    if not unittest.TextTestRunner(verbosity=1).run(tests).wasSuccessful():
        raise SystemExit("restart tests failed")
    config = config or os.path.join(HERE, "pose_estimation", "config_ice.json")
    cfg = common.load_config(config)
    runs = run_poses.main(["--config", config] + (["--skip-existing"] if skip_existing else []))
    detection_figures.main(["--config", config])
    compare_poses.main(["--config", config])
    common.write_manifest(common.project_path(cfg["out_dir"]), "pose_ice", [config],
                          [common.project_path(r["frames"]) for r in cfg["recordings"]], [], runs)


def fusion(config, skip_existing):
    """Fusion comparison on S30 (fusion/config.json): full COLMAP against TSDF on FoundationStereo and on SGBM
    depth, all at the COLMAP poses and on the pose comparison's inputs (run_fusion.py), then CAD scores and
    figures (compare_fusion.py). Needs pose_comparison's outputs."""
    import compare_fusion
    import run_fusion

    config = config or os.path.join(HERE, "fusion", "config.json")
    cfg = common.load_config(config)
    runs = run_fusion.main(["--config", config] + (["--skip-existing"] if skip_existing else []))
    compare_fusion.main(["--config", config])
    rec = cfg["recording"]
    inputs = [common.project_path(rec["frames"]), common.project_path(rec["tag_masks"])]
    inputs += [common.project_path(d) for d in rec["depth"].values()]
    common.write_manifest(os.path.join(common.project_path(cfg["out_dir"]), rec["name"]), "fusion", [config],
                          inputs, [], runs, {"poses": common.rel(common.project_path(rec["poses"]))})


EXPERIMENTS = {"channels": channels, "detector_sweep": detector_sweep, "compensation": compensation,
               "pose_estimation": pose_estimation, "depth_matchers": depth_matchers,
               "pose_comparison": pose_comparison, "pose_ice": pose_ice, "fusion": fusion}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--config", help="another config, for a single experiment")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep pipeline runs (and the sweep's evaluation cache) that already exist")
    ap.add_argument("--allow-dirty", action="store_true", help="run even though src/ has uncommitted changes")
    args = ap.parse_args(argv)
    if args.config and args.experiment == "all":
        ap.error("--config is for a single experiment")
    common.require_clean(args.allow_dirty)
    for name in (EXPERIMENTS if args.experiment == "all" else [args.experiment]):
        print(f"\n######## {name} ########")
        EXPERIMENTS[name](args.config and os.path.abspath(args.config), args.skip_existing)


if __name__ == "__main__":
    main()
