# Experiments

Every result in the thesis comes from one command:

```
src/venv/bin/python src/experiments/reproduce.py <experiment>      # channels | detector_sweep | compensation | pose_estimation | all
```

It starts from the raw frames in `data/`, makes the experiment's pipeline runs, runs its analyses and draws its
figures. Results go to `results/experiments/<experiment>/`, with a `manifest.json` that records the git commit,
library versions, the config, the resolved detector presets and a digest of every input folder.

## Rules that keep results repeatable

1. **The pipeline is fixed.** `src/02_calibration/Calibration.py` is the only code that detects tags and fits
   cameras. Experiments call it with `Calibration.calibrate(argv)`; they never re-implement it. The analysis
   scripts only read what those runs saved (`*_stereo.yaml`, `*_corners.json`).
2. **Every setting is in the config.** Each experiment's `config.json` states every pipeline setting (detector,
   preset, channel, outlier rejection, min tags). `common.calibration_argv()` refuses a config that leaves one
   out. Changing a default in Calibration.py therefore never changes an experiment.
3. **Results come from a commit.** `reproduce.py` refuses to run while `src/` has uncommitted changes.
   `--allow-dirty` overrides that for a trial run, and the manifest then lists the uncommitted files.

## The standard pipeline

`Calibration.py` at its defaults: preset `water_final` (the `default` in `detector_presets.json`), channel
`merge`, outlier rejection on. Each YAML records the arguments it was run with (`metadata/arguments`) and whether
rejection ran (`calibration_board/outlier_rejection`).

## The experiments

| # | experiment | question | config | output (`results/experiments/...`) | in the report |
|---|---|---|---|---|---|
| 1 | `channels` | Which image channel should the detector see? Gray, Y, R, G, B and merged, in air and water, with OpenCV's defaults | `channels/config.json` | `channels/` | ch. 5: `fig_recall_mean_by_medium`, `table_channel_summary`, `fig_channel_overlap_L3`, `fig_air_vs_water_C1_right_60`, `air_vs_water_crops/fig_single_tag_close_far` |
| 2 | `detector_sweep` | Which `DetectorParameters` find the most tags without misreading them, and do they calibrate better? | `detector_sweep/config.json` | `detector_sweep/` (`sweep/`, `calibration/`, `presets/summary.md`, `accuracy/`) | not yet |
| 3 | `compensation` | Which calibration model suits the flat-port underwater cameras: Brown-Conrady, Kannala-Brandt, close-range training, or Pinax refraction from an in-air calibration? | `compensation/config.json`, `compensation/config_pinax_sensitivity.json` | `compensation/` | `compensation/methods_draft.md` |
| 4 | `pose_estimation` | Without reference tags (ice), how well do COLMAP SfM and ICP on SGBM or FoundationStereo depth recover the rig pose of every frame? Scored against the tag poses | `pose_estimation/config.json` | `pose_estimation/<recording>/` (`poses/`, `evaluation/`) | not yet |

### 1. channels
Steps: one `Calibration.py` run per medium and channel (`run_study.py`) → detection, triangulated tag size and
matched-tag RMS (`analyse_study.py`) → figures named in the config's `figures` block (`plot_channel_overlap.py`,
`plot_air_vs_water.py`, `plot_air_water_crops.py`). `variants/` holds the same comparison with preset `wide` and
with outlier rejection on: `reproduce.py channels --config src/experiments/channels/variants/wide.json`.

### 2. detector_sweep
Steps: coordinate-descent search from `opencv_default` on the underwater frames (`sweep_detector.py`), checked
against the preset `detector_presets.json` stores for it (`water_sweep_from_default`; a mismatch is printed and
recorded in the manifest) → every preset calibrated in air and water, compared on tags found and common-set RMS
(`summarise.py`) → {opencv_default, water_final} x {gray, merge} measured on the held-out WaterComparison target
(`accuracy_check.py`). `water_final` is the sweep's result with `polygonalApproxAccuracyRate` set back to 0.03 by
hand; its `_why` in `detector_presets.json` gives the evidence.

### 3. compensation
Steps: the `Calibration.py` runs listed under `pipeline_runs` in each config → `run_comparison.py` for each
config (28 leave-two-recordings-out splits). See `compensation/README.md`. Unit tests:
`src/venv/bin/python -m unittest discover -s src/experiments/compensation/tests -v`.

### 4. pose_estimation
Steps: reference tag poses (`tag_poses.py`) → tag masks (`make_tag_masks.py`) → per-frame camera-frame depth for
each depth source (`depth_frames.py`) → every estimator (`colmap_poses.py`, `icp_poses.py`) → scores against the
tag poses (`evaluate_poses.py`). Steps run as subprocesses with the interpreter each needs (FoundationStereo:
`envs/ffs`). See `pose_estimation/README.md`.

## Adding an experiment

Make `src/experiments/<name>/` with a `config.json` that states every pipeline setting, scripts whose `main(argv)`
reads that config, and a function in `reproduce.py` listing the steps in order. Use `common.calibrate()` for
pipeline runs and `common.write_manifest()` at the end.
