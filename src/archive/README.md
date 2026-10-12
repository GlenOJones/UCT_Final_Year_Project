# Archive

Scripts no current experiment or report figure depends on. They are kept for reference and are not maintained:
some may no longer run against the current pipeline or folder layout.

| file | what it was |
|---|---|
| `gray_vs_rgb_calibration.py` | Early (26 Aug) study: recorded luma Y vs grayscale computed from colour, with JPEG controls, on ChArUco frames. Superseded by experiment 1 (`experiments/channels`). |
| `gray_vs_channels.py` | Y vs R, G, B decoded again from the raw YUYV recordings (18 Sep). Superseded by experiment 1, which uses `Calibration.py --channel`. Output in `results/analysis/gray_vs_channels/`. |
| `compare_presets.py` | Compared the `wide` and `opencv_default` channel studies with merged detection (`results/analysis/preset_merge_comparison/`). Imports `run_study` from its old folder, so it no longer runs as is. Experiment 2's `summarise.py` covers the preset comparison. |
| `filtering.py` | Image filters (unsharp, upscale, CLAHE, ...) tried before detection on the 18 Sep frames. Never wired into `Calibration.py`. |
| `pinax_18sep.py` | First Pinax flat-port test on the 18 Sep frames, sweeping the in-air focal length (was `02_Compensation/compensation.py`). Experiment 3 replaces it; the refraction functions it used now live in `experiments/compensation/calibmethods/refraction.py`, which this script imports. |
| `UWARUco/` | Re-implementation of the underwater ArUco detector of Cejka et al. 2019. Still selectable with `Calibration.py --detector uwaruco`, which loads it from here. |
| `calibration_configs/` | JSON configs (`baseline`, `opencv_defaults`, `rational_pruned`) written for a planned `calibrate.py` that was never built; nothing reads them. Detector tuning lives in `01_Calibration/detector_presets.json`. |
