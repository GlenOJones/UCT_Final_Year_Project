# Calibration method comparison

Compares ways of calibrating the StellarHD + Aquagon cameras underwater, on held-out data only:

| key | technique | model | OpenCV fit |
|---|---|---|---|
| `bc` | baseline | Brown-Conrady: fx fy cx cy, k1 k2 k3, p1 p2 (9) | `cv2.calibrateCamera` |
| `kb` | wide-angle model | Kannala-Brandt: fx fy cx cy, k1..k4 (8) | `cv2.fisheye.calibrate` |
| `bc_close` | distance transfer | Brown-Conrady trained on boards 200-800 mm away only | `cv2.calibrateCamera` |
| `pinax` | flat-port physics | in-air Brown-Conrady calibration + Snell refraction; nothing fitted in water | in air: `cv2.calibrateCamera` (via `Calibration.py`) |

Input is a `Calibration.py` run (`*_stereo.yaml` + `*_corners.json`); Pinax also reads an in-air `Calibration.py`
run of the same rig (`results/experiments/compensation/calibration/Air/Air_water_final_stereo.yaml`). Both runs are
listed under `pipeline_runs` in the config, and `reproduce.py compensation` makes them before the comparison. Nothing is re-detected: each frame is
reduced to exactly the images and tags that run kept after its own outlier rejection. Both files are only read.

## Use (from the project root)

```
src/venv/bin/python src/experiments/reproduce.py compensation     # pipeline runs + both configs, from raw frames
src/venv/bin/python src/experiments/compensation/run_comparison.py            # one config, once the runs exist
src/venv/bin/python src/experiments/compensation/run_comparison.py --config src/experiments/compensation/config_pinax_sensitivity.json
src/venv/bin/python -m unittest discover -s src/experiments/compensation/tests -v
```

## Full comparison and methods report (`config_full.json`)

The five tests, numbered as in the report, plus two checks, all on the same 28 splits and detections. The
report shows the left (reference) camera only (`report_cameras`); both cameras are still fitted and scored, and
the stereo results use both:

| key | test | model |
|---|---|---|
| `air` | T1 | in-air calibration (K, D and stereo R, T) used unchanged underwater (`models.AirDirect`) |
| `pinax` | T2 | Pinax, n = 1.333 |
| `bc` | T3 | underwater Brown-Conrady |
| `kb` | T4 | underwater Kannala-Brandt |
| `bc_axial` | T5 | Brown-Conrady + axial ray shift dz(r) = a1 r^2 + a2 r^4, a non-single-viewpoint model (`models.BrownConradyAxial`) |
| `bc_close`, `bc_axial_close` | check | T3 and T5 trained on boards 200-800 mm only (single-viewpoint check) |

On top of the outputs listed below, this config also adds:
- **`target.json`, `target_tags.csv`, `fig_target(_fitted)`:** every technique (fitted on all recordings) triangulates the same detected WaterComparison tags (145 mm, labels 50-250 cm). The detector is `single_viewpoint.py`'s: CLAHE + contour refinement.
- **`fig_quiver(_fitted)`:** mean held-out residual vectors on a 16 x 12 grid.
- **`covariance.json`, `fig_correlation`:** intrinsic covariance sigma^2 (J^T J)^-1, with the board poses eliminated (`calibmethods/covariance.py`; matches `calibrateCameraExtended`). The in-air fit's covariance is computed on the Air run.
- **`report/methods_report.pdf`:** built by `make_report.py` from the saved files (pdflatex). It covers the maths, how to run it, every table and figure, and the hand-written discussion in `report_discussion.tex`.

```
src/venv/bin/python src/experiments/compensation/run_comparison.py --config src/experiments/compensation/config_full.json   # ~20 min
src/venv/bin/python src/experiments/compensation/make_report.py                                                           # ~10 s
src/venv/bin/python src/experiments/compensation/run_comparison.py --config src/experiments/compensation/config_full.json --figures-only   # redraw figures, no fitting
src/venv/bin/python src/experiments/compensation/run_comparison.py --config src/experiments/compensation/config_full.json --max-splits 2 --out /tmp/trial   # code-path trial
```
`reproduce.py compensation` runs all three configs and builds the report.

`config_full_opencv_default.json` is the detector check. It is the same comparison, but every detection uses the
untuned `opencv_default` preset instead of the sweep-chosen `water_final`, because the sweep scored candidates
by running `calibrateCamera`, which could favour Brown-Conrady. It has its own calibrations
(`results/experiments/compensation/calibration_opencv_default/`), output (`.../full_comparison_opencv_default/`)
and discussion (`report_discussion_opencv_default.tex`). Run it with
`src/venv/bin/python src/experiments/reproduce.py compensation --config src/experiments/compensation/config_full_opencv_default.json`.

Other configs:
- `config_axial.json` is the earlier axial-only benchmark.
- `single_viewpoint.py` + `config_single_viewpoint.json` is the stand-alone tag-size-vs-distance test of the `water_final` calibration.

`config_pinax_sensitivity.json` reruns Pinax with sea water (n = 1.339) and with the `wide` in-air calibration
(output `results/experiments/compensation/water_final_pinax_sensitivity/`).
Settings (input run, techniques, distance bands, zone radii, grid, scale-bar lengths) are in `config.json`.
Output goes to `results/experiments/compensation/water_final/`:

| file | contents |
|---|---|
| `summary.md`, `summary.csv` | every metric per technique and camera: mean [2.5, 97.5 percentile] over splits |
| `comparisons.csv` | paired difference of every two techniques per metric, with a within-noise flag |
| `params/<technique>.json` | fitted parameters per split and on all recordings, with training frames |
| `points.npz`, `bars.npz` | held-out residual rows and scale-bar rows; every number above comes from these |
| `fig_heatmap` | mean residual per image cell + mean residual vectors, one colour scale (log with Pinax) and arrow gain |
| `fig_heatmap_fitted` | the same for the three fitted models only, on a linear scale of their own |
| `fig_pinax_mapping` | underwater ray angle per pixel, Pinax vs the in-water calibration |
| `fig_radial`, `fig_distance` | RMS vs normalised radius and vs board distance, band = range over splits |
| `fig_scale` | triangulated length error vs distance |
| `fig_summary` | headline metrics with intervals |
| `run_info.json` | inputs, versions, splits, settings |

## How it works

- **Splits** (`splits.py`): every combination of 2 held-out recordings out of 8 (28 splits). Recordings, not
  frames, are held out because each recording is one sweep at one tilt and its frames are near-duplicates.
- **Held-out scoring** (`evaluate.py`): each held-out frame is posed with the technique's intrinsics fixed,
  minimising pixel error (`cv2.solvePnP`; `cv2.fisheye.solvePnP` + pixel-space LM), and its corners reprojected.
- **Metrics** (`metrics.py`, tested in `tests/test_metrics.py`): RMS; centre/mid/outer zones on the radius
  normalised to the farthest image corner, and the outer/centre and mid/centre ratios; binned profiles; image
  grid statistics with cells under 5 points greyed out.
- **Scale** (`evaluate.scale_bars`): stereo extrinsics fitted on the training pairs with intrinsics fixed, held-out
  pairs triangulated, and every corner-to-corner distance of 250-450 mm compared with the board.
- **Noise** (`stats.py`): a difference between two techniques is within noise when the 2.5-97.5 percentile range
  of their per-split difference contains 0.

`cv2.fisheye.stereoCalibrate` (OpenCV 5.0.0) fails unless every pair has the same number of points, so the
Kannala-Brandt extrinsics come from `models.stereo_lm`, the same pixel-error fit; on Brown-Conrady it matches
`cv2.stereoCalibrate` (tested, and to 1e-7 px on the real data).

**Pinax** (`models.Pinax`) uses the in-air calibration to turn a pixel into its ray in air, refracts it into water
with Snell's law (`02_Compensation/compensation.refract_radial`), and projects the reverse way. The rays are taken
from Pinax's single virtual centre; its offset along the axis (set by the lens-to-port distance and port thickness,
which have no physical value for a water-contact lens) is a constant shift every board pose absorbs.
