# Experiment 5: depth map generation

**Question:** how much of each frame do SGBM, AquaStereo and FoundationStereo turn into depth, and how fast?

**Run:** `src/venv/bin/python src/experiments/reproduce.py depth_matchers` (add `--skip-existing` to keep finished
runs). Settings: `config.json`. Results: `results/experiments/depth_matchers/`. Bulky intermediates (points and
disparity per frame): `work/experiments/depth_matchers/`, safe to delete.

**Check first:** `run_depth.py --trial 3` then `analyse_depth.py --trial 3` run only the first 3 frames of each
recording, into `results/experiments/depth_matchers_trial/` and `work/experiments/depth_matchers_trial/`.

## Design

- **Depth:** `depth_frames.py`, the pipeline's per-frame depth without poses, on every `frame_step`th pair (set per recording, or the top-level default): the
  same rectification and matchers as `dense_stereo.py`, limited only by a camera depth range per recording (which
  also sets SGBM's disparity search). `--save-disparity` keeps each frame's disparity for the figures.
- **Metrics** (`summary.md`, `summary.csv`), medians over frames:
  - seconds per frame, for the matcher alone and for the whole frame. The median leaves out a learned matcher's
    first-frame warm-up.
  - pixels with depth within the range, and as a share of the image.
  - points after merging into 1 mm voxels, which is what reconstruction uses.
  Points count **coverage, not correctness**. SGBM's scattered speckle on featureless water counts as much as a
  real surface. Read the numbers alongside the depth images.
- **Figures** (`analyse_depth.py`, which only reads saved output):
  - `<recording>/<matcher>/depth/<frame>.jpg`: every frame, rectified left image | depth map.
  - `<recording>/depth_compare.jpg`: one frame, every matcher's depth map, stacked.
  - `<recording>/matches.jpg` (and `matches_<matcher>.jpg`): one frame, left above right, a line from each pixel
    of a fixed grid to where the matcher matched it. The pixels are the same for every matcher; a grey cross
    marks no depth. A dense matcher has no keypoints, so the grid stands in for them.

## Another recording or matcher

Add it to `recordings` or `matchers` in `config.json`. For a one-off run without editing the config:

    src/venv/bin/python src/experiments/depth_matchers/run_depth.py --frames <frames folder> --depth-min 400 \
        --depth-max 1400 --matchers sgbm foundationstereo --out-dir results/experiments/depth_x --work-dir work/experiments/depth_x

then `analyse_depth.py` with the same options.
