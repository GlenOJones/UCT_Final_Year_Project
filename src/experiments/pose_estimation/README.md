# Experiment 4: markerless pose estimation

**Question:** without the reference tags (ice has none), how well can the rig's pose in every frame be
recovered from the images alone, and with which method?

**Run:** `src/venv/bin/python src/experiments/reproduce.py pose_estimation` (add `--skip-existing` to keep
finished steps). Settings: `config.json`. Results: `results/experiments/pose_estimation/<recording>/`. Bulky
intermediates (depth, COLMAP workspace): `work/experiments/pose_estimation/<recording>/`, safe to delete.

## Design

- **Reference:** `tag_poses.py` on every frame of the recording. The tag poses are the yardstick, not the truth;
  their layout-fit residual (about 0.3 mm on 3DPrintFinal s30) is the floor below which differences mean nothing.
- **Tags hidden:** `make_tag_masks.py` masks every tag, and its white surround, in every image the estimators see.
  A tag is masked where it is detected in that image and wherever the tag layout projects it in a tag-posed frame,
  which also catches tags cut off at the image edge. Frames with neither stay unmasked (17 of 344 on s30).
- **Same frames for everyone:** every `frame_step`th frame pair of the recording, posed by tags or not.
- **Estimators** (pipeline scripts in `src/03_Reconstruction/`, used unchanged if one is adopted):
  - `colmap_poses.py`: COLMAP SfM, intrinsics and stereo rig fixed (the stages of `colmap_mvs.py`), with SIFT kept
    off the masked pixels.
  - `icp_poses.py` on the depth from `depth_frames.py`, which is made in each frame's own camera frame with
    only a depth range (no tag-based crop), once with SGBM and once with FoundationStereo:
    frame-to-frame or frame-to-local-map, point-to-plane or coloured ICP.
- **Scoring** (`evaluate_poses.py`): one rigid fit of each trajectory onto the tag poses, then the absolute
  trajectory error (ATE, mm and deg per frame), the relative error of the motion over every 50 mm of travel
  (RPE, % and deg), the best-fit scale (from the camera centres), the share of frames posed, and runtime.
  For ICP it also reports how well the surface pinned the translation (smallest/largest eigenvalue of the summed
  normal outer products: near 0 means a flat surface that lets ICP slide). The `tags` row scores the reference
  against itself and must be zero.

Every estimator writes the same trajectory file (`pose_files.py`: one `T_cam_world` per frame, world = the left
camera of its first posed frame), so whichever is adopted feeds the rest of the pipeline the same way.

## Testing a new recording

Add an entry to `recordings` in `config.json`: a name, the frames folder, the calibration, the printed tag size and
the camera depth range to keep (`depth_min_mm`, `depth_max_mm`: the nearest and furthest surface, from a frame or
two). A recording with no tags at all can still be posed by every estimator (run `run_poses.py` with
`tag_masks.enabled` false), but has no reference to score against.
