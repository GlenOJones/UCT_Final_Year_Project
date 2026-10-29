# Stereo Reconstruction of the Sea Ice Underside

Code for EEE4022F final year project (UCT): calibrating a stereo camera rig underwater and reconstructing
the underside of sea ice, tested on 3D-printed shapes with known CAD geometry and foam blocks that imitate
ice.

**Live point cloud viewer:** <https://glenojones.github.io/UCT_Final_Year_Project/viewer/>

The viewer compares the fused clouds from COLMAP and from TSDF fusion of FoundationStereo and SGBM depth for
each test object in the browser. Its files are in `docs/viewer/`.

## Pipeline

The code is split into six stages. Each stage is a standalone script that reads the previous stage's output,
so a stage can be swapped out without rerunning the ones before it.

| Stage | Script(s) | Output |
|---|---|---|
| 1. Frame extraction | `01_frame_extraction/extract_stereo_frames.py` | left/right PNG frame pairs |
| 2. Calibration | `02_calibration/Calibration.py` | stereo calibration (`*_stereo.yaml`), detected corners (`*_corners.json`) |
| 3. Depth | `03_depth/depth_frames.py` (SGBM, FoundationStereo, AquaStereo, RAFT-Stereo) | per-frame depth points (`.npz`) |
| 4. Pose | `04_pose/tag_poses.py`, `colmap_poses.py`, `icp_poses.py` | camera trajectory (`.yaml`) |
| 5. Fusion | `05_fusion/tsdf_fuse.py`, `colmap_mvs.py` | fused point cloud (`cloud.ply`) |
| 6. Clean-up | `06_cleanup/postprocess_cloud.py` | cleaned cloud and mesh (`cloud_clean.ply`, `mesh.ply`) |

Where a stage lists several scripts, they are alternative methods that the thesis compares.

## Folder structure

```
src/
  01_frame_extraction/ ... 06_cleanup/   the pipeline stages above
  evaluation/       scoring reconstructions against the CAD models
  experiments/      one folder per experiment: config.json, analysis and plotting scripts
  tools/            viewers and older one-off scripts, not needed to reproduce the results
  envs/             requirements for the learned stereo matchers
  stereo_rig.py     shared rig geometry (loads a calibration, triangulates)
  project_paths.py  where data/, results/ and work/ live
CAD/                test rig, calibration board and mount models
docs/viewer/        the live point cloud viewer
results/            figures and tables used in the report
```

## Setup

Linux, Python 3.14, COLMAP 4.3 and FFmpeg. The learned matchers also need an NVIDIA GPU with CUDA 13.0.

```bash
bash src/setup_envs.sh main         # main environment only
bash src/setup_envs.sh --weights all  # also the learned matchers and their weights
```

See [`src/SETUP.md`](src/SETUP.md) for details and third-party licences.

## Running

Each stage can be run on its own. Pass `--help` to any script for its options, for example:

```bash
src/venv/bin/python src/02_calibration/Calibration.py --frames-dir <frames> --preset water_final --channel merge
src/venv/bin/python src/04_pose/tag_poses.py --session Sep24 --scan mjpg_pyr_lights_2
```

Every experiment in the thesis is defined by a JSON config that states every setting, and is rerun from the raw
frames with:

```bash
src/venv/bin/python src/experiments/reproduce.py <experiment>   # or: all
```

Each run writes a `manifest.json` with the config, git commit and library versions that produced it. See
[`src/experiments/README.md`](src/experiments/README.md) for the list of experiments.

The raw recordings (`data/`, about 30 GB) are not in this repository.
