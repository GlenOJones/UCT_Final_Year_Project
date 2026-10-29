# Setting up the environment

The project is all Python, but the learned stereo matchers need PyTorch and conflicting
dependencies, so it runs in up to four separate virtual environments:

| Environment         | Requirements file                           | Used for |
|---------------------|---------------------------------------------|----------|
| `src/venv`          | `src/requirements.txt`                      | Everything else: calibration, tag detection, SGBM, COLMAP wrappers, ICP, TSDF fusion, comparison, figures |
| `envs/ffs`          | `src/envs/requirements-foundationstereo.txt`| `--matcher foundationstereo` |
| `envs/aquastereo`   | `src/envs/requirements-aquastereo.txt`      | `--matcher aquastereo` |
| `envs/raft`         | `src/envs/requirements-raft.txt`            | `--matcher raft` |

Only `src/venv` is needed unless you run a learned matcher.

## Requirements

- **Linux and Python 3.14.** Tested on Ubuntu with Python 3.14.4.
- **For the learned matchers:** an NVIDIA GPU and a driver that supports CUDA 13.0. Tested on an RTX 5060 Laptop GPU (8 GB), driver 595.91.
- **COLMAP 4.3** (built with CUDA), on `PATH` as `colmap`. Used for pose estimation and multi-view stereo. Install it from <https://colmap.github.io/install.html>. The results were made with 4.3.0.dev0, commit `f773e790`.
- **FFmpeg** (`ffmpeg` and `ffprobe`), used to decode the raw YUYV recordings: `sudo apt install ffmpeg`. Tested with 8.0.1.

## Setup

From the project root:

```bash
bash src/setup_envs.sh main                       # just the main environment
bash src/setup_envs.sh --weights all              # everything, including the network weights
bash src/setup_envs.sh --weights foundationstereo # one matcher
```

For each matcher, the script:

1. Creates the venv.
2. Installs the pinned packages, with PyTorch from the CUDA 13.0 wheel index.
3. Clones the matcher's repository into `external/` at the commit the results were made with.
4. With `--weights`, downloads the weights:
   - **FoundationStereo:** about 4 GB from Google Drive.
   - **AquaStereo:** about 0.6 GB from Hugging Face.
   - **RAFT-Stereo:** from Dropbox.

At the end the script reports whether `colmap`, `ffmpeg` and `ffprobe` were found. Re-running it is safe.

If Python 3.14 isn't your `python3.14`, point the script at it: `PYTHON=/path/to/python bash src/setup_envs.sh main`.

To do the same by hand without the script:

```bash
python3.14 -m venv src/venv
src/venv/bin/pip install -r src/requirements.txt
```

## Running

Call each environment's Python directly. No activation is needed:

```bash
src/venv/bin/python src/03_depth/dense_stereo.py --session Sep24 --scan mjpg_pyr_lights_2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True envs/ffs/bin/python src/03_depth/dense_stereo.py \
    --session <session> --scan <scan> --matcher foundationstereo --net-scale 0.5 --lr-tolerance 3
```

## Third-party code and licences

| Matcher | Repository | Commit | Licence |
|---|---|---|---|
| FoundationStereo | <https://github.com/NVlabs/FoundationStereo> | `6e88068` | NVIDIA, research and non-commercial use only |
| AquaStereo | <https://github.com/qz-wei/AquaStereo> | `6091725` | none stated |
| RAFT-Stereo | <https://github.com/princeton-vl/RAFT-Stereo> | `6e93ed2` | MIT |

None of them are included in this repository. The setup script fetches them.

**The authors' environment files are not used.** They target PyTorch 2.0–2.6 with CUDA 11–12, which cannot run on RTX 50-series GPUs (`sm_120`).
