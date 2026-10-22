#!/usr/bin/env bash
# Sets up the Python environments this project runs in. See src/SETUP.md.
#
#   bash src/setup_envs.sh main                  # src/venv: calibration, SGBM, COLMAP wrappers, fusion, figures
#   bash src/setup_envs.sh foundationstereo      # envs/ffs + external/FoundationStereo
#   bash src/setup_envs.sh aquastereo            # envs/aquastereo + external/AquaStereo
#   bash src/setup_envs.sh raft                  # envs/raft + external/RAFT-Stereo
#   bash src/setup_envs.sh all                   # all four
#   bash src/setup_envs.sh --weights aquastereo  # also download the network weights
#
# Run from anywhere; paths are relative to the project root. Re-running is safe: existing
# environments are updated in place and existing clones are just checked out at the pinned commit.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$PROJECT_ROOT/src"
PYTHON="${PYTHON:-python3.14}"
WEIGHTS=0

# Third-party matchers: repository URL and the commit the report's results were made with.
FS_URL="https://github.com/NVlabs/FoundationStereo.git";  FS_COMMIT="6e88068"
AQUA_URL="https://github.com/qz-wei/AquaStereo.git";      AQUA_COMMIT="6091725"
RAFT_URL="https://github.com/princeton-vl/RAFT-Stereo.git"; RAFT_COMMIT="6e93ed2"
FS_WEIGHTS_FOLDER="https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"

make_venv() {
    # make_venv <folder> <requirements file>
    local venv="$1" requirements="$2"
    echo "==> $venv"
    [ -x "$venv/bin/python" ] || "$PYTHON" -m venv "$venv"
    "$venv/bin/pip" install --upgrade pip
    "$venv/bin/pip" install -r "$requirements"
}

clone_at() {
    # clone_at <url> <folder> <commit>
    local url="$1" folder="$2" commit="$3"
    [ -d "$folder/.git" ] || git clone "$url" "$folder"
    git -C "$folder" fetch --quiet origin
    git -C "$folder" checkout --quiet "$commit"
    echo "    $folder at $(git -C "$folder" log -1 --format='%h %cs')"
}

setup_main() {
    make_venv "$SRC/venv" "$SRC/requirements.txt"
}

setup_foundationstereo() {
    make_venv "$PROJECT_ROOT/envs/ffs" "$SRC/envs/requirements-foundationstereo.txt"
    clone_at "$FS_URL" "$PROJECT_ROOT/external/FoundationStereo" "$FS_COMMIT"
    if [ "$WEIGHTS" = 1 ]; then
        # The whole folder (both models, ~4 GB). Only 23-51-11 (ViT-L, default) and 11-33-40 (ViT-S) are used.
        "$PROJECT_ROOT/envs/ffs/bin/gdown" --folder "$FS_WEIGHTS_FOLDER" \
            -O "$PROJECT_ROOT/external/FoundationStereo/pretrained_models"
    fi
}

setup_aquastereo() {
    local venv="$PROJECT_ROOT/envs/aquastereo"
    make_venv "$venv" "$SRC/envs/requirements-aquastereo.txt"
    "$venv/bin/pip" install --no-deps pytorchvideo==0.1.5   # X3D layers only
    clone_at "$AQUA_URL" "$PROJECT_ROOT/external/AquaStereo" "$AQUA_COMMIT"
    if [ "$WEIGHTS" = 1 ]; then
        "$venv/bin/python" -c "
from huggingface_hub import hf_hub_download
for name in ('AquaStereo_vitb_best.pth', 'AquaStereo_vits_best.pth'):
    print(hf_hub_download('qz-wei/AquaStereo', name, local_dir='$PROJECT_ROOT/external/AquaStereo/checkpoints'))
"
    fi
}

setup_raft() {
    make_venv "$PROJECT_ROOT/envs/raft" "$SRC/envs/requirements-raft.txt"
    clone_at "$RAFT_URL" "$PROJECT_ROOT/external/RAFT-Stereo" "$RAFT_COMMIT"
    if [ "$WEIGHTS" = 1 ]; then
        (cd "$PROJECT_ROOT/external/RAFT-Stereo" && bash download_models.sh)
    fi
}

check_system_tools() {
    # COLMAP and FFmpeg are system programs, not pip packages: only report whether they are there.
    echo "==> system tools"
    local tool
    for tool in colmap ffmpeg ffprobe; do
        if command -v "$tool" >/dev/null; then
            echo "    $tool: $(command -v "$tool")"
        else
            echo "    $tool: NOT FOUND (see src/SETUP.md)"
        fi
    done
}

main() {
    if [ "${1:-}" = "--weights" ]; then WEIGHTS=1; shift; fi
    if [ $# -eq 0 ]; then
        sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 1
    fi
    command -v "$PYTHON" >/dev/null || { echo "$PYTHON not found (set PYTHON=/path/to/python3.14)"; exit 1; }

    local target
    for target in "$@"; do
        case "$target" in
            main)             setup_main ;;
            foundationstereo) setup_foundationstereo ;;
            aquastereo)       setup_aquastereo ;;
            raft)             setup_raft ;;
            all)              setup_main; setup_foundationstereo; setup_aquastereo; setup_raft ;;
            *) echo "unknown target: $target"; exit 1 ;;
        esac
    done
    check_system_tools
}

main "$@"
