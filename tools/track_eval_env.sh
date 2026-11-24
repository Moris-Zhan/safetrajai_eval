#!/bin/bash
# Keep the `track_eval` conda env on the mounted disk so it survives container restarts.
#
# The container's root filesystem (/root/miniconda3/envs/) is reset on restart, which deleted the
# env on 2026-11-24 and killed a training run. The env now lives in ignore/conda_envs/track_eval
# (gitignored) and /root/miniconda3/envs/track_eval is only a symlink to it.
#
#   bash tools/track_eval_env.sh link    # after every container restart: recreate the symlink (seconds)
#   bash tools/track_eval_env.sh build   # first time only: create the env (downloads ~3-4 GB)
#   bash tools/track_eval_env.sh freeze  # write ignore/conda_envs/track_eval.freeze.txt (pip freeze)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="$REPO/ignore/conda_envs/track_eval"
LINK="/root/miniconda3/envs/track_eval"
CONDA="/root/miniconda3/bin/conda"

link() {
    [ -x "$ENV_DIR/bin/python" ] || { echo "$ENV_DIR not built; run: bash $0 build" >&2; exit 1; }
    if [ -L "$LINK" ]; then
        echo "symlink already present: $LINK -> $(readlink "$LINK")"
    elif [ -e "$LINK" ]; then
        echo "$LINK exists and is not a symlink; leaving it alone" >&2; exit 1
    else
        ln -s "$ENV_DIR" "$LINK"
        echo "linked $LINK -> $ENV_DIR"
    fi
}

build() {
    [ -e "$ENV_DIR" ] && { echo "$ENV_DIR already exists; remove it first to rebuild" >&2; exit 1; }
    mkdir -p "$(dirname "$ENV_DIR")"
    "$CONDA" create -y -p "$ENV_DIR" python=3.10
    local pip="$ENV_DIR/bin/pip"
    # CUDA 12.8 build (RTX 5090); same versions as the env that was lost.
    "$pip" install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
    # Everything else from pyproject.toml; torch is already satisfied, so pip keeps the CUDA build.
    "$pip" install ultralytics==8.3.141 opencv-python numpy scipy openpyxl pyyaml matplotlib pillow lap numba \
        requests torchinfo scikit-learn timm pytest
    (cd "$REPO" && "$pip" install -e . --no-deps)
    "$ENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.version.cuda)"
    link
    freeze
}

freeze() {
    "$ENV_DIR/bin/pip" freeze --exclude-editable > "$REPO/ignore/conda_envs/track_eval.freeze.txt"
    echo "wrote $REPO/ignore/conda_envs/track_eval.freeze.txt"
}

case "${1:-}" in
    link) link ;;
    build) build ;;
    freeze) freeze ;;
    *) echo "usage: $0 link|build|freeze" >&2; exit 2 ;;
esac
