#!/usr/bin/env bash
# Build the `grasp` conda env: graspnet-baseline now, AnyGrasp SDK later.
#
# This has to be its own env and can never be merged into robocasa_dp:
#   - robocasa/__init__.py hard-asserts numpy==2.2.5, while graspnetAPI and the
#     pointnet2 extension are numpy-1.x code.
#   - robocasa_dp is python 3.11 / torch 2.7.1+cu126; the pointnet2 kernels are old
#     CUDA-C that the 12.x/13.x toolchains reject.
# Same reasoning as serve_vlm.sh, and the same consequence: the eval talks to the
# detector over HTTP, nothing imports across the boundary.
#
# CUDA 11.8 is the newest line whose nvcc still compiles pointnet2 cleanly. There is no
# system CUDA toolkit on this box, so we install one into the env from the nvidia
# channel. CUDA 11.8 also rejects the system gcc 13, hence gxx_linux-64=11.
#
# Usage:  ./setup_grasp_env.sh          (idempotent-ish; safe to re-run)

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/home/felix/miniforge3}"
ENV_NAME="${GRASP_ENV_NAME:-grasp}"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
THIRD_PARTY="${THIRD_PARTY:-/home/felix/Desktop/robocasa_sim/third_party}"
REPO_DIR="$THIRD_PARTY/graspnet-baseline"
CKPT_DIR="${CKPT_DIR:-/home/felix/Desktop/robocasa_sim/third_party/checkpoints}"

step() { echo -e "\n\033[1;36m==> $*\033[0m"; }

step "1/6  create env $ENV_NAME (python 3.10)"
if [ ! -d "$ENV_PREFIX" ]; then
    "$CONDA_BASE/bin/conda" create -n "$ENV_NAME" python=3.10 -y
else
    echo "    exists, skipping"
fi
PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

step "2/6  CUDA 11.8 toolkit + gcc 11 into the env"
# Full toolkit rather than cuda-nvcc alone: building the extension needs the dev
# headers (cuda_runtime.h) and the cublas/cusparse headers torch's own headers pull in.
if [ ! -x "$ENV_PREFIX/bin/nvcc" ]; then
    "$CONDA_BASE/bin/conda" install -n "$ENV_NAME" -y \
        -c "nvidia/label/cuda-11.8.0" cuda-toolkit
else
    echo "    nvcc present, skipping"
fi
"$CONDA_BASE/bin/conda" install -n "$ENV_NAME" -y -c conda-forge gxx_linux-64=11

step "3/6  torch 2.0.1+cu118 and python deps"
"$PIP" install --no-input torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
# numpy<2 is mandatory: graspnetAPI and the compiled ops are numpy-1.x ABI.
"$PIP" install --no-input "numpy<2" scipy pillow tqdm open3d \
    fastapi "uvicorn[standard]" gdown transforms3d

step "4/6  clone graspnet-baseline"
mkdir -p "$THIRD_PARTY"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/graspnet/graspnet-baseline.git "$REPO_DIR"
else
    echo "    exists, skipping"
fi

step "5/6  build pointnet2 (and knn, which is allowed to fail)"
export CUDA_HOME="$ENV_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"          # RTX 3090 = sm_86
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"

( cd "$REPO_DIR/pointnet2" && "$PY" setup.py install )

# knn is NOT needed for inference -- only training and ModelFreeCollisionDetector use
# it, and it commonly fails to build on torch>=1.11 because THC/THC.h was removed.
# open3d covers collision filtering, so a failure here is not fatal.
if ( cd "$REPO_DIR/knn" && "$PY" setup.py install ); then
    echo "    knn built"
else
    echo -e "\033[1;33m    knn FAILED to build -- expected on torch>=1.11, continuing\033[0m"
fi

step "6/6  graspnetAPI + pretrained checkpoint"
"$PIP" install --no-input graspnetAPI || echo "    graspnetAPI pip failed; try source install"

mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/checkpoint-rs.tar" ]; then
    # RealSense checkpoint; Google Drive is rate-limited, so this is the flaky step.
    "$ENV_PREFIX/bin/gdown" 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk \
        -O "$CKPT_DIR/checkpoint-rs.tar" \
        || echo -e "\033[1;33m    checkpoint download failed -- fetch it by hand\033[0m"
else
    echo "    checkpoint present, skipping"
fi

step "done"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
import numpy; print("numpy", numpy.__version__)
try:
    import pointnet2_utils; print("pointnet2 OK")
except Exception as e:
    print("pointnet2 IMPORT FAILED:", e)
EOF
