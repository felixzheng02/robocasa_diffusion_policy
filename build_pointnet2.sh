#!/usr/bin/env bash
# Compile graspnet-baseline's pointnet2 CUDA extension into the `grasp` env.
#
# Split out of setup_grasp_env.sh because this is the one step that actually fails, and
# it is worth being able to re-run on its own without redoing a 3 GB toolkit install.
#
# pointnet2 has no prebuilt substitute: models/backbone.py needs PointnetSAModuleVotes,
# which is votenet-specific and absent from the PyPI `pointnet2_ops` package.
#
# knn is deliberately NOT built. It vendors KNN_CUDA, which uses the TH/THC C API that
# PyTorch removed in 1.11. It is also unnecessary: knn is reached only via
# utils/label_generation.py, which serves training and evaluation, never inference.
# graspnet_backend.py installs a stub module so the transitive import at
# models/graspnet.py module scope still resolves.

set -euo pipefail

G="${GRASP_ENV:-/home/felix/miniforge3/envs/grasp}"
REPO="${REPO:-/home/felix/Desktop/robocasa_sim/third_party/graspnet-baseline}"

export CUDA_HOME="$G"
export PATH="$CUDA_HOME/bin:$G/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"     # RTX 3090 = sm_86; without this nvcc builds the
                                      # full arch list and can die on compute_90a
export MAX_JOBS=8

# CUDA 11.8's crt/host_config.h hard-errors on gcc > 11, and Ubuntu 24.04 ships gcc 13.3:
#     #error -- unsupported GNU version! gcc versions later than 11 are not supported!
# The C++ objects compile fine with the system compiler, so the failure only appears once
# the first .cu is reached -- well into the build. Use the conda gcc 11 for *both* halves
# so the C++ and CUDA objects share an ABI. NVCC_PREPEND_FLAGS is what actually reaches
# nvcc; CC/CXX alone are not enough for the -ccbin choice.
export CC="$G/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$G/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

echo "nvcc: $(nvcc --version | tail -1)"
"$G/bin/python" -c "import torch; print('torch', torch.__version__, torch.version.cuda)"

# AT_CHECK was renamed TORCH_CHECK in torch 1.5; the vendored sources predate that.
cd "$REPO/pointnet2"
if grep -rql "AT_CHECK" _ext_src/ 2>/dev/null; then
    echo "patching AT_CHECK -> TORCH_CHECK"
    grep -rl "AT_CHECK" _ext_src/ | xargs sed -i 's/\bAT_CHECK\b/TORCH_CHECK/g'
fi

rm -rf build/ dist/ *.egg-info    # stale objects were built with the wrong host compiler
"$G/bin/python" setup.py install

cd "$REPO"
"$G/bin/python" -c "import pointnet2._ext as e; print('pointnet2._ext OK')"
