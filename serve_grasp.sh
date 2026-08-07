#!/usr/bin/env bash
# Serve the 6-DoF grasp detector for eval_anygrasp_pick.py.
#
# Runs in its own conda env for the same reason serve_vlm.sh does: this env pins
# numpy<2 and torch 2.0.1+cu118 so a 2019-vintage CUDA extension will compile, while
# robocasa/__init__.py hard-asserts numpy==2.2.5. The two cannot coexist, and a detector
# crash must not take a multi-hour sweep with it. Nothing imports across the boundary.
#
# Port 8100, not 8000, so this can run alongside the vLLM server -- the agentic arm and
# the grasp arm should be able to coexist on one box.
#
# Three things that are easy to get wrong and fail late:
#   - $GRASP_ENV/bin must be on PATH (nvcc + ninja live there), the same trap serve_vlm.sh
#     documents.
#   - PYTHONPATH must include graspnet-baseline's root AND its models/ utils/ dataset/
#     subdirs: upstream uses flat sibling imports (`from backbone import ...`).
#   - The checkpoint must exist. Without it the server would start and every rollout would
#     score `no_grasp_proposed`, which reads exactly like a bad detector. Preflight instead.
#
# Start this BEFORE the eval. Usage:  ./serve_grasp.sh [extra uvicorn args...]

set -euo pipefail

GRASP_ENV="${GRASP_ENV:-/home/felix/miniforge3/envs/grasp}"
GRASPNET_ROOT="${GRASPNET_ROOT:-/home/felix/Desktop/robocasa_sim/third_party/graspnet-baseline}"
GRASP_CHECKPOINT="${GRASP_CHECKPOINT:-/home/felix/Desktop/robocasa_sim/third_party/checkpoints/checkpoint-rs.tar}"
GRASP_BACKEND="${GRASP_BACKEND:-graspnet-baseline}"
PORT="${GRASP_PORT:-8100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_HOME="$GRASP_ENV"
export PATH="$CUDA_HOME/bin:$GRASP_ENV/bin:$PATH"
export GRASPNET_ROOT GRASP_CHECKPOINT GRASP_BACKEND
export PYTHONPATH="$HERE:$GRASPNET_ROOT:$GRASPNET_ROOT/models:$GRASPNET_ROOT/utils:$GRASPNET_ROOT/dataset:$GRASPNET_ROOT/pointnet2:${PYTHONPATH:-}"

if [ ! -d "$GRASPNET_ROOT" ]; then
    echo "graspnet-baseline not found at $GRASPNET_ROOT -- run ./setup_grasp_env.sh" >&2
    exit 1
fi

if [ "$GRASP_BACKEND" = "graspnet-baseline" ] && [ ! -f "$GRASP_CHECKPOINT" ]; then
    echo "checkpoint missing: $GRASP_CHECKPOINT" >&2
    echo "fetch it with:" >&2
    echo "  $GRASP_ENV/bin/gdown 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O $GRASP_CHECKPOINT" >&2
    exit 1
fi

# Import torch before the compiled extension: pointnet2._ext links against libc10.so and
# fails with 'cannot open shared object file' if torch has not been loaded first.
"$GRASP_ENV/bin/python" - <<'EOF'
import torch, sys
assert torch.cuda.is_available(), "CUDA not available in the grasp env"
import pointnet2._ext  # noqa: F401
print(f"[preflight] torch {torch.__version__} cuda {torch.version.cuda} + pointnet2 OK")
EOF

# --workers 1: the model is on the GPU and requests are serialised anyway. The startup
# self-test in grasp_server lives inside the app, so a bad checkpoint path refuses to serve.
exec "$GRASP_ENV/bin/python" -m uvicorn grasp_server:app \
    --host 127.0.0.1 --port "$PORT" --workers 1 "$@"
