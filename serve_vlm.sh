#!/usr/bin/env bash
# Serve the planner/monitor VLM for eval_agentic_pick_place.py.
#
# Runs in its own conda env: vLLM pins torch 2.11, robocasa_dp pins 2.7.1 alongside
# numpy==2.2.5 / mujoco==3.3.1, and the two cannot coexist. A separate process also means
# a VLM crash does not take a multi-hour sweep with it.
#
# CUDA_HOME is the non-obvious part. vLLM compiles CUDA graphs through inductor at
# startup and needs nvcc; without this it gets all the way through loading weights and
# sizing the KV cache, then dies with
#     RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
# There is no system CUDA toolkit here, but pip pulled one in with vllm, so point at that.
#
# Start this BEFORE the eval script: vLLM preallocates its memory pool, and the eval
# process needs ~3.5 GiB for the two diffusion policies plus EGL rendering.
#
# Usage:  ./serve_vlm.sh [extra vllm args...]

set -euo pipefail

VLM_ENV="${VLM_ENV:-/home/felix/miniforge3/envs/vlm}"
MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
PORT="${VLM_PORT:-8000}"

# $VLM_ENV/bin must be on PATH too, not just nvcc: FlashInfer JIT-builds its sampling
# kernel on first use and shells out to `ninja`, which pip put in the env's bin. Without
# it startup gets all the way past capturing CUDA graphs and then dies with
#     FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
export CUDA_HOME="$VLM_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$VLM_ENV/bin:$PATH"

# FlashInfer JIT-builds its sampling kernel against its own bundled CCCL headers, which
# disagree with the CUDA 13.3 toolkit headers above:
#     error "CUDA compiler and CUDA toolkit headers are incompatible"
# Use the PyTorch-native sampler instead. We sample greedily (temperature 0) and emit ~6
# output tokens per monitor call, so FlashInfer's sampler buys nothing here.
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "no nvcc under $CUDA_HOME — set CUDA_HOME to a real CUDA toolkit, or add" >&2
    echo "--enforce-eager to skip torch.compile (slower decode, but no nvcc needed)" >&2
    exit 1
fi

# AWQ rather than bf16: ~6 GiB instead of ~16.6, and int4 roughly halves decode latency.
# In an async monitor loop latency *is* staleness, so that matters more than the VRAM.
# --max-num-seqs 2 because this workload is one in-flight request at a time.
exec "$VLM_ENV/bin/vllm" serve "$MODEL" \
    --gpu-memory-utilization "${VLM_GPU_FRAC:-0.45}" \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --limit-mm-per-prompt '{"image":2}' \
    --port "$PORT" \
    "$@"
