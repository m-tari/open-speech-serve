#!/usr/bin/env bash
# Foreground Whisper Triton (world_size=1).
# NVIDIA launch_triton_server.py uses Popen and exits, so it cannot be Docker PID 1.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-/workspace/artifacts/triton/model_repo}"
CUDA_MEM_POOL_BYTES="${MULTIMODAL_GPU0_CUDA_MEM_POOL_BYTES:-300000000}"

exec mpirun --allow-run-as-root -n 1 \
  /opt/tritonserver/bin/tritonserver \
  --model-repository="${MODEL_REPO}" \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --disable-auto-complete-config \
  --backend-config=python,shm-region-prefix-name=prefix0_ \
  --cuda-memory-pool-byte-size="0:${CUDA_MEM_POOL_BYTES}"
