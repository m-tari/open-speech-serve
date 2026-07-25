#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openai/whisper-large-v3-turbo}"
PORT="${PORT:-8000}"

exec vllm serve "${MODEL}" \
  --served-model-name "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype float16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
