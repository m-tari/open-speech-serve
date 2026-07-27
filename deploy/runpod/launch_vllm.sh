#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openai/whisper-large-v3-turbo}"
PORT="${PORT:-8000}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

# Stock vllm-openai omits Whisper audio extras; install before serve so the
# API process imports real deps (not PlaceholderModule).
if ! python3 -c "import soundfile, av" >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ffmpeg libsndfile1
  python3 -m pip install --no-cache-dir "vllm[audio]==${VLLM_VERSION}"
fi

exec vllm serve "${MODEL}" \
  --served-model-name "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype float16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
