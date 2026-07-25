#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openai/whisper-large-v3-turbo}"
PORT="${PORT:-30000}"

exec python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --served-model-name "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype float16
