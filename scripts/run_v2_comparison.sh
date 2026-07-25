#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

N="${N:-25}"
RESULTS_DIR="${RESULTS_DIR:-results/v2_comparison}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_TRT_PREP="${SKIP_TRT_PREP:-0}"
COMPOSE=(docker compose -f docker-compose.backends.yml)

case "${RESULTS_DIR}" in
  results | results/*) ;;
  *)
    echo "RESULTS_DIR must be results or a directory under results/" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULTS_DIR}"

cleanup() {
  "${COMPOSE[@]}" \
    --profile vllm --profile sglang --profile triton \
    down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_http() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-600}"
  python3 - "${url}" "${name}" "${timeout_s}" <<'PY'
import sys
import time
import urllib.error
import urllib.request

url, name, timeout_raw = sys.argv[1:]
deadline = time.monotonic() + int(timeout_raw)
last_error = "not started"
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if 200 <= response.status < 300:
                print(f"{name} is ready")
                raise SystemExit(0)
            last_error = f"HTTP {response.status}"
    except (OSError, urllib.error.URLError) as exc:
        last_error = str(exc)
    time.sleep(5)
raise SystemExit(f"{name} did not become ready: {last_error}")
PY
}

run_remote_cells() {
  local make_target="$1"
  local prefix="$2"
  local concurrency
  for concurrency in 1 8 32; do
    make "${make_target}" \
      CELL="configs/cells/${prefix}_c${concurrency}_concurrent.yaml --results-dir ${RESULTS_DIR}"
  done
}

echo "== Preparing benchmark client and data =="
"${COMPOSE[@]}" --profile vllm build bench-client
if [[ "${SKIP_PREP}" != "1" ]]; then
  make gpu-build
  make gpu-prepare N="${N}"
fi

echo "== Running HF Transformers and faster-whisper baselines =="
make gpu-sweep \
  SWEEP="configs/sweeps/v1.yaml --results-dir ${RESULTS_DIR}"

echo "== Running vLLM concurrent cells =="
"${COMPOSE[@]}" --profile vllm up -d vllm
wait_http "http://127.0.0.1:8001/health" "vLLM"
run_remote_cells "vllm-cell" "vllm_turbo_l4"
"${COMPOSE[@]}" --profile vllm stop vllm
"${COMPOSE[@]}" --profile vllm rm -f vllm

echo "== Running SGLang concurrent cells =="
"${COMPOSE[@]}" --profile sglang up -d sglang
wait_http "http://127.0.0.1:8002/v1/models" "SGLang"
run_remote_cells "sglang-cell" "sglang_turbo_l4"
"${COMPOSE[@]}" --profile sglang stop sglang
"${COMPOSE[@]}" --profile sglang rm -f sglang

echo "== Preparing and running TensorRT-LLM/Triton cells =="
if [[ "${SKIP_TRT_PREP}" != "1" ]]; then
  make triton-prepare
fi
"${COMPOSE[@]}" --profile triton up -d triton
wait_http "http://127.0.0.1:8003/v2/health/ready" "Triton"
run_remote_cells "triton-cell" "trtllm_turbo_l4"
"${COMPOSE[@]}" --profile triton stop triton
"${COMPOSE[@]}" --profile triton rm -f triton

echo "== Writing comparison summary and plots =="
"${COMPOSE[@]}" --profile vllm run --rm bench-client \
  analyze --results-dir "${RESULTS_DIR}" >"${RESULTS_DIR}/summary.md"
"${COMPOSE[@]}" --profile vllm run --rm bench-client \
  plot --results-dir "${RESULTS_DIR}" \
  --out-dir "${RESULTS_DIR}/published"

echo "v2 comparison complete:"
echo "  ${RESULTS_DIR}/summary.md"
echo "  ${RESULTS_DIR}/published/gpu_sweep.md"
echo "  ${RESULTS_DIR}/published/gpu_sweep.png"
