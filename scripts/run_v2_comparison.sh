#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

N="${N:-25}"
RESULTS_DIR="${RESULTS_DIR:-results/v2_comparison}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_TRT_PREP="${SKIP_TRT_PREP:-0}"
DOCKER=(./scripts/docker.sh)

case "${RESULTS_DIR}" in
  results | results/*) ;;
  *)
    echo "RESULTS_DIR must be results or a directory under results/" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULTS_DIR}"

cleanup() {
  "${DOCKER[@]}" stop-backends >/dev/null 2>&1 || true
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

# Ready only when ASR works (avoids false ready from /health or /v1/models).
wait_transcription() {
  local base_url="$1"
  local name="$2"
  local wav="${3:-data/tone16k.wav}"
  local model="${4:-openai/whisper-large-v3-turbo}"
  local timeout_s="${5:-600}"
  local deadline=$((SECONDS + timeout_s))
  local out last_error="not started"

  [[ -f "${wav}" ]] || {
    echo "${name} smoke: missing ${wav}; run prepare first" >&2
    exit 2
  }

  while (( SECONDS < deadline )); do
    if out="$(curl -sS --max-time 60 \
      -H "Authorization: Bearer not-needed" \
      -F "file=@${wav};type=audio/wav" \
      -F "model=${model}" \
      -F "language=en" \
      -F "response_format=json" \
      "${base_url%/}/v1/audio/transcriptions" 2>&1)" \
      && [[ "${out}" == *'"text"'* ]]; then
      echo "${name} transcription smoke ok"
      return 0
    fi
    last_error="${out}"
    sleep 5
  done
  echo "${name} transcription smoke failed: ${last_error}" >&2
  exit 1
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

echo "== Preparing Docker images and data =="
"${DOCKER[@]}" build-bench
if [[ "${SKIP_PREP}" != "1" ]]; then
  make gpu-build
  make gpu-prepare N="${N}"
fi

echo "== Running HF Transformers and faster-whisper baselines =="
make gpu-sweep \
  SWEEP="configs/sweeps/v1.yaml --results-dir ${RESULTS_DIR}"

echo "== Running vLLM concurrent cells =="
"${DOCKER[@]}" up-vllm
wait_transcription "http://127.0.0.1:8001" "vLLM"
run_remote_cells "vllm-cell" "vllm_turbo"
"${DOCKER[@]}" stop-vllm

echo "== Running SGLang concurrent cells =="
"${DOCKER[@]}" up-sglang
wait_transcription "http://127.0.0.1:8002" "SGLang"
run_remote_cells "sglang-cell" "sglang_turbo"
"${DOCKER[@]}" stop-sglang

echo "== Preparing and running TensorRT-LLM/Triton cells =="
if [[ "${SKIP_TRT_PREP}" != "1" ]]; then
  make triton-prepare
fi
"${DOCKER[@]}" up-triton
wait_http "http://127.0.0.1:8003/v2/health/ready" "Triton"
run_remote_cells "triton-cell" "trtllm_turbo"
"${DOCKER[@]}" stop-triton

echo "== Writing comparison summary and plots =="
"${DOCKER[@]}" run-bench -- analyze --results-dir "${RESULTS_DIR}" \
  >"${RESULTS_DIR}/summary.md"
"${DOCKER[@]}" run-bench -- plot \
  --results-dir "${RESULTS_DIR}" \
  --out-dir "${RESULTS_DIR}/published"

echo "v2 comparison complete:"
echo "  ${RESULTS_DIR}/summary.md"
echo "  ${RESULTS_DIR}/published/gpu_sweep.md"
echo "  ${RESULTS_DIR}/published/gpu_sweep.png"
