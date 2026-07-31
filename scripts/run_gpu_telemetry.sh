#!/usr/bin/env bash
# HF / vLLM / TensorRT-LLM cells with nvidia-smi telemetry (1 pass each).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

N="${N:-25}"
RESULTS_DIR="${RESULTS_DIR:-results/gpu_telemetry}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_TRT_PREP="${SKIP_TRT_PREP:-0}"
DOCKER=(./scripts/docker.sh)
FLAGS="--results-dir ${RESULTS_DIR} --gpu-telemetry --passes 1"

case "${RESULTS_DIR}" in
  results | results/*) ;;
  *) echo "RESULTS_DIR must be under results/" >&2; exit 2 ;;
esac
mkdir -p "${RESULTS_DIR}"

cleanup() { "${DOCKER[@]}" stop-backends >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_http() {
  python3 - "$1" "$2" "${3:-600}" <<'PY'
import sys, time, urllib.error, urllib.request
url, name, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.monotonic() + timeout
err = "not started"
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if 200 <= r.status < 300:
                print(f"{name} is ready"); raise SystemExit(0)
            err = f"HTTP {r.status}"
    except (OSError, urllib.error.URLError) as e:
        err = str(e)
    time.sleep(5)
raise SystemExit(f"{name} did not become ready: {err}")
PY
}

wait_transcription() {
  local base_url="$1" name="$2" wav="${3:-data/tone16k.wav}"
  local model="${4:-openai/whisper-large-v3-turbo}" deadline=$((SECONDS + 600)) out err="not started"
  [[ -f "${wav}" ]] || { echo "${name}: missing ${wav}" >&2; exit 2; }
  while (( SECONDS < deadline )); do
    if out="$(curl -sS --max-time 60 -H "Authorization: Bearer not-needed" \
      -F "file=@${wav};type=audio/wav" -F "model=${model}" -F "language=en" \
      -F "response_format=json" "${base_url%/}/v1/audio/transcriptions" 2>&1)" \
      && [[ "${out}" == *'"text"'* ]]; then
      echo "${name} transcription smoke ok"; return 0
    fi
    err="${out}"; sleep 5
  done
  echo "${name} transcription smoke failed: ${err}" >&2; exit 1
}

echo "== Prepare =="
"${DOCKER[@]}" build-bench
if [[ "${SKIP_PREP}" != "1" ]]; then
  make gpu-build
  make gpu-prepare N="${N}"
fi

echo "== HF serialized =="
for c in 1 8 32; do
  make gpu-cell CELL="configs/cells/hf_turbo_c${c}.yaml ${FLAGS}"
done

echo "== vLLM concurrent =="
"${DOCKER[@]}" up-vllm
wait_transcription "http://127.0.0.1:8001" "vLLM"
for c in 1 8 32; do
  make vllm-cell CELL="configs/cells/vllm_turbo_c${c}_concurrent.yaml ${FLAGS}"
done
"${DOCKER[@]}" stop-vllm

echo "== TensorRT-LLM concurrent =="
if [[ "${SKIP_TRT_PREP}" != "1" ]]; then
  make triton-prepare
fi
"${DOCKER[@]}" up-triton
wait_http "http://127.0.0.1:8003/v2/health/ready" "Triton"
for c in 1 8 32; do
  make triton-cell CELL="configs/cells/trtllm_turbo_c${c}_concurrent.yaml ${FLAGS}"
done
"${DOCKER[@]}" stop-triton

echo "== Plot =="
"${DOCKER[@]}" run-bench -- plot-gpu-telemetry \
  --results-dir "${RESULTS_DIR}" --out-dir "${RESULTS_DIR}/published"

echo "Done: ${RESULTS_DIR}/published/gpu_util_vs_concurrency.png"
echo "      ${RESULTS_DIR}/published/gpu_util_vs_time_c8.png"
