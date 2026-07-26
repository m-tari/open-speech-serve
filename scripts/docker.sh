#!/usr/bin/env bash
# Plain Docker helpers — no Compose required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CPU_IMAGE="${CPU_IMAGE:-open-speech-serve:cpu}"
GPU_IMAGE="${GPU_IMAGE:-open-speech-serve:gpu}"
BENCH_IMAGE="${BENCH_IMAGE:-open-speech-serve:bench}"
TRITON_IMAGE_LOCAL="${TRITON_IMAGE_LOCAL:-open-speech-serve:triton-whisper}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.25.1}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.15}"
TRITON_BASE_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3}"
TRTLLM_VERSION="${TRTLLM_VERSION:-v1.1.0}"

HF_CACHE_HOST="${HF_CACHE_HOST:-${ROOT}/.cache/huggingface}"
ARTIFACTS_HOST="${ARTIFACTS_HOST:-${ROOT}/artifacts/triton}"

CONTAINER_CPU="${CONTAINER_CPU:-oss-cpu}"
CONTAINER_GPU="${CONTAINER_GPU:-oss-gpu}"
CONTAINER_VLLM="${CONTAINER_VLLM:-oss-vllm}"
CONTAINER_SGLANG="${CONTAINER_SGLANG:-oss-sglang}"
CONTAINER_TRITON="${CONTAINER_TRITON:-oss-triton}"

DOCKER_SH="${ROOT}/scripts/docker.sh"

usage() {
  cat <<'EOF'
Usage: scripts/docker.sh <command> [args...]

Build:
  build-cpu | build-gpu | build-bench | build-triton

CPU / GPU local images:
  run-cpu ENTRYPOINT [args...]
  run-gpu ENTRYPOINT [args...]
  up-server | up-gpu-server
  stop-server | stop-gpu-server

Remote backends:
  up-vllm | up-sglang | up-triton
  stop-vllm | stop-sglang | stop-triton
  stop-backends
  triton-prepare

Benchmark client (host network → localhost backends):
  run-bench [-e KEY=VAL ...] -- ENTRYPOINT [args...]
  cell-vllm CELL_ARGS...
  cell-sglang CELL_ARGS...
  cell-triton CELL_ARGS...
  sweep-vllm SWEEP_ARGS...
  sweep-sglang SWEEP_ARGS...
  sweep-triton SWEEP_ARGS...
EOF
}

ensure_dirs() {
  mkdir -p data results configs "${HF_CACHE_HOST}" "${ARTIFACTS_HOST}"
}

app_mounts=(
  -v "${ROOT}/data:/app/data"
  -v "${ROOT}/results:/app/results"
  -v "${ROOT}/configs:/app/configs"
  -v "${HF_CACHE_HOST}:/app/.cache/huggingface"
)

rm_container() {
  local name="$1"
  if docker ps -aq --filter "name=^${name}$" | grep -q .; then
    docker rm -f "${name}" >/dev/null 2>&1 || true
  fi
}

stop_named() {
  local name="$1"
  if docker ps -aq --filter "name=^${name}$" | grep -q .; then
    docker stop "${name}" >/dev/null 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
  fi
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  build-cpu)
    docker build -t "${CPU_IMAGE}" -f Dockerfile .
    ;;
  build-gpu)
    docker build -t "${GPU_IMAGE}" -f Dockerfile.gpu .
    ;;
  build-bench)
    docker build -t "${BENCH_IMAGE}" -f docker/Dockerfile.bench .
    ;;
  build-triton)
    docker build \
      -t "${TRITON_IMAGE_LOCAL}" \
      -f docker/Dockerfile.triton \
      --build-arg "TRITON_IMAGE=${TRITON_BASE_IMAGE}" \
      --build-arg "TRTLLM_VERSION=${TRTLLM_VERSION}" \
      .
    ;;

  run-cpu)
    ensure_dirs
    entrypoint="${1:?entrypoint required}"
    shift
    # Host network so TTFS can reach a published local server on :8000.
    docker run --rm --network host \
      "${app_mounts[@]}" \
      -e "OSS_FRAMEWORK=${OSS_FRAMEWORK:-faster_whisper}" \
      -e "OSS_MODEL=${OSS_MODEL:-tiny}" \
      -e "OSS_DEVICE=${OSS_DEVICE:-cpu}" \
      -e "OSS_COMPUTE_TYPE=${OSS_COMPUTE_TYPE:-int8}" \
      --entrypoint "${entrypoint}" \
      "${CPU_IMAGE}" \
      "$@"
    ;;
  run-gpu)
    ensure_dirs
    entrypoint="${1:?entrypoint required}"
    shift
    docker run --rm --gpus all --network host \
      "${app_mounts[@]}" \
      -e "OSS_FRAMEWORK=${OSS_FRAMEWORK:-faster_whisper}" \
      -e "OSS_MODEL=${OSS_MODEL:-large-v3-turbo}" \
      -e "OSS_DEVICE=cuda" \
      -e "OSS_COMPUTE_TYPE=${OSS_COMPUTE_TYPE:-float16}" \
      --entrypoint "${entrypoint}" \
      "${GPU_IMAGE}" \
      "$@"
    ;;

  up-server)
    ensure_dirs
    rm_container "${CONTAINER_CPU}"
    docker run -d --name "${CONTAINER_CPU}" \
      -p 8000:8000 \
      "${app_mounts[@]}" \
      -e "OSS_FRAMEWORK=${OSS_FRAMEWORK:-faster_whisper}" \
      -e "OSS_MODEL=${OSS_MODEL:-tiny}" \
      -e "OSS_DEVICE=${OSS_DEVICE:-cpu}" \
      -e "OSS_COMPUTE_TYPE=${OSS_COMPUTE_TYPE:-int8}" \
      "${CPU_IMAGE}"
    ;;
  up-gpu-server)
    ensure_dirs
    rm_container "${CONTAINER_GPU}"
    docker run -d --name "${CONTAINER_GPU}" --gpus all \
      -p 8000:8000 \
      "${app_mounts[@]}" \
      -e "OSS_FRAMEWORK=${OSS_FRAMEWORK:-faster_whisper}" \
      -e "OSS_MODEL=${OSS_MODEL:-large-v3-turbo}" \
      -e "OSS_DEVICE=cuda" \
      -e "OSS_COMPUTE_TYPE=${OSS_COMPUTE_TYPE:-float16}" \
      "${GPU_IMAGE}"
    ;;
  stop-server)
    stop_named "${CONTAINER_CPU}"
    ;;
  stop-gpu-server)
    stop_named "${CONTAINER_GPU}"
    ;;

  up-vllm)
    ensure_dirs
    rm_container "${CONTAINER_VLLM}"
    docker run -d --name "${CONTAINER_VLLM}" --gpus all --ipc=host \
      -p 8001:8000 \
      -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
      -e "HF_TOKEN=${HF_TOKEN:-}" \
      "${VLLM_IMAGE}" \
      --model openai/whisper-large-v3-turbo \
      --served-model-name openai/whisper-large-v3-turbo \
      --host 0.0.0.0 \
      --port 8000 \
      --dtype float16 \
      --gpu-memory-utilization 0.90
    ;;
  up-sglang)
    ensure_dirs
    rm_container "${CONTAINER_SGLANG}"
    docker run -d --name "${CONTAINER_SGLANG}" --gpus all --ipc=host \
      -p 8002:30000 \
      -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
      -e "HF_TOKEN=${HF_TOKEN:-}" \
      --entrypoint python3 \
      "${SGLANG_IMAGE}" \
      -m sglang.launch_server \
      --model-path openai/whisper-large-v3-turbo \
      --served-model-name openai/whisper-large-v3-turbo \
      --host 0.0.0.0 \
      --port 30000 \
      --dtype float16
    ;;
  up-triton)
    ensure_dirs
    rm_container "${CONTAINER_TRITON}"
    docker run -d --name "${CONTAINER_TRITON}" --gpus all --ipc=host \
      -p 8003:8000 \
      -p 8004:8001 \
      -p 8005:8002 \
      -v "${ARTIFACTS_HOST}:/workspace/artifacts/triton" \
      --entrypoint python3 \
      "${TRITON_IMAGE_LOCAL}" \
      /opt/TensorRT-LLM/tensorrt_llm/triton_backend/scripts/launch_triton_server.py \
      --world_size 1 \
      --model_repo /workspace/artifacts/triton/model_repo \
      --tensorrt_llm_model_name tensorrt_llm,whisper_bls \
      --multimodal_gpu0_cuda_mem_pool_bytes 300000000
    ;;
  stop-vllm)
    stop_named "${CONTAINER_VLLM}"
    ;;
  stop-sglang)
    stop_named "${CONTAINER_SGLANG}"
    ;;
  stop-triton)
    stop_named "${CONTAINER_TRITON}"
    ;;
  stop-backends)
    stop_named "${CONTAINER_VLLM}"
    stop_named "${CONTAINER_SGLANG}"
    stop_named "${CONTAINER_TRITON}"
    ;;

  triton-prepare)
    ensure_dirs
    "${DOCKER_SH}" build-triton
    docker run --rm --gpus all \
      -v "${ARTIFACTS_HOST}:/workspace/artifacts/triton" \
      -v "${ROOT}/scripts/triton:/workspace/open-speech-serve/scripts/triton:ro" \
      --entrypoint bash \
      "${TRITON_IMAGE_LOCAL}" \
      /workspace/open-speech-serve/scripts/triton/build_whisper_engines.sh
    ;;

  run-bench)
    ensure_dirs
    docker_args=()
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "--" ]]; then
        shift
        break
      fi
      docker_args+=("$1")
      shift
    done
    entrypoint="${1:?entrypoint required after --}"
    shift || true
    docker run --rm --network host \
      "${app_mounts[@]}" \
      "${docker_args[@]}" \
      --entrypoint "${entrypoint}" \
      "${BENCH_IMAGE}" \
      "$@"
    ;;

  cell-vllm)
    "${DOCKER_SH}" run-bench -e OSS_BASE_URL=http://127.0.0.1:8001 -- cell "$@"
    ;;
  cell-sglang)
    "${DOCKER_SH}" run-bench -e OSS_BASE_URL=http://127.0.0.1:8002 -- cell "$@"
    ;;
  cell-triton)
    "${DOCKER_SH}" run-bench -e OSS_GRPC_URL=127.0.0.1:8004 -- cell "$@"
    ;;
  sweep-vllm)
    "${DOCKER_SH}" run-bench -e OSS_BASE_URL=http://127.0.0.1:8001 -- sweep "$@"
    ;;
  sweep-sglang)
    "${DOCKER_SH}" run-bench -e OSS_BASE_URL=http://127.0.0.1:8002 -- sweep "$@"
    ;;
  sweep-triton)
    "${DOCKER_SH}" run-bench -e OSS_GRPC_URL=127.0.0.1:8004 -- sweep "$@"
    ;;

  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
