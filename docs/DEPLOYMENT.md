# Serving backend deployment

The v2 benchmark uses a lightweight benchmark client and one isolated GPU
backend. Do not install vLLM, SGLang, and TensorRT-LLM into one Python
environment: their pinned Torch/CUDA stacks differ.

## Docker GPU VM

Requirements: x86_64 Linux, NVIDIA driver, Docker, Compose, and NVIDIA Container
Toolkit. Confirm `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04
nvidia-smi` succeeds.

Prepare the common data once:

```bash
make gpu-build
make gpu-prepare N=25
```

Run the complete comparison sequentially on one GPU:

```bash
make v2-comparison
```

The orchestrator cleans up active backend containers on success, failure, or
interruption. Set `RESULTS_DIR=results/<name>`, `N=<count>`, `SKIP_PREP=1`, or
`SKIP_TRT_PREP=1` as needed. TensorRT preparation remains enabled by default
because its engines must match the target GPU and runtime.

Start one backend in terminal 1, then run its cells in terminal 2:

```bash
# vLLM 0.25.1
make vllm-up
make vllm-cell CELL=configs/cells/vllm_turbo_l4_c8_serialized.yaml
make vllm-cell CELL=configs/cells/vllm_turbo_l4_c8_concurrent.yaml

# SGLang 0.5.15 (Whisper support is experimental)
make sglang-up
make sglang-cell CELL=configs/cells/sglang_turbo_l4_c8_concurrent.yaml

# Triton 26.02 + TensorRT-LLM 1.1.0
make triton-prepare       # builds L4-specific engines; run once
make triton-up
make triton-cell CELL=configs/cells/trtllm_turbo_l4_c8_concurrent.yaml
```

Stop the active profile before switching frameworks. Image tags can be
overridden with `VLLM_IMAGE`, `SGLANG_IMAGE`, `TRITON_IMAGE`, and
`TRTLLM_VERSION`. Record overrides with published results.

The exposed host endpoints are:

- vLLM HTTP: `8001`
- SGLang HTTP: `8002`
- Triton HTTP/gRPC/metrics: `8003` / `8004` / `8005`

Inside Compose, Make targets override endpoint addresses to service DNS names.
Direct cells default to local backend ports and can be overridden with
`OSS_BASE_URL` or `OSS_GRPC_URL`.

## TensorRT preparation

`scripts/triton/build_whisper_engines.sh` downloads the official
`large-v3-turbo` checkpoint, converts encoder/decoder checkpoints, builds FP16
engines with batch size 32, and creates the NVIDIA Whisper BLS model repository
under `artifacts/triton/`.

Engine artifacts are intentionally gitignored. Rebuild them when the GPU
architecture, TensorRT-LLM version, precision, or max batch size changes:

```bash
MAX_BATCH_SIZE=32 make triton-prepare
```

## RunPod

RunPod Pods are already containers and do not support nested Docker Compose.
Use the direct instructions in
[`deploy/runpod/README.md`](../deploy/runpod/README.md). Use one pod/backend
session at a time and retain `/workspace` on a network volume.

## Validation before publishing

1. Confirm `/v1/models` reports the exact model for vLLM/SGLang, or Triton
   reports `whisper_bls` ready.
2. Run concurrency 1 first and verify WER is comparable.
3. Run paired serialized/concurrent cells at 8 and 32.
4. Inspect errors in every JSON result; do not report a throughput number from
   a run with failed requests.
5. Run `plot` to regenerate the grouped dispatch-mode plots.
