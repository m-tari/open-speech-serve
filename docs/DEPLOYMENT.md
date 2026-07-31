# Serving backend deployment

The v2 benchmark uses a lightweight benchmark client and one isolated GPU
backend. Do not install vLLM, SGLang, and TensorRT-LLM into one Python
environment: their pinned Torch/CUDA stacks differ.

All Docker workflows use plain `docker` via [`scripts/docker.sh`](../scripts/docker.sh)
— **Compose is not required**.

## Docker GPU VM

Requirements: x86_64 Linux, NVIDIA driver, Docker, and NVIDIA Container
Toolkit. Confirm:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

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
# vLLM 0.25.1 (builds open-speech-serve:vllm-audio with Whisper audio deps)
make vllm-up
make vllm-cell CELL=configs/cells/vllm_turbo_c8_serialized.yaml
make vllm-cell CELL=configs/cells/vllm_turbo_c8_concurrent.yaml
make stop-vllm

# SGLang 0.5.15 (Whisper support is experimental)
make sglang-up
make sglang-cell CELL=configs/cells/sglang_turbo_c8_concurrent.yaml
make stop-sglang

# Triton 26.02 + TensorRT-LLM 1.1.0
make triton-prepare       # builds GPU-specific engines; run once
make triton-up
make triton-cell CELL=configs/cells/trtllm_turbo_c8_concurrent.yaml
make stop-triton
```

Image tags can be overridden with `VLLM_IMAGE`, `VLLM_BASE_IMAGE`,
`SGLANG_IMAGE`, `TRITON_IMAGE`, and `TRTLLM_VERSION`. Record overrides with
published results.

`make vllm-up` / `build-vllm` wraps `VLLM_BASE_IMAGE` (default
`vllm/vllm-openai:v0.25.1`) into `VLLM_IMAGE` (default
`open-speech-serve:vllm-audio`) and installs the full `vllm[audio]` extra
set (plus system `ffmpeg`/`libsndfile`). Stock `vllm-openai` omits those
extras, which makes `/v1/audio/transcriptions` return HTTP 400 for valid WAV
files.

Host endpoints (bench client uses `--network host`):

- vLLM HTTP: `http://127.0.0.1:8001`
- SGLang HTTP: `http://127.0.0.1:8002`
- Triton HTTP / gRPC / metrics: `8003` / `8004` / `8005`

Override cell endpoints with `OSS_BASE_URL` or `OSS_GRPC_URL` if needed.

HF cache is stored at `./.cache/huggingface` (host bind mount). Triton engines
live under `./artifacts/triton`.

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

## GPU telemetry

`make gpu-telemetry` reuses the standard HF / vLLM / TensorRT-LLM cells with
`--gpu-telemetry --passes 1` into `results/gpu_telemetry/`. The bench client
uses `--gpus all` so `nvidia-smi` can observe the same device as the server.

`make plot-gpu-telemetry` writes util-vs-concurrency and util-vs-time (c8)
figures — no Nsight GUI required.

```bash
make gpu-telemetry
make plot-gpu-telemetry
```

## Validation before publishing

1. Confirm a real transcription smoke succeeds for vLLM/SGLang
   (`POST /v1/audio/transcriptions` with `data/tone16k.wav`), or Triton
   reports `whisper_bls` ready. `make v2-comparison` waits on that smoke
   before remote cells.
2. Run concurrency 1 first and verify WER is comparable.
3. Run paired serialized/concurrent cells at 8 and 32.
4. Inspect errors in every JSON result; do not report a throughput number from
   a run with failed requests.
5. Run `plot` to regenerate the grouped dispatch-mode plots.
6. For util/plateau writeups, confirm each telemetry JSON has
   `gpu_telemetry.n_samples > 0` before plotting.
