# open-speech-serve

**Whisper serving benchmark** across HF Transformers, faster-whisper, vLLM,
SGLang, and TensorRT-LLM/Triton, plus a **streaming WebSocket TTFS** cell.

Inspired by [whisper-serving-bench](https://github.com/dyl5051/whisper-serving-bench).

|                | whisper-serving-bench | open-speech-serve      |
| -------------- | --------------------- | ---------------------- |
| Frameworks     | 5                     | 5                      |
| GPUs           | A100 + L4             | 1 (L4 recommended)     |
| Concurrency    | 1/8/32/64/128         | 1/8/32                 |
| Streaming TTFS | no                    | yes                    |
| Docker-first   | yes                   | yes                    |

## Quick start (Docker / CPU)

```bash
make build
make prepare
make mock                                          # plumbing only
make smoke                                         # mock + faster-whisper tiny
```

Streaming server + TTFS:

```bash
make server          # in one terminal (or: docker compose up -d server)
make ttfs
```

## GPU v1 (in-process baselines)

Needs a GPU machine with Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
RunPod Pods do not support nested Docker; use the direct workflow in
[deploy/runpod/README.md](deploy/runpod/README.md) there.

1. **Clone and enter the repo**

```bash
git clone https://github.com/m-tari/open-speech-serve.git
cd open-speech-serve
```

2. **Confirm the GPU is visible**

```bash
nvidia-smi
```

3. **Build the GPU image**

```bash
make gpu-build
```

4. **Prepare data** (LibriSpeech; default `n=25`)

```bash
make gpu-prepare                 # override sample count with N=50
```

5. **Run a cell** (or the full sweep)

```bash
make gpu-cell                    # default: configs/cells/fw_turbo_l4_c1.yaml
make gpu-cell CELL=configs/cells/hf_turbo_l4_c8.yaml
make gpu-sweep                   # full 6-cell v1 matrix
```

**Optional — streaming server + TTFS** (two terminals)

```bash
make gpu-server                  # terminal 1 — listens on :8000
make gpu-ttfs                    # terminal 2
```

Results land in `./results`; data in `./data`.

## GPU v2 (serving frameworks)

The remote backends run in isolated environments because their CUDA/Torch
requirements conflict. On a Docker GPU VM, run the complete 15-cell publication
comparison with one command:

```bash
make v2-comparison
```

This prepares data, runs the six HF/faster-whisper baselines, starts and stops
each remote backend for its three concurrent cells, then writes:

```text
results/v2_comparison/summary.md
results/v2_comparison/published/gpu_sweep.{md,png}
```

Reuse prepared data and TensorRT engines on later runs:

```bash
make v2-comparison SKIP_PREP=1 SKIP_TRT_PREP=1
```

To run or debug one backend manually:

```bash
# vLLM
make vllm-up
make vllm-cell CELL=configs/cells/vllm_turbo_l4_c8_concurrent.yaml
make vllm-sweep

# SGLang (experimental Whisper path)
make sglang-up
make sglang-cell CELL=configs/cells/sglang_turbo_l4_c8_concurrent.yaml
make sglang-sweep

# TensorRT-LLM/Triton (prepare engines once on the target GPU)
make triton-prepare
make triton-up
make triton-cell CELL=configs/cells/trtllm_turbo_l4_c8_concurrent.yaml
make triton-sweep
```

Each remote backend has paired `serialized` and `concurrent` cells. Serialized
mode gates backend calls at one in flight while preserving concurrent arrivals;
concurrent mode lets the serving engine batch/schedule requests. Existing
HF/faster-whisper adapters are serialized-only because their in-process pipeline
objects are not safely concurrent.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker and RunPod instructions.

Plot CUDA cell results (writes `results/published/gpu_sweep.png` + `.md`):

```bash
pip install -e .
plot
# or: make plot
```

Published plots: [results/published/gpu_sweep.md](results/published/gpu_sweep.md).

## Layout

```
adapters/     # local, OpenAI transcription, and Triton clients
bench/        # manifest, normalize, WER, metrics, loadgen, harness
streaming/    # FastAPI WebSocket server + TTFS client
configs/      # cell YAMLs + sweeps
scripts/      # prepare_data, run_cell, run_sweep, analyze, plot_results
docs/         # methodology + deployment
results/published/  # committed plots + summary
```

## Metrics

- **Latency** p50 / p95 / p99 (warmup excluded; median across 3 passes)
- **Service latency / queue wait** split for serialized-vs-concurrent analysis
- **RTF** = wall_s / audio_s (lower is faster)
- **Throughput** = total audio seconds / wall seconds under concurrency
- **WER** when references exist (shared normalizer)
- **TTFS** for streaming: EOS → final transcript

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Env knobs

| Variable           | Default                                       | Meaning        |
| ------------------ | --------------------------------------------- | -------------- |
| `OSS_FRAMEWORK`    | `faster_whisper`                              | server backend |
| `OSS_MODEL`        | `tiny` (CPU) / `large-v3-turbo` (GPU compose) | Whisper size   |
| `OSS_DEVICE`       | `cpu` / `cuda`                                | device         |
| `OSS_COMPUTE_TYPE` | `int8` / `float16`                            | compute type   |

## License

MIT.
