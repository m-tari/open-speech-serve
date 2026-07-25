# open-speech-serve

**Whisper serving benchmark** (HF Transformers vs faster-whisper under concurrent load) plus a **streaming WebSocket TTFS** cell.

Inspired by [whisper-serving-bench](https://github.com/dyl5051/whisper-serving-bench).

|                | whisper-serving-bench | open-speech-serve v1   |
| -------------- | --------------------- | ---------------------- |
| Frameworks     | 5                     | 2 (HF, faster-whisper) |
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

## GPU (rented L4 / RunPod)

Needs a GPU machine with Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (e.g. a RunPod L4 pod). Expose port `8000` on the pod if you want the streaming server reachable from outside.

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

Plot CUDA cell results (writes `results/published/gpu_sweep.png` + `.md`):

```bash
pip install -e .
plot
# or: make plot
```

Published plots: [results/published/gpu_sweep.md](results/published/gpu_sweep.md).

## Layout

```
adapters/     # FrameworkAdapter: hf_transformers, faster_whisper, mock
bench/        # manifest, normalize, WER, metrics, loadgen, harness
streaming/    # FastAPI WebSocket server + TTFS client
configs/      # cell YAMLs + sweeps
scripts/      # prepare_data, run_cell, run_sweep, analyze, plot_results
docs/         # METHODOLOGY.md
results/published/  # committed plots + summary
```

## Metrics

- **Latency** p50 / p95 / p99 (warmup excluded; median across 3 passes)
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
