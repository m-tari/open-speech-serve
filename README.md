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

## GPU (rented L4)

GPU settings live in `docker-compose.gpu.yml` (overlay).

```bash
make gpu-build
make gpu-prepare                 # LibriSpeech n=25; override with N=50
make gpu-cell                    # default: configs/cells/fw_turbo_l4_c1.yaml
make gpu-cell CELL=configs/cells/hf_turbo_l4_c8.yaml
make gpu-sweep                   # full 6-cell v1 matrix
```

Streaming on GPU:

```bash
make gpu-server
make gpu-ttfs
```

## Layout

```
adapters/     # FrameworkAdapter: hf_transformers, faster_whisper, mock
bench/        # manifest, normalize, WER, metrics, loadgen, harness
streaming/    # FastAPI WebSocket server + TTFS client
configs/      # cell YAMLs + sweeps
scripts/      # prepare_data, run_cell, run_sweep, analyze
docs/         # METHODOLOGY.md
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
