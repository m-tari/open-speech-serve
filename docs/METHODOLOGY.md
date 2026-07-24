# Methodology

Compact v1 of a [whisper-serving-bench](https://github.com/dyl5051/whisper-serving-bench)-style matrix, plus one streaming TTFS cell.

## What we measure

### Offline concurrent cells (6 GPU cells)

| Axis | v1 choice |
|---|---|
| Frameworks | `hf_transformers` (no-batching baseline), `faster_whisper` (CTranslate2) |
| Model | `large-v3-turbo` on GPU; `tiny` for CPU smoke |
| GPU | one L4 (or whatever `nvidia-smi` reports) |
| Concurrency | 1 / 8 / 32 |
| Warmup | excluded from aggregates |
| Passes | 3; report **median** across passes |

Per request we record latency, audio duration, RTF (`latency / audio_duration`), hypothesis text, and WER when a reference exists.

### Streaming TTFS cell

**TTFS** = time from client `EOS` (end of speech) to server `final` transcript.

Protocol: raw PCM s16le mono @ 16 kHz over WebSocket; server emits `partial` every ~1 s of audio and `final` after `EOS`.

This is **chunked re-decode**, not incremental encoder-state streaming. That limitation is intentional and documented — matching the honesty bar of whisper-serving-bench, while still measuring a real serving path.

## What we are not measuring

- True causal / duplex streaming with incremental decoder state
- Continuous batching across streams (neither HF nor vanilla faster-whisper does this well)
- Absolute SOTA WER (weights held constant across frameworks)
- Cost-per-audio-hour across cloud SKUs (can add in v1.1)

## Text normalization

Shared lowercasing + punctuation strip before WER (`bench/normalize.py`). Identical across frameworks so WER deltas reflect decoding differences, not normalizer drift.

## Environment capture

Every result JSON includes hostname, Python version, git SHA, GPU name, and the resolved cell config.

## Reproduce

```bash
# CPU smoke
make build && make smoke

# GPU v1 (rented L4)
make gpu-build
make gpu-prepare
make gpu-cell CELL=configs/cells/fw_turbo_l4_c1.yaml
make gpu-sweep
```
