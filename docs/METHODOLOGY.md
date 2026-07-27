# Methodology

Compact [whisper-serving-bench](https://github.com/dyl5051/whisper-serving-bench)-style matrix, plus one streaming TTFS cell.

## What we measure

### Offline cells

| Axis | Choice |
|---|---|
| Frameworks | HF Transformers, faster-whisper, vLLM, SGLang, TensorRT-LLM/Triton |
| Model | `large-v3-turbo` on GPU; `tiny` for CPU smoke |
| GPU | one GPU (whatever `nvidia-smi` reports) |
| Concurrency | 1 / 8 / 32 |
| Dispatch | `serialized` or `concurrent` |
| Warmup | excluded from aggregates |
| Passes | 3; report **median** across passes |

Per request we record end-to-end latency, service latency, queue wait, audio
duration, RTF (`end-to-end latency / audio duration`), hypothesis text, errors,
and WER when a reference exists.

### Dispatch semantics

Both modes submit work through the same concurrent client pool:

- **Serialized** holds a client-side gate around the backend call. Requests may
  arrive concurrently, but only one reaches inference at a time. End-to-end
  latency therefore includes queueing. This is the behavior of the original v1
  benchmark and the only supported mode for the in-process HF/faster-whisper
  adapters.
- **Concurrent** removes that gate and sends simultaneous requests to a remote
  server. vLLM, SGLang, and Triton may batch, schedule, or queue them internally.

`service_latency_s` is measured inside the adapter call. `queue_wait_s` is
`end_to_end - service`, clamped to zero. For remote clients, service latency
includes HTTP/gRPC transport; server-only compute time is not claimed.

Concurrency 1 is the common baseline. At concurrency 8/32, compare paired
serialized/concurrent cells for the same remote framework. Do not interpret the
serialized HF/faster-whisper cells as dynamic batching.

### Streaming TTFS cell

**TTFS** = time from client `EOS` (end of speech) to server `final` transcript.

Protocol: raw PCM s16le mono @ 16 kHz over WebSocket; server emits `partial` every ~1 s of audio and `final` after `EOS`.

This is **chunked re-decode**, not incremental encoder-state streaming. That limitation is intentional and documented — matching the honesty bar of whisper-serving-bench, while still measuring a real serving path.

## What we are not measuring

- True causal / duplex streaming with incremental decoder state
- Server-internal compute time isolated from transport
- Cross-framework TTFS for the remote servers
- Absolute SOTA WER (weights held constant across frameworks)
- Cost-per-audio-hour across cloud SKUs (can add in v1.1)

## Text normalization

Shared lowercasing + punctuation strip before WER (`bench/normalize.py`). Identical across frameworks so WER deltas reflect decoding differences, not normalizer drift.

## Environment capture

Every result JSON includes hostname, Python version, git SHA, GPU name, resolved
cell config, adapter/server identity, dispatch mode, and per-request records.

## Fairness controls

- Exact `openai/whisper-large-v3-turbo` weights (TensorRT uses the equivalent
  OpenAI checkpoint)
- FP16, English transcription, greedy decoding / beam width 1
- Same 16 kHz WAV manifest, references, warmup, passes, and one GPU
- Same client-side latency boundary and text normalization

SGLang's Whisper path is experimental; publish its WER beside performance.
TensorRT engines are GPU-architecture and TensorRT-version specific and must be
built on the target runtime.

## Reproduce

```bash
# CPU smoke
make build && make smoke

# GPU v1
make gpu-build
make gpu-prepare
make gpu-cell CELL=configs/cells/fw_turbo_c1.yaml
make gpu-sweep

# Remote serving cells (server and client in separate terminals)
make vllm-up
make vllm-cell CELL=configs/cells/vllm_turbo_c8_concurrent.yaml
```
