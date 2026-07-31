# GPU v2 comparison results

Publication numbers from the 15-cell concurrent comparison
(`make v2-comparison`). Model: `openai/whisper-large-v3-turbo`, FP16, English
transcription. Concurrency 1 / 8 / 32; warmup excluded; **median across 3
passes**. Metrics are defined in [METHODOLOGY.md](METHODOLOGY.md).

**Hardware:** one NVIDIA RTX 6000 Ada Generation

## Headline

On this box, concurrent serving throughput ranks:

**TensorRT-LLM/Triton > vLLM ≫ SGLang (c1/c8) > HF Transformers ≈ faster-whisper.**

In-process baselines stay ~30× realtime with flat throughput under load (client
serialization). Engineered servers scale into the hundreds of audio-seconds per
wall-second until the GPU saturates.

![GPU sweep plots](../results/published/gpu_sweep.png)

Invalid cells (majority of passes failed) are plotted as an **x** with an
`invalid` label and are **not** connected into the framework trend line. On this
sweep that is SGLang at concurrency 32 only.

## Summary matrix

Throughput = audio-seconds / wall-seconds. Latency and RTF are end-to-end p50
unless noted. SGLang at concurrency 32 is **invalid** (see below) and is shown
as `—`.

| framework       | mode       | conc | e2e p50 (s) | service p50 (s) | queue p50 (s) | p95 (s) | RTF p50 | throughput |   WER |
| --------------- | ---------- | ---: | ----------: | --------------: | ------------: | ------: | ------: | ---------: | ----: |
| faster_whisper  | serialized |    1 |       0.213 |           0.213 |         0.000 |   0.399 |   0.040 |       29.5 | 0.023 |
| faster_whisper  | serialized |    8 |       1.787 |           0.218 |         1.545 |   3.719 |   0.287 |       28.5 | 0.023 |
| faster_whisper  | serialized |   32 |       3.939 |           0.210 |         3.735 |   6.693 |   0.528 |       29.1 | 0.023 |
| hf_transformers | serialized |    1 |       0.174 |           0.174 |         0.000 |   0.361 |   0.037 |       32.8 | 0.021 |
| hf_transformers | serialized |    8 |       1.692 |           0.173 |         1.456 |   2.648 |   0.263 |       32.7 | 0.021 |
| hf_transformers | serialized |   32 |       3.315 |           0.181 |         3.116 |   6.255 |   0.460 |       31.8 | 0.021 |
| sglang          | concurrent |    1 |       0.148 |           0.148 |         0.000 |   0.189 |   0.030 |       46.0 | 0.021 |
| sglang          | concurrent |    8 |       0.737 |           0.730 |         0.002 |   1.288 |   0.142 |       68.4 | 0.021 |
| sglang          | concurrent |   32 |           — |               — |             — |       — |       — |          — |     — |
| tensorrt_llm    | concurrent |    1 |       0.043 |           0.042 |         0.001 |   0.100 |   0.008 |      131.1 | 0.052 |
| tensorrt_llm    | concurrent |    8 |       0.165 |           0.162 |         0.001 |   0.335 |   0.026 |      310.7 | 0.052 |
| tensorrt_llm    | concurrent |   32 |       0.445 |           0.441 |         0.004 |   0.651 |   0.060 |      302.9 | 0.052 |
| vllm            | concurrent |    1 |       0.055 |           0.054 |         0.000 |   0.098 |   0.011 |      113.8 | 0.021 |
| vllm            | concurrent |    8 |       0.189 |           0.188 |         0.000 |   0.446 |   0.034 |      242.4 | 0.021 |
| vllm            | concurrent |   32 |       0.672 |           0.662 |         0.003 |   0.747 |   0.095 |      255.2 | 0.021 |

Raw cell JSON and the machine-generated table live under
`results/v2_comparison/` when you run the sweep locally. Committed plot + table:
[results/published/gpu_sweep.md](../results/published/gpu_sweep.md).

## Findings

- **Serialized HF / faster-whisper:** service latency stays ~flat (~0.17–0.22 s);
  end-to-end latency is almost all client-side queue wait; throughput stays
  ~29–33×. Expected for one-at-a-time in-process inference.
- **Concurrent engines:** client queue ≈ 0; e2e ≈ service. Throughput rises with
  concurrency then plateaus (TensorRT-LLM ~310, vLLM ~250 at c8–c32).
- **WER:** HF, vLLM, and SGLang (c1/c8) agree at ~2.1%; faster-whisper ~2.3%.
  TensorRT-LLM is ~5.2% — a real quality gap on this engine/build, not noise.
- **SGLang c1/c8** are stable but ~3–4× slower than vLLM. The Whisper serving
  path is still experimental upstream.

## GPU util vs concurrency (telemetry)

From `make gpu-telemetry` on **NVIDIA RTX 6000 Ada** with **N=205**, **3 passes**,
`nvidia-smi` every 0.5s during timed passes. Full table:
[`results/gpu_telemetry/published/gpu_util_vs_concurrency.md`](../results/gpu_telemetry/published/gpu_util_vs_concurrency.md).

| framework | mode | conc | throughput | util_gpu_mean % | util_gpu_p95 % | mem_used_mib_max | samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hf_transformers | serialized | 1 | 32.0 | 38.3 | 56 | 2490 | 285 |
| hf_transformers | serialized | 8 | 31.9 | 39.8 | 58 | 2604 | 285 |
| hf_transformers | serialized | 32 | 31.9 | 39.3 | 58 | 3100 | 286 |
| vllm | concurrent | 1 | 112.4 | 46.4 | 55 | 44544 | 82 |
| vllm | concurrent | 8 | 278.5 | 82.3 | 100 | 44544 | 33 |
| vllm | concurrent | 32 | 305.7 | 92.9 | 100 | 44544 | 30 |
| tensorrt_llm | concurrent | 1 | 174.2 | 71.6 | 79 | 12018 | 55 |
| tensorrt_llm | concurrent | 8 | 347.0 | 90.0 | 99 | 12050 | 27 |
| tensorrt_llm | concurrent | 32 | 331.4 | 97.0 | 99 | 12066 | 28 |

**Findings:**

- **Throughput plateaus near c8 for TensorRT-LLM** (~347→331×) while **vLLM
  still climbs modestly** c8→c32 (~278→306×) as mean util rises 82%→93%.
- **GPU util explains the knee:** engines jump from moderate util at c1 to
  ~82–90% mean / ~99–100% p95 at c8; HF stays ~39% mean under serialization no
  matter the concurrency.
- **Memory residency separates the stacks:** HF ~2.5–3.1 GiB, TensorRT-LLM
  ~12 GiB, vLLM ~44 GiB (`--gpu-memory-utilization 0.90`).
- **Util-vs-time at c8** contrasts HF’s mid-util series with vLLM’s dense high
  occupancy — the duty-cycle figure the short N=30 run could not produce.

Plots:

- [`gpu_util_vs_concurrency.png`](../results/gpu_telemetry/published/gpu_util_vs_concurrency.png)
- [`gpu_util_vs_time_c8.png`](../results/gpu_telemetry/published/gpu_util_vs_time_c8.png)

## SGLang concurrency 32 (invalid)

`sglang_turbo_c32_concurrent` is **not a valid datapoint**. Under 32 in-flight
transcriptions the SGLang server hits request timeouts in
`tokenizer_manager._wait_one_response`, returns HTTP 500s, then
`running_phase_sigquit_handler` kills the process tree. Later passes see a dead
server; failed passes report throughput `0`, so the **median across passes**
collapses to `0` while latency/WER can still reflect the one good pass.

This matches SGLang’s own Whisper guidance: the path is experimental and they
advise keeping Whisper at **encoder batch size 1**
([Whisper ASR cookbook](https://sgl-project.github.io/sglang-omni/cookbook/whisper_asr.html)).
Capping `--max-running-requests` can avoid crashes but does not unlock true
concurrent Whisper scaling comparable to vLLM/TensorRT-LLM.

## Reproduce

```bash
make v2-comparison
# later runs with data/engines already prepared:
make v2-comparison SKIP_PREP=1 SKIP_TRT_PREP=1
```

Outputs:

```text
results/v2_comparison/summary.md
results/v2_comparison/published/gpu_sweep.{md,png}
```
