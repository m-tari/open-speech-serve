| cell | framework | mode | conc | e2e p50 | service p50 | queue p50 | p95 | RTF | throughput | WER |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fw_turbo_c1 | faster_whisper | serialized | 1 | 0.213 | 0.213 | 0.000 | 0.399 | 0.040 | 29.504 | 0.023 |
| fw_turbo_c32 | faster_whisper | serialized | 32 | 3.939 | 0.210 | 3.735 | 6.693 | 0.528 | 29.127 | 0.023 |
| fw_turbo_c8 | faster_whisper | serialized | 8 | 1.787 | 0.218 | 1.545 | 3.719 | 0.287 | 28.507 | 0.023 |
| hf_turbo_c1 | hf_transformers | serialized | 1 | 0.174 | 0.174 | 0.000 | 0.361 | 0.037 | 32.825 | 0.021 |
| hf_turbo_c32 | hf_transformers | serialized | 32 | 3.315 | 0.181 | 3.116 | 6.255 | 0.460 | 31.759 | 0.021 |
| hf_turbo_c8 | hf_transformers | serialized | 8 | 1.692 | 0.173 | 1.456 | 2.648 | 0.263 | 32.665 | 0.021 |
| sglang_turbo_c1_concurrent | sglang | concurrent | 1 | 0.148 | 0.148 | 0.000 | 0.189 | 0.030 | 45.985 | 0.021 |
| sglang_turbo_c32_concurrent | sglang | concurrent | 32 | 2.537 | 2.527 | 0.000 | 2.595 | 0.378 | 0.000 | 0.021 |
| sglang_turbo_c8_concurrent | sglang | concurrent | 8 | 0.737 | 0.730 | 0.002 | 1.288 | 0.142 | 68.407 | 0.021 |
| trtllm_turbo_c1_concurrent | tensorrt_llm | concurrent | 1 | 0.043 | 0.042 | 0.001 | 0.100 | 0.008 | 131.142 | 0.052 |
| trtllm_turbo_c32_concurrent | tensorrt_llm | concurrent | 32 | 0.445 | 0.441 | 0.004 | 0.651 | 0.060 | 302.902 | 0.052 |
| trtllm_turbo_c8_concurrent | tensorrt_llm | concurrent | 8 | 0.165 | 0.162 | 0.001 | 0.335 | 0.026 | 310.719 | 0.052 |
| vllm_turbo_c1_concurrent | vllm | concurrent | 1 | 0.055 | 0.054 | 0.000 | 0.098 | 0.011 | 113.832 | 0.021 |
| vllm_turbo_c32_concurrent | vllm | concurrent | 32 | 0.672 | 0.662 | 0.003 | 0.747 | 0.095 | 255.186 | 0.021 |
| vllm_turbo_c8_concurrent | vllm | concurrent | 8 | 0.189 | 0.188 | 0.000 | 0.446 | 0.034 | 242.380 | 0.021 |
