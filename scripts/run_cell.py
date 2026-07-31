from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.harness import load_cell_config, run_cell


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one open-speech-serve cell")
    parser.add_argument("config", help="Path to cell YAML")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--gpu-telemetry",
        action="store_true",
        help="Sample nvidia-smi during timed passes (also OSS_GPU_TELEMETRY=1)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Override cell YAML passes (e.g. 1 for telemetry sweeps)",
    )
    args = parser.parse_args()

    cfg = load_cell_config(Path(args.config))
    if args.gpu_telemetry:
        cfg["gpu_telemetry"] = True
    if args.passes is not None:
        if args.passes < 1:
            raise SystemExit("--passes must be >= 1")
        cfg["passes"] = args.passes
    payload = run_cell(cfg, Path(args.results_dir))
    s = payload["summary"]
    out = {
        "cell": payload["cell"],
        "dispatch_mode": s["dispatch_mode"],
        "latency_p50_s": s["latency_p50_s"],
        "latency_p95_s": s["latency_p95_s"],
        "service_latency_p50_s": s["service_latency_p50_s"],
        "queue_wait_p50_s": s["queue_wait_p50_s"],
        "rtf_p50": s["rtf_p50"],
        "throughput": s["throughput_audio_s_per_wall_s"],
        "wer_pooled": s["wer_pooled"],
    }
    if "gpu_telemetry" in payload:
        out["gpu_telemetry"] = {
            k: payload["gpu_telemetry"].get(k)
            for k in (
                "util_gpu_mean",
                "util_gpu_p50",
                "util_gpu_p95",
                "mem_used_mib_max",
                "n_samples",
            )
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
