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
    args = parser.parse_args()

    cfg = load_cell_config(Path(args.config))
    payload = run_cell(cfg, Path(args.results_dir))
    s = payload["summary"]
    print(
        json.dumps(
            {
                "cell": payload["cell"],
                "dispatch_mode": s["dispatch_mode"],
                "latency_p50_s": s["latency_p50_s"],
                "latency_p95_s": s["latency_p95_s"],
                "service_latency_p50_s": s["service_latency_p50_s"],
                "queue_wait_p50_s": s["queue_wait_p50_s"],
                "rtf_p50": s["rtf_p50"],
                "throughput": s["throughput_audio_s_per_wall_s"],
                "wer_pooled": s["wer_pooled"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
