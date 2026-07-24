from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latest cell results")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    root = Path(args.results_dir)
    files = sorted(root.glob("*_latest.json"))
    if not files:
        print(f"No *_latest.json in {root}")
        return

    rows = []
    for path in files:
        data = json.loads(path.read_text())
        s = data.get("summary", {})
        cfg = data.get("config", {})
        rows.append(
            {
                "cell": data.get("cell"),
                "framework": cfg.get("framework"),
                "model": cfg.get("model"),
                "device": cfg.get("device"),
                "concurrency": cfg.get("concurrency"),
                "latency_p50_s": s.get("latency_p50_s"),
                "latency_p95_s": s.get("latency_p95_s"),
                "rtf_p50": s.get("rtf_p50"),
                "throughput": s.get("throughput_audio_s_per_wall_s"),
                "wer_pooled": s.get("wer_pooled"),
                "file": str(path),
            }
        )

    # Markdown table for README paste.
    print("| cell | framework | model | conc | p50 (s) | p95 (s) | RTF p50 | throughput | WER |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['cell']} | {r['framework']} | {r['model']} | {r['concurrency']} | "
            f"{_fmt(r['latency_p50_s'])} | {_fmt(r['latency_p95_s'])} | "
            f"{_fmt(r['rtf_p50'])} | {_fmt(r['throughput'])} | {_fmt(r['wer_pooled'])} |"
        )


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    main()
