from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _is_invalid_cell(data: dict) -> bool:
    """True when a majority of passes failed (e.g. SGLang c32 server crash)."""
    passes = data.get("passes") or []
    if not passes:
        return False
    failed = sum(1 for p in passes if (p.get("n_errors") or 0) > 0)
    if failed > len(passes) / 2:
        return True
    thr = (data.get("summary") or {}).get("throughput_audio_s_per_wall_s")
    try:
        thr_f = float(thr) if thr is not None else None
    except (TypeError, ValueError):
        thr_f = None
    return bool(failed and thr_f is not None and thr_f == 0.0)


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _load_gpu_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*_latest.json")):
        data = json.loads(path.read_text())
        cfg = data.get("config") or {}
        summary = data.get("summary") or {}
        if not cfg or cfg.get("device") != "cuda":
            continue
        if cfg.get("concurrency") is None:
            continue
        rows.append(
            {
                "cell": data.get("cell") or cfg.get("name"),
                "framework": cfg.get("framework"),
                "model": cfg.get("model"),
                "gpu": (data.get("env") or {}).get("gpu"),
                "concurrency": int(cfg["concurrency"]),
                "dispatch_mode": cfg.get("dispatch_mode", "serialized"),
                "latency_p50_s": summary.get("latency_p50_s"),
                "latency_p95_s": summary.get("latency_p95_s"),
                "service_p50_s": summary.get("service_latency_p50_s"),
                "queue_p50_s": summary.get("queue_wait_p50_s"),
                "rtf_p50": summary.get("rtf_p50"),
                "throughput": summary.get("throughput_audio_s_per_wall_s"),
                "wer_pooled": summary.get("wer_pooled"),
                "invalid": _is_invalid_cell(data),
            }
        )
    return rows


def _series_by_framework(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["framework"]:
            label = f"{row['framework']} [{row['dispatch_mode']}]"
            grouped[label].append(row)
    for framework in grouped:
        grouped[framework].sort(key=lambda r: r["concurrency"])
    return dict(grouped)


def plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_fw = _series_by_framework(rows)
    if not by_fw:
        raise SystemExit("No CUDA cell results found to plot")

    colors = {
        "hf_transformers": "#2563eb",
        "faster_whisper": "#64748b",
        "vllm": "#059669",
        "sglang": "#7c3aed",
        "tensorrt_llm": "#dc2626",
    }
    markers = {
        "hf_transformers": "o",
        "faster_whisper": "s",
        "vllm": "^",
        "sglang": "D",
        "tensorrt_llm": "v",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    gpu = next((r["gpu"] for r in rows if r.get("gpu")), "GPU")
    model = next((r["model"] for r in rows if r.get("model")), "whisper")
    fig.suptitle(f"open-speech-serve · {gpu} · {model}", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "latency_p50_s", "Latency p50 (s)", False),
        (axes[0, 1], "latency_p95_s", "Latency p95 (s)", False),
        (axes[1, 0], "rtf_p50", "RTF p50 (wall / audio)", True),
        (axes[1, 1], "throughput", "Throughput (audio-s / wall-s)", False),
    ]

    for ax, key, ylabel, rtf_ref in panels:
        invalid_legend_added = False
        for framework, series in by_fw.items():
            base_framework = framework.split(" [", 1)[0]
            dispatch_mode = series[0]["dispatch_mode"]
            color = colors.get(base_framework, None)
            marker = markers.get(base_framework, "o")

            valid = [r for r in series if not r.get("invalid") and _finite(r.get(key))]
            invalid = [r for r in series if r.get("invalid") and _finite(r.get(key))]

            if valid:
                ax.plot(
                    [r["concurrency"] for r in valid],
                    [float(r[key]) for r in valid],
                    marker=marker,
                    color=color,
                    linestyle="-" if dispatch_mode == "concurrent" else "--",
                    linewidth=2,
                    markersize=7,
                    label=framework,
                )

            if invalid:
                xs = [r["concurrency"] for r in invalid]
                ys = [float(r[key]) for r in invalid]
                ax.scatter(
                    xs,
                    ys,
                    marker="x",
                    s=90,
                    color=color,
                    linewidths=2.5,
                    zorder=5,
                    label="invalid (server failure)" if not invalid_legend_added else None,
                )
                invalid_legend_added = True
                for r in invalid:
                    ax.annotate(
                        "invalid",
                        (r["concurrency"], float(r[key])),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=7,
                        color=color,
                    )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({r["concurrency"] for r in rows}))
        ax.grid(True, alpha=0.3)
        if rtf_ref:
            ax.axhline(1.0, color="#b45309", linestyle="--", linewidth=1, label="RTF=1")
        ax.legend(fontsize=8, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_md(rows: list[dict], out_path: Path) -> None:
    rows = sorted(
        rows,
        key=lambda r: (
            r["framework"] or "",
            r["dispatch_mode"],
            r["concurrency"],
        ),
    )
    lines = [
        "# GPU sweep results",
        "",
        "Generated by `plot` from `results/*_latest.json` CUDA cells.",
        "",
        "| cell | framework | mode | conc | e2e p50 | service p50 | queue p50 | p95 | RTF | throughput | WER |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("invalid"):
            lines.append(
                f"| {r['cell']} | {r['framework']} | {r['dispatch_mode']} | {r['concurrency']} | "
                f"— | — | — | — | — | invalid | — |"
            )
        else:
            lines.append(
                f"| {r['cell']} | {r['framework']} | {r['dispatch_mode']} | {r['concurrency']} | "
                f"{_fmt(r['latency_p50_s'])} | {_fmt(r['service_p50_s'])} | "
                f"{_fmt(r['queue_p50_s'])} | {_fmt(r['latency_p95_s'])} | "
                f"{_fmt(r['rtf_p50'])} | {_fmt(r['throughput'])} | {_fmt(r['wer_pooled'])} |"
            )
    if any(r.get("invalid") for r in rows):
        lines.extend(
            [
                "",
                "Rows marked `invalid` had a majority of passes fail (e.g. remote "
                "server crash). They are plotted with an **x** and are not connected "
                "into the framework trend line. See [docs/RESULTS.md](../../docs/RESULTS.md).",
            ]
        )
    lines.extend(
        [
            "",
            "![GPU sweep plots](./gpu_sweep.png)",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CUDA cell results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--out-dir",
        default="results/published",
        help="Directory for png + markdown summary (git-tracked)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    rows = _load_gpu_rows(results_dir)
    if not rows:
        raise SystemExit(f"No CUDA *_latest.json cells in {results_dir}")

    png = out_dir / "gpu_sweep.png"
    md = out_dir / "gpu_sweep.md"
    plot(rows, png)
    write_summary_md(rows, md)
    print(f"Wrote {png}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
