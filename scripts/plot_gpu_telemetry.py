"""Plot GPU util vs concurrency and util-vs-time duty-cycle (HF c8 vs vLLM c8)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from bench.gpu_telemetry import DEFAULT_INTERVAL_S, load_samples_csv

COLORS = {
    "hf_transformers": "#2563eb",
    "vllm": "#059669",
    "tensorrt_llm": "#dc2626",
    "faster_whisper": "#64748b",
    "sglang": "#7c3aed",
}


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _resolve_csv(csv_path: str | None, results_dir: Path, cell: str | None) -> Path | None:
    candidates: list[Path] = []
    if csv_path:
        p = Path(csv_path)
        candidates.extend([p, results_dir / p, results_dir / "gpu_telemetry" / p.name])
    if cell:
        candidates.append(results_dir / "gpu_telemetry" / f"{cell}.csv")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_telemetry_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*_latest.json")):
        data = json.loads(path.read_text())
        cfg = data.get("config") or {}
        summary = data.get("summary") or {}
        telem = data.get("gpu_telemetry") or {}
        if cfg.get("device") != "cuda" or cfg.get("concurrency") is None:
            continue
        cell = data.get("cell") or cfg.get("name")
        rows.append(
            {
                "cell": cell,
                "framework": cfg.get("framework"),
                "model": cfg.get("model"),
                "gpu": (data.get("env") or {}).get("gpu"),
                "concurrency": int(cfg["concurrency"]),
                "dispatch_mode": cfg.get("dispatch_mode", "serialized"),
                "throughput": summary.get("throughput_audio_s_per_wall_s"),
                "util_gpu_mean": telem.get("util_gpu_mean"),
                "util_gpu_p95": telem.get("util_gpu_p95"),
                "mem_used_mib_max": telem.get("mem_used_mib_max"),
                "n_samples": telem.get("n_samples"),
                "csv_path": _resolve_csv(telem.get("csv_path"), results_dir, cell),
                "has_telemetry": bool(telem) and _finite(telem.get("util_gpu_mean")),
            }
        )
    return rows


def _by_framework(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["framework"]:
            grouped[f"{row['framework']} [{row['dispatch_mode']}]"].append(row)
    for series in grouped.values():
        series.sort(key=lambda r: r["concurrency"])
    return dict(grouped)


def plot_vs_concurrency(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_fw = _by_framework(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    gpu = next((r["gpu"] for r in rows if r.get("gpu")), "GPU")
    model = next((r["model"] for r in rows if r.get("model")), "whisper")
    fig.suptitle(f"GPU util vs concurrency · {gpu} · {model}", fontsize=13, fontweight="bold")

    for ax, key, ylabel in (
        (axes[0], "throughput", "Throughput (audio-s / wall-s)"),
        (axes[1], "util_gpu_mean", "Mean GPU util (%)"),
    ):
        for label, series in by_fw.items():
            fw = label.split(" [", 1)[0]
            valid = [r for r in series if _finite(r.get(key))]
            if not valid:
                continue
            ax.plot(
                [r["concurrency"] for r in valid],
                [float(r[key]) for r in valid],
                marker="o",
                color=COLORS.get(fw),
                linestyle="-" if series[0]["dispatch_mode"] == "concurrent" else "--",
                linewidth=2,
                label=label,
            )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({r["concurrency"] for r in rows}))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        if key == "util_gpu_mean":
            ax.set_ylim(0, 105)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_util_vs_time(rows: list[dict], out_path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pick(fw: str, mode: str) -> dict | None:
        for r in rows:
            if (
                r.get("framework") == fw
                and r.get("dispatch_mode") == mode
                and r.get("concurrency") == 8
                and r.get("csv_path")
            ):
                return r
        return None

    hf, vllm = pick("hf_transformers", "serialized"), pick("vllm", "concurrent")
    if not hf or not vllm:
        return False

    panels = [
        (hf, COLORS["hf_transformers"], "HF Transformers [serialized]"),
        (vllm, COLORS["vllm"], "vLLM [concurrent]"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), constrained_layout=True, sharey=True)
    gpu = next((r["gpu"] for r in rows if r.get("gpu")), "GPU")
    fig.suptitle(f"GPU duty cycle at concurrency 8 · {gpu}", fontsize=13, fontweight="bold")

    for ax, (row, color, title) in zip(axes, panels):
        samples = load_samples_csv(Path(row["csv_path"]))
        ys = [float(s["util_gpu"]) for s in samples if s.get("util_gpu") is not None]
        if not ys:
            return False
        xs = [i * DEFAULT_INTERVAL_S for i in range(len(ys))]
        mean = sum(ys) / len(ys)
        ax.plot(xs, ys, color=color, linewidth=1.5)
        ax.fill_between(xs, ys, alpha=0.2, color=color)
        ax.axhline(mean, color=color, linestyle="--", linewidth=1, alpha=0.8, label=f"mean {mean:.0f}%")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time into timed passes (s)")
        ax.set_ylabel("GPU util (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def write_summary_md(rows: list[dict], out_path: Path, *, duty_cycle: bool) -> None:
    rows = sorted(rows, key=lambda r: (r["framework"] or "", r["dispatch_mode"], r["concurrency"]))
    lines = [
        "# GPU util vs concurrency",
        "",
        "From `*_latest.json` cells with `gpu_telemetry` (nvidia-smi during timed passes).",
        "",
        "| cell | framework | mode | conc | throughput | util mean % | util p95 % | mem max MiB | samples |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cell']} | {r['framework']} | {r['dispatch_mode']} | {r['concurrency']} | "
            f"{_fmt(r['throughput'])} | {_fmt(r['util_gpu_mean'])} | {_fmt(r['util_gpu_p95'])} | "
            f"{_fmt(r['mem_used_mib_max'])} | {r['n_samples'] if r['n_samples'] is not None else '—'} |"
        )
    lines += [
        "",
        "![GPU util vs concurrency](./gpu_util_vs_concurrency.png)",
        "",
        "If throughput flattens while mean GPU util is near the ceiling, the plateau is "
        "device saturation rather than client under-load.",
        "",
    ]
    if duty_cycle:
        lines += [
            "## GPU duty cycle at concurrency 8",
            "",
            "Util vs time from the same CSVs (HF serialized vs vLLM concurrent).",
            "",
            "![GPU duty cycle](./gpu_util_vs_time_c8.png)",
            "",
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GPU telemetry results")
    parser.add_argument("--results-dir", default="results/gpu_telemetry")
    parser.add_argument("--out-dir", default="results/gpu_telemetry/published")
    args = parser.parse_args()

    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    rows = load_telemetry_rows(results_dir)
    if not rows:
        raise SystemExit(f"No CUDA *_latest.json cells in {results_dir}")
    if not any(r.get("has_telemetry") for r in rows):
        raise SystemExit(f"No gpu_telemetry summaries in {results_dir}")

    conc_png = out_dir / "gpu_util_vs_concurrency.png"
    duty_png = out_dir / "gpu_util_vs_time_c8.png"
    md = out_dir / "gpu_util_vs_concurrency.md"
    plot_vs_concurrency(rows, conc_png)
    duty_ok = plot_util_vs_time(rows, duty_png)
    write_summary_md(rows, md, duty_cycle=duty_ok)
    print(f"Wrote {conc_png}")
    print(f"Wrote {duty_png}" if duty_ok else "Skipped util-vs-time (need HF c8 + vLLM c8 CSVs)")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
