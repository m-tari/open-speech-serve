from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from adapters.registry import get_adapter
from bench.loadgen import timed_load
from bench.metrics import AggregateMetrics, RequestRecord, aggregate
from bench.normalize import load_manifest


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parents[1],
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _gpu_name() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def capture_env() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "gpu": _gpu_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def load_cell_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Cell config must be a mapping: {path}")
    required = {"framework", "model", "concurrency", "manifest"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Cell config missing keys {missing}: {path}")
    dispatch_mode = cfg.get("dispatch_mode", "serialized")
    if dispatch_mode not in {"serialized", "concurrent"}:
        raise ValueError(
            f"Invalid dispatch_mode {dispatch_mode!r}: choose serialized or concurrent"
        )
    if int(cfg["concurrency"]) < 1:
        raise ValueError("concurrency must be >= 1")
    return cfg


def run_cell(cfg: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    device = cfg.get("device") or os.environ.get("OSS_DEVICE", "cpu")
    compute_type = cfg.get("compute_type", "default")
    warmup = int(cfg.get("warmup", 2))
    passes = int(cfg.get("passes", 3))
    concurrency = int(cfg["concurrency"])
    dispatch_mode = cfg.get("dispatch_mode", "serialized")
    limit = cfg.get("limit")

    manifest_path = Path(cfg["manifest"])
    items = load_manifest(manifest_path)
    if limit is not None:
        items = items[: int(limit)]
    if not items:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    reserved = {
        "name", "framework", "model", "device", "compute_type", "concurrency",
        "manifest", "warmup", "passes", "limit", "dispatch_mode",
    }
    adapter_options = {k: v for k, v in cfg.items() if k not in reserved}
    if os.environ.get("OSS_BASE_URL"):
        adapter_options["base_url"] = os.environ["OSS_BASE_URL"]
    if os.environ.get("OSS_GRPC_URL"):
        adapter_options["grpc_url"] = os.environ["OSS_GRPC_URL"]
    adapter = get_adapter(
        cfg["framework"],
        model=cfg["model"],
        device=device,
        compute_type=compute_type,
        **adapter_options,
    )
    adapter.load()

    # Warmup — excluded from aggregates.
    warmup_items = items[: min(warmup, len(items))]
    if warmup_items:
        timed_load(
            adapter,
            warmup_items,
            concurrency=1,
            pass_idx=-1,
            dispatch_mode="serialized",
        )

    all_records: list[RequestRecord] = []
    pass_metrics: list[AggregateMetrics] = []
    for p in range(passes):
        records, wall = timed_load(
            adapter,
            items,
            concurrency,
            pass_idx=p,
            dispatch_mode=dispatch_mode,
        )
        all_records.extend(records)
        pass_metrics.append(aggregate(records, wall))

    # Report medians across passes (whisper-serving-bench style).
    def _median(attr: str) -> float:
        vals = [getattr(m, attr) for m in pass_metrics]
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            return float("nan")
        vals.sort()
        return vals[len(vals) // 2]

    summary = {
        "latency_p50_s": _median("latency_p50_s"),
        "latency_p95_s": _median("latency_p95_s"),
        "latency_p99_s": _median("latency_p99_s"),
        "service_latency_p50_s": _median("service_latency_p50_s"),
        "service_latency_p95_s": _median("service_latency_p95_s"),
        "queue_wait_p50_s": _median("queue_wait_p50_s"),
        "queue_wait_p95_s": _median("queue_wait_p95_s"),
        "rtf_p50": _median("rtf_p50"),
        "throughput_audio_s_per_wall_s": _median("throughput_audio_s_per_wall_s"),
        "wer_pooled": _median("wer_pooled") if any(m.wer_pooled is not None for m in pass_metrics) else None,
        "n_requests_per_pass": len(items),
        "passes": passes,
        "warmup": warmup,
        "concurrency": concurrency,
        "dispatch_mode": dispatch_mode,
    }

    cell_name = cfg.get("name") or f"{adapter.name}_{cfg['model']}_c{concurrency}"
    if "dispatch_mode" in cfg and not str(cell_name).endswith(f"_{dispatch_mode}"):
        cell_name = f"{cell_name}_{dispatch_mode}"
    payload = {
        "cell": cell_name,
        "config": {
            **cfg,
            "device": device,
            "compute_type": compute_type,
            "dispatch_mode": dispatch_mode,
        },
        "adapter": adapter.info(),
        "env": capture_env(),
        "summary": summary,
        "passes": [m.to_dict() for m in pass_metrics],
        "records": [asdict(r) for r in all_records],
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = results_dir / f"{cell_name}_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2))
    # Also write a stable "latest" pointer for the cell.
    (results_dir / f"{cell_name}_latest.json").write_text(json.dumps(payload, indent=2))
    adapter.unload()
    return payload
