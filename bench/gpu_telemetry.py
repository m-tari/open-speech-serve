"""Sample GPU utilization via nvidia-smi during timed benchmark passes."""

from __future__ import annotations

import csv
import logging
import os
import statistics
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 0.5
QUERY = "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw"
CSV_FIELDS = ["timestamp", "util_gpu", "util_mem", "mem_used_mib", "power_w"]


def nvidia_smi_available() -> bool:
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _parse_float(raw: str | None) -> float | None:
    text = (raw or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    cleaned = text.replace("%", "").replace("MiB", "").replace("W", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_sample_line(line: str) -> dict[str, Any] | None:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 5:
        return None
    util_gpu = _parse_float(parts[-4])
    mem_used = _parse_float(parts[-2])
    if util_gpu is None and mem_used is None:
        return None
    return {
        "timestamp": ",".join(parts[:-4]).strip(),
        "util_gpu": util_gpu,
        "util_mem": _parse_float(parts[-3]),
        "mem_used_mib": mem_used,
        "power_w": _parse_float(parts[-1]),
    }


def load_samples_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [
            {
                "timestamp": row.get("timestamp", ""),
                "util_gpu": _parse_float(row.get("util_gpu")),
                "util_mem": _parse_float(row.get("util_mem")),
                "mem_used_mib": _parse_float(row.get("mem_used_mib")),
                "power_w": _parse_float(row.get("power_w")),
            }
            for row in csv.DictReader(f)
        ]


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    csv_path: str | None = None,
) -> dict[str, Any]:
    util_gpu = [float(s["util_gpu"]) for s in samples if s.get("util_gpu") is not None]
    util_mem = [float(s["util_mem"]) for s in samples if s.get("util_mem") is not None]
    mem_used = [float(s["mem_used_mib"]) for s in samples if s.get("mem_used_mib") is not None]
    power_w = [float(s["power_w"]) for s in samples if s.get("power_w") is not None]
    util_sorted = sorted(util_gpu)
    summary: dict[str, Any] = {
        "n_samples": len(samples),
        "interval_s": interval_s,
        "util_gpu_mean": statistics.fmean(util_gpu) if util_gpu else float("nan"),
        "util_gpu_p50": _pct(util_sorted, 50),
        "util_gpu_p95": _pct(util_sorted, 95),
        "util_mem_mean": statistics.fmean(util_mem) if util_mem else float("nan"),
        "mem_used_mib_max": max(mem_used) if mem_used else float("nan"),
        "power_w_mean": statistics.fmean(power_w) if power_w else float("nan"),
    }
    if csv_path is not None:
        summary["csv_path"] = csv_path
    return summary


class GpuTelemetrySampler:
    """Background nvidia-smi sampler for timed passes only."""

    def __init__(self, csv_path: Path, *, interval_s: float = DEFAULT_INTERVAL_S) -> None:
        self.csv_path = Path(csv_path)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._error: str | None = None

    def start(self) -> bool:
        if not nvidia_smi_available():
            self._error = "nvidia-smi not available"
            logger.warning("GPU telemetry disabled: %s", self._error)
            return False
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()
        return True

    def _query_once(self) -> dict[str, Any] | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu={QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            self._error = str(exc)
            return None
        lines = out.splitlines()
        return parse_sample_line(lines[0]) if lines else None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._query_once()
            if sample is not None:
                self._samples.append(sample)
            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 4))
            self._thread = None
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._samples)
        summary = summarize_samples(
            self._samples, interval_s=self.interval_s, csv_path=str(self.csv_path)
        )
        if self._error and summary["n_samples"] == 0:
            summary["error"] = self._error
        return summary


def want_gpu_telemetry(cfg: dict[str, Any]) -> bool:
    if cfg.get("gpu_telemetry") is True:
        return True
    return str(os.environ.get("OSS_GPU_TELEMETRY", "")).lower() in {"1", "true", "yes"}
