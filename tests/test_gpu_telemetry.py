from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from bench.gpu_telemetry import (
    load_samples_csv,
    parse_sample_line,
    summarize_samples,
    want_gpu_telemetry,
)
from bench.harness import run_cell


def test_parse_and_summarize(tmp_path):
    sample = parse_sample_line("2026/07/31 12:00:00.000, 42, 10, 8192, 120.5")
    assert sample is not None
    assert sample["util_gpu"] == 42.0

    summary = summarize_samples(
        [
            {"util_gpu": 10.0, "util_mem": 5.0, "mem_used_mib": 1000.0, "power_w": 100.0},
            {"util_gpu": 50.0, "util_mem": 20.0, "mem_used_mib": 2000.0, "power_w": 150.0},
            {"util_gpu": 90.0, "util_mem": 40.0, "mem_used_mib": 3000.0, "power_w": 200.0},
        ],
        csv_path="x.csv",
    )
    assert summary["util_gpu_mean"] == 50.0
    assert summary["mem_used_mib_max"] == 3000.0

    path = tmp_path / "telem.csv"
    path.write_text("timestamp,util_gpu,util_mem,mem_used_mib,power_w\nt0,10,5,1000,100\n")
    assert load_samples_csv(path)[0]["util_gpu"] == 10.0


def test_want_gpu_telemetry(monkeypatch):
    assert want_gpu_telemetry({"gpu_telemetry": True}) is True
    assert want_gpu_telemetry({}) is False
    monkeypatch.setenv("OSS_GPU_TELEMETRY", "1")
    assert want_gpu_telemetry({}) is True


def test_harness_attaches_gpu_telemetry_when_smi_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("bench.harness.GpuTelemetrySampler.start", lambda self: False)
    monkeypatch.setattr(
        "bench.harness.GpuTelemetrySampler.stop",
        lambda self: {
            "n_samples": 0,
            "interval_s": 0.5,
            "util_gpu_mean": float("nan"),
            "error": "nvidia-smi not available",
            "csv_path": "gpu_telemetry/x.csv",
        },
    )
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16_000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio_path": str(wav), "reference": "sample"}) + "\n")
    payload = run_cell(
        {
            "name": "mock_telem",
            "framework": "mock",
            "model": "mock",
            "device": "cpu",
            "concurrency": 1,
            "dispatch_mode": "serialized",
            "manifest": str(manifest),
            "warmup": 0,
            "passes": 1,
            "gpu_telemetry": True,
        },
        tmp_path / "results",
    )
    assert payload["gpu_telemetry"]["n_samples"] == 0
    assert payload["gpu_telemetry"]["csv_path"] == "gpu_telemetry/mock_telem_serialized.csv"
