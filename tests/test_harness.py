from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from bench.harness import run_cell


def test_harness_persists_dispatch_and_latency_split(tmp_path):
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16_000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": str(wav), "reference": "sample"}) + "\n"
    )
    results = tmp_path / "results"
    payload = run_cell(
        {
            "name": "mock_concurrency",
            "framework": "mock",
            "model": "mock",
            "device": "cpu",
            "concurrency": 2,
            "dispatch_mode": "concurrent",
            "manifest": str(manifest),
            "warmup": 0,
            "passes": 1,
        },
        results,
    )

    assert payload["cell"] == "mock_concurrency_concurrent"
    assert payload["summary"]["dispatch_mode"] == "concurrent"
    assert payload["summary"]["service_latency_p50_s"] >= 0
    assert payload["summary"]["queue_wait_p50_s"] >= 0
    assert (results / "mock_concurrency_concurrent_latest.json").exists()
