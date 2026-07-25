from __future__ import annotations

from bench.metrics import RequestRecord, aggregate, percentile
from bench.normalize import normalize_text
from bench.wer import wer


def test_normalize_text():
    assert normalize_text("Hello, World!") == "hello world"


def test_wer_identical():
    assert wer("hello world", "Hello, World!") == 0.0


def test_percentile():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 50) == 3.0


def test_mock_adapter(tmp_path):
    import soundfile as sf
    import numpy as np

    from adapters.mock import MockAdapter

    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
    a = MockAdapter()
    a.load()
    r = a.transcribe(wav)
    assert r.audio_duration_s > 0
    assert "mock" in r.text


def test_aggregate_reports_service_and_queue_latency():
    records = [
        RequestRecord(
            audio_path="a.wav",
            latency_s=0.3,
            service_latency_s=0.2,
            queue_wait_s=0.1,
            audio_duration_s=1.0,
            text="a",
        ),
        RequestRecord(
            audio_path="b.wav",
            latency_s=0.5,
            service_latency_s=0.2,
            queue_wait_s=0.3,
            audio_duration_s=1.0,
            text="b",
        ),
    ]
    metrics = aggregate(records, wall_s=0.5)
    assert metrics.service_latency_p50_s == 0.2
    assert metrics.queue_wait_p50_s == 0.2
