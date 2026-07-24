from __future__ import annotations

from bench.metrics import percentile
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
