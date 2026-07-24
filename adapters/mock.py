from __future__ import annotations

import time
from pathlib import Path

from adapters.base import FrameworkAdapter, TranscriptionResult


class MockAdapter(FrameworkAdapter):
    """Deterministic no-model adapter for plumbing smoke tests."""

    name = "mock"

    def __init__(self, model: str = "mock", device: str = "cpu", **_kwargs):
        self.model = model
        self.device = device
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if not self._loaded:
            raise RuntimeError("MockAdapter.load() not called")
        import soundfile as sf

        info = sf.info(str(audio_path))
        duration = float(info.duration)
        # Simulate a tiny fixed cost proportional to duration.
        time.sleep(min(0.05, 0.01 * duration))
        t0 = time.perf_counter()
        text = f"mock transcript for {audio_path.name}"
        latency = time.perf_counter() - t0 + 0.01
        return TranscriptionResult(
            text=text,
            latency_s=latency,
            audio_duration_s=duration,
            extra={"framework": self.name},
        )
