from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from adapters.base import FrameworkAdapter, TranscriptionResult
from bench.loadgen import timed_load


class ProbeAdapter(FrameworkAdapter):
    name = "probe"

    def __init__(self, supports_concurrent: bool, delay_s: float = 0.03):
        self.supports_concurrent = supports_concurrent
        self.delay_s = delay_s
        self.active = 0
        self.peak_active = 0
        self._counter_lock = threading.Lock()

    def load(self) -> None:
        pass

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        with self._counter_lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        started = time.perf_counter()
        time.sleep(self.delay_s)
        service = time.perf_counter() - started
        with self._counter_lock:
            self.active -= 1
        return TranscriptionResult(
            text=audio_path.name,
            latency_s=service,
            audio_duration_s=1.0,
        )


def _items(n: int) -> list[dict]:
    return [{"audio_path": f"{i}.wav"} for i in range(n)]


def test_serialized_dispatch_has_one_in_flight_and_queue_wait():
    adapter = ProbeAdapter(supports_concurrent=True)
    records, wall = timed_load(
        adapter,
        _items(4),
        concurrency=4,
        dispatch_mode="serialized",
    )

    assert adapter.peak_active == 1
    assert wall >= adapter.delay_s * 3
    assert max(r.queue_wait_s or 0 for r in records) >= adapter.delay_s


def test_concurrent_dispatch_reaches_backend_in_parallel():
    adapter = ProbeAdapter(supports_concurrent=True)
    records, wall = timed_load(
        adapter,
        _items(4),
        concurrency=4,
        dispatch_mode="concurrent",
    )

    assert adapter.peak_active > 1
    assert wall < adapter.delay_s * 3
    assert all(r.service_latency_s is not None for r in records)


def test_concurrent_dispatch_rejects_unsafe_adapter():
    adapter = ProbeAdapter(supports_concurrent=False)
    with pytest.raises(ValueError, match="does not support concurrent"):
        timed_load(
            adapter,
            _items(2),
            concurrency=2,
            dispatch_mode="concurrent",
        )


def test_invalid_dispatch_and_concurrency_are_rejected():
    adapter = ProbeAdapter(supports_concurrent=True)
    with pytest.raises(ValueError, match="dispatch_mode"):
        timed_load(adapter, _items(1), 1, dispatch_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="concurrency"):
        timed_load(adapter, _items(1), 0)
