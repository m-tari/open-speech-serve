from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Literal

from adapters.base import FrameworkAdapter
from bench.metrics import RequestRecord
from bench.wer import wer as compute_wer

DispatchMode = Literal["serialized", "concurrent"]


def run_load(
    adapter: FrameworkAdapter,
    items: list[dict],
    concurrency: int,
    pass_idx: int = 0,
    dispatch_mode: DispatchMode = "serialized",
) -> list[RequestRecord]:
    """
    Concurrent offline transcription load.

    ``serialized`` reproduces the original benchmark: concurrent clients queue
    at a client-side gate. ``concurrent`` sends all requests to a backend that
    explicitly declares concurrent calls safe.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if dispatch_mode not in {"serialized", "concurrent"}:
        raise ValueError(
            f"Unknown dispatch_mode {dispatch_mode!r}; choose serialized or concurrent"
        )
    if dispatch_mode == "concurrent" and not adapter.supports_concurrent:
        raise ValueError(
            f"{adapter.name} does not support concurrent dispatch; "
            "use dispatch_mode: serialized"
        )

    lock = threading.Lock()
    records: list[RequestRecord] = []

    def _one(worker_id: int, item: dict) -> RequestRecord:
        audio = Path(item["audio_path"])
        reference = item.get("reference")
        try:
            t0 = time.perf_counter()
            if dispatch_mode == "serialized":
                with lock:
                    result = adapter.transcribe(audio)
            else:
                result = adapter.transcribe(audio)
            e2e = time.perf_counter() - t0
            service = result.latency_s
            queue_wait = max(0.0, e2e - service)
            w = None
            if reference:
                w = compute_wer(reference, result.text)
            return RequestRecord(
                audio_path=str(audio),
                latency_s=e2e,
                audio_duration_s=result.audio_duration_s,
                text=result.text,
                service_latency_s=service,
                queue_wait_s=queue_wait,
                reference=reference,
                wer=w,
                pass_idx=pass_idx,
                worker_id=worker_id,
            )
        except Exception as exc:  # noqa: BLE001 — record and continue
            return RequestRecord(
                audio_path=str(audio),
                latency_s=0.0,
                audio_duration_s=0.0,
                text="",
                service_latency_s=None,
                queue_wait_s=None,
                reference=reference,
                error=f"{type(exc).__name__}: {exc}",
                pass_idx=pass_idx,
                worker_id=worker_id,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = []
        for i, item in enumerate(items):
            futs.append(pool.submit(_one, i % concurrency, item))
        for fut in concurrent.futures.as_completed(futs):
            records.append(fut.result())

    return records


def timed_load(
    adapter: FrameworkAdapter,
    items: list[dict],
    concurrency: int,
    pass_idx: int = 0,
    dispatch_mode: DispatchMode = "serialized",
) -> tuple[list[RequestRecord], float]:
    t0 = time.perf_counter()
    records = run_load(
        adapter,
        items,
        concurrency,
        pass_idx=pass_idx,
        dispatch_mode=dispatch_mode,
    )
    return records, time.perf_counter() - t0
