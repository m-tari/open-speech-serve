from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path

from adapters.base import FrameworkAdapter
from bench.metrics import RequestRecord
from bench.wer import wer as compute_wer


def run_load(
    adapter: FrameworkAdapter,
    items: list[dict],
    concurrency: int,
    pass_idx: int = 0,
) -> list[RequestRecord]:
    """
    Concurrent offline transcription load.

    Note: HF Transformers and faster-whisper are largely single-request-per-GPU.
    Concurrency here measures queuing / multi-thread contention — the scaling
    cliff is the interesting result, not peak GPU occupancy.
    """
    lock = threading.Lock()
    records: list[RequestRecord] = []

    def _one(worker_id: int, item: dict) -> RequestRecord:
        audio = Path(item["audio_path"])
        reference = item.get("reference")
        try:
            # End-to-end latency includes queue wait under the adapter lock.
            # HF pipelines are not thread-safe; faster-whisper is serialized
            # the same way so both frameworks see a fair queuing cliff.
            t0 = time.perf_counter()
            with lock:
                result = adapter.transcribe(audio)
            e2e = time.perf_counter() - t0
            w = None
            if reference:
                w = compute_wer(reference, result.text)
            return RequestRecord(
                audio_path=str(audio),
                latency_s=e2e,
                audio_duration_s=result.audio_duration_s,
                text=result.text,
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
) -> tuple[list[RequestRecord], float]:
    t0 = time.perf_counter()
    records = run_load(adapter, items, concurrency, pass_idx=pass_idx)
    return records, time.perf_counter() - t0
