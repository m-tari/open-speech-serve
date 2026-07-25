from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


@dataclass
class RequestRecord:
    audio_path: str
    latency_s: float
    audio_duration_s: float
    text: str
    service_latency_s: float | None = None
    queue_wait_s: float | None = None
    reference: str | None = None
    wer: float | None = None
    error: str | None = None
    pass_idx: int = 0
    worker_id: int = 0

    @property
    def rtf(self) -> float:
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.latency_s / self.audio_duration_s


@dataclass
class AggregateMetrics:
    n: int
    n_errors: int
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    latency_mean_s: float
    service_latency_p50_s: float
    service_latency_p95_s: float
    queue_wait_p50_s: float
    queue_wait_p95_s: float
    rtf_p50: float
    rtf_mean: float
    throughput_audio_s_per_wall_s: float
    wer_mean: float | None
    wer_pooled: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(records: list[RequestRecord], wall_s: float) -> AggregateMetrics:
    ok = [r for r in records if r.error is None]
    latencies = [r.latency_s for r in ok]
    service_latencies = [
        r.service_latency_s for r in ok if r.service_latency_s is not None
    ]
    queue_waits = [r.queue_wait_s for r in ok if r.queue_wait_s is not None]
    rtfs = [r.rtf for r in ok]
    audio_total = sum(r.audio_duration_s for r in ok)
    wers = [r.wer for r in ok if r.wer is not None]

    wer_pooled = None
    pairs = [(r.reference, r.text) for r in ok if r.reference]
    if pairs:
        from bench.wer import pooled_wer

        wer_pooled = pooled_wer(pairs)

    return AggregateMetrics(
        n=len(records),
        n_errors=len(records) - len(ok),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        latency_p99_s=percentile(latencies, 99),
        latency_mean_s=statistics.fmean(latencies) if latencies else float("nan"),
        service_latency_p50_s=percentile(service_latencies, 50),
        service_latency_p95_s=percentile(service_latencies, 95),
        queue_wait_p50_s=percentile(queue_waits, 50),
        queue_wait_p95_s=percentile(queue_waits, 95),
        rtf_p50=percentile(rtfs, 50),
        rtf_mean=statistics.fmean(rtfs) if rtfs else float("nan"),
        throughput_audio_s_per_wall_s=(audio_total / wall_s) if wall_s > 0 else 0.0,
        wer_mean=statistics.fmean(wers) if wers else None,
        wer_pooled=wer_pooled,
    )
