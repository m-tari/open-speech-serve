from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

from bench.metrics import percentile


async def run_one(url: str, wav_path: Path, chunk_ms: int = 100) -> dict:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        # Naive resample for fixtures already at 16k; fail loudly otherwise.
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz: {wav_path}")

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    frame = int(16000 * (chunk_ms / 1000.0)) * 2

    partials = 0
    final_text = ""
    ttfs_ms = None
    t_connect = time.perf_counter()

    async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
        # Stream as if live (real-time pacing).
        for i in range(0, len(pcm), frame):
            await ws.send(pcm[i : i + frame])
            await asyncio.sleep(chunk_ms / 1000.0)

        t_eos = time.perf_counter()
        await ws.send("EOS")

        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "partial":
                partials += 1
            elif msg.get("type") == "final":
                final_text = msg.get("text", "")
                # Prefer server-reported TTFS (EOS→final inference); also
                # compute client-side wall time for cross-check.
                server_ttfs = msg.get("ttfs_ms")
                client_ttfs = (time.perf_counter() - t_eos) * 1000.0
                ttfs_ms = float(server_ttfs) if server_ttfs is not None else client_ttfs
                break

    return {
        "audio": str(wav_path),
        "duration_s": len(audio) / sr,
        "partials": partials,
        "final_text": final_text,
        "ttfs_ms": ttfs_ms,
        "session_s": time.perf_counter() - t_connect,
    }


async def run_many(url: str, wavs: list[Path], n: int, chunk_ms: int) -> dict:
    results = []
    for i in range(n):
        wav = wavs[i % len(wavs)]
        results.append(await run_one(url, wav, chunk_ms=chunk_ms))

    ttfs = [r["ttfs_ms"] for r in results if r["ttfs_ms"] is not None]
    summary = {
        "n": len(results),
        "ttfs_median_ms": statistics.median(ttfs) if ttfs else None,
        "ttfs_p95_ms": percentile(ttfs, 95) if ttfs else None,
        "ttfs_p99_ms": percentile(ttfs, 99) if ttfs else None,
        "ttfs_mean_ms": statistics.fmean(ttfs) if ttfs else None,
    }
    return {"summary": summary, "runs": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TTFS against streaming server")
    parser.add_argument("--url", default="ws://localhost:8000/v1/stream")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--out", default="results/ttfs_latest.json")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if manifest.exists():
        from bench.normalize import load_manifest

        rows = load_manifest(manifest)
        wavs = [Path(r["audio_path"]) for r in rows]
    else:
        # Fall back to generated tones.
        tone = Path("data/tone16k.wav")
        if not tone.exists():
            raise SystemExit(
                f"No manifest at {manifest} and no {tone}. Run prepare first."
            )
        wavs = [tone]

    payload = asyncio.run(run_many(args.url, wavs, args.n, args.chunk_ms))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    s = payload["summary"]
    print(
        f"TTFS n={s['n']} median={s['ttfs_median_ms']:.1f}ms "
        f"p95={s['ttfs_p95_ms']:.1f}ms → {out}"
    )


if __name__ == "__main__":
    main()
