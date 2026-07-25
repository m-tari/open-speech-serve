from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from bench.normalize import write_manifest


def write_tone(path: Path, duration_s: float = 3.0, sr: int = 16000, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Soft A440 + quiet noise so Whisper has *something* to latch onto.
    audio = 0.2 * np.sin(2 * np.pi * freq * t)
    audio += 0.01 * np.random.default_rng(0).standard_normal(audio.shape)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def fetch_librispeech_sample(dest: Path, n: int = 25) -> list[dict]:
    """
    Pull a tiny LibriSpeech test-clean subset via OpenSLR (tar of full set is large).

    For v1 smoke we prefer local tones + optional HF datasets if available.
    Falls back gracefully when offline.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return []

    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    )
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        audio = ex["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        out = dest / f"ls_{i:04d}.wav"
        import soundfile as sf

        sf.write(str(out), arr, sr)
        rows.append(
            {
                "audio_path": str(out),
                "reference": ex.get("text") or ex.get("transcription") or "",
                "source": "librispeech_test_clean",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare eval audio + manifest")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n", type=int, default=25,
                        help="LibriSpeech clips if available")
    parser.add_argument(
        "--tones",
        type=int,
        default=5,
        help="Synthetic tone clips to always generate (offline-safe)",
    )
    parser.add_argument("--with-librispeech", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for i in range(args.tones):
        path = data_dir / "tones" / f"tone_{i:02d}.wav"
        write_tone(path, duration_s=2.0 + i * 0.5)
        rows.append(
            {
                "audio_path": str(path),
                "reference": None,
                "source": "synthetic_tone",
            }
        )

    # Always write a single canonical tone for streaming demos.
    write_tone(data_dir / "tone16k.wav", duration_s=3.0)

    if args.with_librispeech:
        # Optional dependency; install datasets in the image later if needed.
        try:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "datasets"]
            )
            ls_rows = fetch_librispeech_sample(
                data_dir / "librispeech", n=args.n)
            rows.extend(ls_rows)
            print(f"Fetched {len(ls_rows)} LibriSpeech clips")
        except Exception as exc:  # noqa: BLE001
            print(f"LibriSpeech fetch skipped: {exc}")

    manifest = data_dir / "manifest.jsonl"
    write_manifest(manifest, rows)
    print(f"Wrote {len(rows)} entries → {manifest}")


if __name__ == "__main__":
    main()
