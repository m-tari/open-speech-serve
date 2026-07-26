from __future__ import annotations

import argparse
import io
import os
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from bench.normalize import write_manifest


def write_tone(
    path: Path, duration_s: float = 3.0, sr: int = 16000, freq: float = 440.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * freq * t)
    audio += 0.01 * np.random.default_rng(0).standard_normal(audio.shape)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def fetch_librispeech_sample(dest: Path, n: int = 25) -> list[dict]:
    """Pull LibriSpeech test-clean via Hugging Face datasets (streaming)."""
    from datasets import Audio, load_dataset

    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    )
    ds = ds.decode(False) if hasattr(ds, "decode") else ds.cast_column(
        "audio", Audio(decode=False)
    )

    rows: list[dict] = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        audio = ex["audio"]
        if audio.get("bytes"):
            arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        elif audio.get("path"):
            arr, sr = sf.read(audio["path"], dtype="float32")
        else:
            raise ValueError(
                f"LibriSpeech example {i} has no audio bytes/path")
        out = dest / f"ls_{i:04d}.wav"
        sf.write(str(out), np.asarray(arr, dtype=np.float32), int(sr))
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
                        help="LibriSpeech clips if requested")
    parser.add_argument("--tones", type=int, default=5)
    parser.add_argument("--with-librispeech", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for i in range(args.tones):
        path = data_dir / "tones" / f"tone_{i:02d}.wav"
        write_tone(path, duration_s=2.0 + i * 0.5)
        rows.append(
            {"audio_path": str(path), "reference": None,
             "source": "synthetic_tone"}
        )

    write_tone(data_dir / "tone16k.wav", duration_s=3.0)

    if args.with_librispeech:
        ls_rows = fetch_librispeech_sample(data_dir / "librispeech", n=args.n)
        rows.extend(ls_rows)
        print(f"Fetched {len(ls_rows)} LibriSpeech clips")

    manifest = data_dir / "manifest.jsonl"
    write_manifest(manifest, rows)
    print(f"Wrote {len(rows)} entries → {manifest}")

    # datasets/pyarrow + CUDA Torch crash on normal interpreter shutdown (exit 139).
    if args.with_librispeech:
        os._exit(0)


if __name__ == "__main__":
    main()
