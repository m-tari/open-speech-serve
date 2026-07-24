from __future__ import annotations

import json
import re
from pathlib import Path


# Minimal English ASR normalizer aligned with common Whisper eval practice:
# lower-case, strip punctuation, collapse whitespace. Good enough for relative
# WER across frameworks on the same clips; not a full OpenAI WhisperNormalizer.
_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def load_manifest(path: Path) -> list[dict]:
    """Load JSONL manifest: each line has audio_path + optional reference."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
