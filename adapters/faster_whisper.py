from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from adapters.base import FrameworkAdapter, TranscriptionResult


class FasterWhisperAdapter(FrameworkAdapter):
    """Compiled CTranslate2 path via faster-whisper."""

    name = "faster_whisper"

    def __init__(
        self,
        model: str = "tiny",
        device: str = "cpu",
        compute_type: str = "default",
        beam_size: int = 1,
        cpu_threads: int = 4,
        **_kwargs: Any,
    ):
        self.model_name = model
        self.device = "cuda" if device.startswith("cuda") else device
        if compute_type == "default":
            compute_type = "float16" if self.device == "cuda" else "int8"
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("FasterWhisperAdapter.load() not called")

        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=self.beam_size,
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        latency = time.perf_counter() - t0

        return TranscriptionResult(
            text=text,
            latency_s=latency,
            audio_duration_s=float(info.duration),
            extra={
                "language": info.language,
                "language_probability": float(info.language_probability),
                "framework": self.name,
            },
        )

    def info(self) -> dict[str, Any]:
        return {
            "framework": self.name,
            "supports_concurrent": self.supports_concurrent,
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
        }
