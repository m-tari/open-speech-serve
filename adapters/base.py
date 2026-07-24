from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    text: str
    latency_s: float
    audio_duration_s: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor: wall time / audio duration (lower is faster)."""
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.latency_s / self.audio_duration_s


class FrameworkAdapter(ABC):
    """One adapter per serving stack. Load once; transcribe many times."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights onto the configured device."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Run a single offline transcription and return timing + text."""

    def unload(self) -> None:
        """Optional teardown."""

    def info(self) -> dict[str, Any]:
        return {"framework": self.name}
