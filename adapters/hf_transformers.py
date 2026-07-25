from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from adapters.base import FrameworkAdapter, TranscriptionResult

# Map short aliases used in configs to Hugging Face model ids.
HF_MODEL_IDS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo",
}


class HfTransformersAdapter(FrameworkAdapter):
    """Baseline: Hugging Face Transformers Whisper (no continuous batching)."""

    name = "hf_transformers"

    def __init__(
        self,
        model: str = "tiny",
        device: str = "cpu",
        compute_type: str = "default",
        **_kwargs: Any,
    ):
        self.model_name = model
        self.model_id = HF_MODEL_IDS.get(model, model)
        self.device = device
        self.compute_type = compute_type
        self._pipe = None

    def load(self) -> None:
        import torch
        from transformers import pipeline

        torch_dtype = torch.float32
        if self.device.startswith("cuda"):
            if self.compute_type in {"float16", "fp16", "default"}:
                torch_dtype = torch.float16
            elif self.compute_type in {"bfloat16", "bf16"}:
                torch_dtype = torch.bfloat16

        device_arg: int | str = -1
        if self.device.startswith("cuda"):
            device_arg = 0 if self.device in {"cuda", "cuda:0"} else int(self.device.split(":")[-1])

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=torch_dtype,
            device=device_arg,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if self._pipe is None:
            raise RuntimeError("HfTransformersAdapter.load() not called")

        import soundfile as sf

        info = sf.info(str(audio_path))
        duration = float(info.duration)

        t0 = time.perf_counter()
        out = self._pipe(str(audio_path), return_timestamps=False)
        latency = time.perf_counter() - t0

        text = out["text"] if isinstance(out, dict) else str(out)
        return TranscriptionResult(
            text=text.strip(),
            latency_s=latency,
            audio_duration_s=duration,
            extra={"model_id": self.model_id, "framework": self.name},
        )

    def info(self) -> dict[str, Any]:
        return {
            "framework": self.name,
            "supports_concurrent": self.supports_concurrent,
            "model": self.model_name,
            "model_id": self.model_id,
            "device": self.device,
            "compute_type": self.compute_type,
        }
