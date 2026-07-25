from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf

from adapters.base import FrameworkAdapter, TranscriptionResult


MODEL_IDS = {
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo",
}


class OpenAITranscriptionAdapter(FrameworkAdapter):
    """Remote OpenAI-compatible ASR client used by vLLM and SGLang."""

    supports_concurrent = True

    def __init__(
        self,
        model: str,
        framework_name: str,
        base_url: str,
        language: str = "en",
        timeout_s: float = 120.0,
        api_key: str = "not-needed",
        validate_model: bool = True,
        client: httpx.Client | None = None,
        **_kwargs: Any,
    ):
        if not base_url:
            raise ValueError(f"{framework_name} requires base_url")
        self.name = framework_name
        self.model_name = model
        self.model_id = MODEL_IDS.get(model, model)
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.timeout_s = float(timeout_s)
        self.validate_model = bool(validate_model)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._server_models: list[str] = []

    def load(self) -> None:
        health = self._client.get("/health")
        health.raise_for_status()
        models = self._client.get("/v1/models")
        models.raise_for_status()
        payload = models.json()
        self._server_models = [
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        if self.validate_model and self.model_id not in self._server_models:
            raise RuntimeError(
                f"{self.name} does not serve {self.model_id!r}; "
                f"available models: {self._server_models}"
            )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        info = sf.info(str(audio_path))
        started = time.perf_counter()
        with audio_path.open("rb") as audio:
            response = self._client.post(
                "/v1/audio/transcriptions",
                files={"file": (audio_path.name, audio, "audio/wav")},
                data={
                    "model": self.model_id,
                    "language": self.language,
                    "response_format": "json",
                    "temperature": "0",
                },
            )
        latency = time.perf_counter() - started
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "text" not in payload:
            raise ValueError(f"{self.name} returned no transcription text")
        return TranscriptionResult(
            text=str(payload["text"]).strip(),
            latency_s=latency,
            audio_duration_s=float(info.duration),
            extra={"framework": self.name, "model_id": self.model_id},
        )

    def unload(self) -> None:
        if self._owns_client:
            self._client.close()

    def info(self) -> dict[str, Any]:
        return {
            "framework": self.name,
            "supports_concurrent": self.supports_concurrent,
            "model": self.model_name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "server_models": self._server_models,
            "language": self.language,
        }
