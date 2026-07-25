from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from adapters.base import FrameworkAdapter, TranscriptionResult


class TritonWhisperAdapter(FrameworkAdapter):
    """Client for NVIDIA's TensorRT-LLM Whisper BLS model repository."""

    name = "tensorrt_llm"
    supports_concurrent = True

    def __init__(
        self,
        model: str,
        grpc_url: str = "localhost:8001",
        triton_model_name: str = "whisper_bls",
        language: str = "en",
        padding_duration_s: int = 10,
        client: Any = None,
        protocol_client: Any = None,
        **_kwargs: Any,
    ):
        self.model_name = model
        self.grpc_url = grpc_url
        self.triton_model_name = triton_model_name
        self.language = language
        self.padding_duration_s = int(padding_duration_s)
        self._client = client
        self._grpc = protocol_client

    def load(self) -> None:
        if self._client is None:
            try:
                import tritonclient.grpc as grpcclient
            except ImportError as exc:
                raise RuntimeError(
                    "TensorRT-LLM cells require `pip install -e '.[triton]'`"
                ) from exc
            self._grpc = grpcclient
            self._client = grpcclient.InferenceServerClient(
                url=self.grpc_url,
                verbose=False,
            )
        elif self._grpc is None:
            raise ValueError("An injected Triton client requires protocol_client")
        if not self._client.is_server_ready():
            raise RuntimeError(f"Triton is not ready at {self.grpc_url}")
        if not self._client.is_model_ready(self.triton_model_name):
            raise RuntimeError(
                f"Triton model {self.triton_model_name!r} is not ready"
            )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if self._client is None or self._grpc is None:
            raise RuntimeError("TritonWhisperAdapter.load() not called")
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32")
        if sample_rate != 16_000:
            raise ValueError(f"Triton Whisper requires 16 kHz audio, got {sample_rate}")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        duration = len(waveform) / sample_rate
        padded_seconds = (
            int(duration) // self.padding_duration_s + 1
        ) * self.padding_duration_s
        samples = np.zeros((1, padded_seconds * sample_rate), dtype=np.float32)
        samples[0, : len(waveform)] = waveform
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        prompt = np.array(
            [
                [
                    f"<|startoftranscript|><|{self.language}|>"
                    "<|transcribe|><|notimestamps|>"
                ]
            ],
            dtype=object,
        )

        inputs = [
            self._infer_input("WAV", samples),
            self._infer_input("WAV_LENS", lengths),
            self._infer_input("TEXT_PREFIX", prompt, datatype="BYTES"),
        ]
        outputs = [self._grpc.InferRequestedOutput("TRANSCRIPTS")]
        started = time.perf_counter()
        response = self._client.infer(
            self.triton_model_name,
            inputs,
            outputs=outputs,
        )
        latency = time.perf_counter() - started
        value = response.as_numpy("TRANSCRIPTS")[0]
        if isinstance(value, np.ndarray):
            text = b" ".join(value).decode("utf-8")
        elif isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)
        return TranscriptionResult(
            text=text.strip(),
            latency_s=latency,
            audio_duration_s=float(duration),
            extra={
                "framework": self.name,
                "model": self.model_name,
                "triton_model": self.triton_model_name,
            },
        )

    def _infer_input(
        self,
        name: str,
        value: np.ndarray,
        datatype: str | None = None,
    ):
        if datatype is None:
            datatype = {
                np.dtype("float32"): "FP32",
                np.dtype("float16"): "FP16",
                np.dtype("int32"): "INT32",
            }.get(value.dtype)
            if datatype is None:
                raise ValueError(f"Unsupported Triton dtype: {value.dtype}")
        tensor = self._grpc.InferInput(name, value.shape, datatype)
        tensor.set_data_from_numpy(value)
        return tensor

    def info(self) -> dict[str, Any]:
        return {
            "framework": self.name,
            "supports_concurrent": self.supports_concurrent,
            "model": self.model_name,
            "grpc_url": self.grpc_url,
            "triton_model_name": self.triton_model_name,
            "language": self.language,
        }
