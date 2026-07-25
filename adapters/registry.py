from __future__ import annotations

from adapters.base import FrameworkAdapter
from adapters.faster_whisper import FasterWhisperAdapter
from adapters.hf_transformers import HfTransformersAdapter
from adapters.mock import MockAdapter
from adapters.openai_transcription import OpenAITranscriptionAdapter


def get_adapter(
    framework: str,
    model: str,
    device: str = "cpu",
    compute_type: str = "default",
    **kwargs,
) -> FrameworkAdapter:
    key = framework.strip().lower().replace("-", "_")
    if key in {"faster_whisper", "fw", "faster-whisper"}:
        return FasterWhisperAdapter(
            model=model, device=device, compute_type=compute_type, **kwargs
        )
    if key in {"hf", "hf_transformers", "transformers", "huggingface"}:
        return HfTransformersAdapter(
            model=model, device=device, compute_type=compute_type, **kwargs
        )
    if key == "mock":
        return MockAdapter(model=model, device=device)
    if key in {"vllm", "v_llm"}:
        return OpenAITranscriptionAdapter(
            model=model,
            framework_name="vllm",
            **kwargs,
        )
    if key in {"sglang", "sgl"}:
        return OpenAITranscriptionAdapter(
            model=model,
            framework_name="sglang",
            **kwargs,
        )
    if key in {"tensorrt_llm", "tensorrt", "triton", "trtllm"}:
        from adapters.triton_whisper import TritonWhisperAdapter

        return TritonWhisperAdapter(model=model, **kwargs)
    raise ValueError(
        f"Unknown framework {framework!r}. "
        "Choose: faster_whisper | hf_transformers | vllm | sglang | "
        "tensorrt_llm | mock"
    )
