from __future__ import annotations

import httpx
import numpy as np
import pytest
import soundfile as sf

from adapters.openai_transcription import OpenAITranscriptionAdapter


def _wav(tmp_path):
    path = tmp_path / "sample.wav"
    sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000)
    return path


def test_openai_adapter_health_model_validation_and_transcription(tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "openai/whisper-large-v3-turbo"}]},
            )
        if request.url.path == "/v1/audio/transcriptions":
            assert request.method == "POST"
            assert request.headers["content-type"].startswith("multipart/form-data")
            return httpx.Response(200, json={"text": " hello world "})
        return httpx.Response(404)

    client = httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    adapter = OpenAITranscriptionAdapter(
        model="large-v3-turbo",
        framework_name="vllm",
        base_url="http://test/",
        client=client,
    )
    adapter.load()
    result = adapter.transcribe(_wav(tmp_path))

    assert adapter.supports_concurrent
    assert result.text == "hello world"
    assert result.audio_duration_s == pytest.approx(1.0, abs=0.01)
    assert [r.url.path for r in requests] == [
        "/v1/models",
        "/v1/audio/transcriptions",
    ]


def test_openai_adapter_rejects_wrong_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "other-model"}]})

    adapter = OpenAITranscriptionAdapter(
        model="large-v3-turbo",
        framework_name="sglang",
        base_url="http://test",
        client=httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(RuntimeError, match="does not serve"):
        adapter.load()


def test_openai_adapter_rejects_missing_text(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    adapter = OpenAITranscriptionAdapter(
        model="large-v3-turbo",
        framework_name="vllm",
        base_url="http://test",
        validate_model=False,
        client=httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(ValueError, match="no transcription text"):
        adapter.transcribe(_wav(tmp_path))


def test_openai_adapter_propagates_models_failure():
    adapter = OpenAITranscriptionAdapter(
        model="large-v3-turbo",
        framework_name="vllm",
        base_url="http://test",
        client=httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(503, text="loading")
            ),
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        adapter.load()


def test_openai_adapter_propagates_timeout(tmp_path):
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow backend", request=request)

    adapter = OpenAITranscriptionAdapter(
        model="large-v3-turbo",
        framework_name="vllm",
        base_url="http://test",
        client=httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(timeout),
        ),
    )
    with pytest.raises(httpx.ReadTimeout):
        adapter.transcribe(_wav(tmp_path))
