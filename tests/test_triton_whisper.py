from __future__ import annotations

import types

import numpy as np
import soundfile as sf

from adapters.triton_whisper import TritonWhisperAdapter


class FakeInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.value = None

    def set_data_from_numpy(self, value):
        self.value = value


class FakeResponse:
    def as_numpy(self, name):
        assert name == "TRANSCRIPTS"
        return np.array([b"test transcript"], dtype=object)


class FakeClient:
    def __init__(self):
        self.inputs = None

    def is_server_ready(self):
        return True

    def is_model_ready(self, name):
        return name == "whisper_bls"

    def infer(self, model_name, inputs, outputs):
        assert model_name == "whisper_bls"
        assert outputs[0] == "TRANSCRIPTS"
        self.inputs = inputs
        return FakeResponse()


def test_triton_adapter_builds_official_whisper_bls_request(tmp_path):
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    client = FakeClient()
    protocol = types.SimpleNamespace(
        InferInput=FakeInput,
        InferRequestedOutput=lambda name: name,
    )
    adapter = TritonWhisperAdapter(
        model="large-v3-turbo",
        client=client,
        protocol_client=protocol,
    )
    adapter.load()
    result = adapter.transcribe(wav)

    assert result.text == "test transcript"
    assert result.audio_duration_s == 1.0
    assert [item.name for item in client.inputs] == [
        "WAV",
        "WAV_LENS",
        "TEXT_PREFIX",
    ]
    assert client.inputs[0].value.shape == (1, 160_000)
    assert client.inputs[1].value.tolist() == [[16_000]]
    assert "<|en|>" in client.inputs[2].value[0, 0]
