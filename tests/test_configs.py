from __future__ import annotations

from pathlib import Path

import pytest

from adapters.openai_transcription import OpenAITranscriptionAdapter
from adapters.registry import get_adapter
from bench.harness import load_cell_config


ROOT = Path(__file__).resolve().parents[1]


def test_all_v2_cell_configs_validate():
    paths = sorted((ROOT / "configs" / "cells").glob("*_turbo_c*_*.yaml"))
    assert len(paths) == 18
    for path in paths:
        cfg = load_cell_config(path)
        assert cfg["dispatch_mode"] in {"serialized", "concurrent"}


def test_invalid_dispatch_mode_is_rejected(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "framework: mock\nmodel: mock\nconcurrency: 1\n"
        "manifest: data/manifest.jsonl\ndispatch_mode: parallel\n"
    )
    with pytest.raises(ValueError, match="dispatch_mode"):
        load_cell_config(config)


def test_registry_remote_aliases():
    vllm = get_adapter(
        "vllm",
        model="large-v3-turbo",
        base_url="http://localhost:8000",
    )
    sglang = get_adapter(
        "sgl",
        model="large-v3-turbo",
        base_url="http://localhost:30000",
    )
    assert isinstance(vllm, OpenAITranscriptionAdapter)
    assert vllm.name == "vllm"
    assert isinstance(sglang, OpenAITranscriptionAdapter)
    assert sglang.name == "sglang"
