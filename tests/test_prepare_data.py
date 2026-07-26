from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_data import write_tone


def test_write_tone(tmp_path: Path):
    path = tmp_path / "tone.wav"
    write_tone(path, duration_s=1.0)
    arr, sr = sf.read(str(path))
    assert sr == 16000
    assert len(arr) == 16000
    assert np.max(np.abs(arr)) > 0
