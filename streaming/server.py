from __future__ import annotations

import argparse
import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic_settings import BaseSettings, SettingsConfigDict

from adapters.registry import get_adapter


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OSS_")

    framework: str = "faster_whisper"
    model: str = "tiny"
    device: str = "cpu"
    compute_type: str = "default"
    sample_rate: int = 16000
    # Emit a partial every N seconds of accumulated audio (chunked streaming).
    chunk_s: float = 1.0


settings = Settings()
_adapter = None


def get_loaded_adapter():
    global _adapter
    if _adapter is None:
        _adapter = get_adapter(
            settings.framework,
            model=settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        _adapter.load()
    return _adapter


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Eager load so /health reflects a ready model.
    if settings.framework != "mock":
        await asyncio.to_thread(get_loaded_adapter)
    else:
        get_loaded_adapter()
    yield


app = FastAPI(title="open-speech-serve", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "framework": settings.framework,
        "model": settings.model,
        "device": settings.device,
    }


def _pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    """
    Simple chunked streaming protocol:

    Client -> binary frames: raw PCM s16le mono @ 16 kHz
    Client -> text: "EOS" to finalize
    Server -> JSON text frames:
      {"type":"partial","text":"...","t_ms":...}
      {"type":"final","text":"...","t_ms":...}
    """
    await ws.accept()
    adapter = get_loaded_adapter()
    pcm_buf = bytearray()
    last_emit_samples = 0
    bytes_per_chunk = int(settings.sample_rate * settings.chunk_s) * 2  # int16

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                pcm_buf.extend(msg["bytes"])
                while len(pcm_buf) - last_emit_samples >= bytes_per_chunk:
                    end = last_emit_samples + bytes_per_chunk
                    chunk = bytes(pcm_buf[:end])
                    last_emit_samples = end
                    wav = _pcm16_to_wav_bytes(chunk, settings.sample_rate)
                    text = await asyncio.to_thread(_transcribe_bytes, adapter, wav)
                    await ws.send_json(
                        {"type": "partial", "text": text,
                            "t_ms": int(time.time() * 1000)}
                    )

            elif "text" in msg and msg["text"] is not None:
                if msg["text"].strip().upper() == "EOS":
                    # Final pass over the full buffer.
                    t_eos = time.perf_counter()
                    wav = _pcm16_to_wav_bytes(
                        bytes(pcm_buf), settings.sample_rate)
                    text = await asyncio.to_thread(_transcribe_bytes, adapter, wav)
                    ttfs_ms = (time.perf_counter() - t_eos) * 1000.0
                    await ws.send_json(
                        {
                            "type": "final",
                            "text": text,
                            "ttfs_ms": ttfs_ms,
                            "t_ms": int(time.time() * 1000),
                        }
                    )
                    break
    except WebSocketDisconnect:
        return


def _transcribe_bytes(adapter, wav_bytes: bytes) -> str:
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = Path(f.name)
    try:
        return adapter.transcribe(path).text
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="open-speech-serve streaming server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "streaming.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
