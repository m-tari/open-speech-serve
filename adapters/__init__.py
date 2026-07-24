"""Framework adapters implementing a common Whisper inference interface."""

from adapters.base import FrameworkAdapter, TranscriptionResult
from adapters.registry import get_adapter

__all__ = ["FrameworkAdapter", "TranscriptionResult", "get_adapter"]
