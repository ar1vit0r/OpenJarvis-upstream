"""Abstract base classes and data types for wake-word detection backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class WakeWordDetection:
    """A single wake-word trigger."""

    keyword: str
    confidence: float
    frame_index: int


class WakeWordBackend(ABC):
    """Abstract base class for streaming wake-word detection backends.

    Unlike ``SpeechBackend.transcribe()``, this is a frame-by-frame streaming
    interface: callers feed fixed-size raw PCM frames from a live mic buffer
    and get a detection back only on the frame where the keyword fires.
    """

    backend_id: str = ""

    @property
    @abstractmethod
    def frame_length(self) -> int:
        """Expected number of samples per frame passed to ``process()``."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Required input sample rate in Hz."""

    @abstractmethod
    def process(self, frame: bytes) -> Optional[WakeWordDetection]:
        """Feed one raw 16-bit PCM frame; return a detection if triggered."""

    @abstractmethod
    def health(self) -> bool:
        """Check if the backend/model is ready."""


__all__ = ["WakeWordBackend", "WakeWordDetection"]
