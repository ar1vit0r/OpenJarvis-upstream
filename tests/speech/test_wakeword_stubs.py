"""Tests for wake-word ABC and data types."""

import pytest

from openjarvis.speech.wakeword import WakeWordBackend, WakeWordDetection


def test_wake_word_detection():
    detection = WakeWordDetection(keyword="hey_jarvis", confidence=0.87, frame_index=42)
    assert detection.keyword == "hey_jarvis"
    assert detection.confidence == 0.87
    assert detection.frame_index == 42


def test_wake_word_backend_is_abstract():
    with pytest.raises(TypeError):
        WakeWordBackend()
