"""VoiceChannel: local microphone adapter, wake word -> record -> STT -> handlers.

``send()`` speaks a reply back out loud via TTS. Playback and the wake-word
listener both go through the local speakers/mic (PortAudio via
``sounddevice``), not a network API, so this channel has no inbound
credentials to check, only the optional ``sounddevice``/``webrtcvad``
dependencies (``uv sync --extra speech-mic``).
"""

from __future__ import annotations

import importlib.util
import io
import logging
import threading
import time
import wave
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from openjarvis.channels._stubs import (
    BaseChannel,
    ChannelHandler,
    ChannelMessage,
    ChannelStatus,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import (
    ChannelRegistry,
    SpeechRegistry,
    TTSRegistry,
    WakeWordRegistry,
)


def _import_sounddevice() -> Tuple[Any, str]:
    """Return ``(module, "")``, or ``(None, why)`` with the fix for that cause.

    The channel is optional, so the whole package must still import.
    """
    try:
        import sounddevice
    except ImportError:
        return None, (
            "sounddevice is not installed; install with: uv sync --extra speech-mic"
        )
    except OSError as exc:
        # The wheel is installed but PortAudio's native library (libportaudio2)
        # is not: a different fix than the missing Python package.
        return None, (
            f"the PortAudio system library could not be loaded ({exc}); "
            "install it, e.g. `sudo apt install libportaudio2`"
        )
    return sounddevice, ""


sd, _SD_UNAVAILABLE = _import_sounddevice()

try:
    # Trigger STT/TTS/wake-word backend registration, so callers that never
    # imported openjarvis.speech themselves don't hit an empty registry.
    import openjarvis.speech  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Delay before restarting a crashed mic-capture session (_mic_loop). Fixed, not
# exponential backoff: a mic session either works or it doesn't for
# environmental reasons (device unplugged, config error).
_MIC_LOOP_RETRY_DELAY_S = 2.0
# After TTS playback ends, skip wake-word detection for this long to let room
# echo of the assistant's own voice decay, or its tail self-triggers the wake word.
_POST_PLAYBACK_SUPPRESS_S = 0.5
# Wake-word frames (80 ms each) re-attached to the start of the utterance that
# follows a detection. The detector fires only after the wake phrase and the
# first word or two of the command were already spoken.
_LEAD_IN_FRAMES = 5
_VAD_SAMPLE_RATES = (8000, 16000, 32000, 48000)

# Backend id -> (modules it needs, extra that provides them). Lets connect()
# report every missing package at once instead of one traceback per retry.
_VOICE_BACKEND_DEPS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "openwakeword": (("openwakeword",), "speech-wakeword"),
    "faster-whisper": (("faster_whisper",), "speech"),
    "kokoro": (("kokoro", "soundfile"), "speech-tts"),
}


def _missing_voice_extras(*backend_ids: str) -> List[str]:
    """Extras still to install for the given backends (unknown ids skipped)."""
    extras: List[str] = []
    for backend in backend_ids:
        modules, extra = _VOICE_BACKEND_DEPS.get(backend, ((), ""))
        missing = any(importlib.util.find_spec(m) is None for m in modules)
        if missing and extra not in extras:
            extras.append(extra)
    return extras


def _pcm_to_wav(pcm: bytes, *, sample_rate: int, sample_width: int = 2) -> bytes:
    """Wrap raw mono 16-bit PCM in a WAV container (stdlib, no ffmpeg)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Unwrap a WAV container to raw PCM bytes + its sample rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


class _UtteranceEndpointer:
    """Decides when a recorded utterance has ended, one VAD frame at a time.

    Pure state machine (no I/O), so it is testable without real audio.

    Parameters
    ----------
    frame_ms:
        Duration each :meth:`feed` call represents.
    max_silence_ms:
        Trailing silence, after speech has started, that ends the utterance.
    max_duration_ms:
        Hard cap regardless of VAD, a safety net against runaway recording if
        VAD never detects silence (e.g. loud background noise).
    leading_silence_timeout_ms:
        Give up (treat as no speech) if speech never starts within this
        window after recording begins.
    """

    def __init__(
        self,
        frame_ms: int,
        *,
        max_silence_ms: int = 500,
        max_duration_ms: int = 12000,
        leading_silence_timeout_ms: int = 3000,
    ) -> None:
        self._frame_ms = frame_ms
        self._max_silence_ms = max_silence_ms
        self._max_duration_ms = max_duration_ms
        self._leading_silence_timeout_ms = leading_silence_timeout_ms
        self._elapsed_ms = 0
        self._silence_run_ms = 0
        self._heard_speech = False

    def feed(self, is_speech: bool) -> bool:
        """Feed one frame's VAD decision. Returns True once capture should stop."""
        self._elapsed_ms += self._frame_ms
        if is_speech:
            self._heard_speech = True
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += self._frame_ms

        if self._elapsed_ms >= self._max_duration_ms:
            return True
        if not self._heard_speech:
            return self._elapsed_ms >= self._leading_silence_timeout_ms
        return self._silence_run_ms >= self._max_silence_ms

    def summary(self) -> str:
        """When and why capture ended, for diagnostics."""
        if self._elapsed_ms >= self._max_duration_ms:
            reason = "max duration"
        elif not self._heard_speech:
            reason = "leading silence"
        else:
            reason = "trailing silence"
        return f"after {self._elapsed_ms} ms ({reason})"


def _parse_input_device(value: str) -> int | str | None:
    """Config string -> sounddevice ``device``: None, an index, or a name substring."""
    value = value.strip()
    if not value:
        return None
    return int(value) if value.isdigit() else value


def _close_stream(stream: Any) -> None:
    """Stop and close an input stream, ignoring errors on an already-dead one."""
    try:
        stream.stop()
        stream.close()
    except Exception:
        logger.debug("Failed to close mic stream", exc_info=True)


@ChannelRegistry.register("voice")
class VoiceChannel(BaseChannel):
    """Always-listening local mic channel gated by wake-word detection.

    Parameters
    ----------
    wakeword_backend:
        ``WakeWordRegistry`` key used to detect the wake phrase (default
        ``"openwakeword"``).
    wakeword_keyword:
        The trigger phrase passed to the wake-word backend (default
        ``"hey_jarvis"``).
    wakeword_model_path:
        Path to a custom wake-word model. Empty (default) looks the model up
        by ``wakeword_keyword``.
    wakeword_threshold:
        Detection confidence threshold (default 0.5).
    stt_backend:
        ``SpeechRegistry`` key used to transcribe the utterance that follows
        a wake-word trigger (default ``"faster-whisper"``).
    stt_language:
        Language hint for STT. Empty (default) means auto-detect.
    tts_backend:
        ``TTSRegistry`` key used to speak replies in :meth:`send`. Empty
        string disables spoken output (``send`` returns ``False``).
    tts_voice_id:
        Voice id passed to the TTS backend. Empty uses the backend default.
    record_seconds:
        Maximum recording duration after a trigger. Recording normally ends
        earlier, once ``webrtcvad`` hears the speaker stop; without
        ``webrtcvad`` it always records the full ``record_seconds``.
    vad_max_silence_ms:
        Trailing silence (ms) that ends an utterance (default 500).
    vad_aggressiveness:
        ``webrtcvad`` filtering aggressiveness, 0 (least) to 3 (most).
    input_device:
        ``sounddevice`` input device index or name. Empty is the system default.
    bus:
        Optional event bus for publishing channel events.
    """

    channel_id = "voice"

    def __init__(
        self,
        *,
        wakeword_backend: str = "openwakeword",
        wakeword_keyword: str = "hey_jarvis",
        wakeword_model_path: str = "",
        wakeword_threshold: float = 0.5,
        stt_backend: str = "faster-whisper",
        stt_language: str = "",
        tts_backend: str = "",
        tts_voice_id: str = "",
        record_seconds: float = 4.0,
        vad_max_silence_ms: int = 500,
        vad_aggressiveness: int = 2,
        input_device: str = "",
        bus: Optional[EventBus] = None,
    ) -> None:
        self._wakeword_backend_id = wakeword_backend
        self._wakeword_keyword = wakeword_keyword
        self._wakeword_model_path = wakeword_model_path
        self._wakeword_threshold = wakeword_threshold
        self._stt_backend_id = stt_backend
        self._stt_language = stt_language
        self._tts_backend_id = tts_backend
        self._tts_voice_id = tts_voice_id
        self._record_seconds = record_seconds
        self._vad_max_silence_ms = vad_max_silence_ms
        self._vad_aggressiveness = vad_aggressiveness
        self._input_device = _parse_input_device(input_device)
        self._bus = bus
        self._handlers: List[ChannelHandler] = []
        self._status = ChannelStatus.DISCONNECTED
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_speaking = False
        # time.monotonic() of when playback last ended, see _POST_PLAYBACK_SUPPRESS_S.
        self._playback_ended_at = 0.0
        # The mic loop's open stream. Recording reuses it, because opening a
        # second stream re-acquires the device and drops the first words spoken
        # right after the wake word.
        self._active_stream: Any = None
        # Audio just before the last wake detection; consumed by the next capture.
        self._lead_in = b""
        # STT/TTS back real models that are expensive to load, so they are
        # created once and reused for the channel's lifetime.
        self._stt: Optional[Any] = None
        self._tts: Optional[Any] = None

    def _get_stt(self) -> Any:
        if self._stt is None:
            self._stt = SpeechRegistry.create(self._stt_backend_id)
        return self._stt

    def _get_tts(self) -> Any:
        if self._tts is None:
            self._tts = TTSRegistry.create(self._tts_backend_id)
        return self._tts

    # -- connection lifecycle ---------------------------------------------------

    def connect(self) -> None:
        """Start the mic-capture/wake-word listener thread."""
        if sd is None:
            logger.warning("Voice channel disabled: %s", _SD_UNAVAILABLE)
            self._status = ChannelStatus.ERROR
            return

        missing = _missing_voice_extras(
            self._wakeword_backend_id, self._stt_backend_id, self._tts_backend_id
        )
        if missing:
            flags = " ".join(f"--extra {e}" for e in missing)
            logger.error(
                "Voice channel backends are missing packages and will fail: uv sync %s",
                flags,
            )

        self._stop_event.clear()
        self._status = ChannelStatus.CONNECTING
        self._listener_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._listener_thread.start()
        # The thread may already have failed and set ERROR; don't mask it.
        if self._status == ChannelStatus.CONNECTING:
            self._status = ChannelStatus.CONNECTED

    def disconnect(self) -> None:
        """Stop the listener thread."""
        self._stop_event.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5.0)
            self._listener_thread = None
        self._status = ChannelStatus.DISCONNECTED

    # -- send / receive --------------------------------------------------------

    def send(
        self,
        channel: str,
        content: str,
        *,
        conversation_id: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        """Speak ``content`` out loud via the configured TTS backend."""
        if not self._tts_backend_id:
            logger.warning("No TTS backend configured for voice channel")
            return False
        if sd is None:
            logger.warning("Cannot play audio: %s", _SD_UNAVAILABLE)
            return False

        self._is_speaking = True
        try:
            pcm, sample_rate = _wav_to_pcm(self._synthesize(content))
            if self._bus is not None:
                self._bus.publish(EventType.VOICE_SPEAKING_START, {"channel": "voice"})
            self._play(pcm, sample_rate)
        except Exception as exc:
            logger.exception("Voice channel send failed: %s", exc)
            return False
        finally:
            self._is_speaking = False

        self._publish_sent(content, conversation_id)
        return True

    def _synthesize(self, content: str) -> bytes:
        # An empty voice id must fall through to the backend's own default.
        kwargs = {"voice_id": self._tts_voice_id} if self._tts_voice_id else {}
        result = self._get_tts().synthesize(content, output_format="wav", **kwargs)
        return result.audio

    def status(self) -> ChannelStatus:
        """Return the current connection status."""
        return self._status

    def list_channels(self) -> List[str]:
        """Return available channel identifiers."""
        return ["voice"]

    def on_message(self, handler: ChannelHandler) -> None:
        """Register a callback for incoming (transcribed) messages."""
        self._handlers.append(handler)

    # -- mic loop ---------------------------------------------------------------

    def _mic_loop(self) -> None:
        """Restart the mic session on unexpected errors, so one failure doesn't
        leave the channel deaf for the rest of the process lifetime."""
        while not self._stop_event.is_set():
            try:
                self._run_mic_session()
            except ImportError as exc:
                # A missing package won't appear on its own; retrying only
                # floods the log with the same traceback.
                logger.error("Voice channel stopped, missing dependency: %s", exc)
                self._status = ChannelStatus.ERROR
                return
            except Exception:
                logger.exception(
                    "Voice channel mic loop error, retrying in %.0fs",
                    _MIC_LOOP_RETRY_DELAY_S,
                )
                self._status = ChannelStatus.ERROR
                self._stop_event.wait(_MIC_LOOP_RETRY_DELAY_S)

    def _run_mic_session(self) -> None:
        """One wake-word session: open the mic and listen until stopped or an
        error occurs. Raises on failure; ``_mic_loop`` decides whether to retry."""
        self._status = ChannelStatus.CONNECTED
        wakeword = WakeWordRegistry.create(
            self._wakeword_backend_id,
            keyword=self._wakeword_keyword,
            threshold=self._wakeword_threshold,
            model_path=self._wakeword_model_path,
        )
        stream = sd.RawInputStream(
            samplerate=wakeword.sample_rate,
            blocksize=wakeword.frame_length,
            dtype="int16",
            channels=1,
            device=self._input_device,
        )
        try:
            stream.start()
            self._active_stream = stream
            self._listen(wakeword, stream)
        finally:
            self._active_stream = None
            _close_stream(stream)

    def _listen(self, wakeword: Any, stream: Any) -> None:
        """Feed mic frames to the wake-word detector; capture on each detection."""
        lead_in: deque[bytes] = deque(maxlen=_LEAD_IN_FRAMES)
        while not self._stop_event.is_set():
            frame, _overflowed = stream.read(wakeword.frame_length)
            frame_bytes = bytes(frame)
            lead_in.append(frame_bytes)
            if self._wake_word_suppressed():
                continue
            detection = wakeword.process(frame_bytes)
            if detection is None:
                continue
            logger.info(
                "Wake word %r detected (confidence=%.2f)",
                detection.keyword,
                detection.confidence,
            )
            self._lead_in = b"".join(lead_in)
            if self._bus is not None:
                self._bus.publish(EventType.VOICE_LISTENING_START, {"channel": "voice"})
            try:
                self._capture_and_dispatch(wakeword.sample_rate, stream)
            except ImportError:
                raise
            except Exception:
                # One failed STT call must not tear down the mic session.
                logger.exception("Voice capture/transcription failed")
            # Drop pre-detection frames so they can't leak into the next lead-in
            # if the wake word fires again right away.
            lead_in.clear()

    def _wake_word_suppressed(self) -> bool:
        """True while the assistant is speaking, or just finished: the mic would
        pick up its own voice and self-trigger."""
        if self._is_speaking:
            return True
        elapsed = time.monotonic() - self._playback_ended_at
        return bool(self._playback_ended_at) and elapsed < _POST_PLAYBACK_SUPPRESS_S

    # -- recording --------------------------------------------------------------

    def _record_fixed_window(self, sample_rate: int, stream: Any) -> bytes:
        """Record exactly ``record_seconds`` of audio, no endpointing."""
        frame, _overflowed = stream.read(int(self._record_seconds * sample_rate))
        return bytes(frame)

    def _record_with_vad(self, sample_rate: int, stream: Any) -> bytes:
        """Record until ``webrtcvad`` hears the speaker stop talking.

        Reads from the caller's already-open mic stream (see ``_active_stream``).
        ``RawInputStream.read(frames)`` accepts any frame count regardless of the
        blocksize the stream was opened with, so 30 ms reads are fine here.
        """
        import webrtcvad

        if sample_rate not in _VAD_SAMPLE_RATES:
            raise ValueError(f"webrtcvad does not support sample_rate={sample_rate}")

        frame_ms = 30
        frame_samples = sample_rate * frame_ms // 1000
        vad = webrtcvad.Vad(self._vad_aggressiveness)
        endpointer = _UtteranceEndpointer(
            frame_ms,
            max_silence_ms=self._vad_max_silence_ms,
            max_duration_ms=int(self._record_seconds * 1000),
        )
        chunks: List[bytes] = []
        while True:
            frame, _overflowed = stream.read(frame_samples)
            frame_bytes = bytes(frame)
            chunks.append(frame_bytes)
            if endpointer.feed(vad.is_speech(frame_bytes, sample_rate)):
                break
        logger.info("Utterance capture ended %s", endpointer.summary())
        return b"".join(chunks)

    def _record_utterance(self, sample_rate: int, stream: Any) -> bytes:
        """Record one utterance, preferring VAD endpointing over a fixed window."""
        try:
            return self._record_with_vad(sample_rate, stream)
        except ImportError:
            logger.debug(
                "webrtcvad not installed; using fixed %.1fs recording window",
                self._record_seconds,
            )
        except Exception:
            logger.exception(
                "VAD capture failed; falling back to fixed recording window"
            )
        return self._record_fixed_window(sample_rate, stream)

    def _capture_and_dispatch(self, sample_rate: int, stream: Any) -> bool:
        """Record an utterance, transcribe it, and dispatch it to handlers.

        Returns True if a transcript was produced and dispatched, False if STT
        produced no text.
        """
        lead_in, self._lead_in = self._lead_in, b""
        pcm = lead_in + self._record_utterance(sample_rate, stream)
        result = self._get_stt().transcribe(
            _pcm_to_wav(pcm, sample_rate=sample_rate),
            format="wav",
            language=self._stt_language or None,
        )
        logger.info("Transcribed: %r", result.text)
        if not result.text:
            return False
        self._dispatch(result.text)
        return True

    def _dispatch(self, text: str) -> None:
        """Publish the received event, then hand the message to every handler."""
        cm = ChannelMessage(
            channel="voice",
            sender="mic",
            content=text,
            conversation_id="mic",
        )
        # Publish before dispatching: a handler runs the agent turn and calls
        # send() synchronously, which blocks on TTS playback, so publishing
        # afterward would delay subscribers until the reply had been spoken.
        if self._bus is not None:
            self._bus.publish(
                EventType.CHANNEL_MESSAGE_RECEIVED,
                {
                    "channel": cm.channel,
                    "sender": cm.sender,
                    "content": cm.content,
                    "message_id": cm.message_id,
                },
            )
        for handler in self._handlers:
            try:
                handler(cm)
            except Exception:
                logger.exception("Voice channel handler error")

    # -- playback ---------------------------------------------------------------

    def _play(self, pcm: bytes, sample_rate: int) -> None:
        """Play raw 16-bit PCM through the default output device."""
        import numpy as np

        sd.play(np.frombuffer(pcm, dtype=np.int16), samplerate=sample_rate)
        sd.wait()
        self._drain_mic_after_playback()

    def _drain_mic_after_playback(self) -> None:
        """Drop stale mic audio that piled up while playback blocked the mic thread.

        Without draining, the first reads after playback return a backlog that
        includes room echo of the assistant's own voice, which can self-trigger
        the wake word or produce garbled transcripts.
        """
        self._playback_ended_at = time.monotonic()
        stream = self._active_stream
        # Only the mic thread blocks in send(); from another thread the mic
        # loop keeps reading, and a concurrent read() on the stream is unsafe.
        if stream is None or threading.current_thread() is not self._listener_thread:
            return
        try:
            if stream.read_available:
                stream.read(stream.read_available)
        except Exception:
            logger.debug("Failed to drain mic after playback", exc_info=True)
        self._playback_ended_at = time.monotonic()

    def _publish_sent(self, content: str, conversation_id: str) -> None:
        """Publish a CHANNEL_MESSAGE_SENT event on the bus.

        ``channel`` is always ``"voice"``, never the ``channel`` argument
        ``send()`` receives: per ``BaseChannel.send()`` that argument is the reply
        destination, and for this channel it is the literal ``"mic"``.
        """
        if self._bus is not None:
            self._bus.publish(
                EventType.CHANNEL_MESSAGE_SENT,
                {
                    "channel": self.channel_id,
                    "content": content,
                    "conversation_id": conversation_id,
                },
            )


__all__ = ["VoiceChannel"]
