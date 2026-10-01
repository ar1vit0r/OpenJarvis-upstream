"""Tests for the VoiceChannel adapter."""

from __future__ import annotations

import builtins
import importlib.util
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.voice_channel import (
    VoiceChannel,
    _import_sounddevice,
    _missing_voice_extras,
    _pcm_to_wav,
    _UtteranceEndpointer,
    _wav_to_pcm,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import (
    ChannelRegistry,
    SpeechRegistry,
    TTSRegistry,
    WakeWordRegistry,
)
from openjarvis.speech._stubs import TranscriptionResult
from openjarvis.speech.tts import TTSResult
from openjarvis.speech.wakeword import WakeWordDetection
from tests.channels.channel_test_helpers import make_common_channel_tests


@pytest.fixture(autouse=True)
def _register_voice():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("voice"):
        ChannelRegistry.register_value("voice", VoiceChannel)


TestCommonChannel = make_common_channel_tests(VoiceChannel, "voice")


class TestInit:
    def test_defaults(self):
        ch = VoiceChannel()
        assert ch._wakeword_backend_id == "openwakeword"
        assert ch._stt_backend_id == "faster-whisper"
        assert ch._tts_backend_id == ""
        assert ch._record_seconds == 4.0
        assert ch.status() == ChannelStatus.DISCONNECTED


class TestWavRoundtrip:
    def test_pcm_to_wav_and_back(self):
        pcm = b"\x01\x00\x02\x00" * 100
        wav_bytes = _pcm_to_wav(pcm, sample_rate=16000)
        roundtrip_pcm, sample_rate = _wav_to_pcm(wav_bytes)
        assert roundtrip_pcm == pcm
        assert sample_rate == 16000


class TestConnect:
    def test_connect_without_sounddevice_sets_error(self):
        with patch("openjarvis.channels.voice_channel.sd", new=None):
            ch = VoiceChannel()
            ch.connect()
            assert ch.status() == ChannelStatus.ERROR

    def test_connect_keeps_an_error_the_listener_already_set(self):
        # The listener can fail before connect() returns; connect() must not
        # then overwrite ERROR with CONNECTED (it hid missing-package errors).
        ch = VoiceChannel()
        with (
            patch("openjarvis.channels.voice_channel.sd", MagicMock()),
            patch.object(ch, "_run_mic_session", side_effect=ImportError("missing")),
        ):
            ch.connect()
            assert ch._listener_thread is not None
            ch._listener_thread.join(timeout=2.0)

        assert ch.status() == ChannelStatus.ERROR

    def test_connect_starts_listener_thread(self):
        mock_sd = MagicMock()
        ch = VoiceChannel()
        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(ch, "_mic_loop"),
        ):
            ch.connect()
            assert ch.status() == ChannelStatus.CONNECTED
            assert ch._listener_thread is not None
            ch.disconnect()
            assert ch.status() == ChannelStatus.DISCONNECTED


class TestImportSounddevice:
    @staticmethod
    def _import_raising(exc: Exception):
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "sounddevice":
                raise exc
            return real_import(name, *args, **kwargs)

        return fake

    def test_missing_package_points_at_the_extra(self):
        with patch("builtins.__import__", self._import_raising(ImportError("x"))):
            module, why = _import_sounddevice()

        assert module is None
        assert "uv sync --extra speech-mic" in why

    def test_missing_portaudio_points_at_the_system_library(self):
        with patch(
            "builtins.__import__", self._import_raising(OSError("PortAudio not found"))
        ):
            module, why = _import_sounddevice()

        assert module is None
        assert "libportaudio2" in why
        assert "PortAudio not found" in why
        assert "speech-mic" not in why

    def test_success_returns_the_module_and_no_message(self):
        fake = MagicMock()
        with patch.dict(sys.modules, {"sounddevice": fake}):
            module, why = _import_sounddevice()

        assert module is fake
        assert why == ""


class TestMissingExtras:
    def test_reports_each_missing_extra_once(self):
        with patch("importlib.util.find_spec", return_value=None):
            assert _missing_voice_extras(
                "openwakeword", "faster-whisper", "kokoro", "kokoro"
            ) == ["speech-wakeword", "speech", "speech-tts"]

    def test_installed_and_unknown_backends_are_ignored(self):
        with patch("importlib.util.find_spec", return_value=object()):
            assert _missing_voice_extras("openwakeword", "kokoro") == []
        assert _missing_voice_extras("some-other-backend", "") == []

    def test_connect_logs_one_error_naming_the_extras(self, caplog):
        with (
            patch("openjarvis.channels.voice_channel.sd", MagicMock()),
            patch("importlib.util.find_spec", return_value=None),
            caplog.at_level("ERROR"),
        ):
            ch = VoiceChannel(tts_backend="kokoro")
            ch.connect()
            ch.disconnect()
        assert "uv sync --extra speech-wakeword" in caplog.text
        assert "--extra speech-tts" in caplog.text


class TestSend:
    def test_send_no_tts_backend_returns_false(self):
        ch = VoiceChannel()
        assert ch.send("voice", "hello") is False

    def test_send_no_sounddevice_returns_false(self):
        with patch("openjarvis.channels.voice_channel.sd", new=None):
            ch = VoiceChannel(tts_backend="kokoro")
            assert ch.send("voice", "hello") is False

    def test_send_synthesizes_and_plays(self):
        wav_bytes = _pcm_to_wav(b"\x00\x01" * 10, sample_rate=24000)
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = TTSResult(audio=wav_bytes, format="wav")
        mock_sd = MagicMock()

        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(TTSRegistry, "create", return_value=mock_tts),
        ):
            ch = VoiceChannel(tts_backend="kokoro")
            result = ch.send("voice", "hello there")

        assert result is True
        mock_tts.synthesize.assert_called_once_with("hello there", output_format="wav")
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    def test_send_publishes_event(self):
        wav_bytes = _pcm_to_wav(b"\x00\x01" * 10, sample_rate=24000)
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = TTSResult(audio=wav_bytes, format="wav")
        mock_sd = MagicMock()
        bus = EventBus(record_history=True)

        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(TTSRegistry, "create", return_value=mock_tts),
        ):
            ch = VoiceChannel(tts_backend="kokoro", bus=bus)
            # ChannelAgent's real call passes msg.conversation_id (always
            # "mic" for this channel) as the destination arg here, not the
            # literal string "voice" — regression test for the channel
            # field being mislabeled from that arg instead of hardcoded.
            ch.send("mic", "hi")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types
        sent_event = next(
            e for e in bus.history if e.event_type == EventType.CHANNEL_MESSAGE_SENT
        )
        assert sent_event.data["channel"] == "voice"

    def test_send_publishes_speaking_start_before_playback(self):
        wav_bytes = _pcm_to_wav(b"\x00\x01" * 10, sample_rate=24000)
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = TTSResult(audio=wav_bytes, format="wav")
        mock_sd = MagicMock()
        bus = EventBus(record_history=True)

        history_len_at_play_call = []

        def _play(_pcm, _sample_rate):
            history_len_at_play_call.append(len(bus.history))

        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(TTSRegistry, "create", return_value=mock_tts),
        ):
            ch = VoiceChannel(tts_backend="kokoro", bus=bus)
            with patch.object(ch, "_play", side_effect=_play):
                ch.send("voice", "hello there")

        event_types = [e.event_type for e in bus.history]
        assert EventType.VOICE_SPEAKING_START in event_types
        # Regression guard: published before playback starts, not after --
        # the whole point is to signal "about to speak", matching how
        # VOICE_LISTENING_START signals "about to record" rather than
        # "finished recording".
        assert history_len_at_play_call == [1]

    def test_send_synthesize_failure_returns_false(self):
        mock_tts = MagicMock()
        mock_tts.synthesize.side_effect = RuntimeError("boom")
        mock_sd = MagicMock()

        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(TTSRegistry, "create", return_value=mock_tts),
        ):
            ch = VoiceChannel(tts_backend="kokoro")
            assert ch.send("voice", "hi") is False

    def test_tts_backend_created_once_and_reused(self):
        wav_bytes = _pcm_to_wav(b"\x00\x01" * 10, sample_rate=24000)
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = TTSResult(audio=wav_bytes, format="wav")
        mock_sd = MagicMock()

        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(TTSRegistry, "create", return_value=mock_tts) as mock_create,
        ):
            ch = VoiceChannel(tts_backend="kokoro")
            ch.send("voice", "hi")
            ch.send("voice", "hi again")

        # Regression: TTSRegistry.create() used to be called on every
        # single reply, reloading the (real, expensive, torch-backed) TTS
        # model from disk each time instead of reusing one instance.
        mock_create.assert_called_once()

    def test_tts_voice_id_passed_to_synthesize(self):
        wav_bytes = _pcm_to_wav(b"\x00\x01" * 10, sample_rate=24000)
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = TTSResult(audio=wav_bytes, format="wav")
        mock_sd = MagicMock()

        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(TTSRegistry, "create", return_value=mock_tts),
        ):
            ch = VoiceChannel(tts_backend="kokoro", tts_voice_id="af_sarah")
            ch.send("voice", "hello")

        mock_tts.synthesize.assert_called_once_with(
            "hello", output_format="wav", voice_id="af_sarah"
        )

    def test_tts_voice_id_omitted_when_empty(self):
        wav_bytes = _pcm_to_wav(b"\x00\x01" * 10, sample_rate=24000)
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = TTSResult(audio=wav_bytes, format="wav")
        mock_sd = MagicMock()

        with (
            patch("openjarvis.channels.voice_channel.sd", mock_sd),
            patch.object(TTSRegistry, "create", return_value=mock_tts),
        ):
            ch = VoiceChannel(tts_backend="kokoro")
            ch.send("voice", "hello")

        # Empty id must not be forwarded: the backend's own default applies.
        mock_tts.synthesize.assert_called_once_with("hello", output_format="wav")


class TestCaptureAndDispatch:
    @pytest.mark.skipif(
        importlib.util.find_spec("webrtcvad") is not None,
        reason="webrtcvad is installed in this environment",
    )
    def test_transcribed_speech_dispatches_to_handlers(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(
            text="turn on the lights"
        )
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)

        received = []
        with patch.object(SpeechRegistry, "create", return_value=mock_stt):
            ch = VoiceChannel()
            ch.on_message(lambda cm: received.append(cm))
            ch._capture_and_dispatch(16000, mock_stream)

        mock_stream.read.assert_called_once_with(64000)
        assert len(received) == 1
        assert received[0].content == "turn on the lights"
        assert received[0].channel == "voice"

    def test_stt_language_passed_through(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(text="oi jarvis")
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)

        with patch.object(SpeechRegistry, "create", return_value=mock_stt):
            ch = VoiceChannel(stt_language="pt")
            ch._capture_and_dispatch(16000, mock_stream)

        mock_stt.transcribe.assert_called_once()
        assert mock_stt.transcribe.call_args.kwargs["language"] == "pt"

    def test_stt_backend_created_once_and_reused(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(text="hi")
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)

        with patch.object(
            SpeechRegistry, "create", return_value=mock_stt
        ) as mock_create:
            ch = VoiceChannel()
            ch._capture_and_dispatch(16000, mock_stream)
            ch._capture_and_dispatch(16000, mock_stream)

        # Regression: SpeechRegistry.create() used to be called on every
        # single capture, reloading the (real, expensive) STT model from
        # disk each time instead of reusing one instance.
        mock_create.assert_called_once()

    def test_empty_transcription_does_not_dispatch(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(text="")
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)

        received = []
        with patch.object(SpeechRegistry, "create", return_value=mock_stt):
            ch = VoiceChannel()
            ch.on_message(lambda cm: received.append(cm))
            ch._capture_and_dispatch(16000, mock_stream)

        assert received == []

    def test_handler_exception_does_not_propagate(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(text="hi")
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)

        def _boom(cm):
            raise RuntimeError("handler exploded")

        with patch.object(SpeechRegistry, "create", return_value=mock_stt):
            ch = VoiceChannel()
            ch.on_message(_boom)
            ch._capture_and_dispatch(16000, mock_stream)  # must not raise

    def test_publishes_received_event_before_dispatching_to_handlers(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(
            text="turn on the lights"
        )
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)
        bus = EventBus(record_history=True)

        history_len_at_handler_call = []

        def _handler(cm):
            history_len_at_handler_call.append(len(bus.history))

        with patch.object(SpeechRegistry, "create", return_value=mock_stt):
            ch = VoiceChannel(bus=bus)
            ch.on_message(_handler)
            ch._capture_and_dispatch(16000, mock_stream)

        # Regression: CHANNEL_MESSAGE_RECEIVED must be published before handlers
        # run, not after -- a handler (ChannelAgent._handle_message) runs the
        # agent turn synchronously and blocks on TTS playback, so publishing
        # afterward delayed the frontend's window-show trigger on this event
        # until Jarvis had already finished speaking.
        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_RECEIVED in event_types
        assert history_len_at_handler_call == [1]

    def test_returns_true_when_dispatched(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(
            text="turn on the lights"
        )
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)

        with patch.object(SpeechRegistry, "create", return_value=mock_stt):
            ch = VoiceChannel()
            result = ch._capture_and_dispatch(16000, mock_stream)

        assert result is True

    def test_returns_false_on_empty_transcription(self):
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = TranscriptionResult(text="")
        mock_stream = MagicMock()
        mock_stream.read.return_value = (b"\x00\x00" * 64000, False)

        with patch.object(SpeechRegistry, "create", return_value=mock_stt):
            ch = VoiceChannel()
            result = ch._capture_and_dispatch(16000, mock_stream)

        assert result is False


def _quick_check_then_stop(ch: VoiceChannel, frame_bytes: bytes = b"\x01\x02" * 1280):
    """``stream.read`` side_effect for tests below.

    _run_mic_session opens the stream, then does a quick pre-loop
    audio-level check (up to 5 reads) before entering its real read loop --
    a silent frame there falls back to real PipeWireCapture (a real
    subprocess) instead of raising. ``frame_bytes`` must therefore be
    non-silent so the check passes on the first read; the 2nd-and-later
    read (the real loop's first iteration) sets ``_stop_event`` so that
    loop runs exactly once before stopping.
    """
    calls = {"n": 0}

    def _read(_frames):
        calls["n"] += 1
        if calls["n"] >= 2:
            ch._stop_event.set()
        return (frame_bytes, False)

    return _read


class TestPermanentMicLoopErrors:
    def test_missing_dependency_stops_the_loop_after_one_error(self, caplog):
        ch = VoiceChannel()
        with (
            patch.object(
                ch, "_run_mic_session", side_effect=ImportError("no openwakeword")
            ) as mock_session,
            caplog.at_level("ERROR"),
        ):
            ch._mic_loop()  # returns instead of retrying forever

        mock_session.assert_called_once()
        assert "no openwakeword" in caplog.text
        assert ch.status() == ChannelStatus.ERROR

    def test_other_errors_still_retry(self):
        ch = VoiceChannel()
        calls = []

        def flaky() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("device unplugged")
            ch._stop_event.set()

        with (
            patch.object(ch, "_run_mic_session", side_effect=flaky),
            patch("openjarvis.channels.voice_channel._MIC_LOOP_RETRY_DELAY_S", 0.0),
        ):
            ch._mic_loop()

        assert len(calls) == 2


class TestPostPlaybackSuppress:
    def test_suppresses_wake_word_after_playback(self):
        """Wake-word detection is skipped for a short window after TTS
        playback ends, letting room echo of Jarvis's voice decay."""
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_stream.read_available = 0

        ch = VoiceChannel()
        ch._active_stream = mock_stream

        with patch("openjarvis.channels.voice_channel.sd", mock_sd):
            ch._play(b"\x00\x00" * 100, 16000)

        assert ch._playback_ended_at > 0

    def test_suppression_window_expires(self):
        """After the suppression window elapses, detection resumes normally."""
        ch = VoiceChannel()
        ch._playback_ended_at = time.monotonic() - 1.0  # 1s ago, past the 0.5s window

        # Should not be suppressed anymore
        assert time.monotonic() - ch._playback_ended_at >= 0.5


class TestDrainMicAfterPlayback:
    def test_drain_clears_stale_audio_after_playback(self):
        """After TTS playback, stale mic audio must be drained to prevent
        self-triggering from room echo of Jarvis's own voice."""
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_stream.read_available = 42

        ch = VoiceChannel()
        ch._active_stream = mock_stream
        ch._listener_thread = threading.current_thread()

        with patch("openjarvis.channels.voice_channel.sd", mock_sd):
            ch._play(b"\x00\x00" * 100, 16000)

        mock_stream.read.assert_called_once_with(42)

    def test_drain_noop_when_no_active_mic(self):
        """Drain is a no-op when no mic stream is active."""
        mock_sd = MagicMock()

        ch = VoiceChannel()
        ch._active_stream = None

        with patch("openjarvis.channels.voice_channel.sd", mock_sd):
            ch._play(b"\x00\x00" * 100, 16000)  # must not raise

    def test_drain_swallows_errors(self):
        """Drain failures must never break playback."""
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_stream.read_available = 42
        mock_stream.read.side_effect = RuntimeError("device unplugged")

        ch = VoiceChannel()
        ch._active_stream = mock_stream

        with patch("openjarvis.channels.voice_channel.sd", mock_sd):
            ch._play(b"\x00\x00" * 100, 16000)  # must not raise


class TestUtteranceEndpointer:
    def test_stops_after_trailing_silence_once_speech_heard(self):
        ep = _UtteranceEndpointer(
            30,
            max_silence_ms=90,
            max_duration_ms=100_000,
            leading_silence_timeout_ms=100_000,
        )
        assert ep.feed(True) is False  # speech starts
        assert ep.feed(False) is False  # 30ms silence
        assert ep.feed(False) is False  # 60ms silence
        assert ep.feed(False) is True  # 90ms silence -> stop

    def test_gives_up_if_speech_never_starts(self):
        ep = _UtteranceEndpointer(
            30, leading_silence_timeout_ms=90, max_duration_ms=100_000
        )
        assert ep.feed(False) is False  # 30ms, no speech yet
        assert ep.feed(False) is False  # 60ms
        assert ep.feed(False) is True  # 90ms -> give up, speech never started

    def test_max_duration_caps_even_during_continuous_speech(self):
        ep = _UtteranceEndpointer(
            30,
            max_duration_ms=60,
            max_silence_ms=100_000,
            leading_silence_timeout_ms=100_000,
        )
        assert ep.feed(True) is False  # 30ms
        assert ep.feed(True) is True  # 60ms -> hard cap, even though still speaking

    def test_silence_run_resets_on_renewed_speech(self):
        ep = _UtteranceEndpointer(
            30,
            max_silence_ms=60,
            max_duration_ms=100_000,
            leading_silence_timeout_ms=100_000,
        )
        assert ep.feed(True) is False  # speech
        assert ep.feed(False) is False  # 30ms silence
        assert ep.feed(True) is False  # speech again, silence run resets
        assert ep.feed(False) is False  # 30ms silence (not 60 — reset)
        assert ep.feed(False) is True  # 60ms silence -> stop


def _stop_after_first_read(ch: VoiceChannel, frame_bytes: bytes = b"\x01\x02" * 1280):
    """``stream.read`` side_effect: return one frame, then stop the listener."""

    def _read(_frames):
        ch._stop_event.set()
        return (frame_bytes, False)

    return _read


def _listen_once(ch: VoiceChannel, detection):
    """Run one listener iteration against a mocked wake-word backend and stream."""
    wakeword = MagicMock()
    wakeword.sample_rate = 16000
    wakeword.frame_length = 1280
    wakeword.process.return_value = detection
    stream = MagicMock()
    stream.read.side_effect = _stop_after_first_read(ch)
    sd = MagicMock()
    sd.RawInputStream.return_value = stream
    with (
        patch("openjarvis.channels.voice_channel.sd", sd),
        patch.object(WakeWordRegistry, "create", return_value=wakeword),
        patch.object(ch, "_capture_and_dispatch") as capture,
    ):
        ch._mic_loop()
    return wakeword, stream, capture


class TestWakeWordListener:
    def test_wake_word_triggers_capture(self):
        detection = WakeWordDetection(
            keyword="hey_jarvis", confidence=0.9, frame_index=1
        )
        ch = VoiceChannel()
        _, stream, capture = _listen_once(ch, detection)
        capture.assert_called_once_with(16000, stream)

    def test_detection_publishes_listening_start(self):
        bus = EventBus()
        seen = []
        bus.subscribe(EventType.VOICE_LISTENING_START, seen.append)
        detection = WakeWordDetection(
            keyword="hey_jarvis", confidence=0.9, frame_index=1
        )
        _listen_once(VoiceChannel(bus=bus), detection)
        assert len(seen) == 1

    def test_no_detection_does_not_trigger_capture(self):
        _, _, capture = _listen_once(VoiceChannel(), None)
        capture.assert_not_called()

    def test_frames_ignored_while_speaking(self):
        ch = VoiceChannel()
        ch._is_speaking = True
        wakeword, _, capture = _listen_once(ch, None)
        wakeword.process.assert_not_called()
        capture.assert_not_called()

    def test_stream_is_closed_when_the_session_ends(self):
        _, stream, _ = _listen_once(VoiceChannel(), None)
        stream.stop.assert_called_once()
        stream.close.assert_called_once()


class TestChannelKwargs:
    def test_empty_strings_fall_through_to_channel_defaults(self):
        from openjarvis.core.config import ChannelConfig
        from openjarvis.system._channel_kwargs import build_channel_kwargs

        kw = build_channel_kwargs(ChannelConfig(), "voice")
        assert kw["wakeword_threshold"] == 0.5
        assert "tts_backend" not in kw
        assert "input_device" not in kw

    def test_configured_strings_are_forwarded(self):
        from openjarvis.core.config import ChannelConfig
        from openjarvis.system._channel_kwargs import build_channel_kwargs

        cfg = ChannelConfig()
        cfg.voice.tts_backend = "kokoro"
        cfg.voice.stt_language = "en"
        kw = build_channel_kwargs(cfg, "voice")
        assert kw["tts_backend"] == "kokoro"
        assert kw["stt_language"] == "en"
        # Every forwarded key must be a real VoiceChannel parameter.
        VoiceChannel(**kw)


class TestInputDevice:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("", None), ("  ", None), ("0", 0), ("2", 2), ("pipewire", "pipewire")],
    )
    def test_config_string_becomes_a_sounddevice_device(self, raw, expected):
        assert VoiceChannel(input_device=raw)._input_device == expected
