"""Tests for openWakeWord wake-word backend."""

from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.registry import WakeWordRegistry
from openjarvis.speech.openwakeword_backend import OpenWakeWordBackend
from openjarvis.speech.wakeword import WakeWordDetection


@pytest.fixture(autouse=True)
def _register_openwakeword():
    """Re-register after any registry clear."""
    if not WakeWordRegistry.contains("openwakeword"):
        WakeWordRegistry.register_value("openwakeword", OpenWakeWordBackend)


def test_openwakeword_backend_registers():
    """Backend registers itself in WakeWordRegistry."""
    assert WakeWordRegistry.contains("openwakeword")


def test_openwakeword_frame_and_sample_rate():
    backend = OpenWakeWordBackend()
    assert backend.frame_length == 1280
    assert backend.sample_rate == 16000


def test_openwakeword_process_detects_above_threshold():
    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_jarvis": 0.9}

    with patch(
        "openjarvis.speech.openwakeword_backend.Model",
        return_value=mock_model,
    ):
        backend = OpenWakeWordBackend(keyword="hey_jarvis", threshold=0.5)
        frame = b"\x00\x00" * 1280
        detection = backend.process(frame)

        assert isinstance(detection, WakeWordDetection)
        assert detection.keyword == "hey_jarvis"
        assert detection.confidence == 0.9
        assert detection.frame_index == 1


def test_openwakeword_process_below_threshold_returns_none():
    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_jarvis": 0.1}

    with patch(
        "openjarvis.speech.openwakeword_backend.Model",
        return_value=mock_model,
    ):
        backend = OpenWakeWordBackend(keyword="hey_jarvis", threshold=0.5)
        frame = b"\x00\x00" * 1280
        assert backend.process(frame) is None


def test_openwakeword_missing_dependency_hint():
    with patch("openjarvis.speech.openwakeword_backend.Model", new=None):
        backend = OpenWakeWordBackend()

        with pytest.raises(ImportError) as excinfo:
            backend._ensure_model()

    assert "uv sync --extra speech-wakeword" in str(excinfo.value)


def test_openwakeword_health_no_model():
    with patch("openjarvis.speech.openwakeword_backend.Model", new=None):
        backend = OpenWakeWordBackend()
        assert backend.health() is False
        assert "uv sync --extra speech-wakeword" in (backend.last_error() or "")


def test_openwakeword_falls_back_to_model_paths_on_old_api():
    """openwakeword<0.5 rejects wakeword_models; retry with paths by name."""
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        if "wakeword_models" in kwargs:
            raise TypeError("unexpected keyword argument 'wakeword_models'")
        return mock_model

    fake_models = {"hey_jarvis": {"model_path": "/fake/hey_jarvis_v0.1.onnx"}}
    with (
        patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor),
        patch.dict("sys.modules", {"openwakeword": MagicMock(models=fake_models)}),
    ):
        backend = OpenWakeWordBackend(keyword="hey_jarvis")
        model = backend._ensure_model()

    assert model is mock_model
    assert calls[0] == {"wakeword_models": ["hey_jarvis"]}
    assert calls[1] == {"wakeword_model_paths": ["/fake/hey_jarvis_v0.1.onnx"]}
    # openwakeword keys predict() output by the model file's stem, not the
    # bare keyword — process() must look up scores under that key or every
    # detection silently scores 0.0 regardless of the real confidence.
    assert backend._score_key == "hey_jarvis_v0.1"


def test_openwakeword_custom_model_path_scores_under_file_stem():
    """A custom model_path loads on the first try (openwakeword>=0.5) but its
    predict() output is still keyed by file stem, not the keyword."""
    with patch(
        "openjarvis.speech.openwakeword_backend.Model", return_value=MagicMock()
    ):
        backend = OpenWakeWordBackend(
            keyword="hey_jarvis", model_path="/fake/jarvis_retrain2.onnx"
        )
        backend._ensure_model()

    assert backend._score_key == "jarvis_retrain2"


def test_openwakeword_falls_back_when_verifier_dict_partially_matches():
    """openwakeword<0.5 swallows wakeword_models via **kwargs, loads every
    pretrained model, and raises ValueError when the custom verifier dict
    matches only some of them -- the fallback must key the verifier by the
    model file stem and still load."""
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        if "wakeword_models" in kwargs:
            raise ValueError(
                "Custom verifier models were provided, but some were not "
                "matched with a base model!"
            )
        return mock_model

    fake_models = {"hey_jarvis": {"model_path": "/fake/hey_jarvis_v0.1.onnx"}}
    with (
        patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor),
        patch.dict("sys.modules", {"openwakeword": MagicMock(models=fake_models)}),
    ):
        backend = OpenWakeWordBackend(
            keyword="hey_jarvis",
            verifier_model_path="/fake/hey_jarvis_verifier.pkl",
            verifier_threshold=0.0,
        )
        model = backend._ensure_model()

    assert model is mock_model
    assert calls[0]["wakeword_models"] == ["hey_jarvis"]
    assert calls[1] == {
        "wakeword_model_paths": ["/fake/hey_jarvis_v0.1.onnx"],
        "custom_verifier_models": {"hey_jarvis_v0.1": "/fake/hey_jarvis_verifier.pkl"},
        "custom_verifier_threshold": 0.0,
    }
    assert backend._score_key == "hey_jarvis_v0.1"


def test_openwakeword_process_uses_file_stem_score_key_after_fallback():
    """Regression: fallback-loaded models score under their file stem key."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_jarvis_v0.1": 0.9}

    def _model_ctor(*args, **kwargs):
        if "wakeword_models" in kwargs:
            raise TypeError("unexpected keyword argument 'wakeword_models'")
        return mock_model

    fake_models = {"hey_jarvis": {"model_path": "/fake/hey_jarvis_v0.1.onnx"}}
    with (
        patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor),
        patch.dict("sys.modules", {"openwakeword": MagicMock(models=fake_models)}),
    ):
        backend = OpenWakeWordBackend(keyword="hey_jarvis", threshold=0.5)
        frame = b"\x00\x00" * 1280
        detection = backend.process(frame)

    assert detection is not None
    assert detection.confidence == 0.9


def test_openwakeword_health_captures_load_error():
    with patch(
        "openjarvis.speech.openwakeword_backend.Model",
        side_effect=RuntimeError("missing onnxruntime provider"),
    ):
        backend = OpenWakeWordBackend()
        assert backend.health() is False
        assert "missing onnxruntime provider" in (backend.last_error() or "")


def test_openwakeword_uses_custom_model_path():
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        return mock_model

    with patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor):
        backend = OpenWakeWordBackend(
            keyword="jarvis", model_path="/models/jarvis.onnx"
        )
        backend._ensure_model()

    assert calls[0] == {"wakeword_models": ["/models/jarvis.onnx"]}


def test_openwakeword_passes_custom_verifier_kwargs():
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        return mock_model

    with patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor):
        backend = OpenWakeWordBackend(
            keyword="jarvis",
            model_path="/models/jarvis.onnx",
            verifier_model_path="/models/jarvis_verifier.onnx",
            verifier_threshold=0.7,
        )
        backend._ensure_model()

    assert calls[0] == {
        "wakeword_models": ["/models/jarvis.onnx"],
        "custom_verifier_models": {"jarvis": "/models/jarvis_verifier.onnx"},
        "custom_verifier_threshold": 0.7,
    }


def test_openwakeword_no_verifier_kwargs_when_unset():
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        return mock_model

    with patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor):
        backend = OpenWakeWordBackend()
        backend._ensure_model()

    assert calls[0] == {"wakeword_models": ["hey_jarvis"]}


def test_openwakeword_custom_model_path_falls_back_to_paths_on_old_api():
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        if "wakeword_models" in kwargs:
            raise TypeError("unexpected keyword argument 'wakeword_models'")
        return mock_model

    with patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor):
        backend = OpenWakeWordBackend(
            keyword="jarvis", model_path="/models/jarvis.onnx"
        )
        model = backend._ensure_model()

    assert model is mock_model
    assert calls[1] == {"wakeword_model_paths": ["/models/jarvis.onnx"]}


def test_openwakeword_passes_vad_threshold_to_model():
    """When vad_threshold > 0, it must be passed to the Model constructor."""
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        return mock_model

    with patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor):
        backend = OpenWakeWordBackend(keyword="hey_jarvis", vad_threshold=0.5)
        backend._ensure_model()

    assert calls[0]["vad_threshold"] == 0.5


def test_openwakeword_passes_speex_noise_suppression_to_model():
    """When speex_noise_suppression is True, it must be passed to Model."""
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        return mock_model

    with patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor):
        backend = OpenWakeWordBackend(
            keyword="hey_jarvis", speex_noise_suppression=True
        )
        backend._ensure_model()

    assert calls[0]["speex_noise_suppression"] is True


def test_openwakeword_omits_vad_and_speex_when_disabled():
    """When both are at defaults, neither kwarg is passed to Model."""
    mock_model = MagicMock()
    calls = []

    def _model_ctor(*args, **kwargs):
        calls.append(kwargs)
        return mock_model

    with patch("openjarvis.speech.openwakeword_backend.Model", side_effect=_model_ctor):
        backend = OpenWakeWordBackend(keyword="hey_jarvis")
        backend._ensure_model()

    assert "vad_threshold" not in calls[0]
    assert "speex_noise_suppression" not in calls[0]


def test_openwakeword_merges_double_peak_within_window():
    """Suppress the second peak when two detections arrive within the merge
    window (the 'jar-vis' double-peak syndrome)."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_jarvis": 0.9}

    with patch(
        "openjarvis.speech.openwakeword_backend.Model",
        return_value=mock_model,
    ):
        backend = OpenWakeWordBackend(keyword="hey_jarvis", threshold=0.5)
        frame = b"\x00\x00" * 1280

        # First detection — should succeed
        d1 = backend.process(frame)
        assert d1 is not None
        assert d1.frame_index == 1

        # Second detection 2 frames later (160ms) — within merge window
        d2 = backend.process(frame)
        assert d2 is None

        # Third detection 2 more frames later — still within window from d1
        d3 = backend.process(frame)
        assert d3 is None


def test_openwakeword_allows_detection_after_window():
    """Allow detections that arrive after the merge window expires."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_jarvis": 0.9}

    with patch(
        "openjarvis.speech.openwakeword_backend.Model",
        return_value=mock_model,
    ):
        backend = OpenWakeWordBackend(keyword="hey_jarvis", threshold=0.5)
        frame = b"\x00\x00" * 1280

        # First detection at frame 1
        d1 = backend.process(frame)
        assert d1 is not None
        assert d1.frame_index == 1

        # Feed 4 more frames (total gap = 4 frames = window boundary)
        # Frame 2,3,4 are suppressed; frame 5 is at boundary
        for _ in range(3):
            r = backend.process(frame)
            assert r is None  # suppressed within window

        # Frame 5: gap = 5-1 = 4 >= merge window → allowed
        d2 = backend.process(frame)
        assert d2 is not None
        assert d2.frame_index == 5

        # Frame 6: gap = 6-5 = 1 < window → suppressed again
        d3 = backend.process(frame)
        assert d3 is None
