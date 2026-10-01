"""openWakeWord wake-word detection backend (local, ONNX-based)."""

from __future__ import annotations

import os
from typing import Optional

from openjarvis.core.registry import WakeWordRegistry
from openjarvis.speech.wakeword import WakeWordBackend, WakeWordDetection

try:
    from openwakeword.model import Model
except ImportError:
    Model = None  # type: ignore[assignment, misc]


@WakeWordRegistry.register("openwakeword")
class OpenWakeWordBackend(WakeWordBackend):
    """Local wake-word detection using openWakeWord (ONNX)."""

    backend_id = "openwakeword"

    # Collapse duplicate detections closer than this many frames apart.
    # At 80ms/frame, 4 frames = 320ms — covers the "jar-vis" double-peak
    # gap observed in analysis (80-240ms) with margin.
    _MERGE_WINDOW_FRAMES = 4

    def __init__(
        self,
        keyword: str = "hey_jarvis",
        threshold: float = 0.5,
        model_path: str = "",
        verifier_model_path: str = "",
        verifier_threshold: float = 0.5,
        min_rms_energy: float = 0.0,
        vad_threshold: float = 0.0,
        speex_noise_suppression: bool = False,
    ) -> None:
        self._keyword = keyword
        self._threshold = threshold
        self._model_path = model_path
        self._verifier_model_path = verifier_model_path
        self._verifier_threshold = verifier_threshold
        self._min_rms_energy = min_rms_energy
        self._vad_threshold = vad_threshold
        self._speex_noise_suppression = speex_noise_suppression
        self._model: Optional[Model] = None
        self._frame_index = 0
        self._last_error: Optional[str] = None
        # Score dict key in model.predict()'s output. Equals self._keyword
        # normally, but the wakeword_model_paths fallback below loads the
        # model by file path, and openwakeword keys predictions by the
        # model's file stem (e.g. "hey_jarvis_v0.1"), not the bare keyword.
        self._score_key = keyword
        # Frame index of the last emitted detection, for double-peak merge.
        self._last_detection_frame: int = -9999

    @property
    def frame_length(self) -> int:
        return 1280  # 80ms @ 16kHz, openWakeWord's fixed chunk size

    @property
    def sample_rate(self) -> int:
        return 16000

    def _ensure_model(self) -> Model:
        """Lazy-load the openWakeWord model on first use."""
        if self._model is None:
            if Model is None:
                self._last_error = (
                    "openwakeword is not installed. "
                    "Install with: uv sync --extra speech-wakeword"
                )
                raise ImportError(self._last_error)
            model_id = self._model_path or self._keyword
            verifier_kwargs: dict = {}
            if self._verifier_model_path:
                verifier_kwargs["custom_verifier_models"] = {
                    self._keyword: self._verifier_model_path
                }
                verifier_kwargs["custom_verifier_threshold"] = self._verifier_threshold
            model_kwargs: dict = dict(verifier_kwargs)
            if self._vad_threshold > 0:
                model_kwargs["vad_threshold"] = self._vad_threshold
            if self._speex_noise_suppression:
                model_kwargs["speex_noise_suppression"] = True
            try:
                self._model = Model(wakeword_models=[model_id], **model_kwargs)
            except (TypeError, ValueError):
                # openwakeword<0.5 (only tflite-free option on Python 3.12 —
                # >=0.5 hard-depends on tflite-runtime, which has no cp312
                # wheel on Linux) takes wakeword_model_paths, not
                # wakeword_models, and wants a file path rather than a bare
                # model name. Worse, its **kwargs silently swallows
                # wakeword_models (no TypeError), so Model() loads every
                # pretrained model and a partial custom_verifier_models dict
                # makes it raise ValueError on the first unmatched model.
                # Either way: load only the requested model by explicit path.
                if self._model_path:
                    model_path = self._model_path
                else:
                    import openwakeword as _oww

                    model_path = _oww.models[self._keyword]["model_path"]
                stem = os.path.splitext(os.path.basename(model_path))[0]
                if self._verifier_model_path:
                    model_kwargs["custom_verifier_models"] = {
                        stem: self._verifier_model_path
                    }
                self._model = Model(wakeword_model_paths=[model_path], **model_kwargs)
                self._score_key = stem
        self._last_error = None
        return self._model

    def process(self, frame: bytes) -> Optional[WakeWordDetection]:
        """Feed one raw 16-bit PCM frame; return a detection if triggered."""
        try:
            import numpy as np

            model = self._ensure_model()
            audio = np.frombuffer(frame, dtype=np.int16)

            # Energy gate: skip detection on silence/near-silence to block
            # the model's tendency to hallucinate triggers on room tone.
            if self._min_rms_energy > 0:
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                if rms < self._min_rms_energy:
                    self._frame_index += 1
                    return None

            scores = model.predict(audio)
        except Exception as exc:
            self._last_error = str(exc)
            raise

        self._frame_index += 1
        self._last_error = None

        # score is the model's (or, with a verifier configured, the speaker
        # verifier's) raw per-frame confidence that this audio is the wake
        # phrase -- not a calibrated probability. threshold is the "react at
        # all" gate.
        score = float(scores.get(self._score_key, 0.0))
        if score >= self._threshold:
            # Collapse double peaks: suppress detections that arrive too
            # soon after the previous one (the "jar-vis" syndrome where
            # the model fires on "jar", dips, then fires again on "vis").
            gap = self._frame_index - self._last_detection_frame
            if gap < self._MERGE_WINDOW_FRAMES:
                return None
            self._last_detection_frame = self._frame_index
            return WakeWordDetection(
                keyword=self._keyword,
                confidence=score,
                frame_index=self._frame_index,
            )
        return None

    def health(self) -> bool:
        """Check if model is loaded or loadable."""
        try:
            self._ensure_model()
            return True
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def last_error(self) -> Optional[str]:
        """Return the last model load or detection error, if any."""
        return self._last_error
