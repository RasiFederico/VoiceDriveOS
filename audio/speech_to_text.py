"""Offline speech-to-text via faster-whisper.

The model is loaded exactly once (double-checked locking) and reused for
every transcription call, per the project brief's explicit prohibition on
repeated model loading. ``faster_whisper`` is imported lazily so this module
can be imported and its buffer-conversion logic unit-tested without the
dependency installed.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.config import AudioConfig
from core.exceptions import ModelLoadError, TranscriptionError
from core.logger import get_logger
from core.performance import LatencyProfiler

logger = get_logger(__name__)

PCM_INT16_MAX = 32768.0


def _load_whisper_model_class() -> Any:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise ModelLoadError(
            "faster-whisper is not installed. Run `pip install faster-whisper` "
            "to enable real offline transcription."
        ) from exc
    return WhisperModel


class SpeechToTextEngine:
    """Thread-safe, lazily-loaded, single-instance faster-whisper wrapper.

    Intended to be owned by :class:`intelligence.model_manager.ModelManager`
    so the whole application shares exactly one loaded model.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._model: Any = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model now (idempotent). Called eagerly at startup by ModelManager."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # re-check inside the lock
                return
            whisper_model_cls = _load_whisper_model_class()
            try:
                logger.info(
                    "loading faster-whisper model=%s device=%s compute_type=%s",
                    self._config.whisper_model,
                    self._config.whisper_device,
                    self._config.whisper_compute_type,
                )
                self._model = whisper_model_cls(
                    self._config.whisper_model,
                    device=self._config.whisper_device,
                    compute_type=self._config.whisper_compute_type,
                    download_root=(
                        str(self._config.whisper_cache_dir)
                        if self._config.whisper_cache_dir is not None
                        else None
                    ),
                    local_files_only=self._config.whisper_local_files_only,
                )
            except Exception as exc:
                preparation_hint = ""
                if self._config.whisper_local_files_only:
                    preparation_hint = (
                        " The model is configured for offline use; download it once with "
                        f"`python scripts/download_whisper_model.py --model {self._config.whisper_model}` "
                        "or set VDOS_AUDIO_WHISPER_MODEL to an existing local model directory."
                    )
                raise ModelLoadError(
                    f"failed to load faster-whisper model '{self._config.whisper_model}': {exc}."
                    f"{preparation_hint}"
                ) from exc

    @staticmethod
    def pcm16_to_float32(audio: bytes | bytearray | np.ndarray) -> np.ndarray:
        """Convert 16-bit PCM bytes (or an existing array) to normalized float32."""
        if isinstance(audio, np.ndarray):
            array = audio
        else:
            array = np.frombuffer(bytes(audio), dtype=np.int16)
        return array.astype(np.float32) / PCM_INT16_MAX

    def transcribe(self, audio: bytes | bytearray | np.ndarray, profiler: LatencyProfiler | None = None) -> str:
        """Transcribe one complete speech segment to text.

        Args:
            audio: A complete utterance as 16-bit PCM bytes (typically the
                ``segment`` produced by :class:`audio.vad.EnergyVAD`).
            profiler: Optional profiler to mark ``transcription_start``/``_end``.

        Raises:
            TranscriptionError: on empty input or an empty/failed result.
        """
        if audio is None or (isinstance(audio, (bytes, bytearray)) and len(audio) == 0):
            raise TranscriptionError("audio buffer is empty; nothing to transcribe")
        if isinstance(audio, np.ndarray) and audio.size == 0:
            raise TranscriptionError("audio buffer is empty; nothing to transcribe")

        self.load()

        if profiler is not None:
            profiler.mark("transcription_start")
        try:
            pcm = self.pcm16_to_float32(audio)
            segments, _info = self._model.transcribe(
                pcm,
                language=self._config.whisper_language,
                beam_size=1,
                vad_filter=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"transcription failed: {exc}") from exc
        finally:
            if profiler is not None:
                profiler.mark("transcription_end")

        if not text:
            raise TranscriptionError("transcription produced empty text")

        logger.info("transcript=%r", text)
        return text

    def shutdown(self) -> None:
        """Release the loaded model (e.g. on application exit)."""
        with self._load_lock:
            self._model = None
