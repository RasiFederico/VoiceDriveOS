"""Voice Activity Detection: continuous audio -> discrete speech segments.

The state machine is deliberately explicit and mirrors the project brief:

.. code-block:: text

    WAITING -> VOICE_DETECTED -> LISTENING -> SILENCE_DETECTED -> SEGMENT_READY -> WAITING

:class:`VoiceActivityDetector` is a ``Protocol`` so the reference
:class:`EnergyVAD` implementation (a simple RMS-energy threshold, using only
``numpy`` — no extra heavy dependency) can later be swapped for Silero VAD or
WebRTC VAD without touching any caller. Benchmarking the alternatives is an
explicit open question in the project brief (section 37) rather than a
decision made here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import numpy as np

from app.config import AudioConfig
from core.exceptions import VADError
from core.logger import get_logger

logger = get_logger(__name__)


class VADState(Enum):
    """Coarse state of the voice activity detector."""

    WAITING = auto()
    LISTENING = auto()


@dataclass(frozen=True)
class VADResult:
    """Outcome of feeding one audio chunk into the detector.

    ``segment`` is populated with the complete, concatenated PCM buffer only
    on the transition that closes a valid segment (state goes back to
    ``WAITING`` with enough accumulated speech). It is ``None`` at every
    other step, including when a segment is discarded for being too short.
    """

    state: VADState
    segment: bytes | None = None
    speech_started_at: float | None = None
    """Monotonic timestamp of speech start, populated only when ``segment`` is set."""
    speech_ended_at: float | None = None
    """Monotonic timestamp of speech end, populated only when ``segment`` is set."""


class VoiceActivityDetector(Protocol):
    """Interface every VAD implementation must satisfy."""

    def process_chunk(self, chunk: bytes) -> VADResult: ...

    def reset(self) -> None: ...


def _rms_energy(chunk: bytes) -> float:
    """Root-mean-square energy of a 16-bit PCM buffer, normalized to [0, 1]."""
    if not chunk:
        return 0.0
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return rms / 32768.0


class EnergyVAD:
    """Reference VAD implementation using simple RMS energy thresholding.

    This is intentionally lightweight (no model, no extra dependency) so the
    full audio pipeline can be exercised end-to-end before a more accurate
    VAD (Silero/WebRTC) is benchmarked and swapped in.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._state = VADState.WAITING
        self._buffer = bytearray()
        self._speech_started_at: float | None = None
        self._silence_started_at: float | None = None

    @property
    def state(self) -> VADState:
        return self._state

    def reset(self) -> None:
        self._state = VADState.WAITING
        self._buffer = bytearray()
        self._speech_started_at = None
        self._silence_started_at = None

    def process_chunk(self, chunk: bytes) -> VADResult:
        if not isinstance(chunk, (bytes, bytearray)):
            raise VADError(f"chunk must be bytes-like, got {type(chunk).__name__}")

        try:
            energy = _rms_energy(chunk)
        except Exception as exc:
            raise VADError(f"failed to compute audio energy: {exc}") from exc

        is_voice = energy >= self._config.vad_energy_threshold
        now = time.monotonic()

        if self._state is VADState.WAITING:
            if is_voice:
                logger.debug("VAD WAITING -> LISTENING (energy=%.4f)", energy)
                self._state = VADState.LISTENING
                self._buffer = bytearray(chunk)
                self._speech_started_at = now
                self._silence_started_at = None
            return VADResult(state=self._state)

        # state is LISTENING
        self._buffer.extend(chunk)

        total_duration = now - (self._speech_started_at or now)
        if total_duration >= self._config.max_recording_duration_s:
            logger.warning("VAD max recording duration reached (%.2fs); force-closing segment", total_duration)
            return self._close_segment(now)

        if is_voice:
            self._silence_started_at = None
            return VADResult(state=self._state)

        if self._silence_started_at is None:
            self._silence_started_at = now
            return VADResult(state=self._state)

        silence_duration = now - self._silence_started_at
        if silence_duration >= self._config.silence_timeout_s:
            return self._close_segment(now)

        return VADResult(state=self._state)

    def _close_segment(self, now: float) -> VADResult:
        speech_started_at = self._speech_started_at or now
        speech_duration = now - speech_started_at
        segment_bytes = bytes(self._buffer)
        self.reset()

        if speech_duration < self._config.min_speech_duration_s:
            logger.debug("VAD segment discarded: too short (%.3fs)", speech_duration)
            return VADResult(state=VADState.WAITING, segment=None)

        logger.info("VAD segment ready: %.3fs, %d bytes", speech_duration, len(segment_bytes))
        return VADResult(
            state=VADState.WAITING,
            segment=segment_bytes,
            speech_started_at=speech_started_at,
            speech_ended_at=now,
        )
