"""Tests for audio.vad.EnergyVAD: speech/silence detection and duration rules."""

from __future__ import annotations

import unittest

import numpy as np

from app.config import AudioConfig
from audio.vad import EnergyVAD, VADState
from core.exceptions import VADError


def _tone_chunk(n_samples: int, amplitude: int) -> bytes:
    """A constant-amplitude int16 PCM chunk, used as a stand-in for real audio."""
    samples = np.full(n_samples, amplitude, dtype=np.int16)
    return samples.tobytes()


def _silence_chunk(n_samples: int = 320) -> bytes:
    return _tone_chunk(n_samples, 0)


def _voice_chunk(n_samples: int = 320) -> bytes:
    # ~50% of int16 full scale => energy well above the default 0.02 threshold.
    return _tone_chunk(n_samples, 16000)


class EnergyVADTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AudioConfig(
            silence_timeout_s=0.05,
            min_speech_duration_s=0.05,
            max_recording_duration_s=2.0,
            vad_energy_threshold=0.02,
        )
        self.vad = EnergyVAD(self.config)

    def test_starts_in_waiting_state(self) -> None:
        self.assertEqual(self.vad.state, VADState.WAITING)

    def test_silence_stays_waiting(self) -> None:
        result = self.vad.process_chunk(_silence_chunk())
        self.assertEqual(result.state, VADState.WAITING)
        self.assertIsNone(result.segment)

    def test_voice_transitions_to_listening(self) -> None:
        result = self.vad.process_chunk(_voice_chunk())
        self.assertEqual(result.state, VADState.LISTENING)
        self.assertIsNone(result.segment)

    def test_full_speech_then_silence_produces_segment(self) -> None:
        # Enough voiced chunks to exceed min_speech_duration_s at typical chunk sizes.
        for _ in range(10):
            self.vad.process_chunk(_voice_chunk())

        result = None
        # Feed silence until the configured silence_timeout_s has elapsed.
        import time

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            result = self.vad.process_chunk(_silence_chunk())
            if result.segment is not None:
                break

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsNotNone(result.segment)
        self.assertEqual(result.state, VADState.WAITING)
        self.assertGreater(len(result.segment), 0)
        self.assertIsNotNone(result.speech_started_at)
        self.assertIsNotNone(result.speech_ended_at)

    def test_too_short_segment_is_discarded(self) -> None:
        strict_config = AudioConfig(
            silence_timeout_s=0.02,
            min_speech_duration_s=5.0,  # unreachable in this test
            max_recording_duration_s=10.0,
        )
        vad = EnergyVAD(strict_config)
        vad.process_chunk(_voice_chunk())

        import time

        result = None
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            result = vad.process_chunk(_silence_chunk())
            if result.state == VADState.WAITING:
                break

        assert result is not None
        self.assertIsNone(result.segment)  # discarded: too short

    def test_reset_clears_state(self) -> None:
        self.vad.process_chunk(_voice_chunk())
        self.vad.reset()
        self.assertEqual(self.vad.state, VADState.WAITING)

    def test_invalid_chunk_type_raises(self) -> None:
        with self.assertRaises(VADError):
            self.vad.process_chunk("not bytes")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
