"""Custom exception hierarchy for VoiceDriveOS.

Every pipeline stage (perception, decision, validation, execution) raises a
specific exception subclassing :class:`VoiceDriveError` so that ``app.main``
can catch failures at the right granularity, log them, clean up state, and
return to ``IDLE`` without ever executing a partial or unsafe action.
"""

from __future__ import annotations


class VoiceDriveError(Exception):
    """Base class for all VoiceDriveOS errors."""


# --------------------------------------------------------------------------- #
# Perception layer: audio
# --------------------------------------------------------------------------- #
class AudioError(VoiceDriveError):
    """Generic audio subsystem failure."""


class MicrophoneError(AudioError):
    """Raised when the microphone device cannot be opened, read, or closed."""


class VADError(AudioError):
    """Raised when the Voice Activity Detector fails to process a frame."""


class TranscriptionError(AudioError):
    """Raised when speech-to-text transcription fails or returns garbage."""


# --------------------------------------------------------------------------- #
# Perception layer: vision
# --------------------------------------------------------------------------- #
class VisionError(VoiceDriveError):
    """Generic vision subsystem failure."""


class ScreenCaptureError(VisionError):
    """Raised when a screenshot cannot be captured."""


class UIParsingError(VisionError):
    """Raised when the UI parser fails to produce a valid element list."""


# --------------------------------------------------------------------------- #
# Decision layer
# --------------------------------------------------------------------------- #
class ModelLoadError(VoiceDriveError):
    """Raised when a heavy model (Whisper, Laya, vision model) fails to load."""


class DecisionError(VoiceDriveError):
    """Raised when the decision core fails to produce a structured action."""


class IntentParsingError(VoiceDriveError):
    """Raised when the deterministic command parser cannot classify a command."""


# --------------------------------------------------------------------------- #
# Validation / execution layer
# --------------------------------------------------------------------------- #
class ValidationError(VoiceDriveError):
    """Raised when a structured action fails schema/semantic/target validation."""


class ActionSchemaError(ValidationError):
    """Raised when raw decision output does not conform to the action schema."""


class ExecutionError(VoiceDriveError):
    """Raised when the executor fails to perform a validated action."""


# --------------------------------------------------------------------------- #
# Application layer
# --------------------------------------------------------------------------- #
class ConfigurationError(VoiceDriveError):
    """Raised when configuration values are missing, malformed, or unsafe."""


class StateTransitionError(VoiceDriveError):
    """Raised when an illegal state machine transition is attempted."""
