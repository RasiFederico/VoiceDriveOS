"""Centralized, typed configuration for VoiceDriveOS.

Every tunable parameter lives here as a typed dataclass field with a sane
default. Values can be overridden via environment variables (prefixed with
``VDOS_``) so that behaviour can be tuned without touching source code, e.g.::

    VDOS_AUDIO_WHISPER_MODEL=base VDOS_VISION_ENABLED=false python -m app.main

Nothing outside this module should hardcode a tunable value.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from core.exceptions import ConfigurationError

T = TypeVar("T")

_ENV_PREFIX = "VDOS_"


def _local_model_dir(name: str) -> Path | None:
    """Find model assets beside a frozen executable or in the source checkout."""
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parents[1]
    candidate = root / "models" / name
    return candidate if candidate.is_dir() else None


def _default_whisper_model() -> str:
    override = os.environ.get(f"{_ENV_PREFIX}AUDIO_WHISPER_MODEL")
    if override:
        return override
    bundled = _local_model_dir("whisper-base")
    return str(bundled) if bundled is not None else "base"


def _default_laya_model_path() -> Path | None:
    override = os.environ.get(f"{_ENV_PREFIX}INTELLIGENCE_LAYA_MODEL_PATH")
    if override:
        return Path(override).expanduser()
    return _local_model_dir("laya-multilingual")


def _env(name: str, default: T, cast: type | None = None) -> T:
    """Read ``VDOS_{name}`` from the environment, casting to ``type(default)``."""
    raw = os.environ.get(f"{_ENV_PREFIX}{name}")
    if raw is None:
        return default

    target_type = cast if cast is not None else type(default)
    try:
        if target_type is bool:
            return raw.strip().lower() in {"1", "true", "yes", "on"}  # type: ignore[return-value]
        if target_type in (int, float, str):
            return target_type(raw)  # type: ignore[return-value]
        return default
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Invalid value for {_ENV_PREFIX}{name}={raw!r}: expected {target_type.__name__}"
        ) from exc


def _optional_env_int(name: str) -> int | None:
    raw = os.environ.get(f"{_ENV_PREFIX}{name}")
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid value for {_ENV_PREFIX}{name}={raw!r}: expected an integer device index"
        ) from exc


@dataclass(frozen=True)
class AudioConfig:
    """Microphone capture, VAD, and speech-to-text parameters."""

    sample_rate: int = field(default_factory=lambda: _env("AUDIO_SAMPLE_RATE", 16000))
    channels: int = field(default_factory=lambda: _env("AUDIO_CHANNELS", 1))
    chunk_size: int = field(default_factory=lambda: _env("AUDIO_CHUNK_SIZE", 1024))
    input_device: int | None = field(default_factory=lambda: _optional_env_int("AUDIO_INPUT_DEVICE"))

    whisper_model: str = field(default_factory=_default_whisper_model)
    whisper_cache_dir: Path | None = field(
        default_factory=lambda: (
            Path(os.environ[f"{_ENV_PREFIX}AUDIO_WHISPER_CACHE_DIR"]).expanduser()
            if os.environ.get(f"{_ENV_PREFIX}AUDIO_WHISPER_CACHE_DIR")
            else None
        )
    )
    whisper_local_files_only: bool = field(
        default_factory=lambda: _env("AUDIO_WHISPER_LOCAL_FILES_ONLY", True)
    )
    whisper_device: str = field(default_factory=lambda: _env("AUDIO_WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(
        default_factory=lambda: _env("AUDIO_WHISPER_COMPUTE_TYPE", "int8")
    )
    whisper_language: str | None = field(
        default_factory=lambda: os.environ.get(f"{_ENV_PREFIX}AUDIO_WHISPER_LANGUAGE") or None
    )

    silence_timeout_s: float = field(default_factory=lambda: _env("AUDIO_SILENCE_TIMEOUT_S", 0.6))
    min_speech_duration_s: float = field(
        default_factory=lambda: _env("AUDIO_MIN_SPEECH_DURATION_S", 0.25)
    )
    max_recording_duration_s: float = field(
        default_factory=lambda: _env("AUDIO_MAX_RECORDING_DURATION_S", 12.0)
    )
    vad_energy_threshold: float = field(
        default_factory=lambda: _env("AUDIO_VAD_ENERGY_THRESHOLD", 0.02)
    )

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigurationError("audio.sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ConfigurationError("audio.channels must be 1 or 2")
        model_names = {"tiny", "base", "small", "medium", "large-v3"}
        if not self.whisper_model.strip():
            raise ConfigurationError("audio.whisper_model must not be empty")
        model_path = Path(self.whisper_model).expanduser()
        if self.whisper_model not in model_names:
            if not model_path.is_dir():
                raise ConfigurationError(
                    f"whisper_model must be one of {sorted(model_names)} or an existing local model directory: "
                    f"{self.whisper_model!r}"
                )
            object.__setattr__(self, "whisper_model", str(model_path.resolve()))
        if self.input_device is not None and self.input_device < 0:
            raise ConfigurationError("audio.input_device must be >= 0")
        if self.silence_timeout_s <= 0:
            raise ConfigurationError("audio.silence_timeout_s must be positive")
        if self.max_recording_duration_s <= self.min_speech_duration_s:
            raise ConfigurationError(
                "audio.max_recording_duration_s must exceed min_speech_duration_s"
            )


@dataclass(frozen=True)
class VisionConfig:
    """Screen capture, caching, and UI parsing parameters."""

    enabled: bool = field(default_factory=lambda: _env("VISION_ENABLED", True))
    capture_scale: float = field(default_factory=lambda: _env("VISION_CAPTURE_SCALE", 0.75))
    refresh_interval_s: float = field(
        default_factory=lambda: _env("VISION_REFRESH_INTERVAL_S", 0.5)
    )
    cache_lifetime_s: float = field(default_factory=lambda: _env("VISION_CACHE_LIFETIME_S", 2.0))
    monitor_index: int = field(default_factory=lambda: _env("VISION_MONITOR_INDEX", 0))
    ocr_language: str = field(default_factory=lambda: _env("VISION_OCR_LANGUAGE", "it"))
    ocr_min_confidence: float = field(
        default_factory=lambda: _env("VISION_OCR_MIN_CONFIDENCE", 0.2)
    )
    max_screen_elements: int = field(
        default_factory=lambda: _env("VISION_MAX_SCREEN_ELEMENTS", 300)
    )
    region_of_interest: tuple[int, int, int, int] | None = None
    """Optional (left, top, width, height) capture region, in real screen pixels."""

    def __post_init__(self) -> None:
        if not (0.0 < self.capture_scale <= 1.0):
            raise ConfigurationError("vision.capture_scale must be in (0, 1]")
        if self.refresh_interval_s < 0:
            raise ConfigurationError("vision.refresh_interval_s must be >= 0")
        if self.cache_lifetime_s < 0:
            raise ConfigurationError("vision.cache_lifetime_s must be >= 0")
        if not self.ocr_language.strip():
            raise ConfigurationError("vision.ocr_language must not be empty")
        if not (0.0 <= self.ocr_min_confidence <= 1.0):
            raise ConfigurationError("vision.ocr_min_confidence must be in [0, 1]")
        if self.max_screen_elements <= 0:
            raise ConfigurationError("vision.max_screen_elements must be positive")


@dataclass(frozen=True)
class IntelligenceConfig:
    """Decision core (Laya / mock) parameters."""

    model_name: str = field(default_factory=lambda: _env("INTELLIGENCE_MODEL", "laya"))
    confidence_threshold: float = field(
        default_factory=lambda: _env("INTELLIGENCE_CONFIDENCE_THRESHOLD", 0.6)
    )
    max_candidates: int = field(default_factory=lambda: _env("INTELLIGENCE_MAX_CANDIDATES", 18))
    laya_model_path: Path | None = field(default_factory=_default_laya_model_path)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError("intelligence.confidence_threshold must be in [0, 1]")
        if self.max_candidates <= 0:
            raise ConfigurationError("intelligence.max_candidates must be positive")


@dataclass(frozen=True)
class ExecutionConfig:
    """PyAutoGUI / clipboard execution parameters."""

    mouse_duration_s: float = field(default_factory=lambda: _env("EXECUTION_MOUSE_DURATION_S", 0.05))
    failsafe: bool = field(default_factory=lambda: _env("EXECUTION_FAILSAFE", True))
    pause_s: float = field(default_factory=lambda: _env("EXECUTION_PAUSE_S", 0.01))
    typing_interval_s: float = field(
        default_factory=lambda: _env("EXECUTION_TYPING_INTERVAL_S", 0.0)
    )
    clipboard_paste_hotkey: tuple[str, ...] = ("ctrl", "v")

    def __post_init__(self) -> None:
        if self.mouse_duration_s < 0:
            raise ConfigurationError("execution.mouse_duration_s must be >= 0")
        if self.pause_s < 0:
            raise ConfigurationError("execution.pause_s must be >= 0")


@dataclass(frozen=True)
class PerformanceConfig:
    """Logging / profiling / resource-target parameters."""

    max_memory_target_mb: int = field(
        default_factory=lambda: _env("PERFORMANCE_MAX_MEMORY_TARGET_MB", 1536)
    )
    logging_level: str = field(default_factory=lambda: _env("PERFORMANCE_LOGGING_LEVEL", "INFO"))
    profiling_enabled: bool = field(
        default_factory=lambda: _env("PERFORMANCE_PROFILING_ENABLED", True)
    )
    log_file: str | None = field(
        default_factory=lambda: os.environ.get(f"{_ENV_PREFIX}PERFORMANCE_LOG_FILE")
    )

    def __post_init__(self) -> None:
        if self.max_memory_target_mb <= 0:
            raise ConfigurationError("performance.max_memory_target_mb must be positive")
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.logging_level.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid logging_level: {self.logging_level}")


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration aggregating every subsystem's settings."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def load(cls) -> "AppConfig":
        """Build configuration from defaults + environment variable overrides."""
        return cls(
            audio=AudioConfig(),
            vision=VisionConfig(),
            intelligence=IntelligenceConfig(),
            execution=ExecutionConfig(),
            performance=PerformanceConfig(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a nested plain-dict representation, useful for debug logging."""
        return {
            "audio": vars(self.audio),
            "vision": vars(self.vision),
            "intelligence": vars(self.intelligence),
            "execution": vars(self.execution),
            "performance": vars(self.performance),
        }
