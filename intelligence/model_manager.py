"""Central, thread-safe owner of every heavy model.

Per the project brief (sections 10 and 16), Whisper, the decision core, and
the vision backend must each be loaded exactly once, at application startup,
and reused for the lifetime of the process. ``ModelManager`` is the only
Singleton in the codebase — every other component prefers plain dependency
injection (a ``ModelManager`` instance, or the individual model objects it
owns, are passed to constructors rather than looked up globally).
"""

from __future__ import annotations

import threading

from app.config import AppConfig
from audio.speech_to_text import SpeechToTextEngine
from core.exceptions import ConfigurationError, ModelLoadError
from core.logger import get_logger
from intelligence.laya import DecisionModel, LayaDecisionModel, MockDecisionModel
from vision.ui_parser import HybridUIParser, RapidOCRUIParser, StaticUIParser, UIParser
from vision.uia_parser import UIAutomationParser

logger = get_logger(__name__)


class ModelManager:
    """Loads and owns the STT engine, decision model, and UI parser.

    Usage::

        manager = ModelManager(config)
        manager.load_all()   # once, at startup
        ...
        manager.stt.transcribe(audio)
        manager.decision_model.decide(transcript, elements)
        manager.ui_parser.parse(screenshot)
        ...
        manager.shutdown()   # once, at exit
    """

    _instance: "ModelManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._stt: SpeechToTextEngine | None = None
        self._decision_model: DecisionModel | None = None
        self._ui_parser: UIParser | None = None
        self._loaded = False

    # -- Singleton accessor (opt-in; ModelManager can also be constructed
    #    directly and passed around explicitly, which is preferred in tests) --
    @classmethod
    def instance(cls, config: AppConfig | None = None) -> "ModelManager":
        with cls._instance_lock:
            if cls._instance is None:
                if config is None:
                    raise ModelLoadError(
                        "ModelManager.instance() called before initialization: "
                        "pass an AppConfig on the first call."
                    )
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance_for_tests(cls) -> None:
        """Clear the process-wide singleton. Only ever call this from tests."""
        with cls._instance_lock:
            cls._instance = None

    # -- Lifecycle --
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_all(self) -> None:
        """Load every model exactly once. Safe to call more than once (no-op after the first)."""
        with self._lock:
            if self._loaded:
                logger.info("ModelManager.load_all(): models already loaded, skipping")
                return

            logger.info("ModelManager: loading models...")

            self._stt = SpeechToTextEngine(self._config.audio)
            self._stt.load()

            self._decision_model = self._build_decision_model()
            load_decision = getattr(self._decision_model, "load", None)
            if load_decision is not None:
                load_decision()

            self._ui_parser = self._build_ui_parser()
            load_parser = getattr(self._ui_parser, "load", None)
            if load_parser is not None:
                load_parser()

            self._loaded = True
            logger.info("ModelManager: all models loaded and READY")

    def _build_decision_model(self) -> DecisionModel:
        model_name = self._config.intelligence.model_name.strip().lower()
        if model_name == "mock":
            return MockDecisionModel(self._config.intelligence)
        if model_name == "laya":
            return LayaDecisionModel(self._config.intelligence)
        raise ConfigurationError(
            f"Unknown intelligence model {model_name!r}; choose 'laya' or 'mock'."
        )

    def _build_ui_parser(self) -> UIParser:
        if not self._config.vision.enabled:
            return StaticUIParser()
        return HybridUIParser(
            self._config.vision,
            ocr=RapidOCRUIParser(self._config.vision),
            uia=UIAutomationParser(self._config.vision),
        )

    def shutdown(self) -> None:
        """Release model resources. Safe to call even if never loaded."""
        with self._lock:
            if self._stt is not None:
                self._stt.shutdown()
            close_parser = getattr(self._ui_parser, "close", None)
            if close_parser is not None:
                close_parser()
            self._stt = None
            self._decision_model = None
            self._ui_parser = None
            self._loaded = False
            logger.info("ModelManager: shut down")

    # -- Accessors --
    @property
    def stt(self) -> SpeechToTextEngine:
        if self._stt is None:
            raise ModelLoadError("SpeechToTextEngine accessed before ModelManager.load_all()")
        return self._stt

    @property
    def decision_model(self) -> DecisionModel:
        if self._decision_model is None:
            raise ModelLoadError("DecisionModel accessed before ModelManager.load_all()")
        return self._decision_model

    @property
    def ui_parser(self) -> UIParser:
        if self._ui_parser is None:
            raise ModelLoadError("UIParser accessed before ModelManager.load_all()")
        return self._ui_parser

    def set_ui_parser(self, parser: UIParser) -> None:
        """Swap in a different UI-parsing backend (e.g. a real CV adapter) after loading."""
        with self._lock:
            self._ui_parser = parser
