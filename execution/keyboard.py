"""Keyboard control, isolated from AI decision-making.

Only key names present in :data:`intelligence.action_schema.ALLOWED_KEYS` can
ever reach the backend, so the action-schema validation and the execution
layer always agree on what a "valid key" is.
"""

from __future__ import annotations

import threading
from typing import Protocol

from app.config import ExecutionConfig
from core.exceptions import ExecutionError
from core.logger import get_logger
from intelligence.action_schema import ALLOWED_KEYS

logger = get_logger(__name__)


class KeyboardBackend(Protocol):
    """Subset of the PyAutoGUI API that :class:`KeyboardController` depends on."""

    def press(self, key: str) -> None: ...

    def hotkey(self, *keys: str) -> None: ...


def _load_pyautogui() -> KeyboardBackend:
    try:
        import pyautogui  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise ExecutionError(
            "pyautogui is not installed. Run `pip install pyautogui` to enable "
            "real keyboard control, or inject a test backend."
        ) from exc
    return pyautogui  # type: ignore[return-value]


def _normalize_key(key: str) -> str:
    normalized = key.strip().lower()
    if normalized not in ALLOWED_KEYS:
        raise ExecutionError(f"key not allowed: {key!r}")
    return normalized


class KeyboardController:
    """Thin, validated wrapper around a PyAutoGUI-compatible keyboard backend."""

    def __init__(self, config: ExecutionConfig, backend: KeyboardBackend | None = None) -> None:
        self._config = config
        self._backend = backend
        self._lock = threading.Lock()

    @property
    def backend(self) -> KeyboardBackend:
        if self._backend is None:
            self._backend = _load_pyautogui()
        return self._backend

    def press_key(self, key: str) -> None:
        normalized = _normalize_key(key)
        with self._lock:
            try:
                self.backend.press(normalized)
            except Exception as exc:  # pragma: no cover - depends on display
                raise ExecutionError(f"press_key failed: {exc}") from exc
        logger.info("keypress %s", normalized)

    def hotkey(self, keys: tuple[str, ...]) -> None:
        if len(keys) < 2:
            raise ExecutionError("hotkey requires at least two keys")
        normalized = tuple(_normalize_key(k) for k in keys)
        with self._lock:
            try:
                self.backend.hotkey(*normalized)
            except Exception as exc:  # pragma: no cover - depends on display
                raise ExecutionError(f"hotkey failed: {exc}") from exc
        logger.info("hotkey %s", "+".join(normalized))

    def paste(self) -> None:
        """Trigger the OS paste shortcut, used after ``ClipboardController.copy``."""
        self.hotkey(self._config.clipboard_paste_hotkey)
