"""Clipboard control.

Per the project brief, typed text is never fed to PyAutoGUI character by
character. Instead the deterministically-extracted string is copied to the
clipboard here, and :meth:`execution.keyboard.KeyboardController.paste`
triggers the OS paste shortcut. This is both faster and more reliable than
simulating keystrokes for long strings.
"""

from __future__ import annotations

import threading
from typing import Protocol

from core.exceptions import ExecutionError
from core.logger import get_logger

logger = get_logger(__name__)


class ClipboardBackend(Protocol):
    """Subset of the pyperclip API that :class:`ClipboardController` depends on."""

    def copy(self, text: str) -> None: ...

    def paste(self) -> str: ...


def _load_pyperclip() -> ClipboardBackend:
    try:
        import pyperclip  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise ExecutionError(
            "pyperclip is not installed. Run `pip install pyperclip` to enable "
            "real clipboard access, or inject a test backend."
        ) from exc
    return pyperclip  # type: ignore[return-value]


class ClipboardController:
    """Thin, validated wrapper around a pyperclip-compatible clipboard backend."""

    def __init__(self, backend: ClipboardBackend | None = None) -> None:
        self._backend = backend
        self._lock = threading.Lock()

    @property
    def backend(self) -> ClipboardBackend:
        if self._backend is None:
            self._backend = _load_pyperclip()
        return self._backend

    def copy(self, text: str) -> None:
        if not isinstance(text, str):
            raise ExecutionError("clipboard text must be a string")
        with self._lock:
            try:
                self.backend.copy(text)
            except Exception as exc:  # pragma: no cover - depends on OS clipboard
                raise ExecutionError(f"clipboard copy failed: {exc}") from exc
        logger.debug("clipboard copy len=%d", len(text))

    def read(self) -> str:
        with self._lock:
            try:
                return self.backend.paste()
            except Exception as exc:  # pragma: no cover - depends on OS clipboard
                raise ExecutionError(f"clipboard read failed: {exc}") from exc
