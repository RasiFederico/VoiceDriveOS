"""Mouse control, isolated from AI decision-making.

The AI never touches the mouse directly: it can only request a validated
:class:`intelligence.action_schema.ClickAction` / ``DragAction`` /
``MoveAction``. :class:`MouseController` is the only code allowed to call the
real backend (PyAutoGUI by default).

``pyautogui`` is imported lazily so that this module — and anything that only
needs the *interface* — can be imported in environments without a display
(e.g. CI, or this very sandbox) and fully unit-tested with a fake backend.
"""

from __future__ import annotations

import threading
from typing import Protocol

from app.config import ExecutionConfig
from core.exceptions import ExecutionError
from core.logger import get_logger

logger = get_logger(__name__)

_COORDINATE_MARGIN = 4096
"""Slack allowed beyond the reported screen size, to tolerate multi-monitor
setups where negative or over-sized coordinates are legitimate."""


class MouseBackend(Protocol):
    """Subset of the PyAutoGUI API that :class:`MouseController` depends on."""

    def size(self) -> tuple[int, int]: ...

    def moveTo(self, x: int, y: int, duration: float = 0.0) -> None: ...

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        clicks: int = 1,
        interval: float = 0.0,
        button: str = "left",
        duration: float = 0.0,
    ) -> None: ...

    def dragTo(
        self, x: int, y: int, duration: float = 0.0, button: str = "left"
    ) -> None: ...


def _load_pyautogui() -> MouseBackend:
    try:
        import pyautogui  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise ExecutionError(
            "pyautogui is not installed. Run `pip install pyautogui` to enable "
            "real mouse/keyboard control, or inject a test backend."
        ) from exc
    return pyautogui  # type: ignore[return-value]


class MouseController:
    """Thin, validated wrapper around a PyAutoGUI-compatible mouse backend."""

    def __init__(self, config: ExecutionConfig, backend: MouseBackend | None = None) -> None:
        self._config = config
        self._backend = backend
        self._lock = threading.Lock()

    @property
    def backend(self) -> MouseBackend:
        if self._backend is None:
            self._backend = _load_pyautogui()
        return self._backend

    def _validate_coordinates(self, x: int, y: int) -> None:
        if not isinstance(x, int) or not isinstance(y, int) or isinstance(x, bool) or isinstance(y, bool):
            raise ExecutionError("coordinates must be integers")
        try:
            screen_w, screen_h = self.backend.size()
        except Exception as exc:  # pragma: no cover - depends on display
            raise ExecutionError(f"unable to query screen size: {exc}") from exc
        if not (-_COORDINATE_MARGIN <= x <= screen_w + _COORDINATE_MARGIN):
            raise ExecutionError(f"x coordinate out of plausible range: {x}")
        if not (-_COORDINATE_MARGIN <= y <= screen_h + _COORDINATE_MARGIN):
            raise ExecutionError(f"y coordinate out of plausible range: {y}")

    def move(self, x: int, y: int, duration: float | None = None) -> None:
        self._validate_coordinates(x, y)
        effective_duration = duration if duration is not None else self._config.mouse_duration_s
        with self._lock:
            try:
                self.backend.moveTo(x, y, duration=effective_duration)
            except Exception as exc:  # pragma: no cover - depends on display
                raise ExecutionError(f"mouse move failed: {exc}") from exc
        logger.debug("mouse move -> (%d, %d)", x, y)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self._validate_coordinates(x, y)
        if button not in {"left", "right", "middle"}:
            raise ExecutionError(f"invalid mouse button: {button!r}")
        if clicks not in (1, 2):
            raise ExecutionError("clicks must be 1 or 2")
        with self._lock:
            try:
                self.backend.click(
                    x=x,
                    y=y,
                    clicks=clicks,
                    interval=0.05,
                    button=button,
                    duration=self._config.mouse_duration_s,
                )
            except Exception as exc:  # pragma: no cover - depends on display
                raise ExecutionError(f"mouse click failed: {exc}") from exc
        logger.info("mouse click -> (%d, %d) button=%s clicks=%d", x, y, button, clicks)

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        self.click(x, y, button=button, clicks=2)

    def drag(self, from_x: int, from_y: int, to_x: int, to_y: int, button: str = "left") -> None:
        self._validate_coordinates(from_x, from_y)
        self._validate_coordinates(to_x, to_y)
        if button not in {"left", "right", "middle"}:
            raise ExecutionError(f"invalid mouse button: {button!r}")
        with self._lock:
            try:
                self.backend.moveTo(from_x, from_y, duration=self._config.mouse_duration_s)
                self.backend.dragTo(to_x, to_y, duration=self._config.mouse_duration_s, button=button)
            except Exception as exc:  # pragma: no cover - depends on display
                raise ExecutionError(f"mouse drag failed: {exc}") from exc
        logger.info("mouse drag (%d, %d) -> (%d, %d)", from_x, from_y, to_x, to_y)
