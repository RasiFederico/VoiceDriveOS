"""The executor: the only component allowed to touch the operating system.

``Executor.execute()`` accepts nothing but a validated
:class:`intelligence.action_schema.Action` — never a raw dict, never a
string, never "code". Target ids are resolved to screen coordinates through
an injected resolver callback so this module has no compile-time dependency
on the vision package (keeping the dependency graph a DAG, per the project's
module-independence rule).

This module deliberately performs one more semantic check per action type
(e.g. "does this target_id actually resolve to something clickable?") on top
of whatever ``commands.command_router`` / the validator already did, because
defense in depth costs little and the alternative — a stray physical click —
is not something you get to undo.
"""

from __future__ import annotations

import platform
import re
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from typing import TYPE_CHECKING, Callable

from app.config import ExecutionConfig
from core.exceptions import ExecutionError
from core.logger import get_logger
from execution.clipboard import ClipboardController
from execution.keyboard import KeyboardController
from execution.mouse import MouseController
from intelligence.action_schema import (
    Action,
    ActionType,
    ClickAction,
    CloseAction,
    DragAction,
    HotkeyAction,
    MoveAction,
    NoAction,
    OpenAction,
    PressKeyAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)

if TYPE_CHECKING:
    from vision.element import UIElement

logger = get_logger(__name__)

# Platform-specific defaults for actions with no explicit coordinates.
_BACK_HOTKEY: tuple[str, ...] = ("alt", "left")
_FORWARD_HOTKEY: tuple[str, ...] = ("alt", "right")
_CLOSE_HOTKEY_WINDOWS: tuple[str, ...] = ("alt", "f4")
_CLOSE_HOTKEY_MACOS: tuple[str, ...] = ("cmd", "w")
_CLOSE_HOTKEY_LINUX: tuple[str, ...] = ("alt", "f4")

TargetResolver = Callable[[int], "UIElement | None"]
"""Resolves a UI element id (from the last vision pass) to a UIElement, or None."""


def _default_resolver(_target_id: int) -> "UIElement | None":
    return None


def _launch_application(app_name: str) -> None:
    """Best-effort, cross-platform application launch.

    This intentionally does not attempt a fuzzy "find the right app" search:
    it either finds an executable of that name on PATH, or (on Windows) asks
    the shell to resolve it via ``os.startfile``, or raises.
    """
    system = platform.system()
    try:
        target = app_name.strip().strip('"')
        if re.match(r"^(?:https?://|www\.)", target, re.IGNORECASE):
            url = target if "://" in target else f"https://{target}"
            if not webbrowser.open(url):
                raise ExecutionError(f"could not open URL: {url!r}")
            return

        if system == "Windows":
            aliases = {
                "google chrome": "chrome.exe", "chrome": "chrome.exe",
                "microsoft edge": "msedge.exe", "edge": "msedge.exe",
                "firefox": "firefox.exe", "notepad": "notepad.exe",
                "blocco note": "notepad.exe", "calcolatrice": "calc.exe",
                "calculator": "calc.exe", "calc": "calc.exe",
                "paint": "mspaint.exe", "mspaint": "mspaint.exe",
                "file explorer": "explorer.exe", "explorer": "explorer.exe",
                "esplora file": "explorer.exe",
            }
            requested = aliases.get(target.casefold(), target)
            executable = shutil.which(requested)
            if executable is None and not os.path.exists(requested):
                executable = _windows_app_path(requested)
            if executable is not None:
                subprocess.Popen([executable])  # noqa: S603 - resolved via PATH, no shell
                return
            os.startfile(target)  # type: ignore[attr-defined] - delegates to Windows shell associations
            return
        resolved = shutil.which(target)
        if resolved is None:
            raise ExecutionError(f"application not found on PATH: {target!r}")
        subprocess.Popen([resolved])  # noqa: S603 - resolved via shutil.which, not user shell input
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError(f"failed to launch application {app_name!r}: {exc}") from exc


def _windows_app_path(executable_name: str) -> str | None:
    """Resolve an installed app through the standard Windows App Paths registry."""
    import winreg

    name = executable_name if executable_name.lower().endswith(".exe") else f"{executable_name}.exe"
    key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{name}"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, None)
            if value and os.path.isfile(value):
                return value
        except OSError:
            continue
    return None


class Executor:
    """Dispatches validated :class:`Action` instances to mouse/keyboard/clipboard controllers."""

    def __init__(
        self,
        config: ExecutionConfig,
        mouse: MouseController | None = None,
        keyboard: KeyboardController | None = None,
        clipboard: ClipboardController | None = None,
        resolve_target: TargetResolver = _default_resolver,
    ) -> None:
        self._config = config
        self.mouse = mouse or MouseController(config)
        self.keyboard = keyboard or KeyboardController(config)
        self.clipboard = clipboard or ClipboardController()
        self._resolve_target = resolve_target
        self._lock = threading.Lock()

    def set_target_resolver(self, resolver: TargetResolver) -> None:
        """Wire in the current vision layer's id-to-element lookup."""
        self._resolve_target = resolver

    def _resolve_or_raise(self, target_id: int) -> "UIElement":
        element = self._resolve_target(target_id)
        if element is None:
            raise ExecutionError(f"target_id {target_id} does not resolve to a known UI element")
        if not getattr(element, "clickable", True):
            raise ExecutionError(f"target_id {target_id} is not clickable")
        return element

    def execute(self, action: Action) -> None:
        """Execute a single validated action. Blocking — call off the event loop thread."""
        with self._lock:
            logger.info("execution_start action=%s", action.action.value)
            try:
                self._dispatch(action)
            except ExecutionError:
                raise
            except Exception as exc:  # defense in depth: never let a raw exception look like success
                raise ExecutionError(f"unexpected failure executing {action.action.value}: {exc}") from exc
            finally:
                logger.info("execution_end action=%s", action.action.value)

    def _dispatch(self, action: Action) -> None:
        if isinstance(action, ClickAction):
            if action.target_id is not None:
                element = self._resolve_or_raise(action.target_id)
                x, y = element.center()
            else:
                x, y = action.x, action.y
            assert x is not None and y is not None
            self.mouse.click(x, y, button=action.button, clicks=action.clicks)
            return

        if isinstance(action, TypeAction):
            self.clipboard.copy(action.text)
            self.keyboard.paste()
            return

        if isinstance(action, PressKeyAction):
            self.keyboard.press_key(action.key)
            return

        if isinstance(action, HotkeyAction):
            self.keyboard.hotkey(action.keys)
            return

        if isinstance(action, ScrollAction):
            self._scroll(action)
            return

        if isinstance(action, DragAction):
            self._drag(action)
            return

        if isinstance(action, MoveAction):
            self._move(action)
            return

        if isinstance(action, OpenAction):
            _launch_application(action.app_name)
            return

        if isinstance(action, CloseAction):
            self.keyboard.hotkey(self._close_hotkey())
            return

        if action.action is ActionType.BACK:
            self.keyboard.hotkey(_BACK_HOTKEY)
            return

        if action.action is ActionType.FORWARD:
            self.keyboard.hotkey(_FORWARD_HOTKEY)
            return

        if isinstance(action, WaitAction):
            time.sleep(action.seconds)
            return

        if isinstance(action, NoAction):
            logger.info("no_action reason=%s", action.reason or "<none>")
            return

        raise ExecutionError(f"no dispatcher registered for action type: {action.action}")

    @staticmethod
    def _close_hotkey() -> tuple[str, ...]:
        system = platform.system()
        if system == "Darwin":
            return _CLOSE_HOTKEY_MACOS
        if system == "Windows":
            return _CLOSE_HOTKEY_WINDOWS
        return _CLOSE_HOTKEY_LINUX

    def _scroll(self, action: ScrollAction) -> None:
        # pyautogui.scroll() takes positive=up, negative=down "clicks".
        amount = action.amount if action.direction == "up" else -action.amount
        try:
            self.mouse.backend.scroll(amount)  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise ExecutionError("configured mouse backend does not support scroll()") from exc
        except Exception as exc:  # pragma: no cover - depends on display
            raise ExecutionError(f"scroll failed: {exc}") from exc
        logger.info("scroll direction=%s amount=%d", action.direction, action.amount)

    def _drag(self, action: DragAction) -> None:
        if action.from_target_id is not None and action.to_target_id is not None:
            start = self._resolve_or_raise(action.from_target_id)
            end = self._resolve_or_raise(action.to_target_id)
            from_x, from_y = start.center()
            to_x, to_y = end.center()
        else:
            from_x, from_y = action.from_x, action.from_y  # type: ignore[assignment]
            to_x, to_y = action.to_x, action.to_y  # type: ignore[assignment]
        self.mouse.drag(from_x, from_y, to_x, to_y)

    def _move(self, action: MoveAction) -> None:
        if action.target_id is not None:
            element = self._resolve_or_raise(action.target_id)
            x, y = element.center()
        else:
            x, y = action.x, action.y  # type: ignore[assignment]
        self.mouse.move(x, y)
