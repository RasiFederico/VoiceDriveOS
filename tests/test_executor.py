"""Tests for execution.executor.Executor.

All PyAutoGUI/pyperclip access is replaced with fake, in-memory backends —
no real mouse click, keystroke, or clipboard write ever happens during tests.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.config import ExecutionConfig
from core.exceptions import ExecutionError
from execution.clipboard import ClipboardController
from execution.executor import Executor
from execution.keyboard import KeyboardController
from execution.mouse import MouseController
from intelligence.action_schema import (
    BackAction,
    ClickAction,
    CloseAction,
    ForwardAction,
    HotkeyAction,
    NoAction,
    PressKeyAction,
    ScrollAction,
    TypeAction,
    WaitAction,
    action_from_dict,
)
from vision.element import UIElement


class FakeMouseBackend:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def size(self) -> tuple[int, int]:
        return (1920, 1080)

    def moveTo(self, x: int, y: int, duration: float = 0.0) -> None:
        self.calls.append(("moveTo", x, y))

    def click(self, x=None, y=None, clicks=1, interval=0.0, button="left", duration=0.0) -> None:
        self.calls.append(("click", x, y, button, clicks))

    def dragTo(self, x: int, y: int, duration: float = 0.0, button: str = "left") -> None:
        self.calls.append(("dragTo", x, y, button))

    def scroll(self, amount: int) -> None:
        self.calls.append(("scroll", amount))


class FakeKeyboardBackend:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def press(self, key: str) -> None:
        self.calls.append(("press", key))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", keys))


class FakeClipboardBackend:
    def __init__(self) -> None:
        self.data = ""

    def copy(self, text: str) -> None:
        self.data = text

    def paste(self) -> str:
        return self.data


def _make_executor() -> tuple[Executor, FakeMouseBackend, FakeKeyboardBackend, FakeClipboardBackend]:
    config = ExecutionConfig()
    mouse_backend = FakeMouseBackend()
    keyboard_backend = FakeKeyboardBackend()
    clipboard_backend = FakeClipboardBackend()

    mouse = MouseController(config, backend=mouse_backend)
    keyboard = KeyboardController(config, backend=keyboard_backend)
    clipboard = ClipboardController(backend=clipboard_backend)

    login_element = UIElement(id=3, type="button", text="Login", x=900, y=600, width=140, height=50)

    def resolver(target_id: int):
        return login_element if target_id == 3 else None

    executor = Executor(config, mouse=mouse, keyboard=keyboard, clipboard=clipboard, resolve_target=resolver)
    return executor, mouse_backend, keyboard_backend, clipboard_backend


class ExecutorTests(unittest.TestCase):
    def test_click_resolves_target_and_clicks_center(self) -> None:
        executor, mouse_backend, _, _ = _make_executor()
        executor.execute(action_from_dict({"action": "click", "target_id": 3}))
        self.assertEqual(len(mouse_backend.calls), 1)
        call = mouse_backend.calls[0]
        self.assertEqual(call[0], "click")
        self.assertEqual((call[1], call[2]), (970, 625))  # center of the fixture element

    def test_right_double_click_preserves_button_and_count(self) -> None:
        executor, mouse_backend, _, _ = _make_executor()
        executor.execute(ClickAction(target_id=3, button="right", clicks=2))
        self.assertEqual(mouse_backend.calls, [("click", 970, 625, "right", 2)])

    def test_coordinate_click_does_not_need_a_vision_target(self) -> None:
        executor, mouse_backend, _, _ = _make_executor()
        executor.execute(ClickAction(x=400, y=250))
        self.assertEqual(mouse_backend.calls, [("click", 400, 250, "left", 1)])

    def test_click_unknown_target_raises(self) -> None:
        executor, _, _, _ = _make_executor()
        with self.assertRaises(ExecutionError):
            executor.execute(action_from_dict({"action": "click", "target_id": 999}))

    def test_type_copies_to_clipboard_and_pastes(self) -> None:
        executor, _, keyboard_backend, clipboard_backend = _make_executor()
        executor.execute(TypeAction(text="hello world"))
        self.assertEqual(clipboard_backend.data, "hello world")
        self.assertIn(("hotkey", ("ctrl", "v")), keyboard_backend.calls)

    def test_press_key(self) -> None:
        executor, _, keyboard_backend, _ = _make_executor()
        executor.execute(PressKeyAction(key="enter"))
        self.assertEqual(keyboard_backend.calls, [("press", "enter")])

    def test_hotkey(self) -> None:
        executor, _, keyboard_backend, _ = _make_executor()
        executor.execute(HotkeyAction(keys=("ctrl", "c")))
        self.assertEqual(keyboard_backend.calls, [("hotkey", ("ctrl", "c"))])

    def test_scroll_down_is_negative(self) -> None:
        executor, mouse_backend, _, _ = _make_executor()
        executor.execute(ScrollAction(direction="down", amount=5))
        self.assertEqual(mouse_backend.calls, [("scroll", -5)])

    def test_scroll_up_is_positive(self) -> None:
        executor, mouse_backend, _, _ = _make_executor()
        executor.execute(ScrollAction(direction="up", amount=5))
        self.assertEqual(mouse_backend.calls, [("scroll", 5)])

    def test_back_triggers_alt_left(self) -> None:
        executor, _, keyboard_backend, _ = _make_executor()
        executor.execute(BackAction())
        self.assertEqual(keyboard_backend.calls, [("hotkey", ("alt", "left"))])

    def test_forward_triggers_alt_right(self) -> None:
        executor, _, keyboard_backend, _ = _make_executor()
        executor.execute(ForwardAction())
        self.assertEqual(keyboard_backend.calls, [("hotkey", ("alt", "right"))])

    def test_close_triggers_platform_hotkey(self) -> None:
        executor, _, keyboard_backend, _ = _make_executor()
        executor.execute(CloseAction())
        self.assertEqual(len(keyboard_backend.calls), 1)
        self.assertEqual(keyboard_backend.calls[0][0], "hotkey")

    def test_wait_sleeps_without_error(self) -> None:
        executor, _, _, _ = _make_executor()
        executor.execute(WaitAction(seconds=0.01))  # short, keeps the test fast

    def test_no_action_is_a_noop(self) -> None:
        executor, mouse_backend, keyboard_backend, _ = _make_executor()
        executor.execute(NoAction(reason="nothing to do"))
        self.assertEqual(mouse_backend.calls, [])
        self.assertEqual(keyboard_backend.calls, [])

    def test_open_delegates_to_application_launcher(self) -> None:
        executor, _, _, _ = _make_executor()
        with patch("execution.executor._launch_application") as fake_launch:
            executor.execute(action_from_dict({"action": "open", "app_name": "chrome"}))
            fake_launch.assert_called_once_with("chrome")

    def test_drag_by_target_id(self) -> None:
        config = ExecutionConfig()
        mouse_backend = FakeMouseBackend()
        mouse = MouseController(config, backend=mouse_backend)
        start = UIElement(id=1, type="icon", text="A", x=0, y=0, width=20, height=20)
        end = UIElement(id=2, type="icon", text="B", x=100, y=100, width=20, height=20)

        def resolver(target_id: int):
            return {1: start, 2: end}.get(target_id)

        executor = Executor(config, mouse=mouse, resolve_target=resolver)
        executor.execute(action_from_dict({"action": "drag", "from_target_id": 1, "to_target_id": 2}))
        self.assertEqual(mouse_backend.calls[0], ("moveTo", 10, 10))
        self.assertEqual(mouse_backend.calls[1], ("dragTo", 110, 110, "left"))


if __name__ == "__main__":
    unittest.main()
