"""End-to-end pipeline simulation, without any real audio/vision/OS access.

Exercises: transcript -> intent parser -> (direct action | vision + decision
core) -> validation -> executor, using a mock executor so no real PyAutoGUI
call ever happens.
"""

from __future__ import annotations

import unittest

from app.config import AppConfig, IntelligenceConfig
from commands.command_router import command_to_direct_action, requires_screen
from core.exceptions import ValidationError
from intelligence.action_schema import ClickAction, NoAction, TypeAction
from intelligence.intent_parser import parse_intent
from intelligence.laya import MockDecisionModel
from vision.element import UIElement
from vision.ui_parser import StaticUIParser


class MockExecutor:
    """Records executed actions instead of touching the OS."""

    def __init__(self) -> None:
        self.executed: list = []

    def execute(self, action) -> None:
        self.executed.append(action)


def _validate(action, confidence_threshold: float):
    if isinstance(action, NoAction):
        return action
    if action.confidence < confidence_threshold:
        raise ValidationError(f"confidence {action.confidence:.2f} below threshold {confidence_threshold:.2f}")
    return action


class ScreenFreePipelineTests(unittest.TestCase):
    """'write hello world' -> TYPE command -> TypeAction -> executor, no screen/decision core involved."""

    def test_write_command_end_to_end(self) -> None:
        transcript = "write hello world"
        command = parse_intent(transcript)
        self.assertFalse(requires_screen(command))

        action = command_to_direct_action(command)
        self.assertIsInstance(action, TypeAction)
        self.assertEqual(action.text, "hello world")

        validated = _validate(action, confidence_threshold=0.6)

        executor = MockExecutor()
        executor.execute(validated)
        self.assertEqual(len(executor.executed), 1)
        self.assertEqual(executor.executed[0].text, "hello world")

    def test_press_enter_end_to_end(self) -> None:
        command = parse_intent("press enter")
        action = command_to_direct_action(command)
        executor = MockExecutor()
        executor.execute(_validate(action, 0.6))
        self.assertEqual(executor.executed[0].key, "enter")

    def test_coordinate_click_maps_to_direct_action(self) -> None:
        command = parse_intent("click at 400, 250")
        self.assertFalse(requires_screen(command))
        action = command_to_direct_action(command)
        self.assertIsInstance(action, ClickAction)
        self.assertEqual((action.x, action.y), (400, 250))


class ClickPipelineTests(unittest.TestCase):
    """'click login' -> CLICK command -> vision (static) -> mock decision core -> executor."""

    def setUp(self) -> None:
        self.elements = [
            UIElement(id=0, type="icon", text="Menu", x=0, y=0, width=40, height=40),
            UIElement(id=1, type="input", text="Username", x=100, y=100, width=200, height=30),
            UIElement(id=2, type="input", text="Password", x=100, y=150, width=200, height=30),
            UIElement(id=3, type="button", text="Login", x=100, y=200, width=100, height=40),
            UIElement(id=4, type="button", text="Register", x=250, y=200, width=100, height=40),
        ]
        self.ui_parser = StaticUIParser(self.elements)
        self.decision_model = MockDecisionModel(IntelligenceConfig(confidence_threshold=0.5))

    def test_click_login_button_selects_correct_element(self) -> None:
        command = parse_intent("click the login button")
        self.assertTrue(requires_screen(command))

        elements = self.ui_parser.parse(screenshot=None)  # type: ignore[arg-type]
        action = self.decision_model.decide(command.target_text or command.raw_text, elements)

        self.assertIsInstance(action, ClickAction)
        self.assertEqual(action.target_id, 3)

        validated = _validate(action, confidence_threshold=0.5)
        executor = MockExecutor()
        executor.execute(validated)
        self.assertEqual(executor.executed[0].target_id, 3)

    def test_click_nonexistent_target_yields_no_action(self) -> None:
        command = parse_intent("click the nonexistent quantum button")
        elements = self.ui_parser.parse(screenshot=None)  # type: ignore[arg-type]
        action = self.decision_model.decide(command.target_text or command.raw_text, elements)
        self.assertIsInstance(action, NoAction)

        # NoAction always passes validation (there is nothing unsafe about doing nothing)
        validated = _validate(action, confidence_threshold=0.5)
        executor = MockExecutor()
        executor.execute(validated)
        self.assertIsInstance(executor.executed[0], NoAction)

    def test_low_confidence_action_rejected_by_validation(self) -> None:
        low_confidence_click = ClickAction(target_id=3, confidence=0.1)
        with self.assertRaises(ValidationError):
            _validate(low_confidence_click, confidence_threshold=0.5)

    def test_empty_element_list_yields_no_action(self) -> None:
        action = self.decision_model.decide("click login", elements=[])
        self.assertIsInstance(action, NoAction)

    def test_double_click_label_is_selected_by_screen_match(self) -> None:
        command = parse_intent("double click on login")
        elements = self.ui_parser.parse(screenshot=None)  # type: ignore[arg-type]
        selected = self.decision_model.decide(command.target_text or "", elements)
        action = ClickAction(
            target_id=selected.target_id,
            confidence=selected.confidence,
            clicks=command.clicks,
            button=command.button,
        )
        self.assertEqual(action.target_id, 3)
        self.assertEqual(action.clicks, 2)

    def test_italian_label_matches_english_ui_text(self) -> None:
        model = MockDecisionModel(IntelligenceConfig())
        elements = [UIElement(id=1, type="text", text="Settings", x=0, y=0, width=80, height=20)]
        action = model.decide("impostazioni", elements)
        self.assertIsInstance(action, ClickAction)
        self.assertEqual(action.target_id, 1)


class AppConfigSmokeTest(unittest.TestCase):
    """Confirms the whole config tree constructs without error (used implicitly by app/main.py)."""

    def test_default_config_loads(self) -> None:
        config = AppConfig.load()
        self.assertGreater(config.audio.sample_rate, 0)
        self.assertTrue(0.0 < config.vision.capture_scale <= 1.0)


if __name__ == "__main__":
    unittest.main()
