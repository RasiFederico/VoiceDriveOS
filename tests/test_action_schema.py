"""Tests for intelligence.action_schema: the AI-to-executor validation boundary."""

from __future__ import annotations

import unittest

from core.exceptions import ActionSchemaError
from intelligence.action_schema import (
    ClickAction,
    HotkeyAction,
    NoAction,
    ScrollAction,
    TypeAction,
    action_from_dict,
)


class ActionSchemaTests(unittest.TestCase):
    def test_valid_click_action(self) -> None:
        action = action_from_dict({"action": "click", "target_id": 3})
        self.assertIsInstance(action, ClickAction)
        self.assertEqual(action.target_id, 3)
        self.assertEqual(action.confidence, 1.0)

    def test_click_options_and_coordinates(self) -> None:
        action = action_from_dict(
            {"action": "click", "x": 400, "y": 250, "button": "right", "clicks": 2}
        )
        self.assertEqual((action.x, action.y), (400, 250))
        self.assertEqual(action.button, "right")
        self.assertEqual(action.clicks, 2)

    def test_click_requires_both_coordinates(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "click", "x": 400})

    def test_valid_type_action(self) -> None:
        action = action_from_dict({"action": "type", "text": "hello world"})
        self.assertIsInstance(action, TypeAction)
        self.assertEqual(action.text, "hello world")

    def test_valid_scroll_action(self) -> None:
        action = action_from_dict({"action": "scroll", "direction": "down", "amount": 5})
        self.assertIsInstance(action, ScrollAction)
        self.assertEqual(action.amount, 5)

    def test_valid_hotkey_action_normalizes_list_to_tuple(self) -> None:
        action = action_from_dict({"action": "hotkey", "keys": ["ctrl", "c"]})
        self.assertIsInstance(action, HotkeyAction)
        self.assertEqual(action.keys, ("ctrl", "c"))

    def test_no_action_reason_optional(self) -> None:
        action = action_from_dict({"action": "no_action"})
        self.assertIsInstance(action, NoAction)
        self.assertEqual(action.reason, "")

    def test_unknown_action_type_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "launch_nuclear_missile"})

    def test_missing_action_field_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"target_id": 3})

    def test_missing_required_field_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "type"})  # missing 'text'

    def test_negative_target_id_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "click", "target_id": -1})

    def test_empty_text_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "type", "text": ""})

    def test_invalid_scroll_direction_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "scroll", "direction": "sideways", "amount": 1})

    def test_unsupported_key_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "keypress", "key": "; rm -rf /"})

    def test_extra_unknown_field_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "click", "target_id": 1, "shell_command": "echo hi"})

    def test_confidence_out_of_range_rejected(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict({"action": "click", "target_id": 1, "confidence": 1.5})

    def test_payload_must_be_mapping(self) -> None:
        with self.assertRaises(ActionSchemaError):
            action_from_dict("click 3")  # type: ignore[arg-type]

    def test_to_dict_roundtrip_shape(self) -> None:
        action = action_from_dict({"action": "click", "target_id": 7, "confidence": 0.9})
        data = action.to_dict()
        self.assertEqual(data["action"], "click")
        self.assertEqual(data["target_id"], 7)


if __name__ == "__main__":
    unittest.main()
