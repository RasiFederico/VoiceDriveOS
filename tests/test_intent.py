"""Tests for intelligence.intent_parser.parse_intent."""

from __future__ import annotations

import unittest

from commands.command_types import CommandType
from intelligence.intent_parser import parse_intent


class IntentParserTests(unittest.TestCase):
    def test_click_requires_screen(self) -> None:
        command = parse_intent("click settings")
        self.assertEqual(command.type, CommandType.CLICK)
        self.assertTrue(command.requires_screen)
        self.assertEqual(command.target_text, "settings")

    def test_click_with_filler_words(self) -> None:
        command = parse_intent("click the login button")
        self.assertEqual(command.type, CommandType.CLICK)
        self.assertEqual(command.target_text, "login button")

    def test_type_extracts_literal_text(self) -> None:
        command = parse_intent("write Hello World")
        self.assertEqual(command.type, CommandType.TYPE)
        self.assertFalse(command.requires_screen)
        self.assertEqual(command.text, "Hello World")

    def test_press_key(self) -> None:
        command = parse_intent("press enter")
        self.assertEqual(command.type, CommandType.PRESS_KEY)
        self.assertEqual(command.key, "enter")
        self.assertFalse(command.requires_screen)

    def test_hotkey(self) -> None:
        command = parse_intent("press control c")
        self.assertEqual(command.type, CommandType.HOTKEY)
        self.assertEqual(command.keys, ("ctrl", "c"))

    def test_three_key_hotkey_and_spoken_italian_aliases(self) -> None:
        command = parse_intent("premi Ctrl Maiuscole S")
        self.assertEqual(command.type, CommandType.HOTKEY)
        self.assertEqual(command.keys, ("ctrl", "shift", "s"))

    def test_plus_separated_hotkey(self) -> None:
        command = parse_intent("press ctrl+alt+delete")
        self.assertEqual(command.keys, ("ctrl", "alt", "delete"))

    def test_function_key(self) -> None:
        command = parse_intent("press F11")
        self.assertEqual(command.type, CommandType.PRESS_KEY)
        self.assertEqual(command.key, "f11")

    def test_common_shortcut_aliases(self) -> None:
        command = parse_intent("seleziona tutto")
        self.assertEqual(command.type, CommandType.HOTKEY)
        self.assertEqual(command.keys, ("ctrl", "a"))

    def test_coordinate_click_is_direct(self) -> None:
        command = parse_intent("clicca alle coordinate 800 420")
        self.assertEqual(command.type, CommandType.CLICK)
        self.assertFalse(command.requires_screen)
        self.assertEqual((command.x, command.y), (800, 420))

    def test_double_click(self) -> None:
        command = parse_intent("doppio clic sul documento")
        self.assertEqual(command.type, CommandType.CLICK)
        self.assertEqual(command.clicks, 2)
        self.assertEqual(command.target_text, "documento")

    def test_right_click(self) -> None:
        command = parse_intent("right click the file")
        self.assertEqual(command.button, "right")
        self.assertEqual(command.target_text, "file")

    def test_drag_and_move_use_screen_targets(self) -> None:
        drag = parse_intent("trascina il file sul cestino")
        self.assertEqual(drag.type, CommandType.DRAG)
        self.assertEqual(drag.source_text, "file")
        self.assertEqual(drag.target_text, "cestino")
        self.assertTrue(drag.requires_screen)
        move = parse_intent("sposta il puntatore su Impostazioni")
        self.assertEqual(move.type, CommandType.MOVE)
        self.assertEqual(move.target_text, "Impostazioni")

    def test_scroll_amount(self) -> None:
        command = parse_intent("scorri giù di 7 scatti")
        self.assertEqual(command.type, CommandType.SCROLL)
        self.assertEqual(command.direction, "down")
        self.assertEqual(command.amount, 7)

    def test_insert_text_in_italian(self) -> None:
        command = parse_intent("inserisci Ciao mondo")
        self.assertEqual(command.type, CommandType.TYPE)
        self.assertEqual(command.text, "Ciao mondo")

    def test_scroll_down(self) -> None:
        command = parse_intent("scroll down")
        self.assertEqual(command.type, CommandType.SCROLL)
        self.assertEqual(command.direction, "down")
        self.assertGreater(command.amount or 0, 0)

    def test_scroll_up(self) -> None:
        command = parse_intent("scroll up")
        self.assertEqual(command.direction, "up")

    def test_open_app(self) -> None:
        command = parse_intent("open chrome")
        self.assertEqual(command.type, CommandType.OPEN)
        self.assertEqual(command.app_name, "chrome")
        self.assertFalse(command.requires_screen)

    def test_go_back(self) -> None:
        command = parse_intent("go back")
        self.assertEqual(command.type, CommandType.BACK)

    def test_go_forward(self) -> None:
        command = parse_intent("go forward")
        self.assertEqual(command.type, CommandType.FORWARD)

    def test_close(self) -> None:
        command = parse_intent("close this window")
        self.assertEqual(command.type, CommandType.CLOSE)

    def test_wait_with_explicit_duration(self) -> None:
        command = parse_intent("wait 2 seconds")
        self.assertEqual(command.type, CommandType.WAIT)
        self.assertEqual(command.duration_seconds, 2.0)

    def test_wait_default_duration(self) -> None:
        command = parse_intent("wait")
        self.assertEqual(command.type, CommandType.WAIT)
        self.assertIsNotNone(command.duration_seconds)

    def test_unknown_command(self) -> None:
        command = parse_intent("what is the meaning of life")
        self.assertEqual(command.type, CommandType.UNKNOWN)

    def test_empty_transcript(self) -> None:
        command = parse_intent("")
        self.assertEqual(command.type, CommandType.UNKNOWN)

    def test_italian_click(self) -> None:
        command = parse_intent("clicca sul pulsante impostazioni")
        self.assertEqual(command.type, CommandType.CLICK)
        self.assertTrue(command.requires_screen)


if __name__ == "__main__":
    unittest.main()
