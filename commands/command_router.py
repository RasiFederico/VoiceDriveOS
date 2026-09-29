"""Routes a parsed :class:`Command` to its execution path.

Two paths exist, mirroring the project brief's core pipeline diagram:

* **Screen-free commands** (``TYPE``, coordinate ``CLICK``, ``PRESS_KEY``,
  ``HOTKEY``, ``SCROLL``, ``OPEN``, ``CLOSE``, ``BACK``, ``FORWARD``, ``WAIT``) are mapped straight to
  a validated :class:`~intelligence.action_schema.Action` here, with **no
  involvement of the decision core**. There is nothing to decide: the intent
  parser already extracted every field the action needs.
* **Screen-dependent commands** (label clicks, drags, and pointer moves) cannot
  be resolved without first capturing the screen and matching OCR labels. This module only tells the caller
  *that* a command needs the screen; the orchestration itself lives in
  ``app/main.py``.
"""

from __future__ import annotations

from commands.command_types import Command, CommandType
from core.exceptions import IntentParsingError
from core.logger import get_logger
from intelligence.action_schema import (
    Action,
    BackAction,
    ClickAction,
    CloseAction,
    ForwardAction,
    HotkeyAction,
    OpenAction,
    PressKeyAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)

logger = get_logger(__name__)


def requires_screen(command: Command) -> bool:
    """Return True if resolving this command needs a screenshot + UI parsing."""
    if command.type is CommandType.CLICK and command.x is not None and command.y is not None:
        return False
    return command.requires_screen or command.type in {
        CommandType.CLICK,
        CommandType.DRAG,
        CommandType.MOVE,
    }


def command_to_direct_action(command: Command) -> Action:
    """Deterministically map a screen-free command straight to a validated Action.

    Raises:
        IntentParsingError: if ``command`` requires the screen (use the
            vision + decision pipeline instead), or if a required field the
            parser should have populated is missing.
    """
    if requires_screen(command):
        raise IntentParsingError(
            f"command type '{command.type.value}' requires screen context; "
            "route through vision + the decision core instead"
        )

    try:
        if command.type is CommandType.TYPE:
            assert command.text is not None
            return TypeAction(text=command.text)

        if command.type is CommandType.CLICK:
            assert command.x is not None and command.y is not None
            return ClickAction(
                x=command.x,
                y=command.y,
                button=command.button,
                clicks=command.clicks,
            )

        if command.type is CommandType.PRESS_KEY:
            assert command.key is not None
            return PressKeyAction(key=command.key)

        if command.type is CommandType.HOTKEY:
            assert command.keys is not None
            return HotkeyAction(keys=command.keys)

        if command.type is CommandType.SCROLL:
            assert command.direction is not None
            return ScrollAction(direction=command.direction, amount=command.amount or 3)

        if command.type is CommandType.OPEN:
            assert command.app_name is not None
            return OpenAction(app_name=command.app_name)

        if command.type is CommandType.CLOSE:
            return CloseAction()

        if command.type is CommandType.BACK:
            return BackAction()

        if command.type is CommandType.FORWARD:
            return ForwardAction()

        if command.type is CommandType.WAIT:
            return WaitAction(seconds=command.duration_seconds or 1.0)
    except AssertionError as exc:
        raise IntentParsingError(
            f"command type '{command.type.value}' is missing a required field for direct mapping"
        ) from exc

    raise IntentParsingError(f"unhandled command type: {command.type.value}")
