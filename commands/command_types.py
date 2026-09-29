"""Intermediate command representation.

A :class:`Command` sits between the raw transcript and the final, strictly
validated :class:`intelligence.action_schema.Action`. It is produced by
``intelligence.intent_parser`` and consumed by ``commands.command_router``,
which decides whether the command can be turned directly into an action
(no screen needed) or must be routed through screen capture + the decision
core first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique


@unique
class CommandType(str, Enum):
    """Coarse-grained intent recognized by the deterministic parser."""

    CLICK = "click"
    DRAG = "drag"
    MOVE = "move"
    TYPE = "type"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    OPEN = "open"
    CLOSE = "close"
    BACK = "back"
    FORWARD = "forward"
    WAIT = "wait"
    UNKNOWN = "unknown"


# Command types that never require a screenshot / UI parsing pass.
SCREEN_FREE_TYPES: frozenset[CommandType] = frozenset(
    {
        CommandType.TYPE,
        CommandType.PRESS_KEY,
        CommandType.HOTKEY,
        CommandType.SCROLL,
        CommandType.OPEN,
        CommandType.CLOSE,
        CommandType.BACK,
        CommandType.FORWARD,
        CommandType.WAIT,
    }
)


@dataclass(frozen=True, kw_only=True)
class Command:
    """Structured intent extracted from a raw transcript.

    Only the fields relevant to ``type`` are expected to be populated; the
    rest stay ``None``. ``requires_screen`` is computed by the parser based
    on ``type`` (``CLICK`` with a free-text target always requires the
    screen; everything else typically does not).
    """

    type: CommandType
    raw_text: str
    requires_screen: bool = False

    target_text: str | None = None
    """Free-text description of a UI target, e.g. 'settings' in 'click settings'."""

    source_text: str | None = None
    """Source screen label for DRAG commands."""

    button: str = "left"
    """Mouse button used for CLICK commands."""

    clicks: int = 1
    """Number of clicks: one or two."""

    x: int | None = None
    y: int | None = None
    """Optional desktop coordinates for a direct coordinate click."""

    text: str | None = None
    """Literal text payload for TYPE commands, extracted deterministically."""

    key: str | None = None
    """Single key name for PRESS_KEY commands."""

    keys: tuple[str, ...] | None = None
    """Key combination for HOTKEY commands."""

    direction: str | None = None
    """'up' or 'down' for SCROLL commands."""

    amount: int | None = None
    """Scroll amount, in abstract 'ticks'."""

    app_name: str | None = None
    """Application name for OPEN commands."""

    duration_seconds: float | None = None
    """Wait duration, in seconds, for WAIT commands."""

    def __post_init__(self) -> None:
        # raw_text may legitimately be empty (e.g. an empty transcript classified
        # as UNKNOWN); only its type is enforced here.
        if not isinstance(self.raw_text, str):
            raise ValueError("Command.raw_text must be a string")
