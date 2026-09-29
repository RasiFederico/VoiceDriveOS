"""Strongly typed, validated action schema.

This module is the single choke point between "what the AI decided" and
"what the executor is allowed to do". A dict coming from the decision core
(Laya, the mock model, or the deterministic parser) MUST be converted to an
:class:`Action` subclass via :func:`action_from_dict` before it can reach
``execution.executor.Executor``.

Design guarantees:

* Unknown action names are rejected.
* Missing/extra/mistyped fields are rejected.
* Out-of-range values (negative ids, empty text, impossible coordinates) are
  rejected.
* There is no code path from a dict to arbitrary Python execution: every
  field is a plain, validated primitive (int, float, str, tuple of str).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, unique
from typing import Any, ClassVar, Mapping

from core.exceptions import ActionSchemaError

MAX_TEXT_LENGTH = 10_000
MAX_WAIT_SECONDS = 30.0
MAX_SCROLL_AMOUNT = 100

# Canonical set of key names accepted by ``PRESS_KEY``/``HOTKEY`` actions.
# Kept here (rather than in execution.keyboard) so schema validation never
# depends on the PyAutoGUI-backed execution layer.
ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "enter", "return", "escape", "esc", "tab", "space", "backspace", "delete",
        "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
        "insert", "capslock", "numlock", "scrolllock",
        "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
        "ctrl", "alt", "shift", "win", "cmd", "command", "option",
        *"abcdefghijklmnopqrstuvwxyz",
        *"0123456789",
    }
)


@unique
class ActionType(str, Enum):
    """Closed set of actions the decision core is allowed to request."""

    CLICK = "click"
    TYPE = "type"
    PRESS_KEY = "keypress"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    DRAG = "drag"
    MOVE = "move"
    OPEN = "open"
    CLOSE = "close"
    BACK = "back"
    FORWARD = "forward"
    WAIT = "wait"
    NO_ACTION = "no_action"


def _validate_key(key: str) -> str:
    normalized = key.strip().lower()
    if not normalized:
        raise ActionSchemaError("key must be a non-empty string")
    if normalized not in ALLOWED_KEYS:
        raise ActionSchemaError(f"unsupported key: {key!r}")
    return normalized


@dataclass(frozen=True, kw_only=True)
class Action:
    """Base class for every validated, executable action."""

    action: ActionType
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise ActionSchemaError("confidence must be numeric")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ActionSchemaError("confidence must be within [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dict (e.g. for logging)."""
        data = asdict(self)
        data["action"] = self.action.value
        return data


@dataclass(frozen=True, kw_only=True)
class ClickAction(Action):
    action: ActionType = field(default=ActionType.CLICK, init=False)
    target_id: int | None = None
    x: int | None = None
    y: int | None = None
    button: str = "left"
    clicks: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        by_id = self.target_id is not None
        by_coordinates = self.x is not None and self.y is not None
        if not (by_id or by_coordinates):
            raise ActionSchemaError("click requires target_id or both x and y coordinates")
        if by_id and (not isinstance(self.target_id, int) or isinstance(self.target_id, bool) or self.target_id < 0):
            raise ActionSchemaError("target_id must be a non-negative integer")
        for name in ("x", "y"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ActionSchemaError(f"{name} must be a non-negative integer")
        if not isinstance(self.button, str):
            raise ActionSchemaError("button must be a string")
        button = self.button.strip().lower()
        if button not in {"left", "right", "middle"}:
            raise ActionSchemaError("button must be 'left', 'right', or 'middle'")
        object.__setattr__(self, "button", button)
        if not isinstance(self.clicks, int) or isinstance(self.clicks, bool) or self.clicks not in (1, 2):
            raise ActionSchemaError("clicks must be 1 or 2")


@dataclass(frozen=True, kw_only=True)
class TypeAction(Action):
    action: ActionType = field(default=ActionType.TYPE, init=False)
    text: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.text, str) or not self.text:
            raise ActionSchemaError("text must be a non-empty string")
        if len(self.text) > MAX_TEXT_LENGTH:
            raise ActionSchemaError(f"text exceeds maximum length of {MAX_TEXT_LENGTH}")


@dataclass(frozen=True, kw_only=True)
class PressKeyAction(Action):
    action: ActionType = field(default=ActionType.PRESS_KEY, init=False)
    key: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "key", _validate_key(self.key))


@dataclass(frozen=True, kw_only=True)
class HotkeyAction(Action):
    action: ActionType = field(default=ActionType.HOTKEY, init=False)
    keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.keys, (list, tuple)) or len(self.keys) < 2:
            raise ActionSchemaError("hotkey requires at least two keys")
        validated = tuple(_validate_key(k) for k in self.keys)
        object.__setattr__(self, "keys", validated)


@dataclass(frozen=True, kw_only=True)
class ScrollAction(Action):
    action: ActionType = field(default=ActionType.SCROLL, init=False)
    direction: str = "down"
    amount: int = 1

    _VALID_DIRECTIONS: ClassVar[frozenset[str]] = frozenset({"up", "down"})

    def __post_init__(self) -> None:
        super().__post_init__()
        normalized = str(self.direction).strip().lower()
        if normalized not in self._VALID_DIRECTIONS:
            raise ActionSchemaError("direction must be 'up' or 'down'")
        object.__setattr__(self, "direction", normalized)
        if not isinstance(self.amount, int) or isinstance(self.amount, bool) or not (0 < self.amount <= MAX_SCROLL_AMOUNT):
            raise ActionSchemaError(f"amount must be an integer in (0, {MAX_SCROLL_AMOUNT}]")


@dataclass(frozen=True, kw_only=True)
class DragAction(Action):
    """Drag from a source to a destination, addressed either by UI element id or raw coordinates."""

    action: ActionType = field(default=ActionType.DRAG, init=False)
    from_target_id: int | None = None
    to_target_id: int | None = None
    from_x: int | None = None
    from_y: int | None = None
    to_x: int | None = None
    to_y: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        by_id = self.from_target_id is not None and self.to_target_id is not None
        by_coords = None not in (self.from_x, self.from_y, self.to_x, self.to_y)
        if not (by_id or by_coords):
            raise ActionSchemaError(
                "drag requires either (from_target_id, to_target_id) or all of "
                "(from_x, from_y, to_x, to_y)"
            )
        for name in ("from_target_id", "to_target_id", "from_x", "from_y", "to_x", "to_y"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ActionSchemaError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, kw_only=True)
class MoveAction(Action):
    action: ActionType = field(default=ActionType.MOVE, init=False)
    target_id: int | None = None
    x: int | None = None
    y: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        by_id = self.target_id is not None
        by_coords = self.x is not None and self.y is not None
        if not (by_id or by_coords):
            raise ActionSchemaError("move requires either target_id or (x, y)")
        for name in ("target_id", "x", "y"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ActionSchemaError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, kw_only=True)
class OpenAction(Action):
    action: ActionType = field(default=ActionType.OPEN, init=False)
    app_name: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.app_name, str) or not self.app_name.strip():
            raise ActionSchemaError("app_name must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class CloseAction(Action):
    action: ActionType = field(default=ActionType.CLOSE, init=False)


@dataclass(frozen=True, kw_only=True)
class BackAction(Action):
    action: ActionType = field(default=ActionType.BACK, init=False)


@dataclass(frozen=True, kw_only=True)
class ForwardAction(Action):
    action: ActionType = field(default=ActionType.FORWARD, init=False)


@dataclass(frozen=True, kw_only=True)
class WaitAction(Action):
    action: ActionType = field(default=ActionType.WAIT, init=False)
    seconds: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.seconds, (int, float)) or isinstance(self.seconds, bool):
            raise ActionSchemaError("seconds must be numeric")
        if not (0.0 < float(self.seconds) <= MAX_WAIT_SECONDS):
            raise ActionSchemaError(f"seconds must be in (0, {MAX_WAIT_SECONDS}]")


@dataclass(frozen=True, kw_only=True)
class NoAction(Action):
    """Explicit "do nothing" action, returned when confidence is too low or no match was found."""

    action: ActionType = field(default=ActionType.NO_ACTION, init=False)
    reason: str = ""


_ACTION_TYPE_TO_CLASS: dict[ActionType, type[Action]] = {
    ActionType.CLICK: ClickAction,
    ActionType.TYPE: TypeAction,
    ActionType.PRESS_KEY: PressKeyAction,
    ActionType.HOTKEY: HotkeyAction,
    ActionType.SCROLL: ScrollAction,
    ActionType.DRAG: DragAction,
    ActionType.MOVE: MoveAction,
    ActionType.OPEN: OpenAction,
    ActionType.CLOSE: CloseAction,
    ActionType.BACK: BackAction,
    ActionType.FORWARD: ForwardAction,
    ActionType.WAIT: WaitAction,
    ActionType.NO_ACTION: NoAction,
}


def action_from_dict(data: Mapping[str, Any]) -> Action:
    """Convert a raw dict (e.g. decoded JSON from a decision model) into a validated :class:`Action`.

    Raises:
        ActionSchemaError: if the action name is unknown, or required fields
            are missing/extra/out of range.
    """
    if not isinstance(data, Mapping):
        raise ActionSchemaError("action payload must be a mapping")

    raw_action = data.get("action")
    if raw_action is None:
        raise ActionSchemaError("missing required field: 'action'")

    try:
        action_type = ActionType(str(raw_action).strip().lower())
    except ValueError as exc:
        raise ActionSchemaError(f"unknown action type: {raw_action!r}") from exc

    action_cls = _ACTION_TYPE_TO_CLASS[action_type]
    kwargs = {k: v for k, v in data.items() if k != "action"}

    # tuple/list normalization for hotkey "keys"
    if action_type is ActionType.HOTKEY and "keys" in kwargs and isinstance(kwargs["keys"], list):
        kwargs["keys"] = tuple(kwargs["keys"])

    try:
        return action_cls(**kwargs)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ActionSchemaError(
            f"invalid fields for action '{action_type.value}': {exc}"
        ) from exc
