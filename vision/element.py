"""Typed representation of a single interactable UI element.

This is the common currency between the UI parser (whatever backend produces
it — OmniParser, YOLO, or a static test fixture) and the decision core /
executor. Neither of those ever reasons about raw pixels; they only ever see
a list of :class:`UIElement`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UIElement:
    """A single detected, addressable element on screen, in real screen pixels."""

    id: int
    type: str
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    clickable: bool = True

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("UIElement.id must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("UIElement.width and height must be positive")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("UIElement.confidence must be within [0.0, 1.0]")

    def center(self) -> tuple[int, int]:
        """Return the (x, y) pixel coordinates of this element's center."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, x: int, y: int) -> bool:
        """Return True if the point ``(x, y)`` falls within this element's bounding box."""
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def as_candidate_line(self) -> str:
        """One-line human-readable description, used when building the decision-core prompt/context."""
        label = self.text.strip() or f"<{self.type}>"
        return f"{self.id}: {label}"
