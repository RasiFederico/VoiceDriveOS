"""Explicit, thread-safe state machine for the VoiceDriveOS pipeline.

Only one worker (the pipeline orchestrator in ``app/main.py``) is expected to
drive transitions, but the machine is guarded by a lock regardless so that
diagnostic/UI code can safely read ``current_state`` from another thread, and
so a stray call from a worker thread cannot corrupt state.

Legal transitions are enforced: attempting to jump from ``IDLE`` straight to
``EXECUTING`` (skipping decision/validation) raises
:class:`core.exceptions.StateTransitionError`. This turns pipeline bugs into
loud failures instead of silent, potentially unsafe behaviour.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable

from core.exceptions import StateTransitionError
from core.logger import get_logger

logger = get_logger(__name__)


@unique
class PipelineState(Enum):
    """Every state the voice-to-action pipeline can be in."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING_SCREEN = "ANALYZING_SCREEN"
    DECIDING = "DECIDING"
    VALIDATING = "VALIDATING"
    EXECUTING = "EXECUTING"
    ERROR = "ERROR"


# Adjacency list of legal transitions. ERROR and IDLE are reachable from any
# state (fail-safe reset / abort), which keeps error handling simple.
_LEGAL_TRANSITIONS: dict[PipelineState, frozenset[PipelineState]] = {
    PipelineState.IDLE: frozenset({PipelineState.LISTENING}),
    PipelineState.LISTENING: frozenset({PipelineState.TRANSCRIBING, PipelineState.ERROR, PipelineState.IDLE}),
    PipelineState.TRANSCRIBING: frozenset(
        {PipelineState.ANALYZING_SCREEN, PipelineState.DECIDING, PipelineState.ERROR, PipelineState.IDLE}
    ),
    PipelineState.ANALYZING_SCREEN: frozenset({PipelineState.DECIDING, PipelineState.ERROR, PipelineState.IDLE}),
    PipelineState.DECIDING: frozenset({PipelineState.VALIDATING, PipelineState.ERROR, PipelineState.IDLE}),
    PipelineState.VALIDATING: frozenset({PipelineState.EXECUTING, PipelineState.ERROR, PipelineState.IDLE}),
    PipelineState.EXECUTING: frozenset({PipelineState.IDLE, PipelineState.ERROR}),
    PipelineState.ERROR: frozenset({PipelineState.IDLE}),
}

StateChangeListener = Callable[[PipelineState, PipelineState], None]


@dataclass(frozen=True)
class StateTransition:
    """Record of a single transition, useful for logging/debugging."""

    previous: PipelineState
    current: PipelineState


class StateMachine:
    """Thread-safe explicit state machine guarding pipeline transitions."""

    def __init__(self, initial: PipelineState = PipelineState.IDLE) -> None:
        self._state = initial
        self._lock = threading.RLock()
        self._listeners: list[StateChangeListener] = []

    @property
    def current_state(self) -> PipelineState:
        with self._lock:
            return self._state

    def can_transition(self, target: PipelineState) -> bool:
        with self._lock:
            return target in _LEGAL_TRANSITIONS.get(self._state, frozenset())

    def transition(self, target: PipelineState) -> StateTransition:
        """Move to ``target`` if legal, otherwise raise ``StateTransitionError``."""
        with self._lock:
            previous = self._state
            if target not in _LEGAL_TRANSITIONS.get(previous, frozenset()):
                raise StateTransitionError(f"Illegal transition: {previous.value} -> {target.value}")
            self._state = target
            listeners = list(self._listeners)

        logger.info("STATE=%s (from %s)", target.value, previous.value)
        for listener in listeners:
            listener(previous, target)
        return StateTransition(previous=previous, current=target)

    def force_error(self, reason: str) -> StateTransition:
        """Unconditionally move to ``ERROR`` from any state (used on exceptions)."""
        with self._lock:
            previous = self._state
            self._state = PipelineState.ERROR
            listeners = list(self._listeners)
        logger.error("STATE=ERROR (from %s) reason=%s", previous.value, reason)
        for listener in listeners:
            listener(previous, PipelineState.ERROR)
        return StateTransition(previous=previous, current=PipelineState.ERROR)

    def reset(self) -> StateTransition:
        """Return to ``IDLE`` from ``ERROR`` (or any state, for recovery)."""
        with self._lock:
            previous = self._state
            self._state = PipelineState.IDLE
            listeners = list(self._listeners)
        logger.info("STATE=IDLE (reset from %s)", previous.value)
        for listener in listeners:
            listener(previous, PipelineState.IDLE)
        return StateTransition(previous=previous, current=PipelineState.IDLE)

    def add_listener(self, listener: StateChangeListener) -> None:
        """Register a callback invoked as ``listener(previous, current)`` on every transition."""
        with self._lock:
            self._listeners.append(listener)
