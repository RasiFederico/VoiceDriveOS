"""Latency profiling utilities based on ``time.monotonic()``.

``time.time()`` is never used for duration measurement because it is subject
to NTP adjustments and clock jumps. All durations in VoiceDriveOS are
measured with the monotonic clock as required by the project brief.

Typical usage::

    profiler = LatencyProfiler()
    profiler.mark("speech_start")
    ...
    profiler.mark("speech_end")
    ...
    profiler.mark("transcription_start")
    ...
    profiler.mark("transcription_end")

    report = profiler.report()
    print(report.total_ms)
    print(report.durations_ms)  # {"speech_end -> transcription_start": ...}

Or, for a single named stage::

    with profiler.measure("vision"):
        elements = ui_parser.parse(screen)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from core.logger import get_logger

logger = get_logger(__name__)

# Canonical checkpoint names referenced throughout the pipeline.
SPEECH_START = "speech_start"
SPEECH_END = "speech_end"
TRANSCRIPTION_START = "transcription_start"
TRANSCRIPTION_END = "transcription_end"
VISION_START = "vision_start"
VISION_END = "vision_end"
DECISION_START = "decision_start"
DECISION_END = "decision_end"
VALIDATION_START = "validation_start"
VALIDATION_END = "validation_end"
EXECUTION_START = "execution_start"
EXECUTION_END = "execution_end"

_STAGE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("stt", TRANSCRIPTION_START, TRANSCRIPTION_END),
    ("vision", VISION_START, VISION_END),
    ("decision", DECISION_START, DECISION_END),
    ("validation", VALIDATION_START, VALIDATION_END),
    ("execution", EXECUTION_START, EXECUTION_END),
)


@dataclass(frozen=True)
class LatencyReport:
    """Immutable snapshot of measured stage latencies, in milliseconds."""

    checkpoints: dict[str, float]
    stage_latencies_ms: dict[str, float]
    total_ms: float | None


@dataclass
class LatencyProfiler:
    """Records monotonic timestamps for named checkpoints in one pipeline run.

    A new :class:`LatencyProfiler` instance should be created for each
    voice-command cycle (it is cheap: a dict and a lock).
    """

    _checkpoints: dict[str, float] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark(self, name: str, at: float | None = None) -> float:
        """Record a monotonic time under ``name`` and return it.

        Args:
            name: Checkpoint name (see the module-level constants).
            at: An explicit ``time.monotonic()`` value to record, for
                retroactively attaching a timestamp captured on another
                thread (e.g. the VAD's speech-start/end times). Defaults to
                "now".
        """
        now = at if at is not None else time.monotonic()
        with self._lock:
            if name not in self._checkpoints:
                self._order.append(name)
            self._checkpoints[name] = now
        logger.debug("checkpoint name=%s t=%.6f", name, now)
        return now

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        """Context manager that marks ``{stage_name}_start``/``_end``."""
        self.mark(f"{stage_name}_start")
        try:
            yield
        finally:
            self.mark(f"{stage_name}_end")

    def elapsed_ms(self, start_name: str, end_name: str) -> float | None:
        """Return the elapsed milliseconds between two checkpoints, if both exist."""
        with self._lock:
            start = self._checkpoints.get(start_name)
            end = self._checkpoints.get(end_name)
        if start is None or end is None:
            return None
        return (end - start) * 1000.0

    def report(self) -> LatencyReport:
        """Build a :class:`LatencyReport` summarizing all known stage latencies."""
        with self._lock:
            checkpoints = dict(self._checkpoints)
            order = list(self._order)

        stage_latencies: dict[str, float] = {}
        for label, start_name, end_name in _STAGE_PAIRS:
            start = checkpoints.get(start_name)
            end = checkpoints.get(end_name)
            if start is not None and end is not None:
                stage_latencies[label] = (end - start) * 1000.0

        total_ms: float | None = None
        if len(order) >= 2:
            first_ts = checkpoints[order[0]]
            last_ts = checkpoints[order[-1]]
            total_ms = (last_ts - first_ts) * 1000.0

        return LatencyReport(
            checkpoints=checkpoints,
            stage_latencies_ms=stage_latencies,
            total_ms=total_ms,
        )

    def reset(self) -> None:
        """Clear all recorded checkpoints, allowing profiler reuse."""
        with self._lock:
            self._checkpoints.clear()
            self._order.clear()
