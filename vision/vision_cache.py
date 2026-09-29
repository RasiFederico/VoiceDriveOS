"""Caches the most recent UI element list to avoid redundant CV inference.

Per the project brief's caching strategy (section 24): if the screen has not
plausibly changed since the last parse, reuse the cached element list instead
of capturing + parsing again. Every executed action invalidates the cache,
since the screen is assumed to have changed as a result.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from app.config import VisionConfig
from core.logger import get_logger
from vision.element import UIElement

logger = get_logger(__name__)


@dataclass(frozen=True)
class _CacheEntry:
    elements: tuple[UIElement, ...]
    captured_at: float = field(default_factory=time.monotonic)


class VisionCache:
    """Thread-safe single-slot cache with a configurable TTL."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._entry: _CacheEntry | None = None
        self._lock = threading.Lock()

    def get(self) -> list[UIElement] | None:
        """Return the cached elements if present and not expired, else ``None``."""
        with self._lock:
            entry = self._entry
            if entry is None:
                return None
            age = time.monotonic() - entry.captured_at
            if age > self._config.cache_lifetime_s:
                logger.debug("vision cache expired (age=%.2fs > ttl=%.2fs)", age, self._config.cache_lifetime_s)
                self._entry = None
                return None
            return list(entry.elements)

    def set(self, elements: list[UIElement]) -> None:
        """Store a fresh element list, timestamped now."""
        with self._lock:
            self._entry = _CacheEntry(elements=tuple(elements))
        logger.debug("vision cache updated: %d elements", len(elements))

    def invalidate(self) -> None:
        """Force the next :meth:`get` to miss (call after every executed action)."""
        with self._lock:
            self._entry = None
        logger.debug("vision cache invalidated")

    @property
    def is_valid(self) -> bool:
        return self.get() is not None
