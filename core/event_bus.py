"""Minimal event bus connecting worker threads to the asyncio pipeline.

VoiceDriveOS runs blocking I/O (microphone capture, PyAutoGUI execution) on
dedicated threads, while the decision pipeline is orchestrated with
``asyncio`` in ``app/main.py``. :class:`EventBus` is the single, deliberately
small primitive used to move events between the two worlds:

* A background thread (e.g. the microphone worker) calls
  :meth:`EventBus.publish_threadsafe`, which is safe to call from any thread.
* The asyncio pipeline calls :meth:`EventBus.subscribe` (an async generator)
  or :meth:`EventBus.get` to consume events on a given topic.

Topics are plain strings (``"audio_chunk"``, ``"command"``, ``"action"``,
...). This is intentionally not a general-purpose framework: one
``asyncio.Queue`` per topic, created lazily.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from core.logger import get_logger

logger = get_logger(__name__)

# Sentinel used internally to unblock subscribers during shutdown.
_STOP = object()


class EventBus:
    """Simple per-topic pub/sub bus built on :class:`asyncio.Queue`.

    Thread-safety: :meth:`publish_threadsafe` is the only method meant to be
    called from a thread other than the one running the bound event loop.
    All other methods (``publish``, ``get``, ``subscribe``) must be awaited
    from within the event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._queues: dict[str, asyncio.Queue[Any]] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the running event loop so worker threads can publish safely."""
        self._loop = loop

    def _queue_for(self, topic: str) -> asyncio.Queue[Any]:
        queue = self._queues.get(topic)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[topic] = queue
        return queue

    async def publish(self, topic: str, event: Any) -> None:
        """Publish ``event`` on ``topic``. Must be awaited from the event loop."""
        await self._queue_for(topic).put(event)

    def publish_threadsafe(self, topic: str, event: Any) -> None:
        """Publish ``event`` on ``topic`` from any thread.

        Requires :meth:`bind_loop` to have been called with the running loop.
        """
        if self._loop is None:
            raise RuntimeError("EventBus.bind_loop() must be called before publish_threadsafe().")
        queue = self._queue_for(topic)
        self._loop.call_soon_threadsafe(queue.put_nowait, event)

    async def get(self, topic: str) -> Any:
        """Await and return the next event published on ``topic``."""
        return await self._queue_for(topic).get()

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        """Async-iterate events published on ``topic`` until :meth:`close` is called."""
        queue = self._queue_for(topic)
        while True:
            event = await queue.get()
            if event is _STOP:
                logger.debug("topic=%s closed", topic)
                return
            yield event

    def close(self, topic: str) -> None:
        """Signal subscribers of ``topic`` to stop iterating.

        Safe to call from any thread once :meth:`bind_loop` has run.
        """
        self.publish_threadsafe(topic, _STOP)

    def close_all(self) -> None:
        """Signal subscribers on every known topic to stop iterating."""
        for topic in list(self._queues.keys()):
            self.close(topic)
