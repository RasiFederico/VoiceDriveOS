"""Non-blocking microphone capture.

Audio capture runs entirely on a background thread driven by the
``sounddevice`` (PortAudio) callback API: PortAudio calls back into
:meth:`MicrophoneWorker._on_audio` on its own internal thread whenever a new
chunk is ready, and that callback does nothing but push raw PCM bytes into a
``queue.Queue``. Nothing here ever touches the event loop or blocks on I/O
inside the callback, so audio capture cannot stall the rest of the pipeline.

``sounddevice`` is imported lazily so this module remains importable (and its
queueing logic testable) on systems without PortAudio installed/available.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

from app.config import AudioConfig
from core.exceptions import MicrophoneError
from core.logger import get_logger

logger = get_logger(__name__)

# Bytes per audio sample for 16-bit PCM, VoiceDriveOS's wire format throughout
# the audio pipeline (VAD and faster-whisper both consume/produce this).
BYTES_PER_SAMPLE = 2


def _load_sounddevice():
    try:
        import sounddevice  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise MicrophoneError(
            "sounddevice is not installed. Run `pip install sounddevice` "
            "(requires PortAudio) to enable real microphone capture."
        ) from exc
    return sounddevice


class MicrophoneWorker:
    """Captures 16-bit PCM audio chunks from the default (or configured) input device.

    Usage::

        mic = MicrophoneWorker(config.audio)
        mic.start()
        try:
            chunk = mic.read_chunk(timeout=1.0)  # bytes | None
        finally:
            mic.stop()

    Or as a context manager::

        with MicrophoneWorker(config.audio) as mic:
            ...
    """

    def __init__(
        self,
        config: AudioConfig,
        on_error: Callable[[Exception], None] | None = None,
        max_queue_size: int = 256,
    ) -> None:
        self._config = config
        self._on_error = on_error
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max_queue_size)
        self._stream = None
        self._running = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def _on_audio(self, indata, frames: int, time_info, status) -> None:  # noqa: ANN001 - PortAudio callback signature
        if status:
            logger.warning("microphone status flag: %s", status)
        try:
            chunk = bytes(indata)
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest chunk rather than blocking the audio callback,
            # which must return immediately or PortAudio will glitch/underrun.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(bytes(indata))
            except queue.Empty:  # pragma: no cover - race, harmless
                pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("microphone callback error: %s", exc)
            if self._on_error is not None:
                self._on_error(exc)

    def start(self) -> None:
        """Open the input stream and begin capturing in the background."""
        if self.is_running:
            return
        sounddevice = _load_sounddevice()
        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="int16",
                blocksize=self._config.chunk_size,
                device=self._config.input_device,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            raise MicrophoneError(f"failed to open microphone stream: {exc}") from exc
        self._running.set()
        logger.info(
            "microphone started sample_rate=%d channels=%d chunk_size=%d",
            self._config.sample_rate,
            self._config.channels,
            self._config.chunk_size,
        )

    def stop(self) -> None:
        """Stop and close the input stream. Safe to call multiple times."""
        if not self.is_running:
            return
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover - depends on backend
                logger.error("error closing microphone stream: %s", exc)
            finally:
                self._stream = None
        logger.info("microphone stopped")

    def read_chunk(self, timeout: float | None = 0.1) -> bytes | None:
        """Return the next available audio chunk, or ``None`` on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard any buffered audio (e.g. after finishing a command)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def __enter__(self) -> "MicrophoneWorker":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
