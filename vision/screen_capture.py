"""On-demand screen capture.

Per the project brief's optimization rules, the screen is **never** polled
continuously at full resolution: :meth:`ScreenCapture.capture` is called only
when a command actually needs screen context, and the result is downscaled
according to ``VisionConfig.capture_scale`` before any CV runs on it.

Uses ``mss`` (a minimal, dependency-light, cross-platform screenshot library)
as the backend, imported lazily. Pixel data is exposed as a ``PIL.Image``
since ``Pillow`` is a lightweight, near-universal dependency and downstream
CV/OCR backends generally accept it directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.config import VisionConfig
from core.exceptions import ScreenCaptureError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Screenshot:
    """A single captured (and possibly downscaled) frame."""

    image: Any  # PIL.Image.Image — typed as Any to avoid a hard Pillow import at type-check time
    width: int
    height: int
    scale: float
    monitor_index: int
    captured_at: float
    """``time.monotonic()`` timestamp of capture, for cache-lifetime comparisons."""
    origin_x: int = 0
    origin_y: int = 0
    original_width: int | None = None
    original_height: int | None = None


def _load_mss() -> Any:
    try:
        import mss  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise ScreenCaptureError(
            "mss is not installed. Run `pip install mss` to enable real screen capture."
        ) from exc
    return mss


class ScreenCapture:
    """Captures and downscales a single screenshot on demand."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._sct: Any = None

    def _backend(self) -> Any:
        if self._sct is None:
            mss = _load_mss()
            try:
                self._sct = mss.mss()
            except Exception as exc:
                raise ScreenCaptureError(f"failed to initialize screen capture backend: {exc}") from exc
        return self._sct

    def _select_region(self, sct: Any) -> dict[str, int]:
        if self._config.region_of_interest is not None:
            left, top, width, height = self._config.region_of_interest
            return {"left": left, "top": top, "width": width, "height": height}

        monitors = sct.monitors
        # monitors[0] is the virtual "all monitors" bounding box in mss;
        # real monitors start at index 1.
        index = self._config.monitor_index + 1
        if index >= len(monitors):
            logger.warning(
                "configured monitor_index=%d out of range (%d monitors detected); using primary",
                self._config.monitor_index,
                len(monitors) - 1,
            )
            index = 1
        return monitors[index]

    def capture(self) -> Screenshot:
        """Capture, convert, and downscale the current screen (or region of interest)."""
        sct = self._backend()
        try:
            region = self._select_region(sct)
            raw = sct.grab(region)
        except ScreenCaptureError:
            raise
        except Exception as exc:
            raise ScreenCaptureError(f"failed to capture screen: {exc}") from exc

        try:
            from PIL import Image

            image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        except Exception as exc:
            raise ScreenCaptureError(f"failed to decode captured frame: {exc}") from exc

        scale = self._config.capture_scale
        if 0.0 < scale < 1.0:
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size)

        screenshot = Screenshot(
            image=image,
            width=image.width,
            height=image.height,
            scale=scale,
            monitor_index=self._config.monitor_index,
            captured_at=time.monotonic(),
            origin_x=int(region["left"]),
            origin_y=int(region["top"]),
            original_width=int(raw.size[0]),
            original_height=int(raw.size[1]),
        )
        logger.debug("screen captured %dx%d scale=%.2f", screenshot.width, screenshot.height, scale)
        return screenshot

    def close(self) -> None:
        """Release the backend's resources (e.g. on shutdown)."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self._sct = None
