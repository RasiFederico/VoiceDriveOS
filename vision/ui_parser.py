"""Offline screenshot OCR that turns visible text into clickable screen regions."""

from __future__ import annotations

from typing import Any, Protocol

from app.config import VisionConfig
from core.exceptions import UIParsingError
from core.logger import get_logger
from vision.element import UIElement
from vision.screen_capture import Screenshot

logger = get_logger(__name__)


class UIParser(Protocol):
    """Interface every screenshot parser must satisfy."""

    def parse(self, screenshot: Screenshot) -> list[UIElement]: ...


class StaticUIParser:
    """Injectable deterministic parser used by unit tests and integrations."""

    def __init__(self, elements: list[UIElement] | None = None) -> None:
        self._elements = list(elements or [])

    def set_elements(self, elements: list[UIElement]) -> None:
        self._elements = list(elements)

    def parse(self, screenshot: Screenshot) -> list[UIElement]:
        del screenshot
        return list(self._elements)


class RapidOCRUIParser:
    """Extract visible labels and their desktop coordinates using local RapidOCR.

    OCR boxes are useful interaction targets for application and browser text:
    button captions, tabs, menus, links, and fields. The parser operates on a
    captured frame only when a voice command asks for screen context; it does
    not poll or retain screenshots.
    """

    def __init__(self, config: VisionConfig, engine: Any | None = None) -> None:
        self._config = config
        self._engine = engine

    def load(self) -> None:
        """Load the packaged CPU OCR models once; no network access is needed."""
        if self._engine is not None:
            return
        try:
            from rapidocr import RapidOCR

            self._engine = RapidOCR(
                params={
                    "Global.log_level": "error",
                    "Global.text_score": self._config.ocr_min_confidence,
                    "Rec.lang_type": self._config.ocr_language,
                    "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                }
            )
        except Exception as exc:
            raise UIParsingError(f"could not load the offline OCR engine: {exc}") from exc
        logger.info("offline screen OCR ready (language=%s)", self._config.ocr_language)

    def parse(self, screenshot: Screenshot) -> list[UIElement]:
        self.load()
        try:
            result = self._engine(screenshot.image)
        except Exception as exc:
            raise UIParsingError(f"screen OCR failed: {exc}") from exc

        if result is None or result.boxes is None:
            return []

        original_width = screenshot.original_width or round(screenshot.width / screenshot.scale)
        original_height = screenshot.original_height or round(screenshot.height / screenshot.scale)
        scale_x = original_width / screenshot.width
        scale_y = original_height / screenshot.height
        elements: list[UIElement] = []

        for box, text, score in zip(result.boxes, result.txts, result.scores):
            label = str(text).strip()
            confidence = float(score)
            if not label or confidence < self._config.ocr_min_confidence:
                continue

            points = list(box)
            left = min(float(point[0]) for point in points)
            top = min(float(point[1]) for point in points)
            right = max(float(point[0]) for point in points)
            bottom = max(float(point[1]) for point in points)
            x = screenshot.origin_x + round(left * scale_x)
            y = screenshot.origin_y + round(top * scale_y)
            width = max(1, round((right - left) * scale_x))
            height = max(1, round((bottom - top) * scale_y))
            elements.append(
                UIElement(
                    id=len(elements),
                    type="text",
                    text=label,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=min(1.0, max(0.0, confidence)),
                    clickable=True,
                )
            )

        # OCR can return detections in reading order. Respect a hard cap to
        # keep matching deterministic and avoid a dense page overwhelming it.
        elements = elements[: self._config.max_screen_elements]
        logger.info("screen OCR found %d text targets", len(elements))
        return elements

    def close(self) -> None:
        """Drop model references when the application shuts down."""
        self._engine = None


class HybridUIParser:
    """Prefer semantic Windows controls and use OCR to fill gaps.

    UI Automation identifies roles such as buttons, list items and text boxes.
    OCR remains valuable for custom-drawn controls and labels absent from the
    accessibility tree. Overlapping detections are merged into one target.
    """

    def __init__(self, config: VisionConfig, ocr: UIParser, uia: UIParser) -> None:
        self._config = config
        self._ocr = ocr
        self._uia = uia

    def load(self) -> None:
        load = getattr(self._ocr, "load", None)
        if load is not None:
            load()

    @staticmethod
    def _overlap(a: UIElement, b: UIElement) -> float:
        left = max(a.x, b.x)
        top = max(a.y, b.y)
        right = min(a.x + a.width, b.x + b.width)
        bottom = min(a.y + a.height, b.y + b.height)
        area = max(0, right - left) * max(0, bottom - top)
        smaller_area = min(a.width * a.height, b.width * b.height)
        return area / smaller_area if smaller_area else 0.0

    def parse(self, screenshot: Screenshot) -> list[UIElement]:
        self.load()
        accessible = self._uia.parse(screenshot)
        text_regions = self._ocr.parse(screenshot)
        merged = list(accessible)
        for text_region in text_regions:
            overlaps = [item for item in merged if self._overlap(item, text_region) >= 0.45]
            if overlaps:
                best = max(overlaps, key=lambda item: self._overlap(item, text_region))
                if not best.text.strip():
                    merged[merged.index(best)] = UIElement(
                        id=best.id,
                        type=best.type,
                        text=text_region.text,
                        x=best.x,
                        y=best.y,
                        width=best.width,
                        height=best.height,
                        confidence=min(best.confidence, text_region.confidence),
                        clickable=best.clickable,
                    )
                continue
            merged.append(text_region)

        merged.sort(key=lambda item: (item.y, item.x, item.type, item.text.casefold()))
        result = [
            UIElement(
                id=index,
                type=element.type,
                text=element.text,
                x=element.x,
                y=element.y,
                width=element.width,
                height=element.height,
                confidence=element.confidence,
                clickable=element.clickable,
            )
            for index, element in enumerate(merged[: self._config.max_screen_elements])
        ]
        logger.info(
            "hybrid screen recognition: %d UIA controls + %d OCR targets -> %d elements",
            len(accessible),
            len(text_regions),
            len(result),
        )
        return result

    def close(self) -> None:
        for parser in (self._uia, self._ocr):
            close = getattr(parser, "close", None)
            if close is not None:
                close()
