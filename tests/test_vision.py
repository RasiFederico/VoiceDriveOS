"""Tests for vision.element.UIElement and vision.vision_cache.VisionCache."""

from __future__ import annotations

import time
import unittest

from app.config import VisionConfig
from vision.element import UIElement
from vision.screen_capture import Screenshot
from vision.ui_parser import RapidOCRUIParser, StaticUIParser
from vision.vision_cache import VisionCache


def _element(id_: int = 0, x: int = 100, y: int = 200, width: int = 40, height: int = 20) -> UIElement:
    return UIElement(id=id_, type="button", text="Login", x=x, y=y, width=width, height=height)


class UIElementTests(unittest.TestCase):
    def test_center_computation(self) -> None:
        element = _element(x=100, y=200, width=40, height=20)
        self.assertEqual(element.center(), (120, 210))

    def test_contains_point_inside(self) -> None:
        element = _element(x=0, y=0, width=10, height=10)
        self.assertTrue(element.contains(5, 5))

    def test_contains_point_outside(self) -> None:
        element = _element(x=0, y=0, width=10, height=10)
        self.assertFalse(element.contains(50, 50))

    def test_negative_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UIElement(id=-1, type="button", text="x", x=0, y=0, width=1, height=1)

    def test_non_positive_dimensions_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UIElement(id=0, type="button", text="x", x=0, y=0, width=0, height=10)

    def test_confidence_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UIElement(id=0, type="button", text="x", x=0, y=0, width=1, height=1, confidence=1.5)


class VisionCacheTests(unittest.TestCase):
    def test_empty_cache_returns_none(self) -> None:
        cache = VisionCache(VisionConfig(cache_lifetime_s=5.0))
        self.assertIsNone(cache.get())

    def test_set_then_get_returns_elements(self) -> None:
        cache = VisionCache(VisionConfig(cache_lifetime_s=5.0))
        elements = [_element(0), _element(1)]
        cache.set(elements)
        cached = cache.get()
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(len(cached), 2)

    def test_invalidate_clears_cache(self) -> None:
        cache = VisionCache(VisionConfig(cache_lifetime_s=5.0))
        cache.set([_element(0)])
        cache.invalidate()
        self.assertIsNone(cache.get())

    def test_expired_cache_returns_none(self) -> None:
        cache = VisionCache(VisionConfig(cache_lifetime_s=0.01))
        cache.set([_element(0)])
        time.sleep(0.05)
        self.assertIsNone(cache.get())


class StaticUIParserTests(unittest.TestCase):
    def test_returns_configured_elements_regardless_of_screenshot(self) -> None:
        parser = StaticUIParser([_element(0), _element(1)])
        result = parser.parse(screenshot=None)  # type: ignore[arg-type]
        self.assertEqual(len(result), 2)

    def test_defaults_to_empty(self) -> None:
        parser = StaticUIParser()
        self.assertEqual(parser.parse(screenshot=None), [])  # type: ignore[arg-type]


class _FakeOCRResult:
    boxes = [
        [[10, 20], [110, 20], [110, 60], [10, 60]],
        [[130, 20], [200, 20], [200, 60], [130, 60]],
    ]
    txts = ("Save", "noise")
    scores = (0.95, 0.1)


class _FakeOCREngine:
    def __call__(self, _image):
        return _FakeOCRResult()


class RapidOCRUIParserTests(unittest.TestCase):
    def test_maps_scaled_ocr_box_back_to_virtual_desktop_coordinates(self) -> None:
        parser = RapidOCRUIParser(VisionConfig(ocr_min_confidence=0.2), engine=_FakeOCREngine())
        screenshot = Screenshot(
            image=object(),
            width=400,
            height=200,
            scale=0.5,
            monitor_index=0,
            captured_at=0.0,
            origin_x=100,
            origin_y=50,
            original_width=800,
            original_height=400,
        )

        elements = parser.parse(screenshot)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].text, "Save")
        self.assertEqual((elements[0].x, elements[0].y), (120, 90))
        self.assertEqual((elements[0].width, elements[0].height), (200, 80))
        self.assertTrue(elements[0].clickable)


if __name__ == "__main__":
    unittest.main()
