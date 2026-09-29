"""Best-effort Windows UI Automation parser.

UI Automation exposes control names, roles and desktop bounds for accessible
application controls. It is queried for the foreground window only, then the
OCR parser fills in any text-only targets that the accessibility provider did
not expose.
"""

from __future__ import annotations

import platform
from typing import Any

from app.config import VisionConfig
from core.logger import get_logger
from vision.element import UIElement
from vision.screen_capture import Screenshot

logger = get_logger(__name__)

_INTERACTIVE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "edit",
    "hyperlink",
    "listitem",
    "menuitem",
    "radiobutton",
    "splitbutton",
    "tabitem",
    "treeitem",
    "datagrid",
    "dataitem",
    "document",
    "headeritem",
    "slider",
    "spinner",
    "tab",
    "textbox",
    "toolbar",
    "tree",
}


class UIAutomationParser:
    """Read clickable controls from the foreground Windows application."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._warned_unavailable = False

    def parse(self, screenshot: Screenshot) -> list[UIElement]:
        del screenshot  # UIA gives coordinates in desktop pixels, not image pixels.
        if platform.system() != "Windows":
            return []
        try:
            from pywinauto import Desktop
        except ImportError:
            if not self._warned_unavailable:
                logger.warning("pywinauto is unavailable; using OCR-only screen recognition")
                self._warned_unavailable = True
            return []

        try:
            desktop = Desktop(backend="uia")
            window = self._foreground_window(desktop)
            if window is None:
                return []
            wrappers = [window, *window.descendants()]
        except Exception as exc:
            logger.warning("Windows UI Automation could not inspect the foreground window: %s", exc)
            return []

        elements: list[UIElement] = []
        seen: set[tuple[int, int, int, int, str, str]] = set()
        for wrapper in wrappers:
            if len(elements) >= self._config.max_screen_elements:
                break
            try:
                info: Any = wrapper.element_info
                role = str(getattr(info, "control_type", "") or "Control")
                role_key = role.casefold().replace(" ", "")
                if role_key not in _INTERACTIVE_ROLES:
                    continue
                if not wrapper.is_visible() or not wrapper.is_enabled():
                    continue
                rectangle = wrapper.rectangle()
                left, top, right, bottom = (
                    int(rectangle.left), int(rectangle.top), int(rectangle.right), int(rectangle.bottom)
                )
                width, height = right - left, bottom - top
                if width <= 1 or height <= 1:
                    continue
                # Ignore minimized/off-screen controls and giant container panes.
                if right <= 0 or bottom <= 0 or left > 100000 or top > 100000:
                    continue
                if width * height > 0.92 * 100000000:
                    continue
                name = str(getattr(info, "name", "") or "").strip()
                automation_id = str(getattr(info, "automation_id", "") or "").strip()
                help_text = str(getattr(info, "help_text", "") or "").strip()
                try:
                    item_type = str(
                        getattr(getattr(info, "element", None), "CurrentItemType", "") or ""
                    ).strip()
                except Exception:
                    item_type = ""
                text = name or help_text
                # Unnamed controls are still useful candidates for a visual model,
                # but do not add generic top-level window chrome.
                if not text and role_key in {"window", "pane", "toolbar"}:
                    continue
                signature = (left, top, width, height, role_key, text.casefold())
                if signature in seen:
                    continue
                seen.add(signature)
                item_kind = " ".join(item_type.casefold().split())
                element_type = f"uia:{role_key} [{item_kind}]" if item_kind else f"uia:{role_key}"
                elements.append(
                    UIElement(
                        id=len(elements),
                        type=element_type,
                        text=text,
                        x=left,
                        y=top,
                        width=width,
                        height=height,
                        confidence=0.95,
                        clickable=True,
                    )
                )
            except Exception:
                # Individual controls can disappear while a window is updating.
                continue

        elements.sort(key=lambda item: (item.y, item.x, item.type, item.text.casefold()))
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
            for index, element in enumerate(elements[: self._config.max_screen_elements])
        ]
        logger.info("Windows UI Automation found %d interactive controls", len(result))
        return result

    @staticmethod
    def _foreground_window(desktop: Any) -> Any | None:
        """Resolve the current foreground HWND, with a pywinauto fallback."""
        try:
            import win32gui

            handle = int(win32gui.GetForegroundWindow())
            if handle:
                return desktop.window(handle=handle).wrapper_object()
        except Exception:
            pass
        try:
            windows = desktop.windows(top_level_only=True, visible_only=True)
            active = [window for window in windows if window.is_active()]
            return active[0] if active else (windows[0] if windows else None)
        except Exception:
            return None
