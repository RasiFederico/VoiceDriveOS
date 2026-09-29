"""Italian/English voice-command parser with deterministic, bounded intents."""

from __future__ import annotations

import re

from commands.command_types import Command, CommandType
from commands.text_extractor import extract_text_payload
from core.logger import get_logger

logger = get_logger(__name__)

_KEY_ALIASES: dict[str, str] = {
    "enter": "enter", "invio": "enter", "return": "enter",
    "escape": "escape", "esc": "escape",
    "tab": "tab", "space": "space", "spazio": "space",
    "backspace": "backspace", "delete": "delete", "canc": "delete", "cancella": "delete",
    "insert": "insert", "ins": "insert",
    "up": "up", "su": "up", "sopra": "up",
    "down": "down", "giu": "down", "giù": "down",
    "left": "left", "sinistra": "left", "destra": "right", "right": "right",
    "home": "home", "end": "end", "fine": "end",
    "pageup": "pageup", "pagedown": "pagedown",
    "capslock": "capslock", "numlock": "numlock", "scrolllock": "scrolllock",
}
_KEY_ALIASES.update({f"f{i}": f"f{i}" for i in range(1, 13)})

_MODIFIER_ALIASES: dict[str, str] = {
    "control": "ctrl", "ctrl": "ctrl", "controllo": "ctrl", "controlleft": "ctrl",
    "alt": "alt", "option": "alt",
    "shift": "shift", "maiuscole": "shift",
    "cmd": "cmd", "command": "cmd", "comando": "cmd",
    "win": "win", "windows": "win", "finestra": "win",
}

_RE_KEY_COMMAND = re.compile(
    r"^(?:press|premi|schiaccia)(?:\s+(?:il\s+)?(?:tasto|key))?\s+(.+)$",
    re.IGNORECASE,
)
_RE_OPEN = re.compile(r"^(?:open|apri)\s+(.+)$", re.IGNORECASE)
_RE_CLOSE = re.compile(
    r"^(?:close(?:\s+this\s+window)?|chiudi(?:\s+questa\s+finestra)?)$", re.IGNORECASE
)
_RE_BACK = re.compile(r"^(?:go\s+back|back|indietro|torna\s+indietro)$", re.IGNORECASE)
_RE_FORWARD = re.compile(r"^(?:go\s+forward|forward|avanti)$", re.IGNORECASE)
_RE_SCROLL = re.compile(
    r"^(?:scroll|scorri)\s+(?:(?:up|down|su|gi[uù])(?:\s+(?:by|di)?\s*(\d+)(?:\s*(?:times?|volte|notches|scatti))?)?|(?:by|di)\s*(\d+)(?:\s*(?:times?|volte|notches|scatti))?)$",
    re.IGNORECASE,
)
_RE_WAIT = re.compile(
    r"^(?:wait|aspetta)(?:\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secondi)?)?$", re.IGNORECASE
)
_RE_DRAG = re.compile(
    r"^(?:drag|trascina)\s+(.+?)\s+(?:to|onto|over|su(?:l|lla|i|lle)?|sopra|verso)\s+(.+)$", re.IGNORECASE
)
_RE_MOVE = re.compile(
    r"^(?:move\s+(?:the\s+)?(?:mouse|pointer|cursor)|sposta\s+(?:il\s+)?(?:mouse|puntatore|cursore))\s+"
    r"(?:to|on|onto|su|verso)\s+(.+)$", re.IGNORECASE
)
_RE_CLICK_START = re.compile(
    r"^(?:click|clicca|tap|clic|fai\s+clic|fai\s+click|premi\s+il\s+pulsante|"
    r"premi\s+pulsante)\b\s*(.*)$", re.IGNORECASE
)
_RE_COORDINATES = re.compile(
    r"^(?:(?:at|in\s+position|alle\s+coordinate|in\s+coordinate|coordinate|su)\s+)?"
    r"(?:x\s*)?(\d{1,5})\s*(?:,|;|\s+(?:and|e)\s+|\s+)\s*(?:y\s*)?(\d{1,5})$",
    re.IGNORECASE,
)

_DEFAULT_SCROLL_AMOUNT = 3
_DEFAULT_WAIT_SECONDS = 1.0
_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "save": ("ctrl", "s"), "salva": ("ctrl", "s"),
    "save as": ("ctrl", "shift", "s"), "salva con nome": ("ctrl", "shift", "s"),
    "copy": ("ctrl", "c"), "copia": ("ctrl", "c"),
    "paste": ("ctrl", "v"), "incolla": ("ctrl", "v"),
    "cut": ("ctrl", "x"), "taglia": ("ctrl", "x"),
    "undo": ("ctrl", "z"), "annulla": ("ctrl", "z"),
    "redo": ("ctrl", "y"), "ripeti": ("ctrl", "y"),
    "select all": ("ctrl", "a"), "seleziona tutto": ("ctrl", "a"),
    "new tab": ("ctrl", "t"), "nuova scheda": ("ctrl", "t"),
    "new window": ("ctrl", "n"), "nuova finestra": ("ctrl", "n"),
    "close tab": ("ctrl", "w"), "chiudi scheda": ("ctrl", "w"),
    "reopen tab": ("ctrl", "shift", "t"), "riapri scheda": ("ctrl", "shift", "t"),
    "refresh": ("ctrl", "r"), "refresh page": ("ctrl", "r"), "aggiorna": ("ctrl", "r"),
    "find": ("ctrl", "f"), "find on page": ("ctrl", "f"), "cerca nella pagina": ("ctrl", "f"),
    "print": ("ctrl", "p"), "stampa": ("ctrl", "p"),
    "switch window": ("alt", "tab"), "cambia finestra": ("alt", "tab"),
    "show desktop": ("win", "d"), "mostra desktop": ("win", "d"),
}


def _key_tokens(sequence: str) -> tuple[str, ...] | None:
    """Normalize spoken or plus-separated key names; reject unknown keys."""
    normalized = sequence.casefold().replace("+", " ").replace("più", " ").replace("piu", " ")
    normalized = re.sub(r"\bplus\b", " ", normalized)
    normalized = re.sub(r"\b(?:and|e)\b", " ", normalized)
    normalized = re.sub(r"\bfreccia\s+(su|giu|giù|sinistra|destra|up|down|left|right)\b", r"\1", normalized)
    normalized = re.sub(r"\bpage\s+(up|down)\b", r"page\1", normalized)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    keys: list[str] = []
    for token in tokens:
        key = _MODIFIER_ALIASES.get(token) or _KEY_ALIASES.get(token)
        if key is None and len(token) == 1 and token.isascii() and token.isalnum():
            key = token
        if key is None:
            return None
        keys.append(key)
    if not keys:
        return None
    # Disallow repeated keys and limit combinations to a practical size.
    if len(keys) > 5 or len(set(keys)) != len(keys):
        return None
    return tuple(keys)


def _remove_articles(target: str) -> str:
    return re.sub(r"^(?:the|a|an|il|lo|la|i|gli|le|un|uno|una|sul|sulla|sui|sulle|su|on)\s+", "", target.strip(), flags=re.IGNORECASE)


def _click_mode(transcript: str) -> tuple[str, int, str] | None:
    """Return the requested mouse button/count and command with click-mode prefix removed."""
    text = transcript.strip()
    modes = (
        (r"^(?:right\s+click|click\s+right|click\s+destro|clic\s+destro|clicca\s+con\s+il\s+tasto\s+destro|"
         r"clicca\s+tasto\s+destro|fai\s+clic\s+destro)\b\s*", "right", 1),
        (r"^(?:middle\s+click|click\s+middle|clic\s+centrale|clicca\s+con\s+il\s+tasto\s+centrale)\b\s*", "middle", 1),
        (r"^(?:double\s+click|double-click|click\s+twice|doppio\s+clic(?:ca)?|doppio\s+click|"
         r"clicca\s+due\s+volte|fai\s+doppio\s+clic)\b\s*", "left", 2),
    )
    for pattern, button, clicks in modes:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return button, clicks, text[match.end():].strip()
    return "left", 1, text


def parse_intent(transcript: str) -> Command:
    """Map a transcript to a known command without guessing unknown targets."""
    text = transcript.strip()
    if not text:
        return Command(type=CommandType.UNKNOWN, raw_text=transcript or "")

    text_payload = extract_text_payload(text)
    if text_payload is not None:
        return Command(type=CommandType.TYPE, raw_text=text, text=text_payload)

    lowered = text.casefold()
    shortcut = _SHORTCUTS.get(re.sub(r"[.!?]+$", "", lowered).strip())
    if shortcut:
        return Command(type=CommandType.HOTKEY, raw_text=text, keys=shortcut)
    key_match = _RE_KEY_COMMAND.match(text)
    if key_match:
        keys = _key_tokens(key_match.group(1))
        if keys is not None:
            if len(keys) == 1:
                return Command(type=CommandType.PRESS_KEY, raw_text=text, key=keys[0])
            return Command(type=CommandType.HOTKEY, raw_text=text, keys=keys)

    open_match = _RE_OPEN.match(text)
    if open_match:
        return Command(type=CommandType.OPEN, raw_text=text, app_name=open_match.group(1).strip())

    if _RE_CLOSE.match(lowered):
        return Command(type=CommandType.CLOSE, raw_text=text)
    if _RE_BACK.match(lowered):
        return Command(type=CommandType.BACK, raw_text=text)
    if _RE_FORWARD.match(lowered):
        return Command(type=CommandType.FORWARD, raw_text=text)

    scroll_match = _RE_SCROLL.match(lowered)
    if scroll_match:
        direction_match = re.search(r"\b(up|down|su|gi[uù])\b", lowered)
        direction = "up" if direction_match and direction_match.group(1) in {"up", "su"} else "down"
        amount = int(scroll_match.group(1) or scroll_match.group(2) or _DEFAULT_SCROLL_AMOUNT)
        return Command(
            type=CommandType.SCROLL,
            raw_text=text,
            direction=direction,
            amount=max(1, min(100, amount)),
        )

    wait_match = _RE_WAIT.match(lowered)
    if wait_match:
        seconds = float(wait_match.group(1)) if wait_match.group(1) else _DEFAULT_WAIT_SECONDS
        return Command(type=CommandType.WAIT, raw_text=text, duration_seconds=seconds)

    drag_match = _RE_DRAG.match(text)
    if drag_match:
        return Command(
            type=CommandType.DRAG,
            raw_text=text,
            source_text=_remove_articles(drag_match.group(1)),
            target_text=_remove_articles(drag_match.group(2)),
            requires_screen=True,
        )

    move_match = _RE_MOVE.match(text)
    if move_match:
        return Command(
            type=CommandType.MOVE,
            raw_text=text,
            target_text=_remove_articles(move_match.group(1)),
            requires_screen=True,
        )

    button, clicks, click_text = _click_mode(text)
    click_match = _RE_CLICK_START.match(click_text)
    if click_match or (button != "left" or clicks != 1):
        target = _remove_articles(click_match.group(1) if click_match else click_text)
        coordinates = _RE_COORDINATES.match(target)
        if coordinates:
            return Command(
                type=CommandType.CLICK,
                raw_text=text,
                x=int(coordinates.group(1)),
                y=int(coordinates.group(2)),
                button=button,
                clicks=clicks,
            )
        if target:
            return Command(
                type=CommandType.CLICK,
                raw_text=text,
                target_text=target,
                button=button,
                clicks=clicks,
                requires_screen=True,
            )

    logger.warning("intent_unrecognized transcript=%r", text)
    return Command(type=CommandType.UNKNOWN, raw_text=text)
