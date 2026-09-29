"""Local, bounded decision core for selecting a visible UI target.

Laya is a typed choice/scoring model, not a text-generating LLM. It receives
the spoken request and a short list of real screen elements, then chooses one
candidate (or ``none``). Every choice is mapped back to a validated UIElement;
the model cannot produce arbitrary commands or executable code.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from app.config import IntelligenceConfig
from core.exceptions import ModelLoadError
from core.logger import get_logger
from intelligence.action_schema import Action, ActionType, ClickAction, NoAction
from vision.element import UIElement

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ORDINALS = {
    "first": 1, "1st": 1, "primo": 1, "prima": 1, "primi": 1, "prime": 1,
    "second": 2, "2nd": 2, "secondo": 2, "seconda": 2,
    "third": 3, "3rd": 3, "terzo": 3, "terza": 3,
    "fourth": 4, "4th": 4, "quarto": 4, "quarta": 4,
    "fifth": 5, "5th": 5, "quinto": 5, "quinta": 5,
}
_ORDINAL_TARGET_TYPES = {
    "folder": ("folder", "cartella", "directory"),
    "file": ("file", "document", "documento"),
    "button": ("button", "pulsante", "bottone"),
    "link": ("link", "collegamento"),
    "tab": ("tab", "scheda"),
    "checkbox": ("checkbox", "casella"),
}
_GENERIC_UI_WORDS = frozenset(
    {
        "a", "an", "the", "on", "in", "at", "to", "of", "il", "lo", "la", "i", "gli",
        "le", "un", "uno", "una", "di", "del", "della", "dei", "sul", "sulla", "su",
        "pulsante", "bottone", "button", "link", "tab", "scheda", "menu", "icona", "icon",
        "campo", "field", "casella", "checkbox", "finestra", "window",
        "click", "clicking", "clicca", "cliccare", "premi", "premere", "tap", "tocca",
    }
)
_UI_TRANSLATIONS = {
    "impostazione": "settings", "impostazioni": "settings", "settings": "settings", "setting": "settings",
    "salva": "save", "salvare": "save", "save": "save",
    "annulla": "cancel", "annullare": "cancel", "cancel": "cancel",
    "accedi": "login", "accesso": "login", "login": "login", "registrati": "register",
    "registrazione": "register", "register": "register",
    "cerca": "search", "ricerca": "search", "search": "search",
    "apri": "open", "open": "open", "chiudi": "close", "close": "close",
    "indietro": "back", "back": "back", "torna": "back", "next": "next", "avanti": "next",
    "scarica": "download", "download": "download", "carica": "upload", "upload": "upload",
    "elimina": "delete", "rimuovi": "delete", "delete": "delete", "remove": "delete",
    "modifica": "edit", "edit": "edit", "stampa": "print", "print": "print",
    "conferma": "confirm", "confirm": "confirm", "yes": "yes", "si": "yes", "no": "no",
    "home": "home", "inizio": "home", "esci": "exit", "exit": "exit",
}


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _ui_words(value: str) -> set[str]:
    normalized = _fold(value)
    normalized = re.sub(r"\b(?:sign|log)\s+in\b", "login", normalized)
    normalized = re.sub(r"\b(?:sign|log)\s+out\b", "logout", normalized)
    return {
        _UI_TRANSLATIONS.get(word, word)
        for word in _WORD_RE.findall(normalized)
        if word not in _GENERIC_UI_WORDS
    }


def _ordinal_target(
    transcript: str, elements: list[UIElement]
) -> UIElement | None:
    """Resolve explicit ordinals only when UIA exposes the requested control kind."""
    tokens = _WORD_RE.findall(_fold(transcript))
    words = set(tokens)
    ordinal = next((_ORDINALS[word] for word in tokens if word in _ORDINALS), None)
    if ordinal is None:
        return None

    target_kind = next(
        (
            kind
            for kind, aliases in _ORDINAL_TARGET_TYPES.items()
            if any(alias in words for alias in aliases)
        ),
        None,
    )
    if target_kind is None:
        return None

    aliases = _ORDINAL_TARGET_TYPES[target_kind]
    matches = [
        element
        for element in elements
        if element.clickable
        and element.type.casefold().startswith("uia:")
        and any(alias in _fold(element.type) for alias in aliases)
        and not (target_kind == "file" and "folder" in _fold(element.type))
    ]
    matches.sort(key=lambda element: (element.y, element.x, element.id))
    if len(matches) < ordinal:
        return None
    return matches[ordinal - 1]


class DecisionModel(Protocol):
    """Interface every decision-core backend must satisfy.

    ``decide`` must return an :class:`~intelligence.action_schema.Action` —
    typically a ``ClickAction`` selecting one of ``elements`` by id, or a
    ``NoAction`` when no candidate is a confident enough match. It must never
    return free-form text, arbitrary code, or an action outside
    ``allowed_actions``.
    """

    def decide(
        self,
        transcript: str,
        elements: list[UIElement],
        allowed_actions: frozenset[ActionType] | None = None,
    ) -> Action: ...


def _text_overlap_score(query: str, candidate: str) -> float:
    """Cheap lexical similarity in [0, 1]: word-overlap ratio plus substring bonus."""
    query_norm = _fold(query).strip()
    candidate_norm = _fold(candidate).strip()
    query_words = _ui_words(query_norm)
    candidate_words = _ui_words(candidate_norm)
    if not query_words or not candidate_words:
        return 0.0

    overlap = query_words & candidate_words
    if not overlap:
        return 0.0
    if query_words == candidate_words:
        return 1.0
    # Prefer an exact spoken label even when another longer label contains it.
    if query_norm and re.search(rf"(?<!\w){re.escape(query_norm)}(?!\w)", candidate_norm):
        return 0.96
    if candidate_norm and re.search(rf"(?<!\w){re.escape(candidate_norm)}(?!\w)", query_norm):
        return 0.82

    recall = len(overlap) / len(query_words)
    precision = len(overlap) / len(candidate_words)
    return 2 * recall * precision / (recall + precision)


class MockDecisionModel:
    """Deterministic, dependency-free stand-in for the real decision core.

    Scores every candidate element's visible text against the command text
    and returns a ``ClickAction`` for the best match above
    ``IntelligenceConfig.confidence_threshold``, or a ``NoAction`` otherwise.
    This lets the full CLICK pipeline (vision -> decision -> validation ->
    execution) be built and tested end to end before Laya is integrated.
    """

    def __init__(self, config: IntelligenceConfig) -> None:
        self._config = config

    def decide(
        self,
        transcript: str,
        elements: list[UIElement],
        allowed_actions: frozenset[ActionType] | None = None,
    ) -> Action:
        if not elements:
            logger.info("mock_decision: no candidate elements available")
            return NoAction(reason="no candidate UI elements available", confidence=1.0)

        candidates = elements[: self._config.max_candidates]
        scored = sorted(
            (
                (element, _text_overlap_score(transcript, element.text) * element.confidence)
                for element in candidates
                if element.clickable
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if not scored:
            return NoAction(reason="no clickable UI elements available", confidence=1.0)
        best_element, best_score = scored[0]
        logger.debug(
            "mock_decision transcript=%r best_id=%d best_text=%r score=%.2f",
            transcript,
            best_element.id,
            best_element.text,
            best_score,
        )

        ambiguous = len(scored) > 1 and best_score - scored[1][1] < 0.04
        if best_score < self._config.confidence_threshold or ambiguous:
            reason = (
                f"ambiguous screen target for {transcript!r}"
                if ambiguous
                else f"no confident match for {transcript!r} (best score {best_score:.2f})"
            )
            return NoAction(
                reason=reason,
                confidence=best_score,
            )

        return ClickAction(target_id=best_element.id, confidence=best_score)


class LayaDecisionModel:
    """Local Laya-backed selector using the multilingual checkpoint by default.

    The official ``laya`` Python package downloads its open model weights on
    first load and runs inference in-process. It makes no hosted API calls.
    A local checkpoint directory can be selected with
    ``VDOS_INTELLIGENCE_LAYA_MODEL_PATH``.
    """

    def __init__(self, config: IntelligenceConfig) -> None:
        self._config = config
        self._router = None
        self._agent = None

    def load(self) -> None:
        if self._router is not None or self._agent is not None:
            return
        try:
            import laya

            if self._config.laya_model_path is not None:
                self._agent = laya.load(str(self._config.laya_model_path))
            else:
                # VoiceDriveOS is Italian-first. The multilingual checkpoint
                # also supports English while keeping only one model resident.
                self._agent = laya.load("convaiinnovations/laya-multilingual")
        except ImportError as exc:
            raise ModelLoadError(
                "The local Laya package is missing. Install it with "
                "`pip install -r requirements.txt`; the first model load needs "
                "internet access to cache the weights."
            ) from exc
        except Exception as exc:
            self._router = None
            self._agent = None
            raise ModelLoadError(f"could not load local Laya model: {exc}") from exc
        logger.info("local Laya decision model ready")

    @staticmethod
    def _position_hint(element: UIElement, elements: list[UIElement]) -> str:
        """Describe relative position so ordinal requests have spatial context."""
        ordered = sorted(elements, key=lambda item: (item.y, item.x))
        try:
            index = next(i for i, candidate in enumerate(ordered, 1) if candidate.id == element.id)
        except StopIteration:
            return ""
        return f"screen order {index} of {len(ordered)}"

    def _predict(self, state: dict[str, object], questions: dict[str, object]) -> dict[str, object]:
        if self._agent is not None:
            return self._agent.predict(state, questions)
        if self._router is not None:
            return self._router.predict(state, questions)
        self.load()
        return self._predict(state, questions)

    def decide(
        self,
        transcript: str,
        elements: list[UIElement],
        allowed_actions: frozenset[ActionType] | None = None,
    ) -> Action:
        if allowed_actions is not None and ActionType.CLICK not in allowed_actions:
            return NoAction(reason="click is not an allowed action", confidence=1.0)
        clickable = [element for element in elements if element.clickable]
        if not clickable:
            return NoAction(reason="no clickable UI elements available", confidence=1.0)

        ordinal_target = _ordinal_target(transcript, clickable)
        if ordinal_target is not None:
            return ClickAction(
                target_id=ordinal_target.id,
                confidence=ordinal_target.confidence,
            )

        # A unique, exact visible-label match is safer and more reliable than
        # asking a zero-shot classifier to rediscover the same lexical match.
        # Laya remains the semantic selector for paraphrases and ambiguous UI.
        direct_matches = sorted(
            (
                (element, _text_overlap_score(transcript, element.text) * element.confidence)
                for element in clickable
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if direct_matches and direct_matches[0][1] >= 0.92:
            best_element, best_score = direct_matches[0]
            runner_up = direct_matches[1][1] if len(direct_matches) > 1 else 0.0
            if best_score - runner_up >= 0.04:
                return ClickAction(target_id=best_element.id, confidence=best_score)

        # Laya's choice head works best with a small, discriminative candidate
        # set. Use lexical retrieval only to bound the list; Laya makes the
        # semantic decision among the resulting screen elements.
        max_options = min(18, self._config.max_candidates)
        ranked = sorted(
            clickable,
            key=lambda element: (
                _text_overlap_score(transcript, element.text),
                element.confidence,
            ),
            reverse=True,
        )
        candidates = ranked[:max_options]
        if not candidates:
            return NoAction(reason="no visible candidates", confidence=1.0)

        options: dict[str, str] = {}
        for element in candidates:
            label = element.text.strip() or "unlabeled control"
            position = self._position_hint(element, candidates)
            options[str(element.id)] = (
                f"{element.type}; label: {label}; {position}; "
                f"confidence of screen detection: {element.confidence:.2f}"
            )
        options["none"] = "No visible candidate matches the request well enough."
        state: dict[str, object] = {
            "spoken_request": transcript,
            "visible_candidates": "\n".join(
                f"{key}: {description}" for key, description in options.items()
            ),
        }
        questions: dict[str, object] = {
            "target": {
                "type": "choice",
                "instructions": (
                    "Choose the single visible UI element matching the spoken request, "
                    "including any stated control kind or screen order. Choose none "
                    "when no candidate is a clear match."
                ),
                "criteria": options,
            }
        }
        try:
            result = self._predict(state, questions)
            answer = result["answers"]["target"]  # type: ignore[index]
            selected_key = str(answer["choice"])  # type: ignore[index]
            probabilities = answer.get("probabilities", {})  # type: ignore[union-attr]
            confidence = float(probabilities.get(selected_key, 0.0))
        except Exception as exc:
            logger.warning("Laya target selection failed: %s", exc)
            return NoAction(reason=f"local Laya decision failed: {exc}", confidence=0.0)

        if selected_key == "none" or selected_key not in {str(item.id) for item in candidates}:
            return NoAction(reason="Laya found no confident screen target", confidence=confidence)
        if confidence < self._config.confidence_threshold:
            return NoAction(
                reason=f"Laya target confidence {confidence:.2f} is below threshold",
                confidence=confidence,
            )
        return ClickAction(target_id=int(selected_key), confidence=confidence)
