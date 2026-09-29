"""Deterministic text-payload extraction.

This is one of the most important optimizations in the project brief: when
the user says "write hello world", the literal string ``"hello world"`` is
sliced out of the transcript with plain string code — it is never generated,
paraphrased, or touched by any model. This guarantees byte-for-byte fidelity,
removes an entire class of hallucination risk, and is effectively free
compared to any form of generation.

Supports both English and Italian trigger verbs, since the project brief is
written in Italian while the reference examples in the tech stack are in
English.
"""

from __future__ import annotations

import re

# Ordered by trigger length (longest first) so "write down" (if ever added)
# would not be shadowed by a shorter "write" prefix match.
_TEXT_TRIGGERS: tuple[str, ...] = (
    "write",
    "type",
    "scrivi",
    "digita",
    "inserisci",
)

_TRIGGER_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(re.escape(t) for t in _TEXT_TRIGGERS) + r")\s+(.+)$",
    re.IGNORECASE,
)


def extract_text_payload(transcript: str) -> str | None:
    """Extract the literal text to type from a transcript, or ``None`` if no match.

    Examples:
        >>> extract_text_payload("write hello world")
        'hello world'
        >>> extract_text_payload("scrivi il mio nome è Federico")
        'il mio nome è Federico'
        >>> extract_text_payload("click settings")
        None
    """
    if not transcript or not transcript.strip():
        return None

    match = _TRIGGER_PATTERN.match(transcript.strip())
    if not match:
        return None

    payload = match.group(1).strip()
    return payload or None


def is_text_command(transcript: str) -> bool:
    """Return True if ``transcript`` begins with a recognized text-entry trigger."""
    return extract_text_payload(transcript) is not None
