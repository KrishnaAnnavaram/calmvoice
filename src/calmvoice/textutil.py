"""Token helpers for retrieval features. They never change the stored or displayed text."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset(
    """a about after again all am an and any are as at be because been before being but by can could
    did do does doing down during each few for from further had has have having he her here hers him his
    how i if in into is it its itself just me more most my myself no nor not now of off on once only or
    other our ours out over own same she should so some such than that the their them then there these
    they this those through to too under until up very was we were what when where which while who whom
    why will with would you your yours yourself im ive dont cant""".split()
)


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def stem(token: str) -> str:
    """A light suffix stripper for matching only (anxious/anxiety, sleeping/sleep)."""
    for suffix in ("ingly", "ously", "iety", "ious", "ness", "ing", "ies", "ied", "ous", "ed", "ly", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token
