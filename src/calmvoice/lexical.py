"""Okapi BM25 over the chunk texts (pure Python)."""

from __future__ import annotations

import math
from collections import Counter

from .textutil import content_tokens, stem


def _terms(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_terms(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = _terms(query)
        out = []
        for d, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1.0))
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            out.append(s)
        return out

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in order[:k] if scores[i] > 0]
