"""Embedders behind one small interface.

``HashingEmbedder`` is offline and deterministic (NumPy only). ``MiniLMEmbedder`` uses
sentence-transformers and is imported lazily, so the core package works without it.
"""

from __future__ import annotations

import zlib
from typing import Protocol, Sequence

import numpy as np

from .textutil import content_tokens, stem


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, dim), rows L2-normalised
        ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Signed feature hashing of stemmed words, word bigrams and character trigrams.

    It uses crc32, not Python's salted ``hash``, so vectors are the same in every process.
    """

    def __init__(self, dim: int = 1024):
        if dim < 16:
            raise ValueError("dim must be 16 or more")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [stem(t) for t in content_tokens(text)]
        feats: list[tuple[str, float]] = [("w:" + w, 1.0) for w in words]
        feats += [("b:" + a + "_" + b, 0.7) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"<{w}>"
            feats += [("c:" + padded[i : i + 3], 0.3) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * weight
        return _l2(out)


class MiniLMEmbedder:
    """sentence-transformers ``all-MiniLM-L6-v2`` (optional extra ``embeddings``)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError('MiniLM needs the extra: pip install -e ".[embeddings]"') from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - optional extra
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(kind: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "minilm":
        return MiniLMEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")
