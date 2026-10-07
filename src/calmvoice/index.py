"""A persisted dense index. Build it once, load it on each start.

The index stores the corpus fingerprint, the embedder name and the chunk settings. ``load_or_build``
builds the index again only when one of these values changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_corpus
from .corpus import Corpus
from .embeddings import Embedder

META_FILE = "index_meta.json"
CHUNKS_FILE = "chunks.json"
VECTORS_FILE = "vectors.npy"


@dataclass
class DenseIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    embedder_name: str
    fingerprint: str
    max_chars: int
    overlap: int
    backend: str = "numpy"

    def __post_init__(self) -> None:
        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError("chunks and vectors differ in length")
        self._faiss = None
        if self.backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError('the FAISS backend needs: pip install -e ".[faiss]"') from exc
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(np.ascontiguousarray(self.vectors, dtype=np.float32))

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk position, cosine score) pairs, best first."""
        k = min(k, len(self.chunks))
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        scores = self.vectors @ q[0]
        # Stable order on ties: higher score first, then lower position.
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(int(i), float(scores[i])) for i in order]

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / VECTORS_FILE, self.vectors)
        (d / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        meta = {
            "embedder": self.embedder_name,
            "fingerprint": self.fingerprint,
            "max_chars": self.max_chars,
            "overlap": self.overlap,
            "n_chunks": len(self.chunks),
        }
        (d / META_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path, backend: str = "numpy") -> "DenseIndex":
        d = Path(directory)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in json.loads((d / CHUNKS_FILE).read_text(encoding="utf-8"))]
        vectors = np.load(d / VECTORS_FILE)
        return cls(chunks, vectors, meta["embedder"], meta["fingerprint"], meta["max_chars"], meta["overlap"], backend)


def build_index(corpus: Corpus, embedder: Embedder, max_chars: int = 420, overlap: int = 1, backend: str = "numpy") -> DenseIndex:
    chunks = chunk_corpus(corpus, max_chars=max_chars, overlap_sentences=overlap)
    vectors = embedder.embed([f"{c.title}. {c.text}" for c in chunks])
    return DenseIndex(chunks, vectors, embedder.name, corpus.fingerprint(), max_chars, overlap, backend)


def is_current(directory: str | Path, corpus: Corpus, embedder: Embedder, max_chars: int, overlap: int) -> bool:
    meta_path = Path(directory) / META_FILE
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return (
        meta.get("fingerprint") == corpus.fingerprint()
        and meta.get("embedder") == embedder.name
        and meta.get("max_chars") == max_chars
        and meta.get("overlap") == overlap
    )


def load_or_build(
    directory: str | Path | None,
    corpus: Corpus,
    embedder: Embedder,
    max_chars: int = 420,
    overlap: int = 1,
    backend: str = "numpy",
) -> tuple[DenseIndex, bool]:
    """Return (index, built). ``built`` is False when a current index was loaded from disk."""
    if directory and is_current(directory, corpus, embedder, max_chars, overlap):
        return DenseIndex.load(directory, backend=backend), False
    index = build_index(corpus, embedder, max_chars, overlap, backend)
    if directory:
        index.save(directory)
    return index, True
