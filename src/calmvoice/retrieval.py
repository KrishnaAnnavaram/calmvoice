"""Hybrid retrieval: real multi-query rewriting, dense + BM25 search, reciprocal rank fusion.

Reciprocal rank fusion (RRF) over ONE ranked list returns the same order as that list. Fusion only
changes the ranking when there are several lists. This module always gives several lists in
``hybrid`` mode: each query variant is searched with the dense index AND with BM25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from .chunking import Chunk
from .embeddings import Embedder
from .index import DenseIndex
from .lexical import BM25
from .textutil import content_tokens

Mode = Literal["dense", "lexical", "hybrid"]

# Everyday words -> the vocabulary of the corpus. Used by the offline rewriter.
EXPANSIONS: dict[str, str] = {
    "anxious": "anxiety worry",
    "nervous": "anxiety worry",
    "worried": "worry anxiety",
    "scared": "fear panic",
    "panicking": "panic attack",
    "heart": "panic body sensations",
    "sad": "low mood",
    "down": "low mood",
    "depressed": "low mood activities",
    "unmotivated": "low mood motivation activities",
    "tired": "sleep energy",
    "insomnia": "sleep awake bed",
    "sleep": "sleep bed night",
    "awake": "sleep night",
    "night": "sleep bedtime",
    "overwhelmed": "stress workload",
    "stressed": "stress pressure",
    "deadline": "workload stress tasks",
    "exams": "study workload stress",
    "lonely": "loneliness connection people",
    "alone": "loneliness connection",
    "friends": "connection people",
    "therapist": "therapy professional",
    "therapy": "talking therapies therapist",
    "counselling": "talking therapies",
    "help": "professional support",
    "exercise": "physical activity movement",
    "walk": "physical activity movement",
    "critic": "self-compassion kindness",
    "hate": "self-criticism self-compassion",
    "failure": "unhelpful thoughts self-compassion",
    "worthless": "self-criticism self-compassion unhelpful thoughts",
    "hopeless": "low mood unhelpful thoughts",
    "better": "balanced thought",
    "thoughts": "thoughts thinking",
    "calm": "breathing grounding",
    "breathe": "breathing",
    "breath": "breathing",
}


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> list[str]: ...


class SingleQuery:
    """No rewriting. Gives one query variant."""

    def rewrite(self, query: str) -> list[str]:
        return [query]


class RuleRewriter:
    """Offline multi-query: the original, a keyword-only variant and a vocabulary expansion."""

    def __init__(self, max_variants: int = 3):
        self.max_variants = max_variants

    def rewrite(self, query: str) -> list[str]:
        words = content_tokens(query)
        variants = [query.strip()]
        if words:
            variants.append(" ".join(words))
        extra = [EXPANSIONS[w] for w in words if w in EXPANSIONS]
        if extra:
            variants.append(" ".join(words + extra))
        return _dedupe(variants)[: self.max_variants]


class LLMRewriter:
    """Ask the LLM for paraphrases. On any error it falls back to the original query only."""

    PROMPT = (
        "Write {n} short search queries that mean the same as the user's message, one per line, "
        "with no numbering and no other text.\nMessage: {query}"
    )

    def __init__(self, llm, n: int = 3):
        self.llm, self.n = llm, n

    def rewrite(self, query: str) -> list[str]:
        try:
            raw = self.llm.complete(system="You rewrite search queries.", prompt=self.PROMPT.format(n=self.n, query=query))
        except Exception:
            return [query]
        lines = [ln.strip(" -*0123456789.").strip() for ln in raw.splitlines()]
        return _dedupe([query] + [ln for ln in lines if 3 <= len(ln) <= 200])[: self.n + 1]


def _dedupe(items: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        key = it.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists of item ids. Score = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float
    variants: tuple[str, ...] = field(default=())


class Retriever:
    def __init__(
        self,
        index: DenseIndex,
        embedder: Embedder,
        mode: Mode = "hybrid",
        rewriter: QueryRewriter | None = None,
        depth: int = 20,
        rrf_k: int = 60,
        max_per_doc: int = 1,
    ):
        if mode not in ("dense", "lexical", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        self.index, self.embedder, self.mode = index, embedder, mode
        self.rewriter = rewriter or (RuleRewriter() if mode == "hybrid" else SingleQuery())
        self.depth, self.rrf_k, self.max_per_doc = depth, rrf_k, max_per_doc
        self.bm25 = BM25([f"{c.title}. {c.text}" for c in index.chunks])

    def rankings(self, query: str) -> tuple[list[str], list[list[int]]]:
        variants = self.rewriter.rewrite(query)
        lists: list[list[int]] = []
        if self.mode in ("dense", "hybrid"):
            vecs = self.embedder.embed(variants)
            for v in vecs:
                lists.append([i for i, _ in self.index.search(v, self.depth)])
        if self.mode in ("lexical", "hybrid"):
            for q in variants:
                lists.append([i for i, _ in self.bm25.search(q, self.depth)])
        return variants, [lst for lst in lists if lst]

    def retrieve(self, query: str, k: int = 4) -> list[Retrieved]:
        variants, lists = self.rankings(query)
        fused = reciprocal_rank_fusion(lists, k=self.rrf_k)
        out: list[Retrieved] = []
        per_doc: dict[str, int] = {}
        for pos, score in fused:
            chunk = self.index.chunks[pos]
            if per_doc.get(chunk.doc_id, 0) >= self.max_per_doc:
                continue
            per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
            out.append(Retrieved(chunk, score, tuple(variants)))
            if len(out) == k:
                break
        return out
