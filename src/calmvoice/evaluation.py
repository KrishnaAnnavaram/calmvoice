"""Evaluation of the parts that matter: safety recall, retrieval quality and answer grounding.

These replace the prototype's t-tests on the character length of retrieved documents, which measured
nothing about the system.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Sequence

from .chunking import sentence_spans
from .retrieval import Retriever
from .safety import RiskLevel, SafetyGate, guard_output
from .synthetic import RedTeamItem
from .textutil import content_tokens, stem

LEVELS = ["none", "concern", "crisis"]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------- safety


@dataclass
class SafetyReport:
    n: int
    confusion: dict[str, dict[str, int]]
    crisis_recall: float
    crisis_recall_ci: tuple[float, float]
    crisis_precision: float
    false_alarm_rate: float  # "none" messages that got crisis or concern
    missed_crisis: list[str]

    def to_dict(self) -> dict:
        return self.__dict__ | {"crisis_recall_ci": list(self.crisis_recall_ci)}


def evaluate_safety(gate: SafetyGate, items: Sequence[RedTeamItem]) -> SafetyReport:
    confusion = {t: {p: 0 for p in LEVELS} for t in LEVELS}
    missed = []
    for it in items:
        pred = gate.assess(it.text).level.label
        confusion[it.level][pred] += 1
        if it.level == "crisis" and pred != "crisis":
            missed.append(it.text)
    n_crisis = sum(confusion["crisis"].values())
    tp = confusion["crisis"]["crisis"]
    pred_crisis = sum(confusion[t]["crisis"] for t in LEVELS)
    n_none = sum(confusion["none"].values())
    false_alarm = confusion["none"]["crisis"] + confusion["none"]["concern"]
    return SafetyReport(
        n=len(items),
        confusion=confusion,
        crisis_recall=tp / n_crisis if n_crisis else 0.0,
        crisis_recall_ci=wilson_interval(tp, n_crisis),
        crisis_precision=tp / pred_crisis if pred_crisis else 0.0,
        false_alarm_rate=false_alarm / n_none if n_none else 0.0,
        missed_crisis=missed,
    )


# ---------------------------------------------------------------- retrieval


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    relevant: tuple[str, ...]


def load_retrieval_set(path: str | Path | None = None) -> list[RetrievalQuery]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("calmvoice.data").joinpath("retrieval_eval.jsonl").read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(RetrievalQuery(d["query"], tuple(d["relevant"])))
    return out


@dataclass
class RetrievalReport:
    mode: str
    k: int
    n: int
    recall_at_k: float
    hit_at_k: float
    hit_ci: tuple[float, float]
    mrr: float


def evaluate_retrieval(
    retriever: Retriever, queries: Sequence[RetrievalQuery], k: int = 4, label: str | None = None
) -> RetrievalReport:
    recalls, hits, rr = [], 0, []
    for q in queries:
        docs: list[str] = []
        for r in retriever.retrieve(q.query, k=k):
            if r.chunk.doc_id not in docs:
                docs.append(r.chunk.doc_id)
        rel = set(q.relevant)
        found = rel & set(docs)
        recalls.append(len(found) / len(rel))
        hits += 1 if found else 0
        first = next((i for i, d in enumerate(docs, start=1) if d in rel), None)
        rr.append(1.0 / first if first else 0.0)
    n = len(queries)
    return RetrievalReport(label or retriever.mode, k, n, sum(recalls) / n, hits / n, wilson_interval(hits, n), sum(rr) / n)


# ---------------------------------------------------------------- answer grounding


@dataclass
class AnswerReport:
    n: int
    answered: int
    blocked: int
    citation_validity: float  # share of [n] markers that point to a real source
    grounded_sentence_rate: float  # share of cited sentences whose words appear in the cited source
    mean_words: float


def grounded(sentence: str, source_text: str, threshold: float = 0.6) -> bool:
    words = {stem(t) for t in content_tokens(sentence)}
    if not words:
        return True
    src = {stem(t) for t in content_tokens(source_text)}
    return len(words & src) / len(words) >= threshold


_CLAIM = re.compile(r"([^\[\]]+?)((?:\s*\[\d+\])+)")


def cited_claims(text: str) -> list[tuple[str, list[int]]]:
    """Return (sentence, cited source numbers) for each citation group.

    The claim is the LAST sentence before the citation markers, so an uncited opening sentence is
    not counted as a claim of the source.
    """
    out = []
    for m in _CLAIM.finditer(text):
        segment = m.group(1).strip()
        spans = sentence_spans(segment)
        claim = segment[spans[-1][0] : spans[-1][1]] if spans else segment
        out.append((claim, [int(x) for x in re.findall(r"\d+", m.group(2))]))
    return out


def evaluate_answers(companion, queries: Sequence[RetrievalQuery]) -> AnswerReport:
    """Proxy faithfulness check: each cited sentence must share most of its words with its source."""

    valid = total_cites = 0
    grounded_n = cited_sentences = 0
    blocked = answered = 0
    words = 0
    for q in queries:
        hits = companion.retriever.retrieve(q.query, k=companion.top_k)
        raw = companion.generator.generate(q.query, hits)
        g = guard_output(raw, len(hits))
        if g.blocked:
            blocked += 1
            continue
        answered += 1
        words += len(g.text.split())
        total_cites += len(re.findall(r"\[(\d+)\]", raw))
        valid += len(re.findall(r"\[(\d+)\]", g.text))
        for claim, nums in cited_claims(raw):
            refs = [n for n in nums if 1 <= n <= len(hits)]
            if refs:
                cited_sentences += 1
                grounded_n += any(grounded(claim, hits[n - 1].chunk.text) for n in refs)
    return AnswerReport(
        n=len(queries),
        answered=answered,
        blocked=blocked,
        citation_validity=valid / total_cites if total_cites else 0.0,
        grounded_sentence_rate=grounded_n / cited_sentences if cited_sentences else 0.0,
        mean_words=words / answered if answered else 0.0,
    )


def level_counts(items: Sequence[RedTeamItem]) -> dict[str, int]:
    out = {lv: 0 for lv in LEVELS}
    for it in items:
        out[it.level] += 1
    return out


__all__ = [
    "RiskLevel",
    "evaluate_safety",
    "evaluate_retrieval",
    "evaluate_answers",
    "load_retrieval_set",
    "wilson_interval",
    "level_counts",
]
