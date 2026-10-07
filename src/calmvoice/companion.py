"""The companion pipeline: safety gate -> scope check -> retrieval -> generation -> output guard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .config import Settings
from .corpus import Corpus, load_corpus
from .embeddings import Embedder, make_embedder
from .generation import ExtractiveGenerator, Generator, LLMGenerator
from .index import load_or_build
from .llm import LLMError, OllamaLLM
from .memory import SessionMemory
from .retrieval import Retriever
from .safety import (
    SCOPE_REPLIES,
    RegionResources,
    RiskAssessment,
    RiskLevel,
    SafetyGate,
    check_scope,
    escalation_message,
    format_resources,
    guard_output,
    load_resources,
)

Route = Literal["crisis", "out_of_scope", "answer", "blocked", "empty"]

CONCERN_PREFIX = (
    "Thank you for telling me how you feel. Those feelings sound heavy. "
    "If they ever turn into thoughts of harming yourself, please contact a crisis line straight away."
)


@dataclass(frozen=True)
class SourceRef:
    n: int
    doc_id: str
    title: str
    source: str
    url: str
    license: str


@dataclass
class Reply:
    text: str
    route: Route
    risk_level: str
    risk_categories: tuple[str, ...] = ()
    sources: list[SourceRef] = field(default_factory=list)
    guard_reasons: list[str] = field(default_factory=list)
    generator: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_categories"] = list(self.risk_categories)
        return d


class Companion:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        corpus: Corpus,
        resources: RegionResources,
        gate: SafetyGate | None = None,
        memory: SessionMemory | None = None,
        top_k: int = 4,
        fallback: Generator | None = None,
    ):
        self.retriever, self.generator, self.corpus = retriever, generator, corpus
        self.resources = resources
        self.gate = gate or SafetyGate()
        self.memory = memory if memory is not None else SessionMemory(enabled=False)
        self.top_k = top_k
        self.fallback = fallback
        self._docs = corpus.by_id()

    def respond(self, message: str) -> Reply:
        message = (message or "").strip()
        if not message:
            return Reply("Please type or say a message.", "empty", RiskLevel.NONE.label)

        # 1. Safety gate first. A crisis message never reaches retrieval or the LLM.
        risk: RiskAssessment = self.gate.assess(message)
        if risk.is_crisis:
            text = escalation_message(risk, self.resources)
            self.memory.add(message, text)
            return Reply(text, "crisis", risk.level.label, risk.categories)

        # 2. Scope limits: no diagnosis, no medication advice.
        scope = check_scope(message)
        if not scope.in_scope:
            text = SCOPE_REPLIES[scope.reason]
            if risk.level == RiskLevel.CONCERN:
                text = f"{CONCERN_PREFIX}\n\n{text}\n\n{format_resources(self.resources)}"
            self.memory.add(message, text)
            return Reply(text, "out_of_scope", risk.level.label, risk.categories)

        # 3. Retrieval over the curated corpus.
        hits = self.retriever.retrieve(message, k=self.top_k)

        # 4. Generation. If the LLM fails, use the offline generator.
        used = self.generator
        try:
            raw = used.generate(message, hits, self.memory.history())
        except LLMError:
            if self.fallback is None:
                raise
            used = self.fallback
            raw = used.generate(message, hits, self.memory.history())

        # 5. Output guard: block diagnoses and medication advice, drop invalid citations.
        guarded = guard_output(raw, len(hits))
        text = guarded.text
        if risk.level == RiskLevel.CONCERN:
            text = f"{CONCERN_PREFIX}\n\n{text}\n\n{format_resources(self.resources)}"
        sources = []
        for n, hit in enumerate(hits, start=1):
            doc = self._docs.get(hit.chunk.doc_id)
            if doc is not None:
                sources.append(SourceRef(n, doc.doc_id, doc.title, doc.source, doc.url, doc.license))
        self.memory.add(message, text)
        return Reply(
            text,
            "blocked" if guarded.blocked else "answer",
            risk.level.label,
            risk.categories,
            [] if guarded.blocked else sources,
            guarded.reasons,
            used.name,
        )


def build_companion(settings: Settings, embedder: Embedder | None = None, memory: SessionMemory | None = None) -> Companion:
    """Wire a companion from settings. With CALMVOICE_LLM=offline it needs no network."""
    corpus = load_corpus(settings.corpus_path or None)
    embedder = embedder or make_embedder(settings.embedder)
    index, _ = load_or_build(settings.index_dir or None, corpus, embedder)
    retriever = Retriever(index, embedder, mode="hybrid")
    offline = ExtractiveGenerator()
    if settings.llm == "ollama":
        generator: Generator = LLMGenerator(
            OllamaLLM(settings.ollama_url, settings.ollama_model, settings.temperature), name=f"ollama:{settings.ollama_model}"
        )
    else:
        generator = offline
    return Companion(
        retriever,
        generator,
        corpus,
        load_resources(settings.region),
        memory=memory if memory is not None else SessionMemory(enabled=settings.memory_enabled),
        top_k=settings.top_k,
        fallback=offline,
    )
