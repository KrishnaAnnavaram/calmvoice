"""Answer generation: the system policy, the LLM generator and the offline extractive generator."""

from __future__ import annotations

from typing import Protocol, Sequence

from .llm import LLM
from .chunking import sentence_spans
from .retrieval import Retrieved
from .textutil import content_tokens, stem

SYSTEM_POLICY = """You are calmvoice, a supportive wellbeing companion. Follow these rules:
1. Be warm, calm and brief (at most 150 words). Reflect the feeling of the user first.
2. Give general wellbeing information ONLY from the numbered sources. Cite each fact as [n].
3. If the sources do not cover the question, say so and suggest a professional.
4. Never diagnose a condition. Never name or advise on medication or doses.
5. Never claim to be a therapist, a doctor or a human.
6. Suggest a doctor or a mental health professional when problems last or affect daily life."""


def format_sources(sources: Sequence[Retrieved]) -> str:
    return "\n\n".join(f"[{i}] {r.chunk.title}\n{r.chunk.text}" for i, r in enumerate(sources, start=1))


def build_prompt(message: str, sources: Sequence[Retrieved]) -> str:
    return f"Sources:\n{format_sources(sources)}\n\nUser message: {message}\n\nAnswer with citations like [1]."


class Generator(Protocol):
    name: str

    def generate(self, message: str, sources: Sequence[Retrieved], history: Sequence[tuple[str, str]] = ()) -> str: ...


class LLMGenerator:
    def __init__(self, llm: LLM, name: str = "llm"):
        self.llm, self.name = llm, name

    def generate(self, message: str, sources: Sequence[Retrieved], history: Sequence[tuple[str, str]] = ()) -> str:
        return self.llm.complete(SYSTEM_POLICY, build_prompt(message, sources), history)


OPENINGS = {
    "stress": "That sounds like a lot of pressure to carry.",
    "anxiety": "Anxious feelings can be very uncomfortable, and it makes sense that you want them to ease.",
    "sleep": "Nights like that are exhausting, and they can make the day harder too.",
    "low-mood": "I am sorry that things feel heavy right now.",
    "coping": "Thank you for sharing how you feel.",
    "relationships": "Feeling disconnected from people can hurt a lot.",
    "help-seeking": "It is a good step to think about support.",
}

CLOSING = (
    "If these feelings last for more than two weeks or get in the way of daily life, "
    "a doctor or a mental health professional can help."
)


class ExtractiveGenerator:
    """Offline generator: an empathic opening, the best matching source sentences with citations,
    and a closing referral. It never writes facts that are not in a source."""

    name = "offline-extractive"

    def __init__(self, sentences: int = 3):
        self.sentences = sentences

    def generate(self, message: str, sources: Sequence[Retrieved], history: Sequence[tuple[str, str]] = ()) -> str:
        if not sources:
            return (
                "I do not have information about that in my sources. "
                "A doctor or a mental health professional can give you better advice."
            )
        q = {stem(t) for t in content_tokens(message)}
        scored: list[tuple[float, int, int, str]] = []
        for si, r in enumerate(sources, start=1):
            text = r.chunk.text
            for pos, (s, e) in enumerate(sentence_spans(text)):
                sent = text[s:e].strip()
                words = {stem(t) for t in content_tokens(sent)}
                if len(words) < 4:
                    continue
                overlap = len(q & words) / (len(q) or 1)
                # Prefer the top source and earlier sentences on ties.
                scored.append((overlap - 0.02 * (si - 1) - 0.001 * pos, si, pos, sent))
        scored.sort(key=lambda x: -x[0])
        chosen = sorted(scored[: self.sentences], key=lambda x: (x[1], x[2]))
        body = " ".join(f"{sent} [{si}]" for _, si, _, sent in chosen)
        opening = OPENINGS.get(sources[0].chunk.topic, OPENINGS["coping"])
        return f"{opening} {body}\n\n{CLOSING}"
