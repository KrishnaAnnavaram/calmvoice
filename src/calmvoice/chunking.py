"""Split documents into chunks of UNMODIFIED text.

The chunker never lower-cases, lemmatises or removes stopwords. Each chunk is an exact slice of the
source text, so the LLM and the user see readable text, and the embedder sees text that fits its
input limit.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .corpus import Corpus, CorpusDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    topic: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of the sentences in text."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def chunk_document(doc: CorpusDocument, max_chars: int = 420, overlap_sentences: int = 1) -> list[Chunk]:
    """Pack whole sentences into chunks of at most max_chars characters.

    A sentence longer than max_chars becomes its own chunk. Consecutive chunks share
    ``overlap_sentences`` sentences so that an idea that crosses a boundary stays retrievable.
    """
    if max_chars < 50:
        raise ValueError("max_chars must be 50 or more")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be 0 or more")
    spans = sentence_spans(doc.text)
    chunks: list[Chunk] = []
    i = 0
    while i < len(spans):
        j = i
        while j + 1 < len(spans) and spans[j + 1][1] - spans[i][0] <= max_chars:
            j += 1
        start, end = spans[i][0], spans[j][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                title=doc.title,
                topic=doc.topic,
                text=doc.text[start:end],
                start=start,
                end=end,
            )
        )
        if j + 1 >= len(spans):
            break
        i = max(i + 1, j + 1 - overlap_sentences)
    return chunks


def chunk_corpus(corpus: Corpus, max_chars: int = 420, overlap_sentences: int = 1) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in corpus.documents:
        out.extend(chunk_document(doc, max_chars=max_chars, overlap_sentences=overlap_sentences))
    return out
