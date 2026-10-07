import json

import pytest
from pydantic import ValidationError

from calmvoice.chunking import chunk_corpus, chunk_document, sentence_spans
from calmvoice.corpus import Corpus, CorpusDocument, load_corpus

GOOD = {
    "doc_id": "x1",
    "title": "A test document",
    "topic": "sleep",
    "content_type": "psychoeducation",
    "source": "tests",
    "license": "CC0-1.0",
    "text": "Sleep matters. Keep a regular time to get up. Avoid caffeine late in the day. Rest well.",
}


def test_bundled_corpus_is_valid(corpus):
    assert len(corpus.documents) == 14
    assert all(d.license == "CC0-1.0" for d in corpus.documents)


def test_licence_gate_rejects_unlicensed_text():
    # Problem 10: a copyrighted book must not enter the corpus.
    with pytest.raises(ValidationError, match="reuse allow-list"):
        CorpusDocument(**{**GOOD, "license": "All rights reserved"})


def test_personal_posts_are_not_a_valid_content_type():
    # Problem 2: forum posts of real people are not a knowledge source.
    with pytest.raises(ValidationError):
        CorpusDocument(**{**GOOD, "content_type": "personal_post"})


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        Corpus(corpus_id="c", version="1", documents=[GOOD, GOOD])


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        CorpusDocument(**{**GOOD, "target": 0})


def test_fingerprint_changes_with_content(corpus):
    data = json.loads(corpus.model_dump_json())
    data["documents"][0]["text"] += " One more sentence for the test."
    assert Corpus.model_validate(data).fingerprint() != corpus.fingerprint()


def test_topic_comes_from_each_document(corpus):
    # Problem 6: no hard-coded label. Each chunk keeps the topic of its own document.
    topics = {c.doc_id: c.topic for c in chunk_corpus(corpus)}
    by_id = corpus.by_id()
    assert all(topics[d] == by_id[d].topic for d in topics)
    assert len(set(topics.values())) >= 5


def test_load_corpus_from_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"corpus_id": "t", "version": "1", "documents": [GOOD]}), encoding="utf-8")
    assert load_corpus(p).documents[0].doc_id == "x1"


def test_chunks_are_exact_unmodified_slices(corpus):
    # Problem 5: no lower-casing, no stopword removal, no lemmatisation.
    by_id = corpus.by_id()
    for c in chunk_corpus(corpus):
        assert by_id[c.doc_id].text[c.start : c.end] == c.text
        assert c.text[0].isupper()


def test_chunks_respect_max_chars_and_overlap(corpus):
    doc = corpus.documents[0]
    chunks = chunk_document(doc, max_chars=200, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    # Consecutive chunks share at least one sentence.
    assert chunks[1].start < chunks[0].end


def test_long_sentence_becomes_own_chunk():
    doc = CorpusDocument(**{**GOOD, "text": "Short one. " + "word " * 80 + "end. Last sentence here."})
    chunks = chunk_document(doc, max_chars=100, overlap_sentences=0)
    assert any(len(c.text) > 100 for c in chunks)
    assert "".join(c.text for c in chunks).replace(" ", "") == doc.text.replace(" ", "")


def test_sentence_spans():
    assert len(sentence_spans("One. Two! Three? four")) == 3


def test_chunk_arguments_validated(corpus):
    with pytest.raises(ValueError):
        chunk_document(corpus.documents[0], max_chars=10)
    with pytest.raises(ValueError):
        chunk_document(corpus.documents[0], overlap_sentences=-1)
