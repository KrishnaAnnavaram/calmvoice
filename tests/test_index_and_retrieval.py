import json

import numpy as np
import pytest

from calmvoice.corpus import Corpus
from calmvoice.embeddings import HashingEmbedder
from calmvoice.index import DenseIndex, build_index, is_current, load_or_build
from calmvoice.lexical import BM25
from calmvoice.llm import ScriptedLLM
from calmvoice.retrieval import LLMRewriter, Retriever, RuleRewriter, SingleQuery, reciprocal_rank_fusion


def test_hashing_embedder_is_deterministic_and_normalised():
    a = HashingEmbedder().embed(["slow breathing helps", "sleep"])
    b = HashingEmbedder().embed(["slow breathing helps", "sleep"])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_index_persists_and_is_reused(tmp_path, corpus, embedder):
    # Problem 7: the index is built once and loaded on later starts.
    idx, built = load_or_build(tmp_path, corpus, embedder)
    assert built
    idx2, built2 = load_or_build(tmp_path, corpus, embedder)
    assert not built2
    assert np.allclose(idx.vectors, idx2.vectors)
    assert [c.chunk_id for c in idx.chunks] == [c.chunk_id for c in idx2.chunks]


def test_index_is_rebuilt_when_corpus_changes(tmp_path, corpus, embedder):
    load_or_build(tmp_path, corpus, embedder)
    data = json.loads(corpus.model_dump_json())
    data["documents"][1]["text"] += " An extra sentence changes the fingerprint."
    changed = Corpus.model_validate(data)
    assert not is_current(tmp_path, changed, embedder, 420, 1)
    _, built = load_or_build(tmp_path, changed, embedder)
    assert built


def test_index_rebuilt_when_embedder_changes(tmp_path, corpus):
    load_or_build(tmp_path, corpus, HashingEmbedder(512))
    assert not is_current(tmp_path, corpus, HashingEmbedder(1024), 420, 1)


def test_dense_search_order(index, embedder):
    q = embedder.embed(["heart pounds panic attack"])[0]
    hits = index.search(q, 3)
    assert len(hits) == 3
    assert hits[0][1] >= hits[1][1] >= hits[2][1]


def test_index_length_mismatch_rejected(index):
    with pytest.raises(ValueError):
        DenseIndex(index.chunks[:2], index.vectors, "x", "y", 420, 1)


def test_bm25_prefers_matching_text():
    bm = BM25(["sleep at night in bed", "breathing slowly for stress", "talk to friends"])
    assert bm.search("cannot sleep at night", 3)[0][0] == 0
    assert bm.search("zzzz", 3) == []


def test_rrf_over_one_list_is_a_no_op():
    # Problem 4: the prototype fused ONE list, which returns the same ranking.
    ranking = [5, 2, 9, 1]
    assert [i for i, _ in reciprocal_rank_fusion([ranking])] == ranking


def test_rrf_over_several_lists_changes_ranking():
    fused = [i for i, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1], [2, 3, 1]])]
    assert fused[0] == 2
    assert fused != [1, 2, 3]


def test_rule_rewriter_makes_real_variants():
    variants = RuleRewriter().rewrite("I feel so anxious before my exams")
    assert len(variants) == 3
    assert variants[0] == "I feel so anxious before my exams"
    assert "anxiety" in variants[2]


def test_llm_rewriter_and_fallback():
    llm = ScriptedLLM(["1. coping with exam stress\n2. study pressure tips\n"])
    out = LLMRewriter(llm, n=2).rewrite("exam stress")
    assert out[0] == "exam stress" and "coping with exam stress" in out

    class Broken:
        def complete(self, **kw):
            raise RuntimeError("down")

    assert LLMRewriter(Broken()).rewrite("q") == ["q"]


def test_hybrid_multiquery_gives_several_lists(index, embedder):
    r = Retriever(index, embedder, mode="hybrid", rewriter=RuleRewriter())
    variants, lists = r.rankings("I am anxious and nervous")
    assert len(variants) >= 2
    assert len(lists) == 2 * len(variants)


def test_retrieval_finds_relevant_documents(retriever):
    assert retriever.retrieve("my heart pounds and my hands tingle with sudden fear", k=3)[0].chunk.doc_id == "panic-body"
    docs = [r.chunk.doc_id for r in retriever.retrieve("I lie awake at night and cannot sleep", k=4)]
    assert "sleep-habits" in docs or "racing-thoughts-night" in docs


def test_max_per_doc_gives_diverse_sources(retriever):
    docs = [r.chunk.doc_id for r in retriever.retrieve("sleep night bed awake", k=4)]
    assert len(docs) == len(set(docs))


def test_modes(index, embedder):
    for mode in ("dense", "lexical"):
        r = Retriever(index, embedder, mode=mode)
        assert isinstance(r.rewriter, SingleQuery)
        assert r.retrieve("breathing for stress", k=2)
    with pytest.raises(ValueError):
        Retriever(index, embedder, mode="magic")


def test_faiss_backend_matches_numpy(corpus, embedder):
    pytest.importorskip("faiss")
    a = build_index(corpus, embedder)
    b = build_index(corpus, embedder, backend="faiss")
    q = embedder.embed(["worry all day"])[0]
    assert [i for i, _ in a.search(q, 5)] == [i for i, _ in b.search(q, 5)]
