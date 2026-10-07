import pytest

from calmvoice.corpus import load_corpus
from calmvoice.embeddings import HashingEmbedder
from calmvoice.index import build_index
from calmvoice.retrieval import Retriever


@pytest.fixture(scope="session")
def corpus():
    return load_corpus()


@pytest.fixture(scope="session")
def embedder():
    return HashingEmbedder()


@pytest.fixture(scope="session")
def index(corpus, embedder):
    return build_index(corpus, embedder)


@pytest.fixture()
def retriever(index, embedder):
    return Retriever(index, embedder, mode="hybrid")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith("CALMVOICE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(os.path.dirname(__file__))
