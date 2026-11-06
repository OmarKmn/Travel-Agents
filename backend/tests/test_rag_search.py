"""Tests de la recherche hybride, avec le faux modèle d'embeddings (sans Ollama)."""

import pytest

from app.rag.bm25 import BM25
from app.rag.chunking import Chunk
from app.rag.embeddings import FakeEmbedder
from app.rag.index import HybridIndex
from app.rag.text import tokenize

CHUNKS = [
    Chunk("Lisbonne-0000", "Lisbonne", "Voir", "- Tour de Belém : tour manuéline au bord du Tage.", "src"),
    Chunk("Lisbonne-0001", "Lisbonne", "Manger", "Goûtez les pastéis de nata et le poisson grillé.", "src"),
    Chunk("Rome-0000", "Rome", "Manger", "Pizza, pâtes et poisson frais près du marché.", "src"),
    Chunk("Rome-0001", "Rome", "Voir", "- Colisée : amphithéâtre antique.", "src"),
]


@pytest.fixture
def index() -> HybridIndex:
    return HybridIndex.build(CHUNKS, FakeEmbedder())


def test_tokenize_lowercases_and_removes_accents_and_stopwords():
    assert tokenize("Où voir la Tour de Belém ?") == ["voir", "tour", "belem"]


def test_bm25_prefers_documents_containing_rare_words():
    bm25 = BM25([["tour", "belem"], ["poisson", "grille"], ["tour", "eiffel"]])

    scores = bm25.scores(["belem"])

    assert scores[0] > 0
    assert scores[1] == scores[2] == 0


def test_fake_embedder_is_deterministic():
    embedder = FakeEmbedder(dim=64)

    first, second = embedder.embed(["Tour de Belém", "Tour de Belém"])

    assert first == second
    assert len(first) == 64


def test_search_finds_the_relevant_chunk(index):
    results = index.search("Belém", k=2)

    assert results[0].chunk.id == "Lisbonne-0000"


def test_search_filters_by_city(index):
    results = index.search("poisson", city="lisbonne")

    assert results[0].chunk.id == "Lisbonne-0001"
    assert all(r.chunk.city == "Lisbonne" for r in results)


def test_search_filters_by_section(index):
    results = index.search("poisson", section="Manger")

    assert {r.chunk.id for r in results} == {"Lisbonne-0001", "Rome-0000"}


def test_search_with_unknown_city_returns_nothing(index):
    assert index.search("poisson", city="Atlantide") == []


def test_save_and_load_give_same_results(index, tmp_path):
    index.save(tmp_path)

    loaded = HybridIndex.load(tmp_path, FakeEmbedder())

    assert [r.chunk.id for r in loaded.search("Colisée")] == [r.chunk.id for r in index.search("Colisée")]


def test_load_refuses_a_different_embedding_model(index, tmp_path):
    index.save(tmp_path)

    with pytest.raises(ValueError, match="Reconstruisez"):
        HybridIndex.load(tmp_path, FakeEmbedder(dim=128))
