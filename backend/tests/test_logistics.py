"""Tests du graphe des lieux et de l'agent Logistique."""

import pytest

from app.agents.graph import build_graph
from app.agents.logistics import run_logistics
from app.agents.state import TripRequest
from app.llm import FakeLLM
from app.rag.chunking import Chunk
from app.rag.embeddings import FakeEmbedder
from app.rag.index import HybridIndex, SearchResult
from app.rag.place_graph import PlaceGraph, distance_km, group_by_proximity
from app.rag.wikitext import Place

# Belém et les Jerónimos sont à ~1 km l'un de l'autre, le Castelo à ~7 km
BELEM = Place("Tour de Belém", "voir", lat=38.6916, lon=-9.2160, city="Lisbonne")
JERONIMOS = Place("Mosteiro dos Jerónimos", "voir", lat=38.6979, lon=-9.2068, city="Lisbonne")
CASTELO = Place("Castelo de São Jorge", "voir", lat=38.7139, lon=-9.1335, city="Lisbonne")
NO_COORDS = Place("Musée sans adresse", "voir", city="Lisbonne")
# Mêmes coordonnées que Belém, mais dans une autre ville : ne doit pas lui être relié
ELSEWHERE = Place("Lieu ailleurs", "voir", lat=38.6916, lon=-9.2160, city="Rome")

PLACES = [BELEM, JERONIMOS, CASTELO, NO_COORDS, ELSEWHERE]

CHUNKS = [
    Chunk("Lisbonne-0000", "Lisbonne", "Voir", "- Tour de Belém : tour manuéline.", "src"),
    Chunk("Lisbonne-0001", "Lisbonne", "Voir", "- Mosteiro dos Jerónimos : monastère.", "src"),
    Chunk("Lisbonne-0002", "Lisbonne", "Voir", "- Castelo de São Jorge : château.", "src"),
]


@pytest.fixture
def index() -> HybridIndex:
    return HybridIndex.build(CHUNKS, FakeEmbedder())


@pytest.fixture
def place_graph() -> PlaceGraph:
    return PlaceGraph(PLACES)


def test_distance_paris_london():
    paris = Place("Paris", "autre", lat=48.8566, lon=2.3522)
    london = Place("Londres", "autre", lat=51.5074, lon=-0.1278)

    assert 330 < distance_km(paris, london) < 360


def test_graph_links_only_nearby_places_of_the_same_city(place_graph):
    neighbors = place_graph.neighbors(BELEM)

    assert [place.name for place, _ in neighbors] == ["Mosteiro dos Jerónimos"]
    assert neighbors[0][1] == pytest.approx(1.06, abs=0.05)
    assert place_graph.neighbors(NO_COORDS) == []  # sans coordonnées, pas dans le graphe


def test_places_in_ignores_accents(place_graph):
    found = place_graph.places_in("lisbonne", "Commencez par la tour de belem, puis le château.")

    assert found == [BELEM]


def test_group_by_proximity():
    zones = group_by_proximity([BELEM, CASTELO, JERONIMOS], groups=2)

    assert zones == [[BELEM, JERONIMOS], [CASTELO]]


def test_group_by_proximity_with_more_days_than_places():
    assert len(group_by_proximity([BELEM, CASTELO], groups=5)) == 2


def test_logistics_adds_nearby_place_and_its_passage(index, place_graph):
    state = {
        "trip": TripRequest(city="Lisbonne", days=2),
        "passages": [SearchResult(CHUNKS[0], 1.0), SearchResult(CHUNKS[2], 0.5)],  # Belém et Castelo
    }

    result = run_logistics(index, place_graph, state)

    assert result["nearby"] == ["Mosteiro dos Jerónimos"]
    assert [r.chunk.id for r in result["passages"]] == ["Lisbonne-0000", "Lisbonne-0002", "Lisbonne-0001"]
    assert result["day_plan"] == [
        ["Tour de Belém", "Mosteiro dos Jerónimos"],
        ["Castelo de São Jorge"],
    ]


def test_writer_receives_day_plan(index, place_graph):
    llm = FakeLLM(responses=[
        '{"city": "Lisbonne", "days": 2, "interests": ["monuments"]}',
        '["monuments"]',
        "Jour 1 : Belém [1]\nJour 2 : Castelo [2]",
        '{"approved": true, "feedback": ""}',
    ])

    state = build_graph(llm, index, place_graph).invoke({"request": "2 jours à Lisbonne"})

    assert state["approved"] is True
    writer_prompt = llm.calls[2][-1]["content"]
    assert "Jour 1 : Tour de Belém, Mosteiro dos Jerónimos" in writer_prompt
    assert "Jour 2 : Castelo de São Jorge" in writer_prompt
