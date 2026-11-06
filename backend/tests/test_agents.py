"""Tests des agents et de leur enchaînement, avec un faux LLM aux réponses scénarisées.

Le FakeLLM renvoie ses réponses dans l'ordre des appels : analyste, chercheur, rédacteur, critique...
(le Critique n'appelle pas le LLM quand les règles automatiques échouent).
"""

import pytest

from app.agents.critic import check_rules
from app.agents.graph import build_graph
from app.agents.researcher import run_researcher
from app.agents.state import TripRequest, extract_json
from app.llm import FakeLLM
from app.rag.chunking import Chunk
from app.rag.embeddings import FakeEmbedder
from app.rag.index import HybridIndex

CHUNKS = [
    Chunk("Lisbonne-0000", "Lisbonne", "Voir", "- Tour de Belém : tour manuéline au bord du Tage.", "src"),
    Chunk("Lisbonne-0001", "Lisbonne", "Manger", "Goûtez les pastéis de nata et le poisson grillé.", "src"),
    Chunk("Rome-0000", "Rome", "Voir", "- Colisée : amphithéâtre antique.", "src"),
]

ANALYST_OK = '{"city": "Lisbonne", "days": 2, "interests": ["cuisine"]}'
RESEARCHER_OK = '["poisson grillé", "pastéis de nata"]'
GOOD_ITINERARY = "Jour 1 : Tour de Belém [1]\nJour 2 : pastéis de nata [2]"
BAD_ITINERARY = "Jour 1 : Tour de Belém [1]"  # il manque le jour 2
APPROVED = '{"approved": true, "feedback": ""}'


@pytest.fixture
def index() -> HybridIndex:
    return HybridIndex.build(CHUNKS, FakeEmbedder())


def run(index, responses: list[str]) -> tuple[dict, FakeLLM]:
    llm = FakeLLM(responses=responses)
    state = build_graph(llm, index).invoke({"request": "2 jours à Lisbonne, j'aime la cuisine"})
    return state, llm


def test_extract_json_ignores_surrounding_text():
    assert extract_json('Voici :\n```json\n{"city": "Rome"}\n```') == {"city": "Rome"}


def test_happy_path(index):
    state, llm = run(index, [ANALYST_OK, RESEARCHER_OK, GOOD_ITINERARY, APPROVED])

    assert state["trip"] == TripRequest(city="Lisbonne", days=2, interests=["cuisine"])
    assert all(r.chunk.city == "Lisbonne" for r in state["passages"])
    assert state["itinerary"] == GOOD_ITINERARY
    assert state["approved"] is True
    assert state["drafts"] == 1
    assert len(llm.calls) == 4


def test_critic_rules_send_itinerary_back_to_writer(index):
    state, llm = run(index, [ANALYST_OK, RESEARCHER_OK, BAD_ITINERARY, GOOD_ITINERARY, APPROVED])

    assert state["approved"] is True
    assert state["drafts"] == 2
    # La 2e demande au Rédacteur contient la remarque du Critique
    second_writer_prompt = llm.calls[3][-1]["content"]
    assert "exactement les jours 1 à 2" in second_writer_prompt


def test_critic_llm_feedback_is_passed_to_writer(index):
    rejected = '{"approved": false, "feedback": "Ajoute un restaurant le jour 1."}'
    state, llm = run(index, [ANALYST_OK, RESEARCHER_OK, GOOD_ITINERARY, rejected, GOOD_ITINERARY, APPROVED])

    assert state["approved"] is True
    assert state["drafts"] == 2
    assert "Ajoute un restaurant le jour 1." in llm.calls[4][-1]["content"]


def test_stops_after_max_revisions(index):
    state, _ = run(index, [ANALYST_OK, RESEARCHER_OK, BAD_ITINERARY, BAD_ITINERARY, BAD_ITINERARY])

    assert state["approved"] is False
    assert state["drafts"] == 3  # 1 version + 2 corrections


def test_unknown_city_stops_early(index):
    state, llm = run(index, ['{"city": "Atlantide", "days": 2, "interests": []}', '["plages"]'])

    assert state["error"] == "Je n'ai aucune information sur Atlantide."
    assert "itinerary" not in state
    assert len(llm.calls) == 2


def test_unclear_request_stops_early(index):
    state, llm = run(index, ["Je ne sais pas."])

    assert state.get("trip") is None
    assert "Précisez la ville" in state["error"]
    assert len(llm.calls) == 1


def test_researcher_falls_back_to_interests(index):
    llm = FakeLLM(responses=["réponse sans JSON"])
    trip = TripRequest(city="Lisbonne", days=2, interests=["poisson"])

    result = run_researcher(llm, index, {"trip": trip})

    assert result["queries"] == ["poisson", "incontournables à voir"]
    assert result["passages"]


def test_check_rules_detects_invalid_citations():
    state = {
        "itinerary": "Jour 1 : Belém [1]\nJour 2 : Colisée [7]",
        "trip": TripRequest(city="Lisbonne", days=2),
        "passages": ["p1", "p2"],
    }

    problems = check_rules(state)

    assert problems == ["Ces sources n'existent pas : [7]. Utilise uniquement les numéros 1 à 2."]
