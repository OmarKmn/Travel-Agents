"""Agent 2 — Chercheur : choisit quoi chercher, puis interroge le RAG (c'est le « RAG agentique »)."""

from app.agents.state import TripState, extract_json
from app.llm import LLM
from app.rag.index import HybridIndex

SYSTEM_PROMPT = """Tu es le Chercheur d'une équipe de planification de voyage.
À partir des centres d'intérêt du voyageur, propose jusqu'à 4 requêtes de recherche courtes
pour trouver dans un guide de voyage les lieux et activités adaptés.
Réponds UNIQUEMENT avec un tableau JSON de chaînes, par exemple :
["musées d'art", "restaurants de poisson", "quartiers historiques"]"""

MAX_QUERIES = 4
PASSAGES_PER_QUERY = 2
MAX_PASSAGES = 8
# Toujours chercher les classiques, même si le voyageur n'a rien précisé
DEFAULT_QUERY = "incontournables à voir"


def run_researcher(llm: LLM, index: HybridIndex, state: TripState) -> TripState:
    trip = state["trip"]
    answer = llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ville : {trip.city}\nCentres d'intérêt : {', '.join(trip.interests) or 'aucun précisé'}"},
    ])
    # Si la réponse est inutilisable, on se rabat sur les centres d'intérêt eux-mêmes
    queries = [*(_parse_queries(answer) or trip.interests[:MAX_QUERIES]), DEFAULT_QUERY]

    passages = []
    seen = set()
    for query in queries:
        for result in index.search(query, k=PASSAGES_PER_QUERY, city=trip.city):
            if result.chunk.id not in seen:
                seen.add(result.chunk.id)
                passages.append(result)
    passages = passages[:MAX_PASSAGES]

    if not passages:
        return {"queries": queries, "passages": [], "error": f"Je n'ai aucune information sur {trip.city}."}
    return {"queries": queries, "passages": passages}


def _parse_queries(answer: str) -> list[str]:
    try:
        data = extract_json(answer)
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [str(query).strip() for query in data if str(query).strip()][:MAX_QUERIES]
