"""Agent 5 — Logistique : localise les lieux trouvés, ajoute des lieux voisins et les regroupe par jour.

Agent sans LLM : les calculs géographiques sont plus fiables et plus rapides en code.
"""

from app.agents.state import TripState
from app.rag.chunking import Chunk
from app.rag.index import HybridIndex, SearchResult
from app.rag.place_graph import PlaceGraph, group_by_proximity
from app.rag.text import normalize
from app.rag.wikitext import Place

MAX_NEARBY = 4


def run_logistics(index: HybridIndex, place_graph: PlaceGraph, state: TripState) -> TripState:
    trip = state["trip"]
    passages = list(state["passages"])
    found = place_graph.places_in(trip.city, "\n".join(result.chunk.text for result in passages))

    # Expansion par le graphe : on ajoute des lieux voisins, avec le passage du guide qui les décrit
    # (pour que le Rédacteur puisse les citer comme les autres)
    known = {place.name for place in found}
    seen_chunks = {result.chunk.id for result in passages}
    nearby: list[Place] = []
    for place in found:
        for neighbor, _ in place_graph.neighbors(place):
            if len(nearby) >= MAX_NEARBY:
                break
            if neighbor.name in known:
                continue
            chunk = _chunk_mentioning(index, neighbor)
            if chunk is None:
                continue
            known.add(neighbor.name)
            nearby.append(neighbor)
            if chunk.id not in seen_chunks:
                seen_chunks.add(chunk.id)
                passages.append(SearchResult(chunk, score=0.0))

    zones = group_by_proximity(found + nearby, trip.days)
    return {
        "passages": passages,
        "nearby": [place.name for place in nearby],
        "day_plan": [[place.name for place in zone] for zone in zones],
    }


def _chunk_mentioning(index: HybridIndex, place: Place) -> Chunk | None:
    """Premier passage du guide de la ville qui mentionne le lieu."""
    name = normalize(place.name)
    for chunk in index.chunks:
        if chunk.city.lower() == place.city.lower() and name in normalize(chunk.text):
            return chunk
    return None
