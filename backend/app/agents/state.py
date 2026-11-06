"""État partagé entre les agents : chaque agent lit l'état et renvoie les champs qu'il met à jour."""

import json
import re
from typing import TypedDict

from pydantic import BaseModel, Field

from app.rag.index import SearchResult


class TripRequest(BaseModel):
    """Demande de voyage structurée, produite par l'Analyste."""

    city: str
    days: int = Field(default=3, ge=1, le=14)
    interests: list[str] = []


class TripState(TypedDict, total=False):
    request: str  # demande de l'utilisateur, en texte libre
    trip: TripRequest | None  # Analyste
    queries: list[str]  # Chercheur : requêtes envoyées au RAG
    passages: list[SearchResult]  # Chercheur : passages trouvés (complétés par la Logistique)
    nearby: list[str]  # Logistique : lieux voisins ajoutés grâce au graphe
    day_plan: list[list[str]]  # Logistique : noms des lieux regroupés par zone géographique
    itinerary: str  # Rédacteur
    drafts: int  # Rédacteur : nombre de versions écrites
    approved: bool  # Critique
    feedback: str  # Critique : remarques à corriger
    error: str  # arrêt anticipé (demande incomprise, ville inconnue...)


def extract_json(text: str):
    """Extrait l'objet ou le tableau JSON d'une réponse de LLM (souvent entouré de texte ou de ```)."""
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Aucun JSON dans la réponse : {text[:100]!r}")
    return json.loads(match.group(0))


def format_passages(passages: list[SearchResult]) -> str:
    """Passages numérotés [1], [2]... pour que le Rédacteur puisse les citer."""
    return "\n\n".join(
        f"[{i}] {result.chunk.section}\n{result.chunk.text}" for i, result in enumerate(passages, start=1)
    )
