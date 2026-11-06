"""Agent 4 — Critique : vérifie l'itinéraire et le renvoie au Rédacteur s'il faut le corriger.

Deux niveaux de contrôle :
1. des règles automatiques (nombre de jours, sources citées), fiables même avec un petit modèle ;
2. si elles passent, un avis du LLM sur la qualité (centres d'intérêt, rythme, lieux inventés).
"""

import re

from app.agents.state import TripState, extract_json, format_passages
from app.llm import LLM

SYSTEM_PROMPT = """Tu es le Critique d'une équipe de planification de voyage.
Vérifie l'itinéraire proposé :
- respecte-t-il les centres d'intérêt du voyageur ?
- chaque lieu vient-il bien des passages du guide fournis ?
- le rythme est-il réaliste (pas trop de visites par jour) ?
Réponds UNIQUEMENT avec un objet JSON :
{"approved": true ou false, "feedback": "remarques précises à corriger, vide si approuvé"}"""


def run_critic(llm: LLM, state: TripState) -> TripState:
    problems = check_rules(state)
    if problems:
        return {"approved": False, "feedback": "\n".join(problems)}

    trip = state["trip"]
    answer = llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Voyage : {trip.days} jours à {trip.city}\n"
            f"Centres d'intérêt : {', '.join(trip.interests) or 'aucun précisé'}\n\n"
            f"Passages du guide :\n{format_passages(state['passages'])}\n\n"
            f"Itinéraire proposé :\n{state['itinerary']}"
        )},
    ])
    try:
        verdict = extract_json(answer)
        return {"approved": bool(verdict["approved"]), "feedback": str(verdict.get("feedback", ""))}
    except (ValueError, KeyError, TypeError):
        # Avis illisible : on garde l'itinéraire, qui a déjà passé les règles automatiques
        return {"approved": True, "feedback": ""}


def check_rules(state: TripState) -> list[str]:
    """Problèmes détectables sans LLM. Liste vide si tout est correct."""
    itinerary = state["itinerary"]
    days = state["trip"].days
    passage_count = len(state["passages"])
    problems = []

    found_days = {int(day) for day in re.findall(r"jour\s+(\d+)", itinerary, re.IGNORECASE)}
    if found_days != set(range(1, days + 1)):
        problems.append(
            f"L'itinéraire doit contenir exactement les jours 1 à {days} (trouvés : {sorted(found_days) or 'aucun'})."
        )

    citations = {int(number) for number in re.findall(r"\[(\d+)\]", itinerary)}
    if not citations:
        problems.append("Cite la source de chaque lieu avec son numéro entre crochets, par exemple [2].")
    invalid = sorted(number for number in citations if not 1 <= number <= passage_count)
    if invalid:
        problems.append(f"Ces sources n'existent pas : {invalid}. Utilise uniquement les numéros 1 à {passage_count}.")

    return problems
