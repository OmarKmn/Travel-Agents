"""Agent 3 — Rédacteur : écrit l'itinéraire à partir des passages, et le corrige selon le Critique."""

from app.agents.state import TripState, format_passages
from app.llm import LLM

SYSTEM_PROMPT = """Tu es le Rédacteur d'une équipe de planification de voyage.
Rédige en français un itinéraire jour par jour.
Règles :
- Un titre par jour : « Jour 1 », « Jour 2 »... exactement le nombre de jours demandé.
- Utilise UNIQUEMENT les lieux présents dans les passages du guide fournis.
- Cite la source de chaque lieu avec son numéro entre crochets, par exemple [2].
- Tiens compte des centres d'intérêt du voyageur et garde un rythme réaliste."""


def run_writer(llm: LLM, state: TripState) -> TripState:
    trip = state["trip"]
    prompt = (
        f"Voyage : {trip.days} jours à {trip.city}\n"
        f"Centres d'intérêt : {', '.join(trip.interests) or 'aucun précisé'}\n\n"
        f"Passages du guide :\n{format_passages(state['passages'])}"
    )
    if state.get("day_plan"):
        zones = "\n".join(f"Jour {i} : {', '.join(names)}" for i, names in enumerate(state["day_plan"], start=1))
        prompt += f"\n\nRegroupement géographique proposé (lieux proches le même jour) :\n{zones}"
    # Version corrigée : on renvoie la version précédente et les remarques du Critique
    if state.get("feedback"):
        prompt += (
            f"\n\nTa version précédente :\n{state['itinerary']}\n\n"
            f"Remarques du Critique à corriger :\n{state['feedback']}"
        )

    itinerary = llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return {"itinerary": itinerary, "drafts": state.get("drafts", 0) + 1}
