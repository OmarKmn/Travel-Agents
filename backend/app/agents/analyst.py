"""Agent 1 — Analyste : transforme la demande en texte libre en demande structurée."""

from app.agents.state import TripRequest, TripState, extract_json
from app.llm import LLM

SYSTEM_PROMPT = """Tu es l'Analyste d'une équipe de planification de voyage.
Extrais de la demande de l'utilisateur :
- city : la ville de destination (nom en français)
- days : le nombre de jours (3 si non précisé)
- interests : les centres d'intérêt (liste vide si aucun)
Réponds UNIQUEMENT avec un objet JSON, par exemple :
{"city": "Lisbonne", "days": 3, "interests": ["musées", "cuisine"]}"""


def run_analyst(llm: LLM, state: TripState) -> TripState:
    answer = llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["request"]},
    ])
    try:
        return {"trip": TripRequest.model_validate(extract_json(answer))}
    except ValueError:  # JSON absent ou invalide (les erreurs de validation Pydantic en héritent)
        return {"trip": None, "error": "Je n'ai pas compris la demande. Précisez la ville et le nombre de jours."}
