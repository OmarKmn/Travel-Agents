"""Faux LLM : renvoie des réponses prédéfinies, sans modèle installé.

Utile pour développer la logique de l'application et pour les tests automatiques.
"""

from app.llm.base import Message


class FakeLLM:
    name = "fake"

    def __init__(self, responses: list[str] | None = None) -> None:
        # Réponses renvoyées dans l'ordre ; sinon on renvoie un écho du dernier message.
        self._responses = list(responses or [])
        # Conversations reçues, pour vérifier dans les tests ce que les agents envoient au modèle
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return f"[fake] Réponse simulée à : {messages[-1]['content']}"
