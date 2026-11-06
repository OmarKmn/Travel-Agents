"""Interface commune à tous les fournisseurs de LLM.

Les agents ne dépendent que de cette interface : on peut ainsi passer
de Fake à Ollama (ou à un autre fournisseur) sans toucher à leur code.
"""

from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLM(Protocol):
    name: str

    def chat(self, messages: list[Message]) -> str:
        """Envoie une conversation au modèle et renvoie sa réponse texte."""
        ...
