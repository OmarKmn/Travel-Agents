"""Client pour un modèle servi en local par Ollama (https://ollama.com)."""

import httpx

from app.llm.base import Message


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float = 120.0) -> None:
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, messages: list[Message]) -> str:
        response = self._client.post(
            "/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
