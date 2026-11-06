"""Transformation de textes en vecteurs (embeddings) pour la recherche par le sens.

Même principe que app/llm : une interface commune, un faux modèle pour développer, Ollama pour de vrai.
"""

import hashlib
from typing import Protocol

import httpx

from app.config import Settings
from app.rag.text import tokenize


class Embedder(Protocol):
    name: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Renvoie un vecteur par texte."""
        ...


class FakeEmbedder:
    """Vecteurs « sac de mots » obtenus par hachage.

    Ils ne comprennent pas le sens (« poisson » et « fruits de mer » ne sont pas proches),
    mais deux textes qui partagent des mots le sont : suffisant pour développer et tester.
    """

    name = "fake"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model = f"hash-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in tokenize(text):
                bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "big") % self.dim
                vector[bucket] += 1.0
            vectors.append(vector)
        return vectors


class OllamaEmbedder:
    """Modèle d'embeddings servi en local par Ollama (ex. bge-m3, multilingue)."""

    name = "ollama"

    def __init__(self, model: str, base_url: str, batch_size: int = 32, timeout: float = 300.0) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            response = self._client.post(
                "/api/embed",
                json={"model": self.model, "input": texts[start : start + self.batch_size]},
            )
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return vectors


def create_embedder(settings: Settings) -> Embedder:
    """Crée le modèle d'embeddings choisi dans la configuration (EMBEDDING_PROVIDER)."""
    if settings.embedding_provider == "ollama":
        return OllamaEmbedder(model=settings.embedding_model, base_url=settings.ollama_base_url)
    return FakeEmbedder()
