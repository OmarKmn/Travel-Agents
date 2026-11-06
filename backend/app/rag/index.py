"""Index de recherche hybride : vecteurs (sens) + BM25 (mots-clés) + filtres (ville, section).

Les deux classements sont fusionnés avec la « Reciprocal Rank Fusion » (RRF) : on additionne
1 / (60 + rang) obtenu dans chaque classement. Un passage bien classé par les deux méthodes
remonte en tête, sans avoir à comparer des scores d'échelles différentes.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.rag.bm25 import BM25
from app.rag.chunking import Chunk
from app.rag.embeddings import Embedder
from app.rag.text import tokenize

RRF_K = 60


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors, embedder: Embedder) -> None:
        self.chunks = chunks
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self.embedder = embedder
        self.bm25 = BM25([tokenize(chunk.text_with_context()) for chunk in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "HybridIndex":
        """Calcule les embeddings de tous les chunks (l'étape la plus longue avec un vrai modèle)."""
        vectors = embedder.embed([chunk.text_with_context() for chunk in chunks])
        return cls(chunks, vectors, embedder)

    def search(
        self, query: str, k: int = 5, city: str | None = None, section: str | None = None
    ) -> list[SearchResult]:
        """Les k passages les plus pertinents, éventuellement limités à une ville et/ou une section."""
        candidates = [
            i for i, chunk in enumerate(self.chunks)
            if (city is None or chunk.city.lower() == city.lower())
            and (section is None or chunk.section.lower().startswith(section.lower()))
        ]
        if not candidates:
            return []

        # 1. Recherche par le sens : similarité cosinus entre la question et chaque passage
        query_vector = _normalize(np.asarray(self.embedder.embed([query]), dtype=np.float32))[0]
        vector_scores = self.vectors[candidates] @ query_vector

        # 2. Recherche par mots-clés
        all_bm25 = self.bm25.scores(tokenize(query))
        bm25_scores = np.array([all_bm25[i] for i in candidates])

        # 3. Fusion des deux classements (un passage sans aucun mot en commun n'est pas classé par BM25)
        rankings = [
            np.argsort(-vector_scores, kind="stable"),
            [j for j in np.argsort(-bm25_scores, kind="stable") if bm25_scores[j] > 0],
        ]
        fused = np.zeros(len(candidates))
        for ranking in rankings:
            for rank, j in enumerate(ranking, start=1):
                fused[j] += 1.0 / (RRF_K + rank)

        best = np.argsort(-fused, kind="stable")[:k]
        return [SearchResult(self.chunks[candidates[j]], float(fused[j])) for j in best]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        save_chunks(self.chunks, directory / "chunks.jsonl")
        meta = {"embedder": self.embedder.name, "model": self.embedder.model, "count": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "HybridIndex":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        # Des vecteurs d'un modèle ne sont pas comparables avec ceux d'un autre
        if (meta["embedder"], meta["model"]) != (embedder.name, embedder.model):
            raise ValueError(
                f"Index construit avec {meta['embedder']} ({meta['model']}), mais la configuration utilise "
                f"{embedder.name} ({embedder.model}). Reconstruisez l'index : python -m scripts.build_index"
            )
        chunks = load_chunks(directory / "chunks.jsonl")
        return cls(chunks, np.load(directory / "vectors.npy"), embedder)


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk(**json.loads(line)) for line in lines if line]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Ramène chaque vecteur à une longueur de 1 : le produit scalaire devient la similarité cosinus."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
