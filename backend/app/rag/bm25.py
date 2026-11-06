"""Recherche par mots-clés avec l'algorithme BM25 (celui des moteurs de recherche classiques).

Un passage obtient un score élevé s'il contient souvent les mots de la question,
surtout des mots rares dans l'ensemble des passages (ex. « Belém » plutôt que « ville »).
"""

import math
from collections import Counter


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1  # à quel point répéter un mot augmente le score
        self.b = b  # à quel point on pénalise les passages longs
        self.counts = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = sum(self.lengths) / len(documents) if documents else 0.0

        # Rareté de chaque mot : plus il apparaît dans peu de passages, plus il compte
        n = len(documents)
        doc_freq = Counter(word for doc in documents for word in set(doc))
        self.idf = {word: math.log(1 + (n - df + 0.5) / (df + 0.5)) for word, df in doc_freq.items()}

    def scores(self, query: list[str]) -> list[float]:
        """Score de chaque passage pour la question (0 si aucun mot en commun)."""
        results = []
        for counts, length in zip(self.counts, self.lengths):
            score = 0.0
            for word in query:
                tf = counts.get(word, 0)
                if tf:
                    norm = 1 - self.b + self.b * length / self.avg_length
                    score += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            results.append(score)
        return results
