"""Teste la recherche hybride depuis le terminal.

    python -m scripts.search "où manger du poisson"
    python -m scripts.search "musées" --city Lisbonne --section Voir -k 3
"""

import argparse
import sys

from app.config import INDEX_DIR, get_settings
from app.rag.embeddings import create_embedder
from app.rag.index import HybridIndex


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Recherche dans les guides Wikivoyage indexés")
    parser.add_argument("query", help="la question")
    parser.add_argument("--city", help="limiter à une ville (ex. Lisbonne)")
    parser.add_argument("--section", help="limiter à une section (ex. Manger)")
    parser.add_argument("-k", type=int, default=5, help="nombre de résultats")
    args = parser.parse_args()

    if not (INDEX_DIR / "meta.json").exists():
        print(f"Aucun index dans {INDEX_DIR}. Lancez d'abord : python -m scripts.build_index")
        return

    index = HybridIndex.load(INDEX_DIR, create_embedder(get_settings()))
    for result in index.search(args.query, k=args.k, city=args.city, section=args.section):
        chunk = result.chunk
        print(f"[{result.score:.4f}] {chunk.city} — {chunk.section}")
        print(f"{chunk.text[:300]}\n")


if __name__ == "__main__":
    main()
