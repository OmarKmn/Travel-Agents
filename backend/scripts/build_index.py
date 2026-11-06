"""Construit l'index de recherche à partir des chunks (data/processed -> data/index). 100 % local.

    python -m scripts.build_index
"""

import sys

from app.config import INDEX_DIR, PROCESSED_DIR, get_settings
from app.rag.embeddings import create_embedder
from app.rag.index import HybridIndex, load_chunks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"{chunks_path} introuvable. Lancez d'abord : python -m scripts.build_chunks")
        return

    chunks = load_chunks(chunks_path)
    embedder = create_embedder(get_settings())
    print(f"Calcul des embeddings de {len(chunks)} chunks avec {embedder.name} ({embedder.model})...")

    HybridIndex.build(chunks, embedder).save(INDEX_DIR)
    print(f"✓ Index enregistré dans {INDEX_DIR}")


if __name__ == "__main__":
    main()
