"""Transforme les pages téléchargées (data/raw) en chunks et en lieux (data/processed). 100 % local.

    python -m scripts.build_chunks
"""

import json
import sys
from dataclasses import asdict

from app.config import PROCESSED_DIR, RAW_DIR
from app.rag.chunking import chunk_page


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        print(f"Aucune page dans {RAW_DIR}. Lancez d'abord : python -m scripts.download_wikivoyage")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (
        open(PROCESSED_DIR / "chunks.jsonl", "w", encoding="utf-8") as chunks_file,
        open(PROCESSED_DIR / "places.jsonl", "w", encoding="utf-8") as places_file,
    ):
        for path in files:
            page = json.loads(path.read_text(encoding="utf-8"))
            chunks, places = chunk_page(page)
            for chunk in chunks:
                chunks_file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            for place in places:
                places_file.write(json.dumps(asdict(place), ensure_ascii=False) + "\n")
            print(f"✓ {page['title']} : {len(chunks)} chunks, {len(places)} lieux")


if __name__ == "__main__":
    main()
