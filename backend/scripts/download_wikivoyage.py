"""Télécharge les guides Wikivoyage (texte brut au format wiki) d'une liste de villes.

Seule étape du projet qui utilise Internet : les fichiers sont ensuite lus en local.

    python -m scripts.download_wikivoyage                 # villes par défaut
    python -m scripts.download_wikivoyage Lyon Nice       # villes choisies

Données : Wikivoyage, licence CC BY-SA 4.0 (https://fr.wikivoyage.org).
"""

import json
import sys
import time
from pathlib import Path

import httpx

from app.config import RAW_DIR

API_URL = "https://fr.wikivoyage.org/w/api.php"

# La politique de Wikimedia demande un User-Agent identifiant le projet.
USER_AGENT = "travel-agents/0.1 (projet open source d'apprentissage)"

DEFAULT_CITIES = [
    "Paris", "Lisbonne", "Barcelone", "Rome", "Amsterdam",
    "Berlin", "Prague", "Marrakech", "Istanbul", "Londres",
]


def fetch_wikitext(client: httpx.Client, title: str) -> dict | None:
    """Récupère le contenu wiki d'une page. Renvoie None si la page n'existe pas."""
    response = client.get(
        API_URL,
        params={
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": title,
            "redirects": 1,  # suit les redirections (ex. « Lisbon » -> « Lisbonne »)
            "format": "json",
            "formatversion": 2,
        },
    )
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]

    if page.get("missing"):
        return None

    return {
        "title": page["title"],
        "source": f"https://fr.wikivoyage.org/wiki/{page['title'].replace(' ', '_')}",
        "license": "CC BY-SA 4.0",
        "wikitext": page["revisions"][0]["slots"]["main"]["content"],
    }


def save_page(page: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{page['title']}.json"
    path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cities = sys.argv[1:] or DEFAULT_CITIES

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for city in cities:
            page = fetch_wikitext(client, city)
            if page is None:
                print(f"✗ {city} : page introuvable")
            else:
                path = save_page(page, RAW_DIR)
                print(f"✓ {city} -> {path.name} ({len(page['wikitext']):,} caractères)")
            time.sleep(1)  # rester poli avec les serveurs de Wikimedia


if __name__ == "__main__":
    main()
