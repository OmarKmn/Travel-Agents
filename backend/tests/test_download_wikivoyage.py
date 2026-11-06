"""Tests du script de téléchargement, sans connexion Internet (réponses simulées)."""

import json

import httpx

from scripts.download_wikivoyage import fetch_wikitext, save_page


def make_client(api_response: dict) -> httpx.Client:
    """Client HTTP qui renvoie toujours la même réponse, sans appel réseau."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=api_response))
    return httpx.Client(transport=transport)


def test_fetch_existing_page():
    client = make_client({
        "query": {"pages": [{
            "title": "Lisbonne",
            "revisions": [{"slots": {"main": {"content": "== Voir ==\nTour de Belém"}}}],
        }]}
    })

    page = fetch_wikitext(client, "Lisbonne")

    assert page["title"] == "Lisbonne"
    assert page["wikitext"] == "== Voir ==\nTour de Belém"
    assert page["source"] == "https://fr.wikivoyage.org/wiki/Lisbonne"


def test_fetch_missing_page_returns_none():
    client = make_client({"query": {"pages": [{"title": "Atlantide", "missing": True}]}})

    assert fetch_wikitext(client, "Atlantide") is None


def test_save_page_writes_utf8_json(tmp_path):
    page = {"title": "Rome", "source": "...", "license": "CC BY-SA 4.0", "wikitext": "Colisée"}

    path = save_page(page, tmp_path)

    assert path.name == "Rome.json"
    assert json.loads(path.read_text(encoding="utf-8"))["wikitext"] == "Colisée"
