"""Tests du nettoyage et du découpage, sur un faux guide Wikivoyage."""

from app.rag.chunking import chunk_page, split_sections, split_text
from app.rag.wikitext import clean, extract_listings, parse_template

SAMPLE = """'''Lisbonne''' est la capitale du [[Portugal]].<ref>source</ref>
[[Fichier:Lisboa.jpg|vignette|Vue de [[Lisbonne]]]]

== Voir ==
* {{voir | nom=Tour de Belém | lat=38.6916 | long=-9.2160 | adresse=Av. Brasília | description=Tour [[manuélin|manuéline]] du '''XVIe siècle'''. {{Prix|6|€}} }}
* {{voir | nom=Mosteiro dos Jerónimos | lat= | long= | description=Monastère.}}

=== Musées ===
Le [https://example.org musée Gulbenkian] est incontournable.

== Manger ==
{{Avertissement|texte=bla}}
Goûtez les ''pastéis de nata''.
"""

PAGE = {"title": "Lisbonne", "source": "https://fr.wikivoyage.org/wiki/Lisbonne", "wikitext": SAMPLE}


def test_parse_template_ignores_pipes_inside_links():
    name, params = parse_template("{{voir | nom=Belém | description=Style [[manuélin|manuéline]]}}")

    assert name == "voir"
    assert params == {"nom": "Belém", "description": "Style [[manuélin|manuéline]]"}


def test_clean_removes_wiki_markup():
    text = "Tour [[manuélin|manuéline]] du '''XVIe siècle'''.<ref>x</ref> [[Fichier:a.jpg|Vue de [[Lisbonne]]]]"

    assert clean(text) == "Tour manuéline du XVIe siècle."


def test_clean_keeps_external_link_label():
    assert clean("Le [https://example.org musée Gulbenkian].") == "Le musée Gulbenkian."


def test_extract_listings():
    _, places = extract_listings(split_sections(SAMPLE)[1][1])

    belem, mosteiro = places
    assert belem.name == "Tour de Belém"
    assert belem.kind == "voir"
    assert belem.description == "Tour manuéline du XVIe siècle."
    assert (belem.lat, belem.lon) == (38.6916, -9.2160)
    assert belem.address == "Av. Brasília"
    assert mosteiro.lat is None  # coordonnées absentes


def test_split_sections_builds_section_paths():
    titles = [title for title, _ in split_sections(SAMPLE)]

    assert titles == ["Introduction", "Voir", "Voir > Musées", "Manger"]


def test_split_text_respects_max_chars():
    text = "\n".join(f"Ligne numéro {i} avec un peu de texte." for i in range(20))

    pieces = split_text(text, max_chars=120)

    assert len(pieces) > 1
    assert all(len(piece) <= 120 for piece in pieces)


def test_chunk_page():
    chunks, places = chunk_page(PAGE)

    assert [c.section for c in chunks] == ["Introduction", "Voir", "Voir > Musées", "Manger"]
    assert chunks[0].text == "Lisbonne est la capitale du Portugal."
    assert "- Tour de Belém : Tour manuéline du XVIe siècle." in chunks[1].text
    assert chunks[3].text == "Goûtez les pastéis de nata."
    assert chunks[3].text_with_context() == "Lisbonne — Manger\nGoûtez les pastéis de nata."

    # Plus aucune trace de syntaxe wiki
    for chunk in chunks:
        assert not any(marker in chunk.text for marker in ("{{", "[[", "'''", "<ref"))

    assert [p.city for p in places] == ["Lisbonne", "Lisbonne"]
    assert places[0].section == "Voir"
