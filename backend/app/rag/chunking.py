"""Découpage des guides Wikivoyage en morceaux (chunks) pour le RAG."""

import re
from dataclasses import dataclass

from app.rag.wikitext import Place, clean, extract_listings

# Titres de section : == Voir ==, === Musées ===, ...
HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    city: str
    section: str
    text: str
    source: str

    def text_with_context(self) -> str:
        """Texte précédé de sa ville et de sa section : c'est ce qu'on indexe pour la recherche."""
        return f"{self.city} — {self.section}\n{self.text}"


def split_sections(wikitext: str) -> list[tuple[str, str]]:
    """Découpe une page en (titre, contenu). Une sous-section donne un titre comme « Voir > Musées »."""
    sections = []
    path: list[str] = []
    title = "Introduction"
    last = 0
    for match in HEADING.finditer(wikitext):
        sections.append((title, wikitext[last : match.start()]))
        level = len(match.group(1)) - 2  # "==" -> 0, "===" -> 1, ...
        path = path[:level] + [clean(match.group(2))]
        title = " > ".join(path)
        last = match.end()
    sections.append((title, wikitext[last:]))
    return sections


def split_text(text: str, max_chars: int) -> list[str]:
    """Regroupe les lignes en morceaux d'au plus max_chars caractères (sans couper une phrase)."""
    pieces = []
    current = ""
    for unit in _units(text, max_chars):
        if current and len(current) + 1 + len(unit) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_page(page: dict, max_chars: int = 1000) -> tuple[list[Chunk], list[Place]]:
    """Transforme une page téléchargée (voir scripts/download_wikivoyage.py) en chunks et en lieux."""
    city = page["title"]
    chunks: list[Chunk] = []
    places: list[Place] = []

    for section, raw in split_sections(page["wikitext"]):
        raw, section_places = extract_listings(raw)
        for place in section_places:
            place.city = city
            place.section = section
        places.extend(section_places)

        for piece in split_text(clean(raw), max_chars):
            chunks.append(Chunk(
                id=f"{city}-{len(chunks):04d}",
                city=city,
                section=section,
                text=piece,
                source=page["source"],
            ))

    return chunks, places


def _units(text: str, max_chars: int):
    """Lignes non vides du texte ; une ligne trop longue est coupée en phrases."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            yield line
        else:
            yield from re.split(r"(?<=[.!?])\s+", line)
