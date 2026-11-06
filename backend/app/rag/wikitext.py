"""Nettoyage du wikitext (format source des pages Wikivoyage) et extraction des fiches de lieux."""

import html
import re
from dataclasses import dataclass

# Modèles de fiches de lieux sur fr.wikivoyage -> catégorie du lieu
LISTING_TYPES = {
    "voir": "voir",
    "faire": "faire",
    "acheter": "acheter",
    "manger": "manger",
    "sortir": "sortir",
    "se loger": "se loger",
    "dormir": "se loger",
    "listing": "autre",
}

# Liens vers des images ou des catégories : à supprimer, pas du texte utile
NON_TEXT_LINK = re.compile(r"\s*(fichier|file|image|catégorie|category)\s*:", re.IGNORECASE)


@dataclass
class Place:
    name: str
    kind: str
    description: str = ""
    address: str = ""
    lat: float | None = None
    lon: float | None = None
    city: str = ""
    section: str = ""


def find_spans(text: str, open_tag: str, close_tag: str) -> list[tuple[int, int]]:
    """Positions (début, fin) des blocs de premier niveau, ex. {{...}} ou [[...]], en gérant l'imbrication."""
    spans = []
    depth = 0
    start = 0
    i = 0
    while i < len(text) - 1:
        pair = text[i : i + 2]
        if pair == open_tag:
            if depth == 0:
                start = i
            depth += 1
            i += 2
        elif pair == close_tag and depth > 0:
            depth -= 1
            i += 2
            if depth == 0:
                spans.append((start, i))
        else:
            i += 1
    return spans


def parse_template(template: str) -> tuple[str, dict[str, str]]:
    """Découpe '{{nom | clé=valeur | ...}}' en ('nom', {'clé': 'valeur', ...})."""
    inner = template[2:-2]
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    while i < len(inner):
        pair = inner[i : i + 2]
        if pair in ("{{", "[["):
            depth += 1
            current.append(pair)
            i += 2
            continue
        if pair in ("}}", "]]") and depth > 0:
            depth -= 1
            current.append(pair)
            i += 2
            continue
        # Un « | » ne sépare les paramètres qu'au premier niveau (pas dans un lien ou un modèle imbriqué)
        if inner[i] == "|" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(inner[i])
        i += 1
    parts.append("".join(current))

    name = parts[0].strip().lower()
    params = {}
    for position, part in enumerate(parts[1:], start=1):
        key, sep, value = part.partition("=")
        if sep:
            params[key.strip().lower()] = value.strip()
        else:
            params[str(position)] = part.strip()
    return name, params


def clean(text: str) -> str:
    """Transforme du wikitext en texte lisible."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = _remove_spans(text, "{{", "}}")  # modèles restants (bandeaux, prix, etc.)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)  # tableaux
    text = _replace_links(text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)  # [url texte] -> texte
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"'{2,}", "", text)  # gras / italique
    text = re.sub(r"<[^>]+>", "", text)  # balises HTML
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"^[*#]+\s*", "- ", text, flags=re.MULTILINE)  # listes
    text = re.sub(r"^[:;]+\s*", "", text, flags=re.MULTILINE)  # indentations
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_listings(text: str) -> tuple[str, list[Place]]:
    """Extrait les fiches de lieux et les remplace dans le texte par « Nom : description »."""
    places = []
    output = []
    last = 0
    for start, end in find_spans(text, "{{", "}}"):
        place = _listing_to_place(*parse_template(text[start:end]))
        if place is None:
            continue
        places.append(place)
        output.append(text[last:start])
        output.append(f"{place.name} : {place.description}" if place.description else place.name)
        last = end
    output.append(text[last:])
    return "".join(output), places


def _listing_to_place(name: str, params: dict[str, str]) -> Place | None:
    kind = LISTING_TYPES.get(name)
    if kind is None:
        return None
    if name == "listing":
        kind = params.get("type", kind).lower()

    place_name = clean(params.get("nom") or params.get("name") or "")
    if not place_name:
        return None

    return Place(
        name=place_name,
        kind=kind,
        description=clean(params.get("description", "")),
        address=clean(params.get("adresse", "")),
        lat=_to_float(params.get("lat") or params.get("latitude")),
        lon=_to_float(params.get("long") or params.get("longitude") or params.get("lon")),
    )


def _to_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _remove_spans(text: str, open_tag: str, close_tag: str) -> str:
    output = []
    last = 0
    for start, end in find_spans(text, open_tag, close_tag):
        output.append(text[last:start])
        last = end
    output.append(text[last:])
    return "".join(output)


def _replace_links(text: str) -> str:
    """[[cible|texte]] -> texte, [[cible]] -> cible ; supprime les images et catégories."""
    output = []
    last = 0
    for start, end in find_spans(text, "[[", "]]"):
        output.append(text[last:start])
        target, _, label = text[start + 2 : end - 2].partition("|")
        if not NON_TEXT_LINK.match(target):
            output.append(label or target.split(":")[-1])  # [[w:Belém]] -> Belém
        last = end
    output.append(text[last:])
    return "".join(output)
