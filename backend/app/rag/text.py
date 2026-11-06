"""Découpage d'un texte en mots (tokens) pour la recherche par mots-clés."""

import re
import unicodedata

# Mots trop fréquents pour aider à trouver un passage
STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "au", "aux", "pour",
    "par", "sur", "dans", "est", "ou", "qui", "que", "ce", "il", "se", "ne", "pas", "plus",
}


def normalize(text: str) -> str:
    """'Tour de Belém' -> 'tour de belem' (minuscules, sans accents) pour comparer des textes."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    """'Où voir la Tour de Belém ?' -> ['voir', 'tour', 'belem'] (minuscules, sans accents ni mots vides)."""
    return [word for word in re.findall(r"\w+", normalize(text)) if len(word) > 1 and word not in STOPWORDS]
