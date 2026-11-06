"""Graphe des lieux : chaque lieu est un nœud, deux lieux proches à pied sont reliés par une arête.

C'est la partie « graphe » du RAG : après la recherche hybride, on suit les arêtes
pour ajouter des lieux voisins de ceux trouvés (expansion par le graphe),
puis on regroupe les lieux par jour selon leur proximité.
"""

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

from app.rag.text import normalize
from app.rag.wikitext import Place

NEAR_KM = 1.5  # environ 20 minutes à pied
EARTH_RADIUS_KM = 6371.0
MIN_NAME_LENGTH = 4  # un nom trop court (« Bar ») serait trouvé dans n'importe quel texte


def distance_km(a: Place, b: Place) -> float:
    """Distance à vol d'oiseau entre deux lieux (formule de haversine)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


class PlaceGraph:
    def __init__(self, places: list[Place], near_km: float = NEAR_KM) -> None:
        self.graph = nx.Graph()
        self._by_city: dict[str, list[Place]] = defaultdict(list)

        # Seuls les lieux avec des coordonnées peuvent être placés dans le graphe
        for place in places:
            key = _key(place)
            if place.lat is None or place.lon is None or key in self.graph:
                continue
            self.graph.add_node(key, place=place)
            self._by_city[place.city.lower()].append(place)

        # On compare les lieux d'une même ville deux à deux : suffisant pour quelques milliers de lieux
        for city_places in self._by_city.values():
            for a, b in combinations(city_places, 2):
                km = distance_km(a, b)
                if km <= near_km:
                    self.graph.add_edge(_key(a), _key(b), km=km)

    def places_in(self, city: str, text: str) -> list[Place]:
        """Lieux de la ville dont le nom apparaît dans le texte (sans tenir compte des accents)."""
        text = normalize(text)
        return [
            place for place in self._by_city.get(city.lower(), [])
            if len(name := normalize(place.name)) >= MIN_NAME_LENGTH and name in text
        ]

    def neighbors(self, place: Place) -> list[tuple[Place, float]]:
        """Lieux proches à pied, du plus proche au plus lointain, avec leur distance en km."""
        key = _key(place)
        if key not in self.graph:
            return []
        result = [(self.graph.nodes[other]["place"], edge["km"]) for other, edge in self.graph[key].items()]
        return sorted(result, key=lambda item: item[1])


def group_by_proximity(places: list[Place], groups: int) -> list[list[Place]]:
    """Répartit les lieux en `groups` zones géographiques (au plus).

    Méthode : on relie tous les lieux, on garde l'arbre couvrant minimal (le réseau le plus court
    qui relie tout le monde), puis on coupe ses arêtes les plus longues : il reste des groupes de lieux proches.
    """
    located = [place for place in places if place.lat is not None and place.lon is not None]
    if not located:
        return []

    complete = nx.Graph()
    complete.add_nodes_from(range(len(located)))
    for i, j in combinations(range(len(located)), 2):
        complete.add_edge(i, j, km=distance_km(located[i], located[j]))

    tree = nx.minimum_spanning_tree(complete, weight="km")
    longest = sorted(tree.edges(data="km"), key=lambda edge: edge[2], reverse=True)
    tree.remove_edges_from((i, j) for i, j, _ in longest[: min(groups, len(located)) - 1])

    zones = sorted(sorted(component) for component in nx.connected_components(tree))
    return [[located[i] for i in zone] for zone in zones]


def load_places(path: Path) -> list[Place]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Place(**json.loads(line)) for line in lines if line]


def _key(place: Place) -> str:
    return f"{place.city}/{place.name}".lower()
