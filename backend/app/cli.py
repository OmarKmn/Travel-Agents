"""Planificateur de voyage en ligne de commande, sans serveur.

    python -m app.cli "3 jours à Lisbonne, j'aime les musées et la cuisine"
    python -m app.cli          # mode conversation (taper "quit" pour sortir)
"""

import sys

from app.agents.graph import build_graph
from app.config import INDEX_DIR, PROCESSED_DIR, get_settings
from app.llm import create_llm
from app.rag.embeddings import create_embedder
from app.rag.index import HybridIndex
from app.rag.place_graph import PlaceGraph, load_places


def main() -> None:
    # Afficher correctement les accents dans le terminal Windows
    sys.stdout.reconfigure(encoding="utf-8")

    if not (INDEX_DIR / "meta.json").exists():
        print(
            "Aucun index de recherche. Préparez d'abord les données :\n"
            "  python -m scripts.download_wikivoyage\n"
            "  python -m scripts.build_chunks\n"
            "  python -m scripts.build_index"
        )
        return

    settings = get_settings()
    llm = create_llm(settings)
    index = HybridIndex.load(INDEX_DIR, create_embedder(settings))
    places_path = PROCESSED_DIR / "places.jsonl"
    place_graph = PlaceGraph(load_places(places_path)) if places_path.exists() else None
    graph = build_graph(llm, index, place_graph)
    print(f"LLM : {llm.name} ({settings.llm_model})\n")

    # Une demande passée directement en argument
    if len(sys.argv) > 1:
        plan(graph, " ".join(sys.argv[1:]))
        return

    # Sinon, mode conversation
    while True:
        try:
            request = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if request.lower() in {"quit", "exit", "q"}:
            break
        if request:
            plan(graph, request)
            print()


def plan(graph, request: str) -> None:
    """Lance les agents en affichant chaque étape, puis l'itinéraire et ses sources."""
    state = {}
    for update in graph.stream({"request": request}, stream_mode="updates"):
        for node, changes in update.items():
            state.update(changes)
            print(describe_step(node, changes))

    if state.get("error"):
        print(f"\n⚠️  {state['error']}")
        return

    print(f"\n{state['itinerary']}\n\nSources :")
    for i, result in enumerate(state["passages"], start=1):
        print(f"[{i}] {result.chunk.city} — {result.chunk.section} ({result.chunk.source})")


def describe_step(node: str, changes: dict) -> str:
    if node == "analyste":
        trip = changes.get("trip")
        if trip is None:
            return "🧭 Analyste : demande incomprise"
        return f"🧭 Analyste : {trip.city}, {trip.days} jours, intérêts : {', '.join(trip.interests) or 'aucun'}"
    if node == "chercheur":
        return f"🔎 Chercheur : {len(changes['passages'])} passages trouvés (requêtes : {', '.join(changes['queries'])})"
    if node == "logistique":
        located = sum(len(zone) for zone in changes["day_plan"])
        return (
            f"🗺️  Logistique : {located} lieux localisés en {len(changes['day_plan'])} zones, "
            f"{len(changes['nearby'])} lieux proches ajoutés"
        )
    if node == "redacteur":
        return f"✍️  Rédacteur : version {changes['drafts']} rédigée"
    if node == "critique":
        if changes["approved"]:
            return "✅ Critique : itinéraire approuvé"
        return f"🔁 Critique : à corriger — {changes['feedback']}"
    return node


if __name__ == "__main__":
    main()
