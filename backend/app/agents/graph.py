"""Enchaînement des agents avec LangGraph.

    START → analyste → chercheur → logistique → redacteur → critique → END
                │           │                       ▲           │
                └─ erreur ──┴─→ END                 └─ à corriger (2 fois au maximum)
"""

from langgraph.graph import END, START, StateGraph

from app.agents.analyst import run_analyst
from app.agents.critic import run_critic
from app.agents.logistics import run_logistics
from app.agents.researcher import run_researcher
from app.agents.state import TripState
from app.agents.writer import run_writer
from app.llm import LLM
from app.rag.index import HybridIndex
from app.rag.place_graph import PlaceGraph

MAX_REVISIONS = 2


def build_graph(llm: LLM, index: HybridIndex, place_graph: PlaceGraph | None = None):
    # Sans lieux (places.jsonl absent), la Logistique laisse simplement passer les passages
    place_graph = place_graph or PlaceGraph([])
    graph = StateGraph(TripState)

    # Nœuds : chaque agent reçoit l'état et renvoie les champs qu'il modifie
    graph.add_node("analyste", lambda state: run_analyst(llm, state))
    graph.add_node("chercheur", lambda state: run_researcher(llm, index, state))
    graph.add_node("logistique", lambda state: run_logistics(index, place_graph, state))
    graph.add_node("redacteur", lambda state: run_writer(llm, state))
    graph.add_node("critique", lambda state: run_critic(llm, state))

    # Arêtes : l'ordre de passage, avec des embranchements selon l'état
    graph.add_edge(START, "analyste")
    graph.add_conditional_edges("analyste", _stop_on_error("chercheur"))
    graph.add_conditional_edges("chercheur", _stop_on_error("logistique"))
    graph.add_edge("logistique", "redacteur")
    graph.add_edge("redacteur", "critique")
    graph.add_conditional_edges("critique", _after_critic)

    return graph.compile()


def _stop_on_error(next_node: str):
    return lambda state: END if state.get("error") else next_node


def _after_critic(state: TripState) -> str:
    # drafts compte la première version : on s'arrête après MAX_REVISIONS corrections
    if state["approved"] or state["drafts"] > MAX_REVISIONS:
        return END
    return "redacteur"
