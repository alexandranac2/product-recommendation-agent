from rag.agent.state import AgentState
from rag.intent import analyze_query


def analyze_intent_node(state: AgentState) -> AgentState:
    """Node 1: understand what the user is asking for."""
    state["analyzed_intent"] = analyze_query(state["query"])
    return state
