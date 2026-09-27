from rag.agent.state import AgentState


def format_response_node(state: AgentState) -> AgentState:
    """Node 5: shape the final API payload."""
    intent = state["analyzed_intent"]
    state["formatted_response"] = {
        "query": state["query"],
        "intent": intent.model_dump() if intent else None,
        "recommendations": state["recommendations"],
        "explanation": state["explanation"],
    }
    return state
