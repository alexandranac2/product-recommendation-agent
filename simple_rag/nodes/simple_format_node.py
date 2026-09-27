from typing import Any

from simple_rag.agent.simple_state import SimpleAgentState


def simple_format_node(state: SimpleAgentState) -> SimpleAgentState:
    """Format the final response"""
    formatted_response: dict[str, Any] = {
        "query": state["query"],
        "recommendations": state["recommendations"],
        "explanation": state["explanation"],
    }

    state["formatted_response"] = formatted_response
    return state
