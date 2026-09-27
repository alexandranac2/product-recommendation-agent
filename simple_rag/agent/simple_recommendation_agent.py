from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from simple_rag.agent.simple_state import SimpleAgentState
from simple_rag.nodes import (
    simple_explain_node,
    simple_format_node,
    simple_refine_node,
    simple_search_node,
)


def build_simple_recommendation_graph(vectorstore):
    """Deterministic variant: no intent LLM, no explanation LLM — zero LLM calls."""
    workflow = StateGraph(SimpleAgentState)

    # Add nodes (no analyze_intent_node!)
    workflow.add_node("search", partial(simple_search_node, vectorstore=vectorstore))
    workflow.add_node("refine", simple_refine_node)
    workflow.add_node("explain", simple_explain_node)
    workflow.add_node("format", simple_format_node)

    # Connect nodes - simpler linear flow
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "refine")
    workflow.add_edge("refine", "explain")
    workflow.add_edge("explain", "format")
    workflow.add_edge("format", END)

    compiled_graph = workflow.compile()

    # Return a function that handles state creation and execution
    def run(query: str) -> dict[str, Any]:
        """Execute the simple recommendation graph with a query"""
        state: SimpleAgentState = {
            "query": query,
            "search_results": [],
            "recommendations": [],
            "explanation": "",
            "formatted_response": None,
        }
        final_state = compiled_graph.invoke(state)
        return final_state["formatted_response"] or {}

    return run
