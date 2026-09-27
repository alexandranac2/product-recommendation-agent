from typing import Any, TypedDict

from rag.intent import QueryIntent


class AgentState(TypedDict):
    """State passed between the nodes of the recommendation graph."""

    query: str
    analyzed_intent: QueryIntent | None
    search_results: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    explanation: str
    formatted_response: dict[str, Any] | None
