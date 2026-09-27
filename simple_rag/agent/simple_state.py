from typing import Any, TypedDict


class SimpleAgentState(TypedDict):
    """Simplified state - no intent analysis needed"""

    query: str
    search_results: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    explanation: str
    formatted_response: dict[str, Any] | None
