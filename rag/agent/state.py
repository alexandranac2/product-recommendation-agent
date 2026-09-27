from typing import Any, Dict, List, Optional, TypedDict

from rag.intent import QueryIntent


class AgentState(TypedDict):
    """State passed between the nodes of the recommendation graph."""
    query: str
    analyzed_intent: Optional[QueryIntent]
    search_results: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    explanation: str
    formatted_response: Optional[Dict[str, Any]]
