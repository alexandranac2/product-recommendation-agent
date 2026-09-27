import logging

from config.settings import settings
from rag.agent.state import AgentState
from rag.query import query_vector_store

logger = logging.getLogger(__name__)


def build_search_query(state: AgentState) -> str:
    """Search on the extracted product + use case; fall back to the raw query."""
    intent = state["analyzed_intent"]
    if not intent:
        return state["query"]
    return f"{intent.product} {intent.use_case}" if intent.use_case else intent.product


def search_products_node(state: AgentState, vectorstore) -> AgentState:
    """Node 2: semantic search over the product index."""
    search_query = build_search_query(state)
    state["search_results"] = query_vector_store(
        search_query,
        vectorstore=vectorstore,
        k=settings.DEFAULT_SEARCH_K,
        max_score=settings.MAX_SIMILARITY_SCORE,
    )
    logger.info("Search %r -> %d products", search_query, len(state["search_results"]))
    return state
