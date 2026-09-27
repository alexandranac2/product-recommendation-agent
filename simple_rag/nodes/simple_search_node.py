from config.settings import settings
from rag.query import query_vector_store
from simple_rag.agent.simple_state import SimpleAgentState


def simple_search_node(state: SimpleAgentState, vectorstore) -> SimpleAgentState:
    """Search on the raw query: the embedding model handles the semantics."""
    state["search_results"] = query_vector_store(
        state["query"],
        vectorstore=vectorstore,
        k=settings.DEFAULT_SEARCH_K,
        max_score=settings.MAX_SIMILARITY_SCORE,
    )
    return state
