from config.settings import settings
from simple_rag.agent.simple_state import SimpleAgentState
from simple_rag.pricing import extract_price_range


def simple_refine_node(state: SimpleAgentState) -> SimpleAgentState:
    """Filter by the price range found in the query text."""
    min_price, max_price = extract_price_range(state["query"])
    filtered = [r for r in state["search_results"] if min_price <= r.get("price", 0) <= max_price]
    state["recommendations"] = filtered[: settings.MAX_RECOMMENDATIONS_TO_RETURN]
    return state
