import logging

from config.settings import settings
from rag.agent.state import AgentState

logger = logging.getLogger(__name__)


def refine_results_node(state: AgentState) -> AgentState:
    """
    Node 3: apply hard constraints the vector search cannot express.

    Semantic similarity already handles category relevance, so only the explicit
    price range from the intent is enforced here.
    """
    intent = state["analyzed_intent"]
    results = state["search_results"]

    if intent and intent.price_range:
        min_price = intent.price_range.get("min") or 0
        max_price = intent.price_range.get("max") or float("inf")
        results = [r for r in results if min_price <= r.get("price", 0) <= max_price]

    state["recommendations"] = results[: settings.MAX_RECOMMENDATIONS_TO_RETURN]
    logger.info("%d recommendations after refinement", len(state["recommendations"]))
    return state
