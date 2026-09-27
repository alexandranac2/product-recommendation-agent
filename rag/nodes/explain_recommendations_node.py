from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.settings import settings
from rag.agent.state import AgentState


@lru_cache(maxsize=1)
def _explain_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)


def explain_recommendations_node(state: AgentState) -> AgentState:
    """Node 4: short natural-language explanation grounded in the top results."""
    recommendations = state["recommendations"]
    if not recommendations:
        state["explanation"] = "No products found matching your criteria."
        return state

    products_text = "\n".join(
        f"- {r.get('name')} (${r.get('price', 0)})" for r in recommendations[: settings.MAX_RECOMMENDATIONS_TO_EXPLAIN]
    )
    prompt = (
        f'User asked: "{state["query"]}"\n\n'
        f"Top products:\n{products_text}\n\n"
        "Explain in 2 sentences why these match the user's needs. "
        "Only mention the products listed above."
    )
    state["explanation"] = _explain_llm().invoke(prompt).content.strip()
    return state
