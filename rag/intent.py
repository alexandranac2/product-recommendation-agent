"""LLM-based query understanding: turn free text into a structured intent."""
import logging
from functools import lru_cache
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config.settings import settings

load_dotenv()
logger = logging.getLogger(__name__)


class QueryIntent(BaseModel):
    """What the LLM extracts from a user query."""

    intent: str = Field(description="One of: search, gift, comparison, specific_need")
    product: str = Field(description="The product the user is looking for")
    category: Optional[str] = None
    price_range: Optional[Dict[str, float]] = Field(
        default=None, description='Price bounds, e.g. {"min": 0, "max": 200}'
    )
    attributes: Optional[Dict[str, str]] = None
    use_case: Optional[str] = None


@lru_cache(maxsize=1)
def _intent_llm():
    # temperature=0: extraction should be deterministic
    return ChatOpenAI(model=settings.LLM_MODEL, temperature=0).with_structured_output(QueryIntent)


def analyze_query(query: str) -> QueryIntent:
    """Extract a structured intent from a natural-language product query."""
    intent = _intent_llm().invoke(
        "Extract the shopping intent from this product search query. "
        "Only fill price_range when the user states a price constraint.\n\n"
        f"Query: {query}"
    )
    logger.info("Intent for %r: %s", query, intent)
    return intent
