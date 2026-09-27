import logging
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from config.settings import settings
from rag.query import query_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])

# Built once at startup (see api.main lifespan) and shared by every request.
_vectorstore = None
_recommend: Callable[[str], dict[str, Any]] | None = None
_recommend_fast: Callable[[str], dict[str, Any]] | None = None


def initialize_recommendation_system(vectorstore=None) -> None:
    """Load the index and compile the graph. Called once from the app lifespan."""
    global _vectorstore, _recommend, _recommend_fast
    from rag.agent.recommendation_agent import build_recommendation_graph
    from simple_rag.agent.simple_recommendation_agent import build_simple_recommendation_graph

    if vectorstore is None:
        from rag.create_vector_store import create_load_vector_store

        vectorstore = create_load_vector_store()

    _vectorstore = vectorstore
    _recommend = build_recommendation_graph(vectorstore)
    _recommend_fast = build_simple_recommendation_graph(vectorstore)
    logger.info("Recommendation system initialised")


def _require_ready() -> None:
    if _recommend is None or _vectorstore is None:
        raise HTTPException(status_code=503, detail="Recommendation system is still starting up.")


class RecommendationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, examples=["Best running shoes under $200"])
    max_results: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["llm", "fast"] = Field(
        default="llm",
        description="llm: LLM intent + explanation. fast: regex price parsing, templated explanation, zero LLM calls.",
    )


class RecommendationResponse(BaseModel):
    query: str
    recommendations: list[dict[str, Any]]
    explanation: str
    total_results: int
    intent: dict[str, Any] | None = None


@router.post("/", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Recommend products for a natural-language query.

    - `llm` mode: intent analysis → vector search → price refinement → LLM explanation
    - `fast` mode: vector search → regex price refinement → templated explanation
    """
    _require_ready()
    recommend = _recommend_fast if request.mode == "fast" else _recommend
    try:
        result = recommend(request.query)
    except Exception as exc:
        logger.exception("Recommendation failed for %r", request.query)
        raise HTTPException(status_code=500, detail="Could not generate recommendations.") from exc

    recommendations = result.get("recommendations", [])
    if request.max_results:
        recommendations = recommendations[: request.max_results]

    return RecommendationResponse(
        query=request.query,
        recommendations=recommendations,
        explanation=result.get("explanation", ""),
        total_results=len(recommendations),
        intent=result.get("intent"),
    )


@router.get("/search")
def search_products(
    q: str = Query(min_length=1, max_length=500),
    k: int = Query(default=settings.DEFAULT_SEARCH_K, ge=1, le=100),
    max_score: float = Query(default=settings.MAX_SIMILARITY_SCORE, gt=0),
) -> dict[str, Any]:
    """Raw vector search (no LLM). `max_score` is an L2 distance: lower = closer."""
    _require_ready()
    results = query_vector_store(q, vectorstore=_vectorstore, k=k, max_score=max_score)
    return {"query": q, "results": results, "count": len(results)}
