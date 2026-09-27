import logging

from rag.format_data import format_search_results

logger = logging.getLogger(__name__)


def query_vector_store(
    query: str,
    vectorstore,
    k: int = 5,
    format_results: bool = True,
    max_score: float | None = None,
):
    """
    Similarity search, optionally dropping weak matches.

    FAISS returns L2 distances, so LOWER is better: `max_score` keeps only
    results whose distance is <= the threshold.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    if max_score is not None:
        kept = [(doc, score) for doc, score in results if score <= max_score]
        logger.debug("Score filter %.2f kept %d/%d", max_score, len(kept), len(results))
        results = kept

    return format_search_results(results) if format_results else results
