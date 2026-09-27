def format_search_results(results, is_reranked: bool = False, excluded_fields: list = None):
    """
    Format search results for display - works with ANY product structure.

    Args:
        results: List of tuples (document, score)
        is_reranked: If True, score is a rerank score (higher is better).
                    If False, score is a similarity score (lower is better).
        excluded_fields: Optional list of metadata fields to exclude from output
    """
    if excluded_fields is None:
        excluded_fields = []

    formatted = []

    for doc, score in results:
        # Start with all metadata dynamically
        result = {}

        # Add all metadata fields (except excluded ones)
        for key, value in doc.metadata.items():
            if key not in excluded_fields:
                result[key] = value

        # Add search-specific fields
        result.update(
            {
                "content": doc.page_content,
                "score": float(score),
                "score_type": "rerank_score" if is_reranked else "similarity_score",
            }
        )

        formatted.append(result)

    return formatted
