from rag.query import query_vector_store


def test_max_score_drops_weak_matches(vectorstore):
    # Fake store distance = 1 / shared words, so 0.5 keeps products sharing 2+ words
    unfiltered = query_vector_store("wireless earbuds budget", vectorstore, k=30, max_score=None)
    strict = query_vector_store("wireless earbuds budget", vectorstore, k=30, max_score=0.5)
    assert 0 < len(strict) < len(unfiltered)
    assert all(r["score"] <= 0.5 for r in strict)


def test_formatted_results_carry_metadata_and_score(vectorstore):
    [top] = query_vector_store("kindle paperwhite", vectorstore, k=1)
    assert top["name"] == "Kindle Paperwhite"
    assert top["score_type"] == "similarity_score"
    assert "content" in top


def test_unformatted_results_are_raw_pairs(vectorstore):
    doc, score = query_vector_store("kindle", vectorstore, k=1, format_results=False)[0]
    assert doc.metadata["name"] == "Kindle Paperwhite"
    assert isinstance(score, float)
