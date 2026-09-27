import pytest

from config.settings import settings
from rag.intent import QueryIntent
from rag.nodes.refine_results_node import refine_results_node
from rag.nodes.search_products_node import build_search_query
from simple_rag.nodes.simple_explain_node import simple_explain_node
from simple_rag.pricing import extract_price_range


def _state(query="q", intent=None, results=None, recommendations=None):
    return {
        "query": query,
        "analyzed_intent": intent,
        "search_results": results or [],
        "recommendations": recommendations or [],
        "explanation": "",
        "formatted_response": None,
    }


RESULTS = [{"name": n, "price": p} for n, p in [("a", 10), ("b", 150), ("c", 250), ("d", 90)]]


def test_refine_applies_intent_price_range():
    intent = QueryIntent(intent="search", product="x", price_range={"min": 50, "max": 200})
    out = refine_results_node(_state(intent=intent, results=RESULTS))
    assert [r["name"] for r in out["recommendations"]] == ["b", "d"]


def test_refine_without_price_keeps_order_and_caps_length():
    many = [{"name": str(i), "price": i} for i in range(20)]
    out = refine_results_node(_state(intent=QueryIntent(intent="search", product="x"), results=many))
    assert len(out["recommendations"]) == settings.MAX_RECOMMENDATIONS_TO_RETURN
    assert out["recommendations"][0]["name"] == "0"


def test_search_query_uses_intent_product_and_use_case():
    intent = QueryIntent(intent="search", product="laptop", use_case="gaming")
    assert build_search_query(_state(query="raw", intent=intent)) == "laptop gaming"
    assert build_search_query(_state(query="raw")) == "raw"


@pytest.mark.parametrize(
    "query, expected",
    [
        ("running shoes under $200", (0.0, 200.0)),
        ("headphones over 100", (100.0, float("inf"))),
        ("budget serum up to 9.99", (0.0, 9.99)),
        ("a nice jacket", (0.0, float("inf"))),
    ],
)
def test_extract_price_range(query, expected):
    assert extract_price_range(query) == expected


def test_simple_explanation_is_built_only_from_result_data():
    recs = [
        {"name": "Kindle Paperwhite", "price": 139.99, "rating": 4.8, "category": "Electronics"},
        {"name": "Budget Earbuds", "price": 49.99, "rating": 4.2, "category": "Electronics"},
    ]
    out = simple_explain_node(_state(query="gadgets under $150", recommendations=recs))
    text = out["explanation"]
    assert "Found 2 products" in text
    assert "under $150" in text
    assert "Top recommendation: Kindle Paperwhite at $139.99." in text


def test_simple_explanation_handles_no_results():
    assert simple_explain_node(_state())["explanation"] == "No products found matching your criteria."
