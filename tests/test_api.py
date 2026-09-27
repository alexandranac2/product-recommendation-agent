from fastapi.testclient import TestClient

from tests.conftest import EXPLAIN_NODE, INTENT_NODE


def test_health_and_readiness(client):
    assert client.get("/api/health/").json() == {"status": "healthy"}
    assert client.get("/api/health/ready").json() == {"status": "ready"}


def test_not_ready_before_startup():
    from api.main import app

    raw = TestClient(app)
    assert raw.get("/api/health/ready").status_code == 503
    assert raw.post("/api/recommendations/", json={"query": "shoes"}).status_code == 503


def test_fast_mode_makes_no_llm_calls_and_respects_price(client, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("fast mode must not call an LLM")

    monkeypatch.setattr(INTENT_NODE, "analyze_query", boom)
    monkeypatch.setattr(EXPLAIN_NODE, "_explain_llm", boom)

    body = client.post("/api/recommendations/", json={"query": "running shoes under $180", "mode": "fast"}).json()
    assert body["recommendations"], body
    assert all(r["price"] <= 180 for r in body["recommendations"])
    assert body["intent"] is None
    assert body["total_results"] == len(body["recommendations"])


def test_llm_mode_uses_intent_and_explanation(client, stub_llms):
    body = client.post("/api/recommendations/", json={"query": "shoes for running"}).json()
    names = [r["name"] for r in body["recommendations"]]
    assert "Adidas Ultraboost Running Shoes" in names
    assert all(r["price"] <= 200 for r in body["recommendations"])
    assert body["intent"]["product"] == "running shoes"
    assert body["explanation"] == "Great picks."


def test_max_results_truncates(client, stub_llms):
    body = client.post("/api/recommendations/", json={"query": "shoes", "max_results": 1}).json()
    assert body["total_results"] == 1


def test_llm_failure_returns_clean_500(client, monkeypatch):
    def fail(_query):
        raise RuntimeError("upstream timeout with secret details")

    monkeypatch.setattr(INTENT_NODE, "analyze_query", fail)
    response = client.post("/api/recommendations/", json={"query": "shoes"})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_request_validation(client):
    assert client.post("/api/recommendations/", json={"query": ""}).status_code == 422
    assert client.post("/api/recommendations/", json={"query": "x", "mode": "nope"}).status_code == 422
    assert client.post("/api/recommendations/", json={"query": "x", "max_results": 0}).status_code == 422


def test_search_endpoint_reuses_the_loaded_index(client):
    body = client.get("/api/recommendations/search", params={"q": "kindle paperwhite", "k": 3}).json()
    assert body["results"][0]["name"] == "Kindle Paperwhite"
    assert body["count"] == len(body["results"])
