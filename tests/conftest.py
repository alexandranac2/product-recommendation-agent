"""
Offline fixtures: a keyword-overlap stand-in for FAISS and stubbed LLMs, so the
suite runs without an OpenAI key or a sentence-transformers download.
"""
import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag.ingest import create_documents
from rag.intent import QueryIntent
from rag.load_products import load_products

PRODUCTS_PATH = Path(__file__).parent.parent / "data" / "products.json"

# rag.nodes re-exports each node function under its module's name, so a string
# target like "rag.nodes.analyze_intent_node.x" resolves to the function, not
# the module. Patch the module objects instead.
INTENT_NODE = importlib.import_module("rag.nodes.analyze_intent_node")
EXPLAIN_NODE = importlib.import_module("rag.nodes.explain_recommendations_node")


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class FakeVectorStore:
    """Scores documents by keyword overlap and reports it as an L2-style distance (lower = closer)."""

    def __init__(self, documents):
        self.documents = documents

    def similarity_search_with_score(self, query, k=4):
        q = _words(query)
        scored = []
        for doc in self.documents:
            overlap = len(q & _words(doc.page_content))
            if overlap:
                scored.append((doc, 1.0 / overlap))
        scored.sort(key=lambda pair: pair[1])
        return scored[:k]


@pytest.fixture(scope="session")
def products():
    return load_products(PRODUCTS_PATH)


@pytest.fixture
def vectorstore(products):
    return FakeVectorStore(create_documents(products))


@pytest.fixture
def stub_llms(monkeypatch):
    """Replace both LLM calls in the full pipeline with deterministic stubs."""
    intent = QueryIntent(intent="search", product="running shoes", price_range={"max": 200})
    monkeypatch.setattr(INTENT_NODE, "analyze_query", lambda query: intent)

    explainer = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content=" Great picks. "))
    monkeypatch.setattr(EXPLAIN_NODE, "_explain_llm", lambda: explainer)
    return intent


def _reset_recommendation_system():
    from api.routes import recommendations

    recommendations._vectorstore = None
    recommendations._recommend = None
    recommendations._recommend_fast = None


@pytest.fixture
def client(vectorstore):
    from api.main import app
    from api.routes import recommendations

    # TestClient without a `with` block skips the lifespan, so the real
    # FAISS/HuggingFace startup never runs; we inject the fake store instead.
    recommendations.initialize_recommendation_system(vectorstore)
    yield TestClient(app)
    _reset_recommendation_system()
