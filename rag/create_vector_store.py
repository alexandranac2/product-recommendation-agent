from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from rag.ingest import create_documents
from rag.load_products import load_products

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "data" / "products.json"


def create_load_vector_store(
    name: str | None = None,
    products_path: Path | None = None,
) -> FAISS:
    """
    Load the FAISS index from disk, or build it from the product catalogue.

    The index is a local build artefact (git-ignored): it is created on first
    start from `products_path` and reused afterwards.
    """
    name = name or settings.VECTOR_STORE_NAME
    products_path = products_path or DEFAULT_PRODUCTS_PATH
    faiss_path = PROJECT_ROOT / name

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    if (faiss_path / "index.faiss").exists() and (faiss_path / "index.pkl").exists():
        logger.info("Loading vector store from %s", faiss_path)
        # The pickle is produced locally by this function, never downloaded,
        # which is what makes deserialising it acceptable.
        return FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)

    if not products_path.exists():
        raise FileNotFoundError(f"No vector store at {faiss_path} and no catalogue at {products_path}")

    documents = create_documents(load_products(products_path))
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    faiss_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(faiss_path))
    logger.info("Built FAISS index with %d products at %s", len(documents), faiss_path)
    return vectorstore
