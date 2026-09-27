import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_products(file_path: str | Path) -> list[dict[str, Any]]:
    """Load the product catalogue from a JSON array."""
    with open(file_path, encoding="utf-8") as f:
        products = json.load(f)
    logger.info("Loaded %d products from %s", len(products), file_path)
    return products
