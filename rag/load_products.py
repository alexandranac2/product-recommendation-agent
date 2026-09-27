import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def load_products(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load the product catalogue from a JSON array."""
    with open(file_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    logger.info("Loaded %d products from %s", len(products), file_path)
    return products
