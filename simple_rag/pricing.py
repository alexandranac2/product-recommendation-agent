import re

_MAX_PRICE = re.compile(r"(?:under|below|less than|max|maximum|up to)\s*\$?(\d+(?:\.\d+)?)")
_MIN_PRICE = re.compile(r"(?:over|above|more than|min|minimum|from)\s*\$?(\d+(?:\.\d+)?)")


def extract_price_range(query: str) -> tuple[float, float]:
    """Pull "under $200" / "over 50" style bounds out of a query without an LLM."""
    query = query.lower()
    max_match: re.Match | None = _MAX_PRICE.search(query)
    min_match: re.Match | None = _MIN_PRICE.search(query)
    return (
        float(min_match.group(1)) if min_match else 0.0,
        float(max_match.group(1)) if max_match else float("inf"),
    )
