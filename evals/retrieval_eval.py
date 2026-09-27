"""
Retrieval eval: labelled queries against the real FAISS index + embedding model.

    python -m evals.retrieval_eval [--min-hit3 0.9]

No API key needed. Queries are paraphrased (they avoid product names) so the
score reflects semantic matching, not string overlap. Reports hit@1, hit@3,
recall@5 and MRR, and lists the misses so regressions are easy to inspect.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.create_vector_store import create_load_vector_store
from rag.query import query_vector_store

QUERIES = Path(__file__).parent / "queries.json"
K = 5


def evaluate(vectorstore, cases):
    hits1 = hits3 = mrr = recall5 = 0.0
    misses = []
    for case in cases:
        relevant = set(case["relevant"])
        ranked = [r["id"] for r in query_vector_store(case["query"], vectorstore, k=K, max_score=None)]
        first = next((rank for rank, pid in enumerate(ranked, 1) if pid in relevant), None)

        hits1 += first == 1
        hits3 += first is not None and first <= 3
        mrr += 1 / first if first else 0
        recall5 += len(relevant & set(ranked)) / len(relevant)
        if first != 1:
            misses.append((case["query"], sorted(relevant), ranked[:3]))

    n = len(cases)
    return {"queries": n, "hit@1": hits1 / n, "hit@3": hits3 / n, "recall@5": recall5 / n, "mrr": mrr / n}, misses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-hit3", type=float, default=0.0, help="exit non-zero below this hit@3 (CI gate)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cases = json.loads(QUERIES.read_text())
    metrics, misses = evaluate(create_load_vector_store(), cases)

    print(" | ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()))
    for query, relevant, top3 in misses:
        print(f"  not first: {query!r} expected {relevant}, got {top3}")

    if metrics["hit@3"] < args.min_hit3:
        print(f"FAIL: hit@3 {metrics['hit@3']:.2f} < {args.min_hit3}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
