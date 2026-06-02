"""Pure ranking metrics (no torch/network) — unit-tested.

Each query is represented by ``labels``: the gold relevance (1/0) of candidates
in *predicted* rank order (best first). Recall is proportional (fraction of gold
retrieved), not hit-rate.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def recall_at_k(labels: Sequence[int], k: int) -> float:
    gold = sum(labels)
    return (sum(labels[:k]) / gold) if gold else 0.0


def precision_at_k(labels: Sequence[int], k: int) -> float:
    return (sum(labels[:k]) / k) if k else 0.0


def f1_at_k(labels: Sequence[int], k: int) -> float:
    p, r = precision_at_k(labels, k), recall_at_k(labels, k)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def mrr(labels: Sequence[int]) -> float:
    for rank, label in enumerate(labels, start=1):
        if label:
            return 1.0 / rank
    return 0.0


def dcg_at_k(labels: Sequence[int], k: int) -> float:
    return sum(labels[i] / math.log2(i + 2) for i in range(min(k, len(labels))))


def ndcg_at_k(labels: Sequence[int], k: int) -> float:
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    return (dcg_at_k(labels, k) / ideal) if ideal else 0.0


def aggregate(ranked: List[List[int]], ks=(1, 5, 10)) -> Dict[str, float]:
    """Mean metrics over queries (queries with no gold are skipped)."""
    rows = [r for r in ranked if any(r)]
    n = len(rows)
    if n == 0:
        return {**{f"Recall@{k}": 0.0 for k in ks}, "MRR": 0.0, "nDCG@10": 0.0, "F1@5": 0.0, "n_queries": 0}
    out = {f"Recall@{k}": round(sum(recall_at_k(r, k) for r in rows) / n, 4) for k in ks}
    out["MRR"] = round(sum(mrr(r) for r in rows) / n, 4)
    out["nDCG@10"] = round(sum(ndcg_at_k(r, 10) for r in rows) / n, 4)
    out["F1@5"] = round(sum(f1_at_k(r, 5) for r in rows) / n, 4)
    out["n_queries"] = n
    return out
