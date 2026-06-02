"""Inference wrapper for the trained cross-encoder evidence reranker.

Loads a checkpoint produced by ``trainer/train_reranker.py`` and rescoring a
question against candidate passages. ``sentence-transformers`` is imported lazily
so importing this module costs nothing until a reranker is actually used.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple


def maybe_load_reranker(path: Optional[str]) -> Optional["Reranker"]:
    """Load a reranker if ``path`` points to a checkpoint dir and deps are present;
    otherwise return None (Q&A then falls back to plain dense retrieval). Never raises."""
    if not path or not os.path.isdir(path):
        return None
    try:
        return Reranker(path)
    except Exception:  # noqa: BLE001 - reranker is optional; never break Q&A over it
        return None


def order_by_score(scores: Sequence[float], top_k: int = None) -> List[int]:
    """Indices of ``scores`` sorted high→low (stable), optionally truncated to top_k."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order if top_k is None else order[:top_k]


class Reranker:
    """Score (question, passage) pairs with a fine-tuned cross-encoder."""

    def __init__(self, model_path: str, device: str = None, max_length: int = 256) -> None:
        from sentence_transformers import CrossEncoder  # lazy: optional dependency
        import torch

        self.model = CrossEncoder(
            model_path, max_length=max_length, device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def score(self, question: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([[question, p] for p in passages])]

    def rerank(self, question: str, passages: Sequence[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Return ``[(original_index, score), ...]`` for the top_k passages."""
        scores = self.score(question, passages)
        return [(i, scores[i]) for i in order_by_score(scores, top_k)]
