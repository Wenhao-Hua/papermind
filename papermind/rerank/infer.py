"""Inference wrapper for the trained cross-encoder evidence reranker.

Loads a checkpoint produced by ``trainer/train_reranker.py`` and rescoring a
question against candidate passages. ``sentence-transformers`` is imported lazily
so importing this module costs nothing until a reranker is actually used.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


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
