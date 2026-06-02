"""Retrieve and format the passages that ground an answer."""

from __future__ import annotations

from typing import List, Optional, Tuple

from papermind.llm.base import LLMClient
from papermind.qa.index import Chunk, PaperIndex


class Retriever:
    def __init__(self, index: PaperIndex, client: LLMClient, reranker=None, rerank_candidates: int = 30) -> None:
        self.index = index
        self.client = client
        self.reranker = reranker  # optional trained cross-encoder; None -> plain dense retrieval
        self.rerank_candidates = rerank_candidates

    def retrieve(self, query: str, k: int = 5, section: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        # Over-fetch a larger pool when a reranker (or a section filter) is in play.
        if self.reranker is not None:
            pool = max(self.rerank_candidates, k)
        elif section:
            pool = max(k * 3, k)
        else:
            pool = k
        results = self.index.search(self.client.embed([query]), k=pool)
        if section:
            results = [r for r in results if section.lower() in (r[0].section or "").lower()] or results
        if self.reranker is not None and results:
            ranked = self.reranker.rerank(query, [c.text for c, _ in results], top_k=k)
            return [(results[i][0], float(score)) for i, score in ranked]
        return results[:k]

    @staticmethod
    def format_passages(results: List[Tuple[Chunk, float]]) -> str:
        """Number passages and tag each with its section/page for citation."""
        lines = []
        for n, (chunk, _score) in enumerate(results, start=1):
            loc = chunk.section or ""
            if chunk.page:
                loc = f"{loc}, p.{chunk.page}".strip(", ")
            lines.append(f"[{n}] ({loc})\n{chunk.text}")
        return "\n\n".join(lines)
