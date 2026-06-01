"""Retrieve and format the passages that ground an answer."""

from __future__ import annotations

from typing import List, Optional, Tuple

from papermind.llm.base import LLMClient
from papermind.qa.index import Chunk, PaperIndex


class Retriever:
    def __init__(self, index: PaperIndex, client: LLMClient) -> None:
        self.index = index
        self.client = client

    def retrieve(self, query: str, k: int = 5, section: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        if section:
            # Over-fetch, then keep only chunks in the requested section (fallback to all).
            results = self.index.search(self.client.embed([query]), k=max(k * 3, k))
            filtered = [r for r in results if section.lower() in (r[0].section or "").lower()]
            return (filtered or results)[:k]
        return self.index.search(self.client.embed([query]), k=k)

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
