"""Module 2: technical detail extraction & explanation.

Figure matching/generation is layered on afterward by ``papermind.figures`` so
this module stays focused on identifying and explaining the hard parts.
"""

from __future__ import annotations

from papermind.llm.base import LLMClient
from papermind.llm.prompts import ANALYSIS_SYSTEM, technical_user
from papermind.modules._util import int_or_none, str_or_none
from papermind.output.schema import TechnicalPoint, TechnicalSection
from papermind.parser.pdf import ParsedPaper

_VALID_DIFFICULTY = {"high", "mid", "low"}


def run(parsed: ParsedPaper, client: LLMClient, context: str, max_points: int = 6) -> TechnicalSection:
    data = client.complete_json(ANALYSIS_SYSTEM, technical_user(context, max_points))
    items = data.get("technical_details") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return TechnicalSection()

    points = []
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        difficulty = str(item.get("difficulty", "mid") or "mid").lower()
        if difficulty not in _VALID_DIFFICULTY:
            difficulty = "mid"
        points.append(
            TechnicalPoint(
                name=str(item["name"]),
                explanation=str(item.get("explanation", "") or ""),
                analogy=str(item.get("analogy", "") or ""),
                formula=str_or_none(item.get("formula")),
                source_section=str_or_none(item.get("source_section")),
                page=int_or_none(item.get("page")),
                difficulty=difficulty,
            )
        )
    return TechnicalSection(details=points[:max_points])
