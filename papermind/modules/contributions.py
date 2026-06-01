"""Module 1: contributions & novelty."""

from __future__ import annotations

from papermind.llm.base import LLMClient
from papermind.llm.prompts import CONTRIBUTIONS_SYSTEM, contributions_user
from papermind.modules._util import int_or_none, str_or_none
from papermind.output.schema import Contributions, Source
from papermind.parser.pdf import ParsedPaper


def run(parsed: ParsedPaper, client: LLMClient, context: str) -> Contributions:
    data = client.complete_json(CONTRIBUTIONS_SYSTEM, contributions_user(context))
    if not isinstance(data, dict):
        return Contributions()
    sources = []
    for item in data.get("sources", []) or []:
        if isinstance(item, dict) and item.get("text"):
            sources.append(
                Source(
                    text=str(item["text"]),
                    section=str_or_none(item.get("section")),
                    page=int_or_none(item.get("page")),
                )
            )
    return Contributions(
        main_contribution=str(data.get("main_contribution", "") or ""),
        novelty=str(data.get("novelty", "") or ""),
        problem_solved=str(data.get("problem_solved", "") or ""),
        sources=sources,
    )
