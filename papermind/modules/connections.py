"""Module 3: knowledge connections to prior work."""

from __future__ import annotations

from papermind.llm.base import LLMClient
from papermind.llm.prompts import ANALYSIS_SYSTEM, connections_user
from papermind.modules._util import str_or_none
from papermind.output.schema import Connection, ConnectionsSection
from papermind.parser.pdf import ParsedPaper


def run(parsed: ParsedPaper, client: LLMClient, context: str, max_items: int = 6) -> ConnectionsSection:
    data = client.complete_json(ANALYSIS_SYSTEM, connections_user(context, max_items))
    items = data.get("connections") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return ConnectionsSection()

    works = []
    for item in items:
        if not isinstance(item, dict) or not item.get("concept"):
            continue
        works.append(
            Connection(
                concept=str(item["concept"]),
                paper=str(item.get("paper", "") or ""),
                arxiv_link=_clean_link(item.get("arxiv_link")),
                relationship=str(item.get("relationship", "") or ""),
            )
        )
    return ConnectionsSection(related_works=works[:max_items])


def _clean_link(value):
    link = str_or_none(value)
    if link and link.lower().startswith("http"):
        return link
    return None
