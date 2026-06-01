"""Generate Mermaid diagrams for technical points that have no matched original
figure. Clearly labeled as AI-generated so they are distinguishable from
extracted figures.
"""

from __future__ import annotations

import re
from typing import List

from papermind.errors import LLMError
from papermind.llm.base import LLMClient
from papermind.llm.prompts import MERMAID_SYSTEM, mermaid_user
from papermind.output.schema import Figure, TechnicalPoint


def generate_diagrams(points: List[TechnicalPoint], client: LLMClient) -> None:
    for point in points:
        if point.figure is not None:
            continue  # already has an original figure
        try:
            data = client.complete_json(MERMAID_SYSTEM, mermaid_user(point.name, point.explanation))
        except LLMError:
            continue
        if not isinstance(data, dict):
            continue
        mermaid = _clean_mermaid(data.get("mermaid"))
        if not mermaid:
            continue
        caption = data.get("caption") or "AI 生成示意图"
        point.figure = Figure(type="ai_generated", mermaid=mermaid, caption=str(caption))


def _clean_mermaid(value) -> str:
    if not value or not isinstance(value, str):
        return ""
    text = value.strip()
    fence = re.match(r"^```(?:mermaid)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text
