"""Generate Mermaid diagrams for technical points that have no matched original
figure. Clearly labeled as AI-generated so they are distinguishable from
extracted figures.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from papermind.errors import LLMError
from papermind.llm.base import LLMClient
from papermind.llm.prompts import MERMAID_SYSTEM, image_diagram_prompt, mermaid_user
from papermind.output.schema import Figure, TechnicalPoint


def generate_image_diagrams(
    points: List[TechnicalPoint], client: LLMClient, cache_dir: Path, model: str, on_notice=None
) -> None:
    """Generate an actual image (via an image model) for points lacking a figure.

    Best-effort: any failure (no key, model error) leaves the point figure-less
    so the caller can still fall back to Mermaid.
    """
    fig_dir = cache_dir / "figures"
    for i, point in enumerate(points):
        if point.figure is not None:
            continue
        try:
            data = client.image(image_diagram_prompt(point.name, point.explanation), model)
        except LLMError as exc:
            if on_notice:
                on_notice(f"图像生成失败（{model}）：{exc}")
                on_notice = None  # only warn once
            continue
        fig_dir.mkdir(parents=True, exist_ok=True)
        dest = fig_dir / f"ai_image_{i}.png"
        try:
            dest.write_bytes(data)
        except OSError:
            continue
        point.figure = Figure(type="ai_generated", image_path=str(dest), caption=f"AI 生成示意图：{point.name}")


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
