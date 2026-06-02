"""Generate diagrams for technical points that have no matched original figure.

The default is an architecture-faithful teaching SVG (``generate_svg_diagrams``);
``generate_diagrams`` (Mermaid) is a legacy fallback used only when SVGs are off.
All generated figures are clearly labeled AI-generated, distinct from extracted ones.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from papermind.errors import LLMError
from papermind.llm.base import LLMClient
from papermind.llm.prompts import MERMAID_SYSTEM, SVG_FIGURE_SYSTEM, image_diagram_prompt, mermaid_user, svg_figure_user
from papermind.output.schema import Figure, TechnicalPoint

_MAX_WORKERS = 4


def _parallel_map(fn, items):
    """Map fn over items concurrently, preserving order. Serial if 0/1 item."""
    if len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(items))) as pool:
        return list(pool.map(fn, items))


def generate_image_diagrams(
    points: List[TechnicalPoint], client: LLMClient, cache_dir: Path, model: str, on_notice=None
) -> None:
    """Generate an actual image (via an image model) for points lacking a figure.

    Best-effort: any failure (no key, model error) leaves the point figure-less
    so the caller can still fall back to Mermaid. Network calls run in parallel;
    file writes happen serially on the main thread.
    """
    todo = [(i, p) for i, p in enumerate(points) if p.figure is None]
    if not todo:
        return

    def _fetch(item):
        i, point = item
        try:
            return (i, point, client.image(image_diagram_prompt(point.name, point.explanation), model), None)
        except LLMError as exc:
            return (i, point, None, str(exc))

    fig_dir = cache_dir / "figures"
    for i, point, data, err in _parallel_map(_fetch, todo):
        if data is None:
            if err and on_notice:
                on_notice(f"图像生成失败（{model}）：{err}")
                on_notice = None  # only warn once
            continue
        fig_dir.mkdir(parents=True, exist_ok=True)
        dest = fig_dir / f"ai_image_{i}.png"
        try:
            dest.write_bytes(data)
        except OSError:
            continue
        point.figure = Figure(type="ai_generated", image_path=str(dest), caption=f"AI 生成示意图：{point.name}")


def generate_svg_diagrams(points: List[TechnicalPoint], client: LLMClient, context: str, on_notice=None) -> None:
    """Generate a self-contained, architecture-faithful teaching SVG per point.

    Only points without a figure are processed, so an extracted *original* figure
    (precedence: original > AI SVG) is never overwritten. Best-effort: invalid /
    unsafe output is dropped (the point simply gets no figure — we never fall back
    to a Mermaid flowchart). The SVG is validated (well-formed XML) and sanitized
    (no scripts / event handlers / external refs) before it's stored.
    """
    todo = [p for p in points if p.figure is None]
    if not todo:
        return
    ctx = (context or "")[:6000]

    def _gen(point):
        try:
            raw = client.complete(
                SVG_FIGURE_SYSTEM,
                svg_figure_user(point.name, point.explanation, point.formula, ctx),
                max_tokens=18000,  # headroom: a complete definition-forward SVG can run ~14k chars
                reasoning_effort="low",  # a thinking model would otherwise spend the whole budget reasoning
            )
        except LLMError:
            return None
        svg = _clean_svg(raw)
        return Figure(type="ai_generated", svg=svg, caption=f"教学示意图：{point.name}") if svg else None

    failed = False
    for point, fig in zip(todo, _parallel_map(_gen, todo)):
        if fig is not None:
            point.figure = fig
        else:
            failed = True
    if failed and on_notice:
        on_notice("部分技术点的 SVG 讲解图生成失败，已跳过该图（不回退 Mermaid）。")


def generate_diagrams(points: List[TechnicalPoint], client: LLMClient) -> None:
    todo = [p for p in points if p.figure is None]
    if not todo:
        return

    def _gen(point):
        try:
            data = client.complete_json(MERMAID_SYSTEM, mermaid_user(point.name, point.explanation))
        except LLMError:
            return None
        if not isinstance(data, dict):
            return None
        mermaid = _clean_mermaid(data.get("mermaid"))
        if not mermaid:
            return None
        return Figure(type="ai_generated", mermaid=mermaid, caption=str(data.get("caption") or "AI 生成示意图"))

    for point, fig in zip(todo, _parallel_map(_gen, todo)):
        if fig is not None:
            point.figure = fig


def _clean_mermaid(value) -> str:
    if not value or not isinstance(value, str):
        return ""
    text = value.strip()
    fence = re.match(r"^```(?:mermaid)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text


def _clean_svg(value) -> str:
    """Extract a single <svg> element, strip scripts/handlers/external refs, and
    require it to be well-formed XML. Returns '' if anything is off."""
    if not value or not isinstance(value, str):
        return ""
    text = value.strip()
    fence = re.match(r"^```(?:svg|xml|html)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"<svg\b.*?</svg>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    svg = match.group(0)
    # Sanitize: reports are shared HTML, so drop anything executable/external.
    svg = re.sub(r"<script\b.*?</script>", "", svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"<image\b.*?(?:/>|</image>)", "", svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"<use\b.*?(?:/>|</use>)", "", svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"""\son\w+\s*=\s*(["']).*?\1""", "", svg, flags=re.IGNORECASE | re.DOTALL)
    # The teaching-SVG spec uses no href at all, so drop every href/xlink:href.
    svg = re.sub(r"""\s(?:xlink:)?href\s*=\s*(["']).*?\1""", "", svg, flags=re.IGNORECASE | re.DOTALL)
    # LLM-drawn SVGs routinely contain a raw "&" (e.g. "Add & Norm") which is invalid
    # XML and was silently sinking the whole figure to the Mermaid fallback. Escape any
    # "&" that isn't already a valid entity so well-formedness checks can pass.
    svg = re.sub(r"&(?!(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);)", "&amp;", svg)
    try:
        import xml.etree.ElementTree as ET

        ET.fromstring(svg)  # must be well-formed
    except Exception:  # noqa: BLE001
        return ""
    return svg
