"""Generate diagrams for technical points that have no matched original figure.

The default is an architecture-faithful teaching SVG (``generate_svg_diagrams``);
``generate_diagrams`` (Mermaid) is a legacy fallback used only when SVGs are off.
All generated figures are clearly labeled AI-generated, distinct from extracted ones.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from papermind.errors import LLMError
from papermind.llm.base import LLMClient
from papermind.llm.prompts import (
    FIGURE_EXPLAIN_SYSTEM,
    MERMAID_SYSTEM,
    SVG_FIGURE_SYSTEM,
    figure_explain_user,
    image_diagram_prompt,
    mermaid_user,
    svg_figure_user,
)
from papermind.output.schema import Figure, FigureExplain, TechnicalPoint

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


def generate_svg_diagrams(
    points: List[TechnicalPoint], client: LLMClient, context: str, on_notice=None,
    reasoning_effort: Optional[str] = "low", max_tokens: int = 18000,
) -> None:
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
                max_tokens=max_tokens,  # thinking models need more headroom (reasoning + the SVG body)
                reasoning_effort=reasoning_effort,  # "minimal" for thinking models; None for fast non-thinking ones
            )
        except LLMError:
            return None
        svg = _clean_svg(raw)
        if not svg:
            return None
        fig = Figure(type="ai_generated", svg=svg, caption=f"教学示意图：{point.name}")
        fig.explain = _explain_figure(point, svg, client, reasoning_effort)  # best-effort 读图讲解
        return fig

    failed = False
    for point, fig in zip(todo, _parallel_map(_gen, todo)):
        if fig is not None:
            point.figure = fig
        else:
            failed = True
    if failed and on_notice:
        on_notice("部分技术点的 SVG 讲解图生成失败，已跳过该图（不回退 Mermaid）。")


def _svg_text_labels(svg: str, limit: int = 1200) -> str:
    """Pull the visible <text>/<tspan> strings out of an SVG so the walkthrough can be
    grounded in what the figure actually shows (rather than inventing parts)."""
    def _entity(m, base):
        try:
            return chr(int(m.group(1), base))  # a bad/out-of-range codepoint -> leave the entity as-is
        except (ValueError, OverflowError):
            return m.group(0)

    out, seen = [], set()
    for raw in re.findall(r"<(?:text|tspan)\b[^>]*>(.*?)</(?:text|tspan)>", svg, flags=re.DOTALL | re.IGNORECASE):
        t = re.sub(r"<[^>]+>", "", raw)
        t = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: _entity(m, 16), t)
        t = re.sub(r"&#(\d+);", lambda m: _entity(m, 10), t)
        t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        t = re.sub(r"\s+", " ", t).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return " | ".join(out)[:limit]


def _explain_figure(point: TechnicalPoint, svg: str, client: LLMClient, reasoning_effort) -> "FigureExplain | None":
    """Best-effort structured "how to read this figure" walkthrough, grounded in the
    figure's own labels. Returns None on any failure (the figure is still shown)."""
    try:
        data = client.complete_json(
            FIGURE_EXPLAIN_SYSTEM,
            figure_explain_user(point.name, point.explanation, point.formula, _svg_text_labels(svg)),
            max_tokens=2500,
            reasoning_effort=reasoning_effort,
        )
    except LLMError:
        return None
    if not isinstance(data, dict):
        return None
    gist = str(data.get("gist") or "").strip()
    if not gist:
        return None
    parts = [str(x).strip() for x in (data.get("parts") or []) if str(x).strip()][:4]
    return FigureExplain(gist=gist, parts=parts, takeaway=str(data.get("takeaway") or "").strip())


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


# SVG elements that can execute script or break out into HTML when the figure is
# inlined into the shared report — none are used by the teaching SVGs.
_SVG_FORBIDDEN_TAGS = {
    "script", "foreignobject", "style", "animate", "animatetransform",
    "animatemotion", "set", "image", "use", "iframe", "object", "embed", "handler",
}


def _local_name(tag: str) -> str:
    """Strip an XML namespace ('{ns}rect' -> 'rect') and lowercase."""
    return tag.rsplit("}", 1)[-1].lower()


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
    # <foreignObject> embeds XHTML the browser parses as HTML (arbitrary-markup injection in a
    # shared/online report); <style> injects page-wide CSS. Neither is used by the teaching SVGs
    # — strip them (the parse-tree check below rejects any that survive via nesting tricks).
    svg = re.sub(r"<foreignObject\b.*?</foreignObject>", "", svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"<style\b.*?</style>", "", svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"""\son\w+\s*=\s*(["']).*?\1""", "", svg, flags=re.IGNORECASE | re.DOTALL)
    # The teaching-SVG spec uses no href at all, so drop every href/xlink:href.
    svg = re.sub(r"""\s(?:xlink:)?href\s*=\s*(["']).*?\1""", "", svg, flags=re.IGNORECASE | re.DOTALL)
    # HTML <sub>/<sup>/<br> are "breakout" tags: inside an INLINE <svg> the browser's
    # HTML parser closes the SVG early and dumps the rest of the figure as flowing text
    # (subscripts like d_model / PE_(pos) are extremely common, so this broke most figures).
    # svglib's XML parser tolerated them, which is why offline previews looked fine. Convert
    # to SVG-native <tspan> so it renders correctly in the browser.
    svg = re.sub(r"<sub\b[^>]*>(.*?)</sub>", r'<tspan baseline-shift="sub" font-size="75%">\1</tspan>',
                 svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"<sup\b[^>]*>(.*?)</sup>", r'<tspan baseline-shift="super" font-size="75%">\1</tspan>',
                 svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"<br\s*/?>", "", svg, flags=re.IGNORECASE)
    # Un-double-escaped entities the model sometimes emits: "&amp;#8730;" should be "&#8730;"
    # (√) and "&amp;nbsp;" a space — otherwise the literal string "&#8730;" / "&nbsp;" shows.
    svg = re.sub(r"&amp;(#\d+;|#x[0-9a-fA-F]+;)", r"&\1", svg)
    svg = re.sub(r"&amp;nbsp;", "&#160;", svg)
    # The model sometimes HTML-escapes its own <tspan> tags ("&lt;tspan …&gt;…&lt;/tspan&gt;"),
    # which then show as literal text instead of rendering the sub/superscript. Un-escape tspan
    # tags specifically (attribute quotes inside are real "), restoring them to real elements.
    svg = re.sub(r"&lt;(/?tspan\b[^&]*?)&gt;", r"<\1>", svg)
    # LaTeX-style sub/superscripts don't typeset in SVG — the braces/carets show literally
    # (ℝ^{N×N}, M_{i,j}, ℝ^(d×k)). Convert simple braced/paren groups to native <tspan> so they
    # render as real super/subscripts. (Bare "d_k" is left alone — too ambiguous to touch safely;
    # the SVG generation prompt now forbids LaTeX, so this is a backstop for older/stray output.)
    svg = re.sub(r"\^[\{(]([^{}()<>]{1,28})[\})]", r'<tspan baseline-shift="super" font-size="75%">\1</tspan>', svg)
    svg = re.sub(r"_[\{(]([^{}()<>]{1,28})[\})]", r'<tspan baseline-shift="sub" font-size="75%">\1</tspan>', svg)
    # LLM-drawn SVGs routinely contain a raw "&" (e.g. "Add & Norm") which is invalid
    # XML and was silently sinking the whole figure to the Mermaid fallback. Escape any
    # "&" that isn't already a valid entity so well-formedness checks can pass.
    svg = re.sub(r"&(?!(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);)", "&amp;", svg)
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(svg)  # must be well-formed
    except Exception:  # noqa: BLE001
        return ""
    # Defense in depth: if any executable/breakout element or event/href attribute survived the
    # strips above (e.g. via nested tags that defeat a non-greedy regex), drop the whole figure
    # rather than risk injecting it into the shared report HTML.
    for el in root.iter():
        if _local_name(el.tag) in _SVG_FORBIDDEN_TAGS:
            return ""
        if any(_local_name(a).startswith("on") or _local_name(a) == "href" for a in el.attrib):
            return ""
    # Give the root explicit width/height (from viewBox). Without them a browser renders
    # the SVG at a tiny default size when it's an <img>/standalone file (e.g. the relative
    # SVGs in examples/ on GitHub). Inline in the report it stays responsive (CSS sets
    # max-width:100% + height:auto, which override these).
    tag_end = svg.find(">")
    if tag_end != -1 and "width=" not in svg[:tag_end]:
        vb = re.search(r'viewBox="[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)"', svg[:tag_end])
        if vb:
            svg = svg[:tag_end] + f' width="{vb.group(1)}" height="{vb.group(2)}"' + svg[tag_end:]
    return svg
