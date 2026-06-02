"""Teaching-SVG figure: cleaning/sanitizing (pure) + rendering. No LLM."""

from __future__ import annotations

from papermind.figures.generate import _clean_svg
from papermind.output.schema import Figure, PaperMeta, Report, TechnicalPoint, TechnicalSection


def test_clean_svg_strips_fence_and_keeps_svg():
    out = _clean_svg("```svg\n<svg viewBox='0 0 10 10'><rect/></svg>\n```")
    assert out.startswith("<svg") and out.endswith("</svg>")


def test_clean_svg_removes_script_and_handlers():
    out = _clean_svg('<svg><script>alert(1)</script><rect onclick="x()"/></svg>')
    assert "<script" not in out and "onclick" not in out
    assert "<rect" in out  # the safe content survives


def test_clean_svg_rejects_non_svg_and_malformed():
    assert _clean_svg("no svg here") == ""
    assert _clean_svg("<svg><rect></svg>") == ""  # unclosed <rect> -> not well-formed -> dropped


def test_clean_svg_strips_external_refs():
    out = _clean_svg(
        '<svg><image href="http://x/y.png"/><use xlink:href="#a"/>'
        '<rect/><a href="https://evil"></a></svg>'
    )
    assert out and "<image" not in out and "<use" not in out
    assert "href" not in out and "<rect" in out  # safe content survives


def test_clean_svg_converts_html_sub_sup_to_tspan():
    # HTML <sub>/<sup> are "breakout" tags: in an inline <svg> the browser's HTML parser
    # closes the SVG early and the rest of the figure spills out as text. Convert to <tspan>.
    out = _clean_svg('<svg viewBox="0 0 100 50"><text x="5" y="20">d<sub>model</sub> 10<sup>2</sup></text></svg>')
    assert out and "<sub" not in out and "<sup" not in out
    assert out.count('baseline-shift="sub"') == 1 and out.count('baseline-shift="super"') == 1
    assert "model" in out  # inner text preserved


def test_clean_svg_escapes_bare_ampersand():
    # "Add & Norm" (raw &) is invalid XML; we escape it rather than drop the whole figure
    # (this was the #1 reason SVGs silently fell back to Mermaid).
    out = _clean_svg("<svg viewBox='0 0 10 10'><text>Add & Norm</text></svg>")
    assert out and "Add &amp; Norm" in out


def test_clean_svg_preserves_existing_entities():
    out = _clean_svg("<svg viewBox='0 0 10 10'><text>a &amp; b &#8594; &#x2192;</text></svg>")
    assert out and out.count("&amp;") == 1 and "&#8594;" in out and "&#x2192;" in out


def test_svg_renders_inline_in_html_and_datauri_in_md():
    svg = "<svg viewBox='0 0 20 20'><rect width='20' height='20'/></svg>"
    report = Report(
        paper=PaperMeta(title="T"),
        technical=TechnicalSection(details=[
            TechnicalPoint(name="Attn", explanation="...", figure=Figure(type="ai_generated", svg=svg)),
        ]),
    )
    html = report.to_html()
    assert "<svg viewBox='0 0 20 20'>" in html  # inlined, not escaped
    md = report.to_markdown()
    assert "data:image/svg+xml;base64," in md


def test_figure_explain_renders_in_html_and_md():
    from papermind.output.schema import FigureExplain

    svg = "<svg viewBox='0 0 20 20'><rect width='20' height='20'/></svg>"
    ex = FigureExplain(gist="这张图定义浪费比率", parts=["T_extra 是多开检查", "fee 用 CMS 换算"], takeaway="比率越低越克制")
    report = Report(
        paper=PaperMeta(title="T"),
        technical=TechnicalSection(details=[
            TechnicalPoint(name="X", explanation="...", figure=Figure(type="ai_generated", svg=svg, explain=ex)),
        ]),
    )
    html = report.to_html()
    assert 'class="readfig"' in html and "读图" in html
    assert "T_extra 是多开检查" in html and "比率越低越克制" in html
    md = report.to_markdown()
    assert "> **读图**：这张图定义浪费比率" in md and "> - T_extra 是多开检查" in md and "**关键**：比率越低越克制" in md
