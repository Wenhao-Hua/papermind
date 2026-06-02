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
