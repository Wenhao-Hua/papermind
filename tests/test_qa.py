"""Q&A / figures tests: chunking, passage formatting, answer layering, figures.

Everything here avoids network, FAISS, and real LLM calls.
"""

from __future__ import annotations

from pathlib import Path

from papermind.figures import extract, generate
from papermind.output.schema import PaperMeta, TechnicalPoint
from papermind.parser.pdf import ExtractedFigure, ParsedPaper, TextBlock
from papermind.qa.chat import PaperChat
from papermind.qa.index import Chunk, build_chunks
from papermind.qa.retriever import Retriever


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, system, user, **kwargs):
        return self.payload


def _stub_paper(figures=None):
    blocks = [TextBlock(text="sentence number %d. " % i * 5, page=(i // 3) + 1, section=f"S{i//3}") for i in range(9)]
    return ParsedPaper(
        meta=PaperMeta(title="t"),
        pages=["p"],
        blocks=blocks,
        sections=[],
        figures=figures or [],
        cache_dir=Path("."),
    )


def test_build_chunks_preserve_section_and_page():
    chunks = build_chunks(_stub_paper(), target_chars=200, overlap_blocks=1)
    assert chunks
    assert all(c.section and c.page for c in chunks)
    assert chunks[0].idx == 0


def test_format_passages_tags_location():
    results = [
        (Chunk(idx=0, text="alpha", section="3.1", page=5), 0.9),
        (Chunk(idx=1, text="beta", section="Abstract", page=1), 0.8),
    ]
    text = Retriever.format_passages(results)
    assert "[1] (3.1, p.5)" in text
    assert "[2] (Abstract, p.1)" in text


def _chat_stub(mode):
    chat = PaperChat.__new__(PaperChat)  # bypass __init__ (no network/index)
    chat.mode = mode
    return chat


def test_parse_answer_layers_and_evidence():
    chat = _chat_stub("balanced")
    data = {
        "segments": [
            {"kind": "fact", "text": "paper uses block 64"},
            {"kind": "inference", "text": "larger may be faster", "confidence": "mid", "reasoning": "SRAM"},
        ],
        "evidence": [{"text": "block of size 64", "section": "3.1", "page": "5"}],
        "sources": [{"section": "3.1", "page": 5}],
    }
    ans = chat._parse_answer("q", data)
    assert [s.kind for s in ans.segments] == ["fact", "inference"]
    assert ans.segments[1].confidence == "mid"
    assert ans.evidence[0].page == 5


def test_strict_mode_drops_inference():
    chat = _chat_stub("strict")
    data = {
        "segments": [
            {"kind": "fact", "text": "fact text"},
            {"kind": "inference", "text": "guess", "confidence": "low"},
        ]
    }
    ans = chat._parse_answer("q", data)
    assert [s.kind for s in ans.segments] == ["fact"]


def test_parse_answer_empty_falls_back_to_out_of_scope():
    chat = _chat_stub("balanced")
    ans = chat._parse_answer("q", {"segments": []})
    assert ans.segments[0].kind == "out_of_scope"


def test_match_original_figures():
    figs = [ExtractedFigure(label="Figure 1", caption="cap", page=2, image_path="/tmp/f1.png")]
    parsed = _stub_paper(figures=figs)
    points = [TechnicalPoint(name="Tiling", explanation="e")]
    client = FakeClient({"matches": [{"point": "Tiling", "figure": "Figure 1"}]})
    extract.match_original_figures(points, parsed, client)
    assert points[0].figure is not None
    assert points[0].figure.type == "original"
    assert points[0].figure.image_path == "/tmp/f1.png"


def test_generate_diagrams_only_when_missing():
    points = [TechnicalPoint(name="A", explanation="e")]
    client = FakeClient({"mermaid": "```mermaid\nflowchart TD\n X-->Y\n```", "caption": "示意图"})
    generate.generate_diagrams(points, client)
    assert points[0].figure.type == "ai_generated"
    assert points[0].figure.mermaid.startswith("flowchart TD")  # fences stripped


def test_clean_mermaid_strips_fences():
    assert generate._clean_mermaid("```mermaid\ngraph TD\nA-->B\n```") == "graph TD\nA-->B"
    assert generate._clean_mermaid(None) == ""


class _ImageClient:
    def __init__(self, data=b"\x89PNG\r\nfake"):
        self.data = data
        self.calls = 0

    def image(self, prompt, model, size="1024x1024"):
        self.calls += 1
        return self.data


def test_generate_image_diagrams_writes_png(tmp_path):
    points = [TechnicalPoint(name="Tiling", explanation="e")]
    client = _ImageClient()
    generate.generate_image_diagrams(points, client, tmp_path, "gpt-image-1")
    fig = points[0].figure
    assert fig is not None and fig.type == "ai_generated"
    assert fig.image_path.endswith(".png") and Path(fig.image_path).exists()
    assert fig.mermaid is None  # it's an image, not mermaid


def test_image_client_skipped_when_point_has_figure(tmp_path):
    from papermind.output.schema import Figure

    points = [TechnicalPoint(name="X", explanation="e", figure=Figure(type="original", image_path="/p.png"))]
    client = _ImageClient()
    generate.generate_image_diagrams(points, client, tmp_path, "gpt-image-1")
    assert client.calls == 0  # already had a figure -> no image generated


def test_renderers_handle_ai_generated_image(tmp_path):
    from papermind.output.html import to_html
    from papermind.output.markdown import to_markdown
    from papermind.output.schema import Figure, PaperMeta, Report, TechnicalPoint as TP, TechnicalSection

    img = tmp_path / "ai.png"
    img.write_bytes(b"\x89PNG\r\nfake")
    report = Report(
        paper=PaperMeta(title="T"),
        technical=TechnicalSection(details=[TP(name="N", explanation="e",
                                               figure=Figure(type="ai_generated", image_path=str(img), caption="AI 生成示意图：N"))]),
    )
    md = to_markdown(report)
    assert "(AI 生成示意图)" in md and str(img) in md
    html = to_html(report)
    assert "data:image/png;base64," in html and "AI 生成示意图" in html
