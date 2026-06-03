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


def test_long_paper_context_represents_every_section():
    from papermind.analyze import _build_context

    blocks = [
        TextBlock(text=f"section {s} content " * 30, page=s + 1, section=f"S{s} 节")
        for s in range(5)
        for _ in range(8)
    ]
    parsed = ParsedPaper(
        meta=PaperMeta(title="Long Paper"),
        pages=["filler " * 800],  # full_text well over budget -> triggers aggregation
        blocks=blocks,
        sections=[],
        figures=[],
        cache_dir=Path("."),
    )
    ctx = _build_context(parsed, budget=1500)
    for s in range(5):  # no section dropped (unlike head-truncation)
        assert f"S{s} 节" in ctx
    assert len(ctx) < 1500 * 2  # bounded


def test_short_paper_context_unchanged():
    from papermind.analyze import _build_context

    parsed = ParsedPaper(
        meta=PaperMeta(title="T"),
        pages=["short body text here"],
        blocks=[TextBlock(text="short body text here", page=1, section="Intro")],
        sections=[],
        figures=[],
        cache_dir=Path("."),
    )
    ctx = _build_context(parsed, budget=1000)
    assert "short body text here" in ctx and "[Intro]" not in ctx  # full text, not aggregated


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


def test_verify_evidence_matches_and_snaps_location():
    from papermind.qa.chat import _verify_evidence
    from papermind.output.schema import EvidenceItem

    chunks = [
        Chunk(idx=0, text="we scale the dot products by 1 over sqrt of d_k to keep gradients stable",
              section="3.2.1 Scaled Dot-Product Attention", page=4),
        Chunk(idx=1, text="unrelated experiments on translation datasets", section="6 Results", page=8),
    ]
    grounded = EvidenceItem(text="we scale the dot products by 1 over sqrt of d_k", section="WRONG", page=99)
    fabricated = EvidenceItem(text="this sentence never appears anywhere in the paper at all", section="9", page=1)
    _verify_evidence([grounded, fabricated], chunks)

    assert grounded.verified is True
    assert grounded.section == "3.2.1 Scaled Dot-Product Attention" and grounded.page == 4  # snapped
    assert fabricated.verified is False


def test_verify_evidence_chinese_overlap():
    from papermind.qa.chat import _verify_evidence
    from papermind.output.schema import EvidenceItem

    chunks = [Chunk(idx=0, text="缩放点积注意力把点积除以根号 d_k 以避免梯度消失", section="3.2", page=4)]
    item = EvidenceItem(text="点积除以根号 d_k 避免梯度消失")
    _verify_evidence([item], chunks)
    assert item.verified is True and item.section == "3.2"


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


def test_generate_diagrams_parallel_assigns_each_point():
    points = [TechnicalPoint(name=f"P{i}", explanation="e") for i in range(5)]
    client = FakeClient({"mermaid": "flowchart TD\n A-->B", "caption": "图"})
    generate.generate_diagrams(points, client)
    assert all(p.figure is not None and p.figure.mermaid.startswith("flowchart") for p in points)


def test_usage_accumulation_thread_safe():
    import threading

    from papermind.config import Config
    from papermind.llm.base import LLMClient

    client = LLMClient(model="gpt-4o-mini", config=Config(openai_key="sk-x"))

    n_threads, per = 6, 100

    def hammer():
        for _ in range(per):
            client._record_usage("gpt-4o-mini", 10, 5)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert client.usage.calls == n_threads * per
    assert client.usage.prompt_tokens == n_threads * per * 10
    assert client.usage.completion_tokens == n_threads * per * 5


# --------------------------------------------------------------------------- #
# PaperIndex cache invalidation (the no-faiss early-return branches of _try_load)
# --------------------------------------------------------------------------- #
def test_paper_index_try_load_missing_files_returns_none(tmp_path):
    from papermind.qa.index import PaperIndex

    assert PaperIndex._try_load(tmp_path, "model-x") is None  # nothing cached -> rebuild


def test_paper_index_try_load_rejects_embed_model_mismatch(tmp_path):
    from papermind.qa.index import PaperIndex

    (tmp_path / "index.faiss").write_bytes(b"x")
    (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
    (tmp_path / "index_meta.json").write_text(
        '{"embed_model": "old-model", "dim": 3, "n": 0}', encoding="utf-8"
    )
    # embedding model changed since the cache was built -> must invalidate and rebuild
    assert PaperIndex._try_load(tmp_path, "new-model") is None


def test_paper_index_try_load_corrupt_meta_returns_none(tmp_path):
    from papermind.qa.index import PaperIndex

    (tmp_path / "index.faiss").write_bytes(b"x")
    (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
    (tmp_path / "index_meta.json").write_text("{not valid json", encoding="utf-8")
    # a corrupt cache must degrade to a rebuild, not crash
    assert PaperIndex._try_load(tmp_path, "m") is None


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
