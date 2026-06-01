"""Module + schema tests with a mocked LLM client (no network)."""

from __future__ import annotations

from papermind.modules import connections, contributions, reproduction, technical
from papermind.output.schema import (
    Connection,
    Contributions,
    PaperMeta,
    Report,
    Source,
    TechnicalPoint,
    TechnicalSection,
)


class FakeClient:
    """Returns a canned JSON object regardless of prompt."""

    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, system, user, **kwargs):
        return self.payload


def test_contributions_run_parses_sources():
    client = FakeClient(
        {
            "main_contribution": "faster exact attention",
            "novelty": "better parallelism",
            "problem_solved": "low occupancy",
            "sources": [{"text": "we propose", "section": "Abstract", "page": "1"}],
        }
    )
    result = contributions.run(None, client, "ctx")
    assert result.main_contribution == "faster exact attention"
    assert result.sources[0].page == 1  # coerced from string


def test_technical_run_clamps_difficulty_and_count():
    items = [
        {"name": f"point{i}", "explanation": "e", "difficulty": "bogus", "page": None}
        for i in range(10)
    ]
    client = FakeClient({"technical_details": items})
    section = technical.run(None, client, "ctx", max_points=4)
    assert len(section.details) == 4
    assert section.details[0].difficulty == "mid"  # invalid -> default


def test_connections_drops_bad_links():
    client = FakeClient(
        {
            "connections": [
                {"concept": "Attention", "paper": "Vaswani 2017", "arxiv_link": "not-a-url", "relationship": "improves"},
                {"concept": "MHA", "paper": "x", "arxiv_link": "https://arxiv.org/abs/1706.03762", "relationship": "y"},
            ]
        }
    )
    section = connections.run(None, client, "ctx")
    assert section.related_works[0].arxiv_link is None
    assert section.related_works[1].arxiv_link.startswith("https://")


def test_reproduction_run_full():
    client = FakeClient(
        {
            "official_code": "https://github.com/x/y",
            "key_hyperparams": ["block_size=64", ""],
            "env_setup_steps": [{"title": "install", "command": "pip install x"}],
            "performance_benchmarks": [{"setting": "seq=2048", "speedup": "2x"}],
            "datasets": [{"name": "WikiText", "purpose": "lm"}],
            "common_errors": [{"error": "fp16 only", "fix_command": "model.half()"}],
            "gotchas": ["compiles slowly"],
        }
    )
    repro = reproduction.run(None, client, "ctx")
    assert repro.official_code == "https://github.com/x/y"
    assert repro.key_hyperparams == ["block_size=64"]  # empty dropped
    assert repro.env_setup_steps[0].step == 1  # auto-numbered
    assert repro.performance_benchmarks[0].speedup == "2x"


def test_report_json_round_trip():
    report = Report(
        paper=PaperMeta(title="T", arxiv_id="2307.08691", pdf_url="https://arxiv.org/pdf/2307.08691.pdf"),
        contributions=Contributions(main_contribution="c", sources=[Source(text="s", section="Abstract", page=1)]),
        technical=TechnicalSection(details=[TechnicalPoint(name="n", explanation="e", difficulty="high")]),
    )
    data = report.to_dict()
    assert data["technical_details"][0]["name"] == "n"
    restored = Report.from_dict(data)
    assert restored.technical.details[0].name == "n"
    assert restored.contributions.sources[0].page == 1


def test_markdown_contains_links_and_figure():
    report = Report(
        paper=PaperMeta(title="T", pdf_url="https://arxiv.org/pdf/2307.08691.pdf"),
        technical=TechnicalSection(
            details=[
                TechnicalPoint(
                    name="Tiling",
                    explanation="block compute",
                    source_section="Section 3.1",
                    page=5,
                )
            ]
        ),
    )
    md = report.to_markdown()
    assert "#page=5" in md
    assert "Tiling" in md


def test_page_link():
    meta = PaperMeta(title="t", pdf_url="https://arxiv.org/pdf/x.pdf")
    assert meta.page_link(5) == "https://arxiv.org/pdf/x.pdf#page=5"
    assert PaperMeta(title="t").page_link(5) is None


def test_html_export_self_contained(tmp_path):
    from papermind.output.schema import Figure

    img = tmp_path / "fig1.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fakepngdata")
    report = Report(
        paper=PaperMeta(title="FlashAttention-2 <test>", arxiv_id="2307.08691", pdf_url="https://arxiv.org/pdf/2307.08691.pdf"),
        technical=TechnicalSection(
            details=[
                TechnicalPoint(name="Tiling", explanation="block compute", difficulty="high",
                               figure=Figure(type="original", image_path=str(img), caption="Fig 1")),
                TechnicalPoint(name="Softmax", explanation="x", difficulty="mid",
                               figure=Figure(type="ai_generated", mermaid="flowchart TD\n A-->B", caption="示意图")),
            ]
        ),
    )
    out = tmp_path / "r.html"
    doc = report.to_html(str(out))
    assert out.exists()
    assert doc.startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," in doc          # original image inlined -> shareable
    assert '<pre class="mermaid">flowchart TD' in doc  # mermaid rendered as diagram
    assert "&lt;test&gt;" in doc                      # title HTML-escaped


def test_demo_plays_offline_instantly():
    import io

    from rich.console import Console

    from papermind.demo import build_demo_answer, build_demo_report, play

    assert build_demo_report().paper.arxiv_id == "1706.03762"
    assert build_demo_answer().segments[0].kind == "fact"

    buf = io.StringIO()
    console = Console(file=buf, width=100, legacy_windows=False)
    play(console, speed=0)  # speed=0 -> no sleeps, fully offline
    out = buf.getvalue()
    assert "Attention Is All You Need" in out
    assert "论文事实" in out and "基于论文的推理" in out  # layered answer shown
    assert "原文依据" in out and "Section 3.2.1" in out
