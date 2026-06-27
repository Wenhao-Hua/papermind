"""Module + schema tests with a mocked LLM client (no network)."""

from __future__ import annotations

from papermind.modules import connections, contributions, reproduction, technical
from papermind.output.schema import (
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


def test_technical_parses_formula_and_renders_math():
    client = FakeClient(
        {"technical_details": [{"name": "SDPA", "explanation": "scale by $\\sqrt{d_k}$",
                                 "formula": "\\text{softmax}(\\frac{QK^\\top}{\\sqrt{d_k}})V", "difficulty": "high"}]}
    )
    section = technical.run(None, client, "ctx")
    assert section.details[0].formula.startswith("\\text{softmax}")

    report = Report(paper=PaperMeta(title="T"), technical=section)
    md = report.to_markdown()
    assert "$$" in md and "\\sqrt{d_k}" in md  # display-math block emitted
    html = report.to_html()
    assert "MathJax" in html or "mathjax" in html  # MathJax loaded
    assert 'class="formula"' in html


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


def test_html_has_copy_bibtex_and_collapsible_points():
    from papermind.output.schema import Reproduction, SetupStep

    report = Report(
        paper=PaperMeta(title="Attention", arxiv_id="1706.03762", authors=["A. Vaswani"], year=2017),
        contributions=Contributions(main_contribution="c"),
        technical=TechnicalSection(details=[TechnicalPoint(name="SDPA", explanation="e", difficulty="high")]),
        reproduction=Reproduction(env_setup_steps=[SetupStep(step=1, title="install", command="pip install x")]),
    )
    html = report.to_html()
    assert "复制 BibTeX" in html and 'id="pm-bibtex"' in html  # copy button + payload
    assert "@article{vaswani2017attention" in html               # bibtex embedded
    assert "<details class=\"tech\" open>" in html and "<summary>" in html  # collapsible
    assert 'class="copy-code"' in html and "pip install x" in html  # command copy button
    assert 'id="pm-progress"' in html and 'id="pm-top"' in html      # progress bar + back-to-top


def test_output_base_directory_vs_prefix(tmp_path):
    from papermind.cli import _output_base

    rep = Report(paper=PaperMeta(title="X", arxiv_id="1706.03762"))
    # existing directory -> file inside, named by the paper (the bug: used to make 'dir.md')
    inside = _output_base(rep, str(tmp_path))
    assert inside.name == "1706.03762" and inside.parent == tmp_path
    # trailing separator also means "directory"
    inside2 = _output_base(rep, str(tmp_path) + "/")
    assert inside2.name == "1706.03762"
    # plain file prefix (extension stripped)
    pref = _output_base(rep, "out/report.md")
    assert str(pref).replace("\\", "/").endswith("out/report")
    # arxiv ids with '/' are sanitized into the filename
    rep2 = Report(paper=PaperMeta(title="Y", arxiv_id="hep-th/9901001"))
    assert "/" not in _output_base(rep2, str(tmp_path) + "/").name


def test_markdown_has_toc_with_anchors():
    report = Report(
        paper=PaperMeta(title="T"),
        contributions=Contributions(main_contribution="c"),
        technical=TechnicalSection(details=[TechnicalPoint(name="N", explanation="e")]),
    )
    md = report.to_markdown()
    assert "- [🎯 贡献与创新点](#contributions)" in md
    assert "- [🔬 技术细节解释](#technical)" in md
    assert '<a id="contributions"></a>' in md and '<a id="technical"></a>' in md


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


def test_reading_time_in_markdown_and_html():
    """Reading-time estimate appears in the meta line of both renderers."""
    report = Report(
        paper=PaperMeta(title="FlashAttention-2", arxiv_id="2307.08691", year=2023),
        contributions=Contributions(
            main_contribution="Reorders attention computation to reduce HBM reads/writes",
            novelty="IO-aware tiling strategy applied to both forward and backward pass",
            problem_solved="Memory bandwidth bottleneck in standard attention",
        ),
        technical=TechnicalSection(details=[
            TechnicalPoint(name="Tiling", explanation="Splits Q/K/V into blocks that fit in SRAM"),
        ]),
    )
    md = report.to_markdown()
    assert "阅读时长" in md
    assert "分钟" in md

    html_out = report.to_html()
    assert "阅读时长" in html_out
    assert "分钟" in html_out


def test_reading_time_minimum_one_minute():
    """Even an empty report gives at least 1 minute."""
    from papermind.output.html import _reading_time_minutes as _rt_html
    from papermind.output.markdown import _reading_time_minutes as _rt_md

    empty = Report(paper=PaperMeta(title="X"))
    assert _rt_html(empty) >= 1
    assert _rt_md(empty) >= 1


def test_reading_time_scales_with_content():
    """A content-rich report produces a longer estimate than a sparse one."""
    from papermind.output.markdown import _reading_time_minutes

    sparse = Report(paper=PaperMeta(title="Sparse"))
    rich_report = Report(
        paper=PaperMeta(title="Rich"),
        contributions=Contributions(
            main_contribution=" ".join(["word"] * 100),
            novelty=" ".join(["word"] * 100),
            problem_solved=" ".join(["word"] * 100),
        ),
        technical=TechnicalSection(details=[
            TechnicalPoint(name="T", explanation=" ".join(["word"] * 200)),
        ]),
    )
    assert _reading_time_minutes(rich_report) > _reading_time_minutes(sparse)


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
