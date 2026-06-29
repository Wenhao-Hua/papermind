"""Tests for multi-paper comparison: assembly, rendering, and orchestration."""

from __future__ import annotations

import csv
import io

import papermind.compare as compare_mod
from papermind.compare import build_comparison
from papermind.output.compare_render import to_csv, to_html, to_markdown
from papermind.output.schema import (
    Benchmark,
    ComparedPaper,
    Comparison,
    Contributions,
    PaperMeta,
    Reproduction,
    Report,
    TechnicalPoint,
    TechnicalSection,
)


def _report(title, aid, contrib, method, speedup):
    return Report(
        paper=PaperMeta(title=title, arxiv_id=aid, year=2023),
        contributions=Contributions(main_contribution=contrib, novelty="n"),
        technical=TechnicalSection(details=[TechnicalPoint(name=method, explanation="e")]),
        reproduction=Reproduction(
            official_code="https://github.com/x/y",
            recommended_hardware="A100",
            performance_benchmarks=[Benchmark(setting="seq=2k", speedup=speedup)],
        ),
    )


def _comparison():
    return build_comparison(
        [
            _report("FlashAttention-2", "2307.08691", "faster attention", "Tiling", "2.0x"),
            _report("Transformer", "1706.03762", "attention-only arch", "Self-Attention", None),
        ]
    )


def test_build_comparison_extracts_fields():
    comp = _comparison()
    assert len(comp.papers) == 2
    p0 = comp.papers[0]
    assert p0.main_contribution == "faster attention"
    assert p0.methods == ["Tiling"]
    assert p0.benchmark == "seq=2k: 2.0x"
    assert comp.papers[1].benchmark == "seq=2k"  # no speedup/result -> setting only


def test_comparison_markdown_table():
    md = to_markdown(_comparison())
    assert "| 维度 | 2307.08691 | 1706.03762 |" in md
    assert "| 核心贡献 | faster attention | attention-only arch |" in md
    assert "| 关键方法 | Tiling | Self-Attention |" in md


def test_comparison_html_table_and_links():
    html = to_html(_comparison())
    assert html.startswith("<!DOCTYPE html>")
    assert "<th>2307.08691</th>" in html
    assert "<a href='https://github.com/x/y'>" in html  # official code linked


def test_comparison_json_round_trip():
    comp = _comparison()
    data = comp.to_dict()
    assert data["papers"][0]["arxiv_id"] == "2307.08691"
    assert "usage" not in data  # usage excluded from export


def test_has_compare_modules_rejects_partial_report():
    full = _report("F", "1", "c", "M", "2x")  # contributions + technical + reproduction
    assert compare_mod._has_compare_modules(full) is True
    # a partial `analyze --only contributions` cache would blank out method/benchmark/hardware
    partial = Report(paper=PaperMeta(title="P", arxiv_id="2"),
                     contributions=Contributions(main_contribution="c", novelty="n"))
    assert compare_mod._has_compare_modules(partial) is False


def test_comparison_csv_headers_and_rows():
    csv_text = to_csv(_comparison())
    rows = list(csv.reader(io.StringIO(csv_text)))
    # First row is the header: 维度 + arxiv IDs
    assert rows[0] == ["维度", "2307.08691", "1706.03762"]
    # Verify a data row exists with expected content
    dim_labels = [r[0] for r in rows]
    assert "核心贡献" in dim_labels
    contrib_row = next(r for r in rows if r[0] == "核心贡献")
    assert contrib_row[1] == "faster attention"
    assert contrib_row[2] == "attention-only arch"


def test_comparison_csv_synthesis_appended():
    comp = _comparison()
    comp.synthesis = "Both papers improve Transformer efficiency."
    csv_text = to_csv(comp)
    assert "对比小结" in csv_text
    assert "Both papers improve Transformer efficiency." in csv_text


def test_comparison_csv_no_synthesis_when_empty():
    csv_text = to_csv(_comparison())
    assert "对比小结" not in csv_text


def test_comparison_to_csv_method_matches_render(tmp_path):
    comp = _comparison()
    from papermind.output.compare_render import to_csv as render_csv

    assert comp.to_csv() == render_csv(comp)

    out = tmp_path / "out.csv"
    comp.to_csv(str(out))
    assert out.read_text(encoding="utf-8") == render_csv(comp)


def test_comparison_csv_neutralises_formula_injection():
    # Cells come from LLM-extracted paper text; a value starting with = + - @ must be
    # prefixed so Excel/LibreOffice don't evaluate it as a formula (CWE-1236).
    comp = Comparison(
        papers=[ComparedPaper(title="=HYPERLINK(0)", arxiv_id="1", main_contribution="@SUM(1+1)",
                              methods=["+cmd|calc"])],
        synthesis="-1+1",
    )
    rows = {r[0]: r for r in csv.reader(io.StringIO(to_csv(comp))) if r}
    assert rows["标题"][1] == "'=HYPERLINK(0)"     # leading = neutralised
    assert rows["核心贡献"][1] == "'@SUM(1+1)"       # leading @ neutralised
    assert rows["关键方法"][1] == "'+cmd|calc"        # leading + neutralised
    assert rows["对比小结"][1] == "'-1+1"             # synthesis leading - neutralised


def test_compare_orchestration_reuses_mocked_analyze(monkeypatch):
    reports = {
        "arxiv:2307.08691": _report("FlashAttention-2", "2307.08691", "faster", "Tiling", "2.0x"),
        "arxiv:1706.03762": _report("Transformer", "1706.03762", "attention", "MHA", None),
    }

    def fake_report_for(source, model, refresh, console, config):
        return reports[source]

    monkeypatch.setattr(compare_mod, "_report_for", fake_report_for)

    comp = compare_mod.compare(list(reports.keys()), synthesize=False)
    assert [p.title for p in comp.papers] == ["FlashAttention-2", "Transformer"]
    assert comp.synthesis == ""  # synthesis skipped -> no LLM call
    assert comp.usage is not None
