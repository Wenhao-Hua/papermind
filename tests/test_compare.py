"""Tests for multi-paper comparison: assembly, rendering, and orchestration."""

from __future__ import annotations

import csv
import io

import papermind.compare as compare_mod
from papermind.compare import build_comparison
from papermind.output.compare_render import to_csv, to_html, to_markdown
from papermind.output.schema import (
    Benchmark,
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


def test_comparison_csv_is_parseable():
    csv_text = to_csv(_comparison())
    rows = list(csv.reader(io.StringIO(csv_text)))
    # header row: ["维度", "2307.08691", "1706.03762"]
    assert rows[0] == ["维度", "2307.08691", "1706.03762"]
    labels = [r[0] for r in rows[1:] if r]
    assert "标题" in labels
    assert "核心贡献" in labels
    assert "关键方法" in labels


def test_comparison_csv_contains_paper_data():
    csv_text = to_csv(_comparison())
    rows = list(csv.reader(io.StringIO(csv_text)))
    row_map = {r[0]: r[1:] for r in rows if len(r) >= 2}
    assert row_map["标题"] == ["FlashAttention-2", "Transformer"]
    assert row_map["核心贡献"][0] == "faster attention"
    assert row_map["关键方法"][0] == "Tiling"


def test_comparison_csv_via_schema_method():
    comp = _comparison()
    csv_text = comp.to_csv()
    assert "维度" in csv_text
    assert "2307.08691" in csv_text
    assert "faster attention" in csv_text


def test_comparison_csv_with_synthesis():
    from papermind.output.schema import Comparison, ComparedPaper

    comp = Comparison(
        papers=[
            ComparedPaper(title="A", arxiv_id="0001.00001"),
            ComparedPaper(title="B", arxiv_id="0002.00002"),
        ],
        synthesis="A is faster, B is more general.",
    )
    csv_text = to_csv(comp)
    assert "对比小结" in csv_text
    assert "A is faster" in csv_text
