"""Tests for multi-paper comparison: assembly, rendering, and orchestration."""

from __future__ import annotations

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


def test_comparison_csv_columns_and_rows():
    import csv
    import io

    comp = _comparison()
    text = to_csv(comp)
    rows = list(csv.reader(io.StringIO(text)))
    # header row contains all expected dimension columns
    assert rows[0] == ["标题", "arXiv", "年份", "核心贡献", "新颖之处", "关键方法", "性能/基准", "推荐硬件", "官方代码"]
    # one data row per paper
    assert rows[1][0] == "FlashAttention-2"
    assert rows[1][1] == "2307.08691"
    assert rows[1][2] == "2023"
    assert rows[1][3] == "faster attention"
    assert rows[1][5] == "Tiling"
    assert rows[2][0] == "Transformer"


def test_comparison_csv_synthesis_row():
    import csv
    import io

    comp = _comparison()
    comp.synthesis = "Both papers tackle transformers."
    text = to_csv(comp)
    rows = list(csv.reader(io.StringIO(text)))
    # blank separator + synthesis row appended after data rows
    synth_row = [r for r in rows if r and r[0] == "对比小结"]
    assert len(synth_row) == 1
    assert synth_row[0][1] == "Both papers tackle transformers."


def test_comparison_csv_no_synthesis_when_empty():
    import io
    import csv

    comp = _comparison()
    comp.synthesis = ""
    text = to_csv(comp)
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    assert all(r[0] != "对比小结" for r in rows)


def test_comparison_to_csv_writes_file(tmp_path):
    comp = _comparison()
    out = str(tmp_path / "result.csv")
    text = comp.to_csv(out)
    assert (tmp_path / "result.csv").read_text(encoding="utf-8") == text
    assert "FlashAttention-2" in text


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
