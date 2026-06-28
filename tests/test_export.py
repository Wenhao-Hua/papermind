"""Tests for reproduction export (setup.sh / notebook), BibTeX citation, and CSV compare export."""

from __future__ import annotations

import csv
import io
import json

from papermind.output.cite import to_bibtex
from papermind.output.compare_render import to_csv as compare_to_csv
from papermind.output.reproduce_export import to_notebook, to_setup_script
from papermind.output.schema import CommonError, ComparedPaper, Comparison, PaperMeta, Reproduction, Report, SetupStep


def _report():
    return Report(
        paper=PaperMeta(title="Attention Is All You Need", arxiv_id="1706.03762", authors=["Ashish Vaswani", "Noam Shazeer"], year=2017),
        reproduction=Reproduction(
            official_code="https://github.com/tensorflow/tensor2tensor",
            version_tag="v1.0",
            requirements="PyTorch>=2.0",
            recommended_hardware="8x P100",
            key_hyperparams=["d_model=512", "h=8"],
            env_setup_steps=[
                SetupStep(step=1, title="Install deps", desc="install torch", command="pip install torch"),
                SetupStep(step=2, title="Clone", desc="", command="git clone x"),
            ],
            common_errors=[CommonError(error="loss diverges", cause="no warmup", fix_command="use noam schedule")],
            gotchas=["scale by sqrt(d_k)"],
        ),
    )


def test_setup_script_is_runnable_bash():
    script = to_setup_script(_report())
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script
    assert "git clone https://github.com/tensorflow/tensor2tensor" in script
    assert "pip install torch" in script           # step command present
    assert "# --- Step 1: Install deps ---" in script
    assert "checkout v1.0" in script               # version tag noted


def test_notebook_is_valid_nbformat():
    nb = to_notebook(_report())
    # JSON-serializable + correct shape
    json.dumps(nb)
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and nb["cells"]
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("pip install torch" in "".join(c["source"]) for c in code_cells)
    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert any("Reproduce: Attention Is All You Need" in "".join(c["source"]) for c in md_cells)


def test_setup_script_without_reproduction():
    report = Report(paper=PaperMeta(title="X"))
    assert "No reproduction information" in to_setup_script(report)


def test_bibtex_entry():
    meta = PaperMeta(title="Attention Is All You Need", arxiv_id="1706.03762", authors=["Ashish Vaswani", "Noam Shazeer"], year=2017)
    bib = to_bibtex(meta)
    assert bib.startswith("@article{vaswani2017attention,")
    assert "author       = {Ashish Vaswani and Noam Shazeer}" in bib
    assert "eprint       = {1706.03762}" in bib
    assert "archivePrefix = {arXiv}" in bib


def test_bibtex_without_arxiv_is_misc():
    meta = PaperMeta(title="Some Local Paper", authors=["Jane Doe"], year=2020)
    bib = to_bibtex(meta)
    assert bib.startswith("@misc{doe2020some,")


def _comparison():
    return Comparison(
        papers=[
            ComparedPaper(
                title="Paper A",
                arxiv_id="2301.00001",
                year=2023,
                main_contribution="Contribution A",
                novelty="Novel A",
                methods=["MethodX", "MethodY"],
                benchmark="BLEU: 40.1",
                hardware="8x A100",
                official_code="https://github.com/example/a",
            ),
            ComparedPaper(
                title="Paper B",
                arxiv_id="2301.00002",
                year=2022,
                main_contribution="Contribution B",
                novelty="Novel B",
                methods=["MethodZ"],
                benchmark=None,
                hardware=None,
                official_code=None,
            ),
        ],
        synthesis="Paper A outperforms Paper B.",
    )


def test_compare_to_csv_header_row():
    csv_text = compare_to_csv(_comparison())
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    assert rows[0] == ["维度", "2301.00001", "2301.00002"]


def test_compare_to_csv_contains_all_dimensions():
    csv_text = compare_to_csv(_comparison())
    assert "标题" in csv_text
    assert "核心贡献" in csv_text
    assert "关键方法" in csv_text
    assert "性能/基准" in csv_text


def test_compare_to_csv_values():
    csv_text = compare_to_csv(_comparison())
    reader = csv.reader(io.StringIO(csv_text))
    rows = {r[0]: r[1:] for r in reader if r}
    assert rows["标题"] == ["Paper A", "Paper B"]
    assert rows["年份"] == ["2023", "2022"]
    assert "MethodX" in rows["关键方法"][0]
    assert rows["关键方法"][1] == "MethodZ"
    assert rows["性能/基准"][1] == "-"


def test_compare_to_csv_synthesis_appended():
    csv_text = compare_to_csv(_comparison())
    assert "对比小结" in csv_text
    assert "Paper A outperforms Paper B." in csv_text


def test_compare_to_csv_no_synthesis():
    comp = Comparison(papers=_comparison().papers, synthesis="")
    csv_text = compare_to_csv(comp)
    assert "对比小结" not in csv_text


def test_comparison_to_csv_method(tmp_path):
    comp = _comparison()
    out = tmp_path / "compare.csv"
    result = comp.to_csv(str(out))
    assert out.exists()
    assert "Paper A" in result
    assert "Paper A" in out.read_text(encoding="utf-8")
