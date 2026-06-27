"""Tests for reproduction export (setup.sh / notebook) and BibTeX citation."""

from __future__ import annotations

import json

from papermind.output.cite import to_bibtex
from papermind.output.reproduce_export import to_notebook, to_setup_script
from papermind.output.schema import CommonError, PaperMeta, Reproduction, Report, SetupStep


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


def test_reading_time_in_html_meta():
    from papermind.output.html import to_html
    from papermind.output.schema import Contributions

    report = Report(
        paper=PaperMeta(title="Test", year=2024),
        contributions=Contributions(
            main_contribution="这是一篇" * 100,  # 400 Chinese chars → ~1 min
            novelty="B" * 100,
            problem_solved="",
        ),
    )
    h = to_html(report)
    assert "分钟" in h


def test_reading_time_in_markdown_meta():
    from papermind.output.markdown import to_markdown
    from papermind.output.schema import Contributions

    report = Report(
        paper=PaperMeta(title="Test", year=2024),
        contributions=Contributions(main_contribution="这是一篇" * 100, novelty="", problem_solved=""),
    )
    md = to_markdown(report)
    assert "分钟" in md


def test_reading_time_minimum_one_minute():
    from papermind.output.html import _reading_minutes

    empty = Report(paper=PaperMeta(title="Empty"))
    assert _reading_minutes(empty) == 1
