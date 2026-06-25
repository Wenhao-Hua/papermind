"""Tests for reproduction export (setup.sh / notebook) and BibTeX citation."""

from __future__ import annotations

import json

from papermind.output.cite import to_bibtex
from papermind.output.html import to_html
from papermind.output.markdown import to_markdown
from papermind.output.reproduce_export import to_notebook, to_setup_script
from papermind.output.schema import (
    CommonError,
    Contributions,
    PaperMeta,
    Reproduction,
    Report,
    SetupStep,
    TechnicalSection,
    TechnicalPoint,
    estimate_reading_minutes,
)


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


def _rich_report():
    """A report with substantial content to make reading-time estimates non-trivial."""
    return Report(
        paper=PaperMeta(title="Attention Is All You Need", arxiv_id="1706.03762", year=2017),
        contributions=Contributions(
            main_contribution="A " * 50 + "transformer architecture.",
            novelty="Self-attention " * 20 + "enables parallelism.",
            problem_solved="Removes " * 30 + "recurrence entirely.",
        ),
        technical=TechnicalSection(
            details=[
                TechnicalPoint(
                    name="Multi-Head Attention",
                    difficulty="high",
                    explanation="Each head " * 40 + "learns different relationships.",
                    analogy="Like " * 20 + "multiple perspectives.",
                )
            ]
        ),
    )


def test_estimate_reading_minutes_returns_positive_int():
    mins = estimate_reading_minutes(_rich_report())
    assert isinstance(mins, int)
    assert mins >= 1


def test_estimate_reading_minutes_empty_report_returns_one():
    report = Report(paper=PaperMeta(title="Empty"))
    assert estimate_reading_minutes(report) == 1


def test_reading_time_in_markdown_output():
    md = to_markdown(_rich_report())
    assert "min read" in md


def test_reading_time_in_html_output():
    doc = to_html(_rich_report())
    assert "min read" in doc


def test_reading_time_in_markdown_minimal_report():
    report = Report(paper=PaperMeta(title="Minimal Paper", year=2024))
    md = to_markdown(report)
    assert "min read" in md
