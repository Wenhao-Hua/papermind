"""Tests for reproduction export (setup.sh / notebook) and BibTeX citation."""

from __future__ import annotations

import json

from papermind.output.cite import to_bibtex
from papermind.output.markdown import estimate_reading_min, to_markdown
from papermind.output.html import to_html
from papermind.output.reproduce_export import to_notebook, to_setup_script
from papermind.output.schema import (
    CommonError,
    Contributions,
    PaperMeta,
    Reproduction,
    Report,
    SetupStep,
    TechnicalPoint,
    TechnicalSection,
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


def _rich_report() -> Report:
    """A report with substantive content for reading-time tests."""
    return Report(
        paper=PaperMeta(title="Attention Is All You Need", arxiv_id="1706.03762", year=2017),
        contributions=Contributions(
            main_contribution="Introduces the Transformer architecture based solely on attention mechanisms.",
            novelty="Replaces recurrence and convolutions with multi-head self-attention for sequence modelling.",
            problem_solved="Eliminates sequential computation bottleneck of RNNs enabling full parallelisation.",
        ),
        technical=TechnicalSection(
            details=[
                TechnicalPoint(
                    name="Multi-Head Attention",
                    explanation=(
                        "Queries keys and values are linearly projected h times and attention is computed "
                        "in parallel over all heads then concatenated and projected again."
                    ),
                    analogy="Like looking at a document through multiple lenses simultaneously.",
                    difficulty="mid",
                ),
            ]
        ),
    )


def test_reading_time_estimate_is_positive():
    report = _rich_report()
    mins = estimate_reading_min(report)
    assert mins >= 1


def test_reading_time_empty_report_is_one_minute():
    report = Report(paper=PaperMeta(title="Empty"))
    assert estimate_reading_min(report) == 1


def test_markdown_contains_reading_time():
    md = to_markdown(_rich_report())
    assert "阅读时长" in md
    assert "分钟" in md


def test_html_contains_reading_time():
    h = to_html(_rich_report())
    assert "阅读时长" in h
    assert "分钟" in h
