"""Tests for reproduction export (setup.sh / notebook) and BibTeX citation."""

from __future__ import annotations

import json

from papermind.output.cite import to_bibtex
from papermind.output.html import to_html
from papermind.output.markdown import _reading_minutes, to_markdown
from papermind.output.reproduce_export import to_notebook, to_setup_script
from papermind.output.schema import (
    CommonError,
    Contributions,
    PaperMeta,
    Reproduction,
    Report,
    SetupStep,
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


def test_reading_minutes_minimum_one():
    # An empty report (no content fields) should still return at least 1 minute.
    report = Report(paper=PaperMeta(title="Empty"))
    assert _reading_minutes(report) == 1


def test_reading_minutes_scales_with_content():
    # A report with substantial text should estimate more than 1 minute.
    long_text = "这是一段关于深度学习的技术说明，描述了模型的核心创新点和方法论。" * 20
    report = Report(
        paper=PaperMeta(title="Big Paper"),
        contributions=Contributions(
            main_contribution=long_text,
            novelty=long_text,
            problem_solved=long_text,
        ),
    )
    mins = _reading_minutes(report)
    assert mins > 1


def test_reading_time_appears_in_markdown():
    report = Report(
        paper=PaperMeta(title="Test Paper", authors=["Alice"], year=2024),
        contributions=Contributions(
            main_contribution="核心贡献是提出了一种新的注意力机制。",
            novelty="该方法通过并行化显著提升了训练效率。",
            problem_solved="解决了长序列建模的计算瓶颈问题。",
        ),
    )
    md = to_markdown(report)
    assert "阅读时长" in md
    assert "分钟" in md


def test_reading_time_appears_in_html():
    report = Report(
        paper=PaperMeta(title="Test Paper", authors=["Bob"], year=2023),
        contributions=Contributions(
            main_contribution="A new attention mechanism.",
            novelty="Parallel training.",
            problem_solved="Long sequence bottleneck.",
        ),
    )
    doc = to_html(report)
    assert "阅读时长" in doc
    assert "分钟" in doc
