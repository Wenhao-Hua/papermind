"""Tests for reproduction export (setup.sh / notebook) and BibTeX citation."""

from __future__ import annotations

import json

from papermind.output.cite import to_bibtex
from papermind.output.reading import reading_minutes
from papermind.output.reproduce_export import to_notebook, to_setup_script
from papermind.output.schema import (
    CommonError,
    Contributions,
    PaperMeta,
    RepoRef,
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


def _report_with_repo():
    # The richer, structured code_repo path (a verified RepoRef) — distinct from the
    # plain official_code fallback used by _report() above.
    return Report(
        paper=PaperMeta(title="FlashAttention", arxiv_id="2205.14135"),
        reproduction=Reproduction(
            code_repo=RepoRef(
                url="https://github.com/Dao-AILab/flash-attention.git",  # .git suffix
                source="PapersWithCode", is_official=True, stars=1000,
                install_commands=["pip install flash-attn"],
                run_commands=["python benchmark.py"],
            ),
            version_tag="v2.0",
        ),
    )


def test_setup_script_code_repo_clones_cds_and_installs():
    script = to_setup_script(_report_with_repo())
    assert "git clone https://github.com/Dao-AILab/flash-attention.git" in script
    assert "cd flash-attention" in script             # repo dir from the URL
    assert "cd flash-attention.git" not in script      # ...with the .git suffix stripped (else `cd` fails)
    assert "git checkout v2.0" in script               # version tag checked out
    assert "pip install flash-attn" in script          # real install command emitted
    assert "# python benchmark.py" in script           # run command commented out for review


def test_notebook_code_repo_clones_cds_and_installs():
    nb = to_notebook(_report_with_repo())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "!git clone https://github.com/Dao-AILab/flash-attention.git" in code
    assert "%cd flash-attention" in code
    assert "%cd flash-attention.git" not in code        # .git stripped here too
    assert "!pip install flash-attn" in code


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


def test_reading_time_minimum_one_minute():
    report = Report(paper=PaperMeta(title="Empty Paper"))
    assert reading_minutes(report) == 1


def test_reading_time_grows_with_content():
    short_report = Report(
        paper=PaperMeta(title="T"),
        contributions=Contributions(main_contribution="short", novelty="a", problem_solved="b"),
    )
    long_text = "这是一段很长的解释文字，描述了一个复杂的技术细节和算法原理。 " * 40
    long_report = Report(
        paper=PaperMeta(title="T"),
        technical=TechnicalSection(
            details=[TechnicalPoint(name="X", explanation=long_text, difficulty="high")]
        ),
    )
    assert reading_minutes(long_report) > reading_minutes(short_report)


def test_reading_time_appears_in_markdown():
    report = Report(paper=PaperMeta(title="Test Paper", year=2024))
    md = report.to_markdown()
    assert "预计阅读" in md
    assert "分钟" in md


def test_reading_time_appears_in_html():
    report = Report(paper=PaperMeta(title="Test Paper", year=2024))
    h = report.to_html()
    assert "预计阅读" in h
    assert "分钟" in h
