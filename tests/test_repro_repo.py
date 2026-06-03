"""Paper -> real code repo grounding (offline parts only; no network)."""

from __future__ import annotations

from types import SimpleNamespace

from papermind.output.schema import PaperMeta, Reproduction, RepoRef, Report
from papermind.repro import repo as R


def test_scan_github_finds_skips_and_cleans():
    assert R._scan_github("see code at https://github.com/HazyResearch/flash-attention .") == \
        "https://github.com/HazyResearch/flash-attention"
    # trailing .git is stripped
    assert R._scan_github("https://github.com/foo/bar.git") == "https://github.com/foo/bar"
    # non-repo github paths are skipped
    assert R._scan_github("https://github.com/blog/announcement") is None
    assert R._scan_github("no links here") is None


def test_scan_github_recovers_pdf_mangled_urls():
    # PDF text extraction mangles printed URLs; the scanner must still recover owner/repo.
    # ligature glyph: "tensorﬂow" (single ﬂ char) -> tensorflow
    assert R._scan_github("https://github.com/tensorﬂow/tensor2tensor") == \
        "https://github.com/tensorflow/tensor2tensor"
    # line-wrap: a newline lands right after the host slash
    assert R._scan_github("code at https://github.com/\ngoogle-research/bert here") == \
        "https://github.com/google-research/bert"
    # mangled scheme + www prefix ("htt" dropped by extraction)
    assert R._scan_github("p://www.github.com/goodfeli/adversarial") == \
        "https://github.com/goodfeli/adversarial"


def test_find_repo_url_prefers_paper_link_without_network():
    # A link in the text wins and short-circuits before any PapersWithCode call.
    url, source, official = R.find_repo_url(
        "Our code is available at https://github.com/openai/whisper", None, "2212.04356"
    )
    assert url == "https://github.com/openai/whisper"
    assert source == "论文原文链接" and official is True


def test_extract_run_commands_is_precise():
    readme = (
        "## Usage\n"
        "```\n"
        "python train.py --config configs/base.yaml\n"
        "$ bash scripts/run.sh\n"
        "pip install torch\n"  # install, not a run command -> ignored
        "torchrun --nproc_per_node 8 main.py\n"
        "```\n"
        "Note: python is required.\n"  # prose -> ignored (no script)
    )
    cmds = R._extract_run_commands(readme)
    assert "python train.py --config configs/base.yaml" in cmds
    assert "bash scripts/run.sh" in cmds
    assert "torchrun --nproc_per_node 8 main.py" in cmds
    assert all("python is required" not in c and "pip install" not in c for c in cmds)


def test_find_and_inspect_repo_offline_returns_none():
    # No link in text and no arxiv id -> nothing to look up, no network touched.
    parsed = SimpleNamespace(full_text="a paper with no code link", meta=PaperMeta(title="X"))
    assert R.find_and_inspect_repo(parsed) is None


def test_setup_script_grounds_in_verified_repo():
    repro = Reproduction(
        code_repo=RepoRef(
            url="https://github.com/foo/bar",
            source="论文原文链接",
            is_official=True,
            stars=1234,
            dep_file="requirements.txt",
            install_commands=["pip install -r requirements.txt"],
            run_commands=["python train.py --config a.yaml"],
        )
    )
    report = Report(paper=PaperMeta(title="T", arxiv_id="1234.5678"), reproduction=repro)
    sh = report.to_setup_script()
    assert "git clone https://github.com/foo/bar" in sh
    assert "cd bar" in sh
    assert "pip install -r requirements.txt" in sh
    assert "python train.py --config a.yaml" in sh  # surfaced as a run comment

    nb = report.to_notebook()
    flat = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "git clone https://github.com/foo/bar" in flat
