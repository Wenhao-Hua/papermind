"""Caching tests: report cache path/save/load, inventory, and analyze cache-hit."""

from __future__ import annotations

import papermind.analyze as analyze_mod
from papermind.cache import (
    clear_all,
    clear_paper,
    iter_cached_papers,
    latest_report_path,
    load_cached_report,
    report_cache_path,
    save_report,
)
from papermind.config import Config
from papermind.output.schema import Contributions, PaperMeta, Report
from papermind.parser.arxiv import ResolvedSource


def _report(title="T", arxiv_id="1234.5678"):
    return Report(
        paper=PaperMeta(title=title, arxiv_id=arxiv_id),
        contributions=Contributions(main_contribution="c"),
    )


def test_report_cache_path_is_model_and_module_sensitive(tmp_path):
    p1 = report_cache_path(tmp_path, "gpt-4o-mini", ["a", "b"], True)
    p2 = report_cache_path(tmp_path, "gpt-4o", ["a", "b"], True)
    p3 = report_cache_path(tmp_path, "gpt-4o-mini", ["b", "a"], True)  # order-insensitive
    p4 = report_cache_path(tmp_path, "gpt-4o-mini", ["a", "b"], False)
    assert p1 != p2 and p1 != p4
    assert p1 == p3


def test_save_and_load_round_trip(tmp_path):
    path = report_cache_path(tmp_path, "m", ["contributions"], True)
    save_report(_report(), path)
    loaded = load_cached_report(path)
    assert loaded is not None and loaded.contributions.main_contribution == "c"


def test_load_missing_or_corrupt_returns_none(tmp_path):
    assert load_cached_report(tmp_path / "nope.json") is None
    bad = tmp_path / "report-bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_cached_report(bad) is None


def test_inventory_and_clear(tmp_path):
    cfg = Config(cache_dir=str(tmp_path))
    paper_dir = tmp_path / "2307.08691"
    paper_dir.mkdir()
    (paper_dir / "metadata.json").write_text('{"title": "FA2", "arxiv_id": "2307.08691"}', encoding="utf-8")
    save_report(_report("FA2"), report_cache_path(paper_dir, "m", ["contributions"], True))

    papers = iter_cached_papers(cfg)
    assert len(papers) == 1
    assert papers[0].title == "FA2" and papers[0].has_report
    assert latest_report_path(paper_dir) is not None

    assert clear_paper("2307.08691", cfg) is True
    assert iter_cached_papers(cfg) == []
    assert clear_paper("missing", cfg) is False


def test_analyze_returns_cached_without_parsing(monkeypatch, tmp_path):
    cache_dir = tmp_path / "1234.5678"
    cache_dir.mkdir()
    (cache_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    meta = PaperMeta(title="Cached Paper", arxiv_id="1234.5678")
    resolved = ResolvedSource(meta=meta, pdf_path=cache_dir / "paper.pdf", cache_key="1234.5678", cache_dir=cache_dir)

    monkeypatch.setattr(analyze_mod, "resolve", lambda source, config: resolved)

    def _boom(*a, **k):
        raise AssertionError("parse_pdf must not run on a cache hit")

    monkeypatch.setattr(analyze_mod, "parse_pdf", _boom)

    cfg = Config(openai_key="sk-x", default_model="gpt-4o-mini")
    save_report(
        _report("Cached Paper"),
        report_cache_path(cache_dir, "gpt-4o-mini", analyze_mod.ALL_MODULES, True),
    )

    report = analyze_mod.analyze("arxiv:1234.5678", model="gpt-4o-mini", config=cfg)
    assert report.paper.title == "Cached Paper"
    assert report.contributions.main_contribution == "c"
