"""Tests for arXiv search (mocked httpx) and batch analysis (mocked analyze)."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

import papermind.analyze as analyze_mod
from papermind.cli import app
from papermind.errors import SourceError
from papermind.output.schema import Contributions, PaperMeta, Report

runner = CliRunner()

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2307.08691v1</id>
    <title>FlashAttention-2</title>
    <published>2023-07-17T00:00:00Z</published>
    <author><name>Tri Dao</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>"""


class _FakeResp:
    text = _ATOM

    def raise_for_status(self):
        return None


def test_search_arxiv_parses_entries(monkeypatch):
    from papermind.parser.arxiv import search_arxiv

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp())
    results = search_arxiv("attention", max_results=5)
    assert [r.arxiv_id for r in results] == ["2307.08691v1", "1706.03762v5"]
    assert results[1].title == "Attention Is All You Need"
    assert results[1].year == 2017
    assert results[1].authors == ["Ashish Vaswani", "Noam Shazeer"]


def test_search_command(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp())
    result = runner.invoke(app, ["search", "attention", "-n", "5"])
    assert result.exit_code == 0
    assert "1706.03762" in result.stdout


def test_batch_writes_reports_and_index(monkeypatch, tmp_path):
    def fake_analyze(source, **kwargs):
        if "bad" in source:
            raise SourceError("could not fetch")
        aid = source.split(":")[-1]
        return Report(paper=PaperMeta(title=f"Paper {aid}", arxiv_id=aid), contributions=Contributions(main_contribution="c"))

    monkeypatch.setattr(analyze_mod, "analyze", fake_analyze)

    result = runner.invoke(
        app, ["batch", "arxiv:2307.08691", "arxiv:bad", "-o", str(tmp_path), "-f", "md"]
    )
    assert result.exit_code == 0
    assert (tmp_path / "2307.08691.md").exists()
    index = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert "Paper 2307.08691" in index
    assert "could not fetch" in index  # failure recorded
