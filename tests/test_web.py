"""Web demo tests (skipped if FastAPI isn't installed)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import papermind.parser.arxiv as arxiv_mod  # noqa: E402
from papermind.parser.arxiv import ResolvedSource  # noqa: E402
from papermind.web import create_app  # noqa: E402


def test_healthz_reports_mode():
    client = TestClient(create_app(live=False))
    body = client.get("/healthz").json()
    assert body["ok"] is True and body["live"] is False
    assert TestClient(create_app(live=True)).get("/healthz").json()["live"] is True


def test_index_has_form_and_active_model():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "PaperMind" in r.text and "<form" in r.text
    # No model picker — the active model is shown subtly in the footer instead.
    assert "<select name='model'" not in r.text
    assert "当前模型" in r.text


def test_demo_route_renders_offline_report():
    r = TestClient(create_app()).get("/demo")
    assert r.status_code == 200
    assert "Attention Is All You Need" in r.text  # the bundled demo paper
    assert "MathJax" in r.text                      # full report rendering


def test_all_feature_tabs_present():
    html = TestClient(create_app()).get("/").text
    for href in ("/ask", "/summary", "/compare", "/reproduce", "/search"):
        assert f"href='{href}'" in html


def test_ask_blocked_in_demo_mode():
    client = TestClient(create_app(live=False))
    r = client.post("/ask", data={"source": "1706.03762", "question": "why?", "model": "", "mode": "balanced"})
    assert r.status_code == 200 and "演示模式不支持问答" in r.text


def test_search_works_without_model(monkeypatch):
    import papermind.parser.arxiv as arxiv_mod
    from papermind.output.schema import PaperMeta

    monkeypatch.setattr(arxiv_mod, "search_arxiv", lambda q, max_results=12: [
        PaperMeta(title="FlashAttention-2", arxiv_id="2307.08691", year=2023)
    ])
    r = TestClient(create_app(live=False)).post("/search", data={"query": "attention"})
    assert r.status_code == 200
    assert "2307.08691" in r.text and "/analyze?source=2307.08691" in r.text  # links to analyze


def test_analyze_demo_mode_does_not_run_when_uncached(monkeypatch, tmp_path):
    from papermind.output.schema import PaperMeta

    resolved = ResolvedSource(
        meta=PaperMeta(title="Uncached", arxiv_id="9999.99999"),
        pdf_path=tmp_path / "p.pdf",
        cache_key="9999.99999",
        cache_dir=tmp_path,  # empty -> no cached report
    )
    monkeypatch.setattr(arxiv_mod, "resolve", lambda source, config: resolved)

    client = TestClient(create_app(live=False))
    r = client.post("/analyze", data={"source": "9999.99999", "model": ""})
    assert r.status_code == 200
    assert "演示模式" in r.text  # falls back to the safe note, no analysis run
