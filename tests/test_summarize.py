"""Tests for the one-call summarize() path (pipeline mocked — no network/LLM)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from papermind import summarize as sm
from papermind.config import Config
from papermind.output.schema import PaperMeta, Usage


def _patch_pipeline(monkeypatch, complete_json_return):
    """Stub out everything summarize() touches except its own Summary-building logic."""

    class FakeClient:
        def __init__(self, *a, **k):
            self.usage = Usage()

        def ensure_ready(self):
            pass

        def complete_json(self, system, user):
            return complete_json_return

    meta = PaperMeta(title="Attention Is All You Need")
    monkeypatch.setattr(sm, "LLMClient", FakeClient)
    monkeypatch.setattr(sm, "resolve", lambda source, config: NS(pdf_path="x.pdf", meta=meta, cache_dir="d", cache_key="k"))
    monkeypatch.setattr(sm, "parse_pdf", lambda *a, **k: NS(meta=meta))
    monkeypatch.setattr(sm, "_build_context", lambda parsed: "context")


def test_summarize_builds_summary_from_json(monkeypatch):
    _patch_pipeline(monkeypatch, {"tldr": "A transformer with only attention.", "key_points": ["no recurrence", "parallel"]})
    summary, usage = sm.summarize("https://arxiv.org/abs/1706.03762", config=Config(openai_key="sk-x"))
    assert summary.title == "Attention Is All You Need"
    assert summary.tldr == "A transformer with only attention."
    assert summary.key_points == ["no recurrence", "parallel"]
    assert isinstance(usage, Usage)


def test_summarize_handles_non_dict_response(monkeypatch):
    # A malformed model response (a list, not the expected object) must degrade to an
    # empty summary, never raise.
    _patch_pipeline(monkeypatch, ["oops", "not", "an", "object"])
    summary, _ = sm.summarize("https://arxiv.org/abs/1706.03762", config=Config(openai_key="sk-x"))
    assert summary.title == "Attention Is All You Need"
    assert summary.tldr == ""
    assert summary.key_points == []
