"""Tests for the TL;DR summary parsing and section-focused retrieval."""

from __future__ import annotations

from papermind.modules._util import str_list
from papermind.output.schema import Summary
from papermind.qa.index import Chunk
from papermind.qa.retriever import Retriever


def test_summary_builds_from_json():
    # Mirrors what summarize() does with the model's JSON, without any network.
    data = {"tldr": "A new attention architecture.", "key_points": ["self-attention", "", "multi-head"]}
    summary = Summary(title="Attention", tldr=str(data["tldr"]), key_points=str_list(data["key_points"]))
    assert summary.tldr == "A new attention architecture."
    assert summary.key_points == ["self-attention", "multi-head"]  # blanks dropped


class _FakeIndex:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, vec, k=5):
        return [(c, 0.9 - i * 0.01) for i, c in enumerate(self._chunks)][:k]


class _FakeClient:
    def embed(self, texts):
        return [[0.0]]


def _retriever():
    chunks = [
        Chunk(idx=0, text="intro", section="1 Introduction", page=1),
        Chunk(idx=1, text="sdpa", section="3.2 Attention", page=4),
        Chunk(idx=2, text="multihead", section="3.2 Attention", page=4),
        Chunk(idx=3, text="exp", section="4 Experiments", page=7),
    ]
    return Retriever(_FakeIndex(chunks), _FakeClient())


def test_section_filter_keeps_only_matching():
    results = _retriever().retrieve("how does attention work", k=2, section="3.2")
    assert results
    assert all("3.2" in c.section for c, _ in results)


def test_section_filter_falls_back_when_no_match():
    results = _retriever().retrieve("q", k=2, section="9.9")  # no such section
    assert len(results) == 2  # falls back to unfiltered top-k


def test_no_section_returns_top_k():
    results = _retriever().retrieve("q", k=3)
    assert len(results) == 3
