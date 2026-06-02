"""Reranker wiring into the retriever (fakes; no torch/sentence-transformers)."""

from __future__ import annotations

from papermind.qa.index import Chunk
from papermind.qa.retriever import Retriever
from papermind.rerank.infer import maybe_load_reranker


class _FakeIndex:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, _query_vec, k=5):
        return [(c, 1.0 - i * 0.1) for i, c in enumerate(self._chunks)][:k]


class _FakeClient:
    def embed(self, _texts):
        return [[0.0]]  # ignored by _FakeIndex


class _ReverseReranker:
    """Stand-in cross-encoder: reverses the candidate order (proves reordering)."""

    def rerank(self, _question, passages, top_k=5):
        order = list(range(len(passages)))[::-1][:top_k]
        return [(i, 1.0) for i in order]


def _chunks():
    secs = ["A", "B", "A", "B", "A"]
    return [Chunk(idx=i, text=f"c{i}", section=secs[i], page=i + 1) for i in range(5)]


def test_retrieve_unchanged_without_reranker():
    r = Retriever(_FakeIndex(_chunks()), _FakeClient())
    out = r.retrieve("q", k=3)
    assert [c.idx for c, _ in out] == [0, 1, 2]


def test_retrieve_reranks_and_truncates():
    r = Retriever(_FakeIndex(_chunks()), _FakeClient(), reranker=_ReverseReranker(), rerank_candidates=10)
    out = r.retrieve("q", k=3)
    assert [c.idx for c, _ in out] == [4, 3, 2]  # reversed, top-3


def test_section_filter_without_reranker():
    r = Retriever(_FakeIndex(_chunks()), _FakeClient())
    out = r.retrieve("q", k=2, section="A")
    assert [c.idx for c, _ in out] == [0, 2]  # only section-A chunks, top-2


def test_maybe_load_reranker_is_safe():
    assert maybe_load_reranker(None) is None
    assert maybe_load_reranker("/no/such/checkpoint/dir") is None
