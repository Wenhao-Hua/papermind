"""Reranker training/inference pure helpers (no torch / sentence-transformers)."""

from __future__ import annotations

import json

from papermind.rerank.infer import order_by_score
from trainer.train_reranker import group_by_qid, load_pairs, ranking_metrics


def test_ranking_metrics_recall_and_mrr():
    # q1: first gold at rank 2; q2: gold at rank 1; q3: no gold -> skipped.
    ranked = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    m = ranking_metrics(ranked, ks=(1, 5))
    assert m["n_queries"] == 2
    assert m["Recall@1"] == 0.5     # only q2 has a gold in top-1
    assert m["Recall@5"] == 1.0     # both answerable queries hit within 5
    assert m["MRR"] == round((0.5 + 1.0) / 2, 4)


def test_ranking_metrics_all_unanswerable():
    assert ranking_metrics([[0, 0], [0]])["n_queries"] == 0


def test_group_by_qid():
    pairs = [{"qid": "a", "x": 1}, {"qid": "a", "x": 2}, {"qid": "b", "x": 3}]
    groups = group_by_qid(pairs)
    assert set(groups) == {"a", "b"} and len(groups["a"]) == 2


def test_load_pairs_roundtrip(tmp_path):
    path = tmp_path / "dev.jsonl"
    rows = [{"qid": "q1", "label": 1}, {"qid": "q1", "label": 0}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert load_pairs(path) == rows


def test_order_by_score_sorts_desc_and_truncates():
    assert order_by_score([0.1, 0.9, 0.5]) == [1, 2, 0]
    assert order_by_score([0.1, 0.9, 0.5], top_k=2) == [1, 2]
    assert order_by_score([]) == []
