"""Pure ranking-metric tests (no torch / no data)."""

from __future__ import annotations

from evaluation.metrics import aggregate, f1_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_precision_f1_proportional():
    labels = [0, 1, 0, 1]  # 2 gold, at ranks 2 and 4
    assert recall_at_k(labels, 1) == 0.0
    assert recall_at_k(labels, 2) == 0.5
    assert recall_at_k(labels, 5) == 1.0
    assert precision_at_k(labels, 2) == 0.5
    assert f1_at_k(labels, 2) == 0.5


def test_mrr():
    assert mrr([0, 1, 0, 1]) == 0.5
    assert mrr([1, 0, 0]) == 1.0
    assert mrr([0, 0, 0]) == 0.0


def test_ndcg_known_value():
    # dcg=1/log2(3)+1/log2(5)=1.06161 ; ideal [1,1,0,0]=1.63093 ; ratio≈0.6509
    assert round(ndcg_at_k([0, 1, 0, 1], 4), 4) == 0.6509
    assert ndcg_at_k([1, 1, 0, 0], 4) == 1.0
    assert ndcg_at_k([0, 0], 4) == 0.0


def test_aggregate_skips_goldless_and_means():
    ranked = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]  # last has no gold -> skipped
    out = aggregate(ranked, ks=(1, 5))
    assert out["n_queries"] == 2
    assert out["Recall@1"] == 0.5     # query A hits @1, query B does not
    assert out["MRR"] == round((1.0 + 0.5) / 2, 4)
