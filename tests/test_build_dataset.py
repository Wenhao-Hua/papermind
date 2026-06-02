"""QASPER (raw JSON) -> reranker-pair construction (pure transform; no network)."""

from __future__ import annotations

import random

from trainer.build_dataset import _norm, _paragraphs, build_pairs


def _record() -> dict:
    """One paper in raw QASPER shape (full_text = list of sections; qas = list)."""
    return {
        "title": "Attention Is All You Need",
        "full_text": [
            {"section_name": "Introduction", "paragraphs": [
                "The Transformer is a sequence model.", "We remove recurrence entirely."]},
            {"section_name": "Method", "paragraphs": [
                "We add positional encodings to inject order.", "Multi-head attention runs in parallel."]},
        ],
        "qas": [
            {
                "question": "How is positional information injected?",
                "question_id": "q1",
                "answers": [{"answer": {"evidence": ["We add positional encodings to inject order."]}}],
            }
        ],
    }


def test_paragraphs_flatten_with_section_and_idx():
    paras = _paragraphs(_record()["full_text"])
    assert [p["idx"] for p in paras] == [0, 1, 2, 3]
    assert paras[2]["section"] == "Method"
    assert paras[2]["text"] == "We add positional encodings to inject order."


def test_build_pairs_one_positive_and_in_doc_negatives():
    pairs = build_pairs("1706.03762", _record(), neg_ratio=2, rng=random.Random(0))
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]

    assert len(pos) == 1
    p = pos[0]
    assert p["passage"] == "We add positional encodings to inject order."
    assert p["section"] == "Method" and p["qid"] == "q1" and p["paper_id"] == "1706.03762"
    assert p["passage_id"] == "1706.03762#p2"

    assert 1 <= len(neg) <= 2  # neg_ratio * positives, capped by the 3 other paragraphs
    assert all(n["neg_type"] == "in_doc_hard" for n in neg)
    assert all(n["passage"] != p["passage"] for n in neg)


def test_skips_questions_with_only_figure_evidence():
    rec = _record()
    rec["qas"][0]["answers"] = [{"answer": {"evidence": ["FLOAT SELECTED: Figure 1 shows ..."]}}]
    assert build_pairs("x", rec) == []


def test_norm_collapses_whitespace_and_case():
    assert _norm("  We  ADD\nEncodings ") == "we add encodings"
