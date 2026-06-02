"""QASPER -> reranker-pair construction (pure transform; no network/datasets)."""

from __future__ import annotations

import random

from trainer.build_dataset import _norm, _paragraphs, build_pairs


def _record() -> dict:
    return {
        "id": "1706.03762",
        "full_text": {
            "section_name": ["Introduction", "Method"],
            "paragraphs": [
                ["The Transformer is a sequence model.", "We remove recurrence entirely."],
                ["We add positional encodings to inject order.", "Multi-head attention runs in parallel."],
            ],
        },
        "qas": {
            "question": ["How is positional information injected?"],
            "question_id": ["q1"],
            "answers": [{"answer": [{"evidence": ["We add positional encodings to inject order."]}]}],
        },
    }


def test_paragraphs_flatten_with_section_and_idx():
    paras = _paragraphs(_record()["full_text"])
    assert [p["idx"] for p in paras] == [0, 1, 2, 3]
    assert paras[2]["section"] == "Method"
    assert paras[2]["text"] == "We add positional encodings to inject order."


def test_build_pairs_one_positive_and_in_doc_negatives():
    pairs = build_pairs(_record(), neg_ratio=2, rng=random.Random(0))
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]

    assert len(pos) == 1
    p = pos[0]
    assert p["passage"] == "We add positional encodings to inject order."
    assert p["section"] == "Method" and p["qid"] == "q1" and p["paper_id"] == "1706.03762"
    assert p["passage_id"] == "1706.03762#p2"

    assert 1 <= len(neg) <= 2  # neg_ratio * positives, capped by the pool of 3 other paras
    assert all(n["neg_type"] == "in_doc_hard" for n in neg)
    assert all(n["passage"] != p["passage"] for n in neg)  # negatives are different paragraphs


def test_skips_questions_with_only_figure_evidence():
    rec = _record()
    rec["qas"]["answers"] = [{"answer": [{"evidence": ["FLOAT SELECTED: Figure 1 shows ..."]}]}]
    assert build_pairs(rec) == []  # no body-paragraph evidence -> no pairs


def test_norm_collapses_whitespace_and_case():
    assert _norm("  We  ADD\nEncodings ") == "we add encodings"
