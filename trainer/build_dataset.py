"""Build a paragraph-level reranker training set from QASPER.

QASPER (``allenai/qasper``) gives, for each NLP paper, information-seeking
questions annotated with their gold *evidence* paragraphs. We turn each
``(question, paragraph)`` into a relevance pair:

  - **positive** — a gold-evidence paragraph for the question.
  - **negative** — another paragraph from the *same* paper (an in-document hard
    negative; this is the right negative for paper evidence retrieval, far more
    informative than a random paragraph from a different paper).

Output: ``data/processed/{train,dev,test}.jsonl`` — one pair per line.

The dataset transform (:func:`build_pairs`) is pure and unit-tested; only
:func:`main` touches the network / the ``datasets`` library. Run it with::

    python -m trainer.build_dataset                 # full QASPER
    python -m trainer.build_dataset --max-papers 20 # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

# QASPER marks figure/table evidence with this prefix; we keep to body paragraphs.
_FLOAT_PREFIX = "FLOAT SELECTED"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _paragraphs(full_text: dict) -> List[Dict]:
    """Flatten QASPER ``full_text`` into ``[{idx, section, text}]`` (non-empty)."""
    out: List[Dict] = []
    sections = full_text.get("section_name") or []
    blocks = full_text.get("paragraphs") or []
    for section, paras in zip(sections, blocks):
        for para in paras or []:
            if para and para.strip():
                out.append({"idx": len(out), "section": section or "", "text": para.strip()})
    return out


def _evidence_norms(answers: dict) -> set:
    """Normalized gold-evidence paragraph texts for one question (all annotators)."""
    evidence = set()
    for annot in answers.get("answer", []) or []:
        for item in annot.get("evidence", []) or []:
            if item and not item.startswith(_FLOAT_PREFIX):
                evidence.add(_norm(item))
    return evidence


def _is_positive(para_norm: str, evidence: set) -> bool:
    # Exact match is the common case; containment guards against minor whitespace
    # or boundary differences between the evidence string and the paragraph.
    return any(ev == para_norm or ev in para_norm or para_norm in ev for ev in evidence)


def build_pairs(record: dict, neg_ratio: int = 4, rng: random.Random = None) -> List[dict]:
    """Pure transform: one QASPER paper record -> list of (question, passage) pairs."""
    rng = rng or random.Random(0)
    paper_id = record.get("id") or record.get("paper_id") or ""
    paras = _paragraphs(record.get("full_text") or {})
    if not paras:
        return []
    para_norms = [_norm(p["text"]) for p in paras]

    qas = record.get("qas") or {}
    questions = qas.get("question") or []
    qids = qas.get("question_id") or []
    answers_list = qas.get("answers") or []

    pairs: List[dict] = []
    for qi, question in enumerate(questions):
        qid = qids[qi] if qi < len(qids) else f"{paper_id}_q{qi}"
        answers = answers_list[qi] if qi < len(answers_list) else {}
        evidence = _evidence_norms(answers)
        if not evidence:
            continue
        positives = [i for i, pn in enumerate(para_norms) if _is_positive(pn, evidence)]
        if not positives:
            continue  # evidence didn't match any body paragraph (e.g. figure-only) -> skip
        pos_set = set(positives)
        neg_pool = [i for i in range(len(paras)) if i not in pos_set]
        k = min(len(neg_pool), max(1, neg_ratio) * len(positives))
        negatives = rng.sample(neg_pool, k) if neg_pool else []
        for i in positives:
            pairs.append(_pair(paper_id, qid, question, paras[i], 1, None))
        for i in negatives:
            pairs.append(_pair(paper_id, qid, question, paras[i], 0, "in_doc_hard"))
    return pairs


def _pair(paper_id: str, qid: str, question: str, para: dict, label: int, neg_type) -> dict:
    return {
        "qid": qid,
        "paper_id": paper_id,
        "question": question,
        "passage_id": f"{paper_id}#p{para['idx']}",
        "passage": para["text"],
        "section": para["section"],
        "label": label,
        "neg_type": neg_type,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QASPER reranker training pairs (JSONL).")
    ap.add_argument("--out-dir", default="data/processed", help="Where to write {train,dev,test}.jsonl")
    ap.add_argument("--neg-ratio", type=int, default=4, help="In-document negatives per positive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-papers", type=int, default=0, help="0 = all; small N for a quick smoke run")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("需要 datasets：pip install 'paper-mind[train]'  (或 pip install datasets)")

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("下载 / 加载 QASPER (allenai/qasper)…")
    ds = load_dataset("allenai/qasper")
    for hf_split, out_split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        if hf_split not in ds:
            continue
        out_path = out_dir / f"{out_split}.jsonl"
        n_pairs = n_papers = n_pos = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for record in ds[hf_split]:
                if args.max_papers and n_papers >= args.max_papers:
                    break
                for pair in build_pairs(record, neg_ratio=args.neg_ratio, rng=rng):
                    fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    n_pairs += 1
                    n_pos += pair["label"]
                n_papers += 1
        print(f"  {out_split}: {n_papers} papers -> {n_pairs} pairs ({n_pos} pos) -> {out_path}")


if __name__ == "__main__":
    main()
