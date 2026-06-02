"""Build paragraph-level reranker training pairs from QASPER (raw AllenAI JSON).

We download QASPER's official JSON directly from AllenAI's S3 — newer
``datasets`` versions dropped loading-script support (and the raw files are
stable, reproducible, and dependency-free). Each ``(question, paragraph)``
becomes a relevance pair:

  - **positive** — a gold-evidence paragraph for the question.
  - **negative** — another paragraph from the *same* paper (in-document hard
    negative).

Output: ``data/processed/{train,dev,test}.jsonl`` — one pair per line.

    python -m trainer.build_dataset                  # download + build all splits
    python -m trainer.build_dataset --max-papers 20  # quick smoke

If S3 is slow/blocked, download the two .tgz manually into ``data/raw/`` (same
filenames) and re-run — existing files are reused, nothing is re-downloaded.

The transform (:func:`build_pairs`) is pure and unit-tested; only :func:`main`
touches the network.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Dict, List

# QASPER marks figure/table evidence with this prefix; we keep to body paragraphs.
_FLOAT_PREFIX = "FLOAT SELECTED"

_BASE = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
_TRAIN_DEV_TGZ = f"{_BASE}/qasper-train-dev-v0.3.tgz"
_TEST_TGZ = f"{_BASE}/qasper-test-and-evaluator-v0.3.tgz"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _paragraphs(full_text: list) -> List[Dict]:
    """Flatten raw QASPER ``full_text`` (list of {section_name, paragraphs})."""
    out: List[Dict] = []
    for section in full_text or []:
        name = (section or {}).get("section_name") or ""
        for para in (section or {}).get("paragraphs") or []:
            if para and para.strip():
                out.append({"idx": len(out), "section": name, "text": para.strip()})
    return out


def _evidence_norms(answers: list) -> set:
    """Normalized gold-evidence paragraph texts for one question (all annotators)."""
    evidence = set()
    for annotation in answers or []:
        answer = (annotation or {}).get("answer") or {}
        for item in answer.get("evidence") or []:
            if item and not item.startswith(_FLOAT_PREFIX):
                evidence.add(_norm(item))
    return evidence


def _is_positive(para_norm: str, evidence: set) -> bool:
    return any(ev == para_norm or ev in para_norm or para_norm in ev for ev in evidence)


def build_pairs(paper_id: str, record: dict, neg_ratio: int = 4, rng: random.Random = None) -> List[dict]:
    """Pure transform: one raw QASPER paper -> list of (question, passage) pairs."""
    rng = rng or random.Random(0)
    paras = _paragraphs(record.get("full_text") or [])
    if not paras:
        return []
    para_norms = [_norm(p["text"]) for p in paras]

    pairs: List[dict] = []
    for qa in record.get("qas") or []:
        question = qa.get("question") or ""
        qid = qa.get("question_id") or ""
        evidence = _evidence_norms(qa.get("answers") or [])
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


# --------------------------------------------------------------------------- #
# Download + parse (network)
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {url} …")
    with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    return dest


def _split_json(raw_dir: Path, tgz_url: str, member_keyword: str) -> dict:
    """Download a QASPER .tgz (if absent) and return the JSON dict for the split."""
    tgz = _download(tgz_url, raw_dir / tgz_url.split("/")[-1])
    with tarfile.open(tgz) as tf:
        member = next(
            (n for n in tf.getnames() if member_keyword in n and n.endswith(".json")), None
        )
        if member is None:
            raise SystemExit(f"在 {tgz} 里找不到包含 '{member_keyword}' 的 .json")
        with tf.extractfile(member) as fh:
            return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QASPER reranker training pairs (JSONL) from raw JSON.")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/raw", help="Where the .tgz are cached")
    ap.add_argument("--neg-ratio", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-papers", type=int, default=0, help="0 = all; small N for a quick smoke run")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    splits = [
        ("train", _TRAIN_DEV_TGZ, "train"),
        ("dev", _TRAIN_DEV_TGZ, "dev"),
        ("test", _TEST_TGZ, "test"),
    ]
    for out_split, tgz_url, keyword in splits:
        try:
            papers = _split_json(raw_dir, tgz_url, keyword)
        except Exception as exc:  # noqa: BLE001 - network/extract failures should be clear
            print(f"  [skip] {out_split}: {type(exc).__name__}: {exc}")
            continue
        out_path = out_dir / f"{out_split}.jsonl"
        n_pairs = n_papers = n_pos = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for paper_id, record in papers.items():
                if args.max_papers and n_papers >= args.max_papers:
                    break
                for pair in build_pairs(paper_id, record, neg_ratio=args.neg_ratio, rng=rng):
                    fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    n_pairs += 1
                    n_pos += pair["label"]
                n_papers += 1
        print(f"  {out_split}: {n_papers} papers -> {n_pairs} pairs ({n_pos} pos) -> {out_path}")


if __name__ == "__main__":
    main()
