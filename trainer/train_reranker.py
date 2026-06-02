"""Fine-tune a cross-encoder reranker on the QASPER pairs from build_dataset.

Runs on a single GPU (or CPU). Saves to ``checkpoints/reranker/`` and prints
dev-set ranking metrics so you get a number right after training.

    python -m trainer.train_reranker --smoke                     # tiny sanity run
    python -m trainer.train_reranker --epochs 2                   # full (MiniLM)
    python -m trainer.train_reranker --model BAAI/bge-reranker-base --batch-size 16

The metric/loader helpers below are pure and unit-tested; torch and
sentence-transformers are imported lazily inside ``train`` / ``evaluate`` so this
module imports fine on a machine without them.

Note on the dev metric: build_dataset gives each question its positives plus a
few in-document negatives, so this measures whether the reranker ranks the gold
evidence above hard in-document distractors. Full-corpus Recall/nDCG over *all*
paragraphs is the job of evaluation/ (next step).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no torch)
# --------------------------------------------------------------------------- #
def load_pairs(path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_qid(pairs: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for pair in pairs:
        groups[pair["qid"]].append(pair)
    return groups


def ranking_metrics(ranked_labels: List[List[int]], ks=(1, 5, 10)) -> Dict[str, float]:
    """Given, per query, the gold labels (1/0) ordered by predicted score desc,
    return Recall@k (any gold in top-k) and MRR. Queries with no positive in their
    candidate set are skipped."""
    recall = {k: 0.0 for k in ks}
    mrr = 0.0
    n = 0
    for labels in ranked_labels:
        if not any(labels):
            continue
        n += 1
        for rank, label in enumerate(labels, start=1):
            if label:
                mrr += 1.0 / rank
                break
        for k in ks:
            if any(labels[:k]):
                recall[k] += 1.0
    if n == 0:
        return {**{f"Recall@{k}": 0.0 for k in ks}, "MRR": 0.0, "n_queries": 0}
    out = {f"Recall@{k}": round(recall[k] / n, 4) for k in ks}
    out["MRR"] = round(mrr / n, 4)
    out["n_queries"] = n
    return out


# --------------------------------------------------------------------------- #
# Training / evaluation (lazy heavy imports)
# --------------------------------------------------------------------------- #
def evaluate(model, dev_pairs: List[dict], batch_size: int = 64) -> Dict[str, float]:
    """Rerank each question's candidate paragraphs and score the ordering."""
    ranked: List[List[int]] = []
    for _qid, items in group_by_qid(dev_pairs).items():
        scores = model.predict([[it["question"], it["passage"]] for it in items], batch_size=batch_size)
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        ranked.append([items[i]["label"] for i in order])
    return ranking_metrics(ranked)


def train(args: argparse.Namespace) -> None:
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    data_dir = Path(args.data_dir)
    train_pairs = load_pairs(data_dir / "train.jsonl")
    dev_pairs = load_pairs(data_dir / "dev.jsonl") if (data_dir / "dev.jsonl").exists() else []
    if args.smoke:
        train_pairs, dev_pairs = train_pairs[:2000], dev_pairs[:1000]
    if not train_pairs:
        raise SystemExit(f"没有训练数据：先运行  python -m trainer.build_dataset  生成 {data_dir}/train.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} · train={len(train_pairs)} · dev={len(dev_pairs)} · model={args.model}")

    samples = [InputExample(texts=[p["question"], p["passage"]], label=float(p["label"])) for p in train_pairs]
    loader = DataLoader(samples, shuffle=True, batch_size=args.batch_size)
    model = CrossEncoder(args.model, num_labels=1, max_length=args.max_length, device=device)
    warmup = int(0.1 * len(loader) * args.epochs)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=warmup,
        use_amp=(device == "cuda"),
        output_path=args.out,
    )
    print(f"saved checkpoint -> {args.out}")

    if dev_pairs:
        metrics = evaluate(model, dev_pairs, batch_size=max(args.batch_size, 64))
        print("dev metrics:", json.dumps(metrics, ensure_ascii=False))
        (Path(args.out) / "dev_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a cross-encoder evidence reranker on QASPER pairs.")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF cross-encoder to fine-tune")
    ap.add_argument("--out", default="checkpoints/reranker", help="Where to save the checkpoint")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="Tiny subset for a quick end-to-end sanity run")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
