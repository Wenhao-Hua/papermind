"""Fine-tune a cross-encoder reranker on the QASPER pairs from build_dataset.

Uses a plain transformers training loop (AutoModelForSequenceClassification with
a single regression head + BCE-with-logits) instead of sentence-transformers'
training API, so it's robust across the ST version churn (the saved checkpoint
still loads fine into `sentence_transformers.CrossEncoder` for inference). Runs
on a single GPU (or CPU). Prints/saves dev ranking metrics after training.

    python -m trainer.train_reranker --smoke                     # tiny sanity run
    python -m trainer.train_reranker --epochs 2                   # full (MiniLM)
    python -m trainer.train_reranker --model BAAI/bge-reranker-base --batch-size 16

The metric/loader helpers are pure and unit-tested; torch / transformers are
imported lazily inside train/evaluate so this module imports without them.

Dev metric note: each question's candidate set is its positives + a few
in-document negatives, so this measures ranking gold above hard in-document
distractors. Full-corpus Recall/nDCG vs BM25/dense baselines is evaluation/ (next).
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
    """Per query, gold labels (1/0) ordered by predicted score desc -> Recall@k & MRR.
    Queries with no positive in their candidate set are skipped."""
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
def _score_pairs(model, tok, pairs, device, max_length, batch_size):
    import torch

    model.eval()
    scores: List[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            enc = tok(
                [p["question"] for p in chunk],
                [p["passage"] for p in chunk],
                truncation=True, max_length=max_length, padding=True, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).float().cpu().reshape(-1).tolist())
    return scores


def evaluate(model, tok, dev_pairs, device, max_length=256, batch_size=64) -> Dict[str, float]:
    ranked: List[List[int]] = []
    for _qid, items in group_by_qid(dev_pairs).items():
        scores = _score_pairs(model, tok, items, device, max_length, batch_size)
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        ranked.append([items[i]["label"] for i in order])
    return ranking_metrics(ranked)


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    try:  # torch>=2.x preferred API; fall back to the older namespace
        from torch.amp import GradScaler, autocast

        def make_scaler(enabled):
            return GradScaler("cuda", enabled=enabled)

        def amp_ctx(enabled):
            return autocast("cuda", enabled=enabled)
    except Exception:  # pragma: no cover
        from torch.cuda.amp import GradScaler, autocast

        def make_scaler(enabled):
            return GradScaler(enabled=enabled)

        def amp_ctx(enabled):
            return autocast(enabled=enabled)

    data_dir = Path(args.data_dir)
    train_pairs = load_pairs(data_dir / "train.jsonl")
    dev_pairs = load_pairs(data_dir / "dev.jsonl") if (data_dir / "dev.jsonl").exists() else []
    if args.smoke:
        train_pairs, dev_pairs = train_pairs[:2000], dev_pairs[:1000]
    if not train_pairs:
        raise SystemExit(f"没有训练数据：先运行 python -m trainer.build_dataset 生成 {data_dir}/train.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} · train={len(train_pairs)} · dev={len(dev_pairs)} · model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)

    def collate(batch):
        enc = tok(
            [b["question"] for b in batch],
            [b["passage"] for b in batch],
            truncation=True, max_length=args.max_length, padding=True, return_tensors="pt",
        )
        labels = torch.tensor([float(b["label"]) for b in batch])
        return enc, labels

    loader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(loader) * args.epochs)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)
    use_amp = device == "cuda"
    scaler = make_scaler(use_amp)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            optim.zero_grad()
            with amp_ctx(use_amp):
                logits = model(**enc).logits.squeeze(-1)
                loss = bce(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")
            step += 1

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved checkpoint -> {args.out}")

    if dev_pairs:
        metrics = evaluate(model, tok, dev_pairs, device, args.max_length, max(args.batch_size, 64))
        print("dev metrics:", json.dumps(metrics, ensure_ascii=False))
        (Path(args.out) / "dev_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a cross-encoder evidence reranker on QASPER pairs.")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF cross-encoder / seq-classification base")
    ap.add_argument("--out", default="checkpoints/reranker")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--smoke", action="store_true", help="Tiny subset for a quick end-to-end sanity run")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
