"""Controlled QASPER negative-mining experiment with a small real cross-encoder.

Same initialization, positive pairs, negative count, optimizer and step budget
for random/BM25/dense/mixed negatives. Dev is never used for mining/training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import platform
import subprocess
import time

from evaluation.eval_retrieval import DenseScorer, _argsort_desc, _labels, _tok, load_eval
from evaluation.hybrid import reciprocal_rank_fusion
from evaluation.metrics import aggregate


def select_negatives(strategy, gold, bm_order, dense_order, count, seed):
    if count < 0:
        raise ValueError("negative count")
    if count == 0:
        return []
    pool = [i for i in bm_order if i not in gold]
    shuffled = pool.copy()
    random.Random(seed).shuffle(shuffled)
    if strategy == "random":
        order = shuffled
    elif strategy == "bm25":
        order = bm_order
    elif strategy == "dense":
        order = dense_order
    elif strategy == "mixed":
        order = [i for group in zip(bm_order, dense_order, shuffled) for i in group]
    else:
        raise ValueError(strategy)
    result = []
    for i in order:
        if i not in gold and i not in result:
            result.append(i)
        if len(result) >= min(count, len(pool)):
            break
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(instances, dense):
    from rank_bm25 import BM25Okapi
    rows = []
    for paras, questions in instances:
        embeddings = dense.embed_paragraphs(paras)
        bm = BM25Okapi([_tok(p) for p in paras])
        for question, gold in questions:
            start = time.perf_counter()
            b = _argsort_desc(bm.get_scores(_tok(question)))
            bm_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            d = _argsort_desc(dense.scores(question, embeddings))
            dense_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            h = reciprocal_rank_fusion(b, d)
            hybrid_ms = (time.perf_counter() - start) * 1000 + bm_ms + dense_ms
            rows.append({"paras": paras, "question": question, "gold": sorted(gold),
                         "orders": {"BM25": b, "Dense": d, "Hybrid": h},
                         "latency": {"BM25": bm_ms, "Dense": dense_ms, "Hybrid": hybrid_ms}})
    return rows


def run(args):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    dense = DenseScorer("sentence-transformers/all-MiniLM-L6-v2")
    train = prepare(load_eval(args.raw_dir, "train", args.train_papers), dense)
    dev = prepare(load_eval(args.raw_dir, "dev", args.dev_papers), dense)
    assert not ({digest(r["paras"]) for r in train} & {digest(r["paras"]) for r in dev})
    del dense
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    torch.manual_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, local_files_only=True)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    results, query_rows, training = {}, [], {}

    def encode(pairs):
        return tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                   truncation=True, max_length=128, return_tensors="pt")

    def evaluate(label):
        model.eval()
        ranked = {name: [] for name in ["BM25", "Dense", "Hybrid", "Dense+Reranker", "Hybrid+Reranker"]}
        times = {name: [] for name in ranked}
        for row in dev:
            orders = dict(row["orders"])
            latency = dict(row["latency"])
            for name in ("Dense", "Hybrid"):
                order = orders[name]
                top = order[:args.topk]
                start = time.perf_counter()
                with torch.no_grad():
                    scores = model(**encode([(row["question"], row["paras"][i]) for i in top])).logits.flatten().tolist()
                orders[name + "+Reranker"] = [top[i] for i in _argsort_desc(scores)] + order[len(top):]
                latency[name + "+Reranker"] = latency[name] + (time.perf_counter() - start) * 1000
            for name, order in orders.items():
                ranked[name].append(_labels(order, set(row["gold"])))
                times[name].append(latency[name])
            query_rows.append({"variant": label, "query_sha256": digest(row["question"]),
                               "gold": row["gold"], "rankings": orders, "latency_ms": latency})
        results[label] = {name: {**aggregate(rows, ks=(1, 5, 10, 20)),
                                 "latency_p50_ms": float(np.percentile(times[name], 50)),
                                 "latency_p95_ms": float(np.percentile(times[name], 95))}
                          for name, rows in ranked.items()}
        print(label, results[label]["Dense+Reranker"], flush=True)

    evaluate("untrained_head")
    for strategy in ("random", "bm25", "dense", "mixed"):
        pairs = []
        for j, row in enumerate(train):
            gold = set(row["gold"])
            negative = select_negatives(strategy, gold, row["orders"]["BM25"], row["orders"]["Dense"], 2 * len(gold), args.seed + j)
            pairs.extend((row["question"], row["paras"][i], float(i in gold)) for i in sorted(gold) + negative)
        (root / (strategy + "-pairs.jsonl")).write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")
        model.load_state_dict(initial)
        torch.manual_seed(args.seed)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        losses = []
        start = time.perf_counter()
        for epoch in range(args.epochs):
            order = list(range(len(pairs)))
            random.Random(args.seed + epoch).shuffle(order)
            for offset in range(0, len(order), 16):
                batch = [pairs[i] for i in order[offset:offset + 16]]
                optimizer.zero_grad()
                logits = model(**encode(batch)).logits.flatten()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([p[2] for p in batch]))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        checkpoint = root / "checkpoints" / strategy
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)
        training[strategy] = {"pairs": len(pairs), "pairs_sha256": digest(pairs), "loss": losses,
                              "seconds": time.perf_counter() - start,
                              "checkpoint_sha256": hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()}
        evaluate(strategy)
        (root / "progress.json").write_text(json.dumps({"results": results, "training": training}, indent=2))
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "config": vars(args),
               "train_queries": len(train), "dev_queries": len(dev),
               "train_sha256": digest([{k:v for k,v in r.items() if k != "latency"} for r in train]),
               "dev_sha256": digest([{k:v for k,v in r.items() if k != "latency"} for r in dev]),
               "source_sha256": source_hash, "revision": revision,
               "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__},
               "initialization_sha256": digest({k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in initial.items()}),
               "scope": "CPU pilot; one training seed; per-paper retrieval; no test split tuning; max_length=128",
               "timing_scope": "query scoring only; excludes corpus encoding and model load; baseline rankings/times reused across variants",
               "results": results, "training": training}
    (root / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "queries.json").write_text(json.dumps(query_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--model", default="prajjwal1/bert-tiny")
    ap.add_argument("--train-papers", type=int, default=20)
    ap.add_argument("--dev-papers", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="evaluation/results/negative-ablation")
    run(ap.parse_args())
