"""Aggregate fixed-seed runs and paired paper-cluster bootstrap intervals.

Intervals condition on these trained checkpoints. They do not estimate the full
distribution of future training seeds or domains.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from evaluation.eval_retrieval import load_eval
from evaluation.metrics import recall_at_k


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def paired_paper_interval(deltas, groups, draws=2000, seed=1709):
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        indices = [i for group in rng.choices(groups, k=len(groups)) for i in group]
        means.append(statistics.mean(deltas[i] for i in indices))
    return [percentile(means, .025), percentile(means, .975)]


def run(paths, raw_dir, output):
    paths = [Path(p) for p in paths]
    reports = [json.loads((p / "ablation.json").read_text()) for p in paths]
    if len({r["config"]["seed"] for r in reports}) != len(reports):
        raise ValueError("Repeated training seed")
    protocols = [{k: v for k, v in r["config"].items() if k not in {"seed", "out"}} for r in reports]
    if any(p != protocols[0] for p in protocols[1:]):
        raise ValueError("Training protocols differ beyond seed/output")
    if len({r["dev_sha256"] for r in reports}) != 1:
        raise ValueError("Dev samples differ")
    queries = [json.loads((p / "queries.json").read_text()) for p in paths]
    instances = load_eval(raw_dir, "dev", reports[0]["config"]["dev_papers"])
    groups, hashes = [], []
    for _, questions in instances:
        group = []
        for question, _ in questions:
            group.append(len(hashes))
            hashes.append(hashlib.sha256(json.dumps(question, sort_keys=True).encode()).hexdigest())
        groups.append(group)
    summary = {}
    for variant in ("random", "bm25", "dense", "mixed"):
        by_seed = []
        for report, all_rows in zip(reports, queries):
            rows = [r for r in all_rows if r["variant"] == variant]
            if [r["query_sha256"] for r in rows] != hashes:
                raise ValueError("Query alignment failed")
            deltas = []
            for row in rows:
                gold = set(row["gold"])
                def metric(method):
                    return recall_at_k([int(i in gold) for i in row["rankings"][method]], 5)
                deltas.append(metric("Dense+Reranker") - metric("Dense"))
            by_seed.append(deltas)
        per_seed = [r["results"][variant]["Dense+Reranker"]["Recall@5"] for r in reports]
        mean_delta = [statistics.mean(row[i] for row in by_seed) for i in range(len(hashes))]
        summary[variant] = {"recall5_per_seed": per_seed, "mean_recall5": statistics.mean(per_seed),
                            "sample_std_recall5": statistics.stdev(per_seed),
                            "paired_mean_delta_vs_dense": statistics.mean(mean_delta),
                            "paired_paper_bootstrap_95_interval": paired_paper_interval(mean_delta, groups)}
    payload = {"training_seeds": [r["config"]["seed"] for r in reports], "papers": len(groups),
               "queries": len(hashes), "dense_recall5": reports[0]["results"]["random"]["Dense"]["Recall@5"],
               "bootstrap": {"draws": 2000, "seed": 1709, "unit": "paper", "paired": True,
                             "interpretation": "percentile interval conditional on the three trained models; not uncertainty over all training seeds"},
               "input_sha256": {str(p): hashlib.sha256((p / "ablation.json").read_bytes()).hexdigest() for p in paths},
               "summary": summary}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="At least two complete experiment directories")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="evaluation/results/seed-stability.json")
    args = ap.parse_args()
    if len(args.runs) < 2:
        ap.error("At least two distinct seeds are required")
    run(args.runs, args.raw_dir, args.out)
