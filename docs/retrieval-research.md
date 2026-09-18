# Retrieval and reranking research

PaperMind tests the Retrieval/Memory part of the Search -> Verifier -> Trajectory
-> Learning -> Retrieval/Memory portfolio. Existing paper-reading and RAG features
remain available; research evaluation is independent from generated answer quality.

## Three fixed-seed replication (42, 7, 2026)

All four negative-mining variants were really retrained under the same protocol
for two additional preselected seeds. Dense Recall@5 remains 0.4151. Values below
are Dense+Reranker Recall@5; intervals are differences from Dense.

| Negatives | Seed 42 | Seed 7 | Seed 2026 | Mean +/- SD | Paired delta 95% interval |
|---|---:|---:|---:|---:|---:|
| random | 0.3984 | 0.4167 | 0.3905 | 0.4019 +/- 0.0134 | [-0.0715, 0.0477] |
| bm25 | 0.2733 | 0.3339 | 0.3094 | 0.3055 +/- 0.0305 | [-0.1759, -0.0369] |
| dense | 0.2075 | 0.2672 | 0.2107 | 0.2285 +/- 0.0336 | [-0.2672, -0.1006] |
| mixed | 0.3166 | 0.3664 | 0.3807 | 0.3546 +/- 0.0336 | [-0.1254, 0.0085] |

Random negatives do not establish a robust gain over Dense. Bootstrap intervals
resample 40 papers with replacement, preserving questions within papers and pairing
methods, for 2,000 draws (seed 1709). They condition on these three trained models;
they are not confidence intervals over all possible training seeds. Do not treat
repeated seeds as independent evaluation questions. No new hyperparameter tuning
was performed based on these dev outcomes.

Reproduce after running trainer.negative_ablation with seeds 7 and 2026 into
`evaluation/results/negative-ablation-seed7` and `negative-ablation-seed2026`:

```sh
python -m evaluation.seed_stability evaluation/results/negative-ablation evaluation/results/negative-ablation-seed7 evaluation/results/negative-ablation-seed2026
```

Git stores each raw ranking file as queries.json.gz; decompress to queries.json
before aggregation. Loss histories and checkpoint hashes accompany each seed.
Full local suite now has 273 passing tests. Initial single-seed analysis follows
as historical context; the table above supersedes its one-seed scope limitation.

## Controlled CPU pilot

QASPER train: first 20 eligible papers, 72 queries. QASPER dev: first 40 eligible
papers, 116 queries. Train/dev paper-content hashes are disjoint. Candidate set is
all paragraphs of the question's own paper, not the whole QASPER collection.

Dense: all-MiniLM-L6-v2. Cross-encoder: prajjwal1/bert-tiny, 128-token truncation,
new regression head; 408 pairs per strategy, 2 negatives per positive, 3 epochs,
78 optimizer updates, batch 16, AdamW 5e-5, seed 42, CPU. Four strategies start
from identical encoder/head weights and use the same positives and query ordering.
Hard negatives exclude annotated positives but may still include unlabeled relevant
paragraphs. Mixed interleaves BM25, dense and random rankings, then deduplicates.

Re-ranking operates on top 20; all baselines and ablations share dev queries.
The untrained-head row is a diagnostic, not a pretrained retrieval baseline.
No model selection or hyperparameter tuning on dev was performed.

| Training negatives | Evaluation method | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| untrained_head | Dense+Reranker | 0.0273 | 0.1204 | 0.2886 | 0.8246 | 0.1922 | 0.1642 |
| untrained_head | Hybrid+Reranker | 0.0108 | 0.0692 | 0.2741 | 0.7892 | 0.1655 | 0.1421 |
| random | BM25 | 0.0797 | 0.2995 | 0.4506 | 0.6859 | 0.3249 | 0.2871 |
| random | Dense | 0.0945 | 0.4151 | 0.5952 | 0.8246 | 0.3965 | 0.3879 |
| random | Hybrid | 0.1494 | 0.3731 | 0.6060 | 0.7892 | 0.4473 | 0.4046 |
| random | Dense+Reranker | 0.1078 | 0.3984 | 0.5510 | 0.8246 | 0.4200 | 0.3753 |
| random | Hybrid+Reranker | 0.1049 | 0.3753 | 0.5351 | 0.7892 | 0.3994 | 0.3603 |
| bm25 | Dense+Reranker | 0.0663 | 0.2733 | 0.5029 | 0.8246 | 0.3050 | 0.3007 |
| bm25 | Hybrid+Reranker | 0.0608 | 0.2746 | 0.4743 | 0.7892 | 0.2926 | 0.2882 |
| dense | Dense+Reranker | 0.0274 | 0.2075 | 0.4014 | 0.8246 | 0.2419 | 0.2268 |
| dense | Hybrid+Reranker | 0.0212 | 0.1769 | 0.3746 | 0.7892 | 0.2137 | 0.2054 |
| mixed | Dense+Reranker | 0.0764 | 0.3166 | 0.5102 | 0.8246 | 0.3301 | 0.3142 |
| mixed | Hybrid+Reranker | 0.0837 | 0.2825 | 0.5244 | 0.7892 | 0.3266 | 0.3200 |

Random-negative training improves the untrained head, but its Dense+Reranker
Recall@5 remains below the Dense baseline. Hard-negative variants are worse in
this small fixed-budget run. This is a negative result, not evidence against hard
negative mining in general: limited data, one seed, a tiny encoder, truncation and
short training all constrain interpretation. Next experiments should vary training
budget, seeds and model size before drawing causal conclusions.

Recall is the fraction of annotated relevant paragraphs retrieved, averaged over
queries; it is not hit rate. MRR uses the first relevant paragraph; nDCG@10 uses
binary relevance. Scoring latency includes query encoding and re-ranking but
excludes model loading, corpus encoding and index building. Baseline rankings and
times are computed once and reused across training variants. Runs shared this
workstation with other experiments; latency is not a production performance claim.

## Reproduction

```sh
pip install -e '.[train,dev]'
# Cache all-MiniLM-L6-v2 and prajjwal1/bert-tiny, and obtain QASPER data/raw.
python -m trainer.build_dataset --max-papers 20
python -m trainer.negative_ablation --train-papers 20 --dev-papers 40 --epochs 3 --seed 42
python -m pytest tests/test_negative_ablation.py
```

The training script intentionally loads the tiny reranker from local cache; fetch
that model explicitly before an offline run. Config is recorded in
`configs/negative-ablation.json`; CLI defaults reproduce that protocol.
Results: `evaluation/results/negative-ablation/ablation.json` holds full metrics,
loss histories, data/config/source/initialization/checkpoint hashes and environment.
`queries.json` retains every ranking and scoring time. Training pairs and checkpoints
are included in the delivered archive, rather than bloating Git history.

## Historical result audit

Original ablation.json / ablation_bge_dev.json / ablation_bge_test.json are unchanged.
The prior 40-paper BM25/Dense/Hybrid ranking metrics match this fresh rerun exactly.
Their old latency values are not expected to match. Original full-corpus trained
reranker weights are absent, so those historical headline results are preserved
with hashes but **not newly reproduced**. See historical-audit.json. Do not compare
the tiny pilot with older experiments on different models/samples as a single gain.

Architecture: existing evaluator/data loader -> train-only negative mining ->
identical-initialization cross-encoder training -> per-paper held-out retrieval ->
raw rankings/metrics/report. Existing Docker and CI remain; new pure mining tests
run without training dependencies. Full local suite: 271 tests passed.
