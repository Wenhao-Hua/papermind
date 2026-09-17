# Query traces and reproducible evaluation

The existing `PaperChat` flow now produces `Answer.trace` and exposes the latest
trace as `chat.last_trace` (including failures). Each trace includes embedding,
retrieval, context construction, optional reranking, and LLM timings; candidate
chunk IDs, source text, dense scores, available rerank scores, selected chunks,
and per-turn usage. Total time excludes paper download, parsing and index build.
Usage now includes query embedding as well as answer generation.

Start the existing web application (`papermind ui`); after asking a
question, follow **Inspect query traces** to `/traces`. The page reads only the
current browser session, escapes document content, and does not persist trace
passages to logs. Session expiration also removes its trace UI history.
`/metrics` exposes process-local Prometheus counters and a latency histogram.
For multiple workers, scrape each process separately; no multiprocess aggregate
is claimed. Keep metrics behind the same network policy as the application.

Structured `papermind.trace` logs contain only trace ID, status and duration.
If OpenTelemetry is installed, stage spans are emitted through its API; the host
must configure an SDK/exporter. No collector/backend is bundled. Unknown model
pricing may appear as zero in the existing Usage model: the UI explicitly labels
cost as best effort, not a billing measurement. Stream token counts may be estimated.
Only returned reranker scores are available; unre-ranked candidates display None.

## Evaluation

```bash
pip install -e '.[train,dev,web]'
# CPU-only lexical baseline, real QASPER dev data; no neural models required
python -m evaluation.eval_retrieval --methods bm25 --max-papers 40 --out evaluation/results/bm25-dev-40.json
# All five methods, with an actual trained reranker checkpoint
python -m evaluation.eval_retrieval --reranker checkpoints/reranker --out evaluation/results/full.json
# Candidate-budget ablation: use separate output paths for every experiment
python -m evaluation.eval_retrieval --reranker checkpoints/reranker --rerank-topk 10 --out evaluation/results/topk10.json
python -m evaluation.eval_retrieval --reranker checkpoints/reranker --rerank-topk 30 --out evaluation/results/topk30.json
python -m evaluation.dashboard evaluation/results/bm25-dev-40.json
```

Methods: BM25, Dense, reciprocal-rank-fused Hybrid, Dense+Reranker,
Hybrid+Reranker. BM25 tokenization and gold evidence matching deliberately reuse
the existing evaluator. RRF uses constant 60, equal weights and stable ID tie
breaking; gold labels never affect fusion. Metrics include proportional
Recall@1/5/10/20, MRR, nDCG@10 and F1@5. Query latency P50/P95 excludes model
loading, document embeddings and indexing. Hybrid time sums its sequential
components; these are not concurrent throughput measurements.

Each run saves the unchanged method-to-metrics JSON format plus `.metadata.json`
(configuration, environment, input hash, revision, sample counts, timing scope,
unrun methods) and `.queries.json` (rankings, gold indices and per-query timings).
Use a clean commit for final research runs; the input hash identifies the actual
evaluated sample even when code is being developed.

This delivery runs BM25 on 40 dev papers / 116 eligible questions. It does not
rerun existing Dense/Reranker headline results, and those numbers must not be
compared directly to this smaller lexical sample. Full five-method evaluation,
negative-sampling/training ablations, and answer faithfulness/relevance evaluation
remain follow-up work. The dashboard displays saved measurements only.
