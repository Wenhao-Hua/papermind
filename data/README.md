# data/

Datasets for PaperMind's trainable modules. Nothing here is committed (see
`.gitignore`) — it's downloaded or generated.

```
data/
  raw/                 # raw downloads (QASPER is fetched via 🤗 datasets cache, usually not here)
  processed/           # generated training pairs: train.jsonl / dev.jsonl / test.jsonl
  self_annotated/      # small hand-labeled out-of-domain test set (later)
```

## Build the reranker training set (QASPER → JSONL)

```bash
# No extra deps needed — QASPER's raw JSON is fetched via httpx (a core dep).
python -m trainer.build_dataset                   # full QASPER -> data/processed/*.jsonl
python -m trainer.build_dataset --max-papers 20   # quick smoke run
```

> Downloads the official `.tgz` from AllenAI S3 into `data/raw/` (cached/reused).
> If S3 is slow/blocked, `wget` the two `.tgz` into `data/raw/` manually and re-run.

Each line of `processed/*.jsonl` is one `(question, paragraph)` relevance pair:

```json
{"qid": "...", "paper_id": "1706.03762", "question": "How is order injected?",
 "passage_id": "1706.03762#p37", "passage": "We add positional encodings ...",
 "section": "3.5 Positional Encoding", "label": 1, "neg_type": null}
```

- `label`: `1` = gold evidence paragraph for the question; `0` = negative.
- `neg_type`: `"in_doc_hard"` for negatives sampled from the same paper.

See [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md) for how this feeds the
cross-encoder reranker and the evaluation plan.
