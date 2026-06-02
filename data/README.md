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
pip install 'paper-mind[train]'          # installs `datasets`
python -m trainer.build_dataset          # full QASPER -> data/processed/*.jsonl
python -m trainer.build_dataset --max-papers 20   # quick smoke run
```

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
