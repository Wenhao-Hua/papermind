# PaperMind — Research Plan

> Turning PaperMind from an API-driven tool into a research-grade project: a
> self-trained **evidence reranker**, a **dataset**, and a **reproducible
> evaluation** — grafted onto the existing system, not rebuilt.

## 0. Where we start (already built)

PaperMind already covers the engineering: PDF parsing & sectioning
(`papermind/parser/`), chunking + FAISS index (`papermind/qa/index.py`), dense
retrieval (`papermind/qa/retriever.py`), grounded layered Q&A with citation
verification (`papermind/qa/chat.py`, `papermind/qa/verify.py`), four-module
structured extraction (`papermind/modules/`), multi-paper compare, FastAPI/CLI,
caching/config, and 100+ tests.

**The missing research spine** — and the whole point of this plan — is three
things: (1) a module we **train ourselves**, (2) a **dataset** behind it, and
(3) an **evaluation** with baselines and ablations that proves it helps.

## 1. Problem

Papers are long, structured, evidence-heavy documents. The real need is not
"summarize this" but **"which sentence supports this claim/method/result?"**.
PaperMind makes every output traceable to source — and this plan makes the
*find-the-evidence* step a **trained, measured** component rather than a black-box
embedding call.

**Vs. a generic RAG system:** generic RAG treats retrieval as a fixed black box
(`embed → top-k → LLM`) and never measures it. Here, evidence retrieval is a
first-class, trainable task with its own benchmark (QASPER) and metrics
(Recall@k / MRR / nDCG / Evidence-F1), plus a BM25 / dense / dense+reranker
ablation.

## 2. The trainable module (the crux)

**Main line — a paper evidence reranker (cross-encoder).** Decided over a
sentence-function classifier because it improves PaperMind's core ability
(evidence retrieval), sits on a real benchmark with published baselines, and
hits the RAG/IR direction directly.

- **Task:** score `(question, paragraph) → relevance ∈ [0,1]`. At inference,
  over-fetch candidates from recall, rerank, keep the top-k as evidence.
- **Data:** QASPER (`allenai/qasper`) — NLP papers with questions + gold
  evidence paragraphs.
  - positive = a gold-evidence paragraph for the question.
  - negative = another paragraph from the **same paper** (in-document hard
    negative). Optionally add BM25 hard negatives later.
- **Model:** fine-tune `cross-encoder/ms-marco-MiniLM-L-6-v2` (or
  `BAAI/bge-reranker-base`) via `sentence-transformers` `CrossEncoder`, pointwise
  BCE. Small, single-GPU / Colab-friendly.
- **Metrics:** Recall@{1,5,10}, MRR, nDCG@10, QASPER Evidence-F1.
- **Integration:** insert a rerank step in `papermind/qa/retriever.py`
  (over-fetch top-50 → rerank → top-5), behind a flag so default behavior is
  unchanged until the model is ready.

**Backup / phase-2 — sentence-function classifier** (Background / Objective /
Method / Result / Limitation / Other) on CSAbstruct or PubMed-RCT; SciBERT;
Accuracy + macro-F1. Useful to pre-tag sentences for cheaper/steadier extraction.
Do **not** run both in the MVP.

## 3. Data pipeline (Step 1 — implemented)

- `trainer/build_dataset.py`: QASPER → `data/processed/{train,dev,test}.jsonl`
  of `(question, paragraph, label)` pairs; in-document hard negatives;
  figure/table ("FLOAT SELECTED") evidence skipped. The transform `build_pairs`
  is pure and unit-tested (`tests/test_build_dataset.py`).
- Storage: JSONL (diffable, streamable). A small SQLite relational layer is
  optional and deferred.
- Later: hand-label ~50 real papers' evidence as an **out-of-domain test set**
  to show generalization (and dataset-building skill).

```bash
python -m trainer.build_dataset --max-papers 20   # smoke (downloads raw QASPER JSON)
python -m trainer.build_dataset                   # full
```

## 3b. Training run — Step 2 (AutoDL / Colab)

`trainer/train_reranker.py` fine-tunes the cross-encoder and prints dev
Recall@k / MRR. Pure helpers (`ranking_metrics`, `load_pairs`, `group_by_qid`)
are unit-tested; torch / sentence-transformers load lazily. Inference wrapper:
`papermind/rerank/infer.py` (`Reranker`).

On an **AutoDL** box (`/root/autodl-tmp`, GPU; torch usually preinstalled):

```bash
cd /root/autodl-tmp
git clone https://github.com/Wenhao-Hua/papermind && cd papermind   # or: git pull
export HF_ENDPOINT=https://hf-mirror.com      # HuggingFace mirror (needed in CN)
pip install -e ".[train]"                      # datasets + sentence-transformers
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -m trainer.build_dataset --max-papers 20   # data smoke -> inspect data/processed/*.jsonl
python -m trainer.train_reranker --smoke           # train smoke (minutes)

python -m trainer.build_dataset                    # full
python -m trainer.train_reranker --epochs 2        # full -> checkpoints/reranker/ + dev_metrics.json
# stronger headline number (same single GPU):
# python -m trainer.train_reranker --model BAAI/bge-reranker-base --batch-size 16 --epochs 2
```

GPU: MiniLM needs ≥4 GB; bge-reranker-base ≈6–8 GB; any AutoDL card (3090/4090/
A5000/T4) is plenty. The dev metric ranks gold above in-document distractors;
full-corpus Recall/nDCG vs BM25/dense baselines is Step 3 (`evaluation/`).

## 4. Tech stack (delta over what exists)

- Retrieval: keep FAISS + sentence-transformers; **add BM25** (`rank_bm25`) as a
  baseline / for hybrid recall; consider upgrading the bi-encoder to `bge-m3` /
  `multilingual-e5-base`.
- Training: PyTorch + HuggingFace `transformers` + `sentence-transformers`
  (`CrossEncoder`). No PEFT/LoRA — it's a small full fine-tune.
- Eval: `ranx` (or `pytrec_eval`) for Recall/MRR/nDCG — **don't hand-roll IR
  metrics**.
- PDF (optional upgrade): GROBID for stronger section/reference parsing — later.

## 5. Architecture (inference link, after rerank lands)

```
PDF → parser → paragraphs(section,page) → index (FAISS + BM25)
question → recall top-50 (dense ∪ BM25) → [TRAINED reranker] → top-5 evidence
        → LLM grounded layered answer → verify/select evidence → answer + citations
```

Training link:
```
QASPER → trainer/build_dataset.py → data/processed/*.jsonl
       → trainer/train_reranker.py (CrossEncoder, BCE) → checkpoints/reranker/
       → evaluation/eval_retrieval.py (Recall@k / MRR / nDCG / Evidence-F1) → results/
```

## 6. Directory layout (added by this plan)

```
trainer/            build_dataset.py [done] · train_reranker.py [next] · (train_classifier.py backup)
rerank/             model.py · infer.py        [next] — inference, wired into qa/retriever.py
evaluation/         eval_retrieval.py · eval_qa.py · ablation.py · results/   [step 3]
data/               raw/ · processed/ · self_annotated/  (git-ignored; see data/README.md)
docs/RESEARCH_PLAN.md   this file
```

## 7. MVP (2–3 weeks, on top of existing PaperMind)

**Do:** build_dataset [done] → train cross-encoder reranker → wire into
retriever (flagged) → `evaluation/` reports the 3-row ablation (BM25 / dense /
dense+reranker) on Recall@5, MRR, nDCG@10, Evidence-F1 → one ablation chart + a
before/after evidence demo.

**Defer:** GROBID, figure–text alignment, the classifier, Docker, OOD self-set,
UI polish.

**Show:** results table + bar chart; a single Q where reranked top-5 evidence is
visibly better; the numbers in the README.

## 8. Evaluation

- **Retrieval (core):** Recall@{1,5,10}, MRR, nDCG@10 on QASPER dev/test, plus
  QASPER **Evidence-F1** (vs published baseline). Ablation: BM25 → +dense →
  +reranker; sweep k and negative ratio.
- **Classifier (if built):** Accuracy + macro-P/R/F1 + confusion matrix; baseline
  = majority / TF-IDF+LR vs SciBERT.
- **QA:** evidence-hit rate (cited paragraph ∈ gold), traceability rate (every
  claim has a verified source — reuse `verify.py`'s `verified` flag), Answer-F1
  / small human eval.
- **System:** rerank latency vs quality gain; cache hit rate.
- Target headline: *"reranker lifts Evidence-F1 from a→b, Recall@5 +c"* with a
  chart, fully reproducible via a script.

## 9. Roadmap

| Phase | Goal | Output | ~Time | Risk |
|---|---|---|---|---|
| 1 Data | QASPER → pairs | `data/processed/*.jsonl` | done | neg-sample definition → fixed to in-doc hard |
| 2 MVP | train + wire reranker | checkpoint + flagged rerank step | 1 wk | GPU/tuning → start MiniLM |
| 3 Eval | metrics + ablation | `evaluation/results/` + chart + numbers | 4–6 d | metric bugs → use `ranx` |
| 4 Enhance (opt) | hybrid recall / classifier / OOD set | higher + generalization numbers | 1–2 wk | scope creep → stop on demand |
| 5 Open source | README + demo + reproduce script | star-ready repo | 3–4 d | demo gif recorded locally |

## 10. Risks / anti-fluff

1. **Evaluation is easiest to fake** — use `ranx`, fixed splits, public scripts;
   make sure negatives aren't secretly relevant.
2. **Don't let the "trained module" degrade into another API call** — it must be
   a real fine-tuned cross-encoder with before/after numbers.
3. **Scope creep kills it** — GROBID / figure alignment / extra datasets all go
   to phase 4. The MVP is one line: reranker + evaluation.
4. **Narrow and deep beats wide and shallow** — 80% of effort on
   reranker + QASPER eval + a 3-row ablation + one chart + one headline number.

## 11. Résumé framing

- **CN:** 面向学术论文的结构化理解与证据检索系统（含可训练的证据重排模块）
- **EN:** PaperMind — Evidence-Grounded Scientific Paper Understanding & Retrieval
- Bullets: end-to-end traceable paper Q&A (parse → hybrid recall →
  **self-trained cross-encoder reranker** → grounded layered answers);
  **built a QASPER dataset (19.4k pairs) and fine-tuned an evidence reranker**
  that lifts retrieval on QASPER dev (888 Q, full-paper candidates)
  **Recall@5 0.46→0.66, nDCG@10 0.41→0.61, MRR 0.40→0.61** over a dense
  baseline, with a BM25/dense/reranker ablation; verified citations +
  reproduction grounded in the paper's real code repo.

## 12. Results (QASPER dev, 276 papers / 888 questions)

Candidate set per question = all paragraphs of its paper. Reranker = fine-tuned
`cross-encoder/ms-marco-MiniLM-L-6-v2`; dense = `all-MiniLM-L6-v2`. Full table in
`evaluation/results/ablation.json`; reproduce with `python -m evaluation.eval_retrieval`.

| method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | F1@5 |
|---|---|---|---|---|---|---|
| BM25 | 0.109 | 0.374 | 0.554 | 0.340 | 0.341 | 0.171 |
| Dense | 0.141 | 0.456 | 0.643 | 0.401 | 0.413 | 0.215 |
| Dense+Reranker | 0.307 | 0.657 | 0.787 | 0.611 | 0.607 | 0.311 |

(F1@5 is a cutoff-based evidence F1, not the official QASPER evaluator.)
- Tags: LLM · RAG · information retrieval / reranking · long-document
  understanding · structured extraction · trainable module · full-stack · eval.
