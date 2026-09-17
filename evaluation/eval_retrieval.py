"""Full-corpus retrieval ablation on QASPER dev.

For each question, the candidate set is ALL paragraphs of its paper. We rank them
three ways and compare:

  1. BM25            — lexical baseline
  2. Dense           — bi-encoder cosine (the kind of recall PaperMind uses)
  3. Dense+Reranker  — dense top-K reranked by the trained cross-encoder

Reports Recall@k / MRR / nDCG@10 / F1@5 and writes evaluation/results/ablation.json.
This is the headline ablation: the lift from (2) -> (3) is the reranker's value.

    python -m evaluation.eval_retrieval --reranker checkpoints/reranker
    python -m evaluation.eval_retrieval --reranker checkpoints/reranker --max-papers 40   # quick
"""

from __future__ import annotations

import argparse
import json
import hashlib
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Tuple

from evaluation.metrics import aggregate
from evaluation.hybrid import reciprocal_rank_fusion
from trainer.build_dataset import (
    _TEST_TGZ,
    _TRAIN_DEV_TGZ,
    _evidence_norms,
    _is_positive,
    _norm,
    _paragraphs,
    _split_json,
)


def load_eval(raw_dir: str, split: str = "dev", max_papers: int = 0):
    """-> list of (paragraph_texts, [(question, gold_idx_set)])."""
    tgz = _TEST_TGZ if split == "test" else _TRAIN_DEV_TGZ  # test lives in a separate archive
    papers = _split_json(Path(raw_dir), tgz, split)
    instances: List[Tuple[List[str], List[Tuple[str, Set[int]]]]] = []
    for _pid, record in papers.items():
        if max_papers and len(instances) >= max_papers:
            break
        paras = _paragraphs(record.get("full_text") or [])
        if not paras:
            continue
        norms = [_norm(p["text"]) for p in paras]
        questions: List[Tuple[str, Set[int]]] = []
        for qa in record.get("qas") or []:
            evidence = _evidence_norms(qa.get("answers") or [])
            if not evidence:
                continue
            gold = {i for i, pn in enumerate(norms) if _is_positive(pn, evidence)}
            if gold:
                questions.append((qa.get("question") or "", gold))
        if questions:
            instances.append(([p["text"] for p in paras], questions))
    return instances


def _argsort_desc(scores) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def _labels(order: List[int], gold: Set[int]) -> List[int]:
    return [1 if i in gold else 0 for i in order]


def _tok(text: str) -> List[str]:
    return text.lower().split()


class DenseScorer:
    def __init__(self, model_name: str, query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_instruction = query_instruction  # e.g. bge: "Represent this sentence ..."

    def embed_paragraphs(self, paras: List[str]):
        return self.model.encode(paras, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)

    def scores(self, question: str, para_emb) -> List[float]:
        q = self.model.encode(
            [self.query_instruction + question], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return (para_emb @ q).tolist()


class RerankScorer:
    def __init__(self, path: str, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.max_length = max_length

    def scores(self, question: str, paras: List[str], idxs: List[int]) -> List[float]:
        import torch

        out: List[float] = []
        with torch.no_grad():
            for i in range(0, len(idxs), 64):
                chunk = idxs[i : i + 64]
                enc = self.tok(
                    [question] * len(chunk),
                    [paras[j] for j in chunk],
                    truncation=True, max_length=self.max_length, padding=True, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                out.extend(torch.sigmoid(logits).float().cpu().reshape(-1).tolist())
        return out


def run(args) -> dict:
    from rank_bm25 import BM25Okapi

    if args.rerank_topk < 1 or args.max_papers < 0:
        raise ValueError('rerank_topk must be positive and max_papers nonnegative')
    instances = load_eval(args.raw_dir, args.split, args.max_papers)
    print(f"eval papers={len(instances)} · questions={sum(len(q) for _, q in instances)}")
    if not instances:
        raise ValueError('No evaluable queries; refusing to write an empty benchmark')
    lexical_only = getattr(args, 'methods', 'all') == 'bm25'
    dense = None if lexical_only else DenseScorer(args.dense_model, args.query_instruction)
    reranker = RerankScorer(args.reranker) if args.reranker and not lexical_only else None

    ranked = {name: [] for name in ('BM25', 'Dense', 'Hybrid', 'Dense+Reranker', 'Hybrid+Reranker')}
    timings = {name: [] for name in ranked}
    per_query = []
    for paras, questions in instances:
        bm = BM25Okapi([_tok(p) for p in paras])
        para_emb = dense.embed_paragraphs(paras) if dense is not None else None
        for question, gold in questions:
            start = time.perf_counter()
            bm_order = _argsort_desc(list(bm.get_scores(_tok(question))))
            bm_ms = (time.perf_counter()-start)*1000
            timings['BM25'].append(bm_ms)
            ranked["BM25"].append(_labels(bm_order, gold))
            row = {'question_sha256':hashlib.sha256(question.encode()).hexdigest(), 'candidates':len(paras),
                   'gold':sorted(gold), 'rankings':{'BM25':bm_order}, 'latency_ms':{'BM25':bm_ms}}
            per_query.append(row)
            if dense is None:
                continue

            start = time.perf_counter()
            dense_order = _argsort_desc(dense.scores(question, para_emb))
            dense_ms = (time.perf_counter()-start)*1000
            timings['Dense'].append(dense_ms)
            ranked["Dense"].append(_labels(dense_order, gold))
            start = time.perf_counter()
            hybrid_order = reciprocal_rank_fusion(bm_order, dense_order)
            hybrid_ms = (time.perf_counter()-start)*1000 + bm_ms + dense_ms
            timings['Hybrid'].append(hybrid_ms)
            ranked['Hybrid'].append(_labels(hybrid_order, gold))
            row['rankings'].update({'Dense':dense_order,'Hybrid':hybrid_order})
            row['latency_ms'].update({'Dense':dense_ms,'Hybrid':hybrid_ms})

            if reranker is not None:
                k = min(len(paras), args.rerank_topk)
                for method, order, base_ms in [('Dense+Reranker',dense_order,dense_ms),('Hybrid+Reranker',hybrid_order,hybrid_ms)]:
                    topk = order[:k]
                    start = time.perf_counter()
                    rr = reranker.scores(question, paras, topk)
                    reordered = [topk[i] for i in _argsort_desc(rr)] + order[k:]
                    elapsed = (time.perf_counter()-start)*1000 + base_ms
                    timings[method].append(elapsed)
                    ranked[method].append(_labels(reordered, gold))
                    row['rankings'][method] = reordered
                    row['latency_ms'][method] = elapsed

    import numpy as np
    results = {name: {**aggregate(rows, ks=(1,5,10,20)),
                     'latency_p50_ms':float(np.percentile(timings[name],50)),
                     'latency_p95_ms':float(np.percentile(timings[name],95))}
               for name, rows in ranked.items() if rows}
    _print_table(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    dirty = bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()) if revision else None
    metadata = {'created_at':datetime.now(timezone.utc).isoformat(), 'git_revision':revision, 'git_dirty':dirty,
                'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'python':platform.python_version(),'platform':platform.platform(),'config':vars(args),
                'dataset':'QASPER','split':args.split,'papers':len(instances),'queries':len(per_query),
                'input_sha256':hashlib.sha256(json.dumps([(p,[(q,sorted(g)) for q,g in qs]) for p,qs in instances],ensure_ascii=False).encode()).hexdigest(),
                'timing_scope':'Sequential query scoring; excludes corpus embedding, index build, model load and network download. Includes query embedding for dense methods.',
                'not_run':[name for name,rows in ranked.items() if not rows]}
    out_path.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    out_path.with_suffix('.queries.json').write_text(json.dumps(per_query,indent=2),encoding='utf-8')
    print(f"saved -> {out_path}")
    return results


def _print_table(results: dict) -> None:
    cols = ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@10", "F1@5"]
    head = f"{'method':<16}" + "".join(f"{c:>11}" for c in cols)
    print("\n" + head)
    print("-" * len(head))
    for name, m in results.items():
        print(f"{name:<16}" + "".join(f"{m.get(c, 0):>11.4f}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description="QASPER full-corpus retrieval ablation (BM25/dense/+reranker).")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument('--methods', choices=['all','bm25'], default='all', help='bm25 runs without downloading neural models')
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--query-instruction", default="", help="prepended to the query (bge needs one)")
    ap.add_argument("--reranker", default="checkpoints/reranker", help="checkpoint dir; empty to skip")
    ap.add_argument("--rerank-topk", type=int, default=30)
    ap.add_argument("--max-papers", type=int, default=0)
    ap.add_argument("--out", default="evaluation/results/ablation.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
