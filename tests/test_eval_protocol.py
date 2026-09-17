import argparse
import json
from evaluation import eval_retrieval as ev
from evaluation.dashboard import render


def test_all_methods_produce_complete_rankings_and_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(ev,'load_eval',lambda *a: [(['cat mat','dog food','cat food'],[('cat',{0,2})])])
    class Dense:
        def __init__(self,*a): pass
        def embed_paragraphs(self,paras): return None
        def scores(self,q,emb): return [0.7,0.9,0.6]
    class Rerank:
        def __init__(self,*a): pass
        def scores(self,q,paras,ids): return [float('cat' in paras[i]) for i in ids]
    monkeypatch.setattr(ev,'DenseScorer',Dense)
    monkeypatch.setattr(ev,'RerankScorer',Rerank)
    out=tmp_path/'result.json'
    args=argparse.Namespace(raw_dir='unused',split='dev',max_papers=1,methods='all',dense_model='stub',query_instruction='',reranker='stub',rerank_topk=2,out=str(out))
    result=ev.run(args)
    assert set(result)=={'BM25','Dense','Hybrid','Dense+Reranker','Hybrid+Reranker'}
    rows=json.loads(out.with_suffix('.queries.json').read_text())
    assert all(sorted(ranking)==[0,1,2] for ranking in rows[0]['rankings'].values())
    assert all(row['n_queries']==1 for row in result.values())
    metadata=json.loads(out.with_suffix('.metadata.json').read_text())
    assert metadata['not_run']==[]
    assert len(metadata['input_sha256'])==64
    assert 'Hybrid+Reranker' in render(out)
