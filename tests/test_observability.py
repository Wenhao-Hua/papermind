from types import SimpleNamespace
import pytest
from papermind.observability import current_trace, query_trace, span, prometheus_text
from papermind.qa.index import Chunk
from papermind.qa.retriever import Retriever
from papermind.trace_view import render_traces
from evaluation.hybrid import reciprocal_rank_fusion


def test_retrieval_scores_and_stages():
    chunks=[Chunk(0,'alpha','Intro',1),Chunk(1,'beta','Methods',2)]
    index=SimpleNamespace(search=lambda v,k: [(chunks[0],0.9),(chunks[1],0.7)])
    client=SimpleNamespace(embed=lambda x: [[1,0]])
    reranker=SimpleNamespace(rerank=lambda q,texts,top_k: [(1,0.99)])
    with query_trace() as trace:
        results=Retriever(index,client,reranker).retrieve('q',k=1)
    assert results[0][0].idx==1
    assert trace.selected_chunk_ids==[1]
    assert trace.candidates[1]['dense_score']==0.7
    assert trace.candidates[1]['rerank_score']==0.99
    assert [s['name'] for s in trace.spans]==['embedding','retrieval','reranking']
    assert trace.total_ms>=0 and trace.status=='ok'
    assert current_trace() is None


def test_failure_recorded_and_context_reset():
    with pytest.raises(ValueError):
        with query_trace() as trace:
            with span('llm'):
                raise ValueError('secret')
    assert trace.status=='error'
    assert trace.spans[0]['status']=='error'
    assert current_trace() is None
    assert 'secret' not in prometheus_text()


def test_trace_html_escapes_untrusted_passages():
    with query_trace() as trace:
        trace.candidates=[{'text':'<script>alert(1)</script>'}]
    html=render_traces([trace.to_dict()])
    assert '<script>' not in html
    assert '&lt;script&gt;' in html


def test_rrf_deterministic_and_gold_independent():
    assert reciprocal_rank_fusion([0,1,2],[2,1,0])==[0,2,1]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([1,1])


def test_metrics_and_trace_web_endpoints():
    from fastapi.testclient import TestClient
    from papermind.web import create_app
    c=TestClient(create_app())
    assert c.get('/metrics').status_code==200
    assert 'papermind_queries_total' in c.get('/metrics').text
    assert 'No traces yet' in c.get('/traces').text
