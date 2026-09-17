"""Escaped, dependency-free query inspection UI."""
from html import escape
import json


def render_traces(traces):
    rows = []
    for trace in reversed(traces):
        stages = ''.join('<li>'+escape(s['name'])+f" · {s['latency_ms']:.2f} ms · "+escape(s['status'])+'</li>' for s in trace['spans'])
        candidates = ''.join('<tr>'+''.join('<td>'+escape(str(c.get(k,'')))+'</td>' for k in ('chunk_id','section','dense_score','rerank_score','text'))+'</tr>' for c in trace['candidates'])
        rows.append('<article><h2>'+escape(trace['trace_id'])+'</h2><p>'+escape(trace['status'])+f" · {trace['total_ms']:.2f} ms</p><ol>"+stages+'</ol><p>Selected chunks: '+escape(str(trace['selected_chunk_ids']))+'</p><pre>'+escape(json.dumps(trace['usage'],indent=2))+'</pre><p>'+escape(trace['cost_note'])+'</p><div class="scroll"><table><tr><th>Chunk</th><th>Section</th><th>Dense</th><th>Rerank</th><th>Passage</th></tr>'+candidates+'</table></div></article>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PaperMind Query Traces</title><style>body{font:16px system-ui;max-width:1100px;margin:48px auto;padding:0 24px;color:#21332c;background:#f7f7f2}h1{font-size:40px}h2{font:16px monospace}article{border-top:1px solid #bccbc1;padding:24px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd;min-width:80px}td:last-child{min-width:280px}.scroll{overflow:auto}pre{background:#e9eee8;padding:16px}</style><a href="/ask">← Questions</a><h1>Query traces</h1><p>Only this browser session's queries. Costs are estimates; missing pricing is not proof of free inference.</p>'''+(''.join(rows) or '<p>No traces yet. Ask a question in live mode first.</p>')+'</html>'
