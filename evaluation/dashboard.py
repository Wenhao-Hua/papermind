"""Render saved measurements as standalone HTML; never invent missing methods."""
import argparse
from html import escape
import json
from pathlib import Path


def render(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    metadata = path.with_suffix('.metadata.json')
    meta = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {'provenance':'Legacy result; no metadata sidecar'}
    keys = list(dict.fromkeys(k for row in data.values() for k in row))
    rows = ''.join('<tr><th>'+escape(method)+'</th>'+''.join('<td>'+escape(str(row.get(k,'not measured')))+'</td>' for k in keys)+'</tr>' for method,row in data.items())
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>PaperMind evaluation</title><style>body{font:16px system-ui;margin:48px;color:#173527;background:#f8faf6}table{border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #bdcabb;text-align:left}pre{white-space:pre-wrap}h1{font-size:40px}.scroll{overflow:auto}</style><h1>Retrieval evaluation</h1><p>Source: '+escape(path.name)+'</p><div class="scroll"><table><tr><th>Method</th>'+''.join('<th>'+escape(k)+'</th>' for k in keys)+'</tr>'+rows+'</table></div><h2>Run provenance and limits</h2><pre>'+escape(json.dumps(meta,indent=2))+'</pre></html>'


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('results')
    p.add_argument('--out',default='evaluation/results/dashboard.html')
    args=p.parse_args()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(render(args.results),encoding='utf-8')
