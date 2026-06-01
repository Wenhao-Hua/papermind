"""Render a Report to a single self-contained HTML file.

Images are inlined as base64 data URIs and Mermaid diagrams render via a CDN
script, so the file is shareable as-is (open in any browser, no local assets).
"""

from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import List, Optional

from papermind.output.schema import Connection, Reproduction, Report, Source, TechnicalPoint

_DIFFICULTY = {"high": ("high", "#e5484d"), "mid": ("mid", "#f5a623"), "low": ("low", "#30a46c")}

_CSS = """
:root { --fg:#1a1a2e; --muted:#6b7280; --accent:#5b5bd6; --border:#e5e7eb; --bg:#fbfbfd; --card:#fff; --soft:#f3f4f6; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e6f0; --muted:#9aa3b2; --accent:#a5b4fc; --border:#2a2a3c; --bg:#0f0f17; --card:#16161f; --soft:#1e1e2b; }
}
html { scroll-behavior:smooth; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif; }
.wrap { max-width:860px; margin:0 auto; padding:48px 24px 96px; }
h1 { font-size:2rem; line-height:1.25; margin:0 0 .4em; }
h2 { font-size:1.4rem; margin:2.4em 0 .8em; padding-bottom:.3em; border-bottom:2px solid var(--border); }
h3 { font-size:1.12rem; margin:1.6em 0 .5em; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.meta { color:var(--muted); font-size:.92rem; margin-bottom:1em; }
.meta a { color:var(--muted); }
.toc { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 8px; padding-bottom:18px; border-bottom:1px solid var(--border); }
.toc a { font-size:.86rem; background:var(--soft); border:1px solid var(--border); border-radius:999px; padding:5px 14px; color:var(--fg); }
.toc a:hover { border-color:var(--accent); color:var(--accent); text-decoration:none; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px 22px; margin:14px 0; }
.pill { display:inline-block; font-size:.72rem; font-weight:700; color:#fff; padding:2px 9px; border-radius:999px; vertical-align:middle; margin-left:8px; }
.analogy { background:var(--soft); border-left:4px solid var(--accent); padding:10px 16px; border-radius:0 8px 8px 0; margin:12px 0; }
.formula { overflow-x:auto; padding:6px 0; margin:10px 0; }
.src { color:var(--muted); font-size:.86rem; }
blockquote { margin:12px 0; padding:8px 16px; border-left:3px solid var(--border); color:var(--muted); }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:.95rem; }
th,td { border:1px solid var(--border); padding:8px 12px; text-align:left; vertical-align:top; }
th { background:var(--soft); }
code { background:var(--soft); padding:1px 6px; border-radius:6px; font-size:.88em; }
pre { background:#1e1e2e; color:#e6e6f0; padding:14px 16px; border-radius:10px; overflow-x:auto; }
pre code { background:none; color:inherit; padding:0; }
pre.mermaid { background:#fafaff; border:1px solid var(--border); border-radius:10px; text-align:center; padding:18px; }
pre.mermaid svg { max-width:100%; height:auto; }
img.figure { max-width:100%; border:1px solid var(--border); border-radius:8px; display:block; margin:8px 0; }
figcaption { color:var(--muted); font-size:.85rem; margin-bottom:1em; }
.ai-tag { color:var(--accent); }
footer { color:var(--muted); font-size:.85rem; margin-top:48px; text-align:center; }
"""

_MERMAID = (
    '<script type="module">'
    "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';"
    "mermaid.initialize({startOnLoad:true, theme:'base', themeVariables:{"
    "fontFamily:'-apple-system,Segoe UI,Microsoft YaHei,sans-serif', fontSize:'15px',"
    "primaryColor:'#eef2ff', primaryBorderColor:'#5b5bd6', primaryTextColor:'#1e1b4b',"
    "lineColor:'#94a3b8'}});"
    "</script>"
)

_MATHJAX = (
    "<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],"
    "displayMath:[['$$','$$'],['\\\\[','\\\\]']]},svg:{fontCache:'global'}};</script>"
    '<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>'
)


def to_html(report: Report, path: Optional[str] = None) -> str:
    doc = "\n".join(_render(report))
    if path:
        out = Path(path)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc, encoding="utf-8")
    return doc


def _render(report: Report) -> List[str]:
    paper = report.paper
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(paper.title)} · PaperMind</title>",
        f"<style>{_CSS}</style>",
        _MERMAID,
        _MATHJAX,
        '</head><body><div class="wrap">',
        f"<h1>{esc(paper.title)}</h1>",
        f'<div class="meta">{_meta(report)}</div>',
        _toc(report),
    ]
    if report.contributions:
        parts += _contributions(report)
    if report.technical.details:
        parts.append('<h2 id="technical">🔬 技术细节解释</h2>')
        for i, p in enumerate(report.technical.details, 1):
            parts += _technical_point(report, i, p)
    if report.connections.related_works:
        parts += _connections(report.connections.related_works)
    if report.reproduction:
        parts += _reproduction(report.reproduction)
    parts.append(
        '<footer>Generated by <a href="https://github.com/Wenhao-Hua/papermind">PaperMind</a>.</footer>'
    )
    parts.append("</div></body></html>")
    return parts


def esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def _toc(report: Report) -> str:
    items = []
    if report.contributions:
        items.append(("contributions", "🎯 贡献"))
    if report.technical.details:
        items.append(("technical", "🔬 技术细节"))
    if report.connections.related_works:
        items.append(("connections", "🔗 知识关联"))
    if report.reproduction:
        items.append(("reproduction", "🛠️ 复现指南"))
    if len(items) < 2:
        return ""
    links = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in items)
    return f'<nav class="toc">{links}</nav>'


def _meta(report: Report) -> str:
    paper = report.paper
    bits = []
    if paper.authors:
        shown = ", ".join(paper.authors[:6]) + (" et al." if len(paper.authors) > 6 else "")
        bits.append(f"<b>Authors:</b> {esc(shown)}")
    if paper.year:
        bits.append(f"<b>Year:</b> {paper.year}")
    if paper.arxiv_id:
        bits.append(f'<b>arXiv:</b> <a href="https://arxiv.org/abs/{esc(paper.arxiv_id)}">{esc(paper.arxiv_id)}</a>')
    if paper.pdf_url:
        bits.append(f'<a href="{esc(paper.pdf_url)}">PDF</a>')
    return " &nbsp;•&nbsp; ".join(bits)


def _src_link(report: Report, section: Optional[str], page: Optional[int]) -> str:
    if page and section:
        label = f"{section} (p.{page})"
    elif page:
        label = f"p.{page}"
    else:
        label = section or "source"
    url = report.paper.page_link(page)
    return f'<a href="{esc(url)}">{esc(label)}</a>' if url else esc(label)


def _sources(report: Report, sources: List[Source]) -> str:
    if not sources:
        return ""
    items = "".join(f"<li>{_src_link(report, s.section, s.page)}: {esc(s.text)}</li>" for s in sources)
    return f"<blockquote><b>原文出处</b><ul>{items}</ul></blockquote>"


def _contributions(report: Report) -> List[str]:
    c = report.contributions
    return [
        '<h2 id="contributions">🎯 贡献与创新点</h2>',
        '<div class="card">',
        f"<p><b>核心贡献：</b>{esc(c.main_contribution)}</p>",
        f"<p><b>新颖之处：</b>{esc(c.novelty)}</p>",
        f"<p><b>解决的问题：</b>{esc(c.problem_solved)}</p>",
        _sources(report, c.sources),
        "</div>",
    ]


def _technical_point(report: Report, idx: int, p: TechnicalPoint) -> List[str]:
    label, color = _DIFFICULTY.get(p.difficulty, (p.difficulty, "#888"))
    out = [
        '<div class="card">',
        f'<h3>{idx}. {esc(p.name)}<span class="pill" style="background:{color}">{label}</span></h3>',
        f"<p>{esc(p.explanation)}</p>",
    ]
    if p.formula:
        out.append(f'<div class="formula">$$ {esc(p.formula)} $$</div>')
    if p.analogy:
        out.append(f'<div class="analogy">💡 <b>类比：</b>{esc(p.analogy)}</div>')
    if p.source_section or p.page:
        out.append(f'<p class="src">📍 出处：{_src_link(report, p.source_section, p.page)}</p>')
    if p.figure:
        out += _figure(p)
    out.append("</div>")
    return out


def _figure(p: TechnicalPoint) -> List[str]:
    fig = p.figure
    if fig.image_path:  # original figure, or an AI-generated image
        uri = _data_uri(fig.image_path)
        if uri:
            origin = "论文原图" if fig.type == "original" else "AI 生成示意图"
            cap = esc(fig.caption or origin)
            tag = "" if fig.type == "original" else ' class="ai-tag"'
            return [f'<figure><img class="figure" src="{uri}" alt="{cap}"><figcaption{tag}>{cap}（{origin}）</figcaption></figure>']
    if fig.type == "ai_generated" and fig.mermaid:
        cap = esc(fig.caption or "AI 生成示意图")
        return [f'<pre class="mermaid">{fig.mermaid}</pre>', f'<figcaption class="ai-tag">{cap}</figcaption>']
    return []


def _connections(works: List[Connection]) -> List[str]:
    rows = []
    for w in works:
        paper = f'<a href="{esc(w.arxiv_link)}">{esc(w.paper)}</a>' if w.arxiv_link else esc(w.paper)
        rows.append(f"<tr><td>{esc(w.concept)}</td><td>{paper}</td><td>{esc(w.relationship)}</td></tr>")
    return [
        '<h2 id="connections">🔗 知识关联</h2>',
        "<table><thead><tr><th>概念</th><th>相关论文</th><th>关系</th></tr></thead><tbody>",
        *rows,
        "</tbody></table>",
    ]


def _reproduction(r: Reproduction) -> List[str]:
    out = ['<h2 id="reproduction">🛠️ 复现指南</h2>', '<div class="card"><ul>']
    if r.official_code:
        tag = f" (<code>{esc(r.version_tag)}</code>)" if r.version_tag else ""
        out.append(f'<li><b>官方代码：</b><a href="{esc(r.official_code)}">{esc(r.official_code)}</a>{tag}</li>')
    if r.requirements:
        out.append(f"<li><b>环境要求：</b>{esc(r.requirements)}</li>")
    if r.recommended_hardware:
        out.append(f"<li><b>推荐硬件：</b>{esc(r.recommended_hardware)}</li>")
    if r.key_hyperparams:
        params = ", ".join(f"<code>{esc(h)}</code>" for h in r.key_hyperparams)
        out.append(f"<li><b>关键超参数：</b>{params}</li>")
    out.append("</ul></div>")

    if r.env_setup_steps:
        out.append("<h3>环境配置步骤</h3>")
        for s in r.env_setup_steps:
            out.append(f"<p><b>{s.step}. {esc(s.title)}</b></p>")
            if s.desc:
                out.append(f"<p>{esc(s.desc)}</p>")
            if s.command:
                out.append(f"<pre><code>{esc(s.command)}</code></pre>")

    if r.performance_benchmarks:
        out.append("<h3>性能基准</h3>")
        out.append("<table><thead><tr><th>设置</th><th>Baseline</th><th>结果</th><th>加速</th><th>显存</th></tr></thead><tbody>")
        for b in r.performance_benchmarks:
            out.append(
                f"<tr><td>{esc(b.setting)}</td><td>{esc(b.baseline or '-')}</td>"
                f"<td>{esc(b.result or '-')}</td><td>{esc(b.speedup or '-')}</td><td>{esc(b.memory or '-')}</td></tr>"
            )
        out.append("</tbody></table>")

    if r.datasets:
        out.append("<h3>数据集</h3><ul>")
        for d in r.datasets:
            name = f'<a href="{esc(d.link)}">{esc(d.name)}</a>' if d.link else esc(d.name)
            out.append(f"<li><b>{name}</b> — {esc(d.purpose)}</li>")
        out.append("</ul>")

    if r.common_errors:
        out.append("<h3>常见报错与解决</h3><ul>")
        for e in r.common_errors:
            line = f"<li><b>报错：</b><code>{esc(e.error)}</code>"
            if e.cause:
                line += f"<br>原因：{esc(e.cause)}"
            if e.fix_command:
                line += f"<br>修复：<code>{esc(e.fix_command)}</code>"
            out.append(line + "</li>")
        out.append("</ul>")

    if r.gotchas:
        out.append("<h3>⚠️ 坑点提示</h3><ul>")
        out += [f"<li>{esc(g)}</li>" for g in r.gotchas]
        out.append("</ul>")
    return out


def _data_uri(path: str) -> Optional[str]:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
