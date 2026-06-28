"""Render a Report to a single self-contained HTML file.

AI teaching figures are inlined as SVG; images as base64 data URIs; legacy Mermaid
renders via a CDN script — so the file is shareable as-is (any browser, no assets).
"""

from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import List, Optional

from papermind.output.cite import to_bibtex
from papermind.output.schema import Connection, Reproduction, Report, Source, TechnicalPoint

_DIFFICULTY = {"high": ("high", "#e5484d"), "mid": ("mid", "#f5a623"), "low": ("low", "#30a46c")}

_CSS = """
:root { --fg:#0f172a; --muted:#475569; --faint:#94a3b8; --accent:#2563eb; --accent-press:#1d4ed8;
  --border:#e8edf3; --border-strong:#d4dbe4; --bg:#f5f7fa; --card:#ffffff; --soft:#f1f5f9; --accent-soft:#eff5ff;
  --shadow:0 1px 2px rgba(15,23,42,.04),0 6px 18px -8px rgba(15,23,42,.10); }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8ecf3; --muted:#9aa6b7; --faint:#606b7c; --accent:#5b8cff; --accent-press:#7aa2ff;
    --border:#212733; --border-strong:#2f3744; --bg:#0b0d12; --card:#13161d; --soft:#171b24; --accent-soft:#16203a;
    --shadow:0 1px 2px rgba(0,0,0,.5),0 6px 18px -8px rgba(0,0,0,.6); }
}
html { scroll-behavior:smooth; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); -webkit-font-smoothing:antialiased;
  font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif; }
.wrap { max-width:820px; margin:0 auto; padding:56px 24px 110px; }
h1 { font-size:2rem; line-height:1.22; letter-spacing:-.025em; font-weight:700; margin:0 0 .35em; }
h2 { font-size:1.34rem; letter-spacing:-.02em; font-weight:700; margin:2.6em 0 .9em; padding-bottom:.4em; border-bottom:1px solid var(--border); }
h2::before { content:''; display:inline-block; width:4px; height:.82em; background:var(--accent); border-radius:2px; margin-right:12px; vertical-align:-1px; }
h3 { font-size:1.1rem; letter-spacing:-.01em; font-weight:600; margin:1.7em 0 .5em; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.meta { color:var(--muted); font-size:.92rem; margin-bottom:1.2em; }
.meta a { color:var(--muted); }
.toc { display:flex; flex-wrap:wrap; gap:8px; margin:20px 0 8px; padding-bottom:20px; border-bottom:1px solid var(--border); }
.toc a { font-size:.85rem; background:var(--card); border:1px solid var(--border-strong); border-radius:8px; padding:6px 13px; color:var(--muted); font-weight:500; }
.toc a:hover { border-color:var(--accent); color:var(--accent); text-decoration:none; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px 24px; margin:14px 0; box-shadow:var(--shadow); }
.pill { display:inline-block; font-size:.7rem; font-weight:700; letter-spacing:.02em; color:#fff; padding:2px 9px; border-radius:999px; vertical-align:middle; margin-left:10px; }
.analogy { background:var(--accent-soft); border-left:3px solid var(--accent); padding:11px 16px; border-radius:0 8px 8px 0; margin:14px 0; }
.formula { overflow-x:auto; padding:6px 0; margin:12px 0; }
.src { color:var(--faint); font-size:.85rem; }
blockquote { margin:14px 0; padding:10px 18px; border-left:3px solid var(--border-strong); color:var(--muted); background:var(--soft); border-radius:0 8px 8px 0; }
table { border-collapse:collapse; width:100%; margin:14px 0; font-size:.93rem; }
th,td { border:1px solid var(--border); padding:9px 13px; text-align:left; vertical-align:top; }
th { background:var(--soft); font-weight:600; }
code { background:var(--accent-soft); color:var(--accent-press); padding:1px 6px; border-radius:5px; font-size:.86em; }
pre { background:#0f1117; color:#e7e9ef; padding:15px 17px; border-radius:9px; overflow-x:auto; }
pre code { background:none; color:inherit; padding:0; }
pre.mermaid { background:var(--card); border:1px solid var(--border); border-radius:10px; text-align:center; padding:18px; color:var(--fg); }
pre.mermaid svg { max-width:100%; height:auto; }
img.figure { max-width:100%; border:1px solid var(--border); border-radius:10px; display:block; margin:10px 0; }
figure.svgfig { margin:10px 0; }
figure.svgfig svg { max-width:100%; height:auto; display:block; margin:0 auto;
  border:1px solid var(--border); border-radius:10px; background:#fff; box-shadow:var(--shadow); }
figcaption { color:var(--faint); font-size:.84rem; margin-bottom:1em; }
.ai-tag { color:var(--accent); }
.readfig { background:var(--soft); border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:0 10px 10px 0; padding:13px 17px; margin:6px 0 1.5em; font-size:.92rem; line-height:1.65; }
.readfig .rf-gist { margin:0 0 8px; font-weight:600; color:var(--fg); }
.readfig .rf-gist span, .readfig .rf-take span { display:inline-block; font-size:.7rem; font-weight:600;
  color:#fff; background:var(--accent); border-radius:5px; padding:1px 8px; margin-right:8px; }
.readfig .rf-parts { margin:0 0 8px; padding-left:1.3em; color:var(--fg); }
.readfig .rf-parts li { margin:3px 0; }
.readfig .rf-take { margin:0; color:var(--fg); }
.copy-btn { font:inherit; font-size:.85rem; font-weight:600; cursor:pointer; background:var(--card); color:var(--accent);
  border:1px solid var(--border-strong); border-radius:8px; padding:6px 13px; margin:4px 0 12px; }
.copy-btn:hover { border-color:var(--accent); }
.codeblock { position:relative; }
.codeblock .copy-code { position:absolute; top:8px; right:8px; font:inherit; font-size:.76rem; cursor:pointer;
  background:rgba(255,255,255,.08); color:#e7e9ef; border:1px solid #333842; border-radius:6px; padding:3px 9px; opacity:.75; }
.codeblock .copy-code:hover { opacity:1; border-color:var(--accent); }
#pm-progress { position:fixed; top:0; left:0; height:2px; width:0; background:var(--accent); z-index:100; transition:width .1s; }
#pm-top { position:fixed; bottom:24px; right:24px; display:none; cursor:pointer; z-index:100; width:42px; height:42px;
  border-radius:50%; border:1px solid var(--border-strong); background:var(--card); color:var(--accent); font-size:1.2rem; box-shadow:0 4px 14px rgba(15,17,21,.1); }
details.tech { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:4px 24px; margin:14px 0; box-shadow:var(--shadow); }
details.tech > summary { cursor:pointer; list-style:none; font-size:1.1rem; font-weight:600; letter-spacing:-.01em; padding:14px 0; }
details.tech > summary::-webkit-details-marker { display:none; }
details.tech > summary::before { content:'▸'; color:var(--faint); margin-right:9px; font-size:.8em; }
details.tech[open] > summary::before { content:'▾'; }
details.tech[open] > summary { border-bottom:1px solid var(--border); margin-bottom:12px; }
footer { color:var(--faint); font-size:.84rem; margin-top:54px; padding-top:22px; border-top:1px solid var(--border); text-align:center; }
"""

_MERMAID = (
    '<script type="module">'
    "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';"
    "mermaid.initialize({startOnLoad:true, theme:'base', themeVariables:{"
    "fontFamily:'-apple-system,Segoe UI,Microsoft YaHei,sans-serif', fontSize:'15px',"
    "primaryColor:'#eff5ff', primaryBorderColor:'#2563eb', primaryTextColor:'#0f172a',"
    "lineColor:'#94a3b8'}});"
    "</script>"
)

_JS = (
    "<script>"
    "function pmCopyBib(b){var t=document.getElementById('pm-bibtex').textContent;"
    "pmFlash(b,t,'\\u2713 \\u5df2\\u590d\\u5236');}"
    "function pmCopyCode(b){var p=b.nextElementSibling;pmFlash(b,p.innerText||p.textContent,'\\u2713');}"
    "function pmFlash(b,text,ok){navigator.clipboard.writeText(text).then(function(){"
    "var o=b.textContent;b.textContent=ok;setTimeout(function(){b.textContent=o;},1200);});}"
    "var pb=document.getElementById('pm-progress'),tb=document.getElementById('pm-top');"
    "addEventListener('scroll',function(){var h=document.documentElement;"
    "var p=h.scrollTop/((h.scrollHeight-h.clientHeight)||1)*100;"
    "if(pb)pb.style.width=p+'%';if(tb)tb.style.display=h.scrollTop>400?'block':'none';});"
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
        ("<link rel='icon' href=\"data:image/svg+xml,"
         "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
         "<rect width='32' height='32' rx='7' fill='%232563eb'/>"
         "<g stroke='%23fff' stroke-width='2.4' stroke-linecap='round'>"
         "<path d='M10 11h12'/><path d='M10 16h12'/><path d='M10 21h7'/></g></svg>\">"),
        f"<title>{esc(paper.title)} · PaperMind</title>",
        f"<style>{_CSS}</style>",
        _MERMAID,
        _MATHJAX,
        '</head><body><div id="pm-progress"></div>'
        '<button id="pm-top" title="回到顶部" onclick="scrollTo({top:0,behavior:\'smooth\'})">↑</button>'
        '<div class="wrap">',
        f"<h1>{esc(paper.title)}</h1>",
        f'<div class="meta">{_meta(report)}</div>',
        _bibtex_block(report),
        _toc(report),
    ]
    if report.contributions:
        parts += _contributions(report)
    if report.framework is not None:
        parts += _framework_section(report.framework)
    if report.technical.details:
        parts.append('<h2 id="technical">技术细节解释</h2>')
        for i, p in enumerate(report.technical.details, 1):
            parts += _technical_point(report, i, p)
    if report.connections.related_works:
        parts += _connections(report.connections.related_works)
    if report.reproduction:
        parts += _reproduction(report.reproduction)
    parts.append(
        '<footer>Generated by <a href="https://github.com/Wenhao-Hua/papermind">PaperMind</a>.</footer>'
    )
    parts.append("</div>")
    parts.append(_JS)
    parts.append("</body></html>")
    return parts


def esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def _cmd_block(command: str) -> str:
    return (
        '<div class="codeblock"><button class="copy-code" onclick="pmCopyCode(this)">复制</button>'
        f"<pre><code>{esc(command)}</code></pre></div>"
    )


def _bibtex_block(report: Report) -> str:
    bib = esc(to_bibtex(report.paper))
    return (
        '<button class="copy-btn" onclick="pmCopyBib(this)">复制 BibTeX</button>'
        f'<pre id="pm-bibtex" hidden>{bib}</pre>'
    )


def _toc(report: Report) -> str:
    items = []
    if report.contributions:
        items.append(("contributions", "贡献"))
    if report.framework is not None:
        items.append(("framework", "框架"))
    if report.technical.details:
        items.append(("technical", "技术细节"))
    if report.connections.related_works:
        items.append(("connections", "知识关联"))
    if report.reproduction:
        items.append(("reproduction", "复现指南"))
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
    bits.append(f"<b>阅读时长:</b> 约 {report.reading_time_minutes()} 分钟")
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
    unverified = "<span style='color:var(--amber,#b7791f)'>未核实</span> · "
    items = "".join(
        f"<li>{_src_link(report, s.section, s.page)}: "
        f"{'' if s.verified else unverified}{esc(s.text)}</li>"
        for s in sources
    )
    return f"<blockquote><b>原文出处</b><ul>{items}</ul></blockquote>"


def _contributions(report: Report) -> List[str]:
    c = report.contributions
    return [
        '<h2 id="contributions">贡献与创新点</h2>',
        '<div class="card">',
        f"<p><b>核心贡献：</b>{esc(c.main_contribution)}</p>",
        f"<p><b>新颖之处：</b>{esc(c.novelty)}</p>",
        f"<p><b>解决的问题：</b>{esc(c.problem_solved)}</p>",
        _sources(report, c.sources),
        "</div>",
    ]


def _framework_section(spec) -> List[str]:
    from papermind.figures.framework import render_framework_svg

    return [
        '<h2 id="framework">方法框架</h2>',
        f'<figure class="svgfig">{render_framework_svg(spec)}</figure>',
        '<figcaption class="ai-tag">整篇方法的端到端框架图（虚线＝论文未显式画出的推断步骤）。</figcaption>',
    ]


def _technical_point(report: Report, idx: int, p: TechnicalPoint) -> List[str]:
    label, color = _DIFFICULTY.get(p.difficulty, (p.difficulty, "#888"))
    out = [
        '<details class="tech" open>',
        f'<summary>{idx}. {esc(p.name)}<span class="pill" style="background:{color}">{label}</span></summary>',
        f"<p>{esc(p.explanation)}</p>",
    ]
    if p.formula:
        from papermind.output.markdown import _mathjax

        out.append(f'<div class="formula">$$ {esc(_mathjax(p.formula))} $$</div>')
    if p.analogy:
        out.append(f'<div class="analogy"><b>类比：</b>{esc(p.analogy)}</div>')
    if p.source_section or p.page:
        out.append(f'<p class="src">出处：{_src_link(report, p.source_section, p.page)}</p>')
    if p.figure:
        out += _figure(p)
    out.append("</details>")
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
    if fig.svg:  # self-contained teaching SVG (architecture-faithful), inlined
        cap = esc(fig.caption or "教学示意图")
        return [
            f'<figure class="svgfig">{fig.svg}</figure>',
            f'<figcaption class="ai-tag">{cap}</figcaption>',
            *_explain_html(fig.explain),
        ]
    if fig.type == "ai_generated" and fig.mermaid:
        cap = esc(fig.caption or "AI 生成示意图")
        return [f'<pre class="mermaid">{fig.mermaid}</pre>', f'<figcaption class="ai-tag">{cap}</figcaption>']
    return []


def _explain_html(ex) -> List[str]:
    """Render the structured "how to read this figure" walkthrough as a tidy block."""
    if not ex or not ex.gist:
        return []
    parts = "".join(f"<li>{esc(p)}</li>" for p in ex.parts)
    ul = f'<ul class="rf-parts">{parts}</ul>' if parts else ""
    take = f'<p class="rf-take"><span>关键</span>{esc(ex.takeaway)}</p>' if ex.takeaway else ""
    return [f'<aside class="readfig"><p class="rf-gist"><span>读图</span>{esc(ex.gist)}</p>{ul}{take}</aside>']


def _connections(works: List[Connection]) -> List[str]:
    rows = []
    for w in works:
        paper = f'<a href="{esc(w.arxiv_link)}">{esc(w.paper)}</a>' if w.arxiv_link else esc(w.paper)
        rows.append(f"<tr><td>{esc(w.concept)}</td><td>{paper}</td><td>{esc(w.relationship)}</td></tr>")
    return [
        '<h2 id="connections">知识关联</h2>',
        "<table><thead><tr><th>概念</th><th>相关论文</th><th>关系</th></tr></thead><tbody>",
        *rows,
        "</tbody></table>",
    ]


def _reproduction(r: Reproduction) -> List[str]:
    out = ['<h2 id="reproduction">复现指南</h2>', '<div class="card"><ul>']
    if r.code_repo is not None:
        cr = r.code_repo
        badge = " · ✓官方实现" if cr.is_official else ""
        star = f" · ★{cr.stars}" if cr.stars else ""
        tag = f" · <code>{esc(r.version_tag)}</code>" if r.version_tag else ""
        out.append(
            f'<li><b>代码仓库（{esc(cr.source)}{badge}{star}）：</b>'
            f'<a href="{esc(cr.url)}">{esc(cr.url)}</a>{tag}</li>'
        )
        cmds = list(cr.install_commands) + list(cr.run_commands)
        if cmds:
            body = "\n".join(esc(c) for c in cmds)
            out.append(f"<li><b>安装 / 运行（取自该仓库，非模型生成）：</b><pre>{body}</pre></li>")
    elif r.official_code:
        tag = f" (<code>{esc(r.version_tag)}</code>)" if r.version_tag else ""
        out.append(f'<li><b>代码仓库：</b><a href="{esc(r.official_code)}">{esc(r.official_code)}</a>{tag}</li>')
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
                out.append(_cmd_block(s.command))

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
        out.append("<h3>坑点提示</h3><ul>")
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
