"""Full-featured web app for PaperMind (``papermind serve``).

Tabs for every capability — 分析 / 问答 / 摘要 / 对比 / 复现 / 搜索 — each reusing
the same core functions and HTML renderers as the CLI.

Safe by default (**demo mode**): features that would call a model only run when
their result is already cached; ``--live`` unlocks live analysis using the
server's configured keys. (搜索 is always available — it makes no model calls.)
"""

from __future__ import annotations

import html as _html
from typing import List, Optional

from papermind.errors import PaperMindError

WEB_MODELS = [
    ("", "默认（按服务器配置）"),
    ("gpt-4o-mini", "OpenAI · GPT-4o mini（便宜）"),
    ("gpt-4o", "OpenAI · GPT-4o"),
    ("claude-3-5-sonnet-20241022", "Anthropic · Claude 3.5 Sonnet"),
    ("deepseek/deepseek-v4-pro", "DeepSeek · V4 Pro"),
    ("deepseek/deepseek-v4-flash", "DeepSeek · V4 Flash"),
    ("gemini/gemini-1.5-flash", "Google · Gemini 1.5 Flash"),
    ("ollama/llama3.1", "本地 · Ollama llama3.1"),
]

_TABS = [
    ("/", "🎯 分析"),
    ("/ask", "💬 问答"),
    ("/summary", "📄 摘要"),
    ("/compare", "📊 对比"),
    ("/reproduce", "🛠️ 复现"),
    ("/search", "🔎 搜索"),
]


def create_app(live: bool = False):
    try:
        from fastapi import FastAPI, Form
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("Web demo 需要 FastAPI。安装：pip install 'paper-mind[web]'") from exc

    app = FastAPI(title="PaperMind", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "live": live}

    @app.get("/demo", response_class=HTMLResponse)
    def demo():
        from papermind.demo import build_demo_report

        return build_demo_report().to_html()

    # -- analyze -------------------------------------------------------------- #
    @app.get("/", response_class=HTMLResponse)
    def index():
        return _page("/", _analyze_form(live))

    @app.post("/analyze", response_class=HTMLResponse)
    def analyze_route(source: str = Form(...), model: str = Form("")):
        report, err = _report_for(source, model, live)
        if err:
            return _page("/", _analyze_form(live, err))
        return report.to_html()

    @app.get("/analyze", response_class=HTMLResponse)
    def analyze_get(source: str = ""):
        if not source.strip():
            return _page("/", _analyze_form(live))
        report, err = _report_for(source, "", live)
        if err:
            return _page("/", _analyze_form(live, err))
        return report.to_html()

    # -- ask (single-shot grounded Q&A) -------------------------------------- #
    @app.get("/ask", response_class=HTMLResponse)
    def ask_form():
        return _page("/ask", _ask_form(live))

    @app.post("/ask", response_class=HTMLResponse)
    def ask_route(source: str = Form(...), question: str = Form(...), model: str = Form(""), mode: str = Form("balanced")):
        if not live:
            return _page("/ask", _ask_form(live, "演示模式不支持问答（需调用模型）。请以 --live 启动。"))
        if not source.strip() or not question.strip():
            return _page("/ask", _ask_form(live, "请填写论文和问题。"))
        from papermind.qa.chat import PaperChat

        try:
            chat = PaperChat(source.strip(), model=(model or None), mode=mode)
            answer = chat.ask(question.strip())
        except PaperMindError as exc:
            return _page("/ask", _ask_form(live, str(exc)))
        return _page("/ask", _answer_html(answer) + _ask_form(live))

    # -- summary -------------------------------------------------------------- #
    @app.get("/summary", response_class=HTMLResponse)
    def summary_form():
        return _page("/summary", _simple_form("/summary", "快速 TL;DR（一次调用）", live))

    @app.post("/summary", response_class=HTMLResponse)
    def summary_route(source: str = Form(...), model: str = Form("")):
        if not live:
            return _page("/summary", _simple_form("/summary", "快速 TL;DR", live, "演示模式不支持摘要（需调用模型）。请以 --live 启动。"))
        from papermind.summarize import summarize

        try:
            result, _usage = summarize(source.strip(), model=(model or None))
        except PaperMindError as exc:
            return _page("/summary", _simple_form("/summary", "快速 TL;DR", live, str(exc)))
        points = "".join(f"<li>{_e(p)}</li>" for p in result.key_points)
        body = f"<div class='card'><h2>{_e(result.title)}</h2><p>{_e(result.tldr)}</p><ul>{points}</ul></div>"
        return _page("/summary", body + _simple_form("/summary", "再来一篇", live))

    # -- compare -------------------------------------------------------------- #
    @app.get("/compare", response_class=HTMLResponse)
    def compare_form():
        return _page("/compare", _compare_form(live))

    @app.post("/compare", response_class=HTMLResponse)
    def compare_route(sources: str = Form(...), model: str = Form("")):
        items = [s.strip() for s in sources.splitlines() if s.strip()][:4]
        if len(items) < 2:
            return _page("/compare", _compare_form(live, "请每行一个来源，至少 2 篇。"))
        from papermind.compare import compare as run_compare

        try:
            comparison = run_compare(items, model=(model or None), synthesize=live)
        except PaperMindError as exc:
            return _page("/compare", _compare_form(live, str(exc)))
        return comparison.to_html()

    # -- reproduce ------------------------------------------------------------ #
    @app.get("/reproduce", response_class=HTMLResponse)
    def reproduce_form():
        return _page("/reproduce", _simple_form("/reproduce", "导出复现指南（setup.sh）", live))

    @app.post("/reproduce", response_class=HTMLResponse)
    def reproduce_route(source: str = Form(...), model: str = Form("")):
        report, err = _report_for(source, model, live, need="reproduction")
        if err:
            return _page("/reproduce", _simple_form("/reproduce", "导出复现指南", live, err))
        script = report.to_setup_script()
        body = (
            "<div class='card'><h2>setup.sh</h2>"
            "<button class='copy-btn' onclick=\"navigator.clipboard.writeText("
            "document.getElementById('sh').textContent)\">📋 复制</button>"
            f"<pre id='sh'>{_e(script)}</pre></div>"
        )
        return _page("/reproduce", body + _simple_form("/reproduce", "再导一篇", live))

    # -- search (no model calls -> always available) ------------------------- #
    @app.get("/search", response_class=HTMLResponse)
    def search_form():
        return _page("/search", _search_form())

    @app.post("/search", response_class=HTMLResponse)
    def search_route(query: str = Form(...)):
        from papermind.parser.arxiv import search_arxiv

        try:
            results = search_arxiv(query.strip(), max_results=12)
        except PaperMindError as exc:
            return _page("/search", _search_form(str(exc)))
        rows = "".join(
            f"<tr><td><a href='/analyze?source={_e(r.arxiv_id)}'>{_e(r.arxiv_id)}</a></td>"
            f"<td>{r.year or '-'}</td><td>{_e(r.title)}</td></tr>"
            for r in results
        )
        table = f"<table><thead><tr><th>arXiv</th><th>年份</th><th>标题</th></tr></thead><tbody>{rows}</tbody></table>"
        return _page("/search", _search_form() + f"<div class='card'>{table}</div>")

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, live: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("Web demo 需要 uvicorn。安装：pip install 'paper-mind[web]'") from exc
    uvicorn.run(create_app(live=live), host=host, port=port)


# --------------------------------------------------------------------------- #
# Shared report resolution (cached-or-live)
# --------------------------------------------------------------------------- #
def _report_for(source: str, model: str, live: bool, need: Optional[str] = None):
    """Return (report, error). Serves a cached report; runs live analysis only if ``live``."""
    from papermind.analyze import analyze as run_analyze
    from papermind.cache import latest_report_path, load_cached_report
    from papermind.config import load_config
    from papermind.parser.arxiv import resolve

    source = (source or "").strip()
    if not source:
        return None, "请输入 arXiv id / URL。"
    config = load_config()
    try:
        resolved = resolve(source, config)
    except PaperMindError as exc:
        return None, str(exc)

    report_path = latest_report_path(resolved.cache_dir)
    report = load_cached_report(report_path) if report_path else None
    if report is not None and (need is None or getattr(report, need, None) is not None):
        return report, None
    if not live:
        return None, "演示模式仅展示已缓存的论文。新论文请用 CLI 分析，或以 --live 启动服务。"
    try:
        # Web stays snappy: skip figure generation (the extra per-point model calls);
        # the report still has all four modules + formulas. Use CLI for figures.
        return run_analyze(source, model=(model or None), config=config, with_figures=False), None
    except PaperMindError as exc:
        return None, str(exc)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = (
    "body{font:16px/1.6 -apple-system,Segoe UI,Roboto,'PingFang SC','Microsoft YaHei',sans-serif;"
    "max-width:820px;margin:0 auto;padding:24px 20px 80px;color:#1a1a2e;background:#fbfbfd}"
    "@media(prefers-color-scheme:dark){body{background:#0f0f17;color:#e6e6f0}"
    ".card{background:#16161f!important;border-color:#2a2a3c!important}"
    "input,select,textarea{background:#16161f!important;color:#e6e6f0!important;border-color:#2a2a3c!important}"
    ".nav a{color:#9aa3b2}.nav a.on{color:#a5b4fc}}"
    "h1{font-size:1.7rem;margin:.2em 0}.sub{color:#6b7280;margin:0 0 18px}"
    ".nav{display:flex;flex-wrap:wrap;gap:14px;border-bottom:1px solid #e5e7eb;padding-bottom:10px;margin-bottom:20px}"
    ".nav a{text-decoration:none;color:#6b7280;font-weight:600}.nav a.on{color:#5b5bd6}"
    ".card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px;margin:14px 0}"
    "label{font-weight:600;font-size:.92rem}input,select,textarea{width:100%;font:inherit;padding:10px 12px;"
    "border:1px solid #e5e7eb;border-radius:8px;margin:6px 0 14px}textarea{min-height:90px}"
    "button{font:inherit;font-weight:600;background:#5b5bd6;color:#fff;border:0;border-radius:8px;padding:11px 18px;cursor:pointer}"
    "button:hover{background:#4a4ac0}.err{color:#e5484d}.note{color:#b45309}"
    "table{border-collapse:collapse;width:100%}th,td{border:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top}"
    "pre{background:#1e1e2e;color:#e6e6f0;padding:14px;border-radius:10px;overflow-x:auto;white-space:pre-wrap}"
    ".copy-btn{background:#eef;color:#333;border:1px solid #ccd;border-radius:8px;padding:4px 12px;margin-bottom:8px}"
    ".seg{padding:10px 14px;border-radius:10px;margin:10px 0}.fact{background:#dcfce7}.inf{background:#fef9c3}.oos{background:#fee2e2}"
    "@media(prefers-color-scheme:dark){.fact{background:#14321f}.inf{background:#3a3410}.oos{background:#3a1414}}"
    ".tag{font-weight:700;font-size:.8rem}a{color:#5b5bd6}"
)


def _page(active: str, body: str) -> str:
    nav = "".join(
        f"<a href='{href}' class='{'on' if href == active else ''}'>{label}</a>" for href, label in _TABS
    )
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>PaperMind</title><style>{_CSS}</style></head><body>"
        "<h1>📄 PaperMind</h1><p class='sub'>读懂一篇 arXiv 论文：分析 · 问答 · 复现</p>"
        f"<nav class='nav'>{nav}</nav>{body}"
        "<script>document.addEventListener('submit',function(e){var b=e.target.querySelector('button');"
        "if(b){b.disabled=true;b.dataset.t=b.textContent;"
        "b.textContent='⏳ 处理中…（首次分析新论文约 20–40 秒，请勿重复点击）';}});</script>"
        "</body></html>"
    )


def _models_select() -> str:
    opts = "".join(f"<option value='{_e(v)}'>{_e(t)}</option>" for v, t in WEB_MODELS)
    return f"<label>模型</label><select name='model'>{opts}</select>"


def _mode_note(live: bool, error: str) -> str:
    out = f"<p class='err'>{_e(error)}</p>" if error else ""
    out += f"<p class='note'>当前：{'LIVE（调用模型，有成本）' if live else '演示模式（仅已缓存论文）'} · <a href='/demo'>看离线示例 →</a></p>"
    return out


def _analyze_form(live: bool, error: str = "") -> str:
    return (
        "<div class='card'><form method='post' action='/analyze'>"
        f"{_mode_note(live, error)}"
        "<label>论文（arXiv id / URL）</label>"
        "<input name='source' placeholder='2307.08691 或 https://arxiv.org/abs/2307.08691' autofocus>"
        f"{_models_select()}<button>分析</button></form></div>"
    )


def _ask_form(live: bool, error: str = "") -> str:
    return (
        "<div class='card'><form method='post' action='/ask'>"
        f"{_mode_note(live, error)}"
        "<label>论文</label><input name='source' placeholder='2307.08691'>"
        "<label>问题</label><input name='question' placeholder='为什么要除以 √d_k？'>"
        f"{_models_select()}"
        "<label>模式</label><select name='mode'><option value='balanced'>balanced</option>"
        "<option value='strict'>strict</option><option value='explore'>explore</option></select>"
        "<button>提问</button></form></div>"
    )


def _simple_form(action: str, label: str, live: bool, error: str = "") -> str:
    return (
        f"<div class='card'><form method='post' action='{action}'>"
        f"{_mode_note(live, error)}<label>{_e(label)} · 论文</label>"
        "<input name='source' placeholder='2307.08691'>"
        f"{_models_select()}<button>运行</button></form></div>"
    )


def _compare_form(live: bool, error: str = "") -> str:
    return (
        "<div class='card'><form method='post' action='/compare'>"
        f"{_mode_note(live, error)}"
        "<label>论文（每行一个，2–4 篇）</label>"
        "<textarea name='sources' placeholder='2307.08691&#10;1706.03762'></textarea>"
        f"{_models_select()}<button>对比</button></form></div>"
    )


def _search_form(error: str = "") -> str:
    err = f"<p class='err'>{_e(error)}</p>" if error else ""
    return (
        "<div class='card'><form method='post' action='/search'>"
        f"{err}<label>搜索 arXiv（不调用模型）</label>"
        "<input name='query' placeholder='flash attention' autofocus>"
        "<button>搜索</button></form></div>"
    )


def _answer_html(answer) -> str:
    kind = {"fact": ("论文事实", "fact"), "inference": ("基于论文的推理", "inf"), "out_of_scope": ("超出论文范围", "oos")}
    segs = []
    for s in answer.segments:
        label, cls = kind.get(s.kind, (s.kind, "seg"))
        head = f"<span class='tag'>【{label}】</span>"
        if s.kind == "inference" and s.confidence:
            head += f" 置信度: {_e(s.confidence)}"
        reason = f"<br><small>推理依据: {_e(s.reasoning)}</small>" if s.reasoning else ""
        segs.append(f"<div class='seg {cls}'>{head}<br>{_e(s.text)}{reason}</div>")
    ev_rows = ""
    for e in answer.evidence:
        loc = (e.section or "") + (f" p.{e.page}" if e.page else "")
        mark = "" if e.verified else "<span class='note'>⚠ 未核实 </span>"
        ev_rows += f"<tr><td>{_e(loc) or '-'}</td><td>{mark}{_e(e.text)}</td></tr>"
    ev = f"<h3>原文依据</h3><table><tbody>{ev_rows}</tbody></table>" if ev_rows else ""
    return f"<div class='card'>{''.join(segs)}{ev}</div>"


def _e(text) -> str:
    return _html.escape(str(text)) if text is not None else ""
