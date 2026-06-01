"""Full-featured web app for PaperMind (``papermind serve`` / ``papermind ui``).

Tabs for every capability — 分析 / 问答 / 速读 / 对比 / 复现 / 搜索 — each reusing
the same core functions and HTML renderers as the CLI.

Safe by default (**demo mode**): features that would call a model only run when
their result is already cached; ``--live`` unlocks live analysis using the
server's configured keys. (搜索 is always available — it makes no model calls.)

The look is a single minimal-academic design system (system serif display + sans
body, warm paper, one ink-navy accent, light/dark via CSS variables). No model
picker — the active model is shown subtly in the footer.
"""

from __future__ import annotations

import html as _html
import secrets
import threading
from typing import Optional

from papermind.errors import PaperMindError

try:  # so route annotations (e.g. UploadFile) resolve under `from __future__ import annotations`
    from fastapi import Cookie, File, Form, Request, UploadFile  # noqa: F401
except ImportError:  # pragma: no cover - a friendly error is raised in create_app
    pass

# In-memory chat sessions for multi-turn 问答 (keyed by a per-browser cookie).
# Each entry: {"paper": str, "chat": PaperChat, "log": [(question, Answer)]}.
_ASK_SESSIONS: dict = {}
_ASK_LOCK = threading.Lock()
_ASK_MAX_SESSIONS = 32

# Kept for reference / tests: the models the server can talk to. The web UI no
# longer renders a picker — the active model comes from the server's config.
WEB_MODELS = [
    ("gpt-4o-mini", "OpenAI · GPT-4o mini"),
    ("gpt-4o", "OpenAI · GPT-4o"),
    ("claude-3-5-sonnet-20241022", "Anthropic · Claude 3.5 Sonnet"),
    ("deepseek/deepseek-v4-pro", "DeepSeek · V4 Pro"),
    ("deepseek/deepseek-v4-flash", "DeepSeek · V4 Flash"),
    ("gemini/gemini-1.5-flash", "Google · Gemini 1.5 Flash"),
    ("ollama/llama3.1", "本地 · Ollama llama3.1"),
]

_MODEL_LABELS = {
    "deepseek/deepseek-v4-pro": "DeepSeek-V4 Pro",
    "deepseek/deepseek-v4-flash": "DeepSeek-V4 Flash",
    "deepseek/deepseek-reasoner": "DeepSeek Reasoner",
    "deepseek/deepseek-chat": "DeepSeek Chat",
    "gpt-4o-mini": "OpenAI GPT-4o mini",
    "gpt-4o": "OpenAI GPT-4o",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
}

_TABS = [
    ("/", "分析"),
    ("/ask", "问答"),
    ("/summary", "速读"),
    ("/compare", "对比"),
    ("/reproduce", "复现"),
    ("/search", "搜索"),
]


def create_app(live: bool = False):
    try:
        from fastapi import Cookie, FastAPI, File, Form, UploadFile
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
        return _page("/", _analyze_form(live), live)

    @app.post("/analyze", response_class=HTMLResponse)
    def analyze_route(source: str = Form(""), model: str = Form(""), file: Optional[UploadFile] = File(None)):
        src = _resolve_upload(source, file)
        if not src:
            return _page("/", _analyze_form(live, "请填入 arXiv id / URL，或上传 PDF。"), live)
        report, err = _report_for(src, model, live)
        if err:
            return _page("/", _analyze_form(live, err), live)
        return report.to_html()

    @app.get("/analyze", response_class=HTMLResponse)
    def analyze_get(source: str = ""):
        if not source.strip():
            return _page("/", _analyze_form(live), live)
        report, err = _report_for(source, "", live)
        if err:
            return _page("/", _analyze_form(live, err), live)
        return report.to_html()

    # -- ask (multi-turn grounded Q&A) --------------------------------------- #
    @app.get("/ask", response_class=HTMLResponse)
    def ask_form(pm_sid: Optional[str] = Cookie(None)):
        sess = _session_for(pm_sid)
        log = _chat_log_html(sess["log"]) if sess and sess.get("log") else ""
        return _page("/ask", log + _ask_form(live), live)

    @app.post("/ask", response_class=HTMLResponse)
    def ask_route(
        source: str = Form(...),
        question: str = Form(...),
        model: str = Form(""),
        mode: str = Form("balanced"),
        pm_sid: Optional[str] = Cookie(None),
    ):
        if not live:
            return _page("/ask", _ask_form(live, "演示模式不支持问答（需调用模型）。请以 --live 启动。"), live)

        sid = pm_sid
        sess = _session_for(sid)
        prior = _chat_log_html(sess["log"]) if sess and sess.get("log") else ""
        if not source.strip() or not question.strip():
            return _page("/ask", prior + _ask_form(live, "请填写论文和问题。"), live)

        from papermind.qa.chat import PaperChat

        sid = sid or secrets.token_hex(8)
        paper = source.strip()
        if sess is None or sess.get("paper") != paper:
            try:
                sess = {"paper": paper, "chat": PaperChat(paper, model=(model or None), mode=mode), "log": []}
            except PaperMindError as exc:
                return _page("/ask", _ask_form(live, str(exc)), live)
            prior = ""  # new paper -> fresh conversation
        sess["chat"].mode = mode  # honor a mode change between turns
        try:
            answer = sess["chat"].ask(question.strip())
        except PaperMindError as exc:
            _store_session(sid, sess)
            body = _chat_log_html(sess["log"]) + _ask_form(live, str(exc))
            return _with_session(HTMLResponse(_page("/ask", body, live)), sid)

        sess["log"].append((question.strip(), answer))
        _store_session(sid, sess)
        body = _chat_log_html(sess["log"]) + _ask_form(live)
        return _with_session(HTMLResponse(_page("/ask", body, live)), sid)

    # -- summary -------------------------------------------------------------- #
    @app.get("/summary", response_class=HTMLResponse)
    def summary_form():
        return _page("/summary", _summary_form(live), live)

    @app.post("/summary", response_class=HTMLResponse)
    def summary_route(source: str = Form(...), model: str = Form("")):
        if not live:
            return _page("/summary", _summary_form(live, "演示模式不支持速读（需调用模型）。请以 --live 启动。"), live)
        from papermind.summarize import summarize

        try:
            result, _usage = summarize(source.strip(), model=(model or None))
        except PaperMindError as exc:
            return _page("/summary", _summary_form(live, str(exc)), live)
        points = "".join(f"<li>{_e(p)}</li>" for p in result.key_points)
        body = f"<section class='panel'><h2>{_e(result.title)}</h2><p>{_e(result.tldr)}</p><ul>{points}</ul></section>"
        return _page("/summary", body + _summary_form(live), live)

    # -- compare -------------------------------------------------------------- #
    @app.get("/compare", response_class=HTMLResponse)
    def compare_form():
        return _page("/compare", _compare_form(live), live)

    @app.post("/compare", response_class=HTMLResponse)
    def compare_route(sources: str = Form(...), model: str = Form("")):
        items = [s.strip() for s in sources.splitlines() if s.strip()][:4]
        if len(items) < 2:
            return _page("/compare", _compare_form(live, "请每行一个来源，至少 2 篇。"), live)
        from papermind.compare import compare as run_compare

        try:
            comparison = run_compare(items, model=(model or None), synthesize=live)
        except PaperMindError as exc:
            return _page("/compare", _compare_form(live, str(exc)), live)
        return comparison.to_html()

    # -- reproduce ------------------------------------------------------------ #
    @app.get("/reproduce", response_class=HTMLResponse)
    def reproduce_form():
        return _page("/reproduce", _reproduce_form(live), live)

    @app.post("/reproduce", response_class=HTMLResponse)
    def reproduce_route(source: str = Form(...), model: str = Form("")):
        report, err = _report_for(source, model, live, need="reproduction")
        if err:
            return _page("/reproduce", _reproduce_form(live, err), live)
        script = report.to_setup_script()
        body = (
            "<section class='panel'><h2>setup.sh</h2>"
            "<button class='copy-btn' onclick=\"navigator.clipboard.writeText("
            "document.getElementById('sh').textContent)\">复制</button>"
            f"<pre id='sh'>{_e(script)}</pre></section>"
        )
        return _page("/reproduce", body + _reproduce_form(live), live)

    # -- search (no model calls -> always available) ------------------------- #
    @app.get("/search", response_class=HTMLResponse)
    def search_form():
        return _page("/search", _search_form(), live)

    @app.post("/search", response_class=HTMLResponse)
    def search_route(query: str = Form(...)):
        from papermind.parser.arxiv import search_arxiv

        try:
            results = search_arxiv(query.strip(), max_results=12)
        except PaperMindError as exc:
            return _page("/search", _search_form(str(exc)), live)
        rows = ""
        for r in results:
            abs_url = f"https://arxiv.org/abs/{_e(r.arxiv_id)}"
            rows += (
                f"<tr><td class='aid'><a href='/analyze?source={_e(r.arxiv_id)}'>{_e(r.arxiv_id)}</a></td>"
                f"<td>{r.year or '—'}</td>"
                f"<td class='ttl'><a href='{abs_url}' target='_blank' rel='noopener'>{_e(r.title)}</a></td></tr>"
            )
        table = (
            "<table><thead><tr><th>arXiv</th><th>年份</th><th>标题（原文）</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        return _page("/search", _search_form() + f"<section class='panel'>{table}</section>", live)

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
# Sessions (multi-turn 问答) and PDF upload
# --------------------------------------------------------------------------- #
def _session_for(sid: Optional[str]):
    if not sid:
        return None
    with _ASK_LOCK:
        return _ASK_SESSIONS.get(sid)


def _store_session(sid: str, sess: dict) -> None:
    with _ASK_LOCK:
        _ASK_SESSIONS[sid] = sess
        while len(_ASK_SESSIONS) > _ASK_MAX_SESSIONS:
            _ASK_SESSIONS.pop(next(iter(_ASK_SESSIONS)))


def _with_session(response, sid: str):
    response.set_cookie("pm_sid", sid, httponly=True, samesite="lax", max_age=86400)
    return response


def _resolve_upload(source: str, file) -> Optional[str]:
    """An uploaded PDF (saved to a temp file) wins; otherwise the pasted id/URL."""
    if file is not None and getattr(file, "filename", ""):
        import pathlib
        import tempfile

        data = file.file.read()
        if data:
            tmp = pathlib.Path(tempfile.gettempdir()) / f"papermind_{file.filename}"
            tmp.write_bytes(data)
            return str(tmp)
    return (source or "").strip() or None


# --------------------------------------------------------------------------- #
# Design system — minimal academic (system serif display + sans body, warm
# paper, one ink-navy accent). Light/dark via CSS variables; reduced-motion safe.
# --------------------------------------------------------------------------- #
_CSS = """
:root{
  --paper:#f5f4ef;--surface:#fffdf8;--ink:#211f1b;--soft:#5d5a52;--line:#e4e1d7;
  --accent:#2c3a66;--accent-soft:#ecedf4;--green:#3f7d52;--amber:#946312;--red:#a8443f;
  --rp:12px;--rc:8px;
  --serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,"Songti SC",Georgia,serif;
  --sans:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
  --mono:"SF Mono","Cascadia Code","JetBrains Mono",Consolas,monospace;
}
@media(prefers-color-scheme:dark){:root{
  --paper:#15141a;--surface:#1d1c24;--ink:#ece9e1;--soft:#a6a299;--line:#2d2c35;
  --accent:#9fb0e8;--accent-soft:#23222e;--green:#7fb98f;--amber:#d6ab63;--red:#e08a85;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 var(--sans);
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.pm-wrap{max-width:880px;margin:0 auto;padding:0 22px 96px}
.pm-mast{padding:44px 0 18px}
.pm-logo{font:600 1.95rem/1 var(--serif);letter-spacing:-.01em}
.pm-tag{display:block;margin-top:9px;color:var(--soft);font-size:.92rem}
.pm-nav{display:flex;flex-wrap:wrap;gap:24px;border-bottom:1px solid var(--line);margin-bottom:30px}
.pm-nav a{text-decoration:none;color:var(--soft);font-weight:600;font-size:.95rem;
  padding-bottom:13px;border-bottom:2px solid transparent;transition:color .15s,border-color .15s}
.pm-nav a:hover{color:var(--ink)}
.pm-nav a.on{color:var(--accent);border-bottom-color:var(--accent)}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--rp);
  padding:26px 28px;margin:18px 0;box-shadow:0 1px 2px rgba(33,31,27,.04),0 10px 28px rgba(33,31,27,.05)}
h1,h2,h3{font-family:var(--serif);font-weight:600;letter-spacing:-.01em;line-height:1.25}
.panel h2{margin:0 0 .35em;font-size:1.4rem}.panel h3{margin:1.4em 0 .4em;font-size:1.1rem}
.panel p{margin:.2em 0 .9em}.lead{color:var(--soft);margin:0 0 18px;font-size:.96rem}
.panel ul{padding-left:1.15em;margin:.4em 0}.panel li{margin:.35em 0}
label{display:block;font-weight:600;font-size:.92rem;margin:2px 0 6px}
.hint{color:var(--soft);font-weight:400}
code{font-family:var(--mono);font-size:.86em;background:var(--accent-soft);padding:1px 6px;border-radius:5px}
input,textarea,select{width:100%;font:inherit;color:var(--ink);background:var(--paper);
  border:1px solid var(--line);border-radius:var(--rc);padding:11px 13px;margin-bottom:18px}
input:focus,textarea:focus,select:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-soft)}
textarea{min-height:96px;resize:vertical}
input[type=file]{padding:9px 12px;font-size:.9rem;color:var(--soft)}
input[type=file]::file-selector-button{font:600 .85rem var(--sans);color:var(--accent);background:var(--accent-soft);
  border:0;border-radius:6px;padding:6px 12px;margin-right:12px;cursor:pointer}
button{font:600 .98rem var(--sans);background:var(--accent);color:#fff;border:0;border-radius:var(--rc);
  padding:11px 24px;cursor:pointer;transition:filter .15s,transform .08s}
button:hover{filter:brightness(1.08)}button:active{transform:translateY(1px)}
@media(prefers-color-scheme:dark){button{color:#15141a}}
.ex{margin:16px 0 0;color:var(--soft);font-size:.88rem}
.ex a{display:inline-block;margin-left:6px;padding:3px 11px;border:1px solid var(--line);border-radius:var(--rc);
  text-decoration:none;color:var(--accent)}
.ex a:hover{background:var(--accent-soft)}
.chat .turn{padding:20px 0;border-top:1px solid var(--line)}
.chat .turn:first-child{padding-top:0;border-top:0}
.q{font-weight:600;margin:0 0 12px}
.q::before{content:'问 ';color:var(--accent);font-weight:700}
.err{color:var(--red);margin:0 0 14px;font-size:.92rem}
.seg{border-left:3px solid var(--line);padding:2px 0 2px 16px;margin:16px 0}
.seg .tag{display:block;font:600 .8rem var(--sans);letter-spacing:.02em;margin-bottom:5px}
.seg small{display:block;margin-top:6px;color:var(--soft)}
.seg.fact{border-left-color:var(--green)}.seg.fact .tag{color:var(--green)}
.seg.inf{border-left-color:var(--amber)}.seg.inf .tag{color:var(--amber)}
.seg.oos{border-left-color:var(--red)}.seg.oos .tag{color:var(--red)}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th{text-align:left;font:600 .76rem var(--sans);letter-spacing:.04em;text-transform:uppercase;
  color:var(--soft);padding:0 12px 10px}
td{padding:12px;border-top:1px solid var(--line);vertical-align:top}
.aid{font-family:var(--mono);font-size:.9rem;white-space:nowrap}
.ttl a{color:var(--ink)}.ttl a:hover{color:var(--accent)}
pre{font:.9rem/1.6 var(--mono);background:#1b1a22;color:#e9e6df;padding:16px 18px;
  border-radius:var(--rc);overflow-x:auto;white-space:pre-wrap}
.copy-btn{font:600 .85rem var(--sans);background:transparent;color:var(--accent);
  border:1px solid var(--line);border-radius:var(--rc);padding:6px 14px;margin-bottom:12px}
.copy-btn:hover{background:var(--accent-soft);filter:none}
a{color:var(--accent)}
.pm-foot{display:flex;flex-wrap:wrap;align-items:center;color:var(--soft);font-size:.85rem;
  border-top:1px solid var(--line);margin-top:42px;padding-top:18px}
.pm-foot>*+*{margin-left:18px;padding-left:18px;border-left:1px solid var(--line)}
.pm-foot .m{font-weight:600;color:var(--ink)}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
@media(max-width:600px){.pm-wrap{padding:0 16px 64px}.pm-mast{padding:30px 0 14px}.panel{padding:22px}}
"""


def _active_model_label() -> str:
    from papermind.config import load_config

    m = load_config().default_model
    return _MODEL_LABELS.get(m, m)


def _page(active: str, body: str, live: bool) -> str:
    nav = "".join(
        f"<a href='{href}' class='{'on' if href == active else ''}'>{label}</a>" for href, label in _TABS
    )
    mode = "实时分析 · 用本机 key（有成本）" if live else "演示模式 · 仅已缓存论文"
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>PaperMind</title><style>{_CSS}</style></head><body><div class='pm-wrap'>"
        "<header class='pm-mast'><span class='pm-logo'>PaperMind</span>"
        "<span class='pm-tag'>读懂一篇 arXiv 论文 · 结构化分析 · 带原文依据的问答 · 复现指南</span></header>"
        f"<nav class='pm-nav'>{nav}</nav><main>{body}</main>"
        "<footer class='pm-foot'>"
        f"<span>当前模型 <span class='m'>{_e(_active_model_label())}</span></span>"
        f"<span>{mode}</span><a href='/demo'>离线示例</a></footer>"
        "</div><script>document.addEventListener('submit',function(e){var b=e.target.querySelector('button');"
        "if(b&&!b.dataset.t){b.dataset.t=b.textContent;b.disabled=true;b.textContent='处理中…';}});"
        "document.addEventListener('click',function(e){var a=e.target.closest('.ex a');if(!a)return;"
        "e.preventDefault();var i=document.querySelector(\"input[name='source']\");"
        "if(i){i.value=a.dataset.id;i.focus();}});</script>"
        "</body></html>"
    )


def _err(error: str) -> str:
    return f"<p class='err'>{_e(error)}</p>" if error else ""


_EXAMPLES = [("1706.03762", "Transformer"), ("2307.08691", "FlashAttention-2"), ("1810.04805", "BERT")]


def _examples_row() -> str:
    chips = "".join(f"<a href='#' data-id='{_e(i)}'>{_e(name)}</a>" for i, name in _EXAMPLES)
    return f"<p class='ex'>没有目标？试试 {chips}</p>"


def _analyze_form(live: bool, error: str = "") -> str:
    note = "" if live else "<p class='lead'>演示模式：仅展示已缓存论文。分析新论文请用 CLI，或以 <code>--live</code> 启动。</p>"
    return (
        "<section class='panel'><h2>分析一篇论文</h2>"
        "<p class='lead'>四模块结构化解读：核心贡献 · 方法与图示 · 关联工作 · 复现要点。</p>"
        f"{note}{_err(error)}"
        "<form method='post' action='/analyze' enctype='multipart/form-data'>"
        "<label>论文 <span class='hint'>arXiv id / URL</span></label>"
        "<input name='source' placeholder='2307.08691  或  https://arxiv.org/abs/2307.08691' autofocus>"
        "<label>或上传 PDF <span class='hint'>本地论文</span></label>"
        "<input type='file' name='file' accept='application/pdf'>"
        "<button>分析</button></form>"
        f"{_examples_row()}</section>"
    )


def _ask_form(live: bool, error: str = "") -> str:
    return (
        "<section class='panel'><h2>问答</h2>"
        "<p class='lead'>基于原文回答，分层标注：论文事实 · 基于论文的推理 · 超出论文范围，并附原文依据。</p>"
        f"{_err(error)}"
        "<form method='post' action='/ask'>"
        "<label>论文 <span class='hint'>arXiv id / URL</span></label><input name='source' placeholder='2307.08691'>"
        "<label>问题</label><input name='question' placeholder='为什么要除以 √d_k？'>"
        "<label>模式 <span class='hint'>strict 更保守 · explore 更发散</span></label>"
        "<select name='mode'><option value='balanced'>balanced</option>"
        "<option value='strict'>strict</option><option value='explore'>explore</option></select>"
        "<button>提问</button></form></section>"
    )


def _summary_form(live: bool, error: str = "") -> str:
    return (
        "<section class='panel'><h2>速读</h2>"
        "<p class='lead'>一句话 TL;DR + 几条要点（单次调用，比完整分析更快）。</p>"
        f"{_err(error)}"
        "<form method='post' action='/summary'>"
        "<label>论文 <span class='hint'>arXiv id / URL</span></label><input name='source' placeholder='2307.08691'>"
        "<button>速读</button></form></section>"
    )


def _compare_form(live: bool, error: str = "") -> str:
    return (
        "<section class='panel'><h2>多篇对比</h2>"
        "<p class='lead'>2–4 篇论文的问题 · 方法 · 结果横向对照。</p>"
        f"{_err(error)}"
        "<form method='post' action='/compare'>"
        "<label>论文 <span class='hint'>每行一个，2–4 篇</span></label>"
        "<textarea name='sources' placeholder='2307.08691&#10;1706.03762'></textarea>"
        "<button>对比</button></form></section>"
    )


def _reproduce_form(live: bool, error: str = "") -> str:
    return (
        "<section class='panel'><h2>复现指南</h2>"
        "<p class='lead'>导出可一键运行的环境与步骤脚本（setup.sh）。</p>"
        f"{_err(error)}"
        "<form method='post' action='/reproduce'>"
        "<label>论文 <span class='hint'>arXiv id / URL</span></label><input name='source' placeholder='2307.08691'>"
        "<button>导出</button></form></section>"
    )


def _search_form(error: str = "") -> str:
    return (
        "<section class='panel'><h2>搜索 arXiv</h2>"
        "<p class='lead'>关键词检索，不调用模型。点结果中的 arXiv id 即可分析 / 问答。</p>"
        f"{_err(error)}"
        "<form method='post' action='/search'>"
        "<label>关键词</label><input name='query' placeholder='flash attention' autofocus>"
        "<button>搜索</button></form></section>"
    )


def _answer_segments_html(answer) -> str:
    kind = {"fact": ("论文事实", "fact"), "inference": ("基于论文的推理", "inf"), "out_of_scope": ("超出论文范围", "oos")}
    segs = []
    for s in answer.segments:
        label, cls = kind.get(s.kind, (s.kind, ""))
        tag = label
        if s.kind == "inference" and s.confidence:
            tag += f" · 置信度 {_e(s.confidence)}"
        reason = f"<small>推理依据：{_e(s.reasoning)}</small>" if s.reasoning else ""
        segs.append(f"<div class='seg {cls}'><span class='tag'>{tag}</span><div>{_e(s.text)}</div>{reason}</div>")
    ev = ""
    if answer.evidence:
        rows = ""
        for e in answer.evidence:
            loc = (e.section or "") + (f" p.{e.page}" if e.page else "")
            mark = "" if e.verified else "<span class='hint'>未核实 · </span>"
            rows += f"<tr><td class='aid'>{_e(loc) or '—'}</td><td>{mark}{_e(e.text)}</td></tr>"
        ev = f"<h3>原文依据</h3><table><tbody>{rows}</tbody></table>"
    return f"{''.join(segs)}{ev}"


def _chat_log_html(log) -> str:
    turns = ""
    for question, answer in log:
        turns += f"<div class='turn'><p class='q'>{_e(question)}</p>{_answer_segments_html(answer)}</div>"
    return f"<section class='panel chat'>{turns}</section>" if turns else ""


def _e(text) -> str:
    return _html.escape(str(text)) if text is not None else ""
