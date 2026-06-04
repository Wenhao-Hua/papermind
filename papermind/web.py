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
import os
import secrets
import sys
import threading
import time
from typing import Optional

from papermind.errors import PaperMindError, SourceError

# X-Forwarded-For / X-Real-IP are appended by the client and trivially forged, so
# trusting them would let anyone reset their per-IP quota. Only cf-connecting-ip
# (set and overwritten by Cloudflare) is trustworthy, and only when we're actually
# deployed behind it — opt in with PAPERMIND_TRUST_PROXY=1.
_TRUST_PROXY = os.getenv("PAPERMIND_TRUST_PROXY", "").strip().lower() in ("1", "true", "yes", "on")

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


# --------------------------------------------------------------------------- #
# Rate limiting — only gates LIVE, model-calling requests so a public --live
# deployment can't drain the host's API key. Free ops (demo, search, cached
# reports) are never limited. In-memory, per-instance; resets every window.
# --------------------------------------------------------------------------- #
class RateLimiter:
    def __init__(self, per_ip: int = 8, global_max: int = 300, window: int = 86400):
        self.per_ip = per_ip
        self.global_max = global_max
        self.window = window
        self._lock = threading.Lock()
        self._ip: dict = {}
        self._global = 0
        self._reset = time.time() + window

    def _rollover(self) -> None:
        if time.time() >= self._reset:
            self._ip.clear()
            self._global = 0
            self._reset = time.time() + self.window

    def take(self, ip: str):
        """Reserve one slot for ``ip``. Returns (allowed, scope) — scope in {'ip','global',''}."""
        with self._lock:
            self._rollover()
            if self.global_max and self._global >= self.global_max:
                return False, "global"
            if self.per_ip and self._ip.get(ip, 0) >= self.per_ip:
                return False, "ip"
            self._ip[ip] = self._ip.get(ip, 0) + 1
            self._global += 1
            return True, ""

    def release(self, ip: str) -> None:
        """Give a slot back (a reserved request failed before producing any model output).
        Guarded so a window rollover between take/release can't drive a counter negative."""
        with self._lock:
            if self._ip.get(ip, 0) > 0:
                self._ip[ip] -= 1
            if self._global > 0:
                self._global -= 1


def _client_ip(request) -> str:
    if _TRUST_PROXY:
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.split(",")[0].strip()
    return request.client.host if getattr(request, "client", None) else "unknown"


def _quota_msg(scope: str, limiter: "RateLimiter") -> str:
    if scope == "global":
        return f"本服务今日总额度已用完（{limiter.global_max} 次/天）。请明天再来，或本地运行：pip install paper-mind。"
    return f"今日额度已用完（每人 {limiter.per_ip} 次/天）。请明天再来，或本地零成本运行：pip install paper-mind 后 papermind analyze。"


# --------------------------------------------------------------------------- #
# Async analysis jobs — a long analysis runs in a background thread so the HTTP
# request returns immediately (no Cloudflare 100s origin timeout / 524). The
# page polls /job/{id} for real progress, then loads /result/{id}.
# --------------------------------------------------------------------------- #
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOBS_MAX = 64
_JOBS_TTL = 1800  # keep a finished job's result reachable for 30 min (counted from completion)
_JOBS_RUNNING_MAX = 900  # a job still "running" after 15 min is treated as hung -> timed out


def _new_job() -> str:
    jid = secrets.token_hex(8)
    with _JOBS_LOCK:
        now = time.time()
        for k, v in list(_JOBS.items()):
            age = now - v["ts"]
            if v["status"] in ("done", "error"):
                if age > _JOBS_TTL:  # result expired -> reclaim
                    _JOBS.pop(k, None)
            elif age > _JOBS_RUNNING_MAX:  # stuck running far too long -> mark timed out (now evictable)
                v.update(status="error", error="分析超时，请稍后重试，或本地运行（pip install paper-mind）。", ts=now)
        _JOBS[jid] = {"status": "running", "step": "开始", "html": "", "error": "", "kind": "analyze", "ts": now}
        # over capacity: drop the oldest finished job; only if there are none (sustained overload)
        # fall back to dropping the oldest running job, so _JOBS can never grow without bound.
        while len(_JOBS) > _JOBS_MAX:
            finished = [(k, v["ts"]) for k, v in _JOBS.items() if v["status"] in ("done", "error")]
            pool = finished or [(k, v["ts"]) for k, v in _JOBS.items() if k != jid]
            if not pool:
                break
            _JOBS.pop(min(pool, key=lambda kv: kv[1])[0], None)
    return jid


def _set_job(jid: str, **kw) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS:
            if kw.get("status") in ("done", "error"):
                kw.setdefault("ts", time.time())  # TTL is measured from completion, not creation
            _JOBS[jid].update(kw)


def _get_job(jid: str):
    with _JOBS_LOCK:
        return dict(_JOBS[jid]) if jid in _JOBS else None


def _start_job(work, kind: str = "analyze") -> str:
    """Run ``work(job_id) -> html`` in a daemon thread; store result/error on the job.

    ``kind`` ("analyze" | "ask") tells /result which tab/retry-form to render.
    """
    jid = _new_job()
    _set_job(jid, kind=kind)

    def run():
        try:
            _set_job(jid, status="done", html=work(jid))
        except PaperMindError as exc:
            _set_job(jid, status="error", error=str(exc))  # curated, user-facing message
        except Exception as exc:  # noqa: BLE001 - unexpected: log internally, never echo to the public page
            print(f"[papermind] job {jid} ({kind}) failed: {exc!r}", file=sys.stderr)
            _set_job(jid, status="error", error="分析失败，请稍后重试，或本地运行（pip install paper-mind）。")

    threading.Thread(target=run, daemon=True).start()
    return jid


def _poll_js(job_id: str) -> str:
    # Drives the inline progress bar on the job page (elements jp-* are rendered
    # just above this script, so they exist when it runs). The bar fills as
    # analyze reports each real step and creeps with elapsed time so it never
    # freezes; jumps to 100% on done. Any non-running state -> /result (escaped).
    return (
        "(function(){var ORDER=['解析 PDF','贡献与创新点','技术细节','知识关联','复现指南','图示匹配/生成'];"
        "var se=document.getElementById('jp-step'),sec=document.getElementById('jp-sec'),"
        "pf=document.getElementById('jp-fill'),pp=document.getElementById('jp-pct'),t0=Date.now(),pct=5,sp=0;"
        "function paint(){var w=Math.min(pct,99);if(pf){pf.style.width=w+'%';}if(pp){pp.textContent=Math.round(w);}}"
        "setInterval(function(){var el=(Date.now()-t0)/1000;if(sec){sec.textContent=Math.round(el);}"
        "pct=Math.max(pct,Math.min(90,5+el/100*85));paint();},300);"
        "function poll(){fetch('/job/" + job_id + "').then(function(r){return r.json();}).then(function(j){"
        "if(j.step){if(se){se.textContent=j.step;}var i=ORDER.indexOf(j.step);"
        "if(i>=0){sp=Math.round((i+1)/(ORDER.length+1)*100);pct=Math.max(pct,sp);paint();}}"
        "if(j.status==='running'){setTimeout(poll,2000);}"
        "else{pct=100;paint();location.replace('/result/" + job_id + "');}"
        "}).catch(function(){setTimeout(poll,3000);});}paint();poll();})();"
    )


def _job_page(job_id: str, live: bool, tab: str = "/") -> str:
    # Progress bar lives INLINE in the card (not the shared overlay) so the
    # elements exist before the inline poll script runs.
    body = (
        "<section class='panel'><h2>分析中…</h2>"
        "<div class='pm-bar' style='margin:14px 0'><span id='jp-fill'></span></div>"
        "<p class='pm-busy-step' id='jp-step'>开始</p>"
        "<p class='pm-busy-time'><b id='jp-pct'>0</b>% · 已用 <b id='jp-sec'>0</b> 秒</p>"
        "<p class='lead'>完成后自动显示结果。重论文可能要 1–3 分钟，可以离开本页稍后再回来。</p></section>"
        f"<script>{_poll_js(job_id)}</script>"
    )
    return _page(tab, body, live)


def create_app(live: bool = False, rate_per_ip: int = 8, rate_global: int = 300,
               with_figures: bool = True, svg_figures: bool = True, no_cache: bool = False):
    try:
        from fastapi import Cookie, FastAPI, File, Form, Request, UploadFile
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("Web demo 需要 FastAPI。安装：pip install 'paper-mind[web]'") from exc

    app = FastAPI(title="PaperMind", docs_url=None, redoc_url=None)
    limiter = RateLimiter(rate_per_ip, rate_global)

    def gate(request) -> Optional[str]:
        """Reserve a quota slot for this request's IP; return an error message if over."""
        ok, scope = limiter.take(_client_ip(request))
        return None if ok else _quota_msg(scope, limiter)

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
        return _page("/", _hero() + _analyze_form(live), live)

    def _analyze_async(request, src, model, refresh):
        """Live analysis -> background job + polling page (avoids the 100s timeout).
        A cached result still resolves in one quick poll. Quota is reserved up front."""
        blocked = gate(request)
        if blocked:
            return _page("/", _analyze_form(live, blocked), live)
        figs = with_figures

        def work(jid):
            from papermind.analyze import analyze as run_analyze
            from papermind.config import load_config

            report = run_analyze(
                src, model=(model or None), config=load_config(),
                with_figures=figs, svg_figures=(figs and svg_figures), refresh=refresh,
                on_progress=lambda step: _set_job(jid, step=step),
            )
            return report.to_html()

        return _job_page(_start_job(work), live, "/")

    @app.post("/analyze", response_class=HTMLResponse)
    def analyze_route(request: Request, source: str = Form(""), model: str = Form(""),
                      refresh: str = Form(""), file: Optional[UploadFile] = File(None)):
        try:
            src = _resolve_upload(source, file)
        except PaperMindError as exc:
            return _page("/", _analyze_form(live, str(exc)), live)
        if not src:
            return _page("/", _analyze_form(live, "请填入论文 URL（arXiv 链接或 PDF 直链），或上传 PDF。"), live)
        if not live:  # demo: only already-cached reports; never runs a model or hits the network
            report = _demo_cached(src)
            return report.to_html() if report is not None else _page("/", _analyze_form(live, _DEMO_MSG), live)
        return _analyze_async(request, src, model, no_cache or bool(refresh))

    @app.get("/analyze", response_class=HTMLResponse)
    def analyze_get(request: Request, source: str = ""):
        if not source.strip():
            return _page("/", _analyze_form(live), live)
        if not live:
            report = _demo_cached(source)
            return report.to_html() if report is not None else _page("/", _analyze_form(live, _DEMO_MSG), live)
        return _analyze_async(request, source, "", no_cache)

    @app.get("/job/{job_id}")
    def job_status(job_id: str):
        job = _get_job(job_id)
        if job is None:
            return {"status": "missing"}
        return {"status": job["status"], "step": job.get("step", ""), "error": job.get("error", "")}

    @app.get("/result/{job_id}", response_class=HTMLResponse)
    def job_result(job_id: str):
        job = _get_job(job_id)
        kind = (job or {}).get("kind", "analyze")
        tab = "/ask" if kind == "ask" else "/"
        form = _ask_form if kind == "ask" else _analyze_form
        if job is None:
            # job lost (server restarted / expired) -> give back a usable retry form
            return _page(tab, form(live, "结果已过期或服务刚更新过，请重新提交。"), live)
        if job["status"] == "done":
            return job["html"]
        if job["status"] == "error":
            return _page(tab, form(live, job["error"]), live)
        return _job_page(job_id, live, tab)  # still running -> keep polling

    # -- ask (multi-turn grounded Q&A) --------------------------------------- #
    @app.get("/ask", response_class=HTMLResponse)
    def ask_form(pm_sid: Optional[str] = Cookie(None)):
        sess = _session_for(pm_sid)
        log = _chat_log_html(sess["log"]) if sess and sess.get("log") else ""
        return _page("/ask", log + _ask_form(live), live)

    @app.post("/ask", response_class=HTMLResponse)
    def ask_route(
        request: Request,
        source: str = Form(...),
        question: str = Form(...),
        model: str = Form(""),
        mode: str = Form("balanced"),
        pm_sid: Optional[str] = Cookie(None),
    ):
        if not live:
            return _page("/ask", _ask_form(live, "演示模式不支持问答（需调用模型）。请以 --live 启动。"), live)

        sess = _session_for(pm_sid)
        prior = _chat_log_html(sess["log"]) if sess and sess.get("log") else ""
        if not source.strip() or not question.strip():
            return _page("/ask", prior + _ask_form(live, "请填写论文和问题。"), live)

        blocked = gate(request)
        if blocked:
            return _page("/ask", prior + _ask_form(live, blocked), live)

        # Building the chat (download + parse + index) and answering can take tens of
        # seconds on a fresh paper, which would blow Cloudflare's 100s origin timeout
        # (524). Run it in a background job and return a polling page at once, like the
        # other tabs. The result/error is rendered by /result (kind="ask").
        sid = pm_sid or secrets.token_hex(8)
        ip = _client_ip(request)
        paper, q, md, mdl = source.strip(), question.strip(), mode, (model or None)

        def work(jid):
            from papermind.qa.chat import PaperChat

            try:
                sess2 = _session_for(sid)
                if sess2 is None or sess2.get("paper") != paper:
                    _set_job(jid, step="下载并解析论文、建立索引…")
                    sess2 = {"paper": paper, "chat": PaperChat(paper, model=mdl, mode=md), "log": []}
                sess2["chat"].mode = md  # honor a mode change between turns
                _set_job(jid, step="检索原文并生成回答…")
                answer = sess2["chat"].ask(q)
            except PaperMindError:
                limiter.release(ip)  # the user got no answer -> refund their daily quota slot
                raise
            sess2["log"].append((q, answer))
            _store_session(sid, sess2)
            return _page("/ask", _chat_log_html(sess2["log"]) + _ask_form(live), live)

        jid = _start_job(work, kind="ask")
        return _with_session(HTMLResponse(_job_page(jid, live, "/ask")), sid)

    # -- summary -------------------------------------------------------------- #
    @app.get("/summary", response_class=HTMLResponse)
    def summary_form():
        return _page("/summary", _summary_form(live), live)

    @app.post("/summary", response_class=HTMLResponse)
    def summary_route(request: Request, source: str = Form(...), model: str = Form("")):
        if not live:
            return _page("/summary", _summary_form(live, "演示模式不支持速读（需调用模型）。请以 --live 启动。"), live)
        blocked = gate(request)
        if blocked:
            return _page("/summary", _summary_form(live, blocked), live)
        src = source.strip()

        def work(jid):
            from papermind.summarize import summarize

            result, _usage = summarize(src, model=(model or None))
            points = "".join(f"<li>{_e(p)}</li>" for p in result.key_points)
            body = f"<section class='panel'><h2>{_e(result.title)}</h2><p>{_e(result.tldr)}</p><ul>{points}</ul></section>"
            return _page("/summary", body + _summary_form(live), live)

        return _job_page(_start_job(work), live, "/summary")

    # -- compare -------------------------------------------------------------- #
    @app.get("/compare", response_class=HTMLResponse)
    def compare_form():
        return _page("/compare", _compare_form(live), live)

    @app.post("/compare", response_class=HTMLResponse)
    def compare_route(request: Request, sources: str = Form(...), model: str = Form("")):
        items = [s.strip() for s in sources.splitlines() if s.strip()][:4]
        if len(items) < 2:
            return _page("/compare", _compare_form(live, "请每行一个来源，至少 2 篇。"), live)
        if not live:
            return _page("/compare", _compare_form(live, "演示模式不支持对比（需调用模型）。请以 --live 启动。"), live)
        blocked = gate(request)
        if blocked:
            return _page("/compare", _compare_form(live, blocked), live)

        def work(jid):
            from papermind.compare import compare as run_compare

            return run_compare(items, model=(model or None), synthesize=True).to_html()

        return _job_page(_start_job(work), live, "/compare")

    # -- reproduce ------------------------------------------------------------ #
    @app.get("/reproduce", response_class=HTMLResponse)
    def reproduce_form():
        return _page("/reproduce", _reproduce_form(live), live)

    @app.post("/reproduce", response_class=HTMLResponse)
    def reproduce_route(request: Request, source: str = Form(...), model: str = Form("")):
        if not live:  # demo: only already-cached reports
            report = _demo_cached(source)
            if report is not None and report.reproduction is not None:
                return _page("/reproduce", _repro_body(report) + _reproduce_form(live), live)
            return _page("/reproduce", _reproduce_form(live, "演示模式仅展示已缓存论文的复现。请以 --live 启动。"), live)
        blocked = gate(request)
        if blocked:
            return _page("/reproduce", _reproduce_form(live, blocked), live)
        src = source.strip()

        def work(jid):
            from papermind.analyze import analyze as run_analyze
            from papermind.config import load_config

            report = run_analyze(src, model=(model or None), config=load_config(),
                                 with_figures=False, refresh=no_cache,
                                 on_progress=lambda step: _set_job(jid, step=step))
            if report.reproduction is None:
                raise PaperMindError("这篇论文没有可导出的复现信息。")
            return _page("/reproduce", _repro_body(report) + _reproduce_form(live), live)

        return _job_page(_start_job(work), live, "/reproduce")

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
            from urllib.parse import quote

            abs_url = f"https://arxiv.org/abs/{r.arxiv_id}"
            rows += (
                f"<tr><td class='aid'><a href='/analyze?source={quote(abs_url, safe='')}'>{_e(r.arxiv_id)}</a></td>"
                f"<td>{r.year or '—'}</td>"
                f"<td class='ttl'><a href='{_e(abs_url)}' target='_blank' rel='noopener'>{_e(r.title)}</a></td></tr>"
            )
        table = (
            "<table><thead><tr><th>arXiv</th><th>年份</th><th>标题（原文）</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        return _page("/search", _search_form() + f"<section class='panel'>{table}</section>", live)

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, live: bool = False,
          rate_per_ip: int = 8, rate_global: int = 300,
          with_figures: bool = True, svg_figures: bool = True, no_cache: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("Web demo 需要 uvicorn。安装：pip install 'paper-mind[web]'") from exc
    app = create_app(
        live=live, rate_per_ip=rate_per_ip, rate_global=rate_global,
        with_figures=with_figures, svg_figures=svg_figures, no_cache=no_cache,
    )
    uvicorn.run(app, host=host, port=port)


# --------------------------------------------------------------------------- #
# Demo (cached-only) lookups + small renderers
# --------------------------------------------------------------------------- #
_DEMO_MSG = "演示模式仅展示已缓存的论文。分析新论文请用 CLI，或以 --live 启动服务。"


def _demo_cached(source: str):
    """A cached report for ``source`` WITHOUT any network/download (demo mode)."""
    from papermind.cache import latest_report_path, load_cached_report
    from papermind.parser.arxiv import cache_dir_for

    cache_dir = cache_dir_for(source)
    if cache_dir is None:
        return None
    path = latest_report_path(cache_dir)
    return load_cached_report(path) if path else None


def _repro_body(report) -> str:
    script = report.to_setup_script()
    return (
        "<section class='panel'><h2>setup.sh</h2>"
        "<button class='copy-btn' onclick=\"navigator.clipboard.writeText("
        "document.getElementById('sh').textContent)\">复制</button>"
        f"<pre id='sh'>{_e(script)}</pre></section>"
    )


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
        _ASK_SESSIONS.pop(sid, None)  # re-insert at the end: evict by least-recently-stored, not creation order
        _ASK_SESSIONS[sid] = sess
        while len(_ASK_SESSIONS) > _ASK_MAX_SESSIONS:
            _ASK_SESSIONS.pop(next(iter(_ASK_SESSIONS)))


def _with_session(response, sid: str):
    response.set_cookie("pm_sid", sid, httponly=True, samesite="lax", max_age=86400)
    return response


def _resolve_upload(source: str, file) -> Optional[str]:
    """An uploaded PDF (saved to a temp file) wins; otherwise the pasted id/URL.

    The client-supplied ``file.filename`` is never used to build the path — it is
    attacker-controlled (path traversal: ``../../x``) and collides across users
    (two ``paper.pdf`` uploads would share a cache key and leak reports). We write
    to a fresh random temp file and validate size + PDF magic first.
    """
    if file is not None and getattr(file, "filename", ""):
        import tempfile

        from papermind.net import MAX_DOWNLOAD_BYTES

        data = file.file.read(MAX_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise SourceError(f"上传的 PDF 过大（>{MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB）。")
        if data:
            if not data.startswith(b"%PDF"):
                raise SourceError("上传的文件不是 PDF。")
            fd, tmp = tempfile.mkstemp(prefix="papermind_", suffix=".pdf")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            return tmp
    return (source or "").strip() or None


# --------------------------------------------------------------------------- #
# Design system — minimal academic (system serif display + sans body, warm
# paper, one ink-navy accent). Light/dark via CSS variables; reduced-motion safe.
# --------------------------------------------------------------------------- #
_CSS = """
:root{
  --paper:#f5f4ef;--surface:#fffdf8;--ink:#211f1b;--soft:#5d5a52;--line:#e4e1d7;
  --accent:#2c3a66;--accent-soft:#ecedf4;--green:#3f7d52;--amber:#946312;--red:#a8443f;
  --rp:14px;--rc:9px;
  --shadow:0 1px 2px rgba(33,31,27,.05), 0 10px 30px rgba(33,31,27,.06);
  --shadow-lg:0 2px 4px rgba(33,31,27,.05), 0 18px 44px rgba(33,31,27,.08), 0 40px 80px rgba(33,31,27,.05);
  --serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,"Songti SC",Georgia,serif;
  --sans:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
  --mono:"SF Mono","Cascadia Code","JetBrains Mono",Consolas,monospace;
}
@media(prefers-color-scheme:dark){:root{
  --paper:#15141a;--surface:#1d1c24;--ink:#ece9e1;--soft:#a6a299;--line:#2d2c35;
  --accent:#9fb0e8;--accent-soft:#23222e;--green:#7fb98f;--amber:#d6ab63;--red:#e08a85;
}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;color:var(--ink);font:16px/1.7 var(--sans);
  background:var(--paper) radial-gradient(1100px 520px at 50% -8%, rgba(44,58,102,.06), transparent 72%) no-repeat fixed;
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
.hero{padding:6px 0 22px}
.hero-h{font:600 3.3rem/1.08 var(--serif);letter-spacing:-.025em;margin:.1em 0 .3em;text-wrap:balance}
.hero-sub{color:var(--soft);font-size:1.12rem;line-height:1.6;max-width:56ch;margin:0 0 20px}
.hero-mods{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 24px}
.hero-mods span{font-size:.85rem;background:var(--accent-soft);border:1px solid var(--line);
  border-radius:999px;padding:6px 14px;transition:transform .15s,box-shadow .15s}
.hero-mods span:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(33,31,27,.08)}
.hero-fig{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--rp);
  padding:20px 22px;box-shadow:var(--shadow-lg);
  transition:transform .25s cubic-bezier(.2,.7,.2,1),box-shadow .25s}
.hero-fig:hover{transform:translateY(-3px)}
.hero-fig svg{max-width:100%;height:auto;display:block;margin:0 auto}
.hero-fig figcaption{margin-top:14px;color:var(--soft);font-size:.84rem;text-align:center}
@media(max-width:640px){.hero-h{font-size:2.3rem}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--rp);
  padding:28px 30px;margin:20px 0;box-shadow:var(--shadow)}
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
  padding:12px 26px;cursor:pointer;box-shadow:0 1px 2px rgba(44,58,102,.24),0 6px 16px rgba(44,58,102,.20);
  transition:transform .12s,box-shadow .18s,filter .15s}
button:hover{filter:brightness(1.06);transform:translateY(-1px);
  box-shadow:0 2px 4px rgba(44,58,102,.26),0 12px 26px rgba(44,58,102,.28)}
button:active{transform:translateY(0)}
@media(prefers-color-scheme:dark){button{color:#15141a}}
.ex{margin:16px 0 0;color:var(--soft);font-size:.88rem}
.ex a{display:inline-block;margin-left:6px;padding:4px 12px;border:1px solid var(--line);border-radius:999px;
  text-decoration:none;color:var(--accent);transition:transform .15s,background .15s,border-color .15s}
.ex a:hover{background:var(--accent-soft);border-color:var(--accent);transform:translateY(-1px)}
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
.pm-busy{display:none;position:fixed;inset:0;z-index:50;background:rgba(20,20,30,.45);
  align-items:center;justify-content:center}
.pm-busy.on{display:flex}
.pm-busy-card{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:26px 32px;min-width:300px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.28)}
.pm-bar{height:9px;border-radius:6px;background:var(--accent-soft);overflow:hidden;margin-bottom:16px}
.pm-bar span{display:block;height:100%;width:5%;border-radius:6px;background:var(--accent);transition:width .5s ease}
.pm-busy-title{margin:0;font-weight:600}.pm-busy-step{margin:8px 0 0;color:var(--accent)}
.pm-busy-time{margin:6px 0 0;color:var(--soft);font-size:.85rem}
@media(max-width:600px){.pm-wrap{padding:0 16px 64px}.pm-mast{padding:30px 0 14px}.panel{padding:22px}}
@media(prefers-reduced-motion:no-preference){
  @keyframes pm-rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
  .hero{animation:pm-rise .55s cubic-bezier(.2,.7,.2,1) both}
  main>.panel{animation:pm-rise .55s cubic-bezier(.2,.7,.2,1) .08s both}
}
"""


# On submit of a heavy op, show the busy overlay immediately; the real,
# step-driven progress bar then takes over on the job page (see _poll_js).
_BUSY_JS = """
var PM_HEAVY={'/analyze':1,'/summary':1,'/compare':1,'/reproduce':1,'/ask':1};
document.addEventListener('submit',function(e){
  var f=e.target, b=f.querySelector('button'); if(b){b.disabled=true;}
  if(!PM_HEAVY[f.getAttribute('action')||'']){return;}
  var ov=document.getElementById('pm-busy'); if(ov){ov.classList.add('on');}
  var se=document.getElementById('pm-step'); if(se){se.textContent='提交中…';}
});
document.addEventListener('click',function(e){
  var a=e.target.closest('.ex a'); if(!a){return;} e.preventDefault();
  var i=document.querySelector("input[name='source']"); if(i){i.value=a.dataset.id; i.focus();}
});
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
        "<span class='pm-tag'>读懂一篇论文（arXiv 链接或 PDF 直链）· 结构化分析 · 带原文依据的问答 · 复现指南</span></header>"
        f"<nav class='pm-nav'>{nav}</nav><main>{body}</main>"
        "<footer class='pm-foot'>"
        f"<span>当前模型 <span class='m'>{_e(_active_model_label())}</span></span>"
        f"<span>{mode}</span><a href='/demo'>离线示例</a></footer>"
        "<div id='pm-busy' class='pm-busy'><div class='pm-busy-card'>"
        "<div class='pm-bar'><span id='pm-fill'></span></div>"
        "<p class='pm-busy-title'>分析中…</p><p class='pm-busy-step' id='pm-step'>开始</p>"
        "<p class='pm-busy-time'><b id='pm-pct'>0</b>% · 已用 <b id='pm-sec'>0</b> 秒</p></div></div>"
        f"<script>{_BUSY_JS}</script>"
        "</body></html>"
    )


def _err(error: str) -> str:
    return f"<p class='err'>{_e(error)}</p>" if error else ""


_EXAMPLES = [("https://arxiv.org/abs/1706.03762", "Transformer"), ("https://arxiv.org/abs/2307.08691", "FlashAttention-2"), ("https://arxiv.org/abs/1810.04805", "BERT")]


def _examples_row() -> str:
    chips = "".join(f"<a href='#' data-id='{_e(i)}'>{_e(name)}</a>" for i, name in _EXAMPLES)
    return f"<p class='ex'>没有目标？试试 {chips}</p>"


def _load_hero_svg(name: str = "hero.svg") -> str:
    """The homepage hero figure. Bundled at ``papermind/assets/`` so it ships in the wheel
    (a pip-installed ``papermind serve`` shows it too); falls back to the repo's examples
    during local development. Absent entirely -> the hero renders without a figure."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    for path in (here / "assets" / name, here.parent / "examples" / "figures" / "transformer-fig2.svg"):
        try:
            return path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
    return ""


_HERO_SVG = _load_hero_svg()


def _hero() -> str:
    fig = (
        f"<figure class='hero-fig'>{_HERO_SVG}"
        "<figcaption>↑ PaperMind 为论文技术点自动生成的论文式教学示意图 · "
        "<a href='/demo'>看一份完整示例报告 →</a></figcaption></figure>"
    ) if _HERO_SVG else ""
    return (
        "<section class='hero'>"
        "<h1 class='hero-h'>把任意论文，读懂到能复现。</h1>"
        "<p class='hero-sub'>结构化分析 · 带原文出处并自动核验的问答 · 复现接论文的真实代码仓库。"
        "粘贴论文链接、DOI 或标题（arXiv、论文页面、PDF…），也可上传 PDF。</p>"
        "<div class='hero-mods'><span>🎯 核心贡献</span><span>🔬 技术细节 + 教学示意图</span>"
        "<span>🔗 知识关联</span><span>🛠️ 复现指南</span></div>"
        f"{fig}</section>"
    )


def _analyze_form(live: bool, error: str = "") -> str:
    note = "" if live else "<p class='lead'>演示模式：仅展示已缓存论文。分析新论文请用 CLI，或以 <code>--live</code> 启动。</p>"
    return (
        "<section class='panel'><h2>分析一篇论文</h2>"
        "<p class='lead'>四模块结构化解读：核心贡献 · 方法与图示 · 关联工作 · 复现要点。</p>"
        f"{note}{_err(error)}"
        "<form method='post' action='/analyze' enctype='multipart/form-data'>"
        "<label>论文 <span class='hint'>链接 / DOI / 标题</span></label>"
        "<input name='source' placeholder='arXiv / 论文页面 / PDF / DOI，或直接输入论文标题' autofocus>"
        "<label>或上传 PDF <span class='hint'>本地论文</span></label>"
        "<input type='file' name='file' accept='application/pdf'>"
        "<label style='font-weight:400;color:var(--soft)'>"
        "<input type='checkbox' name='refresh' value='1' style='width:auto;margin-right:8px'>忽略缓存，重新分析</label>"
        "<button>分析</button></form>"
        f"{_examples_row()}</section>"
    )


def _ask_form(live: bool, error: str = "") -> str:
    return (
        "<section class='panel'><h2>问答</h2>"
        "<p class='lead'>基于原文回答，分层标注：论文事实 · 基于论文的推理 · 超出论文范围，并附原文依据。</p>"
        f"{_err(error)}"
        "<form method='post' action='/ask'>"
        "<label>论文 <span class='hint'>论文 URL（arXiv 链接或 PDF 直链）</span></label><input name='source' placeholder='https://arxiv.org/abs/2307.08691'>"
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
        "<label>论文 <span class='hint'>论文 URL（arXiv 链接或 PDF 直链）</span></label><input name='source' placeholder='https://arxiv.org/abs/2307.08691'>"
        "<button>速读</button></form></section>"
    )


def _compare_form(live: bool, error: str = "") -> str:
    return (
        "<section class='panel'><h2>多篇对比</h2>"
        "<p class='lead'>2–4 篇论文的问题 · 方法 · 结果横向对照。</p>"
        f"{_err(error)}"
        "<form method='post' action='/compare'>"
        "<label>论文 <span class='hint'>每行一个，2–4 篇</span></label>"
        "<textarea name='sources' placeholder='https://arxiv.org/abs/2307.08691&#10;https://arxiv.org/abs/1706.03762'></textarea>"
        "<button>对比</button></form></section>"
    )


def _reproduce_form(live: bool, error: str = "") -> str:
    return (
        "<section class='panel'><h2>复现指南</h2>"
        "<p class='lead'>导出可一键运行的环境与步骤脚本（setup.sh）。</p>"
        f"{_err(error)}"
        "<form method='post' action='/reproduce'>"
        "<label>论文 <span class='hint'>论文 URL（arXiv 链接或 PDF 直链）</span></label><input name='source' placeholder='https://arxiv.org/abs/2307.08691'>"
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
