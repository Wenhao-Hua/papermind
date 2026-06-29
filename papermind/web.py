"""Full-featured web app for PaperMind (``papermind serve`` / ``papermind ui``).

A focused tool, not a marketing page: the home is 分析 itself (with an inline
"问 AI" block that reuses the same paper), plus 搜索. The whole-method framework
diagram and the reproduction guide ship inside the analysis report; every result
reuses the same core functions and HTML renderers as the CLI.

Safe by default (**demo mode**): features that would call a model only run when
their result is already cached; ``--live`` unlocks live analysis using the
server's configured keys. (搜索 is always available — it makes no model calls.)

The look is a clean, modern light-product design system (system sans, a soft
cool canvas, one blue accent, light/dark via CSS variables).
"""

from __future__ import annotations

import contextvars
import html as _html
import os
import secrets
import sys
import threading
import time
from typing import Optional

from papermind.errors import PaperMindError, SourceError

# --------------------------------------------------------------------------- #
# Language (bilingual UI) — resolved per request and stored in a ContextVar so the
# render helpers don't need a lang argument threaded through every signature.
# Default English (10k-star reach); zh via Accept-Language or an explicit ?lang=zh.
# --------------------------------------------------------------------------- #
_LANG: "contextvars.ContextVar[str]" = contextvars.ContextVar("pm_lang", default="en")


def _lang() -> str:
    return "zh"  # Chinese-only site


def _t(en: str, zh: str) -> str:
    """Chinese-only site: always return the Chinese string. (The English argument is
    kept at call sites so the strings remain available if bilingual is restored.)"""
    return zh


def _resolve_lang(cookie: Optional[str], accept_language: Optional[str]) -> str:
    if cookie in ("en", "zh"):
        return cookie
    return "zh" if (accept_language or "").lower().lstrip().startswith("zh") else "en"

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

_TABS = [
    ("/", "Analyze", "分析"),
    ("/search", "Search", "搜索"),
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
        return _t(
            f"The service's daily quota is used up ({limiter.global_max}/day). Try tomorrow, or run locally: pip install papermind-ai.",
            f"本服务今日总额度已用完（{limiter.global_max} 次/天）。请明天再来，或本地运行：pip install papermind-ai。",
        )
    return _t(
        f"Your daily quota is used up ({limiter.per_ip}/day per person). Try tomorrow, or run locally for free: pip install papermind-ai then papermind analyze.",
        f"今日额度已用完（每人 {limiter.per_ip} 次/天）。请明天再来，或本地零成本运行：pip install papermind-ai 后 papermind analyze。",
    )


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
                v.update(status="error", error="分析超时，请稍后重试，或本地运行（pip install papermind-ai）。", ts=now)
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


def _start_job(work, kind: str = "analyze", on_fail=None) -> str:
    """Run ``work(job_id) -> html`` in a daemon thread; store result/error on the job.

    ``kind`` ("analyze" | "ask") tells /result which tab/retry-form to render.
    ``on_fail`` runs on any failure — gated routes pass it to refund the reserved
    rate-limit slot (a request that produced no output shouldn't cost a daily quota).
    """
    jid = _new_job()
    _set_job(jid, kind=kind)

    def run():
        try:
            _set_job(jid, status="done", html=work(jid))
        except PaperMindError as exc:
            if on_fail:
                on_fail()
            _set_job(jid, status="error", error=str(exc))  # curated, user-facing message
        except Exception as exc:  # noqa: BLE001 - unexpected: log internally, never echo to the public page
            if on_fail:
                on_fail()
            print(f"[papermind] job {jid} ({kind}) failed: {exc!r}", file=sys.stderr)
            _set_job(jid, status="error", error="分析失败，请稍后重试，或本地运行（pip install papermind-ai）。")

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
        f"<section class='panel'><h2>{_t('Working…', '分析中…')}</h2>"
        "<div class='pm-bar' style='margin:14px 0'><span id='jp-fill'></span></div>"
        f"<p class='pm-busy-step' id='jp-step'>{_t('Starting', '开始')}</p>"
        f"<p class='pm-busy-time'><b id='jp-pct'>0</b>% · {_t('elapsed', '已用')} <b id='jp-sec'>0</b>{_t('s', ' 秒')}</p>"
        f"<p class='lead'>{_t('The result appears automatically. A heavy paper can take 1–3 min — you can leave and come back.', '完成后自动显示结果。重论文可能要 1–3 分钟，可以离开本页稍后再回来。')}</p></section>"
        f"<script>{_poll_js(job_id)}</script>"
    )
    return _page(tab, body, live)


def create_app(live: bool = False, rate_per_ip: int = 8, rate_global: int = 300,
               with_figures: bool = True, svg_figures: bool = True, no_cache: bool = False):
    try:
        from fastapi import Cookie, FastAPI, File, Form, Request, UploadFile  # noqa: F401 - Request/UploadFile used in runtime-evaluated route annotations
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("Web demo 需要 FastAPI。安装：pip install 'papermind-ai[web]'") from exc

    app = FastAPI(title="PaperMind", docs_url=None, redoc_url=None)
    limiter = RateLimiter(rate_per_ip, rate_global)

    @app.middleware("http")
    async def _set_language(request, call_next):
        q = request.query_params.get("lang")
        explicit = q if q in ("en", "zh") else None
        _LANG.set(_resolve_lang(explicit or request.cookies.get("pm_lang"),
                                request.headers.get("accept-language")))
        response = await call_next(request)
        if explicit and explicit != request.cookies.get("pm_lang"):
            response.set_cookie("pm_lang", explicit, max_age=31536000, samesite="lax")
        return response

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
        return _page("/", _analyze_form(live), live)

    def _analyze_async(request, src, model, refresh):
        """Live analysis -> background job + polling page (avoids the 100s timeout).
        A cached result still resolves in one quick poll. Quota is reserved up front."""
        blocked = gate(request)
        if blocked:
            return _page("/", _analyze_form(live, blocked), live)
        figs = with_figures
        ip = _client_ip(request)

        def work(jid):
            from papermind.analyze import analyze as run_analyze
            from papermind.config import load_config

            report = run_analyze(
                src, model=(model or None), config=load_config(),
                with_figures=figs, svg_figures=(figs and svg_figures), refresh=refresh,
                on_progress=lambda step: _set_job(jid, step=step),
            )
            return report.to_html()

        return _job_page(_start_job(work, on_fail=lambda: limiter.release(ip)), live, "/")

    @app.post("/analyze", response_class=HTMLResponse)
    def analyze_route(request: Request, source: str = Form(""), model: str = Form(""),
                      refresh: str = Form(""), file: Optional[UploadFile] = File(None)):
        try:
            src = _resolve_upload(source, file)
        except PaperMindError as exc:
            return _page("/", _analyze_form(live, str(exc)), live)
        if not src:
            return _page("/", _analyze_form(live, _t("Enter a paper link / DOI / title, or upload a PDF.", "请填入论文链接 / DOI / 标题，或上传 PDF。")), live)
        if not live:  # demo: only already-cached reports; never runs a model or hits the network
            report = _demo_cached(src)
            return report.to_html() if report is not None else _page("/", _analyze_form(live, _demo_msg()), live)
        return _analyze_async(request, src, model, no_cache or bool(refresh))

    @app.get("/analyze", response_class=HTMLResponse)
    def analyze_get(request: Request, source: str = ""):
        if not source.strip():
            return _page("/", _analyze_form(live), live)
        if not live:
            report = _demo_cached(source)
            return report.to_html() if report is not None else _page("/", _analyze_form(live, _demo_msg()), live)
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
        tab = "/"  # Q&A now lives inside 分析, so every result keeps the 分析 tab active
        form = _ask_form if kind == "ask" else _analyze_form
        if job is None:
            # job lost (server restarted / expired) -> give back a usable retry form
            return _page(tab, form(live, _t("The result expired or the service just updated — please resubmit.", "结果已过期或服务刚更新过，请重新提交。")), live)
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
        return _page("/", log + _ask_form(live), live)

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
            return _page("/", _ask_form(live, _t("Q&A needs a model, so it's off in demo mode. Start with --live.", "演示模式不支持问答（需调用模型）。请以 --live 启动。")), live)

        sess = _session_for(pm_sid)
        prior = _chat_log_html(sess["log"]) if sess and sess.get("log") else ""
        if not source.strip() or not question.strip():
            return _page("/", prior + _ask_form(live, _t("Please fill in both the paper and the question.", "请填写论文和问题。"), source=source), live)

        blocked = gate(request)
        if blocked:
            return _page("/", prior + _ask_form(live, blocked, source=source), live)

        # Building the chat (download + parse + index) and answering can take tens of
        # seconds on a fresh paper, which would blow Cloudflare's 100s origin timeout
        # (524). Run it in a background job and return a polling page at once, like the
        # other tabs. The result/error is rendered by /result (kind="ask").
        sid = pm_sid or secrets.token_hex(8)
        ip = _client_ip(request)
        paper, q, md, mdl = source.strip(), question.strip(), mode, (model or None)

        def work(jid):
            from papermind.qa.chat import PaperChat

            sess2 = _session_for(sid)
            if sess2 is None or sess2.get("paper") != paper:
                _set_job(jid, step="下载并解析论文、建立索引…")
                sess2 = {"paper": paper, "chat": PaperChat(paper, model=mdl, mode=md), "log": []}
            sess2["chat"].mode = md  # honor a mode change between turns
            _set_job(jid, step="检索原文并生成回答…")
            answer = sess2["chat"].ask(q)
            sess2["log"].append((q, answer))
            _store_session(sid, sess2)
            return _page("/", _chat_log_html(sess2["log"]) + _ask_form(live, source=paper), live)

        jid = _start_job(work, kind="ask", on_fail=lambda: limiter.release(ip))
        return _with_session(HTMLResponse(_job_page(jid, live, "/")), sid)

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
            f"<table><thead><tr><th>arXiv</th><th>{_t('Year', '年份')}</th><th>{_t('Title', '标题（原文）')}</th></tr></thead>"
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
        raise PaperMindError("Web demo 需要 uvicorn。安装：pip install 'papermind-ai[web]'") from exc
    app = create_app(
        live=live, rate_per_ip=rate_per_ip, rate_global=rate_global,
        with_figures=with_figures, svg_figures=svg_figures, no_cache=no_cache,
    )
    uvicorn.run(app, host=host, port=port)


# --------------------------------------------------------------------------- #
# Demo (cached-only) lookups + small renderers
# --------------------------------------------------------------------------- #
def _demo_msg() -> str:
    return _t(
        "Demo mode shows only already-cached papers. Analyze a new one via the CLI, or start the server with --live.",
        "演示模式仅展示已缓存的论文。分析新论文请用 CLI，或以 --live 启动服务。",
    )


def _demo_cached(source: str):
    """A cached report for ``source`` WITHOUT any network/download (demo mode)."""
    from papermind.cache import latest_report_path, load_cached_report
    from papermind.parser.arxiv import cache_dir_for

    cache_dir = cache_dir_for(source)
    if cache_dir is None:
        return None
    path = latest_report_path(cache_dir)
    return load_cached_report(path) if path else None


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
        import hashlib

        from papermind.config import load_config
        from papermind.net import MAX_DOWNLOAD_BYTES

        data = file.file.read(MAX_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise SourceError(f"上传的 PDF 过大（>{MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB）。")
        if data:
            if not data.startswith(b"%PDF"):
                raise SourceError("上传的文件不是 PDF。")
            # Persist into the cache keyed by content hash (not a random /tmp file that
            # would leak one PDF per upload): identical re-uploads dedupe, and the bytes
            # live under normal cache management instead of accumulating in the OS tmp dir.
            dest = load_config().paper_cache(f"upload-{hashlib.sha1(data).hexdigest()[:12]}") / "paper.pdf"
            if not dest.exists() or dest.stat().st_size != len(data):
                dest.write_bytes(data)
            return str(dest)
    return (source or "").strip() or None


# --------------------------------------------------------------------------- #
# Design system — modern light product (system sans, soft cool canvas, one blue
# accent). Light/dark via CSS variables; reduced-motion safe.
# --------------------------------------------------------------------------- #
_FAVICON = (
    "<link rel='icon' href=\"data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='7' fill='%232563eb'/>"
    "<g stroke='%23fff' stroke-width='2.4' stroke-linecap='round'>"
    "<path d='M10 11h12'/><path d='M10 16h12'/><path d='M10 21h7'/></g></svg>\">"
)

_CSS = """
:root{
  --bg:#f5f7fa;--surface:#ffffff;--ink:#0f172a;--soft:#475569;--faint:#94a3b8;
  --line:#e8edf3;--line-strong:#d4dbe4;
  --accent:#2563eb;--accent-press:#1d4ed8;--accent-soft:#eff5ff;--accent-ring:rgba(37,99,235,.18);
  --warn:#c2620e;
  --green:#15803d;--amber:#b45309;--red:#dc2626;
  --r:12px;--rs:9px;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Microsoft YaHei","Source Han Sans SC",sans-serif;
  --mono:"SF Mono","Cascadia Code","JetBrains Mono",ui-monospace,Consolas,monospace;
  --shadow:0 1px 2px rgba(15,23,42,.04),0 6px 18px -8px rgba(15,23,42,.10);
  --shadow-lift:0 14px 32px -14px rgba(15,23,42,.22);
  --bar:rgba(255,255,255,.85);
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0b0d12;--surface:#13161d;--ink:#e8ecf3;--soft:#9aa6b7;--faint:#606b7c;
  --line:#212733;--line-strong:#2f3744;
  --accent:#5b8cff;--accent-press:#7aa2ff;--accent-soft:#16203a;--accent-ring:rgba(91,140,255,.28);
  --warn:#e0934a;
  --green:#34d399;--amber:#e0b252;--red:#f0726c;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 6px 18px -8px rgba(0,0,0,.6);
  --shadow-lift:0 16px 36px -14px rgba(0,0,0,.65);--bar:rgba(11,13,18,.85);
}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.65 var(--sans);overflow-x:hidden;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
a{color:var(--accent);text-decoration:none}
.pm-wrap{max-width:1060px;margin:0 auto;padding:0 24px 96px;overflow-x:clip}
main{min-width:0}

/* sticky top bar: brand · nav · GitHub · language */
.pm-bar-top{position:sticky;top:0;z-index:30;border-bottom:1px solid var(--line);
  background:var(--bar);backdrop-filter:saturate(1.6) blur(10px);-webkit-backdrop-filter:saturate(1.6) blur(10px)}
.pm-bar-in{max-width:1060px;margin:0 auto;padding:0 24px;height:56px;display:flex;align-items:center;gap:18px}
.pm-brand{display:flex;align-items:center;gap:9px;color:var(--ink);font-weight:700;font-size:1.04rem;letter-spacing:-.015em}
.pm-brand:hover{color:var(--ink)}
.pm-brand .mk{width:19px;height:19px;border-radius:6px;display:inline-block;
  background:linear-gradient(135deg,var(--accent),var(--accent-press));box-shadow:0 1px 3px var(--accent-ring)}
.pm-nav{display:flex;flex-wrap:wrap;gap:3px;flex:1;min-width:0}
.pm-nav a{color:var(--soft);font-weight:500;font-size:.9rem;padding:7px 11px;border-radius:7px;
  transition:color .15s,background .15s;white-space:nowrap}
.pm-nav a:hover{color:var(--ink);background:var(--accent-soft)}
.pm-nav a.on{color:var(--accent);background:var(--accent-soft);font-weight:600}
.pm-ghs{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line-strong);color:var(--ink);
  border-radius:8px;padding:6px 11px;font-size:.84rem;font-weight:600;white-space:nowrap;transition:border-color .15s,color .15s}
.pm-ghs:hover{border-color:var(--accent);color:var(--accent)}

/* home: a focused analyze tool, grounded in one card on a soft canvas */
.pm-tool{max-width:640px;margin:0 auto;padding:40px 0 56px}
main>.pm-tool:only-child{min-height:calc(100dvh - 150px);display:flex;flex-direction:column;justify-content:center;padding:28px 0 48px}
.tool-card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:30px 32px;box-shadow:var(--shadow)}
.tool-h{font-size:1.55rem;font-weight:700;letter-spacing:-.022em;margin:0 0 .3em}
.tool-sub{color:var(--soft);font-size:.96rem;line-height:1.6;margin:0 0 22px}
.tool-note{font-size:.85rem;color:var(--faint);margin:0 0 14px}
.tool-form{margin:0}
.tool-in{display:flex;gap:10px}
.tool-in input{flex:1;min-width:0;font-size:1.05rem;padding:15px 16px}
.tool-in button{margin:0;white-space:nowrap;padding:15px 30px;font-size:1.02rem}
.tool-opts{display:flex;flex-wrap:wrap;align-items:center;gap:9px 18px;margin-top:12px;font-size:.85rem;color:var(--soft)}
.tool-opts .upl input[type=file]{width:auto;font-size:.82rem}
.tool-opts .chk{display:inline-flex;align-items:center;gap:7px;cursor:pointer;white-space:nowrap}
.tool-opts .chk input{width:auto;margin:0}
.ask-inline{margin-top:26px;padding-top:22px;border-top:1px dashed var(--line-strong)}
.ask-h{font-weight:600;font-size:.96rem;color:var(--ink);margin:0 0 11px}
.ask-h .hint{font-weight:400}
.chk2{display:inline-flex;align-items:center;gap:9px;margin-top:12px;font-size:.85rem;font-weight:500;color:var(--soft)}
.chk2 select{width:auto;padding:7px 11px;font-size:.85rem}
button.btn-2{background:var(--surface);color:var(--accent);border:1px solid var(--accent)}
button.btn-2:hover{background:var(--accent-soft);color:var(--accent-press);box-shadow:none}
.tool-demo{margin-top:18px;font-size:.88rem}
@media(max-width:640px){
  .pm-bar-in{height:auto;flex-wrap:wrap;padding:9px 16px;gap:10px 12px}
  .pm-brand{flex:1}
  .pm-nav{order:3;flex-basis:100%;flex-wrap:nowrap;overflow-x:auto;gap:2px;scrollbar-width:none}
  .pm-nav::-webkit-scrollbar{display:none}
  .pm-tool,main>.pm-tool:only-child{min-height:0;padding:28px 0 40px}
  .tool-h{font-size:1.7rem}
  .tool-in{flex-direction:column}
  .tool-in button{padding:13px}
}

/* panels & type */
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
  padding:26px 28px;margin:18px 0;box-shadow:var(--shadow)}
h1,h2,h3{font-weight:700;letter-spacing:-.02em;line-height:1.25;color:var(--ink)}
.panel h2{margin:0 0 .3em;font-size:1.22rem;letter-spacing:-.015em}.panel h3{margin:1.3em 0 .4em;font-size:1.04rem}
.panel p{margin:.2em 0 .9em}.lead{color:var(--soft);margin:0 0 18px;font-size:.95rem}
.panel ul{padding-left:1.15em;margin:.4em 0}.panel li{margin:.35em 0}
label{display:block;font-weight:600;font-size:.88rem;margin:16px 0 7px}
label:first-of-type{margin-top:4px}
.hint{color:var(--faint);font-weight:400}
code{font-family:var(--mono);font-size:.85em;background:var(--accent-soft);color:var(--accent-press);padding:1px 6px;border-radius:5px}

/* forms */
input,textarea,select{width:100%;font:inherit;color:var(--ink);background:var(--bg);
  border:1px solid var(--line-strong);border-radius:var(--rs);padding:11px 13px}
input:focus,textarea:focus,select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-ring)}
input::placeholder,textarea::placeholder{color:var(--faint)}
textarea{min-height:96px;resize:vertical}
input[type=file]{padding:9px 12px;font-size:.9rem;color:var(--soft);background:var(--surface)}
input[type=file]::file-selector-button{font:600 .85rem var(--sans);color:var(--accent);background:var(--accent-soft);
  border:0;border-radius:6px;padding:6px 12px;margin-right:12px;cursor:pointer}
button{font:600 .95rem var(--sans);background:var(--accent);color:#fff;border:0;border-radius:var(--rs);
  padding:11px 24px;cursor:pointer;margin-top:18px;transition:background .15s,transform .08s,box-shadow .15s;
  box-shadow:0 1px 2px var(--accent-ring)}
button:hover{background:var(--accent-press);box-shadow:0 4px 14px var(--accent-ring)}
button:active{transform:translateY(1px)}
@media(prefers-color-scheme:dark){button{color:#0c0d10}}

/* example chips (kept .ex a hook) */
.ex{margin:18px 0 0;color:var(--faint);font-size:.86rem}
.ex a{display:inline-block;margin:4px 0 0 6px;padding:4px 11px;border:1px solid var(--line-strong);border-radius:7px;
  background:var(--surface);transition:background .15s,border-color .15s}
.ex a:hover{background:var(--accent-soft);border-color:var(--accent)}

/* Q&A chat + evidence segments */
.chat .turn{padding:20px 0;border-top:1px solid var(--line)}
.chat .turn:first-child{padding-top:0;border-top:0}
.q{font-weight:600;margin:0 0 12px}
.q::before{content:'Q ';color:var(--accent);font-weight:700}
.err{color:var(--red);margin:0 0 14px;font-size:.92rem}
.seg{border-left:3px solid var(--line);padding:2px 0 2px 16px;margin:16px 0}
.seg .tag{display:block;font:600 .78rem var(--sans);letter-spacing:.02em;margin-bottom:5px}
.seg small{display:block;margin-top:6px;color:var(--soft)}
.seg.fact{border-left-color:var(--green)}.seg.fact .tag{color:var(--green)}
.seg.inf{border-left-color:var(--amber)}.seg.inf .tag{color:var(--amber)}
.seg.oos{border-left-color:var(--red)}.seg.oos .tag{color:var(--red)}

/* tables */
table{border-collapse:collapse;width:100%;font-size:.93rem}
th{text-align:left;font:600 .72rem var(--sans);letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);padding:0 12px 10px}
td{padding:12px;border-top:1px solid var(--line);vertical-align:top}
.aid{font-family:var(--mono);font-size:.88rem;white-space:nowrap}
.ttl a{color:var(--ink)}.ttl a:hover{color:var(--accent)}

/* code blocks */
pre{font:.88rem/1.6 var(--mono);background:#0f1117;color:#e7e9ef;padding:16px 18px;
  border-radius:var(--rs);overflow-x:auto;white-space:pre-wrap}
.copy-btn{font:600 .84rem var(--sans);background:var(--surface);color:var(--accent);
  border:1px solid var(--line-strong);border-radius:var(--rs);padding:6px 13px;margin-bottom:12px}
.copy-btn:hover{background:var(--accent-soft);border-color:var(--accent)}

/* footer */
.pm-foot{display:flex;flex-wrap:wrap;align-items:center;color:var(--faint);font-size:.84rem;
  border-top:1px solid var(--line);margin-top:48px;padding-top:20px}
.pm-foot>*+*{margin-left:16px;padding-left:16px;border-left:1px solid var(--line)}
.pm-foot .m{font-weight:600;color:var(--soft)}
.pm-foot a{color:var(--soft)}.pm-foot a:hover{color:var(--accent)}

/* busy overlay + progress bar (JS hooks) */
.pm-busy{display:none;position:fixed;inset:0;z-index:50;background:rgba(12,13,16,.5);
  align-items:center;justify-content:center;backdrop-filter:blur(2px)}
.pm-busy.on{display:flex}
.pm-busy-card{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:26px 32px;min-width:300px;text-align:center;box-shadow:0 24px 60px rgba(0,0,0,.28)}
.pm-bar{height:8px;border-radius:6px;background:var(--accent-soft);overflow:hidden;margin-bottom:16px}
.pm-bar span{display:block;height:100%;width:5%;border-radius:6px;background:var(--accent);transition:width .5s ease}
.pm-busy-title{margin:0;font-weight:600}.pm-busy-step{margin:8px 0 0;color:var(--accent)}
.pm-busy-time{margin:6px 0 0;color:var(--faint);font-size:.85rem}
@media(max-width:600px){.pm-wrap{padding:0 16px 72px}.panel{padding:22px}}
.js .reveal{opacity:0;transform:translateY(16px)}
.js .reveal.in{opacity:1;transform:none;transition:opacity .6s ease,transform .6s cubic-bezier(.2,.7,.2,1)}
@media(prefers-reduced-motion:reduce){.js .reveal{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media(prefers-reduced-motion:no-preference){
  @keyframes pm-rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  .pm-tool{animation:pm-rise .5s cubic-bezier(.2,.7,.2,1) both}
  main>.panel{animation:pm-rise .5s cubic-bezier(.2,.7,.2,1) .07s both}
}
"""


# On submit of a heavy op, show the busy overlay immediately; the real,
# step-driven progress bar then takes over on the job page (see _poll_js).
_BUSY_JS = """
var PM_HEAVY={'/analyze':1,'/ask':1};
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
(function(){
  var els=document.querySelectorAll('.reveal');
  if(!els.length) return;
  if(!('IntersectionObserver' in window)){els.forEach(function(e){e.classList.add('in');});return;}
  var io=new IntersectionObserver(function(es){es.forEach(function(en){
    if(en.isIntersecting){en.target.classList.add('in'); io.unobserve(en.target);}
  });},{rootMargin:'0px 0px -8% 0px'});
  els.forEach(function(e){io.observe(e);});
})();
"""


def _page(active: str, body: str, live: bool) -> str:
    nav = "".join(
        f"<a href='{href}' class='{'on' if href == active else ''}'>{_t(en, zh)}</a>"
        for href, en, zh in _TABS
    )
    mode = _t("Live · uses the server key (billed)", "实时分析 · 用本机 key（有成本）") if live \
        else _t("Demo · cached papers only", "演示模式 · 仅已缓存论文")
    return (
        f"<!DOCTYPE html><html lang='{_lang()}'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"{_FAVICON}"
        f"<title>PaperMind</title><style>{_CSS}</style>"
        "<script>document.documentElement.classList.add('js')</script></head><body>"
        "<header class='pm-bar-top'><div class='pm-bar-in'>"
        "<a class='pm-brand' href='/'><span class='mk'></span>PaperMind</a>"
        f"<nav class='pm-nav'>{nav}</nav>"
        "<a class='pm-ghs' href='https://github.com/Wenhao-Hua/papermind' target='_blank' rel='noopener'>★ GitHub</a>"
        "</div></header>"
        f"<div class='pm-wrap'><main>{body}</main>"
        "<footer class='pm-foot'>"
        f"<span>{mode}</span><a href='/demo'>{_t('Offline demo', '离线示例')}</a></footer>"
        "<div id='pm-busy' class='pm-busy'><div class='pm-busy-card'>"
        "<div class='pm-bar'><span id='pm-fill'></span></div>"
        f"<p class='pm-busy-title'>{_t('Analyzing…', '分析中…')}</p>"
        f"<p class='pm-busy-step' id='pm-step'>{_t('Starting', '开始')}</p>"
        f"<p class='pm-busy-time'><b id='pm-pct'>0</b>% · {_t('elapsed', '已用')} <b id='pm-sec'>0</b>{_t('s', ' 秒')}</p></div></div>"
        f"<script>{_BUSY_JS.replace('提交中…', _t('Submitting…', '提交中…'))}</script>"
        "</body></html>"
    )


def _err(error: str) -> str:
    return f"<p class='err'>{_e(error)}</p>" if error else ""


_EXAMPLES = [("https://arxiv.org/abs/1706.03762", "Transformer"), ("https://arxiv.org/abs/2307.08691", "FlashAttention-2"), ("https://arxiv.org/abs/1810.04805", "BERT")]


def _examples_row() -> str:
    chips = "".join(f"<a href='#' data-id='{_e(i)}'>{_e(name)}</a>" for i, name in _EXAMPLES)
    return f"<p class='ex'>{_t('No target? Try', '没链接？点一篇名作先试试')} {chips}</p>"


def _tool_page(title: str, sub: str, inner: str, note: str = "", error: str = "") -> str:
    """The one focused, viewport-centered layout the tool pages share
    (分析 home, 搜索, and the Q&A follow-up) — no marketing chrome."""
    return (
        "<section class='pm-tool'><div class='tool-card'>"
        f"<h1 class='tool-h'>{title}</h1>"
        f"<p class='tool-sub'>{sub}</p>"
        f"{note}{_err(error)}{inner}"
        "</div></section>"
    )


def _analyze_form(live: bool, error: str = "") -> str:
    note = "" if live else "<p class='tool-note'>演示模式：仅展示已缓存论文；分析新论文请用 CLI 或以 <code>--live</code> 启动。</p>"
    inner = (
        "<form class='tool-form' method='post' action='/analyze' enctype='multipart/form-data'>"
        "<label>论文 <span class='hint'>arXiv 链接 / DOI / 标题 / PDF</span></label>"
        "<div class='tool-in'><input name='source' placeholder='arXiv 链接 / DOI / 论文标题 / PDF…' autofocus>"
        "<button formaction='/analyze'>开始读</button></div>"
        "<div class='tool-opts'><span class='upl'>或上传 PDF <input type='file' name='file' accept='application/pdf'></span>"
        "<label class='chk'><input type='checkbox' name='refresh' value='1'>忽略缓存重分析</label></div>"
        # 问答就在分析里：复用上面那篇论文，不必再选一次
        "<div class='ask-inline'>"
        "<div class='ask-h'>或，直接问上面这篇论文 <span class='hint'>不必再选，就用上面填的那篇</span></div>"
        "<div class='tool-in'><input name='question' id='q-inline' placeholder='例：为什么要除以 √d_k？'>"
        "<button class='btn-2' formaction='/ask'>问 AI</button></div>"
        "<label class='chk2'>回答风格"
        "<select name='mode'><option value='balanced'>均衡</option>"
        "<option value='strict'>严格 · 更保守</option><option value='explore'>发散 · 更开放</option></select></label>"
        "</div>"
        "</form>"
        f"{_examples_row()}"
        "<p class='tool-demo'><a href='/demo'>没装也想看？看一份完整示例报告 →</a></p>"
        # Enter in the question box should ask (the form's default submit is 分析).
        "<script>(function(){var q=document.getElementById('q-inline');if(!q)return;"
        "q.addEventListener('keydown',function(e){if(e.key==='Enter'){e.preventDefault();"
        "var b=document.querySelector(\"button[formaction='/ask']\");if(b)b.click();}});})();</script>"
    )
    return _tool_page(
        "分析一篇论文",
        "粘 arXiv 链接 / DOI / 标题，或直接传 PDF —— 得到四模块解读和整篇方法的框架图。也可以就这篇论文直接问 AI。",
        inner, note=note, error=error,
    )


def _ask_form(live: bool, error: str = "", source: str = "") -> str:
    af_src, af_q = ("", " autofocus") if source else (" autofocus", "")
    inner = (
        "<form class='tool-form' method='post' action='/ask'>"
        "<label>论文 <span class='hint'>链接 / DOI / 标题</span></label>"
        f"<input name='source' value='{_e(source)}' placeholder='arXiv 链接 / DOI / 论文标题'{af_src}>"
        "<label>你的问题</label>"
        f"<input name='question' placeholder='例：为什么要除以 √d_k？'{af_q}>"
        "<label>回答风格 <span class='hint'>均衡 · 严格更保守 · 发散更开放</span></label>"
        "<select name='mode'><option value='balanced'>均衡</option>"
        "<option value='strict'>严格</option><option value='explore'>发散</option></select>"
        "<button>提问</button></form>"
    )
    return _tool_page("继续问这篇" if source else "问答",
                      "基于原文回答——每句都标注是论文事实 / 推理 / 超纲，并附带页码的原文出处。",
                      inner, error=error)


def _search_form(error: str = "") -> str:
    inner = (
        "<form class='tool-form' method='post' action='/search'>"
        "<div class='tool-in'><input name='query' placeholder='关键词，例：flash attention' autofocus>"
        "<button>搜索</button></div></form>"
    )
    return _tool_page("搜索 arXiv", "关键词检索，不调用模型。点结果里的论文即可分析 / 问答。", inner, error=error)


def _answer_segments_html(answer) -> str:
    kind = {
        "fact": (_t("Paper fact", "论文事实"), "fact"),
        "inference": (_t("Inference from the paper", "基于论文的推理"), "inf"),
        "out_of_scope": (_t("Out of scope", "超出论文范围"), "oos"),
    }
    segs = []
    for s in answer.segments:
        label, cls = kind.get(s.kind, (s.kind, ""))
        tag = label
        if s.kind == "inference" and s.confidence:
            tag += f" · {_t('confidence', '置信度')} {_e(s.confidence)}"
        reason = f"<small>{_t('Reasoning', '推理依据')}：{_e(s.reasoning)}</small>" if s.reasoning else ""
        segs.append(f"<div class='seg {cls}'><span class='tag'>{tag}</span><div>{_e(s.text)}</div>{reason}</div>")
    ev = ""
    if answer.evidence:
        rows = ""
        for e in answer.evidence:
            loc = (e.section or "") + (f" p.{e.page}" if e.page else "")
            mark = "" if e.verified else f"<span class='hint'>{_t('unverified', '未核实')} · </span>"
            rows += f"<tr><td class='aid'>{_e(loc) or '—'}</td><td>{mark}{_e(e.text)}</td></tr>"
        ev = f"<h3>{_t('Sources', '原文依据')}</h3><table><tbody>{rows}</tbody></table>"
    return f"{''.join(segs)}{ev}"


def _chat_log_html(log) -> str:
    turns = ""
    for question, answer in log:
        turns += f"<div class='turn'><p class='q'>{_e(question)}</p>{_answer_segments_html(answer)}</div>"
    return f"<section class='panel chat'>{turns}</section>" if turns else ""


def _e(text) -> str:
    return _html.escape(str(text)) if text is not None else ""
