"""Minimal web demo for PaperMind (``papermind serve``).

Paste an arXiv id/URL, pick a model, and get the full HTML report rendered in
the browser. By default it runs in **demo mode**: it only serves reports that
are already cached (so exposing it publicly — e.g. via a Cloudflare tunnel —
cannot burn your API key). Pass ``live=True`` to allow live analysis using the
server's configured keys.
"""

from __future__ import annotations

import html as _html
from typing import Optional

from papermind.errors import PaperMindError

# Curated model choices for the dropdown (anything litellm routes also works via CLI).
WEB_MODELS = [
    ("", "默认（按服务器配置）"),
    ("gpt-4o-mini", "OpenAI · GPT-4o mini（便宜）"),
    ("gpt-4o", "OpenAI · GPT-4o"),
    ("claude-3-5-sonnet-20241022", "Anthropic · Claude 3.5 Sonnet"),
    ("deepseek/deepseek-chat", "DeepSeek · Chat (V3)"),
    ("deepseek/deepseek-reasoner", "DeepSeek · Reasoner (R1)"),
    ("gemini/gemini-1.5-flash", "Google · Gemini 1.5 Flash"),
    ("ollama/llama3.1", "本地 · Ollama llama3.1"),
]


def create_app(live: bool = False):
    try:
        from fastapi import FastAPI, Form
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("Web demo 需要 FastAPI。安装：pip install 'paperis[web]'") from exc

    app = FastAPI(title="PaperMind", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "live": live}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _shell("PaperMind", _landing(live))

    @app.get("/demo", response_class=HTMLResponse)
    def demo():
        from papermind.demo import build_demo_report

        return build_demo_report().to_html()

    @app.post("/analyze", response_class=HTMLResponse)
    def analyze_route(source: str = Form(...), model: str = Form("")):
        from papermind.analyze import analyze as run_analyze
        from papermind.cache import latest_report_path, load_cached_report
        from papermind.config import load_config
        from papermind.parser.arxiv import resolve

        source = (source or "").strip()
        if not source:
            return _shell("PaperMind", _landing(live, error="请输入 arXiv id / URL 或论文。"))

        config = load_config()
        try:
            resolved = resolve(source, config)
        except PaperMindError as exc:
            return _shell("PaperMind", _landing(live, error=str(exc)))

        report_path = latest_report_path(resolved.cache_dir)
        report = load_cached_report(report_path) if report_path else None

        if report is None:
            if not live:
                return _shell(
                    "PaperMind",
                    _landing(
                        live,
                        error="演示模式仅展示已缓存的论文。新论文请用 CLI 分析，或以 --live 启动服务。",
                    ),
                )
            try:
                report = run_analyze(source, model=(model or None), config=config)
            except PaperMindError as exc:
                return _shell("PaperMind", _landing(live, error=str(exc)))

        return report.to_html()

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, live: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("Web demo 需要 uvicorn。安装：pip install 'paperis[web]'") from exc
    uvicorn.run(create_app(live=live), host=host, port=port)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_SHELL_CSS = (
    "body{font:16px/1.6 -apple-system,Segoe UI,Roboto,'PingFang SC','Microsoft YaHei',sans-serif;"
    "max-width:720px;margin:8vh auto;padding:0 20px;color:#1a1a2e;background:#fbfbfd}"
    "@media(prefers-color-scheme:dark){body{background:#0f0f17;color:#e6e6f0}"
    ".card{background:#16161f!important;border-color:#2a2a3c!important}input,select{background:#16161f;color:#e6e6f0;border-color:#2a2a3c}}"
    "h1{font-size:2rem;margin:0 0 .2em}.sub{color:#6b7280;margin-bottom:1.5em}"
    ".card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:22px}"
    "input,select{width:100%;font:inherit;padding:10px 12px;border:1px solid #e5e7eb;border-radius:8px;margin:6px 0 14px}"
    "button{font:inherit;font-weight:600;background:#5b5bd6;color:#fff;border:0;border-radius:8px;padding:11px 18px;cursor:pointer;width:100%}"
    "button:hover{background:#4a4ac0}.err{color:#e5484d;margin:0 0 12px}"
    ".muted{color:#6b7280;font-size:.85rem;margin-top:14px}a{color:#5b5bd6}"
)


def _shell(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_SHELL_CSS}</style></head><body>{body}</body></html>"
    )


def _landing(live: bool, error: str = "") -> str:
    options = "".join(f"<option value='{_e(v)}'>{_e(label)}</option>" for v, label in WEB_MODELS)
    err = f"<p class='err'>{_e(error)}</p>" if error else ""
    mode = "Live（会调用模型，有成本）" if live else "演示模式（仅展示已缓存论文）"
    return (
        "<h1>📄 PaperMind</h1>"
        "<p class='sub'>读懂一篇 arXiv 论文：结构化分析 · 带原文依据的问答 · 复现指南</p>"
        "<div class='card'>"
        f"{err}"
        "<form method='post' action='/analyze'>"
        "<label>论文（arXiv id / URL）</label>"
        "<input name='source' placeholder='2307.08691 或 https://arxiv.org/abs/2307.08691' autofocus>"
        "<label>模型</label>"
        f"<select name='model'>{options}</select>"
        "<button type='submit'>分析</button>"
        "</form>"
        f"<p class='muted'>当前：{_e(mode)} · <a href='/demo'>看一个离线示例报告 →</a></p>"
        "</div>"
        "<p class='muted'>由 <a href='https://github.com/Wenhao-Hua/papermind'>PaperMind</a> 驱动。</p>"
    )


def _e(text) -> str:
    return _html.escape(str(text)) if text is not None else ""
