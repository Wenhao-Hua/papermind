"""Typer CLI entry point for PaperMind.

Heavy work is imported lazily inside each command so that ``papermind --help``
and ``papermind config`` work even before optional dependencies are installed.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.markup import escape

from papermind import __version__
from papermind.errors import PaperMindError


def _force_utf8() -> None:
    """Avoid UnicodeEncodeError on legacy (GBK/cp1252) Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8()
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="papermind",
    help="Read any paper from its URL: structured analysis, grounded Q&A, and reproduction tutoring.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Manage API keys, default model, and cache settings.", no_args_is_help=True)
app.add_typer(config_app, name="config")
cache_app = typer.Typer(help="查看与清理缓存（PDF / 向量索引 / 分析报告）。", no_args_is_help=True)
app.add_typer(cache_app, name="cache")

ALL_MODULES = ["contributions", "technical", "connections", "reproduction"]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"papermind {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit.", callback=_version_callback, is_eager=True
    ),
) -> None:
    """PaperMind command-line interface."""


def _fail(message: str) -> "typer.Exit":
    """Print a clean error and return an Exit(1) to raise."""
    err_console.print(f"[bold red]Error:[/bold red] {escape(message)}")
    return typer.Exit(code=1)


def _apply_local(local: bool, model: Optional[str], env=None) -> Optional[str]:
    """With --local: force the whole stack (LLM + embeddings) to local Ollama.

    Implemented via env overrides that every command's ``load_config()`` reads,
    so no signature plumbing is needed. Returns the model to pass downstream
    (None lets the env-overridden config default drive it).
    """
    if not local:
        return model
    import os

    from papermind.config import load_config
    from papermind.llm.base import OLLAMA_DEFAULT_MODEL

    env = os.environ if env is None else env
    env["PAPERMIND_MODEL"] = model or load_config().local_model or OLLAMA_DEFAULT_MODEL
    env["PAPERMIND_EMBEDDING_PROVIDER"] = "local"
    env.pop("PAPERMIND_EMBEDDING_MODEL", None)
    return None


def _parse_only(only: Optional[str]) -> List[str]:
    if not only:
        return ALL_MODULES
    requested = [m.strip() for m in only.split(",") if m.strip()]
    invalid = [m for m in requested if m not in ALL_MODULES]
    if invalid:
        raise _fail(f"Unknown module(s): {', '.join(invalid)}. Choose from: {', '.join(ALL_MODULES)}")
    return requested


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
@app.command()
def demo(
    speed: float = typer.Option(1.0, "--speed", help="Playback speed (higher = faster; 0 = instant)."),
) -> None:
    """Play a built-in, offline demo (no API key) — quick look + ideal for recording a GIF."""
    from papermind.demo import play

    play(console, speed=speed)


@app.command()
def ui(
    host: str = typer.Option("localhost", "--host", help="Bind host."),
    port: int = typer.Option(8501, "--port", "-p", help="Bind port."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open a browser window."),
) -> None:
    """Launch the local web app — full features, uses your configured keys."""
    import threading
    import webbrowser

    from papermind.web import serve as run_serve

    url = f"http://{host}:{port}"
    console.print(f"[bold cyan]PaperMind[/bold cyan]  →  {url}  ·  Ctrl-C 停止")
    if not no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        # Local: unlimited, figures on, SVG teaching figures (you run it on your own machine/key).
        run_serve(host=host, port=port, live=True, rate_per_ip=0, rate_global=0,
                  with_figures=True, svg_figures=True)
    except PaperMindError as exc:
        raise _fail(str(exc))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port."),
    live: bool = typer.Option(
        False, "--live", help="Allow live analysis using the server's keys (has cost). Default: cached-only."
    ),
    rate_per_ip: int = typer.Option(8, "--rate-per-ip", help="Max live requests per IP per day (0 = unlimited)."),
    rate_global: int = typer.Option(300, "--rate-global", help="Max live requests total per day (0 = unlimited)."),
    figures: bool = typer.Option(True, "--figures/--no-figures", help="Generate figures in reports (off = fastest/cheapest)."),
    svg_figures: bool = typer.Option(True, "--svg-figures/--mermaid", help="Architecture-faithful teaching SVGs (default); --mermaid for legacy Mermaid flowcharts."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Always re-analyze (never reuse cached reports). Costs more every time."),
) -> None:
    """Run the web service. Default is demo mode (cached reports only, key-safe).

    With --live, model-calling requests are rate-limited per IP and globally so a
    public deployment can't drain your API key (tune with --rate-per-ip / --rate-global).
    Figures default to architecture-faithful teaching SVGs (--mermaid for the legacy
    flowcharts, --no-figures to skip). --no-cache forces a fresh analysis every request.
    """
    from papermind.web import serve as run_serve

    mode = "[yellow]LIVE — 实时分析会用本机 key（有成本）[/yellow]" if live else "演示模式（仅展示已缓存论文，安全）"
    limits = f"  ·  限流 每IP {rate_per_ip}/天 · 全局 {rate_global}/天" if live else ""
    nocache = "  ·  [yellow]不缓存（每次都重跑）[/yellow]" if no_cache else ""
    console.print(f"[bold cyan]PaperMind web[/bold cyan]  →  http://{host}:{port}\n[dim]{mode}{limits}{nocache}  ·  Ctrl-C 停止[/dim]")
    try:
        run_serve(host=host, port=port, live=live, rate_per_ip=rate_per_ip, rate_global=rate_global,
                  with_figures=figures, svg_figures=svg_figures, no_cache=no_cache)
    except PaperMindError as exc:
        raise _fail(str(exc))


@app.command()
def analyze(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM (litellm format). Default: the configured model."),
    fmt: str = typer.Option("md", "--format", "-f", help="Output format: md | json | html | all."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path/prefix. Prints to terminal if omitted."),
    only: Optional[str] = typer.Option(None, "--only", help="Comma-separated modules to run."),
    quick: bool = typer.Option(False, "--quick", help="Fast/cheap subset: contributions+technical, no figures."),
    no_figures: bool = typer.Option(False, "--no-figures", help="Skip figure extraction/generation."),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cached result and re-analyze."),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama (LLM + embeddings)."),
    image_figures: bool = typer.Option(False, "--image-figures", help="Use an image model for AI figures instead of the default teaching SVGs (needs `config set image-model`)."),
    svg_figures: bool = typer.Option(True, "--svg-figures/--mermaid", help="Architecture-faithful teaching SVGs per technical point (default); --mermaid for legacy flowcharts."),
    open_report: bool = typer.Option(False, "--open", help="Open the HTML report in a browser when done."),
    estimate: bool = typer.Option(False, "--estimate", help="Estimate token cost and exit (no model calls)."),
) -> None:
    """Analyze a paper and produce the four-module report."""
    if fmt not in ("md", "json", "html", "all"):
        raise _fail("--format must be one of: md, json, html, all")
    if image_figures:
        from papermind.config import load_config

        if not load_config().image_model:
            console.print(
                "[yellow]提示：未配置图像模型，--image-figures 不生效，将使用默认教学 SVG。"
                "配置示例：papermind config set image-model gpt-image-1（需对应 key）。[/yellow]"
            )
    if quick:
        modules = ["contributions", "technical"]
        no_figures = True
    else:
        modules = _parse_only(only)
    model = _apply_local(local, model)

    if estimate:
        _estimate_cost(source, model, modules, not no_figures)
        return

    from papermind.analyze import analyze as run_analyze
    from papermind.output.terminal import render_report, render_usage

    try:
        report = run_analyze(
            source, model=model, modules=modules, with_figures=not no_figures, console=console,
            refresh=refresh, image_figures=image_figures, svg_figures=svg_figures,
        )
    except PaperMindError as exc:
        raise _fail(str(exc))

    if output:
        _write_outputs(report, fmt, output)
    else:
        render_report(report, console)
        if fmt in ("json", "all"):
            console.print(report.to_json())
    render_usage(report.usage, console)
    if open_report:
        _open_html_report(report, output)


def _write_outputs(report, fmt: str, output: str) -> None:

    base_path = _output_base(report, output)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    writers = {"md": report.to_markdown, "json": report.to_json, "html": report.to_html}
    written = []
    for ext, writer in writers.items():
        if fmt in (ext, "all"):
            path = f"{base_path}.{ext}"
            writer(path)
            written.append(path)
    console.print(f"[green]✓[/green] Wrote: {', '.join(written)}")


def _output_base(report, output: str):
    """Resolve the output path prefix (without extension).

    If ``output`` is an existing directory or ends with a path separator, write
    files *inside* it named by the paper; otherwise treat it as a file prefix.
    """
    from pathlib import Path

    out = Path(output)
    if output.endswith(("/", "\\")) or out.is_dir():
        name = report.paper.arxiv_id or _safe_name(report.paper.title)
        return out / name.replace("/", "_")
    for ext in (".md", ".json", ".html"):
        if output.lower().endswith(ext):
            return Path(output[: -len(ext)])
    return out


def _open_path(path) -> None:
    import webbrowser
    from pathlib import Path

    uri = Path(path).resolve().as_uri()
    webbrowser.open(uri)
    console.print(f"[green]✓[/green] 已在浏览器打开: {path}")


def _open_html_report(report, output: Optional[str]) -> None:
    import tempfile
    from pathlib import Path

    if output:
        base_path = _output_base(report, output)
        html_path = f"{base_path}.html"
        if not Path(html_path).exists():
            base_path.parent.mkdir(parents=True, exist_ok=True)
            report.to_html(html_path)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="papermind-", suffix=".html", delete=False)
        tmp.close()
        report.to_html(tmp.name)
        html_path = tmp.name
    _open_path(html_path)


def _estimate_cost(source: str, model: Optional[str], modules: List[str], with_figures: bool) -> None:
    from rich.table import Table

    from papermind.analyze import _build_context
    from papermind.config import load_config
    from papermind.llm import base, prompts
    from papermind.parser.arxiv import resolve
    from papermind.parser.pdf import parse_pdf

    cfg = load_config()
    eff_model = model or cfg.default_model
    try:
        with console.status("[cyan]估算中（解析论文，不调用模型）…[/cyan]"):
            resolved = resolve(source, cfg)
            parsed = parse_pdf(resolved.pdf_path, resolved.meta, resolved.cache_dir, extract_figures=False)
    except PaperMindError as exc:
        raise _fail(str(exc))

    context = _build_context(parsed)
    builders = {
        "contributions": (prompts.ANALYSIS_SYSTEM, prompts.contributions_user(context)),
        "technical": (prompts.ANALYSIS_SYSTEM, prompts.technical_user(context)),
        "connections": (prompts.ANALYSIS_SYSTEM, prompts.connections_user(context)),
        "reproduction": (prompts.ANALYSIS_SYSTEM, prompts.reproduction_user(context)),
    }
    litellm = base._import_litellm()
    est_completion_per_call = 700
    total_prompt = 0
    table = Table(title=f"成本预估 · {eff_model}", title_justify="left", header_style="bold cyan")
    table.add_column("模块")
    table.add_column("输入 tokens", justify="right")
    for m in modules:
        system, user = builders[m]
        n = base._count_tokens(litellm, eff_model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        total_prompt += n
        table.add_row(m, f"{n:,}")

    n_calls = len(modules) + (2 if with_figures and "technical" in modules else 0)
    est_completion = est_completion_per_call * n_calls
    table.add_row("[dim]估算输出[/dim]", f"[dim]~{est_completion:,}[/dim]")
    console.print(table)
    try:
        pc, cc = litellm.cost_per_token(model=eff_model, prompt_tokens=total_prompt, completion_tokens=est_completion)
        cost = float(pc) + float(cc)
    except Exception:  # noqa: BLE001
        cost = 0.0
    cost_str = f" · 估算成本 ~${cost:.4f}" if cost else "（该模型无定价信息）"
    console.print(f"约 {total_prompt + est_completion:,} tokens{cost_str}  [dim](仅供参考，实际取决于模型输出长度)[/dim]")


# --------------------------------------------------------------------------- #
# chat (interactive)
# --------------------------------------------------------------------------- #
@app.command()
def chat(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM (litellm format)."),
    mode: str = typer.Option("balanced", "--mode", help="Reasoning mode: strict | balanced | explore."),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable live streaming output."),
) -> None:
    """Interactive, grounded Q&A about the paper (multi-turn)."""
    _validate_mode(mode)
    model = _apply_local(local, model)
    from papermind.qa.chat import PaperChat
    from papermind.output.terminal import render_answer, render_usage

    try:
        chat_session = PaperChat(source, model=model, mode=mode, console=console)
    except PaperMindError as exc:
        raise _fail(str(exc))

    console.print(
        f"[bold cyan]PaperMind chat[/bold cyan] — {chat_session.report.paper.title}\n"
        f"[dim]mode={mode}  •  type 'exit' or Ctrl-D to quit, 'mode <name>' to switch[/dim]"
    )
    while True:
        try:
            question = console.input("\n[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", ":q"):
            console.print("[dim]bye[/dim]")
            break
        if question.lower().startswith("mode "):
            new_mode = question.split(" ", 1)[1].strip()
            try:
                _validate_mode(new_mode)
                chat_session.mode = new_mode
                console.print(f"[dim]mode set to {new_mode}[/dim]")
            except typer.Exit:
                console.print("[red]invalid mode[/red]")
            continue
        try:
            answer = _run_answer(lambda on_delta: chat_session.ask(question, on_delta=on_delta), not no_stream)
        except PaperMindError as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            continue
        render_answer(answer, console)
        render_usage(answer.usage, console, label="本轮")
    render_usage(chat_session.usage, console, label="本次会话合计")


def _run_answer(call, stream: bool):
    """Run an answer-producing call, optionally with a live streaming view."""
    from papermind.output.terminal import streaming_view

    if stream:
        with streaming_view(console) as on_delta:
            return call(on_delta)
    with console.status("[cyan]思考中…[/cyan]"):
        return call(None)


# --------------------------------------------------------------------------- #
# ask (one-shot)
# --------------------------------------------------------------------------- #
@app.command()
def summary(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama."),
) -> None:
    """Quick TL;DR of the paper — a single, cheap model call."""
    model = _apply_local(local, model)
    from papermind.output.terminal import render_summary, render_usage
    from papermind.summarize import summarize

    try:
        result, usage = summarize(source, model=model, console=console)
    except PaperMindError as exc:
        raise _fail(str(exc))
    render_summary(result, console)
    render_usage(usage, console)


@app.command()
def ask(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    question: str = typer.Argument(..., help="Your question about the paper."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    mode: str = typer.Option("balanced", "--mode"),
    section: Optional[str] = typer.Option(None, "--section", help="Focus retrieval on a section, e.g. '3.2'."),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable live streaming output."),
) -> None:
    """Ask a single question and print a grounded answer."""
    _validate_mode(mode)
    model = _apply_local(local, model)
    from papermind.qa.chat import PaperChat
    from papermind.output.terminal import render_answer, render_usage

    try:
        session = PaperChat(source, model=model, mode=mode, console=console)
        answer = _run_answer(
            lambda on_delta: session.ask(question, on_delta=on_delta, section=section), not no_stream
        )
    except PaperMindError as exc:
        raise _fail(str(exc))
    render_answer(answer, console)
    render_usage(answer.usage, console)


# --------------------------------------------------------------------------- #
# tutor
# --------------------------------------------------------------------------- #
@app.command()
def tutor(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable live streaming output."),
) -> None:
    """Reproduction tutor: debugging, environment, porting, and tuning help (explore mode)."""
    model = _apply_local(local, model)
    from papermind.qa.chat import PaperChat
    from papermind.output.terminal import render_answer, render_usage

    try:
        session = PaperChat(source, model=model, mode="explore", console=console)
    except PaperMindError as exc:
        raise _fail(str(exc))

    console.print(
        f"[bold cyan]PaperMind tutor[/bold cyan] — {session.report.paper.title}\n"
        "[dim]Ask about errors, environment setup, porting to your project, or tuning. "
        "'exit' to quit.[/dim]"
    )
    while True:
        try:
            question = console.input("\n[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", ":q"):
            break
        try:
            answer = _run_answer(lambda on_delta: session.tutor(question, on_delta=on_delta), not no_stream)
        except PaperMindError as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            continue
        render_answer(answer, console)
        render_usage(answer.usage, console, label="本轮")
    render_usage(session.usage, console, label="本次会话合计")


# --------------------------------------------------------------------------- #
# debug (paste an error)
# --------------------------------------------------------------------------- #
@app.command()
def debug(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    error: str = typer.Option(..., "--error", "-e", help="The error message / traceback you hit."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable live streaming output."),
) -> None:
    """Diagnose an error encountered while reproducing the paper."""
    model = _apply_local(local, model)
    from papermind.qa.chat import PaperChat
    from papermind.output.terminal import render_answer, render_usage

    try:
        session = PaperChat(source, model=model, mode="explore", console=console)
        answer = _run_answer(lambda on_delta: session.debug(error, on_delta=on_delta), not no_stream)
    except PaperMindError as exc:
        raise _fail(str(exc))
    render_answer(answer, console)
    render_usage(answer.usage, console)


# --------------------------------------------------------------------------- #
# reproduce / cite
# --------------------------------------------------------------------------- #
@app.command()
def reproduce(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    fmt: str = typer.Option("both", "--format", "-f", help="Output: sh | notebook | both."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output base path (default: ./repro)."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama (if analysis is needed)."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-analyze instead of using a cached report."),
) -> None:
    """Export the paper's reproduction guide as a runnable setup.sh and/or notebook."""
    if fmt not in ("sh", "notebook", "both"):
        raise _fail("--format must be one of: sh, notebook, both")
    model = _apply_local(local, model)
    report = _report_for_reproduction(source, model, refresh)
    if report.reproduction is None:
        raise _fail("该论文没有可导出的复现信息。")

    base = output or "repro"
    for ext in (".sh", ".ipynb"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    written = []
    if fmt in ("sh", "both"):
        path = f"{base}.sh"
        report.to_setup_script(path)
        written.append(path)
    if fmt in ("notebook", "both"):
        path = f"{base}.ipynb"
        report.to_notebook(path)
        written.append(path)
    console.print(f"[green]✓[/green] Wrote: {', '.join(written)}")


def _report_for_reproduction(source: str, model: Optional[str], refresh: bool):
    import json

    from papermind.cache import latest_report_path
    from papermind.config import load_config
    from papermind.output.schema import Report
    from papermind.parser.arxiv import resolve

    try:
        resolved = resolve(source, load_config())
    except PaperMindError as exc:
        raise _fail(str(exc))
    if not refresh:
        report_path = latest_report_path(resolved.cache_dir)
        if report_path is not None:
            cached = Report.from_dict(json.loads(report_path.read_text(encoding="utf-8")))
            if cached.reproduction is not None:
                return cached

    from papermind.analyze import analyze as run_analyze

    try:
        return run_analyze(
            source, model=model, modules=["reproduction"], with_figures=False, console=console, refresh=refresh
        )
    except PaperMindError as exc:
        raise _fail(str(exc))


# --------------------------------------------------------------------------- #
# search / batch
# --------------------------------------------------------------------------- #
@app.command()
def search(
    query: str = typer.Argument(..., help="Search terms."),
    num: int = typer.Option(10, "--num", "-n", help="Max results."),
) -> None:
    """Search arXiv and list matching papers (then `papermind analyze https://arxiv.org/abs/<id>`)."""
    from rich.table import Table

    from papermind.parser.arxiv import search_arxiv

    try:
        results = search_arxiv(query, num)
    except PaperMindError as exc:
        raise _fail(str(exc))
    if not results:
        console.print("[dim]没有结果。[/dim]")
        return
    table = Table(title=f"arXiv 搜索: {query}", header_style="bold cyan", title_justify="left")
    table.add_column("arXiv ID", style="cyan", no_wrap=True)
    table.add_column("年份", justify="right")
    table.add_column("标题")
    table.add_column("作者")
    for r in results:
        authors = (", ".join(r.authors[:3]) + (" et al." if len(r.authors) > 3 else "")) if r.authors else "-"
        table.add_row(r.arxiv_id or "-", str(r.year or "-"), r.title[:60], authors[:36])
    console.print(table)
    console.print("[dim]分析其中一篇: papermind analyze https://arxiv.org/abs/<arXiv ID>[/dim]")


@app.command()
def batch(
    sources: Optional[List[str]] = typer.Argument(None, help="One or more sources."),
    from_file: Optional[str] = typer.Option(None, "--from-file", help="File with one source per line."),
    directory: Optional[str] = typer.Option(None, "--dir", help="Analyze every *.pdf in a directory."),
    output: str = typer.Option("papermind-reports", "--output", "-o", help="Output directory."),
    fmt: str = typer.Option("md", "--format", "-f", help="md | json | html | all."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    only: Optional[str] = typer.Option(None, "--only", help="Comma-separated modules."),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache and re-analyze."),
) -> None:
    """Analyze multiple papers; write a report per paper plus an index.md."""
    from pathlib import Path

    items = list(sources or [])
    if from_file:
        items += [
            ln.strip()
            for ln in Path(from_file).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    if directory:
        items += [str(p) for p in sorted(Path(directory).glob("*.pdf"))]
    if not items:
        raise _fail("请提供至少一个来源：位置参数 / --from-file / --dir。")
    if fmt not in ("md", "json", "html", "all"):
        raise _fail("--format must be one of: md, json, html, all")
    modules = _parse_only(only)

    from papermind.analyze import analyze as run_analyze

    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for i, src in enumerate(items, 1):
        console.print(f"[bold]({i}/{len(items)})[/bold] {src}")
        try:
            report = run_analyze(src, model=model, modules=modules, console=console, refresh=refresh)
        except PaperMindError as exc:
            err_console.print(f"  [red]跳过: {exc}[/red]")
            failed.append((src, str(exc)))
            continue
        key = report.paper.arxiv_id or _safe_name(report.paper.title)
        _write_outputs(report, fmt, str(outdir / key))
        done.append((report.paper.title, key))

    _write_batch_index(outdir, done, failed, fmt)
    console.print(f"[green]✓[/green] 完成 {len(done)}/{len(items)}，输出目录: {outdir}")
    if failed:
        console.print(f"[yellow]{len(failed)} 篇失败（见 index.md）[/yellow]")


def _safe_name(title: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9]+", "-", title or "paper").strip("-").lower()
    return (slug or "paper")[:60]


def _write_batch_index(outdir, done, failed, fmt: str) -> None:
    ext = "md" if fmt in ("md", "all") else ("html" if fmt == "html" else "json")
    lines = ["# PaperMind 批量分析结果", "", f"共 {len(done)} 篇成功，{len(failed)} 篇失败。", ""]
    for title, key in done:
        lines.append(f"- [{title}]({key}.{ext})")
    if failed:
        lines += ["", "## 失败", ""]
        lines += [f"- `{src}` — {err}" for src, err in failed]
    (outdir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command()
def compare(
    sources: List[str] = typer.Argument(..., help="2-4 sources to compare side by side."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    fmt: str = typer.Option("md", "--format", "-f", help="md | json | html | all (with --output)."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path/prefix; prints to terminal if omitted."),
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="Skip the LLM comparison summary."),
    local: bool = typer.Option(False, "--local", help="Run fully local & free via Ollama."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-analyze instead of reusing cache."),
) -> None:
    """Compare 2-4 papers side by side (contributions / methods / benchmarks)."""
    if len(sources) < 2:
        raise _fail("compare 需要至少 2 篇论文。")
    if len(sources) > 4:
        raise _fail("一次最多对比 4 篇论文。")
    if fmt not in ("md", "json", "html", "csv", "all"):
        raise _fail("--format must be one of: md, json, html, csv, all")
    model = _apply_local(local, model)

    from papermind.compare import compare as run_compare
    from papermind.output.terminal import render_comparison, render_usage

    try:
        comparison = run_compare(
            sources, model=model, synthesize=not no_synthesis, console=console, refresh=refresh
        )
    except PaperMindError as exc:
        raise _fail(str(exc))

    if output:
        _write_comparison_outputs(comparison, fmt, output)
    else:
        render_comparison(comparison, console)
    render_usage(comparison.usage, console)


def _write_comparison_outputs(comparison, fmt: str, output: str) -> None:
    base = output
    for ext in (".md", ".json", ".html", ".csv"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    written = []
    if fmt in ("md", "all"):
        comparison.to_markdown(f"{base}.md")
        written.append(f"{base}.md")
    if fmt in ("json", "all"):
        comparison.to_json(f"{base}.json")
        written.append(f"{base}.json")
    if fmt in ("html", "all"):
        comparison.to_html(f"{base}.html")
        written.append(f"{base}.html")
    if fmt in ("csv", "all"):
        comparison.to_csv(f"{base}.csv")
        written.append(f"{base}.csv")
    console.print(f"[green]✓[/green] Wrote: {', '.join(written)}")


@app.command()
def cite(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write BibTeX to a file."),
) -> None:
    """Print a BibTeX entry for the paper (metadata only — no model calls)."""
    from papermind.config import load_config
    from papermind.output.cite import to_bibtex, write_bibtex
    from papermind.parser.arxiv import resolve

    try:
        resolved = resolve(source, load_config())
    except PaperMindError as exc:
        raise _fail(str(exc))
    if output:
        write_bibtex(resolved.meta, output)
        console.print(f"[green]✓[/green] Wrote {output}")
    else:
        console.print(to_bibtex(resolved.meta))


def _validate_mode(mode: str) -> None:
    if mode not in ("strict", "balanced", "explore"):
        raise _fail("--mode must be one of: strict, balanced, explore")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="e.g. openai-key, anthropic-key, model, embedding-provider."),
    value: str = typer.Argument(..., help="The value to store."),
) -> None:
    """Persist a config value to ~/.papermind/config.json."""
    from papermind import config as cfg

    try:
        field = cfg.set_value(key, value)
    except PaperMindError as exc:
        raise _fail(str(exc))
    console.print(f"[green]✓[/green] Set [bold]{field}[/bold] in {cfg.config_path()}")


@config_app.command("show")
def config_show() -> None:
    """Show the current effective configuration (secrets masked)."""
    from rich.table import Table

    from papermind import config as cfg

    view = cfg.masked_view()
    table = Table(title="PaperMind configuration", show_header=True, header_style="bold cyan")
    table.add_column("Key")
    table.add_column("Value")
    for key, value in view.items():
        table.add_row(key, "[dim]-[/dim]" if value is None else str(value))
    console.print(table)


# --------------------------------------------------------------------------- #
# open / list (library)
# --------------------------------------------------------------------------- #
@app.command("open")
def open_(
    source: str = typer.Argument(..., help="arXiv URL, any PDF URL, or local PDF path."),
    pdf: bool = typer.Option(False, "--pdf", help="Open the cached PDF instead of the report."),
) -> None:
    """Open the cached HTML report (or the PDF) in your browser."""
    import json

    from papermind.cache import latest_report_path
    from papermind.config import load_config
    from papermind.output.schema import Report
    from papermind.parser.arxiv import resolve

    try:
        resolved = resolve(source, load_config())
    except PaperMindError as exc:
        raise _fail(str(exc))
    if pdf:
        _open_path(resolved.pdf_path)
        return
    report_path = latest_report_path(resolved.cache_dir)
    if report_path is None:
        raise _fail("尚无缓存的分析结果，请先运行: papermind analyze <source>")
    report = Report.from_dict(json.loads(report_path.read_text(encoding="utf-8")))
    html = resolved.cache_dir / "report.html"
    report.to_html(str(html))
    _open_path(html)


@app.command("list")
def list_() -> None:
    """List papers you've analyzed (your local library)."""
    from rich.table import Table

    from papermind.cache import iter_cached_papers

    analyzed = [p for p in iter_cached_papers() if p.has_report]
    if not analyzed:
        console.print("[dim]还没有分析过的论文。运行 papermind analyze <paper> 开始。[/dim]")
        return
    table = Table(title="我的论文库", header_style="bold cyan", title_justify="left")
    table.add_column("arXiv / Key", style="cyan")
    table.add_column("标题")
    for p in analyzed:
        table.add_row(p.arxiv_id or p.key, (p.title or "-")[:60])
    console.print(table)


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
@cache_app.command("path")
def cache_path_cmd() -> None:
    """Print the cache root directory."""
    from papermind.config import load_config

    console.print(str(load_config().cache_root()))


@cache_app.command("list")
def cache_list() -> None:
    """List cached papers (PDF / index / report)."""
    from rich.table import Table

    from papermind.cache import human_size, iter_cached_papers

    papers = iter_cached_papers()
    if not papers:
        console.print("[dim]缓存为空。[/dim]")
        return
    table = Table(title="PaperMind 缓存", header_style="bold cyan", title_justify="left")
    table.add_column("Key", style="cyan")
    table.add_column("标题")
    table.add_column("报告", justify="center")
    table.add_column("索引", justify="center")
    table.add_column("大小", justify="right")
    for p in papers:
        table.add_row(
            p.key,
            (p.title or "-")[:46],
            "✓" if p.has_report else "-",
            "✓" if p.has_index else "-",
            human_size(p.size_bytes),
        )
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    key: Optional[str] = typer.Argument(None, help="Cache key to remove (see `cache list`)."),
    all_: bool = typer.Option(False, "--all", help="Clear the entire cache."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Remove one paper's cache, or everything with --all."""
    from papermind.cache import clear_all, clear_paper

    if all_:
        if not yes and not typer.confirm("清空所有缓存？"):
            raise typer.Exit()
        count = clear_all()
        console.print(f"[green]✓[/green] 已清空 {count} 个论文缓存。")
        return
    if not key:
        raise _fail("请指定要清理的 key，或用 --all 清空全部（papermind cache list 查看 key）。")
    if clear_paper(key):
        console.print(f"[green]✓[/green] 已清理缓存: {key}")
    else:
        raise _fail(f"未找到缓存 key: {key}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
