"""Main analysis orchestration: resolve source -> parse PDF -> run modules ->
attach figures -> assemble Report.

This is the single entry point behind both ``papermind analyze`` and the public
``papermind.analyze`` Python API.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

from papermind.cache import load_cached_report, report_cache_path, save_report
from papermind.config import Config, load_config
from papermind.llm.base import LLMClient
from papermind.llm.prompts import paper_context
from papermind.modules import connections as m_connections
from papermind.modules import contributions as m_contributions
from papermind.modules import reproduction as m_reproduction
from papermind.modules import technical as m_technical
from papermind.output.schema import Report
from papermind.parser.arxiv import resolve
from papermind.parser.pdf import ParsedPaper, parse_pdf

ALL_MODULES = ["contributions", "technical", "connections", "reproduction"]
_BODY_BUDGET = 48_000  # chars of body text fed to analysis prompts


def analyze(
    source: str,
    model: Optional[str] = None,
    modules: Optional[List[str]] = None,
    with_figures: bool = True,
    config: Optional[Config] = None,
    console=None,
    refresh: bool = False,
) -> Report:
    """Analyze a paper and return a structured :class:`Report`.

    Args:
        source: arXiv URL, ``arxiv:ID``, bare id, or local PDF path.
        model: litellm model name (defaults to the configured model).
        modules: subset of {contributions, technical, connections, reproduction}.
        with_figures: extract/generate figures for technical points.
        config: explicit Config (otherwise loaded from env + file).
        console: optional rich Console to show a progress bar.
        refresh: ignore any cached report and re-run the analysis.
    """
    config = config or load_config()
    modules = modules or ALL_MODULES
    notice = (lambda msg: console.print(f"[dim]{msg}[/dim]")) if console is not None else None
    client = LLMClient(model=model, config=config, on_notice=notice)

    resolved = resolve(source, config)
    cache_path = report_cache_path(resolved.cache_dir, client.model, modules, with_figures)
    if not refresh:
        cached = load_cached_report(cache_path)
        if cached is not None:
            if notice:
                notice("已从缓存读取分析结果（未调用模型）；用 --refresh 重新分析。")
            return cached

    steps = ["parse"] + [m for m in ALL_MODULES if m in modules]
    if with_figures and "technical" in modules:
        steps.append("figures")

    with _progress(console, steps) as advance:
        parsed = parse_pdf(resolved.pdf_path, resolved.meta, resolved.cache_dir)
        advance("parse")

        context = _build_context(parsed)
        report = Report(paper=parsed.meta)

        if "contributions" in modules:
            report.contributions = m_contributions.run(parsed, client, context)
            advance("contributions")
        if "technical" in modules:
            report.technical = m_technical.run(parsed, client, context)
            advance("technical")
        if "connections" in modules:
            report.connections = m_connections.run(parsed, client, context)
            advance("connections")
        if "reproduction" in modules:
            report.reproduction = m_reproduction.run(parsed, client, context)
            advance("reproduction")

        if with_figures and "technical" in modules and report.technical.details:
            from papermind.figures.extract import match_original_figures
            from papermind.figures.generate import generate_diagrams

            match_original_figures(report.technical.details, parsed, client)
            generate_diagrams(report.technical.details, client)
            advance("figures")

    report.usage = client.usage
    save_report(report, cache_path)
    return report


def _build_context(parsed: ParsedPaper, budget: int = _BODY_BUDGET) -> str:
    body = parsed.full_text
    if len(body) > budget:
        body = body[:budget] + "\n...[truncated]..."
    return paper_context(
        title=parsed.meta.title,
        abstract=parsed.meta.abstract,
        outline=parsed.section_outline(),
        body=body,
    )


@contextmanager
def _progress(console, steps: List[str]):
    """Yield an ``advance(step_name)`` callable; uses a rich bar if a console is given."""
    labels = {
        "parse": "解析 PDF",
        "contributions": "贡献与创新点",
        "technical": "技术细节",
        "connections": "知识关联",
        "reproduction": "复现指南",
        "figures": "图示匹配/生成",
    }
    if console is None:
        yield lambda step: None
        return

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("分析中…", total=len(steps))

        def advance(step: str) -> None:
            progress.update(task, advance=1, description=labels.get(step, step))

        yield advance
