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
    image_figures: bool = False,
    svg_figures: bool = False,
    on_progress=None,
) -> Report:
    """Analyze a paper and return a structured :class:`Report`.

    Args:
        source: arXiv URL, any PDF URL, or local PDF path.
        model: litellm model name (defaults to the configured model).
        modules: subset of {contributions, technical, connections, reproduction}.
        with_figures: extract/generate figures for technical points.
        config: explicit Config (otherwise loaded from env + file).
        console: optional rich Console to show a progress bar.
        refresh: ignore any cached report and re-run the analysis.
        image_figures: use an image model for figures (needs config.image_model).
        svg_figures: generate architecture-faithful teaching SVGs (else Mermaid).
        on_progress: optional callback(step_label) for external progress (e.g. web).
    """
    config = config or load_config()
    modules = modules or ALL_MODULES
    notice = (lambda msg: console.print(f"[dim]{msg}[/dim]")) if console is not None else None
    client = LLMClient(model=model, config=config, on_notice=notice)

    resolved = resolve(source, config)
    # image_figures and svg_figures aren't mutually exclusive (svg defaults on), so decide a
    # single mode: image wins when usable, else svg, else mermaid. This keeps the cache key
    # honest (an image run and an svg run don't collide on fig=svg) and avoids running both.
    use_image = bool(image_figures and config.image_model)
    fig_mode = ("off" if not with_figures else "image" if use_image else "svg" if svg_figures else "mermaid")
    cache_path = report_cache_path(resolved.cache_dir, client.model, modules, fig_mode)
    if not refresh:
        cached = load_cached_report(cache_path)
        if cached is not None:
            if notice:
                notice("已从缓存读取分析结果（未调用模型）；用 --refresh 重新分析。")
            return cached

    client.ensure_ready()  # cache miss -> we will call the model: fail fast if it can't run

    steps = ["parse"] + [m for m in ALL_MODULES if m in modules]
    if with_figures and "technical" in modules:
        steps.append("figures")

    with _progress(console, steps, on_progress=on_progress) as advance:
        parsed = parse_pdf(resolved.pdf_path, resolved.meta, resolved.cache_dir)
        advance("parse")

        context = _build_context(parsed, client, notice=notice)
        report = Report(paper=parsed.meta)

        # The four modules are independent -> run them concurrently (network-bound).
        runners = {
            "contributions": lambda: m_contributions.run(parsed, client, context),
            "technical": lambda: m_technical.run(parsed, client, context),
            "connections": lambda: m_connections.run(parsed, client, context),
            "reproduction": lambda: m_reproduction.run(parsed, client, context),
        }
        selected = [m for m in ALL_MODULES if m in modules]
        results = _run_modules(selected, runners, advance, notice=notice)

        if "contributions" in results:
            report.contributions = results["contributions"]
        if "technical" in results:
            report.technical = results["technical"]
        if "connections" in results:
            report.connections = results["connections"]
        if "reproduction" in results:
            report.reproduction = results["reproduction"]

        # Verify citations against the real paper text (no model calls): flag any
        # quoted source we can't find, and snap each to its true section/page.
        _verify_citations(report, parsed)

        if with_figures and "technical" in modules and report.technical.details:
            from papermind.figures.extract import match_original_figures
            from papermind.figures.generate import generate_diagrams, generate_image_diagrams, generate_svg_diagrams

            match_original_figures(report.technical.details, parsed, client)
            if use_image:
                generate_image_diagrams(
                    report.technical.details, client, parsed.cache_dir, config.image_model, on_notice=notice
                )
            elif svg_figures:
                # Architecture-faithful teaching SVGs, conditioned on the paper context.
                # No Mermaid fallback here: with SVGs on we'd rather show no figure than
                # a generic candy-colored flowchart. A separate (typically fast, non-thinking)
                # figure model can be configured to speed this up, at some cost to layout polish.
                if config.figure_model:
                    # A configured (typically fast, non-thinking) figure model: no reasoning, modest budget.
                    fig_client = LLMClient(model=config.figure_model, config=config, on_notice=notice)
                    generate_svg_diagrams(
                        report.technical.details, fig_client, context, on_notice=notice,
                        reasoning_effort=None, max_tokens=18000,
                    )
                else:
                    # Default = the main model, usually a thinking model: keep reasoning minimal (else it
                    # spends the whole budget deliberating over the layout rules and starves the SVG) and
                    # give it ample headroom for reasoning + the SVG body.
                    generate_svg_diagrams(
                        report.technical.details, client, context, on_notice=notice,
                        reasoning_effort="minimal", max_tokens=32000,
                    )
            else:
                # Legacy path (only when SVGs aren't requested): Mermaid fills any gap.
                generate_diagrams(report.technical.details, client)
            advance("figures")

    report.usage = client.usage
    save_report(report, cache_path)
    return report


_MODULE_LABELS = {
    "contributions": "核心贡献", "technical": "技术细节",
    "connections": "关联工作", "reproduction": "复现指南",
}


def _run_modules(selected: List[str], runners: dict, advance, notice=None) -> dict:
    """Run the selected analysis modules concurrently (or serially if just one).

    A module that fails (e.g. the model returns JSON we can't repair) is skipped —
    its section stays at its default (empty) and the rest of the report is still
    produced. One bad module must not take down the whole analysis.
    """
    results: dict = {}

    def _safe(name: str):
        try:
            return runners[name]()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the report
            if notice:
                notice(f"「{_MODULE_LABELS.get(name, name)}」模块分析失败，已跳过：{exc}")
            return None

    if len(selected) <= 1:
        for name in selected:
            result = _safe(name)
            if result is not None:
                results[name] = result
            advance(name)
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {pool.submit(_safe, name): name for name in selected}
        for future in as_completed(futures):  # iterated on the main thread
            name = futures[future]
            result = future.result()  # _safe swallowed any module error -> won't raise
            if result is not None:
                results[name] = result
            advance(name)
    return results


def _build_context(parsed: ParsedPaper, client=None, budget: int = _BODY_BUDGET, notice=None) -> str:
    body = parsed.full_text
    if len(body) > budget:
        # Long paper: condense window-by-window so later sections (experiments /
        # reproduction details) aren't truncated away. With a client we run an
        # extractive map step (keeps verbatim wording); without one we fall back
        # to proportional per-section sampling.
        body = _long_paper_digest(parsed, client, budget, notice) if client is not None \
            else _aggregate_sections(parsed, budget)
    return paper_context(
        title=parsed.meta.title,
        abstract=parsed.meta.abstract,
        outline=parsed.section_outline(),
        body=body,
    )


def _section_segments(parsed: ParsedPaper) -> List[tuple]:
    """Merge consecutive same-section blocks into (section, text) segments."""
    segments: List[tuple] = []
    for block in parsed.blocks:
        if segments and segments[-1][0] == block.section:
            segments[-1] = (block.section, segments[-1][1] + " " + block.text)
        else:
            segments.append((block.section, block.text))
    return segments


def _pack_windows(segments: List[tuple], budget: int) -> List[tuple]:
    """Pack consecutive sections into ``<= budget`` char windows (splitting any
    single oversized section). Each window is (label, text)."""
    windows: List[tuple] = []
    label, parts, size = "", [], 0
    for section, text in segments:
        for start in range(0, len(text), budget):
            piece = text[start : start + budget]
            if parts and size + len(piece) > budget:
                windows.append((label, " ".join(parts)))
                label, parts, size = "", [], 0
            if not parts:
                label = section
            parts.append(piece)
            size += len(piece)
    if parts:
        windows.append((label, " ".join(parts)))
    return windows


def _long_paper_digest(parsed: ParsedPaper, client, budget: int, notice=None) -> str:
    """Extractive map-reduce: condense each window verbatim, then concatenate."""
    from papermind.llm.prompts import CONDENSE_SYSTEM, condense_user

    segments = _section_segments(parsed) or [("", parsed.full_text)]
    windows = _pack_windows(segments, budget)
    if not windows:
        return _aggregate_sections(parsed, budget)
    if notice:
        notice(f"长论文（约 {len(parsed.full_text) // 1000}k 字）：分 {len(windows)} 段提炼关键内容…")

    fallback_len = budget // len(windows)

    def _condense(window):
        label, text = window
        try:
            piece = client.complete(CONDENSE_SYSTEM, condense_user(label, text), max_tokens=1500)
        except Exception:  # noqa: BLE001 - fall back to a verbatim slice for just this window
            piece = ""
        return piece.strip() if piece and piece.strip() else text[:fallback_len]

    # Windows are independent -> condense them concurrently (network-bound). map() keeps order,
    # so the digest stays in reading order. LLMClient.usage is lock-protected for concurrent use.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(windows), 8)) as pool:
        digests = list(pool.map(_condense, windows))
    digest = "\n\n".join(d for d in digests if d)
    return digest[: int(budget * 1.3)] if digest else _aggregate_sections(parsed, budget)


def _aggregate_sections(parsed: ParsedPaper, budget: int) -> str:
    """Keep a per-section slice proportional to the section's length, so every
    section of a long paper is represented within the token budget."""
    segments = _section_segments(parsed)
    if not segments:
        return parsed.full_text[:budget]

    total = sum(len(text) for _, text in segments) or 1
    parts = []
    for section, text in segments:
        keep = max(150, int(budget * len(text) / total))  # floor so short sections still show
        snippet = text[:keep] + (" …" if len(text) > keep else "")
        parts.append(f"\n[{section}]\n{snippet}")
    out = "\n".join(parts)
    return out[: int(budget * 1.2)]  # cap if per-section floors overshot


def _verify_citations(report: Report, parsed: ParsedPaper) -> None:
    """Mark each quoted citation verified/unverified against the real paper text
    and snap its section/page to the true source. No model calls."""
    from papermind.qa.index import build_chunks
    from papermind.qa.verify import verify_items

    chunks = build_chunks(parsed)
    if chunks and report.contributions and report.contributions.sources:
        verify_items(report.contributions.sources, chunks)

    # Technical points cite by section/page only (no quote): sanity-check the page.
    npages = len(parsed.pages)
    if npages:
        for point in report.technical.details:
            if point.page is not None and not (1 <= point.page <= npages):
                point.page = None


@contextmanager
def _progress(console, steps: List[str], on_progress=None):
    """Yield an ``advance(step_name)`` callable; uses a rich bar if a console is
    given, and/or forwards the human label to ``on_progress`` (e.g. a web job)."""
    labels = {
        "parse": "解析 PDF",
        "contributions": "贡献与创新点",
        "technical": "技术细节",
        "connections": "知识关联",
        "reproduction": "复现指南",
        "figures": "图示匹配/生成",
    }

    def emit(step: str) -> None:
        if on_progress is not None:
            on_progress(labels.get(step, step))

    if console is None:
        yield emit
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
            emit(step)

        yield advance
