"""Compare 2+ papers side by side (contributions / methods / benchmarks).

Reuses each paper's cached analysis when available (cheap), assembles a
structured :class:`Comparison`, and optionally adds a one-call LLM synthesis.
"""

from __future__ import annotations

from typing import List, Optional

from papermind.cache import latest_report_path, load_cached_report
from papermind.config import Config, load_config
from papermind.errors import LLMError
from papermind.llm.base import LLMClient
from papermind.llm.prompts import COMPARE_SYSTEM, compare_user
from papermind.output.schema import ComparedPaper, Comparison, Report, Usage
from papermind.parser.arxiv import resolve

_COMPARE_MODULES = ["contributions", "technical", "reproduction"]


def compare(
    sources: List[str],
    model: Optional[str] = None,
    synthesize: bool = True,
    config: Optional[Config] = None,
    console=None,
    refresh: bool = False,
) -> Comparison:
    """Analyze each source (reusing cache) and return a side-by-side Comparison."""
    if len(sources) < 2:
        from papermind.errors import PaperMindError

        raise PaperMindError("compare 需要至少 2 篇论文。")
    config = config or load_config()

    reports: List[Report] = []
    total = Usage()
    for i, src in enumerate(sources, 1):
        if console is not None:
            console.print(f"[dim]({i}/{len(sources)}) 分析 {src}[/dim]")
        report = _report_for(src, model, refresh, console, config)
        reports.append(report)
        _accumulate(total, report.usage)

    comparison = build_comparison(reports)

    if synthesize:
        client = LLMClient(model=model, config=config)
        try:
            comparison.synthesis = client.complete(COMPARE_SYSTEM, compare_user(_briefs(comparison)))
        except LLMError:
            comparison.synthesis = ""
        _accumulate(total, client.usage)

    comparison.usage = total
    return comparison


def build_comparison(reports: List[Report]) -> Comparison:
    return Comparison(papers=[_compared(r) for r in reports])


def _compared(r: Report) -> ComparedPaper:
    c = r.contributions
    methods = [p.name for p in r.technical.details[:3]]
    benchmark = None
    if r.reproduction and r.reproduction.performance_benchmarks:
        b = r.reproduction.performance_benchmarks[0]
        result = b.speedup or b.result
        benchmark = f"{b.setting}: {result}" if result else b.setting
    return ComparedPaper(
        title=r.paper.title,
        arxiv_id=r.paper.arxiv_id,
        year=r.paper.year,
        main_contribution=c.main_contribution if c else "",
        novelty=c.novelty if c else "",
        methods=methods,
        benchmark=benchmark,
        hardware=r.reproduction.recommended_hardware if r.reproduction else None,
        official_code=r.reproduction.official_code if r.reproduction else None,
    )


def _has_compare_modules(r: Report) -> bool:
    """Reuse a cached report for comparison only if it actually has the modules compare
    reads. A partial run (e.g. `analyze --only contributions`) leaves reproduction/technical
    empty, which would silently blank out the comparison's method/benchmark/hardware columns."""
    return bool(r.contributions and r.technical and r.technical.details and r.reproduction)


def _report_for(source: str, model: Optional[str], refresh: bool, console, config: Config) -> Report:
    resolved = resolve(source, config)
    if not refresh:
        report_path = latest_report_path(resolved.cache_dir)
        if report_path is not None:
            cached = load_cached_report(report_path)
            if cached is not None and _has_compare_modules(cached):
                return cached

    from papermind.analyze import analyze as run_analyze

    return run_analyze(
        source, model=model, modules=_COMPARE_MODULES, with_figures=False, config=config,
        console=console, refresh=refresh,
    )


def _briefs(comparison: Comparison) -> str:
    parts = []
    for p in comparison.papers:
        parts.append(
            f"【{p.title}】\n贡献: {p.main_contribution}\n新颖: {p.novelty}\n"
            f"方法: {', '.join(p.methods)}\n基准: {p.benchmark or '-'}"
        )
    return "\n\n".join(parts)


def _accumulate(total: Usage, u: Optional[Usage]) -> None:
    if u is None:
        return
    total.calls += u.calls
    total.prompt_tokens += u.prompt_tokens
    total.completion_tokens += u.completion_tokens
    total.total_tokens += u.total_tokens
    total.cost_usd += u.cost_usd
