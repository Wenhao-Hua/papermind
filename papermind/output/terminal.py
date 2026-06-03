"""Rich terminal rendering for reports and answers."""

from __future__ import annotations

from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from papermind.output.schema import Answer, Comparison, Report, Summary, Usage

_DIFFICULTY_STYLE = {"high": "bold red", "mid": "yellow", "low": "green"}
_KIND_META = {
    "fact": ("论文事实", "green"),
    "inference": ("基于论文的推理", "yellow"),
    "out_of_scope": ("超出论文范围", "dim red"),
}
_CONFIDENCE_LABEL = {"high": "置信度: 高", "mid": "置信度: 中", "low": "置信度: 低"}


def render_report(report: Report, console: Console) -> None:
    paper = report.paper
    subtitle = []
    if paper.authors:
        subtitle.append(", ".join(paper.authors[:4]) + (" et al." if len(paper.authors) > 4 else ""))
    if paper.year:
        subtitle.append(str(paper.year))
    if paper.arxiv_id:
        subtitle.append(f"arXiv:{paper.arxiv_id}")
    console.print(
        Panel(
            Text(paper.title, style="bold cyan"),
            subtitle=" • ".join(subtitle) if subtitle else None,
            border_style="cyan",
        )
    )
    stats = []
    if report.technical.details:
        stats.append(f"{len(report.technical.details)} 个技术点")
    if report.connections.related_works:
        stats.append(f"{len(report.connections.related_works)} 条知识关联")
    if report.reproduction:
        stats.append("复现指南")
    if stats:
        console.print(f"[dim]📊 {' · '.join(stats)}[/dim]")

    if report.contributions:
        _render_contributions(report, console)
    if report.technical.details:
        _render_technical(report, console)
    if report.connections.related_works:
        _render_connections(report, console)
    if report.reproduction:
        _render_reproduction(report, console)


def _render_contributions(report: Report, console: Console) -> None:
    c = report.contributions
    body = Text()
    body.append("核心贡献: ", style="bold")
    body.append(c.main_contribution + "\n")
    body.append("新颖之处: ", style="bold")
    body.append(c.novelty + "\n")
    body.append("解决问题: ", style="bold")
    body.append(c.problem_solved)
    console.print(Panel(body, title="🎯 贡献与创新点", border_style="blue", title_align="left"))


def _render_technical(report: Report, console: Console) -> None:
    console.print()
    console.rule("[bold blue]🔬 技术细节解释[/bold blue]", style="blue")
    console.print("[dim]难度:[/dim] [bold red]high[/] · [yellow]mid[/] · [green]low[/]")
    for i, point in enumerate(report.technical.details, start=1):
        style = _DIFFICULTY_STYLE.get(point.difficulty, "white")
        header = Text()
        header.append(f"{i}. {point.name}  ", style="bold")
        header.append(f"[{point.difficulty}]", style=style)
        body = Text()
        body.append(point.explanation + "\n")
        if point.formula:
            body.append("\n🔢 公式: ", style="bold")
            body.append(point.formula, style="cyan")
        if point.analogy:
            body.append("\n💡 类比: ", style="bold")
            body.append(point.analogy)
        if point.source_section or point.page:
            loc = point.source_section or ""
            if point.page:
                loc = f"{loc} (p.{point.page})".strip()
            body.append(f"\n📍 出处: {loc}", style="dim")
        if point.figure:
            fig = point.figure
            if fig.image_path:
                origin = "原图" if fig.type == "original" else "AI 生成示意图(图片)"
                body.append(f"\n🖼  {origin}: {fig.caption or fig.image_path}", style="dim cyan")
            elif fig.svg:
                body.append("\n📊 AI 教学示意图 (SVG，见 Markdown/HTML 报告)", style="dim cyan")
            elif fig.mermaid:
                body.append("\n📊 AI 生成示意图 (Mermaid，见 Markdown/HTML 报告)", style="dim cyan")
            ex = fig.explain
            if ex and ex.gist:
                body.append(f"\n📖 读图：{ex.gist}", style="cyan")
                for part in ex.parts:
                    body.append(f"\n   · {part}", style="dim")
                if ex.takeaway:
                    body.append(f"\n   关键：{ex.takeaway}", style="dim cyan")
        console.print(Panel(body, title=header, border_style=style, title_align="left"))


def _render_connections(report: Report, console: Console) -> None:
    table = Table(title="🔗 知识关联", title_style="bold blue", show_lines=False, title_justify="left")
    table.add_column("概念", style="cyan", no_wrap=False)
    table.add_column("相关论文")
    table.add_column("关系")
    for w in report.connections.related_works:
        paper = w.paper + (f"\n{w.arxiv_link}" if w.arxiv_link else "")
        table.add_row(w.concept, paper, w.relationship)
    console.print(table)


def _render_reproduction(report: Report, console: Console) -> None:
    r = report.reproduction
    info = Text()
    if r.code_repo is not None:
        cr = r.code_repo
        badge = " · ✓官方实现" if cr.is_official else ""
        star = f" · ★{cr.stars}" if cr.stars else ""
        info.append(f"代码仓库 ({cr.source}{badge}{star}): ", style="bold")
        info.append(f"{cr.url}\n")
        cmds = list(cr.install_commands) + list(cr.run_commands)
        if cmds:
            info.append("安装/运行 (取自该仓库): ", style="bold")
            info.append("; ".join(cmds) + "\n")
    elif r.official_code:
        tag = f" ({r.version_tag})" if r.version_tag else ""
        info.append("代码仓库: ", style="bold")
        info.append(f"{r.official_code}{tag}\n")
    if r.requirements:
        info.append("环境要求: ", style="bold")
        info.append(r.requirements + "\n")
    if r.recommended_hardware:
        info.append("推荐硬件: ", style="bold")
        info.append(r.recommended_hardware + "\n")
    if r.key_hyperparams:
        info.append("关键超参: ", style="bold")
        info.append(", ".join(r.key_hyperparams))
    console.print(Panel(info, title="🛠️  复现指南", border_style="blue", title_align="left"))

    if r.env_setup_steps:
        steps = Table(title="环境配置步骤", title_style="bold", show_lines=True, title_justify="left")
        steps.add_column("#", style="cyan", width=3)
        steps.add_column("步骤")
        steps.add_column("命令", style="green")
        for s in r.env_setup_steps:
            steps.add_row(str(s.step), f"{s.title}\n[dim]{s.desc}[/dim]" if s.desc else s.title, s.command or "")
        console.print(steps)

    if r.performance_benchmarks:
        bench = Table(title="性能基准", title_style="bold", title_justify="left")
        for col in ("设置", "Baseline", "结果", "加速", "显存"):
            bench.add_column(col)
        for b in r.performance_benchmarks:
            bench.add_row(b.setting, b.baseline or "-", b.result or "-", b.speedup or "-", b.memory or "-")
        console.print(bench)

    if r.common_errors:
        console.print("[bold]常见报错:[/bold]")
        for e in r.common_errors:
            console.print(f"  [red]✗[/red] {e.error}")
            if e.cause:
                console.print(f"     [dim]原因: {e.cause}[/dim]")
            if e.fix_command:
                console.print(f"     [green]修复: {e.fix_command}[/green]")

    if r.gotchas:
        console.print("[bold yellow]⚠️  坑点:[/bold yellow]")
        for g in r.gotchas:
            console.print(f"  • {g}")


# --------------------------------------------------------------------------- #
# Answers (chat / ask / tutor / debug)
# --------------------------------------------------------------------------- #
def render_answer(answer: Answer, console: Console) -> None:
    for seg in answer.segments:
        label, color = _KIND_META.get(seg.kind, (seg.kind, "white"))
        tag = Text()
        tag.append(f"[{label}]", style=f"bold {color}")
        if seg.kind == "inference" and seg.confidence:
            tag.append(f"  {_CONFIDENCE_LABEL.get(seg.confidence, '')}", style=color)
        console.print(tag)
        console.print(seg.text)
        if seg.kind == "inference" and seg.reasoning:
            console.print(f"[dim]  推理依据: {seg.reasoning}[/dim]")
        console.print()

    if answer.evidence:
        ev = Table(title="原文依据", title_style="bold cyan", show_lines=False, title_justify="left", box=None)
        ev.add_column("出处", style="cyan", no_wrap=True)
        ev.add_column("片段")
        for item in answer.evidence:
            loc = item.section or ""
            if item.page:
                loc = f"{loc} p.{item.page}".strip()
            mark = "" if item.verified else "[yellow]⚠ 未在检索段落中核实[/yellow]\n"
            ev.add_row(loc or "-", mark + item.text)
        console.print(ev)


# --------------------------------------------------------------------------- #
# Token usage + streaming
# --------------------------------------------------------------------------- #
def render_summary(summary: "Summary", console: Console) -> None:
    body = Text()
    body.append(summary.tldr or "(无)")
    console.print(Panel(body, title=f"📄 TL;DR — {summary.title}", border_style="cyan", title_align="left"))
    if summary.key_points:
        console.print("[bold]要点:[/bold]")
        for point in summary.key_points:
            console.print(f"  • {point}")


def render_comparison(comparison: "Comparison", console: Console) -> None:
    from papermind.output.compare_render import _headers, _rows

    table = Table(title="📊 论文对比", header_style="bold cyan", title_justify="left", show_lines=True)
    table.add_column("维度", style="bold", no_wrap=True)
    for header in _headers(comparison):
        table.add_column(header, overflow="fold")
    for label, values in _rows(comparison):
        table.add_row(label, *values)
    console.print(table)
    if comparison.synthesis:
        console.print(Panel(Text(comparison.synthesis), title="对比小结", border_style="cyan", title_align="left"))


def render_usage(usage: "Usage", console: Console, label: str = "用量") -> None:
    if usage is None or usage.calls == 0:
        return
    cost = f" · ~${usage.cost_usd:.4f}" if usage.cost_usd else ""
    console.print(
        f"[dim]📊 {label}: {usage.calls} calls · {usage.total_tokens:,} tokens "
        f"(prompt {usage.prompt_tokens:,} / completion {usage.completion_tokens:,}){cost}[/dim]"
    )


@contextmanager
def streaming_view(console: Console):
    """Yield an ``on_delta`` callback that live-renders streamed text in a transient
    panel; the panel disappears on exit so the final structured answer can be shown."""
    from rich.live import Live

    state = {"text": ""}
    with Live(console=console, transient=True, refresh_per_second=12) as live:

        def on_delta(delta: str) -> None:
            state["text"] += delta
            tail = state["text"][-500:]
            live.update(
                Panel(Text(tail, style="dim"), title="生成中…", border_style="dim", title_align="left")
            )

        yield on_delta
