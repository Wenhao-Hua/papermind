"""One-call TL;DR summary of a paper (cheap, no full analysis)."""

from __future__ import annotations

from typing import Optional, Tuple

from papermind.analyze import _build_context
from papermind.config import Config, load_config
from papermind.llm.base import LLMClient
from papermind.llm.prompts import SUMMARY_SYSTEM, summary_user
from papermind.modules._util import str_list
from papermind.output.schema import Summary, Usage
from papermind.parser.arxiv import resolve
from papermind.parser.pdf import parse_pdf


def summarize(
    source: str,
    model: Optional[str] = None,
    config: Optional[Config] = None,
    console=None,
) -> Tuple[Summary, Usage]:
    """Return a (Summary, Usage) for the paper using a single model call."""
    config = config or load_config()
    notice = (lambda msg: console.print(f"[dim]{msg}[/dim]")) if console is not None else None
    client = LLMClient(model=model, config=config, on_notice=notice)
    client.ensure_ready()  # fail fast before download/parse if the model can't run

    resolved = resolve(source, config)
    if console is not None:
        with console.status("[cyan]解析论文…[/cyan]"):
            parsed = parse_pdf(resolved.pdf_path, resolved.meta, resolved.cache_dir, extract_figures=False)
    else:
        parsed = parse_pdf(resolved.pdf_path, resolved.meta, resolved.cache_dir, extract_figures=False)

    data = client.complete_json(SUMMARY_SYSTEM, summary_user(_build_context(parsed)))
    summary = Summary(
        title=parsed.meta.title,
        tldr=str(data.get("tldr", "") or "") if isinstance(data, dict) else "",
        key_points=str_list(data.get("key_points")) if isinstance(data, dict) else [],
    )
    return summary, client.usage
