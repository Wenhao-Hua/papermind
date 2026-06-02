"""Match extracted original figures to technical points by semantics (LLM).

Only figures that actually have a saved image are attached. Figure handling is
best-effort: any failure here leaves technical points without a figure (the
SVG/AI diagram generator then fills the gap) rather than aborting the analysis.
"""

from __future__ import annotations

from typing import Dict, List

from papermind.errors import LLMError
from papermind.llm.base import LLMClient
from papermind.llm.prompts import FIGURE_MATCH_SYSTEM, figure_match_user
from papermind.output.schema import Figure, TechnicalPoint
from papermind.parser.pdf import ParsedPaper


def match_original_figures(
    points: List[TechnicalPoint], parsed: ParsedPaper, client: LLMClient
) -> None:
    usable = [f for f in parsed.figures if f.image_path]
    if not usable or not points:
        return

    fig_by_label: Dict[str, object] = {f.label: f for f in usable}
    try:
        data = client.complete_json(
            FIGURE_MATCH_SYSTEM,
            figure_match_user([p.name for p in points], parsed.figures_digest()),
        )
    except LLMError:
        return

    matches = data.get("matches") if isinstance(data, dict) else data
    if not isinstance(matches, list):
        return

    point_by_name = {p.name.strip().lower(): p for p in points}
    for m in matches:
        if not isinstance(m, dict):
            continue
        label = m.get("figure")
        point = point_by_name.get(str(m.get("point", "")).strip().lower())
        if point is None or not label:
            continue
        fig = fig_by_label.get(str(label).strip())
        if fig is None:
            continue
        point.figure = Figure(
            type="original",
            image_path=fig.image_path,
            caption=fig.caption or label,
        )
