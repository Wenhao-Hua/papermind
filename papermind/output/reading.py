"""Estimated reading time for a Report — shared by the HTML and Markdown renderers."""

from __future__ import annotations

from typing import List

from papermind.output.schema import Report

_UNITS_PER_MIN = 200  # rough academic-reading pace; 1 CJK char ~= 1 unit


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF       # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF    # CJK Extension A
        or 0x3000 <= o <= 0x303F    # CJK punctuation/symbols（，。！？ live here / fullwidth below）
        or 0xFF00 <= o <= 0xFFEF    # fullwidth & halfwidth forms
    )


def reading_minutes(report: Report) -> int:
    """Estimate reading time in minutes from the report's prose.

    Counts whitespace-separated tokens (English/mixed) plus individual CJK characters
    — including CJK punctuation — each a reading unit in Chinese, then divides by
    200 units/min (academic pace). Always at least 1.
    """
    texts: List[str] = []
    if report.contributions:
        c = report.contributions
        texts += [c.main_contribution, c.novelty, c.problem_solved]
    for p in report.technical.details:
        texts += [p.explanation, p.analogy or ""]
    for w in report.connections.related_works:
        texts.append(w.relationship)
    if report.reproduction:
        r = report.reproduction
        texts += [r.requirements or "", r.recommended_hardware or ""]
        for step in r.env_setup_steps:
            texts.append(step.desc or "")
        texts += list(r.gotchas)
    total = 0
    for text in texts:
        if not text:
            continue
        total += len(text.split())
        total += sum(1 for ch in text if _is_cjk(ch))
    return max(1, round(total / _UNITS_PER_MIN))
