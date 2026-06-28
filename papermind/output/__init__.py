"""Output models and renderers (schema, markdown, json, terminal)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papermind.output.schema import Report

_READING_WPM = 200  # academic reading pace


def estimate_reading_time(report: "Report") -> int:
    """Return estimated reading time in minutes (academic pace, ~200 wpm)."""
    parts: list[str] = []
    if report.contributions:
        c = report.contributions
        parts += [c.main_contribution or "", c.novelty or "", c.problem_solved or ""]
        for s in c.sources:
            parts.append(s.text or "")
    for tp in report.technical.details:
        parts += [tp.explanation or "", tp.analogy or ""]
    for rw in report.connections.related_works:
        parts += [rw.concept or "", rw.relationship or ""]
    if report.reproduction:
        r = report.reproduction
        parts += [r.requirements or "", r.recommended_hardware or ""]
        for step in r.env_setup_steps:
            parts += [step.title or "", step.desc or ""]
        for err in r.common_errors:
            parts += [err.error or "", err.cause or ""]
    word_count = sum(len(p.split()) for p in parts if p.strip())
    return max(1, round(word_count / _READING_WPM))
