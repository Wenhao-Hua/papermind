"""PaperMind — read arXiv papers for real.

Public API::

    from papermind import analyze, PaperChat
    report = analyze("arxiv:2307.08691", model="gpt-4o")
    chat = PaperChat("arxiv:2307.08691", mode="balanced")

Heavy dependencies (litellm, faiss, pymupdf) are imported lazily so that
``import papermind`` and ``papermind --help`` stay fast and do not require a
full install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["analyze", "compare", "summarize", "PaperChat", "Report", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from papermind.analyze import analyze
    from papermind.compare import compare
    from papermind.output.schema import Report
    from papermind.qa.chat import PaperChat
    from papermind.summarize import summarize


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    if name == "analyze":
        from papermind.analyze import analyze

        return analyze
    if name == "compare":
        from papermind.compare import compare

        return compare
    if name == "summarize":
        from papermind.summarize import summarize

        return summarize
    if name == "PaperChat":
        from papermind.qa.chat import PaperChat

        return PaperChat
    if name == "Report":
        from papermind.output.schema import Report

        return Report
    raise AttributeError(f"module 'papermind' has no attribute {name!r}")
