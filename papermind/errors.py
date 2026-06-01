"""Shared exception types so the CLI can present clean, actionable messages."""

from __future__ import annotations


class PaperMindError(Exception):
    """Base class for all expected PaperMind failures (shown to the user without a traceback)."""


class SourceError(PaperMindError):
    """Bad input source: unrecognized arXiv id/URL, missing PDF, download failure."""


class ParseError(PaperMindError):
    """PDF could not be parsed."""


class LLMError(PaperMindError):
    """Model call failed or returned unparseable output."""


class ConfigError(PaperMindError):
    """Invalid configuration access."""
