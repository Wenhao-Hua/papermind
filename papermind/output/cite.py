"""Export a paper's metadata as a BibTeX entry."""

from __future__ import annotations

import re
from pathlib import Path

from papermind.output.schema import PaperMeta


def to_bibtex(meta: PaperMeta) -> str:
    key = _cite_key(meta)
    fields = [f"  title        = {{{meta.title}}}"]
    if meta.authors:
        fields.append(f"  author       = {{{' and '.join(meta.authors)}}}")
    if meta.year:
        fields.append(f"  year         = {{{meta.year}}}")
    if meta.arxiv_id:
        fields.append(f"  eprint       = {{{meta.arxiv_id}}}")
        fields.append("  archivePrefix = {arXiv}")
        fields.append(f"  url          = {{https://arxiv.org/abs/{meta.arxiv_id}}}")
    entry_type = "article" if meta.arxiv_id else "misc"
    return f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def write_bibtex(meta: PaperMeta, path: str) -> str:
    text = to_bibtex(meta)
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return text


def _cite_key(meta: PaperMeta) -> str:
    parts = meta.authors[0].split() if meta.authors else []
    last = parts[-1] if parts else "anon"  # a blank/whitespace author must not IndexError
    year = str(meta.year) if meta.year else "0000"
    first_word = ""
    for token in re.split(r"\s+", meta.title or ""):
        word = re.sub(r"[^A-Za-z0-9]", "", token).lower()
        if len(word) >= 3:
            first_word = word
            break
    key = f"{_slug(last)}{year}{first_word}"
    return key or "paper"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()
