"""Citation verification shared by Q&A and analysis.

Checks that a cited snippet actually appears in the paper, and snaps its
section/page to the matched source (parser-derived, reliable). Pure text
matching — no embeddings, no network, no model calls.

Matching is substring-first (the strongest signal for a real quote), then a
token/shingle overlap ratio as a fallback for paraphrases and translations.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

VERIFY_THRESHOLD = 0.4
_MIN_SUBSTR = 12  # only treat short-enough quotes as substrings to avoid trivial hits


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _shingles(text: str) -> set:
    text = (text or "").lower()
    tokens = set(re.findall(r"[a-z0-9]{2,}", text))  # english / numeric words
    cjk = re.findall(r"[一-鿿]", text)
    tokens |= {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}  # chinese char bigrams
    return tokens


def best_match(text: str, chunks) -> Tuple[Optional[object], float]:
    """Return (chunk, score in [0,1]) of the chunk that best supports ``text``."""
    norm_cited = _norm(text)
    cited = _shingles(text)
    best, best_score = None, 0.0
    for chunk in chunks:
        ctext = getattr(chunk, "text", "") or ""
        if len(norm_cited) >= _MIN_SUBSTR and norm_cited in _norm(ctext):
            return chunk, 1.0  # exact (normalized) containment — a real quote
        shing = _shingles(ctext)
        score = (len(cited & shing) / len(cited)) if cited else 0.0
        if score > best_score:
            best, best_score = chunk, score
    return best, best_score


def verify_items(items, chunks, threshold: float = VERIFY_THRESHOLD) -> None:
    """For each item with a ``.text``, set ``.verified`` and, when matched, snap
    ``.section``/``.page`` to the matched chunk. Items must expose ``verified``,
    ``section`` and ``page`` attributes (e.g. ``EvidenceItem`` and ``Source``)."""
    for item in items:
        chunk, score = best_match(getattr(item, "text", ""), chunks)
        if chunk is not None and score >= threshold:
            item.verified = True
            section = getattr(chunk, "section", None)
            page = getattr(chunk, "page", None)
            if section:
                item.section = section
            if page:
                item.page = page
        else:
            item.verified = False
