"""Citation verification shared by Q&A and analysis.

Checks that a cited snippet actually appears in the paper, and snaps its
section/page to the matched source (parser-derived, reliable). Pure text
matching — no embeddings, no network, no model calls.

Verification must be *strict*: a citation is "verified" only if the quoted text
actually occurs in the paper (verbatim, modulo PDF extraction noise). We match
substring-first, then fall back to the longest contiguous run the quote shares
with a chunk. An unordered token/word-bag overlap was deliberately removed — a
fabricated sentence built from the paper's own vocabulary trivially overlaps some
chunk, which made the verifier mark hallucinated quotes as "verified" (a trust
killer). A paraphrase or a translation is *not* verifiable as a quote and is
honestly left unverified.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

VERIFY_THRESHOLD = 0.55
_MIN_SUBSTR = 12  # only treat short-enough quotes as substrings to avoid trivial hits

# PDF text renders "fl"/"fi" etc. as single ligature glyphs; fold them so a real
# quote still matches the extracted text.
_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"})


def _norm(text: str) -> str:
    t = (text or "").translate(_LIGATURES).lower()
    t = re.sub(r"-\s+", "", t)  # join line-break hyphenation: "atten- tion" -> "attention"
    t = re.sub(r"[^0-9a-z一-鿿]+", " ", t)  # punctuation/symbols -> space
    return re.sub(r"\s+", " ", t).strip()


def _contiguous_ratio(a: str, b: str) -> float:
    """Longest contiguous run of ``a`` that also occurs in ``b``, as a fraction of
    ``a`` — order-aware, so a word-bag of paper terms can't fake a match."""
    if not a:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size / len(a)


def best_match(text: str, chunks) -> Tuple[Optional[object], float]:
    """Return (chunk, score in [0,1]) of the chunk that best supports ``text``."""
    norm_cited = _norm(text)
    if not norm_cited:
        return None, 0.0
    best, best_score = None, 0.0
    for chunk in chunks:
        nctext = _norm(getattr(chunk, "text", "") or "")
        if len(norm_cited) >= _MIN_SUBSTR and norm_cited in nctext:
            return chunk, 1.0  # verbatim (normalized) containment — a real quote
        score = _contiguous_ratio(norm_cited, nctext)
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
