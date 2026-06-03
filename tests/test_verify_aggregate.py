"""Citation verification (analysis + Q&A shared) and long-paper aggregation.

No network, no FAISS, no real model calls.
"""

from __future__ import annotations

from pathlib import Path

from papermind.output.schema import Contributions, PaperMeta, Report, Source
from papermind.parser.pdf import ParsedPaper, TextBlock
from papermind.qa.index import Chunk
from papermind.qa.verify import best_match, verify_items


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def test_best_match_substring_is_strongest():
    chunks = [Chunk(idx=0, text="the quick brown fox jumps over the lazy dog", section="A", page=1)]
    chunk, score = best_match("quick brown fox jumps", chunks)
    assert chunk is not None and score == 1.0


def test_verify_items_flags_and_snaps_sources():
    chunks = [
        Chunk(idx=0, text="we scale the dot products by 1 over sqrt of d_k to keep gradients stable",
              section="3.2", page=4),
    ]
    grounded = Source(text="we scale the dot products by 1 over sqrt of d_k", section="WRONG", page=99)
    fabricated = Source(text="a totally invented claim that appears nowhere in the paper")
    verify_items([grounded, fabricated], chunks)
    assert grounded.verified and grounded.section == "3.2" and grounded.page == 4  # snapped
    assert not fabricated.verified


def test_fabricated_quote_from_paper_vocabulary_is_not_verified():
    # The trust-critical case: a hallucinated sentence built from the paper's OWN
    # words must NOT be marked verified. An unordered word-overlap match false-
    # positived here; the order-aware contiguous match rejects it.
    chunks = [Chunk(
        idx=0,
        text="we propose the transformer based solely on attention mechanisms dispensing with recurrence and convolutions",
        section="1", page=1,
    )]
    fake = Source(text="the transformer applies a convolutional recurrence module before the attention mechanism for speed")
    real = Source(text="based solely on attention mechanisms dispensing with recurrence")
    verify_items([fake, real], chunks)
    assert not fake.verified   # reuses 'transformer/attention/convolution/recurrence' but is absent -> must fail
    assert real.verified       # a genuine contiguous quote still verifies


def test_verify_citations_runs_over_report():
    from papermind.analyze import _verify_citations

    blocks = [TextBlock(text="the model uses rotary position embeddings for long context windows",
                        page=2, section="Method")]
    parsed = ParsedPaper(meta=PaperMeta(title="T"), pages=["pg one", "pg two"],
                         blocks=blocks, sections=[], figures=[], cache_dir=Path("."))
    report = Report(paper=parsed.meta, contributions=Contributions(
        main_contribution="x",
        sources=[
            Source(text="the model uses rotary position embeddings", section="?", page=9),
            Source(text="this sentence is fabricated and absent from the paper"),
        ],
    ))
    _verify_citations(report, parsed)
    grounded, fabricated = report.contributions.sources
    assert grounded.verified and grounded.section == "Method" and grounded.page == 2
    assert not fabricated.verified


# --------------------------------------------------------------------------- #
# Long-paper aggregation
# --------------------------------------------------------------------------- #
def _long_parsed(filler_chars: int) -> ParsedPaper:
    blocks = [
        TextBlock(text=f"key sentence in section {s}. " * 6, page=s + 1, section=f"S{s}")
        for s in range(3)
        for _ in range(4)
    ]
    return ParsedPaper(meta=PaperMeta(title="Long"), pages=["x" * filler_chars],
                       blocks=blocks, sections=[], figures=[], cache_dir=Path("."))


class _CondenseClient:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=None):
        self.calls += 1
        return f"[DIGEST{self.calls}] extracted key sentences"


def test_long_paper_uses_extractive_map_when_client_given():
    from papermind.analyze import _build_context

    parsed = _long_parsed(5000)  # full_text well over budget
    client = _CondenseClient()
    ctx = _build_context(parsed, client, budget=1500)
    assert client.calls >= 1  # the map step actually ran
    assert "DIGEST1" in ctx


def test_long_paper_falls_back_to_sampling_without_client():
    from papermind.analyze import _build_context

    parsed = _long_parsed(5000)
    ctx = _build_context(parsed, budget=1500)  # no client -> extractive sampling
    for s in range(3):
        assert f"S{s}" in ctx  # every section represented, none truncated away


def test_pack_windows_bounds_size_and_splits_big_section():
    from papermind.analyze import _pack_windows

    windows = _pack_windows([("BIG", "a" * 3500), ("small", "b" * 100)], budget=1000)
    assert all(len(text) <= 1000 for _, text in windows)
    assert len(windows) >= 4  # the oversized section is split into budget-sized pieces
