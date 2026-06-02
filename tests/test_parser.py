"""Parser tests: arXiv id/URL detection and PDF helpers (no network, no LLM)."""

from __future__ import annotations

import pytest

from papermind.parser.arxiv import parse_arxiv_id
from papermind.parser.pdf import (
    _band_for,
    _detect_heading,
    _infer_title,
    _nearest_image_above,
    _x_overlap,
)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("https://arxiv.org/abs/2307.08691", "2307.08691"),
        ("https://arxiv.org/pdf/2307.08691.pdf", "2307.08691"),
        ("https://arxiv.org/abs/2307.08691v2", "2307.08691v2"),
    ],
)
def test_parse_arxiv_id_matches_urls(source, expected):
    assert parse_arxiv_id(source) == expected


@pytest.mark.parametrize(
    # Inputs are unified on URLs: bare ids / 'arxiv:ID' are no longer accepted.
    "source",
    ["arxiv:2307.08691", "2307.08691", "1706.03762", "hep-th/9901001",
     "./paper.pdf", "/home/me/x.pdf", "not an id", "report.json"],
)
def test_parse_arxiv_id_rejects_non_urls(source):
    assert parse_arxiv_id(source) is None


def test_resolve_rejects_bare_id_requires_url():
    from papermind.errors import SourceError
    from papermind.parser.arxiv import resolve

    for bad in ("2307.08691", "arxiv:2307.08691", "not a source"):
        with pytest.raises(SourceError):
            resolve(bad)  # bare ids no longer accepted; needs a URL or local PDF (no network hit)


def test_detect_heading_numbered():
    assert _detect_heading("3.1 Tiling and Recomputation") == ("3.1", "Tiling and Recomputation")
    assert _detect_heading("2 Background") == ("2", "Background")


def test_detect_heading_named():
    assert _detect_heading("Abstract") == (None, "Abstract")
    assert _detect_heading("References") == (None, "References")


def test_detect_heading_rejects_paragraph():
    para = "In this paper we propose a new method that improves throughput by reordering the loops."
    assert _detect_heading(para) is None


def test_infer_title():
    pages = ["FlashAttention-2: Faster Attention\nTri Dao\nAbstract\nWe present..."]
    assert _infer_title(pages) == "FlashAttention-2: Faster Attention"


# -- figure geometry helpers ------------------------------------------------- #
def test_x_overlap():
    assert _x_overlap((0, 0, 10, 10), (5, 0, 15, 10)) == 5
    assert _x_overlap((0, 0, 4, 10), (6, 0, 10, 10)) == 0  # disjoint columns


def test_band_for_picks_containing_column():
    columns = [(0, 100), (110, 210)]
    assert _band_for(columns, 50) == (0, 100)      # left column
    assert _band_for(columns, 150) == (110, 210)   # right column
    assert _band_for(columns, 105) == (0, 100)     # gutter -> nearest center


def test_nearest_image_above_requires_overlap_and_above():
    cap = (100, 500, 300, 520)  # caption bbox
    images = [
        (1, (100, 300, 300, 480)),  # above + overlaps  -> chosen
        (2, (100, 540, 300, 700)),  # below caption     -> rejected
        (3, (600, 300, 800, 480)),  # other column      -> rejected
    ]
    match = _nearest_image_above(images, cap)
    assert match is not None and match[0] == 1


def test_nearest_image_above_none_when_only_vector():
    cap = (100, 500, 300, 520)
    images = [(2, (100, 540, 300, 700))]  # nothing above in column -> triggers render path
    assert _nearest_image_above(images, cap) is None
