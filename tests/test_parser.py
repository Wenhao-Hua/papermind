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


# --------------------------------------------------------------------------- #
# parse_pdf boundary paths (fitz stubbed; no real PDF)
# --------------------------------------------------------------------------- #
class _FakePage:
    def __init__(self, text, blocks):
        self._text, self._blocks = text, blocks

    def get_text(self, kind="text"):
        return self._text if kind == "text" else self._blocks


class _FakeDoc:
    def __init__(self, pages):
        self._pages, self.page_count = pages, len(pages)

    def load_page(self, pno):
        return self._pages[pno]

    def close(self):
        pass


def _boom_open(*a, **k):
    raise RuntimeError("corrupt or unreadable PDF")


def test_parse_pdf_open_failure_raises_parseerror(monkeypatch, tmp_path):
    import fitz

    from papermind.errors import ParseError
    from papermind.output.schema import PaperMeta
    from papermind.parser.pdf import parse_pdf

    monkeypatch.setattr(fitz, "open", _boom_open)
    with pytest.raises(ParseError):
        parse_pdf(tmp_path / "x.pdf", PaperMeta(title="T"), tmp_path, extract_figures=False)


def test_parse_pdf_scanned_pdf_has_no_text_raises(monkeypatch, tmp_path):
    import fitz

    from papermind.errors import ParseError
    from papermind.output.schema import PaperMeta
    from papermind.parser.pdf import parse_pdf

    page = _FakePage(text="", blocks=[])  # no extractable text blocks -> scanned image
    monkeypatch.setattr(fitz, "open", lambda *a, **k: _FakeDoc([page]))
    with pytest.raises(ParseError) as ei:
        parse_pdf(tmp_path / "scan.pdf", PaperMeta(title="T"), tmp_path, extract_figures=False)
    assert "scanned" in str(ei.value).lower()


def test_parse_pdf_dehyphenates_and_infers_title(monkeypatch, tmp_path):
    import fitz

    from papermind.output.schema import PaperMeta
    from papermind.parser.pdf import parse_pdf

    blocks = [
        (0, 0, 200, 12, "A Great Title", 0, 0),
        (0, 30, 200, 60, "we use atten-\ntion for long-\nrange deps", 1, 0),
    ]
    page = _FakePage(text="A Great Title\nwe use attention for long range deps", blocks=blocks)
    monkeypatch.setattr(fitz, "open", lambda *a, **k: _FakeDoc([page]))

    parsed = parse_pdf(tmp_path / "x.pdf", PaperMeta(title="paper"), tmp_path, extract_figures=False)
    joined = " ".join(b.text for b in parsed.blocks)
    assert "attention" in joined and "atten- tion" not in joined  # hyphenated line-break merged
    assert "longrange" in joined                                  # second hyphenation merged
    assert parsed.meta.title == "A Great Title"                   # 'paper' placeholder -> inferred title
