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

    for bad in ("2307.08691", "arxiv:2307.08691"):  # bare ids -> guided to a URL, never title-searched
        with pytest.raises(SourceError) as exc:
            resolve(bad)  # no network hit — rejected before any search/download
        assert "arxiv.org/abs/2307.08691" in str(exc.value)


def test_parse_doi_forms():
    from papermind.parser.arxiv import _parse_doi

    assert _parse_doi("10.1145/3292500.3330701") == "10.1145/3292500.3330701"
    assert _parse_doi("doi:10.1038/s41586-021-03819-2") == "10.1038/s41586-021-03819-2"
    assert _parse_doi("DOI: 10.1101/2020.01.01.000001") == "10.1101/2020.01.01.000001"
    assert _parse_doi("not a doi") is None
    assert _parse_doi("https://arxiv.org/abs/2307.08691") is None


def test_resolve_doi_routes_through_doi_org(monkeypatch):
    from papermind.config import Config
    from papermind.parser import arxiv as ax

    captured = {}

    def _fake_resolve_url(url, config):
        captured["url"] = url
        return "RESOLVED"

    monkeypatch.setattr(ax, "_resolve_url", _fake_resolve_url)
    assert ax.resolve("10.1145/3292500.3330701", config=Config()) == "RESOLVED"
    assert captured["url"] == "https://doi.org/10.1145/3292500.3330701"


def test_resolve_title_searches_arxiv(monkeypatch):
    from papermind.config import Config
    from papermind.output.schema import PaperMeta
    from papermind.parser import arxiv as ax

    monkeypatch.setattr(
        ax, "search_arxiv", lambda q, max_results=1: [PaperMeta(title="Attention", arxiv_id="1706.03762")]
    )
    monkeypatch.setattr(ax, "_resolve_arxiv", lambda aid, config: f"ARXIV:{aid}")
    assert ax.resolve("Attention is all you need", config=Config()) == "ARXIV:1706.03762"


def test_resolve_title_no_match_raises(monkeypatch):
    from papermind.config import Config
    from papermind.errors import SourceError
    from papermind.parser import arxiv as ax

    monkeypatch.setattr(ax, "search_arxiv", lambda q, max_results=1: [])
    with pytest.raises(SourceError):
        ax.resolve("an obscure paper that does not exist anywhere", config=Config())


def test_resolve_arxiv_falls_back_to_pdf_when_metadata_throttled(monkeypatch, tmp_path):
    # arXiv's metadata API 429s (cloud IPs get throttled) -> analysis must still proceed
    # by downloading the PDF and inferring a title, not hard-fail the whole request.
    from papermind.config import load_config
    from papermind.errors import SourceError
    from papermind.parser import arxiv as ax

    monkeypatch.setenv("PAPERMIND_HOME", str(tmp_path))
    cfg = load_config()

    def _boom_metadata(arxiv_id):
        raise SourceError("arXiv 请求失败（可能被限流）")

    monkeypatch.setattr(ax, "_fetch_arxiv_metadata", _boom_metadata)
    monkeypatch.setattr(ax, "_download_pdf", lambda url, dest: dest.write_bytes(b"%PDF-1.5 x"))
    monkeypatch.setattr(ax, "_title_from_pdf", lambda pdf_path, url: "Adam: A Method for Stochastic Optimization")

    resolved = ax._resolve_arxiv("1412.6980", cfg)
    assert resolved.meta.title == "Adam: A Method for Stochastic Optimization"
    assert resolved.meta.arxiv_id == "1412.6980"
    assert resolved.pdf_path.exists()
    # metadata.json must NOT be cached, so a later run retries the API once it recovers.
    assert not (resolved.cache_dir / "metadata.json").exists()


# --------------------------------------------------------------------------- #
# Any-paper resolution: a landing page (HTML) -> its citation_pdf_url
# --------------------------------------------------------------------------- #
def test_citation_pdf_url_both_attribute_orders_and_relative():
    from papermind.parser.arxiv import _citation_pdf_url

    a = '<meta name="citation_pdf_url" content="https://x.org/p.pdf">'
    assert _citation_pdf_url(a, "https://x.org/abs/1") == "https://x.org/p.pdf"
    b = "<meta content='/files/p.pdf' name='citation_pdf_url'>"  # reversed order + relative
    assert _citation_pdf_url(b, "https://x.org/abs/1") == "https://x.org/files/p.pdf"
    c = '<meta content="https://aclanthology.org/2020.acl-main.1.pdf" name=citation_pdf_url>'  # unquoted name (ACL)
    assert _citation_pdf_url(c, "https://aclanthology.org/2020.acl-main.1/") == "https://aclanthology.org/2020.acl-main.1.pdf"
    assert _citation_pdf_url("<html>no tag here</html>", "https://x.org") is None


def _fake_safe_stream(body: bytes):
    import contextlib

    @contextlib.contextmanager
    def _cm(method, url, *, headers=None, timeout=30.0, max_redirects=5):
        class _Resp:
            headers = {}

            def raise_for_status(self):
                pass

            def iter_bytes(self, chunk_size=65536):
                for i in range(0, len(body), chunk_size):
                    yield body[i : i + chunk_size]

        yield _Resp()

    return _cm


def test_resolve_pdf_url_passes_through_direct_pdf(monkeypatch):
    import papermind.net as net
    from papermind.parser.arxiv import _resolve_pdf_url

    monkeypatch.setattr(net, "safe_stream", _fake_safe_stream(b"%PDF-1.7 ...bytes..."))
    assert _resolve_pdf_url("https://host.org/paper.pdf") == "https://host.org/paper.pdf"


def test_resolve_pdf_url_finds_pdf_on_landing_page(monkeypatch):
    import papermind.net as net
    from papermind.parser.arxiv import _resolve_pdf_url

    html = b'<html><head><meta name="citation_pdf_url" content="https://openreview.net/pdf?id=abc"></head></html>'
    monkeypatch.setattr(net, "safe_stream", _fake_safe_stream(html))
    assert _resolve_pdf_url("https://openreview.net/forum?id=abc") == "https://openreview.net/pdf?id=abc"


def test_resolve_pdf_url_page_without_pdf_raises(monkeypatch):
    import papermind.net as net
    from papermind.errors import SourceError
    from papermind.parser.arxiv import _resolve_pdf_url

    monkeypatch.setattr(net, "safe_stream", _fake_safe_stream(b"<html><body>just a webpage</body></html>"))
    with pytest.raises(SourceError):
        _resolve_pdf_url("https://example.com/some-page")


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


def test_detect_heading_numbered_with_trailing_period():
    from papermind.parser.pdf import _detect_heading

    # PDF extraction often leaves a trailing period on a numbered heading line.
    assert _detect_heading("3.1 Experiments.") == ("3.1", "Experiments")
    assert _detect_heading("3.1 Experiments") == ("3.1", "Experiments")
    assert _detect_heading("Abstract") == (None, "Abstract")
    assert _detect_heading("We propose a transformer that does many useful things in this long sentence") is None


def test_figures_digest_only_with_image_filters_caption_only():
    from pathlib import Path

    from papermind.output.schema import PaperMeta
    from papermind.parser.pdf import ExtractedFigure, ParsedPaper

    figs = [
        ExtractedFigure(label="Figure 1", caption="has image", page=1, image_path="/tmp/a.png"),
        ExtractedFigure(label="Figure 2", caption="caption only", page=2, image_path=None),
    ]
    parsed = ParsedPaper(meta=PaperMeta(title="T"), pages=["p"], blocks=[], sections=[],
                         figures=figs, cache_dir=Path("."))
    assert "Figure 1" in parsed.figures_digest() and "Figure 2" in parsed.figures_digest()
    only = parsed.figures_digest(only_with_image=True)
    assert "Figure 1" in only and "Figure 2" not in only  # caption-only figure excluded from matching
