"""PDF parsing with PyMuPDF: page text, section segmentation, paragraph blocks
(each tagged with page + section for accurate citations), and figure/caption
extraction.

The parsed representation (``ParsedPaper``) is the shared input for every
analysis module and for the RAG index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from papermind.errors import ParseError
from papermind.output.schema import PaperMeta

# Headings like "3", "3.1", "3.1.2" followed by a Title-cased name.
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{1,80})$")
# Well-known unnumbered sections.
_NAMED_HEADING = re.compile(
    r"^\s*(Abstract|Introduction|Background|Related Work|Method(?:s|ology)?|"
    r"Experiments?|Results?|Evaluation|Discussion|Conclusions?|Limitations?|"
    r"References|Acknowledg(?:e)?ments?|Appendix)\b",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.?|Table)\s*([0-9]+)", re.IGNORECASE)

_MIN_FIGURE_DIM = 64  # px; smaller raster images are treated as icons/logos


@dataclass
class TextBlock:
    text: str
    page: int  # 1-based
    section: str


@dataclass
class Section:
    title: str
    number: Optional[str]
    page: int  # 1-based page where the heading appears


@dataclass
class ExtractedFigure:
    label: str  # e.g. "Figure 1"
    caption: str
    page: int  # 1-based
    image_path: Optional[str] = None  # saved PNG/JPEG, or None if no raster found


@dataclass
class ParsedPaper:
    meta: PaperMeta
    pages: List[str]  # raw text per page
    blocks: List[TextBlock]
    sections: List[Section]
    figures: List[ExtractedFigure]
    cache_dir: Path

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.pages)

    def section_outline(self) -> str:
        """A compact textual outline used as context for the LLM."""
        lines = []
        for sec in self.sections:
            prefix = f"{sec.number} " if sec.number else ""
            lines.append(f"- {prefix}{sec.title} (p.{sec.page})")
        return "\n".join(lines)

    def figures_digest(self) -> str:
        """One line per figure with its caption, for figure-matching prompts."""
        return "\n".join(
            f"- {fig.label} (p.{fig.page}): {fig.caption[:200]}" for fig in self.figures
        )


def parse_pdf(
    pdf_path: Path,
    meta: PaperMeta,
    cache_dir: Path,
    extract_figures: bool = True,
    render_vector_figures: bool = True,
) -> ParsedPaper:
    """Parse a PDF into pages, blocks, sections, and figures.

    Args:
        extract_figures: set False to skip all figure work (e.g. chat indexing).
        render_vector_figures: when a caption has no embedded raster above it
            (a vector figure), render that page region to a PNG instead.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ParseError(
            "PyMuPDF is required for PDF parsing. Install it with: pip install pymupdf"
        ) from exc

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:  # fitz raises a variety of errors
        raise ParseError(f"Could not open PDF {pdf_path}: {exc}") from exc

    pages: List[str] = []
    blocks: List[TextBlock] = []
    sections: List[Section] = []
    figures: List[ExtractedFigure] = []
    fig_dir = cache_dir / "figures"

    current_section = "Frontmatter"
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            page_no = pno + 1
            pages.append(page.get_text("text"))

            raw_blocks = _sorted_text_blocks(page)
            for btext in raw_blocks:
                heading = _detect_heading(btext)
                if heading is not None:
                    number, title = heading
                    current_section = f"{number} {title}".strip() if number else title
                    sections.append(Section(title=title, number=number, page=page_no))
                blocks.append(TextBlock(text=btext, page=page_no, section=current_section))

            if extract_figures:
                figures.extend(
                    _extract_page_figures(doc, page, page_no, fig_dir, render_vector_figures)
                )
    finally:
        doc.close()

    if not meta.title or meta.title.strip() in ("", "paper"):
        inferred = _infer_title(pages)
        if inferred:
            meta.title = inferred

    if not any(b.text.strip() for b in blocks):
        raise ParseError(
            f"No extractable text found in {pdf_path}. The PDF may be a scanned image "
            f"(OCR is not supported)."
        )

    return ParsedPaper(
        meta=meta,
        pages=pages,
        blocks=blocks,
        sections=sections,
        figures=figures,
        cache_dir=cache_dir,
    )


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _sorted_text_blocks(page) -> List[str]:
    """Text blocks in reading order, with intra-block line breaks collapsed."""
    raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = [b for b in raw if len(b) >= 6 and b[6] == 0 and b[4].strip()] if raw else []
    # Reading order: top-to-bottom, then left-to-right.
    text_blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))
    out = []
    for b in text_blocks:
        text = re.sub(r"-\n(?=\w)", "", b[4])  # join hyphenated line breaks
        text = re.sub(r"\s*\n\s*", " ", text).strip()
        if text:
            out.append(text)
    return out


def _detect_heading(text: str):
    """Return (number, title) if a block looks like a section heading, else None."""
    # Headings are short single lines; skip long paragraphs.
    if len(text) > 90 or text.endswith("."):
        # A named heading can still be a short standalone line ending without a period.
        named = _NAMED_HEADING.match(text)
        if named and len(text) <= 90:
            return (None, named.group(1).strip())
        return None
    m = _NUMBERED_HEADING.match(text)
    if m:
        return (m.group(1), m.group(2).strip())
    named = _NAMED_HEADING.match(text)
    if named:
        return (None, named.group(1).strip())
    return None


def _infer_title(pages: List[str]) -> Optional[str]:
    """Best-effort title from the first page: first substantial line near the top."""
    if not pages:
        return None
    for line in pages[0].splitlines():
        stripped = line.strip()
        if 8 <= len(stripped) <= 200 and not stripped.lower().startswith(("abstract", "arxiv:")):
            return stripped
    return None


# --------------------------------------------------------------------------- #
# Figure extraction
# --------------------------------------------------------------------------- #
def _extract_page_figures(
    doc, page, page_no: int, fig_dir: Path, render_vector: bool = True
) -> List[ExtractedFigure]:
    """Pair 'Figure N' / 'Table N' captions with the figure above them.

    Prefers an embedded raster image directly above the caption; when none exists
    (a vector figure) and ``render_vector`` is set, renders that page region to PNG.
    """
    captions = _find_captions(page)
    if not captions:
        return []

    images = _find_images(page)  # list of (xref, bbox)
    raw_blocks = [
        b for b in (page.get_text("blocks") or []) if len(b) >= 6 and b[6] == 0 and b[4].strip()
    ]
    columns = _page_columns(page) if render_vector else None
    results: List[ExtractedFigure] = []
    for label, caption_text, cap_bbox in captions:
        image_path = None
        match = _nearest_image_above(images, cap_bbox)
        if match is not None:
            image_path = _save_image(doc, match[0], fig_dir, page_no, label)
        # Vector figure fallback: render the region above the caption (figures only).
        if image_path is None and render_vector and label.startswith("Figure"):
            image_path = _render_figure_region(
                page, page_no, cap_bbox, fig_dir, label, raw_blocks, columns
            )
        results.append(
            ExtractedFigure(label=label, caption=caption_text, page=page_no, image_path=image_path)
        )
    return results


def _find_captions(page):
    out = []
    for b in page.get_text("blocks") or []:
        if len(b) < 6 or b[6] != 0:
            continue
        text = re.sub(r"\s*\n\s*", " ", b[4]).strip()
        m = _CAPTION_RE.match(text)
        if m:
            label = f"{'Table' if m.group(1).lower().startswith('tab') else 'Figure'} {m.group(2)}"
            out.append((label, text[:400], (b[0], b[1], b[2], b[3])))
    return out


def _find_images(page):
    out = []
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return out
    for info in infos:
        xref = info.get("xref", 0)
        if not xref:
            continue
        w = info.get("width", 0)
        h = info.get("height", 0)
        if w < _MIN_FIGURE_DIM or h < _MIN_FIGURE_DIM:
            continue
        out.append((xref, info.get("bbox", (0, 0, 0, 0))))
    return out


def _nearest_image_above(images, cap_bbox):
    """Closest raster whose bottom is above the caption AND shares its column."""
    cap_top = cap_bbox[1]
    candidates = [
        (xref, bbox)
        for (xref, bbox) in images
        if bbox[3] <= cap_top + 5 and _x_overlap(bbox, cap_bbox) > 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda ib: cap_top - ib[1][3])  # smallest gap above caption


def _x_overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _page_columns(page):
    """Detect 1- or 2-column layout; return list of (x0, x1) column bands."""
    page_rect = page.rect
    mid = (page_rect.x0 + page_rect.x1) / 2
    left = right = cross = 0
    for b in page.get_text("blocks") or []:
        if len(b) < 6 or b[6] != 0 or not b[4].strip():
            continue
        if b[2] <= mid + 5:
            left += 1
        elif b[0] >= mid - 5:
            right += 1
        else:
            cross += 1
    total = left + right + cross
    if total and left >= 3 and right >= 3 and cross <= 0.25 * total:
        gutter = 0.02 * page_rect.width
        return [(page_rect.x0, mid - gutter), (mid + gutter, page_rect.x1)]
    return [(page_rect.x0, page_rect.x1)]


def _band_for(columns, x: float):
    for band in columns:
        if band[0] <= x <= band[1]:
            return band
    return min(columns, key=lambda c: abs((c[0] + c[1]) / 2 - x))


def _render_figure_region(page, page_no, cap_bbox, fig_dir, label, raw_blocks, columns) -> Optional[str]:
    """Render the page region above a caption (captures vector figures)."""
    import fitz

    page_rect = page.rect
    cap_x0, cap_y0, cap_x1, _cap_y1 = cap_bbox
    bx0, bx1 = _band_for(columns, (cap_x0 + cap_x1) / 2)

    # Top boundary = bottom of the nearest text block above the caption in this column.
    band_box = (bx0, page_rect.y0, bx1, cap_y0)
    tops = [b[3] for b in raw_blocks if b[3] <= cap_y0 - 2 and _x_overlap(b, band_box) > 0]
    top = max(tops) if tops else page_rect.y0 + 4
    top = max(top, cap_y0 - 0.7 * page_rect.height, page_rect.y0)  # cap region height
    if cap_y0 - top < 30:
        return None

    rect = fitz.Rect(bx0, top, bx1, cap_y0 - 1) & page_rect
    if rect.is_empty or rect.height < 30 or rect.width < 40:
        return None
    try:
        pix = page.get_pixmap(clip=rect, dpi=150)
    except Exception:
        return None
    if pix.width < 40 or pix.height < 40:
        return None

    fig_dir.mkdir(parents=True, exist_ok=True)
    dest = fig_dir / f"p{page_no}_{label.lower().replace(' ', '')}_render.png"
    try:
        pix.save(str(dest))
    except Exception:
        return None
    return str(dest)


def _save_image(doc, xref: int, fig_dir: Path, page_no: int, label: str) -> Optional[str]:
    try:
        extracted = doc.extract_image(xref)
    except Exception:
        return None
    if not extracted or "image" not in extracted:
        return None
    fig_dir.mkdir(parents=True, exist_ok=True)
    ext = extracted.get("ext", "png")
    safe_label = label.lower().replace(" ", "")
    dest = fig_dir / f"p{page_no}_{safe_label}_x{xref}.{ext}"
    try:
        dest.write_bytes(extracted["image"])
    except OSError:
        return None
    return str(dest)
