# Changelog

All notable changes to PaperMind are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Bilingual web UI** — English by default with a 中文 toggle (resolved per request
  via `?lang=`, a cookie, or `Accept-Language`).
- Whole-method **framework diagram** — a per-paper end-to-end architecture figure
  (paper-style SVG that also reconstructs steps the paper only implies, marked as
  inferred). Available on the web `/framework` page and downloadable as SVG.
- Broader paper sources, in addition to arXiv links / PDF URLs / local & uploaded
  PDFs: a **DOI** (bare, `doi:`, or `doi.org`), a free-text **paper title** (resolved
  via OpenAlex, then arXiv), and any academic **landing page** (resolved via its
  `citation_pdf_url` meta tag — OpenReview, bioRxiv, ACL, PMLR, …).
- **Gemini** as a first-class provider (chat + embeddings), alongside OpenAI,
  Anthropic, and DeepSeek.

### Changed
- **Redesigned the web UI** (landing shell + report page) into one modern, light
  "dev-tool" design system — system sans (no serif), near-white + indigo palette,
  sticky top bar, dark mode, mobile-friendly — replacing the previous serif/beige,
  emoji-heavy styling.
- Title search resolves through OpenAlex (keyless, reliable) instead of the
  rate-limited arXiv search API.

### Fixed
- Teaching SVG figures no longer show raw LaTeX (`^{}`, `_{}`, `\frac`, …) or
  mis-rendered sub/superscripts, and multi-line report formulas render under MathJax
  (wrapped in `aligned`); fixed several incorrect formulas in the example gallery.
- The offline demo report no longer renders a "Syntax error · mermaid" box — its
  figure is now a real teaching SVG.
- A corrupt/partial `metadata.json` no longer crashes every subsequent analysis of
  that paper (guarded read + atomic write).
- Uploaded PDFs are content-addressed in the cache instead of leaking one temp file
  per upload.
- Title search degrades (PDF → DOI → arXiv) instead of hard-failing on a dead link.
- Live web routes refund the per-IP rate-limit slot when a request fails.

### Internal
- CI runs the FastAPI web suite and a ruff (pyflakes) lint gate across Python 3.9–3.12.

## [0.1.0]
- Initial release: 4-module structured analysis, citation-verified grounded Q&A, a
  trained cross-encoder retrieval reranker, a reproduction guide built from the
  paper's real code repo, teaching SVG figures, and a CLI + web UI.
