# Changelog

All notable changes to PaperMind are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Web UI + report restyled into a warmer, more distinctive "research reading"
  look: an ink-green-teal accent (retiring the generic indigo), warm paper
  background, serif headings, a citation-spine + source-badge motif, and
  plainer-language copy throughout.

## [0.1.2] — 2026-06-20

### Added
- Whole-method **framework diagram** — a per-paper end-to-end architecture figure
  (paper-style SVG that also reconstructs steps the paper only implies, marked as
  inferred). On the web `/framework` page, downloadable as SVG.
- Broader paper sources beyond arXiv links / PDF URLs / local & uploaded PDFs: a
  **DOI** (bare, `doi:`, or `doi.org`), a free-text **paper title** (via OpenAlex
  then arXiv), and any academic **landing page** (via its `citation_pdf_url`).
- **Gemini** (chat + embeddings) and **Qwen (Aliyun Bailian / DashScope)** as
  providers — `papermind config set qwen-key …`, then any `dashscope/…` model
  (e.g. `figure-model dashscope/qwen3-max`).

### Changed
- **Web UI is now Chinese-only**, with a denser, redesigned landing: a two-column
  hero with the analyze tool inline, a compact capability grid, and an evidence
  section led by the reranker benchmark. (The earlier English-default + 中文 toggle
  was removed.)
- Restyled the shell + report into one modern, light "dev-tool" design system —
  system sans, near-white + indigo, soft elevation, dark mode, mobile-friendly.
- **README rewritten as a single concise Chinese `README.md`** (the bilingual
  `README.zh-CN.md` was removed).
- **Published on PyPI as `papermind-ai`** — `pip install papermind-ai` (the import
  package, CLI command, and repo stay `papermind`).
- Title search resolves through OpenAlex (keyless) instead of the rate-limited arXiv
  search API.

### Fixed
- Structured-analysis modules (esp. **technical detail**) no longer intermittently
  fail on math-heavy papers: models emit raw LaTeX in JSON string values (`\frac`,
  `\zeta`, `\mathbb`, …) where a lone backslash is an invalid JSON escape that broke
  `json.loads` even after the repair retry — dropping the whole module and its
  figures. JSON parsing now falls back to `json_repair`, preserving the content.
- Teaching SVG figures no longer show raw LaTeX or mis-rendered sub/superscripts;
  multi-line report formulas render under MathJax; fixed several gallery formulas.
- The offline demo report no longer renders a "Syntax error · mermaid" box.
- A corrupt/partial `metadata.json` no longer crashes subsequent analyses of that
  paper (guarded read + atomic write).
- Uploaded PDFs are content-addressed in the cache instead of leaking a temp file.
- Title search degrades (PDF → DOI → arXiv) instead of hard-failing on a dead link.
- Live web routes refund the per-IP rate-limit slot when a request fails.

### Internal
- CI runs the FastAPI web suite and a ruff (pyflakes) lint gate across Python 3.9–3.12.

## [0.1.1] — 2026-06-20
- Same-day interim publish: first PyPI release as `papermind-ai`, plus the
  `json_repair` JSON-parse fix; superseded by 0.1.2.

## [0.1.0]
- Initial release: 4-module structured analysis, citation-verified grounded Q&A, a
  trained cross-encoder retrieval reranker, a reproduction guide built from the
  paper's real code repo, teaching SVG figures, and a CLI + web UI.
