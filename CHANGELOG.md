# Changelog

All notable changes to PaperMind are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [0.1.4] — 2026-06-29

### Added
- **CSV export for `compare`** — `papermind compare --format csv` (and the
  `Comparison.to_csv` API) writes the side-by-side table as UTF-8 CSV.
- **Reading-time estimate** in the report meta line ("预计阅读 N 分钟"), counted
  CJK-aware (each Chinese character/punctuation is a reading unit).

### Fixed
- **Security — escape Mermaid figure bodies** in the served report: a
  model-generated diagram could otherwise break out of `<pre>` and inject script
  (stored XSS in `papermind serve` / shared report HTML).
- **Security — redact Google/Gemini keys** (`AIza…`) and `?key=` params in
  user-facing error messages (only `sk-`/`Bearer` were caught before).
- **Security — neutralise CSV formula injection** in the compare export: a
  paper-derived cell starting with `= + - @` is now prefixed so spreadsheets
  don't evaluate it.
- **`_record_usage` thread-safety** — it no longer triggers litellm's first
  import from a worker thread, where a native (pyo3) panic could kill the thread
  and drop usage accounting.

### Internal
- Removed dead hero-image loaders; deduped the reading-time helper into
  `papermind.output.reading`; added `.dockerignore`; refreshed stale docs/comments.

## [0.1.3] — 2026-06-20

### Changed
- **Web UI rebuilt as a focused tool, not a marketing page.** The home is the
  analyze function itself, sharing one consistent focused-tool layout — dropping
  the hero pitch, capability cards, benchmark section, and gallery.
- **Q&A folded into 分析.** The analyze page now carries a dedicated "问 AI" block
  that reuses the paper you already entered — no second paper selection.
- **The framework diagram now ships inside the analysis report** (a 方法框架 section
  with the whole-method end-to-end diagram), generated alongside the figures — so
  it's part of every `papermind analyze` (CLI + web), not a separate tab. The web
  nav is trimmed to 分析 · 搜索.
- **Removed the redundant web tools** 速读 / 对比 / 复现 (their output is already
  part of a full analysis). The CLI commands are unchanged.
- **Restyled into a clean, modern light-product look** across every surface (home,
  tools, result/state pages, and the report): cool-neutral palette on a soft canvas,
  a single blue accent, system sans throughout (no serif), the home grounded in one
  focused card, and the framework diagram recolored to match.
- Dropped the footer model badge.

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
