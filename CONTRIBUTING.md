# Contributing to PaperMind

Thanks for your interest! PaperMind aims to stay small, dependency-light, and easy to read.

## Development setup

```bash
git clone https://github.com/Wenhao-Hua/papermind
cd papermind
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

Tests mock the LLM and avoid network, FAISS, and PyMuPDF, so they run fast and offline:

```bash
pytest
```

Please add a test alongside any new parsing rule, schema field, or answer-layering behavior.

## Architecture at a glance

```
papermind/
├── cli.py            # typer entry point (lazy-imports heavy work)
├── analyze.py        # orchestration: resolve → parse → modules → figures → Report
├── config.py         # keys / model / cache resolution (env > file > default)
├── parser/           # arxiv.py (download+meta), pdf.py (text+figures)
├── modules/          # contributions / technical / connections / reproduction
├── figures/          # extract.py (match originals), generate.py (Mermaid)
├── qa/               # index.py (chunk+FAISS), retriever.py, chat.py (PaperChat)
├── llm/              # base.py (litellm wrapper), prompts.py (all prompts)
└── output/           # schema.py (pydantic, single source of truth), markdown/json/terminal
```

## Guidelines

- **Keep dependencies minimal.** No LangChain or heavyweight frameworks. Heavy imports
  (`litellm`, `faiss`, `fitz`, `numpy`) must stay **lazy** so `papermind --help`/`config`
  work without them.
- **Prompts live in one place** — [`papermind/llm/prompts.py`](papermind/llm/prompts.py).
- **`output/schema.py` is the single source of truth.** Change it first, then renderers.
- **Parse LLM JSON defensively.** A single malformed field should never sink a whole report
  (see [`papermind/modules/_util.py`](papermind/modules/_util.py)).
- **Type-annotate public functions and all pydantic models.**
- Match the existing style; keep changes surgical.

## Submitting changes

1. Branch from `main`.
2. Add/update tests; run `pytest`.
3. Open a PR describing the change and its motivation.

By contributing you agree your work is licensed under the project's [MIT License](LICENSE).
