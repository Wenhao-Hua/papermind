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

The web demo (`test_web.py`) needs the `web` extra — `pip install -e ".[dev,web]"`.

## Running against a model

Every provider is routed through `litellm`, so they're config, not code. Set a key
(env var, or `papermind config set <name>-key ...`) and pick a model:

| Provider | Key | Example model |
| --- | --- | --- |
| OpenAI (default) | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-latest` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| Gemini | `GEMINI_API_KEY` | `gemini/gemini-2.5-flash` |
| Local / free | — | `--local` (Ollama) |

Override the model with `--model` / `PAPERMIND_MODEL` and embeddings with
`PAPERMIND_EMBEDDING_MODEL`. `papermind demo` runs fully offline with no key.

## Architecture at a glance

```
papermind/
├── cli.py            # typer entry point (lazy-imports heavy work)
├── analyze.py        # orchestration: resolve → parse → modules → figures → Report
├── config.py         # keys / model / cache resolution (env > file > default)
├── parser/           # arxiv.py (download+meta), pdf.py (text+figures)
├── modules/          # contributions / technical / connections / reproduction
├── figures/          # extract.py (match originals), generate.py (teaching SVGs; legacy Mermaid), framework.py (whole-method diagram: spec + deterministic SVG renderer)
├── qa/               # index.py (chunk+FAISS), retriever.py, chat.py (PaperChat)
├── repro/            # repo.py (locate & verify the paper's official code repo)
├── rerank/           # infer.py (trained cross-encoder reranker inference)
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
