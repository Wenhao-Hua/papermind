# PaperMind web service — public-safe by default (demo mode: cached reports only).
# Build:  docker build -t papermind .
# Run:    docker run -p 8080:8080 papermind
# Live:   docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... papermind \
#                papermind serve --host 0.0.0.0 --port 8080 --live
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# faiss-cpu / pymupdf / litellm ship manylinux wheels, so no build toolchain needed.
RUN pip install --no-cache-dir ".[web]"

EXPOSE 8080
# Demo mode by default: serves only already-cached reports, makes no model calls,
# so exposing it publicly won't burn an API key. Add `--live` (and a key) for real analysis.
CMD ["papermind", "serve", "--host", "0.0.0.0", "--port", "8080"]
