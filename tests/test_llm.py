"""LLM client tests: usage accounting + streaming (litellm fully mocked)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from papermind.config import Config
from papermind.llm import base
from papermind.llm.base import OLLAMA_DEFAULT_MODEL, LLMClient, _chunk_delta, _usage_value
from papermind.output.schema import Usage


def test_usage_record_snapshot_minus():
    u = Usage()
    u.record(100, 20, 0.001)
    u.record(50, 10, 0.0005)
    assert u.calls == 2 and u.prompt_tokens == 150 and u.completion_tokens == 30
    assert u.total_tokens == 180
    snap = u.snapshot()
    u.record(10, 5, 0.0)
    delta = u.minus(snap)
    assert delta.calls == 1 and delta.prompt_tokens == 10 and delta.completion_tokens == 5


def test_usage_value_reads_dict_and_object():
    assert _usage_value({"usage": {"prompt_tokens": 7}}, "prompt_tokens") == 7
    assert _usage_value(NS(usage=NS(completion_tokens=3)), "completion_tokens") == 3
    assert _usage_value({"no_usage": 1}, "prompt_tokens") == 0


def test_chunk_delta_object_and_empty():
    chunk = NS(choices=[NS(delta=NS(content="hello"))], usage=None)
    assert _chunk_delta(chunk) == "hello"
    assert _chunk_delta(NS(choices=[], usage=NS())) == ""  # usage-only chunk


def _fake_litellm(stream_chunks=None, completion_content=None):
    def completion(**kwargs):
        if kwargs.get("stream"):
            return iter(stream_chunks)
        return {"choices": [{"message": {"content": completion_content}}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}}

    return NS(
        completion=completion,
        cost_per_token=lambda **k: (0.0, 0.0),
        token_counter=lambda **k: 0,
        drop_params=True,
    )


def test_non_stream_records_usage(monkeypatch):
    fake = _fake_litellm(completion_content='{"segments": []}')
    monkeypatch.setattr(base, "_import_litellm", lambda: fake)
    client = LLMClient(model="gpt-4o-mini")
    data = client.complete_json("sys", "user")
    assert data == {"segments": []}
    assert client.usage.calls == 1
    assert client.usage.prompt_tokens == 12 and client.usage.completion_tokens == 8


def test_stream_forwards_deltas_and_records_usage(monkeypatch):
    chunks = [
        NS(choices=[NS(delta=NS(content='{"seg'))], usage=None),
        NS(choices=[NS(delta=NS(content='ments": []}'))], usage=None),
        NS(choices=[], usage=NS(prompt_tokens=30, completion_tokens=9)),  # final usage chunk
    ]
    fake = _fake_litellm(stream_chunks=chunks)
    monkeypatch.setattr(base, "_import_litellm", lambda: fake)

    received = []
    client = LLMClient(model="gpt-4o-mini")
    data = client.complete_json_messages(
        [{"role": "user", "content": "q"}], on_delta=received.append
    )
    assert data == {"segments": []}                 # streamed JSON parsed
    assert "".join(received) == '{"segments": []}'  # deltas forwarded in order
    assert client.usage.prompt_tokens == 30 and client.usage.completion_tokens == 9


# --------------------------------------------------------------------------- #
# Free local path resolution
# --------------------------------------------------------------------------- #
def _no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_auto_local_when_no_key_and_ollama_up(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setattr(base, "ollama_available", lambda timeout=1.0: True)
    cfg = Config(openai_key=None, default_model="gpt-4o-mini")
    client = LLMClient(config=cfg)  # no explicit model
    assert client.model == OLLAMA_DEFAULT_MODEL
    assert cfg.embedding_provider == "local"  # embeddings auto-switched too


def test_keeps_hosted_when_key_present(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setattr(base, "ollama_available", lambda timeout=1.0: True)
    cfg = Config(openai_key="sk-test", default_model="gpt-4o-mini")
    client = LLMClient(config=cfg)
    assert client.model == "gpt-4o-mini"
    assert cfg.embedding_provider == "openai"


def test_no_key_no_ollama_keeps_hosted_model(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setattr(base, "ollama_available", lambda timeout=1.0: False)
    cfg = Config(openai_key=None, default_model="gpt-4o-mini")
    client = LLMClient(config=cfg)
    assert client.model == "gpt-4o-mini"  # left as-is; first call raises a clear error


def test_explicit_ollama_model_uses_local_embeddings(monkeypatch):
    _no_keys(monkeypatch)
    cfg = Config(openai_key=None, default_model="gpt-4o-mini")
    client = LLMClient(model="ollama/llama3.1", config=cfg)
    assert client.model == "ollama/llama3.1"
    assert cfg.embedding_provider == "local"


def test_deepseek_keeps_model_and_switches_embeddings(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(base, "ollama_available", lambda timeout=1.0: True)  # must NOT be used
    cfg = Config(deepseek_key="sk-x", default_model="deepseek/deepseek-chat")
    client = LLMClient(config=cfg)
    assert client.model == "deepseek/deepseek-chat"  # has deepseek key -> not swapped to ollama
    assert cfg.embedding_provider == "local"         # no openai key -> local embeddings for RAG


def test_apply_local_forces_whole_stack(monkeypatch, tmp_path):
    # Isolate config: empty PAPERMIND_HOME, no PAPERMIND_MODEL leaking in.
    monkeypatch.setenv("PAPERMIND_HOME", str(tmp_path))
    monkeypatch.delenv("PAPERMIND_MODEL", raising=False)
    from papermind.cli import _apply_local

    env = {}
    # local=False is a no-op and touches nothing.
    assert _apply_local(False, "gpt-4o", env) == "gpt-4o"
    assert env == {}

    # local=True forces model + embeddings local via env overrides.
    assert _apply_local(True, None, env) is None
    assert env["PAPERMIND_EMBEDDING_PROVIDER"] == "local"
    assert env["PAPERMIND_MODEL"].startswith("ollama/")

    # An explicit model with --local is honored as the local model.
    env2 = {}
    assert _apply_local(True, "ollama/mistral", env2) is None
    assert env2["PAPERMIND_MODEL"] == "ollama/mistral"
