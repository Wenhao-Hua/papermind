"""Unified LLM access via litellm.

One thin client works across OpenAI / Anthropic / Ollama (anything litellm
routes).  Provides plain text completion, robust JSON completion (with a single
repair retry), and embeddings (OpenAI by default, optional local
sentence-transformers).
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import TYPE_CHECKING, Callable, List, Optional

from papermind.config import Config, load_config
from papermind.errors import LLMError
from papermind.output.schema import Usage

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

OnDelta = Optional[Callable[[str], None]]
OnNotice = Optional[Callable[[str], None]]

OLLAMA_DEFAULT_MODEL = "ollama/llama3.1"
_KEYLESS_PREFIXES = ("ollama/", "ollama_chat/", "local/")

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class LLMClient:
    """Stateless-ish wrapper around litellm bound to a model + config."""

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[Config] = None,
        temperature: float = 0.2,
        on_notice: OnNotice = None,
    ) -> None:
        self.config = config or load_config()
        self.temperature = temperature
        self.usage = Usage()  # accumulated across all calls on this client
        self._usage_lock = threading.Lock()  # usage is updated from parallel workers

        # Free local path: with no API key but Ollama running, default to a local model.
        if model is None:
            self.model = _resolve_default_model(self.config, on_notice)
        else:
            self.model = model
        # Non-OpenAI LLM + no OpenAI key -> OpenAI embeddings can't work, use local.
        # (Silent: only the Q&A family actually needs embeddings; if local deps are
        # missing, embed() raises a clear, actionable error at that point.)
        if not _is_openai_model(self.model) and not _has_openai_key(self.config):
            if self.config.embedding_provider == "openai" and not self.config.embedding_model:
                self.config.embedding_provider = "local"

    # -- preflight ---------------------------------------------------------- #
    def ensure_ready(self) -> None:
        """Fail fast — before any download / parse / indexing — if this client's
        model cannot run, with a clear, actionable message instead of a confusing
        provider error 40 seconds in."""
        if not _needs_key(self.model):  # ollama / local model
            if not ollama_available():
                name = self.model.split("/", 1)[-1]
                raise LLMError(
                    f"本地模式需要 Ollama，但没有检测到在运行的服务（模型 {self.model}）。\n"
                    "  1) 安装：https://ollama.com\n"
                    f"  2) 拉取模型：ollama pull {name}\n"
                    "  3) 启动 Ollama 后重试。\n"
                    "想零配置先体验，可看离线示例：papermind demo"
                )
            return
        if _has_key_for(self.model, self.config):
            return
        raise LLMError(
            f"模型 {self.model!r} 需要 API key，但没检测到，本地 Ollama 也不可用。三选一：\n"
            "  A) 配置 key：papermind config set deepseek-key sk-...（或 openai-key / anthropic-key）\n"
            "  B) 本地免费跑：装 Ollama(https://ollama.com) 后加 --local，例：papermind analyze <paper> --local\n"
            "  C) 零配置先看离线示例：papermind demo"
        )

    # -- text --------------------------------------------------------------- #
    def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._call(
            messages, temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort
        )

    def complete_messages(
        self, messages: List[dict], temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        return self._call(messages, temperature=temperature, max_tokens=max_tokens)

    # -- json --------------------------------------------------------------- #
    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Return parsed JSON (dict or list). Retries once to repair invalid JSON."""
        return self.complete_json_messages(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def complete_json_messages(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        on_delta: OnDelta = None,
        reasoning_effort: Optional[str] = None,
    ):
        """JSON completion from a full message list (for multi-turn chat).

        If ``on_delta`` is given, the first attempt is streamed (deltas are
        forwarded as they arrive); the repair attempt is always non-streamed.
        """
        raw = self._call(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=True, on_delta=on_delta,
            reasoning_effort=reasoning_effort,
        )
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed

        repair = self._call(
            [
                {"role": "system", "content": "You output only valid JSON. No prose, no code fences."},
                {
                    "role": "user",
                    "content": (
                        "The following was supposed to be valid JSON but could not be parsed. "
                        "Return the corrected JSON only:\n\n" + raw
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
            reasoning_effort=reasoning_effort,
        )
        parsed = _try_parse_json(repair)
        if parsed is None:
            raise LLMError(
                f"Model {self.model!r} did not return valid JSON after a repair attempt."
            )
        return parsed

    # -- image generation --------------------------------------------------- #
    def image(self, prompt: str, model: str, size: str = "1024x1024") -> bytes:
        """Generate an image via litellm and return raw PNG/JPEG bytes."""
        import base64

        litellm = _import_litellm()
        try:
            resp = litellm.image_generation(model=model, prompt=prompt, size=size)
        except Exception as exc:  # noqa: BLE001
            raise _wrap_llm_error(exc, model) from exc
        try:
            item = resp["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected image response from {model!r}: {exc}") from exc

        b64 = item.get("b64_json") if isinstance(item, dict) else getattr(item, "b64_json", None)
        if b64:
            return base64.b64decode(b64)
        url = item.get("url") if isinstance(item, dict) else getattr(item, "url", None)
        if url:
            import httpx

            from papermind.net import MAX_DOWNLOAD_BYTES, BlockedURLError, safe_stream

            try:
                with safe_stream("GET", url, timeout=60.0) as r:
                    r.raise_for_status()
                    data = b""
                    for chunk in r.iter_bytes(chunk_size=65536):
                        data += chunk
                        if len(data) > MAX_DOWNLOAD_BYTES:
                            raise LLMError("Generated image is too large to download.")
                    return data
            except BlockedURLError as exc:
                raise LLMError(f"Refused to download image from a blocked URL: {exc}") from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Failed to download generated image: {exc}") from exc
        raise LLMError(f"Image response from {model!r} had neither b64_json nor url.")

    # -- embeddings --------------------------------------------------------- #
    def embed(self, texts: List[str]) -> "np.ndarray":
        import numpy as np

        if not texts:
            return np.zeros((0, 1), dtype="float32")
        if self.config.embedding_provider == "local":
            return self._embed_local(texts)
        return self._embed_openai(texts)

    def _embed_openai(self, texts: List[str]) -> "np.ndarray":
        import numpy as np

        litellm = _import_litellm()
        model = self.config.resolved_embedding_model()
        try:
            resp = litellm.embedding(model=model, input=texts)
        except Exception as exc:  # noqa: BLE001 - litellm raises provider-specific errors
            raise _wrap_llm_error(exc, model) from exc
        self._record_usage(model, _usage_value(resp, "prompt_tokens"), 0)
        vectors = [item["embedding"] for item in resp["data"]]
        return np.asarray(vectors, dtype="float32")

    def _embed_local(self, texts: List[str]) -> "np.ndarray":
        import numpy as np

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LLMError(
                "Local embeddings need sentence-transformers. Install with: "
                "pip install 'papermind-ai[local-embeddings]'"
            ) from exc
        model_name = self.config.resolved_embedding_model()
        encoder = _get_local_encoder(model_name, SentenceTransformer)
        vectors = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype="float32")

    # -- internals ---------------------------------------------------------- #
    def _call(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        on_delta: OnDelta = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        litellm = _import_litellm()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort:
            # Drawing needs little chain-of-thought; lowering reasoning frees the
            # token budget for the (large) SVG body. Passed through as-is (NOT via
            # drop_params, which would strip it for models litellm doesn't recognize
            # — then a thinking model burns the whole budget and returns nothing).
            # A model that genuinely rejects it surfaces as a caught LLMError (no figure).
            kwargs["reasoning_effort"] = reasoning_effort
        if json_mode and _supports_json_mode(self.model):
            kwargs["response_format"] = {"type": "json_object"}

        if on_delta is not None:
            return self._call_stream(litellm, kwargs, json_mode, on_delta)

        try:
            resp = litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Retry once after dropping params some providers reject: reasoning_effort
            # (a non-reasoning model 400s on it) and/or a json response_format.
            retry = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "response_format")}
            if len(retry) == len(kwargs):  # nothing droppable -> the error is real
                raise _wrap_llm_error(exc, self.model) from exc
            try:
                resp = litellm.completion(**retry)
            except Exception as exc2:  # noqa: BLE001
                raise _wrap_llm_error(exc2, self.model) from exc2

        self._record_usage(
            self.model, _usage_value(resp, "prompt_tokens"), _usage_value(resp, "completion_tokens")
        )
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from {self.model!r}: {exc}") from exc

    def _call_stream(self, litellm, kwargs: dict, json_mode: bool, on_delta: Callable[[str], None]) -> str:
        """Stream a completion, forwarding text deltas and accumulating usage."""
        stream_kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        parts: List[str] = []
        usage_obj = None
        try:
            for chunk in litellm.completion(**stream_kwargs):
                found = _chunk_usage(chunk)
                if found is not None:
                    usage_obj = found
                delta = _chunk_delta(chunk)
                if delta:
                    parts.append(delta)
                    on_delta(delta)
        except Exception as exc:  # noqa: BLE001
            if json_mode and "response_format" in stream_kwargs:
                # Provider rejected json response_format under streaming; fall back — but keep
                # reasoning_effort, else a thinking model may burn the whole budget reasoning.
                return self._call(
                    kwargs["messages"], kwargs.get("temperature"), kwargs.get("max_tokens"),
                    json_mode=False, reasoning_effort=kwargs.get("reasoning_effort"),
                )
            raise _wrap_llm_error(exc, self.model) from exc

        text = "".join(parts)
        if usage_obj is not None:
            pt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        else:  # provider didn't return usage in the stream; estimate
            pt = _count_tokens(litellm, self.model, messages=kwargs["messages"])
            ct = _count_tokens(litellm, self.model, text=text)
        self._record_usage(self.model, pt, ct)
        return text

    def _record_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        cost = 0.0
        try:
            litellm = _import_litellm()
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
            cost = float(prompt_cost) + float(completion_cost)
        except (KeyboardInterrupt, SystemExit):  # never swallow control signals
            raise
        except BaseException:  # noqa: BLE001 - cost is best-effort; pyo3 panics are BaseException
            cost = 0.0
        with self._usage_lock:
            self.usage.record(prompt_tokens, completion_tokens, cost)


# --------------------------------------------------------------------------- #
# Model resolution (free local path)
# --------------------------------------------------------------------------- #
def _resolve_default_model(config: Config, on_notice: OnNotice = None) -> str:
    """When no model is specified and the configured one needs a key we don't have,
    fall back to a local Ollama model if Ollama is reachable."""
    model = config.default_model
    if not _needs_key(model) or _has_key_for(model, config):
        return model
    if ollama_available():
        if on_notice:
            on_notice(
                f"未检测到 API key，已自动改用本地模型 {OLLAMA_DEFAULT_MODEL}（Ollama）。"
                "可用 --model 覆盖。"
            )
        return OLLAMA_DEFAULT_MODEL
    return model  # no key, no Ollama: the first call raises a clear, actionable error


def _needs_key(model: str) -> bool:
    return not model.lower().startswith(_KEYLESS_PREFIXES)


def _has_key_for(model: str, config: Config) -> bool:
    m = model.lower()
    if "claude" in m or m.startswith("anthropic/"):
        return bool(config.anthropic_key or os.environ.get("ANTHROPIC_API_KEY"))
    if "deepseek" in m or m.startswith("deepseek/"):
        return bool(config.deepseek_key or os.environ.get("DEEPSEEK_API_KEY"))
    if "gemini" in m:  # litellm routes the gemini/ prefix via GEMINI_API_KEY
        return bool(config.gemini_key or os.environ.get("GEMINI_API_KEY"))
    if "qwen" in m or m.startswith("dashscope/"):  # Aliyun Bailian / DashScope
        return bool(config.dashscope_key or os.environ.get("DASHSCOPE_API_KEY"))
    return bool(config.openai_key or os.environ.get("OPENAI_API_KEY"))


def _has_openai_key(config: Config) -> bool:
    return bool(config.openai_key or os.environ.get("OPENAI_API_KEY"))


def _is_openai_model(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt", "o1", "o3", "openai/", "chatgpt"))


def ollama_available(timeout: float = 1.0) -> bool:
    """Best-effort check that a local Ollama server is reachable."""
    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        import httpx

        return httpx.get(f"{host}/api/tags", timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #
def _usage_value(resp, field: str) -> int:
    """Read a token count from a litellm response's usage, tolerating shapes."""
    try:
        usage = resp["usage"] if isinstance(resp, dict) else getattr(resp, "usage", None)
        if usage is None:
            return 0
        value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, 0)
        return int(value or 0)
    except (KeyError, AttributeError, TypeError, ValueError):
        return 0


def _chunk_usage(chunk):
    return getattr(chunk, "usage", None) or (chunk.get("usage") if isinstance(chunk, dict) else None)


def _chunk_delta(chunk) -> str:
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError):
        try:
            return chunk["choices"][0]["delta"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""


def _count_tokens(litellm, model: str, messages=None, text: str = "") -> int:
    try:
        if messages is not None:
            return int(litellm.token_counter(model=model, messages=messages))
        return int(litellm.token_counter(model=model, text=text))
    except Exception:  # noqa: BLE001
        return 0


_LOCAL_ENCODERS: dict = {}


def _get_local_encoder(model_name: str, cls):
    if model_name not in _LOCAL_ENCODERS:
        _LOCAL_ENCODERS[model_name] = cls(model_name)
    return _LOCAL_ENCODERS[model_name]


_litellm_import_lock = threading.Lock()
_litellm_module = None
_litellm_import_failed = False  # set on first BaseException to skip future attempts


def _import_litellm():
    global _litellm_module, _litellm_import_failed
    if _litellm_module is not None:
        return _litellm_module
    if _litellm_import_failed:
        raise LLMError("litellm is required. Install with: pip install litellm")
    with _litellm_import_lock:
        if _litellm_module is not None:
            return _litellm_module
        if _litellm_import_failed:
            raise LLMError("litellm is required. Install with: pip install litellm")
        try:
            import litellm

            litellm.drop_params = True  # silently drop unsupported params per provider
            _litellm_module = litellm
            return litellm
        except ImportError as exc:  # pragma: no cover
            raise LLMError("litellm is required. Install with: pip install litellm") from exc
        except BaseException:
            _litellm_import_failed = True
            raise


def _supports_json_mode(model: str) -> bool:
    m = model.lower()
    return any(tag in m for tag in ("gpt-4", "gpt-3.5", "gpt-4o", "o1", "o3", "gpt-5", "deepseek"))


def _try_parse_json(text: str):
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    for candidate in (text, _extract(text, _JSON_OBJECT_RE), _extract(text, _JSON_ARRAY_RE)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # Fallback: repair malformed LLM JSON. Math-heavy papers make models emit raw
    # LaTeX in string values ("\frac", "\zeta", "\mathbb"), where the lone backslash
    # is an invalid JSON escape that breaks json.loads; json_repair fixes these (plus
    # trailing commas, unquoted keys, …) while preserving the content.
    try:
        import json_repair

        repaired = json_repair.loads(text)
        if isinstance(repaired, (dict, list)) and repaired:
            return repaired
    except Exception:  # noqa: BLE001 - repair is best-effort; never raise here
        pass
    return None


def _extract(text: str, pattern: re.Pattern) -> Optional[str]:
    m = pattern.search(text)
    return m.group(0) if m else None


# Some providers echo the offending API key / bearer token in an auth error; never
# let that reach a user-facing message (these errors are shown on the public web too).
_SECRET_RE = re.compile(r"\b(?:sk|pk|rk)[-_][A-Za-z0-9]{12,}\b|\bBearer\s+[A-Za-z0-9._-]{12,}\b", re.IGNORECASE)


def _redact_secrets(s: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", s)


def _wrap_llm_error(exc: Exception, model: str) -> LLMError:
    msg = _redact_secrets(str(exc))
    low = msg.lower()
    if model.lower().startswith(_KEYLESS_PREFIXES):  # local / Ollama
        if any(s in low for s in ("connection", "connect", "refused", "max retries", "timed out", "timeout")):
            return LLMError(
                f"无法连接本地 Ollama（模型 {model}）。请确认已安装并在运行：\n"
                "  安装 https://ollama.com，启动后重试（默认端口 11434）。\n"
                f"原始错误：{msg}"
            )
        if any(s in low for s in ("not found", "does not exist", "try pulling", "no such model")):
            name = model.split("/", 1)[-1]
            return LLMError(f"本地模型未下载：{model}。先执行  ollama pull {name}  再重试。\n原始错误：{msg}")
    if "api key" in low or "authentication" in low or "no api_key" in low or "openai_api_key" in low:
        return LLMError(
            f"模型 {model!r} 鉴权失败。\n"
            f"方式 A —— 配置 key：  papermind config set openai-key sk-...\n"
            f"           （或 export OPENAI_API_KEY / ANTHROPIC_API_KEY）\n"
            f"方式 B —— 用 Ollama 全本地免费运行（无需 key）：\n"
            f"           从 https://ollama.com 安装后：\n"
            f"           papermind analyze <paper> --model {OLLAMA_DEFAULT_MODEL}\n"
            f"           （本地向量：pip install 'papermind-ai[local-embeddings]'）\n"
            f"原始错误：{msg}"
        )
    if "model" in low and ("not found" in low or "does not exist" in low):
        return LLMError(f"模型 {model!r} 不可用：{msg}")
    return LLMError(f"模型 {model!r} 调用失败：{msg}")
