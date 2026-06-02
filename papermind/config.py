"""Configuration: API keys, default model, cache locations.

Resolution order for any value is: explicit env var > ``~/.papermind/config.json``
> built-in default.  API keys found in the config file are also exported into the
process environment so that ``litellm`` (which reads ``OPENAI_API_KEY`` etc.)
picks them up transparently.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from papermind.errors import ConfigError

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
# Multilingual by default: PaperMind's analysis/questions are often Chinese while
# papers are English, so cross-lingual retrieval matters more than raw English score.
DEFAULT_EMBEDDING_MODEL_LOCAL = "paraphrase-multilingual-MiniLM-L12-v2"

# Map CLI-friendly keys (``config set openai-key``) to internal field names.
_CLI_KEY_ALIASES: Dict[str, str] = {
    "openai-key": "openai_key",
    "openai_key": "openai_key",
    "anthropic-key": "anthropic_key",
    "anthropic_key": "anthropic_key",
    "deepseek-key": "deepseek_key",
    "deepseek_key": "deepseek_key",
    "model": "default_model",
    "default-model": "default_model",
    "embedding-provider": "embedding_provider",
    "embedding-model": "embedding_model",
    "local-model": "local_model",
    "image-model": "image_model",
    "cache-dir": "cache_dir",
    "reranker": "reranker_path",
    "reranker-path": "reranker_path",
}

# Config field -> environment variable that overrides it.
_ENV_OVERRIDES: Dict[str, str] = {
    "openai_key": "OPENAI_API_KEY",
    "anthropic_key": "ANTHROPIC_API_KEY",
    "deepseek_key": "DEEPSEEK_API_KEY",
    "default_model": "PAPERMIND_MODEL",
    "embedding_provider": "PAPERMIND_EMBEDDING_PROVIDER",
    "embedding_model": "PAPERMIND_EMBEDDING_MODEL",
    "reranker_path": "PAPERMIND_RERANKER",
}

_SENSITIVE = {"openai_key", "anthropic_key", "deepseek_key"}


def home_dir() -> Path:
    """Root PaperMind directory (overridable via ``PAPERMIND_HOME``)."""
    return Path(os.environ.get("PAPERMIND_HOME", str(Path.home() / ".papermind")))


def config_path() -> Path:
    return home_dir() / "config.json"


@dataclass
class Config:
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    deepseek_key: Optional[str] = None
    default_model: str = DEFAULT_MODEL
    embedding_provider: str = "openai"  # "openai" | "local"
    embedding_model: Optional[str] = None
    local_model: Optional[str] = None  # model used by --local (default: ollama/llama3.1)
    image_model: Optional[str] = None  # image-gen model for --image-figures (e.g. gpt-image-1)
    cache_dir: Optional[str] = None
    reranker_path: Optional[str] = None  # trained cross-encoder checkpoint for Q&A reranking (off if None)

    # -- derived ----------------------------------------------------------- #
    def resolved_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        if self.embedding_provider == "local":
            return DEFAULT_EMBEDDING_MODEL_LOCAL
        return DEFAULT_EMBEDDING_MODEL_OPENAI

    def cache_root(self) -> Path:
        root = Path(self.cache_dir) if self.cache_dir else (home_dir() / "cache")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def paper_cache(self, key: str) -> Path:
        path = self.cache_root() / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_keys_to_env(self) -> None:
        """Push configured keys into the environment for litellm to consume."""
        if self.openai_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.openai_key
        if self.anthropic_key and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_key
        if self.deepseek_key and not os.environ.get("DEEPSEEK_API_KEY"):
            os.environ["DEEPSEEK_API_KEY"] = self.deepseek_key


def _read_file() -> Dict[str, object]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"Could not read config at {path}: {exc}") from exc


def load_config() -> Config:
    """Build a Config from file + environment overrides, then export keys to env."""
    data = _read_file()
    cfg = Config(
        openai_key=data.get("openai_key"),
        anthropic_key=data.get("anthropic_key"),
        deepseek_key=data.get("deepseek_key"),
        default_model=data.get("default_model", DEFAULT_MODEL),
        embedding_provider=data.get("embedding_provider", "openai"),
        embedding_model=data.get("embedding_model"),
        local_model=data.get("local_model"),
        image_model=data.get("image_model"),
        cache_dir=data.get("cache_dir"),
        reranker_path=data.get("reranker_path"),
    )
    # Environment overrides win.
    for field_name, env_var in _ENV_OVERRIDES.items():
        env_val = os.environ.get(env_var)
        if env_val:
            setattr(cfg, field_name, env_val)
    cfg.export_keys_to_env()
    return cfg


def set_value(cli_key: str, value: str) -> str:
    """Persist a single config value. Returns the internal field name set."""
    field_name = _CLI_KEY_ALIASES.get(cli_key)
    if field_name is None:
        valid = ", ".join(sorted(set(_CLI_KEY_ALIASES)))
        raise ConfigError(f"Unknown config key '{cli_key}'. Valid keys: {valid}")
    data = _read_file()
    data[field_name] = value
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return field_name


def masked_view() -> Dict[str, object]:
    """Config as a dict suitable for display, with secrets masked."""
    cfg = load_config()
    view = asdict(cfg)
    for key in _SENSITIVE:
        val = view.get(key)
        view[key] = _mask(val) if val else None
    view["embedding_model"] = cfg.resolved_embedding_model()
    view["config_path"] = str(config_path())
    view["cache_root"] = str(cfg.cache_root())
    return view


def _mask(secret: str) -> str:
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}...{secret[-4:]}"
