"""Analysis-result caching and cache inventory/management.

The parser already caches the PDF, metadata, and FAISS index per paper. This
module adds caching of the analysis :class:`Report` itself, keyed by
(model, modules, with_figures), so re-running ``analyze`` on the same paper is
instant and free unless ``--refresh`` is passed. It also exposes a small
inventory API for the ``papermind cache`` / ``papermind list`` commands.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from papermind.config import Config, load_config
from papermind.output.schema import Report


def report_cache_path(cache_dir: Path, model: str, modules: List[str], figures) -> Path:
    """Cache path keyed by model + modules + figure mode (bool or 'off'/'mermaid'/'image')."""
    key = f"{model}|{','.join(sorted(modules))}|fig={figures}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"report-{digest}.json"


def save_report(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")


def load_cached_report(path: Path) -> Optional[Report]:
    if not path.exists():
        return None
    try:
        return Report.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def latest_report_path(cache_dir: Path) -> Optional[Path]:
    reports = sorted(cache_dir.glob("report-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


# --------------------------------------------------------------------------- #
# Inventory / management
# --------------------------------------------------------------------------- #
@dataclass
class CachedPaper:
    key: str
    title: Optional[str]
    arxiv_id: Optional[str]
    has_report: bool
    has_index: bool
    size_bytes: int
    path: Path


def iter_cached_papers(config: Optional[Config] = None) -> List[CachedPaper]:
    config = config or load_config()
    root = config.cache_root()
    out: List[CachedPaper] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        title, arxiv_id = _read_meta(d)
        report_path = latest_report_path(d)
        has_report = report_path is not None
        if title is None and has_report:
            title = _title_from_report(report_path)
        out.append(
            CachedPaper(
                key=d.name,
                title=title,
                arxiv_id=arxiv_id,
                has_report=has_report,
                has_index=(d / "index.faiss").exists(),
                size_bytes=_dir_size(d),
                path=d,
            )
        )
    return out


def clear_paper(key: str, config: Optional[Config] = None) -> bool:
    root = (config or load_config()).cache_root()
    target = root / key
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        return True
    return False


def clear_all(config: Optional[Config] = None) -> int:
    root = (config or load_config()).cache_root()
    count = 0
    for d in [p for p in root.iterdir() if p.is_dir()]:
        shutil.rmtree(d, ignore_errors=True)
        count += 1
    return count


def _read_meta(d: Path):
    meta = d / "metadata.json"
    if not meta.exists():
        return None, None
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        return m.get("title"), m.get("arxiv_id")
    except (OSError, json.JSONDecodeError):
        return None, None


def _title_from_report(path: Path) -> Optional[str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("paper", {}).get("title")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
