"""JSON export of a Report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def to_json(report, path: Optional[str] = None) -> str:
    text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if path:
        out = Path(path)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text
