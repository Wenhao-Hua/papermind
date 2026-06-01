"""Small coercion helpers shared by the analysis modules.

LLM JSON is tolerated defensively: wrong types are coerced or dropped rather than
raising, so a single malformed field never sinks the whole report.
"""

from __future__ import annotations

from typing import List, Optional


def int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def str_or_none(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def str_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]
