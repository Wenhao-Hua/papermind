"""The JSON parser must tolerate the malformed JSON LLMs emit on math-heavy papers:
raw LaTeX in string values (e.g. ``\\zeta``) where the lone backslash is an invalid
JSON escape that breaks ``json.loads``. _try_parse_json repairs these via json_repair.
"""

import json

from papermind.llm.base import _try_parse_json


def test_invalid_latex_escape_is_repaired():
    # A single backslash before "zeta" -> invalid JSON escape -> json.loads fails.
    raw = '{"name": "attention", "formula": "z = \\zeta(x)"}'
    import pytest

    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    out = _try_parse_json(raw)
    assert isinstance(out, dict)
    assert out["name"] == "attention"
    assert "zeta" in out["formula"]


def test_well_formed_json_unchanged():
    raw = '{"a": "line1\\nline2", "b": [1, 2, 3]}'
    assert _try_parse_json(raw) == json.loads(raw)


def test_fenced_array_still_parses():
    assert _try_parse_json('```json\n[{"x": 1}]\n```') == [{"x": 1}]
