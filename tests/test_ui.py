"""Streamlit GUI smoke test via AppTest (skipped if Streamlit isn't installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_UI = str(Path(__file__).resolve().parent.parent / "papermind" / "ui.py")


def test_ui_runs_without_error():
    at = AppTest.from_file(_UI, default_timeout=30).run()
    assert not at.exception  # the script executes cleanly with no paper selected
    assert [t.label for t in at.tabs] == ["🎯 分析", "📄 速读", "📊 对比", "🛠️ 复现", "🔎 搜索"]
    # Q&A lives in the sidebar, not a main tab.
    assert len(at.sidebar.text_input) >= 1
