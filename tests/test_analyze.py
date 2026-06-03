"""Analysis orchestration resilience: a single module failing (e.g. the model
returns JSON we can't repair) must not take down the whole report.

No network, no model — module runners are plain callables, one of which raises.
"""

from __future__ import annotations

from papermind.analyze import _run_modules
from papermind.errors import LLMError


def _advance_noop(name):
    pass


def _boom():
    raise LLMError("模型返回了无法修复的 JSON")


def test_run_modules_skips_a_failing_module_concurrent():
    notices = []
    runners = {
        "contributions": lambda: "C",
        "technical": _boom,          # this one fails
        "connections": lambda: "N",
        "reproduction": lambda: "R",
    }
    results = _run_modules(
        ["contributions", "technical", "connections", "reproduction"],
        runners, _advance_noop, notice=notices.append,
    )
    # the failing module is simply absent; the other three are still produced
    assert results == {"contributions": "C", "connections": "N", "reproduction": "R"}
    assert "technical" not in results
    assert any("技术细节" in n for n in notices)  # the user is told what was skipped


def test_run_modules_single_failing_module_serial_branch():
    # len(selected) <= 1 takes the serial path — it must degrade too.
    notices = []
    results = _run_modules(["technical"], {"technical": _boom}, _advance_noop, notice=notices.append)
    assert results == {}  # nothing crashed; section left at its default (empty)
    assert notices


def test_run_modules_advances_progress_even_for_failed_module():
    advanced = []
    runners = {"contributions": lambda: "C", "technical": _boom}
    _run_modules(["contributions", "technical"], runners, advanced.append, notice=lambda m: None)
    assert set(advanced) == {"contributions", "technical"}  # progress bar doesn't stall on failure


def test_run_modules_all_succeed_unchanged():
    runners = {"contributions": lambda: "C", "technical": lambda: "T"}
    results = _run_modules(["contributions", "technical"], runners, _advance_noop)
    assert results == {"contributions": "C", "technical": "T"}
