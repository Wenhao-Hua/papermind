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


def test_analyze_attaches_framework_to_report(monkeypatch, tmp_path):
    """The figures step generates the whole-method framework and stores it on the Report
    (no separate tab). Mocks isolate the figures wiring from network/model."""
    import types

    import papermind.analyze as am
    import papermind.figures.framework as fw_mod
    from papermind.config import Config
    from papermind.figures.framework import FNode, FrameworkSpec, _auto_layout
    from papermind.output.schema import PaperMeta
    from papermind.parser.arxiv import ResolvedSource

    cache_dir = tmp_path / "p"
    cache_dir.mkdir()
    meta = PaperMeta(title="P", arxiv_id="x")
    resolved = ResolvedSource(meta=meta, pdf_path=cache_dir / "paper.pdf", cache_key="x", cache_dir=cache_dir)

    class _FakeClient:
        def __init__(self, *a, **k):
            self.model = "gpt-4o-mini"
            self.usage = None

        def ensure_ready(self):
            pass

    spec = _auto_layout(FrameworkSpec(title="FW", nodes=[FNode(id="a", label="输入"), FNode(id="b", label="阶段")]))
    monkeypatch.setattr(am, "resolve", lambda source, config: resolved)
    monkeypatch.setattr(am, "parse_pdf", lambda *a, **k: types.SimpleNamespace(meta=meta, cache_dir=cache_dir))
    monkeypatch.setattr(am, "_build_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(am, "_run_modules", lambda *a, **k: {})  # isolate the figures step from the 4 modules
    monkeypatch.setattr(am, "_verify_citations", lambda *a, **k: None)
    monkeypatch.setattr(am, "LLMClient", _FakeClient)
    monkeypatch.setattr(fw_mod, "generate_framework", lambda context, client, on_notice=None: spec)

    cfg = Config(openai_key="sk-x", default_model="gpt-4o-mini")
    report = am.analyze("x", model="gpt-4o-mini", config=cfg, with_figures=True, refresh=True)
    assert report.framework is not None and report.framework.title == "FW"
    assert 'id="framework"' in report.to_html()  # and it renders into the report

    # with figures off, no framework call is made and the report carries none
    monkeypatch.setattr(fw_mod, "generate_framework", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    report2 = am.analyze("x", model="gpt-4o-mini", config=cfg, with_figures=False, refresh=True)
    assert report2.framework is None
