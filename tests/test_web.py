"""Web demo tests (skipped if FastAPI isn't installed)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import papermind.parser.arxiv as arxiv_mod  # noqa: E402
from papermind.parser.arxiv import ResolvedSource  # noqa: E402
from papermind.web import create_app  # noqa: E402


def test_healthz_reports_mode():
    client = TestClient(create_app(live=False))
    body = client.get("/healthz").json()
    assert body["ok"] is True and body["live"] is False
    assert TestClient(create_app(live=True)).get("/healthz").json()["live"] is True


def test_index_has_form_and_active_model():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "PaperMind" in r.text and "<form" in r.text
    # No model picker — the active model is shown subtly in the footer instead.
    assert "<select name='model'" not in r.text
    assert "当前模型" in r.text  # active model shown in the footer


def test_web_is_chinese_only():
    app = create_app()
    # Chinese regardless of Accept-Language; no English UI, no language toggle.
    for al in ("en-US,en", "zh-CN,zh"):
        t = TestClient(app).get("/", headers={"accept-language": al}).text
        assert "能复现" in t and "Analyze a paper" not in t and "Read any paper" not in t
    assert "pm-lang" not in TestClient(app).get("/").text  # language toggle removed


def test_framework_tab_and_form():
    client = TestClient(create_app(live=False))
    r = client.get("/framework")
    assert r.status_code == 200 and "生成框架图" in r.text
    assert "/framework" in client.get("/").text  # nav tab present


def test_framework_body_is_view_only_with_download():
    from papermind.figures.framework import FNode, FrameworkSpec, _auto_layout
    from papermind.web import _framework_body

    spec = _auto_layout(FrameworkSpec(title="T", nodes=[FNode(id="a", label="输入"), FNode(id="b", label="阶段")]))
    html = _framework_body(spec)
    assert "<svg" in html and "输入" in html                    # the diagram is rendered
    assert "data-fw='download'" in html and "下载 SVG" in html   # the only action: download the SVG
    assert "/framework/edit" not in html and "fw-save" not in html  # no editing affordances


def test_framework_demo_route_renders_cached_spec(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERMIND_HOME", str(tmp_path))
    from papermind.config import load_config
    from papermind.figures.framework import FNode, FrameworkSpec, _auto_layout, save_framework_spec

    cache_dir = load_config().paper_cache("1706.03762")  # cache key for the arXiv URL below
    save_framework_spec(cache_dir, _auto_layout(FrameworkSpec(title="Attn", nodes=[FNode(id="a", label="输入序列")])))
    r = TestClient(create_app(live=False)).post("/framework", data={"source": "https://arxiv.org/abs/1706.03762"})
    assert r.status_code == 200 and "<svg" in r.text and "输入序列" in r.text and "data-fw='download'" in r.text


def test_framework_demo_route_uncached_shows_note(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERMIND_HOME", str(tmp_path))
    r = TestClient(create_app(live=False)).post("/framework", data={"source": "https://arxiv.org/abs/2401.00001"})
    assert r.status_code == 200 and "演示模式" in r.text  # no cached spec -> the demo note, not a crash


def test_demo_route_renders_offline_report():
    r = TestClient(create_app()).get("/demo")
    assert r.status_code == 200
    assert "Attention Is All You Need" in r.text  # the bundled demo paper
    assert "MathJax" in r.text                      # full report rendering


def test_all_feature_tabs_present():
    html = TestClient(create_app()).get("/").text
    for href in ("/ask", "/summary", "/compare", "/reproduce", "/search"):
        assert f"href='{href}'" in html


def test_ask_blocked_in_demo_mode():
    client = TestClient(create_app(live=False))
    r = client.post("/ask", data={"source": "1706.03762", "question": "why?", "model": "", "mode": "balanced"})
    assert r.status_code == 200 and "演示模式" in r.text


def test_search_works_without_model(monkeypatch):
    import papermind.parser.arxiv as arxiv_mod
    from papermind.output.schema import PaperMeta

    monkeypatch.setattr(arxiv_mod, "search_arxiv", lambda q, max_results=12: [
        PaperMeta(title="FlashAttention-2", arxiv_id="2307.08691", year=2023)
    ])
    r = TestClient(create_app(live=False)).post("/search", data={"query": "attention"})
    assert r.status_code == 200
    assert "2307.08691" in r.text and "/analyze?source=" in r.text  # links to analyze (via the paper URL)


def test_analyze_demo_mode_does_not_run_when_uncached(monkeypatch, tmp_path):
    from papermind.output.schema import PaperMeta

    resolved = ResolvedSource(
        meta=PaperMeta(title="Uncached", arxiv_id="9999.99999"),
        pdf_path=tmp_path / "p.pdf",
        cache_key="9999.99999",
        cache_dir=tmp_path,  # empty -> no cached report
    )
    monkeypatch.setattr(arxiv_mod, "resolve", lambda source, config: resolved)

    client = TestClient(create_app(live=False))
    r = client.post("/analyze", data={"source": "9999.99999", "model": ""})
    assert r.status_code == 200
    assert "演示模式" in r.text  # falls back to the safe note, no analysis run


def test_rate_limiter_per_ip_and_global_caps():
    from papermind.web import RateLimiter

    rl = RateLimiter(per_ip=2, global_max=3, window=9999)
    assert rl.take("a") == (True, "")
    assert rl.take("a") == (True, "")
    assert rl.take("a") == (False, "ip")       # per-IP cap (2) hit; not counted globally
    assert rl.take("b") == (True, "")           # 3rd global slot
    assert rl.take("b") == (False, "global")    # global cap (3) reached


def test_rate_limiter_zero_means_unlimited_and_rollover():
    from papermind.web import RateLimiter

    unlimited = RateLimiter(per_ip=0, global_max=0)
    assert all(unlimited.take("x")[0] for _ in range(50))

    rl = RateLimiter(per_ip=1, global_max=10, window=9999)
    assert rl.take("a")[0] is True
    assert rl.take("a") == (False, "ip")
    rl._reset = 0  # force the window to elapse -> counters reset
    assert rl.take("a")[0] is True


def test_rate_limiter_release_refunds_a_slot():
    from papermind.web import RateLimiter

    rl = RateLimiter(per_ip=1, global_max=10, window=9999)
    assert rl.take("a")[0] is True
    assert rl.take("a") == (False, "ip")  # quota used up
    rl.release("a")                        # request failed before output -> refund
    assert rl.take("a")[0] is True         # slot is available again
    rl.release("a"); rl.release("a"); rl.release("a")  # never drives the counter negative
    assert rl.take("a")[0] is True


def _await_job(get_status, jid, tries=80):
    import time

    for _ in range(tries):
        s = get_status(jid)
        if s["status"] != "running":
            return s
        time.sleep(0.05)
    return get_status(jid)


def test_start_job_runs_and_captures_errors():
    from papermind.errors import PaperMindError
    from papermind.web import _get_job, _start_job

    ok = _start_job(lambda j: "<html>ok</html>")
    done = _await_job(_get_job, ok)
    assert done["status"] == "done" and done["html"] == "<html>ok</html>"

    def boom(j):
        raise PaperMindError("nope")

    bad = _await_job(_get_job, _start_job(boom))
    assert bad["status"] == "error" and "nope" in bad["error"]


def test_analyze_live_runs_async_and_polls(monkeypatch):
    import re

    import papermind.analyze as analyze_mod
    from papermind.output.schema import PaperMeta, Report

    monkeypatch.setattr(analyze_mod, "analyze", lambda *a, **k: Report(paper=PaperMeta(title="FakePaper")))
    client = TestClient(create_app(live=True, with_figures=False))

    r = client.post("/analyze", data={"source": "https://arxiv.org/abs/1706.03762"})
    assert r.status_code == 200
    m = re.search(r"/job/([0-9a-f]+)", r.text)
    assert m and "分析中" in r.text  # returned a polling page, not a blocking analysis
    jid = m.group(1)

    status = _await_job(lambda j: client.get(f"/job/{j}").json(), jid)
    assert status["status"] == "done"
    assert "FakePaper" in client.get(f"/result/{jid}").text


def test_start_job_generic_exception_is_not_leaked():
    from papermind.web import _get_job, _start_job

    def boom(j):
        raise ValueError("C:/Users/x/.papermind secret-internal kaboom")

    j = _await_job(_get_job, _start_job(boom))
    assert j["status"] == "error"
    # an unexpected error's internals (paths/URLs/etc.) must NOT reach the public page
    assert "kaboom" not in j["error"] and "secret-internal" not in j["error"]
    assert "请稍后重试" in j["error"]  # a fixed, safe message instead


def test_eviction_keeps_running_drops_finished():
    import papermind.web as web

    with web._JOBS_LOCK:
        saved = dict(web._JOBS)
        web._JOBS.clear()
    try:
        run_ids = [web._new_job() for _ in range(web._JOBS_MAX)]  # all running
        web._set_job(run_ids[0], status="done")
        extra = web._new_job()  # over cap -> must evict the finished one, not a running one
        assert web._get_job(run_ids[0]) is None
        assert all(web._get_job(i) is not None for i in run_ids[1:])
        assert web._get_job(extra) is not None
    finally:
        with web._JOBS_LOCK:
            web._JOBS.clear()
            web._JOBS.update(saved)


def test_job_ts_refreshed_on_completion_so_ttl_counts_from_done():
    import time

    import papermind.web as web

    with web._JOBS_LOCK:
        saved = dict(web._JOBS)
        web._JOBS.clear()
    try:
        jid = web._new_job()
        with web._JOBS_LOCK:
            web._JOBS[jid]["ts"] = time.time() - 10_000  # pretend it was created long ago
        web._set_job(jid, status="done", html="<x>")  # completion must refresh ts
        web._new_job()  # triggers the reclaim sweep
        assert web._get_job(jid) is not None  # survives: TTL is measured from completion, not creation
    finally:
        with web._JOBS_LOCK:
            web._JOBS.clear()
            web._JOBS.update(saved)


def test_hung_running_job_is_timed_out():
    import time

    import papermind.web as web

    with web._JOBS_LOCK:
        saved = dict(web._JOBS)
        web._JOBS.clear()
    try:
        jid = web._new_job()
        with web._JOBS_LOCK:
            web._JOBS[jid]["ts"] = time.time() - (web._JOBS_RUNNING_MAX + 60)  # stuck running too long
        web._new_job()  # the sweep marks the hung job as timed out (now reclaimable)
        j = web._get_job(jid)
        assert j["status"] == "error" and "超时" in j["error"]
    finally:
        with web._JOBS_LOCK:
            web._JOBS.clear()
            web._JOBS.update(saved)


def test_jobs_bounded_even_when_all_running():
    import papermind.web as web

    with web._JOBS_LOCK:
        saved = dict(web._JOBS)
        web._JOBS.clear()
    try:
        for _ in range(web._JOBS_MAX + 5):  # all fresh/running, well over capacity
            web._new_job()
        with web._JOBS_LOCK:
            assert len(web._JOBS) <= web._JOBS_MAX  # bounded: oldest running is dropped, never unbounded
    finally:
        with web._JOBS_LOCK:
            web._JOBS.clear()
            web._JOBS.update(saved)


def test_missing_job_and_result_degrade_gracefully():
    c = TestClient(create_app(live=True))
    assert c.get("/job/deadbeef").json() == {"status": "missing"}
    r = c.get("/result/deadbeef")
    assert r.status_code == 200 and "已过期" in r.text


def test_result_error_renders_escaped_form(monkeypatch):
    import re

    import papermind.analyze as analyze_mod

    def _raise(*a, **k):
        from papermind.errors import PaperMindError
        raise PaperMindError("解析失败<x>")

    monkeypatch.setattr(analyze_mod, "analyze", _raise)
    c = TestClient(create_app(live=True, with_figures=False))
    r = c.post("/analyze", data={"source": "https://arxiv.org/abs/x"})
    jid = re.search(r"/job/([0-9a-f]+)", r.text).group(1)
    assert _await_job(lambda j: c.get(f"/job/{j}").json(), jid)["status"] == "error"
    res = c.get(f"/result/{jid}")
    assert res.status_code == 200 and "解析失败" in res.text and "action='/analyze'" in res.text
    assert "<x>" not in res.text and "&lt;x&gt;" in res.text  # error is HTML-escaped (no XSS)


def test_analyze_get_runs_async(monkeypatch):
    import re

    import papermind.analyze as analyze_mod
    from papermind.output.schema import PaperMeta, Report

    monkeypatch.setattr(analyze_mod, "analyze", lambda *a, **k: Report(paper=PaperMeta(title="GetPaper")))
    c = TestClient(create_app(live=True, with_figures=False))
    r = c.get("/analyze", params={"source": "https://arxiv.org/abs/1706.03762"})
    m = re.search(r"/job/([0-9a-f]+)", r.text)
    assert m and "分析中" in r.text
    assert _await_job(lambda j: c.get(f"/job/{j}").json(), m.group(1))["status"] == "done"
    assert "GetPaper" in c.get(f"/result/{m.group(1)}").text


def test_ask_live_runs_async_and_polls(monkeypatch):
    import re

    import papermind.qa.chat as chat_mod
    from papermind.demo import build_demo_answer

    class FakeChat:  # stands in for PaperChat (no download / parse / index / model)
        def __init__(self, paper, model=None, mode="balanced"):
            self.mode = mode

        def ask(self, q):
            return build_demo_answer()

    monkeypatch.setattr(chat_mod, "PaperChat", FakeChat)
    c = TestClient(create_app(live=True))

    r = c.post("/ask", data={"source": "https://arxiv.org/abs/1706.03762",
                             "question": "why scale by sqrt d_k?", "model": "", "mode": "balanced"})
    assert r.status_code == 200
    m = re.search(r"/job/([0-9a-f]+)", r.text)
    assert m and "分析中" in r.text  # a polling page, NOT a blocking synchronous answer

    jid = m.group(1)
    assert _await_job(lambda j: c.get(f"/job/{j}").json(), jid)["status"] == "done"
    res = c.get(f"/result/{jid}")
    assert res.status_code == 200
    assert "why scale by sqrt d_k?" in res.text  # the chat log (with our question) rendered
    assert "action='/ask'" in res.text           # within the 问答 tab, ready for the next turn


def test_async_analyze_gated_over_quota(monkeypatch):
    import papermind.analyze as analyze_mod
    from papermind.output.schema import PaperMeta, Report

    monkeypatch.setattr(analyze_mod, "analyze", lambda *a, **k: Report(paper=PaperMeta(title="X")))
    c = TestClient(create_app(live=True, with_figures=False, rate_per_ip=1, rate_global=99))
    assert "/job/" in c.post("/analyze", data={"source": "https://arxiv.org/abs/a"}).text
    blocked = c.post("/analyze", data={"source": "https://arxiv.org/abs/b"})
    assert "/job/" not in blocked.text and "额度" in blocked.text  # gated before starting a job


def test_demo_never_hits_network(monkeypatch):
    import papermind.parser.arxiv as arxiv_mod

    def _boom(*a, **k):
        raise AssertionError("demo mode must not resolve/download")

    monkeypatch.setattr(arxiv_mod, "resolve", _boom)
    # an id that won't be cached locally -> demo returns the cached-only message,
    # and crucially must NOT call resolve() (which would raise AssertionError here)
    r = TestClient(create_app(live=False)).post("/analyze", data={"source": "https://arxiv.org/abs/9999.99999"})
    assert r.status_code == 200 and "演示模式" in r.text


def test_on_progress_step_reaches_job(monkeypatch):
    import re

    import papermind.analyze as analyze_mod
    from papermind.output.schema import PaperMeta, Report
    from papermind.web import _get_job

    def fake(*a, on_progress=None, **k):
        if on_progress:
            on_progress("技术细节")
        return Report(paper=PaperMeta(title="P"))

    monkeypatch.setattr(analyze_mod, "analyze", fake)
    c = TestClient(create_app(live=True, with_figures=False))
    jid = re.search(r"/job/([0-9a-f]+)", c.post("/analyze", data={"source": "https://arxiv.org/abs/x"}).text).group(1)
    _await_job(lambda j: c.get(f"/job/{j}").json(), jid)
    assert _get_job(jid)["step"] == "技术细节"  # on_progress -> _set_job -> /job chain works
