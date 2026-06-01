"""Streamlit GUI for PaperMind — run via ``papermind ui``.

Layout: a persistent Q&A chat in the sidebar (with the active model shown at the
bottom), and the main area carries 分析 / 速读 / 对比 / 复现 / 搜索. The model is
fixed to the configured default (no picker); figures are always generated.
Results are kept in session state so reruns don't re-run analysis — a full page
refresh starts fresh.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from papermind.errors import PaperMindError

_MODEL_LABELS = {
    "deepseek/deepseek-v4-pro": "DeepSeek-V4 Pro",
    "deepseek/deepseek-v4-flash": "DeepSeek-V4 Flash",
    "deepseek/deepseek-reasoner": "DeepSeek Reasoner",
    "deepseek/deepseek-chat": "DeepSeek Chat",
    "gpt-4o-mini": "OpenAI GPT-4o mini",
    "gpt-4o": "OpenAI GPT-4o",
}

st.set_page_config(page_title="PaperMind", page_icon="📄", layout="wide")


def _model_label() -> str:
    from papermind.config import load_config

    m = load_config().default_model
    return _MODEL_LABELS.get(m, m)


def _resolve_input(src: str, upload) -> str | None:
    """A pasted id/URL or an uploaded PDF -> a source string."""
    if upload is not None:
        tmp = Path(tempfile.gettempdir()) / f"papermind_{upload.name}"
        tmp.write_bytes(upload.getvalue())
        return str(tmp)
    return (src or "").strip() or None


def _usage_caption(usage):
    if usage and usage.calls:
        cost = f" · ~${usage.cost_usd:.4f}" if usage.cost_usd else ""
        st.caption(f"📊 {usage.calls} calls · {usage.total_tokens:,} tokens{cost}")


# --------------------------------------------------------------------------- #
# Sidebar: persistent Q&A chat + model at the bottom
# --------------------------------------------------------------------------- #
def _sidebar():
    with st.sidebar:
        st.markdown("### 📄 PaperMind")
        src = st.text_input("论文（arXiv id / URL）", key="paper_src", placeholder="2307.08691")
        up = st.file_uploader("或上传 PDF", type="pdf", key="paper_pdf")
        paper = _resolve_input(src, up)
        st.divider()
        st.markdown("**💬 问答**")
        if not paper:
            st.info("填入论文后即可在此提问。")
        else:
            st.caption(f"对象：`{paper[:48]}`")
            for q, ans in st.session_state.get("chat_log", []):
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    _render_answer(ans, compact=True)

            q = st.text_input("问点什么？", key="chat_q", placeholder="为什么要除以 √d_k？")
            if st.button("发送", key="chat_send", use_container_width=True) and q.strip():
                _do_ask(paper, q.strip())
                st.rerun()

        st.divider()
        st.caption(f"模型：**{_model_label()}**")
    return paper


def _do_ask(paper: str, question: str):
    from papermind.qa.chat import PaperChat

    if st.session_state.get("chat_paper") != paper or st.session_state.get("chat_obj") is None:
        st.session_state["chat_paper"] = paper
        st.session_state["chat_log"] = []
        try:
            st.session_state["chat_obj"] = PaperChat(paper, mode="balanced")
        except PaperMindError as exc:
            st.session_state["chat_obj"] = None
            st.session_state.setdefault("chat_log", []).append((question, _error_answer(str(exc))))
            return
    try:
        answer = st.session_state["chat_obj"].ask(question)
    except PaperMindError as exc:
        answer = _error_answer(str(exc))
    st.session_state["chat_log"].append((question, answer))


def _error_answer(msg: str):
    from papermind.output.schema import Answer, AnswerSegment

    return Answer(question="", segments=[AnswerSegment(kind="out_of_scope", text=msg)])


def _render_answer(answer, compact: bool = False):
    boxes = {"fact": st.success, "inference": st.warning, "out_of_scope": st.error}
    labels = {"fact": "论文事实", "inference": "基于论文的推理", "out_of_scope": "超出论文范围"}
    for seg in answer.segments:
        text = f"**【{labels.get(seg.kind, seg.kind)}】**"
        if seg.kind == "inference" and seg.confidence:
            text += f"（置信度: {seg.confidence}）"
        text += f"\n\n{seg.text}"
        if seg.reasoning and not compact:
            text += f"\n\n*推理依据：{seg.reasoning}*"
        boxes.get(seg.kind, st.info)(text)
    if answer.evidence:
        with st.expander("📌 原文依据"):
            for e in answer.evidence:
                loc = (e.section or "") + (f" p.{e.page}" if e.page else "")
                flag = "" if e.verified else "⚠️ 未核实 · "
                st.markdown(f"- **{loc or '—'}** — {flag}{e.text}")
    _usage_caption(getattr(answer, "usage", None))


# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
def _tab_analyze(paper: str | None):
    if not paper:
        st.info("在左侧侧栏填入 arXiv id / URL 或上传 PDF。")
        return
    if st.button("🎯 分析这篇论文", type="primary", key="an_go"):
        from papermind.analyze import analyze as run_analyze
        from papermind.config import load_config

        with st.spinner("分析中…（思考模型 + 图示，约 30–90 秒）"):
            try:
                # Web always generates figures; uses the configured default model.
                report = run_analyze(paper, with_figures=True, config=load_config())
                st.session_state["report"] = report
                st.session_state["report_src"] = paper
            except PaperMindError as exc:
                st.error(str(exc))
                return
    # Persist across reruns: re-show the last report for this paper without re-running.
    if st.session_state.get("report") is not None and st.session_state.get("report_src") == paper:
        report = st.session_state["report"]
        _usage_caption(report.usage)
        components.html(report.to_html(), height=900, scrolling=True)


def _tab_summary(paper: str | None):
    st.caption("速读 = 一句话 TL;DR + 几条要点（一次调用，比完整分析快）。")
    if not paper:
        st.info("在左侧填入论文。")
        return
    if st.button("📄 速读", type="primary", key="sm_go"):
        from papermind.summarize import summarize

        with st.spinner("生成速读…"):
            try:
                result, usage = summarize(paper)
                st.session_state["summary"] = (paper, result, usage)
            except PaperMindError as exc:
                st.error(str(exc))
                return
    cached = st.session_state.get("summary")
    if cached and cached[0] == paper:
        _, result, usage = cached
        st.subheader(result.title or "TL;DR")
        st.write(result.tldr)
        for p in result.key_points:
            st.markdown(f"- {p}")
        _usage_caption(usage)


def _tab_compare():
    raw = st.text_area("论文（每行一个，2–4 篇）", key="cmp_src", placeholder="2307.08691\n1706.03762")
    if st.button("📊 对比", type="primary", key="cmp_go"):
        items = [s.strip() for s in raw.splitlines() if s.strip()][:4]
        if len(items) < 2:
            st.warning("请至少填 2 篇（每行一个）。")
            return
        from papermind.compare import compare as run_compare

        with st.spinner("对比中…"):
            try:
                st.session_state["comparison"] = run_compare(items)
            except PaperMindError as exc:
                st.error(str(exc))
                return
    comp = st.session_state.get("comparison")
    if comp is not None:
        components.html(comp.to_html(), height=700, scrolling=True)
        _usage_caption(comp.usage)


def _tab_reproduce(paper: str | None):
    if not paper:
        st.info("在左侧填入论文。")
        return
    if st.button("🛠️ 生成复现指南", type="primary", key="rp_go"):
        from papermind.analyze import analyze as run_analyze
        from papermind.config import load_config

        with st.spinner("生成复现指南…"):
            try:
                report = run_analyze(paper, modules=["reproduction"], with_figures=False, config=load_config())
                st.session_state["repro"] = (paper, report)
            except PaperMindError as exc:
                st.error(str(exc))
                return
    cached = st.session_state.get("repro")
    if cached and cached[0] == paper and cached[1].reproduction is not None:
        report = cached[1]
        st.download_button("⬇️ setup.sh", report.to_setup_script(), file_name="setup.sh")
        st.download_button("⬇️ repro.ipynb", json.dumps(report.to_notebook(), ensure_ascii=False, indent=1), file_name="repro.ipynb")
        st.code(report.to_setup_script(), language="bash")


def _tab_search():
    query = st.text_input("关键词搜索 arXiv（类似 arXiv 全文检索，不调用模型）", key="se_q", placeholder="flash attention")
    if st.button("🔎 搜索", type="primary", key="se_go") and query.strip():
        from papermind.parser.arxiv import search_arxiv

        with st.spinner("搜索中…"):
            try:
                st.session_state["search"] = search_arxiv(query.strip(), max_results=20)
            except PaperMindError as exc:
                st.error(str(exc))
                return
    results = st.session_state.get("search")
    if results:
        st.dataframe(
            [
                {
                    "arXiv": r.arxiv_id,
                    "原文": f"https://arxiv.org/abs/{r.arxiv_id}",
                    "PDF": f"https://arxiv.org/pdf/{r.arxiv_id}.pdf",
                    "年份": r.year,
                    "标题": r.title,
                }
                for r in results
            ],
            column_config={
                "原文": st.column_config.LinkColumn("原文", display_text="打开"),
                "PDF": st.column_config.LinkColumn("PDF", display_text="PDF"),
            },
            use_container_width=True,
            hide_index=True,
        )
        st.caption("把某个 arXiv id 复制到左侧侧栏，即可分析 / 问答。")


def main():
    paper = _sidebar()
    tabs = st.tabs(["🎯 分析", "📄 速读", "📊 对比", "🛠️ 复现", "🔎 搜索"])
    with tabs[0]:
        _tab_analyze(paper)
    with tabs[1]:
        _tab_summary(paper)
    with tabs[2]:
        _tab_compare()
    with tabs[3]:
        _tab_reproduce(paper)
    with tabs[4]:
        _tab_search()


main()
