"""Streamlit GUI for PaperMind — run via ``papermind ui``.

A graphical front end for the same engine the CLI uses: analyze, summarize,
multi-turn chat, compare, reproduce, and arXiv search. Paste an arXiv id/URL or
upload a PDF, pick a model in the sidebar, and read the result inline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from papermind.errors import PaperMindError

MODELS = [
    ("", "默认（按配置）"),
    ("gpt-4o-mini", "OpenAI · GPT-4o mini"),
    ("gpt-4o", "OpenAI · GPT-4o"),
    ("claude-3-5-sonnet-20241022", "Anthropic · Claude 3.5 Sonnet"),
    ("deepseek/deepseek-chat", "DeepSeek · Chat (V3)"),
    ("deepseek/deepseek-reasoner", "DeepSeek · Reasoner (R1)"),
    ("gemini/gemini-1.5-flash", "Google · Gemini 1.5 Flash"),
    ("ollama/llama3.1", "本地 · Ollama llama3.1"),
]

st.set_page_config(page_title="PaperMind", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Sidebar (model / mode / status)
# --------------------------------------------------------------------------- #
def _sidebar():
    from papermind.config import load_config

    st.sidebar.title("📄 PaperMind")
    st.sidebar.caption("读懂一篇 arXiv 论文：分析 · 问答 · 复现")

    model = st.sidebar.selectbox("模型", MODELS, format_func=lambda x: x[1])[0] or None
    mode = st.sidebar.selectbox("问答模式", ["balanced", "strict", "explore"])
    figures = st.sidebar.checkbox("生成图示（Mermaid，较慢）", value=False)

    cfg = load_config()
    keys = []
    if cfg.openai_key:
        keys.append("OpenAI")
    if cfg.anthropic_key:
        keys.append("Anthropic")
    if cfg.deepseek_key:
        keys.append("DeepSeek")
    st.sidebar.markdown("---")
    st.sidebar.caption("已配置 key: " + ("、".join(keys) if keys else "无（可用本地 Ollama）"))
    st.sidebar.caption(f"默认模型: `{cfg.default_model}`")
    return model, mode, figures


def _source_input(key: str):
    """A paper input + PDF uploader; returns a resolvable source string or None."""
    src = st.text_input("arXiv id / URL", key=f"{key}_src", placeholder="2307.08691 或 https://arxiv.org/abs/2307.08691")
    up = st.file_uploader("或上传 PDF", type="pdf", key=f"{key}_pdf")
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / f"papermind_{up.name}"
        tmp.write_bytes(up.getvalue())
        return str(tmp)
    return src.strip() or None


def _usage_caption(usage):
    if usage and usage.calls:
        cost = f" · ~${usage.cost_usd:.4f}" if usage.cost_usd else ""
        st.caption(f"📊 {usage.calls} calls · {usage.total_tokens:,} tokens{cost}")


def _embed_report(report):
    components.html(report.to_html(), height=900, scrolling=True)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def _tab_analyze(model, figures):
    source = _source_input("an")
    if st.button("🎯 分析", type="primary", key="an_go"):
        if not source:
            st.warning("请输入 arXiv id / URL 或上传 PDF。")
            return
        from papermind.analyze import analyze as run_analyze
        from papermind.config import load_config

        with st.spinner("分析中…（首次分析新论文约 15–40 秒）"):
            try:
                report = run_analyze(source, model=model, with_figures=figures, config=load_config())
            except PaperMindError as exc:
                st.error(str(exc))
                return
        _usage_caption(report.usage)
        _embed_report(report)


def _tab_summary(model):
    source = _source_input("sm")
    if st.button("📄 摘要", type="primary", key="sm_go"):
        if not source:
            st.warning("请输入论文。")
            return
        from papermind.summarize import summarize

        with st.spinner("生成 TL;DR…"):
            try:
                result, usage = summarize(source, model=model)
            except PaperMindError as exc:
                st.error(str(exc))
                return
        st.subheader(result.title or "TL;DR")
        st.write(result.tldr)
        for point in result.key_points:
            st.markdown(f"- {point}")
        _usage_caption(usage)


def _tab_chat(model, mode):
    source = st.text_input("论文（先设定再提问）", key="chat_src", placeholder="2307.08691")
    if not source.strip():
        st.info("先填一篇论文，然后在下方提问（支持多轮）。")
        return

    state_key = f"chat::{source.strip()}::{model}::{mode}"
    if st.session_state.get("chat_key") != state_key:
        st.session_state["chat_key"] = state_key
        st.session_state["chat_obj"] = None
        st.session_state["chat_log"] = []  # list of (question, Answer)

    for question, answer in st.session_state["chat_log"]:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            _render_answer(answer)

    prompt = st.chat_input("问点什么？（如：为什么要除以 √d_k？）")
    if prompt:
        from papermind.qa.chat import PaperChat

        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("检索原文 · 生成分层回答…"):
                try:
                    if st.session_state["chat_obj"] is None:
                        st.session_state["chat_obj"] = PaperChat(source.strip(), model=model, mode=mode)
                    answer = st.session_state["chat_obj"].ask(prompt)
                except PaperMindError as exc:
                    st.error(str(exc))
                    return
            _render_answer(answer)
        st.session_state["chat_log"].append((prompt, answer))


def _render_answer(answer):
    boxes = {"fact": st.success, "inference": st.warning, "out_of_scope": st.error}
    labels = {"fact": "论文事实", "inference": "基于论文的推理", "out_of_scope": "超出论文范围"}
    for seg in answer.segments:
        text = f"**【{labels.get(seg.kind, seg.kind)}】**"
        if seg.kind == "inference" and seg.confidence:
            text += f"（置信度: {seg.confidence}）"
        text += f"\n\n{seg.text}"
        if seg.reasoning:
            text += f"\n\n*推理依据：{seg.reasoning}*"
        boxes.get(seg.kind, st.info)(text)
    if answer.evidence:
        with st.expander("📌 原文依据"):
            for e in answer.evidence:
                loc = (e.section or "") + (f" p.{e.page}" if e.page else "")
                flag = "" if e.verified else "⚠️ 未核实 · "
                st.markdown(f"- **{loc or '—'}** — {flag}{e.text}")
    _usage_caption(answer.usage)


def _tab_compare(model):
    raw = st.text_area("论文（每行一个，2–4 篇）", key="cmp_src", placeholder="2307.08691\n1706.03762")
    if st.button("📊 对比", type="primary", key="cmp_go"):
        items = [s.strip() for s in raw.splitlines() if s.strip()][:4]
        if len(items) < 2:
            st.warning("请至少填 2 篇（每行一个）。")
            return
        from papermind.compare import compare as run_compare

        with st.spinner("对比中…"):
            try:
                comparison = run_compare(items, model=model)
            except PaperMindError as exc:
                st.error(str(exc))
                return
        components.html(comparison.to_html(), height=700, scrolling=True)
        _usage_caption(comparison.usage)


def _tab_reproduce(model):
    source = _source_input("rp")
    if st.button("🛠️ 生成复现指南", type="primary", key="rp_go"):
        if not source:
            st.warning("请输入论文。")
            return
        from papermind.analyze import analyze as run_analyze
        from papermind.config import load_config

        with st.spinner("生成复现指南…"):
            try:
                report = run_analyze(source, model=model, modules=["reproduction"], with_figures=False, config=load_config())
            except PaperMindError as exc:
                st.error(str(exc))
                return
        if report.reproduction is None:
            st.warning("未能提取复现信息。")
            return
        st.download_button("⬇️ setup.sh", report.to_setup_script(), file_name="setup.sh")
        import json

        st.download_button("⬇️ repro.ipynb", json.dumps(report.to_notebook(), ensure_ascii=False, indent=1), file_name="repro.ipynb")
        st.code(report.to_setup_script(), language="bash")
        _usage_caption(report.usage)


def _tab_search():
    query = st.text_input("搜索 arXiv（不调用模型）", key="se_q", placeholder="flash attention")
    if st.button("🔎 搜索", type="primary", key="se_go") and query.strip():
        from papermind.parser.arxiv import search_arxiv

        with st.spinner("搜索中…"):
            try:
                results = search_arxiv(query.strip(), max_results=15)
            except PaperMindError as exc:
                st.error(str(exc))
                return
        st.dataframe(
            [
                {"arXiv": r.arxiv_id, "年份": r.year, "标题": r.title, "作者": ", ".join(r.authors[:3])}
                for r in results
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("复制 arXiv id 到「分析 / 问答」标签使用。")


def main():
    model, mode, figures = _sidebar()
    tabs = st.tabs(["🎯 分析", "💬 问答", "📄 摘要", "📊 对比", "🛠️ 复现", "🔎 搜索"])
    with tabs[0]:
        _tab_analyze(model, figures)
    with tabs[1]:
        _tab_chat(model, mode)
    with tabs[2]:
        _tab_summary(model)
    with tabs[3]:
        _tab_compare(model)
    with tabs[4]:
        _tab_reproduce(model)
    with tabs[5]:
        _tab_search()


main()
