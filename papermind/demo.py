"""Offline, zero-config demo playback — `papermind demo`.

Replays a curated, pre-baked analysis report and a layered Q&A with built-in
pacing, so it works with no API key and no network. Ideal for a quick look and
for recording a tight (<20s) demo GIF (see docs/demo.tape).

Featured paper: "Attention Is All You Need" (Transformer) — maximum recognition.
"""

from __future__ import annotations

import time

from papermind.output.schema import (
    Answer,
    AnswerSegment,
    Benchmark,
    CommonError,
    Connection,
    ConnectionsSection,
    Contributions,
    EvidenceItem,
    Figure,
    PaperMeta,
    Reproduction,
    Report,
    Source,
    TechnicalPoint,
    TechnicalSection,
)
from papermind.output.terminal import render_report


def _demo_framework():
    """A hand-built whole-method framework for the demo paper (Transformer), so the
    offline report showcases the embedded framework diagram with no model call."""
    from papermind.figures.framework import FEdge, FLegend, FNode, FrameworkSpec, _auto_layout

    return _auto_layout(FrameworkSpec(
        title="Transformer 端到端框架",
        nodes=[
            FNode(id="emb", kind="io", label="输入嵌入 + 位置编码"),
            FNode(id="enc", kind="box", label="编码器 ×N", lines=["多头自注意力", "前馈网络 FFN", "残差 + LayerNorm"]),
            FNode(id="dec", kind="box", label="解码器 ×N", lines=["掩码自注意力", "编码器-解码器交叉注意力", "前馈网络 FFN"]),
            FNode(id="head", kind="box", label="线性投影 + Softmax"),
            FNode(id="out", kind="io", label="输出词概率"),
            FNode(id="mha", kind="group", col=1, inferred=True, label="多头注意力（机制）",
                  lines=["softmax(Q·Kᵀ / √d_k)·V", "h 个子空间并行"]),
        ],
        edges=[
            FEdge(src="emb", dst="enc"),
            FEdge(src="enc", dst="dec", style="emph"),
            FEdge(src="dec", dst="head"),
            FEdge(src="head", dst="out"),
            FEdge(src="enc", dst="mha", style="dashed"),
        ],
        legend=[FLegend(style="solid", text="数据流"), FLegend(style="dashed", text="论文未显式画出的推断")],
    ))


def _load_demo_svg() -> str:
    """A real teaching SVG for the demo's first point (Scaled Dot-Product Attention).
    Prefer the gallery figure; fall back to the wheel-bundled hero so ``papermind demo``
    shows a faithful figure even from a bare pip install. Absent -> no figure (not a
    broken diagram)."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    for p in (here.parent / "examples" / "figures" / "transformer-fig1.svg", here / "assets" / "hero.svg"):
        try:
            return p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
    return ""


_DEMO_SVG = _load_demo_svg()

_KIND = {
    "fact": ("论文事实", "green"),
    "inference": ("基于论文的推理", "yellow"),
    "out_of_scope": ("超出论文范围", "dim red"),
}
_CONF = {"high": "高", "mid": "中", "low": "低"}


def build_demo_report() -> Report:
    return Report(
        paper=PaperMeta(
            title="Attention Is All You Need",
            arxiv_id="1706.03762",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
            year=2017,
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        ),
        contributions=Contributions(
            main_contribution="提出完全基于注意力机制的 Transformer 架构，彻底去掉循环与卷积，在机器翻译上以更高质量、更强并行、更短训练时间刷新 SOTA。",
            novelty="用 self-attention 直接建模任意两个位置的依赖（路径长度 O(1)）；多头注意力在不同子空间并行关注不同关系；位置编码在无循环结构下注入顺序信息。",
            problem_solved="RNN 必须按时间步串行、难并行且长程依赖会衰减；Transformer 把依赖建模变成可并行的矩阵运算，既快又能直接捕捉长程关系。",
            sources=[Source(text="based solely on attention mechanisms, dispensing with recurrence and convolutions entirely", section="Abstract", page=1)],
        ),
        technical=TechnicalSection(
            details=[
                TechnicalPoint(
                    name="缩放点积注意力 (Scaled Dot-Product Attention)",
                    explanation="每个 Query 与所有 Key 做点积得到相关性，softmax 归一化后对 Value 加权求和；关键是除以 √d_k——维度大时点积数值大、softmax 会进入梯度极小的饱和区，缩放把方差拉回稳定范围。",
                    analogy="像搜索引擎：查询词(Q)与每篇文档关键词(K)算匹配度，再按匹配度加权汇总文档内容(V)；除以 √d_k 相当于先标准化分数。",
                    source_section="Section 3.2.1",
                    page=4,
                    difficulty="high",
                    figure=(Figure(type="ai_generated", svg=_DEMO_SVG, caption="缩放点积注意力（教学示意图）")
                            if _DEMO_SVG else None),
                ),
                TechnicalPoint(
                    name="多头注意力 (Multi-Head Attention)",
                    explanation="把 Q/K/V 投影到 h 个较低维子空间各自独立做注意力，再拼接并线性变换。不同的“头”可学习关注不同关系（句法、共指、局部 vs 长程），表达力强于单个全维注意力。",
                    analogy="一个委员会从多个专业角度同时审阅同一句话，再汇总意见，比一个人通览更全面。",
                    source_section="Section 3.2.2",
                    page=4,
                    difficulty="mid",
                ),
            ]
        ),
        connections=ConnectionsSection(
            related_works=[
                Connection(
                    concept="Seq2Seq + Attention",
                    paper="Bahdanau et al. 2014",
                    arxiv_link="https://arxiv.org/abs/1409.0473",
                    relationship="注意力机制的来源，本文将其推广为唯一的建模手段",
                ),
            ]
        ),
        reproduction=Reproduction(
            official_code="https://github.com/tensorflow/tensor2tensor",
            recommended_hardware="8× P100（base ~12h，big ~3.5 天）",
            key_hyperparams=["d_model=512", "h=8", "d_ff=2048", "N=6", "warmup_steps=4000"],
            performance_benchmarks=[
                Benchmark(setting="WMT14 EN-DE (BLEU)", baseline="ConvS2S 25.2", result="28.4 (big)", speedup="-", memory="-")
            ],
            common_errors=[
                CommonError(
                    error="训练 loss 不下降 / 发散",
                    cause="未用 warmup 学习率调度，或漏了对注意力分数做 √d_k 缩放",
                    fix_command="scores = Q @ K.transpose(-2,-1) / math.sqrt(d_k)  # 并用 Noam warmup=4000",
                )
            ],
            gotchas=["warmup 学习率与 √d_k 缩放是最常被忽略的两个关键点；注意区分 encoder/decoder 自注意力与 cross-attention 的掩码。"],
        ),
        framework=_demo_framework(),
    )


def build_demo_answer() -> Answer:
    return Answer(
        question="为什么注意力分数要除以 √d_k？",
        segments=[
            AnswerSegment(
                kind="fact",
                text="当 d_k 较大时，Q·K 的点积数值会变大、方差增大，把 softmax 推入梯度极小的饱和区；除以 √d_k 把点积方差拉回 ~1，保持梯度稳定、训练可学。",
            ),
            AnswerSegment(
                kind="fact",
                text="论文在 Section 3.2.1 明确指出：未缩放时大 d_k 下点积注意力表现劣于加性注意力，因此引入 1/√d_k 的缩放。",
            ),
            AnswerSegment(
                kind="inference",
                text="所以 d_k 越大，缩放越关键——若你自定义注意力时忘了它，深层/大维度模型会更容易训练不稳定或不收敛。",
                confidence="mid",
                reasoning="论文把缩放动机归因于点积随 d_k 增大而增大；据此可推广到更大维度的稳定性影响。",
            ),
        ],
        evidence=[
            EvidenceItem(text="for large values of d_k, the dot products grow large in magnitude, pushing the softmax into regions of extremely small gradients", section="Section 3.2.1", page=4),
            EvidenceItem(text="we scale the dot products by 1/√d_k", section="Section 3.2.1", page=4),
        ],
    )


def play(console, speed: float = 1.0) -> None:
    """Render the demo with pacing. ``speed=0`` disables all sleeps (instant)."""

    def pause(seconds: float) -> None:
        if speed > 0:
            time.sleep(seconds / speed)

    def status(label: str, seconds: float) -> None:
        if speed > 0:
            with console.status(f"[cyan]{label}[/cyan]"):
                time.sleep(seconds / speed)

    report = build_demo_report()
    answer = build_demo_answer()
    arxiv = report.paper.arxiv_id

    # Beat 1: analyze -> the structured report.
    if speed > 0:
        console.clear()
    console.print(f"[bold green]$[/bold green] papermind analyze https://arxiv.org/abs/{arxiv} --format all")
    pause(0.4)
    status("分析中… 解析 PDF · 贡献 · 技术细节 · 知识关联 · 复现", 1.0)
    render_report(report, console)
    pause(4.5)  # let the viewer read the report

    # Beat 2: ask -> the grounded, layered answer (the differentiator) on a clean screen.
    if speed > 0:
        console.clear()
    console.print(f'[bold green]$[/bold green] papermind ask https://arxiv.org/abs/{arxiv} "{answer.question}"')
    pause(0.4)
    status("检索原文段落 · 生成分层回答", 0.8)
    _reveal_answer(answer, console, line_delay=0.8 if speed > 0 else 0.0, speed=speed)
    pause(0.8)

    console.print()
    console.print(
        "[bold cyan]理解 → 复现，关键判断都带原文依据并核验。[/bold cyan]  [dim]pip install papermind-ai[/dim]"
    )
    pause(2.0)  # hold the final frame so it lands in a looping GIF


def _reveal_answer(answer: Answer, console, line_delay: float, speed: float) -> None:
    for seg in answer.segments:
        label, color = _KIND.get(seg.kind, (seg.kind, "white"))
        head = f"[bold {color}]【{label}】[/bold {color}]"
        if seg.kind == "inference" and seg.confidence:
            head += f"[{color}]  置信度: {_CONF.get(seg.confidence, '')}[/{color}]"
        console.print(head)
        console.print(seg.text)
        if seg.reasoning:
            console.print(f"[dim]  推理依据: {seg.reasoning}[/dim]")
        console.print()
        if line_delay and speed > 0:
            time.sleep(line_delay / speed)

    console.print("[bold cyan]原文依据[/bold cyan]")
    for e in answer.evidence:
        loc = e.section or ""
        if e.page:
            loc = f"{loc} p.{e.page}".strip()
        console.print(f"  [cyan]{loc or '-'}[/cyan]  {e.text}")
