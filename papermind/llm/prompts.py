"""All prompt templates, centralized.

Analysis modules answer in Chinese (keeping English technical terms), matching
the documented schema examples. Q&A answers in the user's own language.
Every prompt that expects structured output specifies the exact JSON keys so the
parsers stay simple and tolerant.
"""

from __future__ import annotations

from typing import List, Optional

# --------------------------------------------------------------------------- #
# Shared context block
# --------------------------------------------------------------------------- #
def paper_context(title: str, abstract: Optional[str], outline: str, body: str) -> str:
    parts = [f"论文标题: {title}"]
    if abstract:
        parts.append(f"摘要:\n{abstract}")
    if outline:
        parts.append(f"章节大纲:\n{outline}")
    parts.append(f"正文节选:\n{body}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Quick summary (TL;DR)
# --------------------------------------------------------------------------- #
SUMMARY_SYSTEM = (
    "你是高效的论文速读助手。用中文给出一段 TL;DR 和 3-5 条要点，只依据给定文本，不要编造。"
    "严格输出 JSON，不要任何额外文字或代码围栏。"
)


COMPARE_SYSTEM = (
    "你是擅长横向对比论文的研究者。基于给定的多篇论文要点，用中文写一段简明对比小结："
    "它们的共同点、关键差异、各自更适合的场景。3-6 句，只依据给定要点，不要编造。"
    "只输出小结正文，不要标题、不要 JSON、不要代码围栏。"
)


def compare_user(briefs: str) -> str:
    return "以下是各论文的要点：\n\n" + briefs + "\n\n请写一段对比小结："


# --------------------------------------------------------------------------- #
# Long-paper map step: extractive condensation (keeps verbatim wording so the
# downstream analysis still cites real text, and no section gets truncated away).
# --------------------------------------------------------------------------- #
CONDENSE_SYSTEM = (
    "你是论文信息抽取器。从给定的论文片段中**原样摘录**最关键的句子，"
    "覆盖：研究问题与贡献声明、方法与公式、模型/算法细节、数据集、实验设置与超参数、"
    "主要结果与数值、局限与结论。"
    "保留原文措辞与出现的英文术语、公式、数字，**不要改写、不要翻译、不要总结成自己的话**；"
    "保留出现的章节标题/编号。逐条摘录，去掉与复现无关的客套与引用列表。只输出摘录正文。"
)


def condense_user(section_label: str, text: str) -> str:
    head = f"【片段：{section_label}】\n" if section_label else ""
    return head + text + "\n\n请原样摘录上面片段中的关键句子："


def summary_user(context: str) -> str:
    return (
        context
        + "\n\n请输出如下 JSON：\n"
        '{\n  "tldr": "2-3 句话讲清这篇论文做了什么、为什么重要",\n'
        '  "key_points": ["要点1", "要点2", "要点3"]\n}'
    )


# --------------------------------------------------------------------------- #
# Analysis modules
#
# All four modules share ONE system prompt and put the (identical) paper context
# at the FRONT of the user message, with the module-specific task at the END.
# This shared prefix lets providers with automatic prompt caching (DeepSeek,
# OpenAI) discount the repeated ~10k-token context across the four calls.
# --------------------------------------------------------------------------- #
ANALYSIS_SYSTEM = (
    "你是严谨的 AI/ML 论文分析助手。只依据给定的论文文本作答，不要编造；"
    "用中文（保留必要的英文术语）。"
    "凡数学符号、变量、维度或公式——哪怕只是单个字母（如 $d_k$、$N$、$h$、$QK^\\top$、$O(N^2)$）"
    "——一律用 $...$ 包裹成 LaTeX，不要写成纯文本（不要出现 d_k、sqrt、根号 这类未包裹写法）。"
    "严格输出 JSON，不要任何额外文字或代码围栏。"
)


# Module 1: Contributions
def contributions_user(context: str) -> str:
    return (
        context
        + "\n\n【任务】提炼论文的核心贡献，每个判断尽量给出原文出处（章节名+页码）。输出如下 JSON：\n"
        "{\n"
        '  "main_contribution": "一句话点明最核心的那一个贡献（聚焦、不堆砌）",\n'
        '  "novelty": "相比前人工作新在哪里",\n'
        '  "problem_solved": "解决了之前没解决的什么问题",\n'
        '  "sources": [{"text": "支撑上述判断的原文片段", "section": "Abstract", "page": 1}]\n'
        "}\n"
        "sources 给 1-4 条，page 为整数页码（未知则用 null）。"
    )


# Module 2: Technical details
def technical_user(context: str, max_points: int = 6) -> str:
    return (
        context
        + f"\n\n【任务】识别论文中最难懂的技术点（模型结构、关键公式、训练技巧等），最多 {max_points} 个。"
        "解释要讲清它**如何运作**以及**为何这样设计/为何有效**（而非只说是什么），并给一个建立直觉的类比。"
        "若有核心公式，用 LaTeX 写进 formula（不带 $）；解释/类比中的数学符号用 $...$ 包裹（如 $\\sqrt{d_k}$）。"
        "输出如下 JSON：\n"
        "{\n"
        '  "technical_details": [\n'
        "    {\n"
        '      "name": "技术点名称",\n'
        '      "explanation": "用直白语言解释（2-4 句），数学符号用 $...$ LaTeX",\n'
        '      "analogy": "一个建立直觉的类比",\n'
        '      "formula": "核心公式的 LaTeX（不带 $），无公式则为 null",\n'
        '      "source_section": "Section 3.1",\n'
        '      "page": 5,\n'
        '      "difficulty": "high|mid|low"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        r'formula 示例：\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V'
        "\npage 为整数页码（未知则用 null）。按难度从高到低排序。"
    )


# Module 3: Connections
def connections_user(context: str, max_items: int = 6) -> str:
    return (
        context
        + f"\n\n【任务】把本文的关键技术点与已有经典工作挂钩，说明继承/改进/对比关系，最多 {max_items} 条。"
        "arxiv_link 仅在你确信该论文确有对应 arXiv 编号时给出，否则填 null，绝不编造链接。输出如下 JSON：\n"
        "{\n"
        '  "connections": [\n'
        "    {\n"
        '      "concept": "关联的概念/方法名",\n'
        '      "paper": "相关论文（作者 年份），如 Vaswani et al. 2017",\n'
        '      "arxiv_link": "https://arxiv.org/abs/1706.03762 或 null",\n'
        '      "relationship": "本文与它的关系（继承/改进/对比等）"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


# Module 4: Reproduction
def reproduction_user(context: str) -> str:
    return (
        context
        + "\n\n【任务】产出尽量完整、可操作的复现指南。论文中明确的信息（官方代码、超参、基准）以论文为准；"
        "环境步骤、常见报错与修复可结合通用工程经验补充，但不要编造不存在的链接或版本号（不确定填 null）。"
        "命令要可直接复制运行。输出如下 JSON：\n"
        "{\n"
        '  "official_code": "GitHub 链接或 null",\n'
        '  "version_tag": "与论文版本对应的 release/tag 或 null",\n'
        '  "requirements": "环境要求，如 CUDA >= 11.6, PyTorch >= 1.12",\n'
        '  "recommended_hardware": "推荐硬件，如 A100 / H100",\n'
        '  "key_hyperparams": ["block_size=64", "causal=True"],\n'
        '  "env_setup_steps": [{"step": 1, "title": "确认 CUDA 版本", "desc": "...", "command": "nvcc --version"}],\n'
        '  "performance_benchmarks": [{"setting": "seq_len=2048", "baseline": "12 ms", "result": "5 ms", "speedup": "2.4x", "memory": "节省 ~50%"}],\n'
        '  "datasets": [{"name": "WikiText-103", "purpose": "语言模型基准", "link": "https://huggingface.co/datasets/Salesforce/wikitext"}],\n'
        '  "common_errors": [{"error": "报错信息", "cause": "原因分析", "fix_command": "修复命令"}],\n'
        '  "gotchas": ["已知坑点提示"]\n'
        "}\n"
        "尽量给出 3-6 个 env_setup_steps、2-4 条 common_errors。未知字段用 null 或空数组。"
    )


# --------------------------------------------------------------------------- #
# Figures: matching + generation
# --------------------------------------------------------------------------- #
FIGURE_MATCH_SYSTEM = (
    "你要把论文中的技术点与论文里的图（Figure/Table）按语义对应。"
    "只在某张图确实直接展示该技术点时才匹配，否则该技术点匹配 null。"
    "严格输出 JSON，不要额外文字。"
)


def figure_match_user(point_names: List[str], figures_digest: str) -> str:
    points = "\n".join(f"- {name}" for name in point_names)
    return (
        f"技术点列表：\n{points}\n\n"
        f"论文中的图及其 caption：\n{figures_digest or '（无）'}\n\n"
        "请输出如下 JSON（figure 用图的标签如 'Figure 1'，无合适图则为 null）：\n"
        '{"matches": [{"point": "技术点名称", "figure": "Figure 1"}]}'
    )


MERMAID_SYSTEM = (
    "你是把论文里的模型/算法结构画成**论文式结构图（block diagram）**的专家——像论文 Figure 那样"
    "用方块表示模块、箭头表示数据流，用 subgraph 把相关模块成组（如 Encoder/Decoder/某个 Block），"
    "体现层次与包含关系，而不是一条直线的流程图。要求：\n"
    "- 用 Mermaid flowchart（方向 TB 或 LR 取决于结构）；\n"
    "- 用 subgraph 分组体现结构层次；箭头可带标签说明传递的张量/信息；\n"
    "- 节点文字简洁（可含 $ 包裹的简短公式或维度，如 d_model）；\n"
    "- 用 classDef 给不同角色的模块配色，整体像一张干净的论文结构图；\n"
    "- 节点 id 用字母数字，语法合法可渲染。严格输出 JSON，不要额外文字或代码围栏。"
)


def mermaid_user(name: str, explanation: str) -> str:
    return (
        f"技术点：{name}\n解释：{explanation}\n\n"
        "请画一张论文式结构图（用 subgraph 分组、箭头带标签）。输出如下 JSON（mermaid 用 \\n 换行）：\n"
        '{"mermaid": "flowchart TB\\n'
        "  subgraph Encoder\\n    direction TB\\n    E1[Multi-Head\\\\nAttention]:::attn --> E2[Add & Norm]:::norm\\n"
        "    E2 --> E3[Feed Forward]:::ff --> E4[Add & Norm]:::norm\\n  end\\n"
        "  X[输入嵌入 + 位置编码]:::io --> E1\\n  E4 --> Y[编码表示]:::io\\n"
        "  classDef attn fill:#dbeafe,stroke:#3b82f6;classDef ff fill:#fef9c3,stroke:#ca8a04;\\n"
        "  classDef norm fill:#f1f5f9,stroke:#94a3b8;classDef io fill:#dcfce7,stroke:#22c55e;\", "
        '"caption": "AI 生成示意图：……"}'
    )


def image_diagram_prompt(name: str, explanation: str) -> str:
    """Prompt for a diffusion/image model (best-effort technical schematic)."""
    return (
        f"A clean, minimalist technical schematic diagram illustrating the concept "
        f"'{name}' from a machine-learning paper. {explanation} "
        f"Flat infographic style: labeled boxes connected by arrows showing the data flow, "
        f"soft color palette, white background, high contrast, crisp vector look, no photorealism."
    )


# --------------------------------------------------------------------------- #
# Q&A / reasoning / tutoring
# --------------------------------------------------------------------------- #
_LAYERING_RULES = (
    "回答必须分层标注来源，输出为若干 segment，每个 segment 的 kind 取值：\n"
    "- fact：论文原文明确支持（必须能在给定段落中找到依据）。\n"
    "- inference：基于论文推导得出，需要在 reasoning 里写明推理依据，并在 confidence 标注 high/mid/low。\n"
    "- out_of_scope：论文无法支撑的内容，明确告知，不要编造。\n"
    "evidence 给出支撑 fact 的原文片段及其 section/page；sources 汇总引用到的 section/page。"
)

_MODE_RULES = {
    "strict": (
        "模式=strict：只输出 fact 段落。凡论文未明确支持的，用一个 out_of_scope 段落说明“原文未提及”，"
        "不做任何推理。"
    ),
    "balanced": (
        "模式=balanced：先给 fact，再给必要的 inference（标注 reasoning 与 confidence）。"
        "无依据处用 out_of_scope 说明。"
    ),
    "explore": (
        "模式=explore：在 fact 基础上鼓励合理的 inference 与应用建议（仍需标注 reasoning 与 confidence），"
        "但超出论文且无法推导处仍用 out_of_scope 标注。"
    ),
}


def qa_system(mode: str) -> str:
    return (
        "你是基于给定论文回答问题的助手。用与用户提问相同的语言作答。"
        "只能依据提供的论文段落作答，找不到依据时不要编造。\n"
        + _LAYERING_RULES
        + "\n"
        + _MODE_RULES.get(mode, _MODE_RULES["balanced"])
        + "\n严格输出 JSON，不要任何额外文字或代码围栏。"
    )


def _answer_schema_hint() -> str:
    return (
        "请输出如下 JSON：\n"
        "{\n"
        '  "segments": [\n'
        '    {"kind": "fact", "text": "...", "confidence": null, "reasoning": null},\n'
        '    {"kind": "inference", "text": "...", "confidence": "mid", "reasoning": "推理依据..."},\n'
        '    {"kind": "out_of_scope", "text": "...", "confidence": null, "reasoning": null}\n'
        "  ],\n"
        '  "evidence": [{"text": "原文片段", "section": "3.1", "page": 5}],\n'
        '  "sources": [{"section": "3.1", "page": 5}]\n'
        "}"
    )


def qa_user(question: str, passages: str) -> str:
    return (
        f"用户问题：{question}\n\n"
        f"检索到的论文相关段落：\n{passages or '（未检索到相关段落）'}\n\n"
        + _answer_schema_hint()
    )


TUTOR_SYSTEM_EXTRA = (
    "你现在是复现全程辅导助手，覆盖：代码/报错调试、环境配置排查、迁移到用户自己的项目、调参与实验设计。"
    "论文明确支持的标 fact；基于工程经验的推断标 inference（写明依据与置信度）。"
    "代码/命令尽量可直接运行，关键约束（精度要求、维度限制等）要主动提示。"
)


def tutor_system(mode: str = "explore") -> str:
    return qa_system(mode) + "\n" + TUTOR_SYSTEM_EXTRA


def debug_user(error: str, passages: str) -> str:
    return (
        f"我在复现这篇论文时遇到如下报错：\n```\n{error}\n```\n\n"
        f"论文相关段落：\n{passages or '（未检索到相关段落）'}\n\n"
        "请定位原因并给出可直接运行的修复方案。"
        + _answer_schema_hint()
    )
