# PaperMind

> 读懂一篇 arXiv 论文，到能动手复现它 —— 一个命令行工具（含 Python API）。

[![CI](https://github.com/Wenhao-Hua/papermind/actions/workflows/ci.yml/badge.svg)](https://github.com/Wenhao-Hua/papermind/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/paperis)](https://pypi.org/project/paperis/)
[![Python](https://img.shields.io/pypi/pyversions/paperis)](https://pypi.org/project/paperis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PaperMind 不是又一个「论文摘要器」。给它一篇 arXiv 论文，它会产出**结构化分析报告**（贡献、最难懂的技术点+图示、知识脉络、完整复现指南），并提供**带原文依据的 RAG 问答**和**复现全程辅导**。每个回答都把「论文事实 / 基于论文的推理 / 超出论文范围」分层标注，**找不到依据时直说"原文未提及"，绝不编造**。

<p align="center">
  <a href="examples/README.md"><b>▶️ 查看示例报告（Gallery，GitHub 直接渲染含 Mermaid 图）</b></a>
</p>
<!-- 录制 demo 后取消注释（见 docs/DEMO.md）：
<p align="center"><img src="docs/demo.gif" alt="PaperMind demo" width="760"></p>
-->

## ✨ 能做什么

| 命令 | 作用 |
| --- | --- |
| `analyze` | 四模块报告：🎯 贡献 · 🔬 技术点（**含公式 + 原图/AI 示意图**）· 🔗 知识关联 · 🛠️ 复现指南 |
| `summary` | 一句话 TL;DR + 要点（一次调用，便宜） |
| `ask` / `chat` | **带原文依据的分层问答**：论文事实 / 基于论文的推理(+置信度) / 超出论文范围 + 可跳转页码 |
| `tutor` / `debug` | 复现辅导；贴报错直接给可运行的修复 |
| `compare` | 2–4 篇论文**并排对比表** + 对比小结 |
| `reproduce` | 复现指南导出为可运行 `setup.sh` / Jupyter notebook |
| `search` / `batch` / `list` | 搜 arXiv · 批量分析整目录 · 本地论文库 |
| `cite` | 一键 BibTeX（不调模型） |

> 模型走 [litellm](https://github.com/BerriAI/litellm)：OpenAI / Anthropic / **DeepSeek** / 本地 **Ollama** 任选；分析结果与索引本地缓存，二次运行秒出。

## 30 秒上手

**零配置先看一眼**（内置离线回放，无需 key、无需联网）：

```bash
pip install paperis
papermind demo
```

> 📦 安装名是 **`paperis`**，命令名仍是 **`papermind`**（PyPI 上 `papermind` 已被占用）。

**免费，无需 API key**（用本地 [Ollama](https://ollama.com)）：

```bash
pip install paperis            # 或从源码： pip install -e ".[local-embeddings]"
ollama pull llama3.1             # 装好 Ollama 后拉一个模型
papermind analyze https://arxiv.org/abs/2307.08691 --model ollama/llama3.1 --format all -o ./report
```

**或用云端模型**（质量更高）：

```bash
pip install paperis
export OPENAI_API_KEY=sk-...
papermind analyze https://arxiv.org/abs/2307.08691 --format all --output ./report
```

> 没配 key 但本机跑着 Ollama 时，PaperMind 会**自动改用本地模型**，开箱即用。

输出（节选）：

```
╭──────────────────────────── Attention Is All You Need ──────────────────────╮
│ Ashish Vaswani et al. • 2017 • arXiv:1706.03762                             │
╰─────────────────────────────────────────────────────────────────────────────╯
🎯 贡献与创新点
  核心贡献: 提出完全基于注意力的 Transformer，去掉循环与卷积，在机器翻译上刷新 SOTA…
🔬 技术细节解释
  1. 缩放点积注意力 (Scaled Dot-Product Attention)   [high]
     Q·Kᵀ 缩放 √d_k 后 softmax 加权 V；缩放避免点积过大把 softmax 推入梯度饱和区…
     💡 类比: 像搜索引擎按匹配度加权汇总文档内容(V)。
     📊 AI 生成示意图 (Mermaid，见 Markdown 报告)
🛠️ 复现指南
  官方代码: https://github.com/tensorflow/tensor2tensor
  关键超参: d_model=512, h=8, d_ff=2048, N=6, warmup_steps=4000
```

完整样例见 [**Gallery**](examples/README.md)：[FlashAttention-2](examples/flashattention2.md) · [LLaMA 2](examples/llama2.md) · [Transformer](examples/transformer.md)（GitHub 直接渲染，含 Mermaid 图）。

---

## 为什么用 PaperMind

| 痛点 | PaperMind 怎么解决 |
| --- | --- |
| 摘要工具只给你"是什么"，不给"为什么/怎么做" | 主动挑出**最难懂**的技术点，直白解释 + 类比 + **原文图/AI 示意图** |
| 问答工具会一本正经地胡说 | 回答**分层标注**事实/推理/超纲，附**可跳转的原文出处** |
| 想复现，但环境、超参、报错全靠自己踩坑 | 完整**复现指南**：分步环境配置、性能基准、数据集直链、常见报错与修复 |
| 跑起来还是卡住 | **Tutor / Debug** 模式：贴报错直接给可运行的修复代码 |

## 安装

```bash
pip install paperis

# 可选：本地 embedding（配合 Ollama 实现完全离线、零成本）
pip install "paperis[local-embeddings]"
```

或从源码安装（开发）：

```bash
git clone https://github.com/Wenhao-Hua/papermind
cd papermind
pip install -e ".[dev]"
```

需要 Python 3.9+。PDF 解析用 PyMuPDF，向量检索用 faiss-cpu，模型统一走 litellm（OpenAI / Anthropic / Ollama 等）。

## 配置

API key 可通过环境变量或配置文件（`~/.papermind/config.json`）设置：

```bash
# 方式一：环境变量
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 方式二：写入配置文件
papermind config set openai-key sk-...
papermind config set anthropic-key sk-ant-...
papermind config set model gpt-4o          # 默认 gpt-4o-mini
papermind config set embedding-provider local   # 本地多语言 embedding（中文问英文论文也能检索）
papermind config show
```

环境变量优先级高于配置文件。解析结果与向量索引缓存在 `~/.papermind/cache/<arxiv_id>/`，二次运行直接命中缓存。

### 云端 + 本地，一键切换

配一次云端 key 当默认，任何命令加 `--local` 即**整条链路（LLM + embedding）切到本地 Ollama**，无需联网、零成本：

```bash
papermind config set openai-key sk-...        # 默认走云端（gpt-4o-mini）
papermind config set local-model ollama/llama3.1   # 可选：自定义本地模型

papermind analyze <source>                    # 云端
papermind analyze <source> --local            # 本地（需 Ollama 在跑 + 已装 local-embeddings）
papermind chat <source> --local               # chat/ask/tutor/debug/summary/compare/reproduce 同样支持 --local
```

> 没配任何 key 时，只要本机跑着 Ollama，PaperMind 会**自动**走本地，`--local` 可省略。

## CLI 用法

```bash
# 零配置看效果（离线回放，无需 key）
papermind demo

# 分析论文，生成四模块报告（md / json / html / all）
papermind analyze <source> --format all --output ./report
papermind analyze <source> --only contributions,technical    # 只跑部分模块
papermind analyze <source> --quick                           # 快速子集（贡献+技术，无图）
papermind analyze <source> --refresh                         # 忽略缓存重新分析
papermind analyze <source> --open                            # 生成后浏览器打开 HTML
papermind analyze <source> --estimate                        # 只估算 token/成本，不调用模型

# 快速 TL;DR（一次调用，便宜）
papermind summary <source>

# 交互式问答（多轮、带原文依据）
papermind chat <source> --mode balanced     # strict | balanced | explore
papermind ask <source> "调大某超参会怎样？"  --section 3.2    # 定向到某一节
papermind tutor <source>                                      # 复现全程辅导
papermind debug <source> --error "RuntimeError: ... only supports fp16 and bf16"

# 发现 / 批量 / 库 / 对比
papermind search "flash attention"                            # 搜 arXiv
papermind batch <src1> <src2> --dir ./pdfs -o ./reports       # 批量分析 + 索引
papermind compare <src1> <src2> [src3]                        # 2-4 篇并排对比表 + 小结
papermind list                                                # 你分析过的论文库

# 导出
papermind reproduce <source> --format both -o ./repro         # setup.sh + notebook
papermind cite <source>                                       # BibTeX

# 缓存与配置
papermind cache list        # 查看缓存（PDF/索引/报告）
papermind cache clear <key> # 或 --all
papermind open <source>     # 浏览器打开缓存的报告（--pdf 打开 PDF）
papermind config show
```

`<source>` 三种写法等价：`https://arxiv.org/abs/2307.08691`、`arxiv:2307.08691`、`2307.08691`，也可以是本地 `./paper.pdf`。

> **缓存复用**：同一篇 + 同模型 + 同模块的 `analyze` 结果会被缓存，二次运行**秒出、不重复花钱**，需要重算时加 `--refresh`。

### 三档推理强度（`--mode`）

- **strict** — 只答论文明写的事实，不做任何推理。
- **balanced**（默认）— 事实 + 必要推理，分层标注，推理给出依据与置信度。
- **explore** — 鼓励延伸推理与应用建议，仍区分事实/推理/超纲。

### 流式输出与 token 用量

`chat`/`ask`/`tutor`/`debug` 默认**流式生成**，实时显示回答（结构化分层结果在生成结束后渲染）；加 `--no-stream` 可关闭。每轮回答与会话/分析结束都会打印 **token 用量与估算成本**（如 `📊 用量: 3 calls · 4,120 tokens (prompt 3,800 / completion 320) · ~$0.0021`）。Python API 中对应 `report.usage` 与 `answer.usage`。

## Python API

```python
from papermind import analyze, PaperChat

# 一次性分析
report = analyze(
    source="arxiv:2307.08691",
    model="gpt-4o",                       # 默认 gpt-4o-mini
    modules=["contributions", "technical", "connections", "reproduction"],
)
print(report.contributions.main_contribution)
for point in report.technical.details:    # List[TechnicalPoint]，含 figure 字段
    print(point.name, point.difficulty, point.figure)
report.connections.related_works          # List[Connection]
report.reproduction.env_setup_steps       # List[SetupStep]
report.to_markdown("report.md")
report.to_json("report.json")
report.to_html("report.html")            # 自包含可分享网页
report.to_setup_script("setup.sh")       # 复现指南 → 可运行脚本
report.to_notebook("repro.ipynb")        # 复现指南 → Jupyter notebook
report.usage                             # 本次分析 token 用量/成本

# 快速 TL;DR（一次调用）
from papermind.summarize import summarize
summary, usage = summarize("arxiv:2307.08691")
print(summary.tldr, summary.key_points)

# 多论文对比（复用各自缓存）
from papermind import compare
cmp = compare(["arxiv:2307.08691", "arxiv:1706.03762"])
cmp.to_markdown("compare.md"); cmp.to_html("compare.html")

# 多轮问答 + 分层推理
chat = PaperChat("arxiv:2307.08691", mode="balanced")
ans = chat.ask("如果把 block_size 调大到 256 会怎样？", section="3.2")  # 可定向到某节
for seg in ans.segments:                  # kind = fact | inference | out_of_scope
    print(seg.kind, seg.confidence, seg.text)
ans.evidence                              # 原文依据片段（含 section/page）
ans.sources                               # [{"section": "3.1", "page": 5}, ...]

# 辅导 / 调试
chat.tutor("怎么把这个方法迁移到我自己的 decoder-only 模型？")
chat.debug("RuntimeError: CUDA out of memory")
```

## 输出格式

- **JSON** — 由 [`papermind/output/schema.py`](papermind/output/schema.py) 的 pydantic 模型定义，作为单一数据源。
- **Markdown** — 原文出处渲染成可点击的 `[Section 3.1](pdf#page=5)`；论文原图以图片嵌入，AI 示意图以 ` ```mermaid ` 代码块呈现；复现步骤/性能表/报错以表格与代码块组织。
- **终端** — rich 彩色分模块展示，技术点难度用颜色区分，附分析进度条。

## 图示能力

技术点的图示遵循「**先原图，后 AI 图**」：

1. 用 PyMuPDF 提取论文原图：**嵌入的位图**直接抽取；**矢量图**（无位图的结构图）则按 caption 定位、渲染该页区域为 PNG。再由 LLM 按语义把图匹配到对应技术点，标注 Figure 编号与来源。
2. 没有合适原图时，由 LLM 生成 **Mermaid 示意图**（数据流/模块结构/训练流程），明确标注「AI 生成示意图」。

## 工作原理

```
source ──▶ parser ──▶ ┌ contributions ┐
(arxiv/pdf)  (pdf+meta) │ technical      │──▶ Report ──▶ markdown / json / terminal
                        │ connections    │       │
                        └ reproduction ──┘       └─ figures (原图匹配 + Mermaid)

chat:  parser ──▶ chunk + embed ──▶ FAISS (缓存) ──▶ retrieve ──▶ 分层回答(事实/推理/超纲)
```

依赖刻意保持精简，**不使用 LangChain**：`typer` · `rich` · `pydantic` · `pymupdf` · `litellm` · `faiss-cpu`。

## Roadmap

- [x] 流式输出（`chat`/`ask`/`tutor`/`debug` 实时生成，`--no-stream` 关闭）
- [x] token 用量与成本统计（每轮 + 会话/分析合计）
- [x] 向量图（vector figure）渲染抽取，覆盖更多论文图
- [x] 分析结果缓存复用、缓存管理、HTML 导出与一键打开
- [x] 复现指南导出为可运行 `setup.sh` / Jupyter notebook、BibTeX 引用
- [x] arXiv 搜索、批量分析、本地论文库、快速 TL;DR、成本预估、定向问答
- [x] 多论文对比分析（`papermind compare a b` — 贡献/方法/基准并排成表 + 对比小结）
- [ ] OCR 支持（扫描版 PDF）

## 贡献

欢迎 PR / issue，见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

[MIT](LICENSE)
