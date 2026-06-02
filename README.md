# PaperMind

> **把一篇论文读懂、读到能复现 —— 每句话都有原文出处。**（arXiv 链接或任意 PDF 直链）
> *Understand any paper from its URL: structured analysis, grounded & cited Q&A, runnable reproduction.*

[![CI](https://github.com/Wenhao-Hua/papermind/actions/workflows/ci.yml/badge.svg)](https://github.com/Wenhao-Hua/papermind/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/paper-mind)](https://pypi.org/project/paper-mind/)
[![Python](https://img.shields.io/pypi/pyversions/paper-mind)](https://pypi.org/project/paper-mind/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<!-- 录制后取消注释（见 docs/DEMO.md）：
<p align="center"><img src="docs/demo.gif" alt="PaperMind demo" width="820"></p>
-->
<p align="center">
  <a href="examples/transformer.md"><b>▶ 看真实样例报告</b></a> ·
  <a href="examples/README.md">Gallery</a>（GitHub 直接渲染，含 Mermaid 图）
</p>

## 三种用法，挑最方便的

| | 怎么用 | 适合 |
| --- | --- | --- |
| 🌐 **在线用** | 打开 **[papermind.try2026.cn](https://papermind.try2026.cn)**，零安装 | 想立刻试、不装东西 |
| 🖥️ **图形界面** | `pip install "paper-mind[web]"` → `papermind ui` | 本地全功能、浏览器操作 |
| ⌨️ **命令行 / API** | `pip install paper-mind` → `papermind analyze https://arxiv.org/abs/2307.08691` | 脚本化、批量、集成 |

> 🆓 不想花钱：`papermind demo` 离线看效果（无需 key）；任何命令加 `--local` 走本地 Ollama，全程零成本。

## 效果

```
╭──────────────── Attention Is All You Need · 2017 · arXiv:1706.03762 ───────────────╮
🎯 核心贡献   提出完全基于注意力的 Transformer，去掉循环与卷积，机器翻译刷新 SOTA…
🔬 技术细节   1. 缩放点积注意力 [high]  Q·Kᵀ 除以 √d_k 再 softmax 加权 V；缩放避免梯度饱和
              💡 类比：像搜索引擎按匹配度加权汇总文档；📊 附结构示意图（教学 SVG）
💬 问答       「为什么除以 √d_k？」→【论文事实】维度大时点积方差大… 📌 出处 Section 3.2.1 (p.4) ✓已核验
🛠️ 复现       官方代码（已核实·★24k）github.com/... → setup.sh（真实依赖 + README 运行命令）
```

完整渲染样例：[Transformer](examples/transformer.md) · [FlashAttention-2](examples/flashattention2.md) · [LLaMA 2](examples/llama2.md)

## 凭什么不一样

- **🔎 答案有据可查、自动核验**：分层标注「事实 / 推理(带置信度) / 超纲」，出处逐条对着原文核验，核不到标 ⚠️。
- **🛠️ 复现接论文的真实代码仓库**：自动找官方仓库，用仓库里**真实的依赖与运行命令**生成 `setup.sh`，不是模型瞎编。
- **🆓 零配置、本地免费**：`demo` 离线看；`--local` 全本地；没 key 自动回退本地。

## 📊 证据检索是自训练 + 实测的，不是黑箱

在 **QASPER** 上自训练 cross-encoder 重排器（`bge-reranker-base`），全段落候选下相对强稠密基线：

| | Recall@5 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| Dense (`bge-small-en-v1.5`) | 0.519 | 0.463 | 0.469 |
| **+ 自训练 Reranker** | **0.660** | **0.612** | **0.609** |

dev（888 题）与独立 test（1309 题）一致，无过拟合。复现：[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)（`trainer/` 训练 · `evaluation/` 评测）。

## 维护一个在线服务（自托管）

同一套网页，Docker 一行起服务，公网暴露也安全（默认演示模式只读缓存、不烧 key）：

```bash
git clone https://github.com/Wenhao-Hua/papermind && cd papermind
docker build -t papermind . && docker run -p 8080:8080 papermind
docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... papermind \
       papermind serve --host 0.0.0.0 --port 8080 --live    # 实时分析（你的 key 付费）
```

> `--live` 默认**带限流**（每 IP 8 次/天、全局 300 次/天，可用 `--rate-per-ip` / `--rate-global` 调，`0` 表示不限）——公开挂出去也不会被陌生人刷爆你的 key。免费操作（搜索 / 缓存 / 离线 demo）不受限。

## 能做什么

`analyze`（四模块报告）· `summary`（TL;DR）· `ask`/`chat`（带依据问答）· `tutor`/`debug`（复现辅导）· `compare`（多篇对比）· `reproduce`（导出 setup.sh/notebook）· `search`/`batch`/`list` · `cite`。

模型走 [litellm](https://github.com/BerriAI/litellm)：OpenAI / Anthropic / DeepSeek / Gemini / 本地 Ollama 任选；结果本地缓存，二次运行秒出。更多用法：`papermind --help`。

## 贡献 / License

欢迎 PR / issue（[CONTRIBUTING.md](CONTRIBUTING.md)）· [MIT](LICENSE)
