# PaperMind

**[English](README.md) · 简体中文**

> **把任意论文读懂、读到能复现 —— 关键判断都标原文出处、逐条核验。**
> 输入 arXiv 链接 / DOI / 论文标题 / 论文页面或 PDF 直链（也可上传 PDF）：结构化分析、带原文依据并核验的问答、整篇方法的**框架图**（可下载 SVG）、可运行的复现指南。

[![CI](https://github.com/Wenhao-Hua/papermind/actions/workflows/ci.yml/badge.svg)](https://github.com/Wenhao-Hua/papermind/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/paper-mind)](https://pypi.org/project/paper-mind/)
[![Python](https://img.shields.io/pypi/pyversions/paper-mind)](https://pypi.org/project/paper-mind/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<p align="center">
  <img src="examples/figures/transformer-fig2.svg" alt="PaperMind 为论文技术点生成的论文式教学示意图" width="760">
</p>
<p align="center">
  <sub>↑ PaperMind 为论文技术点<b>生成的论文式教学示意图</b>（管线真实产物）·
  <a href="examples/transformer.md"><b>看完整样例报告</b></a> · <a href="examples/README.md">样例画廊 →</a></sub>
</p>

## 凭什么不是「又一个 chat-with-PDF」

- **🔬 我们自训练了检索重排器 —— 相对强稠密基线 Recall@5 +14pt**（QASPER 独立 test）。多数「和论文对话」工具只是套 API；PaperMind 的证据检索是我们**自己微调并实测**的 cross-encoder（见 [实测数据](#实测重排器是自训练实测的不是黑箱)）。
- **🔎 关键判断分层标注、逐条核验。** 回答分 **论文事实 / 推理(带置信度) / 超纲**，每条引用都对着原文核验——核不到就标 ⚠️，绝不默默当真。
- **📐 整篇方法的框架图。** 除了逐技术点的配图，PaperMind 还把论文的端到端方法重建成一张 Figure‑1 式的总览架构图（论文只隐含、未画出的步骤标注为*推断*）——在网页 `/framework` 页查看，可下载为 SVG。
- **🛠️ 复现接论文的真实代码仓库。** 自动定位论文的代码仓库，用仓库里**真实的依赖文件和 README 运行命令**生成 `setup.sh`，不是模型瞎猜。
- **🆓 全本地、零成本可跑。** `papermind demo` 离线看；`--local` 全程走 Ollama；没 key 自动回退本地。

| | **PaperMind** | 常见 chat-with-PDF / SaaS 阅读器 |
| --- | :---: | :---: |
| 逐条引用**并核验**（核不到标 ⚠️） | ✅ | — |
| 复现 → 接**真实仓库**的可运行 `setup.sh` | ✅ | — |
| **自训练** RAG 重排器、QASPER 实测 | ✅ | — |
| **全本地 / 离线、零成本**运行 | ✅ | ✗ |
| **开源、可自托管** | ✅ | ✗ |

<sub>“—” = 据我们所知对方未公开该能力（欢迎 issue 指正）；开源/本地两行对闭源 SaaS 是明确的。</sub>

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
              💡 类比：像搜索引擎按匹配度加权汇总文档；📊 附论文式教学示意图（SVG）
💬 问答       「为什么除以 √d_k？」→【论文事实】维度大时点积方差大… 📌 出处 Section 3.2.1 (p.4) ✓已核验
🛠️ 复现       代码仓库（★24k）github.com/... → setup.sh（真实依赖 + README 运行命令）
```

完整渲染样例（GitHub 直接打开——论文式示意图、可跳转原文出处、复现表格）：
**[Transformer](examples/transformer.md)** · **[ViT](examples/vit.md)** · **[PPO](examples/ppo.md)** · **[完整画廊 →](examples/README.md)**

## 样例画廊 —— 7 篇论文，跨 5 个领域

PaperMind 不只懂 NLP。下面都是管线真实跑出来的报告，按领域分组：

| 领域 | 论文 |
| --- | --- |
| **NLP / 注意力** | [Transformer](examples/transformer.md) · [FlashAttention‑2](examples/flashattention2.md) · [Llama 2](examples/llama2.md) |
| **计算机视觉** | [ViT](examples/vit.md) |
| **生成 / 扩散** | [Latent Diffusion (Stable Diffusion)](examples/latent-diffusion.md) |
| **强化学习** | [PPO](examples/ppo.md) |
| **高效微调** | [LoRA](examples/lora.md) |

## 实测——重排器是自训练、实测的，不是黑箱

在 **QASPER** 上自训练 cross-encoder 重排器（`bge-reranker-base`），全段落候选下相对强稠密基线：

| | Recall@5 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| Dense (`bge-small-en-v1.5`) | 0.519 | 0.463 | 0.469 |
| **+ 自训练 Reranker** | **0.660** | **0.612** | **0.609** |

dev（888 题）与独立 test（1309 题）一致，无过拟合。复现：[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)（`trainer/` 训练 · `evaluation/` 评测）。

## 自托管在线服务

Docker 一行起服务，公网暴露也安全（默认演示模式只读缓存、不烧 key）：

```bash
git clone https://github.com/Wenhao-Hua/papermind && cd papermind
docker build -t papermind . && docker run -p 8080:8080 papermind            # 演示模式

docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... -e PAPERMIND_TRUST_PROXY=1 papermind \
       papermind serve --host 0.0.0.0 --port 8080 --live                    # 实时分析（你的 key 付费）
```

> **`--live` 默认带限流**（每 IP 8 次/天、全局 300 次/天，可用 `--rate-per-ip` / `--rate-global` 调，`0` 表示不限）——公开挂出去也不会被陌生人刷爆你的 key。免费操作（搜索 / 缓存 / 离线 demo）不受限。
>
> **部署在 Cloudflare / 反向代理后面时，要设 `PAPERMIND_TRUST_PROXY=1`** ——否则所有访客共用同一个 per-IP 配额（代理的 IP），per-IP 限流会静默失效。仅当确实在会设置 `cf-connecting-ip` 的可信代理后面才开启。
>
> **想加快出图**：配图默认用主模型（布局最干净但慢）。给配图单独配一个快速的非思考模型即可大幅提速：`papermind config set figure-model deepseek/deepseek-chat`。

## 能做什么

`analyze`（四模块报告）· `summary`（TL;DR）· `ask`/`chat`（带依据问答）· `tutor`/`debug`（复现辅导）· `compare`（多篇对比）· `reproduce`（导出 setup.sh/notebook）· `search`/`batch`/`list` · `cite`。

模型走 [litellm](https://github.com/BerriAI/litellm)：OpenAI / Anthropic / DeepSeek / Gemini / Qwen（百炼）/ 本地 Ollama 任选；结果本地缓存，二次运行秒出。更多用法：`papermind --help`。

## 贡献 / License

欢迎 PR / issue（[CONTRIBUTING.md](CONTRIBUTING.md)）· [MIT](LICENSE)
