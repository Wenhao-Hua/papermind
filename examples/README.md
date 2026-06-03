# 📚 PaperMind Gallery — 不用安装，先看效果

下面是 PaperMind 对三篇经典论文的完整分析报告。**直接点开就能在 GitHub 上看到渲染效果**（包括论文式结构示意图与"读图"讲解、可跳转的原文出处、复现指南表格）——这正是 `papermind analyze <paper>` 的真实产物形态。

> 想看交互式问答与复现辅导的效果？见仓库根目录 [README](../README.md) 的终端截图与用法。

| 论文 | 年份 | PaperMind 抽取了什么 |
| --- | --- | --- |
| **[FlashAttention-2](flashattention2.md)** | 2023 | 改进的在线 Softmax(Logsumexp)、序列维并行、Warp 间 split-Q 分区、块大小调优等 6 个技术点（含 2 张示意图 + 读图讲解）；复现指南指向官方 `flash-attention` 仓库 |
| **[Llama 2](llama2.md)** | 2023 | 安全-帮助性奖励合成、PPO 中的 KL 惩罚、拒绝采样微调、分组查询注意力(GQA) 等技术点（含 3 张示意图 + 读图讲解）；复现指南与关键超参 |
| **[Attention Is All You Need](transformer.md)** | 2017 | 缩放点积注意力、位置编码、多头注意力、自注意力短路径、Noam 学习率等 6 个技术点（含 3 张示意图 + 读图讲解）；复现指南指向官方 `tensor2tensor` 仓库 |

每篇报告都覆盖四个模块：**🎯 贡献与创新点 · 🔬 技术细节解释（含图示）· 🔗 知识关联 · 🛠️ 复现指南**，技术点附原文出处与难度分级，知识关联给出参考论文的 arXiv 链接。

## 生成你自己的报告

```bash
pip install -e .

# 有 OpenAI / Anthropic key：
export OPENAI_API_KEY=sk-...
papermind analyze https://arxiv.org/abs/2307.08691 --format all --output ./report

# 没有 key？用本地 Ollama，完全免费（先 https://ollama.com 装好并 `ollama pull llama3.1`）：
pip install -e ".[local-embeddings]"
papermind analyze https://arxiv.org/abs/2307.08691 --model ollama/llama3.1 --format html -o ./report
```

`--format all` 会同时产出 `report.md` / `report.json` / `report.html`（自包含、可分享的网页报告）。

> 说明：本目录中的样例为**示例输出（illustrative）**，用于展示报告形态；用真实模型对同一篇论文运行会得到内容相近、由模型实时生成的报告。
