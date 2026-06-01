# 📚 PaperMind Gallery — 不用安装，先看效果

下面是 PaperMind 对三篇经典论文的完整分析报告。**直接点开就能在 GitHub 上看到渲染效果**（包括 Mermaid 示意图、可跳转的原文出处、复现指南表格）——这正是 `papermind analyze <paper>` 的真实产物形态。

> 想看交互式问答与复现辅导的效果？见仓库根目录 [README](../README.md) 的终端截图与用法。

| 论文 | 年份 | PaperMind 抽取了什么 |
| --- | --- | --- |
| **[FlashAttention-2](flashattention2.md)** | 2023 | Warp 工作划分、减少 non-matmul、序列维并行 3 个技术点（含 Mermaid 数据流图）；A100 利用率 50–73% 的性能基准；`flash-attn` 编译坑点与 fp16 报错修复 |
| **[Llama 2](llama2.md)** | 2023 | 双奖励模型 RLHF、Ghost Attention、GQA；gated repo / OOM / chat 模板三类常见报错；QLoRA 微调建议 |
| **[Attention Is All You Need](transformer.md)** | 2017 | 缩放点积注意力、多头注意力、位置编码（含两张 Mermaid 结构图）；Noam 学习率 + √d_k 缩放等复现关键点 |

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
