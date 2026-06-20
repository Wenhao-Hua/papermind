# 📚 PaperMind Gallery — 不用安装，先看效果

下面是 PaperMind 对 **7 篇经典论文、跨 5 个领域** 的完整分析报告。**直接点开就能在 GitHub 上看到渲染效果**（论文式教学示意图 + 读图讲解、可跳转的原文出处、复现指南表格）——这正是 `papermind analyze <paper>` 的真实产物形态。

> 想看交互式问答与复现辅导的效果？见仓库 [README](../README.zh-CN.md) 的终端演示与用法。

| 领域 | 论文 | PaperMind 抽取了什么 |
| --- | --- | --- |
| **NLP / 注意力** | [Attention Is All You Need](transformer.md) · 2017 | 缩放点积注意力、位置编码、多头注意力、自注意力短路径、Noam 学习率等技术点（含论文式示意图 + 读图讲解）；复现指南指向 `tensor2tensor` |
| | [FlashAttention‑2](flashattention2.md) · 2023 | 改进的在线 Softmax、序列维并行、Warp 间分区、块大小调优等；复现指南指向官方 `flash-attention` |
| | [Llama 2](llama2.md) · 2023 | 安全‑帮助性奖励合成、PPO 的 KL 惩罚、拒绝采样微调、分组查询注意力(GQA)；复现指南与关键超参 |
| **计算机视觉** | [ViT — An Image is Worth 16×16 Words](vit.md) · 2020 | 图像分块嵌入、ViT 编码器块与预归一化、CLS/池化策略等；复现指南指向 `google-research/vision_transformer` |
| **生成 / 扩散** | [Latent Diffusion (Stable Diffusion)](latent-diffusion.md) · 2021 | 潜空间扩散、感知压缩自编码器、两阶段训练、交叉注意力条件等；复现指南指向 `CompVis/latent-diffusion` |
| **强化学习** | [PPO — Proximal Policy Optimization](ppo.md) · 2017 | 裁剪替代目标、KL 惩罚、广义优势估计、多轮小批量更新等技术点（含示意图 + 读图讲解） |
| **高效微调** | [LoRA — Low‑Rank Adaptation](lora.md) · 2021 | 低秩增量分解、冻结预训练权重、秩/缩放超参、零额外推理延迟等；复现指南指向 `microsoft/LoRA` |

每篇报告都覆盖四个模块：**🎯 贡献与创新点 · 🔬 技术细节解释（含图示）· 🔗 知识关联 · 🛠️ 复现指南**，技术点附原文出处与难度分级，知识关联给出参考论文的 arXiv 链接。

## 生成你自己的报告

```bash
pip install papermind-ai

# 有 OpenAI / Anthropic / DeepSeek key：
export OPENAI_API_KEY=sk-...
papermind analyze https://arxiv.org/abs/2307.08691 --format all --output ./report

# 没有 key？用本地 Ollama，完全免费（先在 https://ollama.com 装好并 `ollama pull llama3.1`）：
pip install "papermind-ai[local-embeddings]"
papermind analyze https://arxiv.org/abs/2307.08691 --model ollama/llama3.1 --format html -o ./report
```

`--format all` 会同时产出 `report.md` / `report.json` / `report.html`（自包含、可分享的网页报告）。

> 这些样例都是 `papermind analyze` **真实跑出来的产物**（非手绘、非杜撰）。模型有随机性，对同一篇论文重跑会得到内容相近、但不完全相同的报告。
