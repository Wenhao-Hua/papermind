# LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang et al.  •  **Year:** 2021  •  **arXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)  •  [PDF](https://arxiv.org/pdf/2106.09685.pdf)

- [🎯 贡献与创新点](#contributions)
- [🔬 技术细节解释](#technical)
- [🛠️ 复现指南](#reproduction)

<a id="contributions"></a>

## 🎯 贡献与创新点

**核心贡献:** 提出 Low-Rank Adaptation (LoRA) 方法，冻结预训练模型权重，注入可训练的低秩分解矩阵，极大减少下游任务可训练参数，且无推理延迟。

**新颖之处:** 将权重更新表示为低秩分解，并行于原权重，可在部署时合并，从而避免 adapter 等方法引入的推理延迟，同时参数效率远超 full fine-tuning。

**解决的问题:** 解决大模型全微调部署成本高昂的问题，以及现有参数高效方法（如 adapter 增加推理延迟、prefix tuning 难优化）的缺陷。

> **原文出处:**
> - [Abstract (p.1)](https://arxiv.org/pdf/2106.09685.pdf#page=1): ⚠️ 未核实 · We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. ... LoRA performs on-par or better than fine-tuning ... and, unlike adapters, no additional inference latency.
> - [4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES (p.4)](https://arxiv.org/pdf/2106.09685.pdf#page=4): No Additional Inference Latency. When deployed in production, we can explicitly compute and store W = W0 + BA and perform inference as usual.
> - [3 AREN’T EXISTING SOLUTIONS GOOD ENOUGH? (p.3)](https://arxiv.org/pdf/2106.09685.pdf#page=3): adapter layers have to be processed sequentially. This makes a difference in the online inference setting ... we see a noticeable increase in latency when using adapters
> - [Frontmatter (p.1)](https://arxiv.org/pdf/2106.09685.pdf#page=1): full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive.

<a id="technical"></a>

## 🔬 技术细节解释

### 1. 低秩参数化更新矩阵  `🔴 high`

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将其适应过程中的更新 $\Delta W$ 约束为低秩分解 $BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。前向传播变为 $h = W_0 x + BA x$，训练时 $W_0$ 冻结，仅优化 $A$ 和 $B$。这极大减少了可训练参数的数量，同时保留了模型对下游任务的适应能力。

$$
h = W_0 x + B A x
$$

> 💡 **类比:** 就像在不改变原有雕像的前提下，仅添加一些小巧的黏土补件来修改雕像的外观，这些补件本身由更少的参数描述。

📍 出处: [Section 4.1 (p.4)](https://arxiv.org/pdf/2106.09685.pdf#page=4)

![教学示意图：低秩参数化更新矩阵](figures/lora-fig1.svg)
*教学示意图：低秩参数化更新矩阵（教学示意图）*

> **读图**：LoRA通过低秩分解减少微调参数量，冻结预训练权重。
>
> - h = W₀ x + B A x：前向传播公式，含低秩更新。
> - ΔW = B A：低秩分解，r ≪ min(d,k)。
> - W₀冻结，仅优化A和B，大幅减少参数量。
> - 对比全微调175B参数，LoRA仅17.5M，显存降约3倍。
>
> **关键**：LoRA在注意力层注入低秩矩阵，高效适配下游任务。

### 2. LoRA 的缩放因子  `🟡 mid`

LoRA 将输出 $\Delta W x$ 乘以 $\frac{\alpha}{r}$，其中 $\alpha$ 是常数。论文指出，在使用 Adam 优化器时，调整 $\alpha$ 大致等同于调整学习率，因此他们将 $\alpha$ 固定为首次尝试的秩 $r$ 值，不再专门调节。这一策略减少了在不同 $r$ 下需要重新调参的麻烦。

$$
\text{scaled output} = \frac{\alpha}{r} \Delta W x
$$

> 💡 **类比:** 就像在添加黏土补件时，先用一个缩放旋钮控制补件的影响力，这个旋钮与学习率联动，使得更换不同大小的补件时无需重新校准旋钮。

📍 出处: [Section 4.1 (p.4)](https://arxiv.org/pdf/2106.09685.pdf#page=4)

![教学示意图：LoRA 的缩放因子](figures/lora-fig2.svg)
*教学示意图：LoRA 的缩放因子（教学示意图）*

> **读图**：LoRA缩放因子α/r的定义与作用
>
> - 缩放输出：h = W₀x + (α/r)ΔWx
> - ΔW = BA，秩r远小于d
> - α为常数，调整α≈调整学习率
> - 例：α固定=8时，r越大缩放因子越小
>
> **关键**：α设为首次尝试的r值，避免不同r下重新调参

### 3. LoRA 可泛化为全微调  `🟡 mid`

当将 LoRA 应用于所有权重矩阵并训练所有偏置项时，若将 LoRA 的秩 $r$ 设置为预训练权重矩阵本身的秩，则可以大致恢复全微调的表达能力。这表明 LoRA 是一种更广义的微调形式，全微调是其特例。

> 💡 **类比:** 当补件的复杂程度与原有结构完全相当时，就相当于重新塑造了整个雕像。

📍 出处: [Section 4.1 (p.4)](https://arxiv.org/pdf/2106.09685.pdf#page=4)

![教学示意图：LoRA 可泛化为全微调](figures/lora-fig3.svg)
*教学示意图：LoRA 可泛化为全微调（教学示意图）*

> **读图**：LoRA 可泛化为全微调，全微调是其特例
>
> - LoRA 更新：h = W₀x + BAx，冻结 W₀，训练 A、B
> - 全微调：h = (W₀ + ΔW_full)x，更新所有参数
> - 当 r = d 时，LoRA 可表示任意 ΔW，等价全微调
>
> **关键**：LoRA 表达能力 ≥ 全微调（r 足够大时）

### 4. LoRA 的初始化策略  `🟢 low`

LoRA 对 $A$ 使用随机高斯初始化，对 $B$ 使用零初始化，因此在训练开始时 $\Delta W = BA = 0$。这保证模型从原始的预训练权重出发，逐步学习任务相关的低秩更新，避免一开始就引入随机扰动。

$$
A \sim \mathcal{N}(0, \sigma^2), B = 0
$$

> 💡 **类比:** 刚开始时，黏土补件是完全透明的，雕像保持原样；随着训练的进行，补件逐渐成型，赋予雕像新的特征。

📍 出处: [Section 4.1 (p.4)](https://arxiv.org/pdf/2106.09685.pdf#page=4)

### 5. LoRA 在 Transformer 上的应用  `🟢 low`

LoRA 仅应用于自注意力模块的权重矩阵（$W_q, W_k, W_v, W_o$），并冻结整个 MLP 模块（即下游任务不训练 MLP 层）。在大多数实验中，只对 $W_q$ 和 $W_v$ 应用 LoRA，以兼顾简单性和参数效率。

> 💡 **类比:** 只修改发动机中几个关键活塞的特性，而保持引擎的其他部分完全不变。

📍 出处: [Section 4.2 (p.5)](https://arxiv.org/pdf/2106.09685.pdf#page=5)

![教学示意图：LoRA 在 Transformer 上的应用](figures/lora-fig4.svg)
*教学示意图：LoRA 在 Transformer 上的应用（教学示意图）*

> **读图**：LoRA通过低秩更新矩阵适配Transformer自注意力层。
>
> - 预训练权重W冻结，不更新梯度。
> - 低秩更新ΔW=BA，B和A可训练，r≪d。
> - LoRA仅应用于W_q和W_v，MLP冻结。
> - 输出为Wx+BAx，无推理延迟。
>
> **关键**：LoRA仅微调自注意力的W_q和W_v，大幅减少参数量。

### 6. 无额外推理延迟的部署  `🟢 low`

生产部署时，可以预先计算 $W = W_0 + BA$ 并存储，推理时就像使用普通模型一样。切换下游任务时，只需减去旧的 $BA$ 并加上新的 $B'A'$，几乎没有额外内存开销。通过这种构造，LoRA 保证与全微调模型相比不会引入额外的推断延迟。

$$
W = W_0 + B A
$$

> 💡 **类比:** 就像提前将新轮胎组装到轮毂上，行驶时无需任何额外操作；换胎只需快速拆装，毫不影响驾驶速度。

📍 出处: [Section 4.1 (p.4)](https://arxiv.org/pdf/2106.09685.pdf#page=4)

![教学示意图：无额外推理延迟的部署](figures/lora-fig5.svg)
*教学示意图：无额外推理延迟的部署（教学示意图）*

> **读图**：LoRA通过低秩分解实现零额外推理延迟的部署
>
> - 训练时注入低秩矩阵B和A，前向传播h=W0x+BAx
> - 部署时合并为单一权重矩阵W=W0+BA，推理零延迟
> - 任务切换时仅更新BA部分，无需重载预训练权重
> - 低秩分解(r<<d)将参数量从d²降至2rd
>
> **关键**：LoRA部署时合并权重，推理延迟与标准模型完全相同

<a id="reproduction"></a>

## 🛠️ 复现指南

- **官方代码（已核实 · 论文原文链接 ✓官方 · ★13576）:** [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)
- **安装 / 运行（取自仓库）:**
  - `pip install -e .`
- **推荐硬件:** NVIDIA Tesla V100 (as used in experiments) or higher
- **关键超参数:** `r=4 (for GPT-3 175B, but adjustable; low rank such as 4 or 8 typical)`, `target_modules=['q','v'] (apply LoRA to query and value projection matrices)`, `alpha set equal to first r tried (e.g., 4) and not tuned further`, `dropout applied to LoRA layers (rate unspecified in paper)`

### 环境配置步骤

**1. Clone repository**

Clone the official LoRA GitHub repository.

```bash
git clone https://github.com/microsoft/LoRA.git && cd LoRA
```

**2. Install PyTorch**

Install PyTorch with CUDA support (version 1.12 or later recommended).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Install dependencies**

Install required packages, including transformers and datasets.

```bash
pip install -r requirements.txt
```

**4. Verify installation**

Run a quick test to ensure LoRA modules can be imported.

```bash
python -c 'import loralib; print("LoRA installed successfully")'
```

### 性能基准

| 设置 | Baseline | 结果 | 加速 | 显存 |
| --- | --- | --- | --- | --- |
| GPT-3 175B training throughput | 32.5 tokens/s per V100 GPU (full fine-tuning) | 43.1 tokens/s per V100 GPU (LoRA) | 1.33x (25% training speedup) | - |
| GPT-3 175B training VRAM | 1.2 TB (full fine-tuning) | 350 GB (LoRA) | - | 3x reduction in GPU memory requirement |
| GPT-3 175B checkpoint size | 350 GB per model (full fine-tuning) | 35 MB per task (LoRA) | - | ~10,000x reduction in storage for adapted models |
| GPT-2 medium inference latency (batch=1, seq_len=128) | 23.9±2.1 ms (AdapterL) | 19.8±2.7 ms (Fine-Tune/LoRA) | LoRA matches fine-tuning, ~20.7% faster than AdapterL | - |

### 数据集

- **[GLUE benchmark](https://huggingface.co/datasets/glue)** — Natural Language Understanding tasks (MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B)
- **[E2E NLG Challenge](http://www.macs.hw.ac.uk/InteractionLab/E2E/)** — End-to-end natural language generation
- **[WikiSQL](https://huggingface.co/datasets/wikisql)** — Natural language to SQL queries
- **[SAMSum](https://huggingface.co/datasets/samsum)** — Conversation summarization

### 常见报错与解决

- **报错:** `CUDA out of memory when training large models (e.g., GPT-3 175B)`
  - 原因: Model and optimizer states exceed GPU VRAM despite LoRA savings; model parallelism still required for very large models.
  - 修复: `Reduce batch size, enable gradient accumulation, or increase the number of GPUs with model parallelism (e.g., use DeepSpeed).`
- **报错:** `KeyError: 'lora_A' when loading a checkpoint`
  - 原因: Checkpoint was saved with a different LoRA configuration (e.g., different set of target modules or rank).
  - 修复: `Ensure the model definition matches the checkpoint exactly. Set the same `r` and target modules before loading.`
- **报错:** `ImportError: No module named 'loralib'`
  - 原因: The LoRA package is not installed or PYTHONPATH not set.
  - 修复: `Install from source: pip install -e . inside the cloned repository.`

### ⚠️ 坑点提示

- Alpha hyperparameter: Set α equal to the first rank you try (e.g., 4) and do not tune it. The scaling factor α/r helps reduce the need to retune learning rates when changing r.
- Weight matrices choice: Applying LoRA to Wq and Wv is sufficient for most tasks; adding Wk or Wo may improve performance marginally but increases parameters.
- Rank selection: A very low rank (e.g., r=4 or even r=1 or 2) often works well for large models; higher rank does not necessarily give better performance and may cause overfitting.
- Inference latency: To avoid extra latency, merge LoRA weights (W0 + BA) before deployment. Do not merge if you need to batch requests for different tasks in a single forward pass.


---
*Generated by [PaperMind](https://github.com/Wenhao-Hua/papermind).*