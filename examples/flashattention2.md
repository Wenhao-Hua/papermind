<!-- PaperMind 示例输出（illustrative sample）。运行 `papermind analyze https://arxiv.org/abs/2307.08691` 可生成你自己的版本。-->

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**Authors:** Tri Dao  •  **Year:** 2023  •  **arXiv:** [2307.08691](https://arxiv.org/abs/2307.08691)  •  [PDF](https://arxiv.org/pdf/2307.08691.pdf)

- [🎯 贡献与创新点](#contributions)
- [🔬 技术细节解释](#technical)
- [🔗 知识关联](#connections)
- [🛠️ 复现指南](#reproduction)

<a id="contributions"></a>

## 🎯 贡献与创新点

**核心贡献:** 提出FlashAttention-2算法，通过改进的并行性和工作分区实现注意力计算速度约2倍提升，达到理论最大FLOPs/s的50-73%。

**新颖之处:** 相比FlashAttention，调整算法减少非矩阵乘法FLOPs，并在序列长度维度上并行化注意力计算以提高占用率，同时优化线程块内warp之间的工作分区以减少共享内存通信。

**解决的问题:** FlashAttention由于GPU上线程块和warp之间工作分区次优，导致低占用率或不必要的共享内存读写，无法充分利用计算资源（仅达25-40%理论最大FLOPs/s）。

> **原文出处:**
> - [Frontmatter (p.1)](https://arxiv.org/pdf/2307.08691.pdf#page=1): We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2× speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100...
> - [Frontmatter (p.1)](https://arxiv.org/pdf/2307.08691.pdf#page=1): We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes.

<a id="technical"></a>

## 🔬 技术细节解释

### 1. 改进的在线Softmax：维护未缩放输出与Logsumexp存储  `🔴 high`

在遍历key/value块时，不立即用$\ell^{(j)}$重新缩放输出，而是保留未缩放的$\tilde{O}^{(j)}$，仅在最后除以$\ell^{(\text{last})}$。同时只存储logsumexp $L = m + \log(\ell)$，免去存储$m$和$\ell$两个值。这大幅减少了每次迭代中的非矩阵乘FLOPs（如除法、指数、标量运算），因为GPU的矩阵乘单元速度是标量单元的约16倍，将更多时间用于矩阵乘可提升吞吐量。

$$
\tilde{O}^{(j)} = \operatorname{diag}(e^{m^{(j-1)} - m^{(j)}})^{-1}\tilde{O}^{(j-1)} + e^{S^{(j)} - m^{(j)}}V^{(j)}
$$

> 💡 **类比:** 就像做加权平均时，先累加带权数值，最后再除以总权重，而不是每加入一个数就重新计算整个平均值，从而减少重复的除法运算。

📍 出处: [Section 3.1.1 (p.5)](https://arxiv.org/pdf/2307.08691.pdf#page=5)

![Figure 1: Diagram of how FlashAttention forward pass is performed, when the key K is partitioned into two blocks and the value V is also partitioned into two blocks. By computing attention with respect to each block and rescaling the output, we get the right answer at the end, while avoiding expensive memory reads/writes of the intermediate matrices S and P. We simplify the diagram, omitting the s](C:\Users\25343\.papermind\cache\2307.08691\figures\p4_figure1_x163.png)
*Figure 1: Diagram of how FlashAttention forward pass is performed, when the key K is partitioned into two blocks and the value V is also partitioned into two blocks. By computing attention with respect to each block and rescaling the output, we get the right answer at the end, while avoiding expensive memory reads/writes of the intermediate matrices S and P. We simplify the diagram, omitting the s (论文原图)*

### 2. 序列长度维度的并行化  `🔴 high`

除了batch和head维度，FlashAttention‑2将外层循环（前向的行块或后向的列块）分配给不同线程块并行执行。前向各线程块独立计算一个输出行块，无需通信；后向各线程块处理一个列块，通过原子加（atomic adds）累积对$\mathbf{dQ}$的更新。这显著提升了GPU占用率，尤其在batch和head数量少的长序列场景，使更多流多处理器被利用。

> 💡 **类比:** 如同将一个大任务切成多个小份，原本一个人要逐个处理多个箱子，现在一群人每人只负责一个箱子，同时开工，整体速度倍增。

📍 出处: [Section 3.2 (p.7)](https://arxiv.org/pdf/2307.08691.pdf#page=7)

![Figure 2: In the forward pass (left), we parallelize the workers (thread blocks) where each worker takes care of a block of rows of the attention matrix. In the backward pass (right), each worker takes care of a block of columns of the attention matrix.](C:\Users\25343\.papermind\cache\2307.08691\figures\p8_figure2_x250.png)
*Figure 2: In the forward pass (left), we parallelize the workers (thread blocks) where each worker takes care of a block of rows of the attention matrix. In the backward pass (right), each worker takes care of a block of columns of the attention matrix. (论文原图)*

### 3. Warp间工作分区：从split‑K到split‑Q  `🔴 high`

在一个线程块内部，FlashAttention将$\mathbf{K}$和$\mathbf{V}$分配给不同warp，导致warp间需要共享内存通信来累加部分结果（split‑K）。FlashAttention‑2改为分割$\mathbf{Q}$，每个warp持有$\mathbf{Q}$的一部分，$\mathbf{K}$和$\mathbf{V}$对全体warp可见。每个warp独立计算其$\mathbf{Q}$块对应的输出，无需warp间通信，消除了额外的共享内存读写。

> 💡 **类比:** 好比几个人合作计算一个大矩阵乘法，以前每人负责一部分列，结果需要汇总；现在每人负责一部分行，各自的结果直接拼接在一起，无需相互等待或传递中间数据。

📍 出处: [Section 3.3 (p.9)](https://arxiv.org/pdf/2307.08691.pdf#page=9)

![Figure 3: Work partitioning between different warps in the forward pass](C:\Users\25343\.papermind\cache\2307.08691\figures\p9_figure3_x264.png)
*Figure 3: Work partitioning between different warps in the forward pass (论文原图)*

### 4. 后向传播中仅使用Logsumexp  `🟡 mid`

后向传播过去需要前向保存的行最大值$m$和指数和$\ell$来计算softmax梯度。FlashAttention‑2改为只保存logsumexp $L = m + \log(\ell)$，后向通过$P^{(j)}_i = \exp(S^{(j)}_i - L_i)$恢复概率矩阵。这减少了对HBM的存储量和读写量，因为每行块只需一个标量，且避免了同时加载$m$和$\ell$。

$$
P^{(j)}_i = \exp(S^{(j)}_i - L_i)
$$

> 💡 **类比:** 只需记住一个数（对数总权重）就能重建整个概率分布，而不必分别记住分子和分母两个数。

📍 出处: [Section 3.1.2 (p.7)](https://arxiv.org/pdf/2307.08691.pdf#page=7)

![教学示意图：后向传播中仅使用Logsumexp](figures/flashattention2-fig1.svg)
*教学示意图：后向传播中仅使用Logsumexp（教学示意图）*

> **读图**：FlashAttention-2后向仅用logsumexp，存储减半
>
> - 旧方法：前向保存m和ℓ，后向需同时加载
> - 新方法：前向仅存L=m+log(ℓ)，后向只需L
> - 核心公式：P=exp(S-L)，由S和L恢复概率
> - 存储开销：每行块从2个float32减为1个
>
> **关键**：每行块仅存一个标量L，HBM读写量减半

### 5. 因果掩码的块级优化  `🟡 mid`

当计算因果注意力时，如果一个块的全部列索引大于行索引，则整个块输出为零，可跳过；若行索引严格小于列索引，则无需应用掩码。这使计算量减少约1.7‑1.8倍，并且每个行块最多只需要对对角线所在的一个块应用掩码，大幅减少条件判断和零填充开销。

> 💡 **类比:** 如同只填写一个下三角表格，右上角大片区域可以直接跳过，仅在边界时仔细处理那条对角线。

📍 出处: [Section 3.1.1 (Causal masking) (p.5)](https://arxiv.org/pdf/2307.08691.pdf#page=5)

### 6. 分块大小的手动调谐  `🟢 low`

块大小$B_r$和$B_c$的选择直接影响性能：增大块可减少共享内存的加载/存储次数，但会占用更多寄存器和共享内存，过大会导致寄存器溢出或超出硬件限制。针对不同的头维度$d$，需手工在少量候选项中（如{64,128}）测试选择，以平衡资源压力和内存IO。

> 💡 **类比:** 好比一次搬运货物的量，太少则来回次数多浪费时间，太多则超出负重能力，需要找到一个效率最高的手推车大小。

📍 出处: [Section 3.3 (Tuning block sizes) (p.9)](https://arxiv.org/pdf/2307.08691.pdf#page=9)

![教学示意图：分块大小的手动调谐](figures/flashattention2-fig2.svg)
*教学示意图：分块大小的手动调谐（教学示意图）*

> **读图**：FlashAttention-2中分块大小Br和Bc的手动调谐方法
>
> - Br和Bc候选集{64,128}，需根据头维度d选择
> - 块大小影响共享内存加载次数和寄存器压力
> - 小块(64)IO频繁但寄存器压力低，大块(128)反之
> - 过大(>128)导致寄存器溢出，不可用
>
> **关键**：在{64,128}中测试选择最小运行时的块大小

<a id="connections"></a>

## 🔗 知识关联

| 概念 | 相关论文 | 关系 |
| --- | --- | --- |
| 标准注意力机制 | [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) | 解决标准注意力实现的内存和速度瓶颈，提出更高效的精确注意力算法。 |
| FlashAttention | [Dao et al. 2022](https://arxiv.org/abs/2205.14135) | 直接改进，通过更好的并行性和工作划分实现约2倍加速，提高GPU利用率。 |
| 在线softmax | [Rabe and Staats 2021](https://arxiv.org/abs/2112.05682) | 采用其分块softmax计算技巧，并进一步调整以减少非矩阵乘法FLOPs（只存储logsumexp）。 |
| Triton语言与编译器 | Tillet et al. 2019 | 借鉴其实现中交换循环顺序和沿序列长度维度并行化的思想，用于提高GPU occupancy。 |
| Megatron-LM | [Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053) | 采用其FLOPs计算公式来评估端到端训练的模型利用率。 |

<a id="reproduction"></a>

## 🛠️ 复现指南

- **官方代码（已核实 · 论文原文链接 ✓官方 · ★24021）:** [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **安装 / 运行（取自仓库）:**
  - `pip install -e .`
  - `python setup.py install`
- **推荐硬件:** A100 / H100
- **关键超参数:** `block_size=[64, 128]`, `causal=True`

### 环境配置步骤

**1. 检查CUDA版本**

确认已安装CUDA工具包，FlashAttention-2需要CUDA环境。

```bash
nvcc --version
```

**2. 安装PyTorch**

安装支持CUDA的PyTorch，版本根据CUDA选择，示例为CUDA 11.8。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. 克隆并安装FlashAttention-2**

从官方仓库克隆并编译安装。

```bash
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && pip install -e .
```

**4. 验证安装**

运行简单测试确保flash-attn正确导入。

```bash
python -c "from flash_attn import flash_attn_func; print('OK')"
```

### 性能基准

| 设置 | Baseline | 结果 | 加速 | 显存 |
| --- | --- | --- | --- | --- |
| seq_len=16384, head_dim=64, causal=False, forward+backward | 46 TFLOPs/s (FlashAttention) | 176 TFLOPs/s | 3.8x | - |
| GPT3-1.3B, seq_len=8192, training on 8xA100 | 170 TFLOPs/s/GPU (FlashAttention) | 220 TFLOPs/s/GPU | 1.3x | - |


---
*Generated by [PaperMind](https://github.com/Wenhao-Hua/papermind).*