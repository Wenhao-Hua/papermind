# High-Resolution Image Synthesis with Latent Diffusion Models

**Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  •  **Year:** 2021  •  **arXiv:** [2112.10752](https://arxiv.org/abs/2112.10752)  •  [PDF](https://arxiv.org/pdf/2112.10752.pdf)

- [🎯 贡献与创新点](#contributions)
- [🔬 技术细节解释](#technical)
- [🔗 知识关联](#connections)
- [🛠️ 复现指南](#reproduction)

<a id="contributions"></a>

## 🎯 贡献与创新点

**核心贡献:** 提出在预训练自编码器的潜在空间中训练扩散模型（Latent Diffusion Models），在显著降低训练和推理计算开销的同时，首次达到复杂度降低与细节保留之间的近最优平衡。

**新颖之处:** 与先前需要过度空间压缩的潜在生成模型（如 VQ-VAE、VQGAN）不同，本方法凭借扩散模型的卷积 UNet 对空间数据的归纳偏置，可在温和压缩的潜在空间中工作，避免因过高压缩率导致的细节损失，且无需在重构与生成能力之间进行精细加权。

**解决的问题:** 解决了像素空间扩散模型优化消耗数百 GPU 天、推理评估缓慢且耗费大量资源的问题，使有限计算资源下的高分辨率图像合成训练成为可能。

> **原文出处:**
> - [Frontmatter (p.1)](https://arxiv.org/pdf/2112.10752.pdf#page=1): To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity.
> - [Abstract (p.2)](https://arxiv.org/pdf/2112.10752.pdf#page=2): Importantly, and in contrast to previous work [23,66], we do not need to rely on excessive spatial compression, as we train DMs in the learned latent space, which exhibits better scaling properties with respect to the spatial dimensionality.
> - [Abstract (p.2)](https://arxiv.org/pdf/2112.10752.pdf#page=2): We propose latent diffusion models (LDMs) as an effective generative model and a separate mild compression stage that only eliminates imperceptible details.

<a id="technical"></a>

## 🔬 技术细节解释

### 1. Conditioning via Cross-Attention Layers  `🔴 high`

通过在 U-Net 的中间层插入交叉注意力机制，将任意形式的条件（如文本、边界框）编码后注入去噪过程。具体地，用 U-Net 的中间特征作为查询 $Q$，将条件信息的嵌入作为键 $K$ 和值 $V$，计算 $\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V$，使生成过程每一步都能关注条件信息。该方法使得单个模型可灵活处理多模态条件，无需重新训练。

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

> 💡 **类比:** 就像在写一篇文章时，不断参考一份提纲（条件），每次决定下一个词时都对照提纲的相关部分，以确保生成内容与提纲一致。

📍 出处: [4.3 (p.7)](https://arxiv.org/pdf/2112.10752.pdf#page=7)

### 2. Convolutional Sampling for High-Resolution Synthesis  `🔴 high`

对于超分辨率和修复等密集条件任务，利用 U-Net 的全卷积性质，在推理时将模型以滑动窗口的形式应用于任意尺寸的潜在表示，逐块去噪并拼接，从而合成远超训练分辨率的图像（如 1024² 像素）。该方法无需修改模型架构即可实现从低分辨率到超高分辨率的泛化。

> 💡 **类比:** 类似拼图：模型只学会了恢复单个小块，但在恢复整张大图时，可以一格一格地处理，最后拼接成完整的大图。

📍 出处: [4.3.2 (p.7)](https://arxiv.org/pdf/2112.10752.pdf#page=7)

![Figure 9. A LDM trained on 2562 resolution can generalize to larger resolution (here: 512×1024) for spatially conditioned tasks such as semantic synthesis of landscape images. See Sec. 4.3.2.](figures/latent-diffusion-orig1.jpeg)
*Figure 9. A LDM trained on 2562 resolution can generalize to larger resolution (here: 512×1024) for spatially conditioned tasks such as semantic synthesis of landscape images. See Sec. 4.3.2. (论文原图)*

### 3. Latent Diffusion Models  `🔴 high`

将扩散模型从高维像素空间迁移到预训练自编码器压缩得到的低维潜在空间中进行训练和推理。首先，一个感知优化的自编码器将图像 $x$ 编码为潜在表示 $z = E(x)$，解码器 $D$ 可实现 $D(z) \approx x$；然后，扩散模型在这个潜在空间上学习去噪过程。这种解耦使得扩散模型专注于语义生成，同时大幅降低计算开销，并可通过一次自编码器训练支持多个下游扩散模型。

> 💡 **类比:** 好比先用有损压缩将高清图片变成占用空间更小的 JPEG 文件，然后训练一个智能绘图程序直接从这个压缩文件生成图像，最后再解压得到高清结果，既快又省资源。

📍 出处: [3.2 (p.4)](https://arxiv.org/pdf/2112.10752.pdf#page=4)

![Figure 4. Samples from LDMs trained on CelebAHQ [39], FFHQ [41], LSUN-Churches [102], LSUN-Bedrooms [102] and class- conditional ImageNet [12], each with a resolution of 256 × 256. Best viewed when zoomed in. For more samples cf. the supplement.](figures/latent-diffusion-orig2.jpeg)
*Figure 4. Samples from LDMs trained on CelebAHQ [39], FFHQ [41], LSUN-Churches [102], LSUN-Bedrooms [102] and class- conditional ImageNet [12], each with a resolution of 256 × 256. Best viewed when zoomed in. For more samples cf. the supplement. (论文原图)*

### 4. Perceptual Compression Tradeoff  `🟡 mid`

通过对下采样因子 $f$ 的控制，在自动编码器的重建保真度和扩散模型的学习效率之间取得平衡。较小的 $f$（例如 $f=4$）保留了更丰富的细节，使得扩散模型能够生成更高质量的图像，但潜在空间维度仍远小于像素空间。实验表明，过于激进的压缩（$f=16$）会损害可达到的生成质量上限。

> 💡 **类比:** 类似照片压缩：压缩率太高会丢失人脸细节，压缩率太低文件仍然很大；找到一个恰当的压缩率能让照片看起来几乎无损，同时文件变得很小。

📍 出处: [4.1 (p.5)](https://arxiv.org/pdf/2112.10752.pdf#page=5)

![Figure 2. Illustrating perceptual and semantic compression: Most bits of a digital image correspond to imperceptible details. While DMs allow to suppress this semantically meaningless information by minimizing the responsible loss term, gradients (during train- ing) and the neural network backbone (training and inference) still need to be evaluated on all pixels, leading to superﬂuous compu- tatio](figures/latent-diffusion-orig3.jpeg)
*Figure 2. Illustrating perceptual and semantic compression: Most bits of a digital image correspond to imperceptible details. While DMs allow to suppress this semantically meaningless information by minimizing the responsible loss term, gradients (during train- ing) and the neural network backbone (training and inference) still need to be evaluated on all pixels, leading to superﬂuous compu- tatio (论文原图)*

### 5. Reweighted Variational Objective  `🟡 mid`

扩散模型训练时采用重新加权的变分目标，通过对不同时间步的损失赋予不同权重并欠采样初始去噪步骤，使模型忽略那些难以感知的高频细节，将建模能力集中于语义内容。这相当于让扩散模型扮演一个有损压缩器的角色，在训练过程中自动抑制不可感知的信息。

> 💡 **类比:** 好比老师批改作文时，重点看逻辑和情节，而不过分计较每一个字的笔画轻微抖动，从而让学生更专注于内容表达。

📍 出处: [2. Related Work (p.2)](https://arxiv.org/pdf/2112.10752.pdf#page=2)

![教学示意图：Reweighted Variational Objective](figures/latent-diffusion-fig1.svg)
*教学示意图：Reweighted Variational Objective（教学示意图）*

> **读图**：重新加权变分目标使扩散模型聚焦语义、抑制高频细节。
>
> - w(t)欠采样初始去噪步骤(t→0)，抑制不可感知高频细节。
> - 模型聚焦中高时间步(t>T/2)的语义内容。
> - 扩散模型扮演有损压缩器，自动抑制不可感知信息。
>
> **关键**：权重调度w(t)引导模型忽略噪声，专注语义压缩。

### 6. Two-Stage Training Strategy  `🟢 low`

将生成任务明确分解为两个独立阶段：第一阶段训练一个通用的感知压缩自编码器，学习出一个感知等价但维度更低的表示空间；第二阶段在该固定编码器的潜在空间中训练扩散模型。这种分离不仅降低了单个阶段的训练难度，还使得编码器可跨任务复用，并允许扩散模型专注于捕获语义和概念的分布。

> 💡 **类比:** 类似先建立一个标准化的压缩工具，之后不同的创作任务（如写实绘画、卡通生成）都直接在这个压缩后的草稿上进行，节省了每次从头建立底稿的成本。

📍 出处: [3.1, 3.2 (p.4)](https://arxiv.org/pdf/2112.10752.pdf#page=4)

![教学示意图：Two-Stage Training Strategy](figures/latent-diffusion-fig2.svg)
*教学示意图：Two-Stage Training Strategy（教学示意图）*

> **读图**：LDM两阶段训练：感知压缩+潜在空间扩散
>
> - Stage1: 自编码器将图像压缩为低维潜在表示z
> - Stage2: 在固定潜在空间训练扩散模型去噪
> - 下采样因子f控制压缩率与重建质量权衡
>
> **关键**：分离感知压缩与语义生成，降低训练难度

<a id="connections"></a>

## 🔗 知识关联

| 概念 | 相关论文 | 关系 |
| --- | --- | --- |
| Diffusion Probabilistic Models | [Ho et al. 2020](https://arxiv.org/abs/2006.11239) | 继承其去噪扩散框架，但将生成过程从像素空间移至潜在空间，以降低计算成本并保持质量。 |
| Perceptual Image Compression with Autoencoders | [Esser et al. 2020 (Taming Transformers)](https://arxiv.org/abs/2012.09841) | 继承其以感知损失和对抗损失训练的自编码器作为压缩阶段，但不使用离散潜在空间，改为连续表示以适配扩散模型。 |
| Cross-Attention for Conditional Generation | [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) | 引入Transformer中的交叉注意力机制，将其融入UNet backbone，实现文本、边界框等多种模态的通用条件生成。 |
| Class-Conditional Diffusion Models in Pixel Space | [Dhariwal & Nichol 2021](https://arxiv.org/abs/2105.05233) | 与之对比，本文LDMs在潜在空间训练和采样，显著降低计算开销，同时达到可比或更优的生成质量。 |
| Classifier-Free Guidance | [Ho & Salimans 2021](https://arxiv.org/abs/2207.12598) | 继承其无分类器引导技术，用于条件扩散模型的采样阶段，提升生成样本的保真度与一致性。 |
| Two-Stage Generative Models with Discrete Latent Spaces | [Razavi et al. 2019 (VQ-VAE-2)](https://arxiv.org/abs/1906.00446) | 对比基于自回归先验的两阶段方法，LDMs利用连续潜在空间和扩散模型，在更温和的压缩率下实现高效生成与细节保持。 |

<a id="reproduction"></a>

## 🛠️ 复现指南

- **官方代码（已核实 · 论文原文链接 ✓官方 · ★14053）:** [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- **安装 / 运行（取自仓库）:**
  - `conda env create -f environment.yaml`
  - `python scripts/knn2img.py  --prompt "a happy bear reading a newspaper, oil on canvas"`
  - `python scripts/train_searcher.py`
  - `python scripts/knn2img.py  --prompt "a happy pineapple" --use_neighbors --knn <number_of_neighbors>`
  - `python scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50`
  - `python scripts/txt2img.py --prompt "a sunset behind a mountain range, vector image" --ddim_eta 1.0 --n_samples 1 --n_iter 1 --H 384 --W 1024 --scale 5.0`
  - `python scripts/inpaint.py --indir data/inpainting_examples/ --outdir outputs/inpainting_results`
- **环境要求:** PyTorch >= 1.10, CUDA >= 11.0 (recommended), Python 3.8+
- **推荐硬件:** NVIDIA V100 or A100 GPU (at least 32GB VRAM for training large models)
- **关键超参数:** `f=4`, `cross_attention=True`

### 环境配置步骤

**1. 克隆仓库**

获取官方代码

```bash
git clone https://github.com/CompVis/latent-diffusion.git && cd latent-diffusion
```

**2. 创建 Conda 环境**

使用 Python 3.8 创建虚拟环境

```bash
conda create -n ldm python=3.8 -y && conda activate ldm
```

**3. 安装 PyTorch**

根据 CUDA 版本安装 PyTorch，以下示例为 CUDA 11.3

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

**4. 安装项目依赖**

使用 requirements.txt 安装其余依赖

```bash
pip install -r requirements.txt
```

**5. 下载预训练模型**

根据官方 README 下载 autoencoder 和 diffusion 模型权重，或使用项目提供的下载脚本

### 性能基准

| 设置 | Baseline | 结果 | 加速 | 显存 |
| --- | --- | --- | --- | --- |
| ImageNet 256x256 class-conditional, f=4 | - | PSNR: 27.4, R-FID: 0.58 (reconstruction from latent) | - | - |

### 数据集

- **[ImageNet](http://www.image-net.org)** — 类条件图像生成、无条件图像生成、超分辨率评估
- **[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)** — 超分辨率训练与评估
- **[MS-COCO](https://cocodataset.org/)** — 文本到图像生成、布局到图像生成

### 常见报错与解决

- **报错:** `RuntimeError: CUDA out of memory`
  - 原因: 批次大小或图像分辨率过大，超出 GPU 显存
  - 修复: `减小配置文件中的 batch_size 或图像尺寸，或使用梯度累积`
- **报错:** `ModuleNotFoundError: No module named 'omegaconf'`
  - 原因: 缺少依赖，未正确安装 requirements.txt
  - 修复: `pip install omegaconf`
- **报错:** `RuntimeError: Unable to download model weights`
  - 原因: 网络问题或链接失效
  - 修复: `检查官方仓库的下载链接，或手动下载并放置在正确路径`

### ⚠️ 坑点提示

- 高分辨率合成（>256x256）需要大量 GPU 显存，建议使用 A100 或 V100 32GB 以上
- 预训练自编码器权重需与扩散模型使用的下采样因子 f 匹配，不可混用
- 条件生成时，文本编码器（如 CLIP）可能需要单独下载并放置在指定目录


---
*Generated by [PaperMind](https://github.com/Wenhao-Hua/papermind).*