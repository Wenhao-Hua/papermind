# Proximal Policy Optimization Algorithms

**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov  •  **Year:** 2017  •  **arXiv:** [1707.06347](https://arxiv.org/abs/1707.06347)  •  [PDF](https://arxiv.org/pdf/1707.06347.pdf)

- [🎯 贡献与创新点](#contributions)
- [🔬 技术细节解释](#technical)
- [🔗 知识关联](#connections)
- [🛠️ 复现指南](#reproduction)

<a id="contributions"></a>

## 🎯 贡献与创新点

**核心贡献:** 提出 proximal policy optimization (PPO)，一种使用裁剪概率比的替代目标函数，支持多轮小批量随机梯度更新的策略梯度方法。

**新颖之处:** 引入 clipped surrogate objective (式(7))，对概率比进行裁剪并取最小值，形成对原目标的悲观下界，从而仅用一阶优化即可实现类似 TRPO 的稳定更新。

**解决的问题:** 解决了 TRPO 实现复杂且与 dropout、参数共享不兼容，而普通策略梯度方法数据效率与鲁棒性差的问题，提供了一种实现简单、数据效率高且鲁棒性好的通用算法。

> **原文出处:**
> - [Frontmatter (p.1)](https://arxiv.org/pdf/1707.06347.pdf#page=1): We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a “surrogate” objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates.
> - [Frontmatter (p.1)](https://arxiv.org/pdf/1707.06347.pdf#page=1): The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically).
> - [Abstract (p.1)](https://arxiv.org/pdf/1707.06347.pdf#page=1): We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy.
> - [Abstract (p.1)](https://arxiv.org/pdf/1707.06347.pdf#page=1): trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks).

<a id="technical"></a>

## 🔬 技术细节解释

### 1. Clipped Surrogate Objective $L^{CLIP}$  `🔴 high`

PPO 的核心目标函数，对概率比率 $r_t(	heta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 进行截断，再与未截断的 $r_t(\theta)\hat{A}_t$ 取最小值。当优势 $\hat{A}_t>0$ 时，若 $r_t(\theta)>1+\epsilon$，截断阻止目标继续增大；当 $\hat{A}_t<0$ 时，若 $r_t(\theta)<1-\epsilon$，截断阻止目标进一步减小。最终目标成为未截断目标的一个悲观下界，避免过大的策略更新。

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

> 💡 **类比:** 就像给车速设置上下限：下坡时限速防止超速，上坡时限速防止溜车，只在安全范围内调节油门。

📍 出处: [Section 3 (p.3)](https://arxiv.org/pdf/1707.06347.pdf#page=3)

![教学示意图：Clipped Surrogate Objective $L^{CLIP}$](figures/ppo-fig1.svg)
*教学示意图：Clipped Surrogate Objective $L^{CLIP}$（教学示意图）*

> **读图**：PPO截断代理目标函数定义与机制图解
>
> - 核心目标LCLIP：对概率比率rt截断后取最小值
> - 概率比率rt：新旧策略在动作上的概率比值
> - 截断机制：将rt限制在[1-ε,1+ε]内
> - 优势At正负决定截断方向：阻止过度更新
>
> **关键**：截断使目标成为悲观下界，防止策略更新过大

### 2. Adaptive KL Penalty Coefficient $\beta$  `🟡 mid`

作为截断目标的替代，在目标中直接加入 KL 散度惩罚项 $\beta\,\text{KL}$，并自适应调整 $\beta$：每次策略更新后计算实际 KL 散度 $d$，若 $d < d_{\text{targ}}/1.5$（更新幅度太小）则将 $\beta$ 减半，若 $d > d_{\text{targ}}\times 1.5$（更新幅度太大）则将 $\beta$ 加倍，其余情况保持不变。这样将 KL 散度维持在一个目标值附近。

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t - \beta \, \text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)] \right]
$$

> 💡 **类比:** 像恒温器自动调节制冷强度：室温偏高就加大制冷，偏低就减少制冷，使室温稳定在设定值附近。

📍 出处: [Section 4 (p.4)](https://arxiv.org/pdf/1707.06347.pdf#page=4)

![教学示意图：Adaptive KL Penalty Coefficient $\beta$](figures/ppo-fig2.svg)
*教学示意图：Adaptive KL Penalty Coefficient $\beta$（教学示意图）*

> **读图**：自适应KL惩罚系数β的调整规则与算法流程
>
> - L^{KLPEN}含重要性采样比和KL惩罚项
> - β根据实际KL散度d与目标d_targ比较调整
> - d<d_targ/1.5则β减半，d>1.5d_targ则β加倍
> - 算法每轮迭代：收集轨迹、计算损失、梯度更新、调整β
>
> **关键**：β自适应维持KL散度在目标范围内，避免更新过大或过小

### 3. Truncated Generalized Advantage Estimation (GAE) for Fixed-Length Segments  `🟡 mid`

PPO 使用固定长度 $T$ 的轨迹段，优势估计采用截断版 GAE：$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$，其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。当 $\lambda=1$ 时退化为有限时步的蒙特卡洛优势估计。这种截断避免了对整条轨迹的依赖，与固定段采样匹配。

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

> 💡 **类比:** 像预报未来几天的天气，只参考接下来一周的数据（有限窗口），而不需要全年的气候模型。

📍 出处: [Section 5 (p.5)](https://arxiv.org/pdf/1707.06347.pdf#page=5)

![教学示意图：Truncated Generalized Advantage Estimation (GAE) for Fixed-Length Segments](figures/ppo-fig3.svg)
*教学示意图：Truncated Generalized Advantage Estimation (GAE) for Fixed-Length Segments（教学示意图）*

> **读图**：截断GAE在固定长度段上的优势估计公式与性质。
>
> - 核心公式：截断求和，从δ_t到(γλ)^{T-t+1}δ_{T-1}。
> - δ_t：TD残差，r_t+γV(s_{t+1})-V(s_t)。
> - 截断至T：避免整条轨迹依赖，匹配固定段采样。
> - λ=1时：退化为有限时步蒙特卡洛优势估计。
>
> **关键**：截断GAE在固定段内高效估计优势，适用于PPO。

### 4. Multi-Epoch Minibatch SGD on Sampled Data  `🟡 mid`

与标准策略梯度每批数据只做一次更新不同，PPO 在收集 $NT$ 个时间步的数据后，使用这些数据构造代理损失，并执行 $K$ 个 epoch 的 minibatch SGD 优化（典型 $K=3$ 或 $10$）。这相当于对同一批数据进行多次梯度上升，提高了样本利用效率，但与简单的重复优化不同，代理损失通过截断或 KL 惩罚防止过大的策略偏移。

> 💡 **类比:** 就像反复研读同一份试题，每次找出薄弱点并修正，而不是做完一遍就扔掉，从而提高学习效果。

📍 出处: [Section 5, Algorithm 1 (p.4)](https://arxiv.org/pdf/1707.06347.pdf#page=4)

### 5. Combined Loss with Value Function and Entropy Bonus  `🟢 low`

最终优化的目标函数整合了三部分：截断代理损失 $L^{CLIP}$、价值函数均方误差损失 $L^{VF}$（训练 $V_\theta$ 逼近目标值）以及策略熵奖励 $S[\pi_\theta](s_t)$（鼓励探索）。通过系数 $c_1, c_2$ 平衡各项，共享网络参数时一起优化。

$$
L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

> 💡 **类比:** 像一份综合食谱，平衡了主食（策略提升）、蛋白质（价值预测）和维生素（探索多样性），确保整体营养。

📍 出处: [Section 5 (p.5)](https://arxiv.org/pdf/1707.06347.pdf#page=5)

![教学示意图：Combined Loss with Value Function and Entropy Bonus](figures/ppo-fig4.svg)
*教学示意图：Combined Loss with Value Function and Entropy Bonus（教学示意图）*

> **读图**：PPO最终目标函数整合截断代理损失、价值损失和熵奖励。
>
> - L̃CLIP: 截断代理损失，限制策略更新幅度。
> - L̃VF: 价值函数均方误差，训练Vθ逼近目标值。
> - S[πθ]: 策略熵奖励，鼓励探索防止早熟收敛。
> - c₁, c₂: 平衡价值损失和熵正则化的系数。
>
> **关键**：三项损失加权求和，共享网络参数联合优化。

### 6. Fixed-Length Trajectory Segments and Parallel Actors  `🟢 low`

PPO 的 actor-critic 实现采用 $N$ 个并行 actor，每个运行 $T$ 步并收集数据，然后合并为 $NT$ 个时间步的批量用于更新。这种固定长度段的设计便于使用截断优势估计，并与循环网络兼容，同时并行化提高了采样效率。

> 💡 **类比:** 好比多个人同时在不同副本上玩游戏，每人都玩固定的时间后把经验汇集成一个大训练集，再集中学习。

📍 出处: [Section 5, Algorithm 1 (p.4)](https://arxiv.org/pdf/1707.06347.pdf#page=4)

![教学示意图：Fixed-Length Trajectory Segments and Parallel Actors](figures/ppo-fig5.svg)
*教学示意图：Fixed-Length Trajectory Segments and Parallel Actors（教学示意图）*

> **读图**：PPO用N个并行actor各跑T步收集NT条轨迹更新
>
> - N个并行actor各运行T步收集数据
> - 合并为NT个时间步的批量用于更新
> - 固定长度段结构便于GAE和RNN
> - 更新步骤：运行、计算优势、优化、SGD
>
> **关键**：固定长度段+并行actor提高采样效率

<a id="connections"></a>

## 🔗 知识关联

| 概念 | 相关论文 | 关系 |
| --- | --- | --- |
| Trust Region Policy Optimization (TRPO) | [Schulman et al. 2015b](https://arxiv.org/abs/1502.05477) | 继承 TRPO 的信任域思想，但用简单的剪辑替代硬约束，实现更简单、更通用、样本复杂度更好 |
| Conservative Policy Iteration (CPI) | Kakade & Langford 2002 | 基于 CPI 的代理目标 L^CPI，引入概率比剪辑，形成更稳定的 L^CLIP 目标 |
| Generalized Advantage Estimation (GAE) | [Schulman et al. 2015a](https://arxiv.org/abs/1506.02438) | 使用 GAE 计算优势估计，作为策略梯度估计器的组成部分 |
| Vanilla policy gradient methods | [Mnih et al. 2016](https://arxiv.org/abs/1602.01783) | 对比；标准策略梯度每次采样只执行一次梯度更新，数据效率低且不稳定，PPO 允许多轮小批量更新，克服该缺点 |
| Advantage Actor-Critic (A2C) | [Mnih et al. 2016](https://arxiv.org/abs/1602.01783) | 实验对比；在 Atari 游戏上，PPO 的样本复杂度显著优于 A2C，且简单性相当 |
| Actor-Critic with Experience Replay (ACER) | [Wang et al. 2016](https://arxiv.org/abs/1611.01224) | 实验对比；PPO 在 Atari 上的最终性能与 ACER 相似，但实现更简单 |

<a id="reproduction"></a>

## 🛠️ 复现指南

- **官方代码（已核实 · 论文原文链接 ✓官方 · ★1654）:** [https://github.com/berkeleydeeprlcourse/homework](https://github.com/berkeleydeeprlcourse/homework)
- **关键超参数:** `clip_epsilon=0.2 (optimal for continuous control from Table 1)`, `horizon_T=2048 (MuJoCo), 512 (Roboschool), 128 (Atari)`, `adam_stepsize=3e-4 (MuJoCo), 2.5e-4*alpha with alpha annealing (Atari)`, `num_epochs=10 (MuJoCo), 15 (Roboschool), 3 (Atari)`, `minibatch_size=64 (MuJoCo), 4096 (Roboschool), 32*8=256 (Atari)`, `discount_gamma=0.99`, `GAE_lambda=0.95`, `value_function_coefficient_c1=1 (Atari)`, `entropy_coefficient_c2=0.01 (Atari)`

### 数据集

- **MuJoCo continuous control tasks (HalfCheetah-v1, Hopper-v1, InvertedDoublePendulum-v1, InvertedPendulum-v1, Reacher-v1, Swimmer-v1, Walker2d-v1)** — continuous control benchmark
- **Roboschool humanoid tasks (RoboschoolHumanoid-v0, RoboschoolHumanoidFlagrun-v0, RoboschoolHumanoidFlagrunHarder-v0)** — high-dimensional continuous control
- **Atari 2600 games (49 games via Arcade Learning Environment)** — discrete control from pixels

### ⚠️ 坑点提示

- MuJoCo may require a separate license; use mujoco-py with the open-source license or consider PyBullet as an alternative.
- Roboschool environments are no longer actively maintained and may be difficult to install; PyBullet provides similar humanoid tasks.
- Atari preprocessing (grayscale, frame stacking, action repeats) must exactly match the A2C setup from Mnih et al. 2016 for fair comparison.
- The clipped surrogate objective with epsilon=0.2 generally performed best; adaptive KL penalty was implemented but not primary.


---
*Generated by [PaperMind](https://github.com/Wenhao-Hua/papermind).*