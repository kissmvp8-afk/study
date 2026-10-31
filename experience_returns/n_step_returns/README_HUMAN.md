# D03 · 连续多步回报｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 已完成讲解与润色 |
| 技术核对状态 | 已核对实际函数、7 项测试、D02 有效片段规则和 D01 终止开关；未开展全面代码审核 |
| 语言润色状态 | 2026-10-31，Gemini 完成自然语言润色，面向具备 Python 基础、正在学习强化学习与机器人控制的读者，阐明了 n 步贝尔曼目标累计、动态截断步数 k 与真终止优先自举机制；严格保留全部技术信息、公式、张量形状、容差分析与验证限制 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现/测试来自 `c3b2fe09b752cc8f0ee713d674af5b057c675558`，读取时无后续源码变化 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 本目录 AI README、[returns.py](returns.py)、[test_returns.py](test_returns.py) 全文；D01/D02 实现与测试，D04/D05 衔接测试；固定 PPO `rollout_storage.py` 的 `compute_returns`。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 7 项原结果见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。人工数据，无目标 Q、仿真或训练；本次未重跑算法测试 |

## n 步回报解决什么问题

在标准的时间差分学习（TD-Learning）中，单步 Q 学习（1-step TD）只看眼前一步的即时奖励，随后立即使用神经网络的价值估计进行自举（Bootstrap）：
$$y_t = r_t + \gamma Q(s_{t+1}, a_{t+1})$$
这种做法的局限在于**奖励信号传播极其缓慢**。在机器人复杂的运动控制中（例如机器人在连续跑出 10 步后才因为一次成功越障获得显著奖励），单步 TD 需要极多次训练才能把这一关键奖励逐步倒推传回初始动作。

为了加速价值传播、减少对初始随机 Q 网络的过度依赖，强化学习普遍引入了 **n 步回报（n-step Returns）**机制：
我们将连续发生的多个时间步的即时奖励进行带折扣的累计，直到第 $k$ 步（$1 \le k \le n$）再接入价值自举：
$$R = \sum_{j=0}^{k-1} \gamma^j r_j = r_0 + \gamma r_1 + \gamma^2 r_2 + \dots + \gamma^{k-1} r_{k-1}$$
对应在下游 N05 价值更新中，完整的贝尔曼目标为：
$$y = R + \gamma^k \times \text{mask} \times Q(\text{尾部到达状态}, a^*)$$

这里有两点极为重要的时序概念：
1. **时间流动方向的区别**：在 D05 模块中，历史观测窗口是**向后看过去**（覆盖时刻 $t-4 \dots t$）；而在本模块 D03 中，n 步回报是**向前看未来**（从当前动作开始累计未来的连续收益）。两者的物理时间轴方向相反，切勿混为一谈。
2. **即时奖励的时序归属**：公式中的 $r_0$ 是起点动作执行**之后**收到的即时环境评分，而不是动作做出前看到的观测。

D03 专注于从物理连续轨迹中精确计算出累计折扣奖励 $R$、动态步数 $k$、有效折扣率 $\gamma^k$ 以及自举所需的尾部状态数据；它自身不调用神经网络，完整的 Q 目标合成留待 N05 统一完成。

## 输入与输出：尾部记录不等于尾部状态

D03 接收由 D02 经验池输出的连续时序矩阵 `SequenceBatch`（包含 $M$ 个起点样本，每个样本向前读取长度为 $n$ 的连续序列）。

处理完成后，D03 输出封装好的数据对象 `NStepBatch`：

| `NStepBatch` 字段 | 对应形状与数据类型 | 核心用途与时序含义 |
|---|---|---|
| `first` | 标准 D01 记录批次，`[M, ...]` | 起始时刻的转移快照。Q 网络评估当前选定动作时读取 `first.q_observation` 与 `first.action`；策略网络编码器使用其 `first.policy_history` |
| `tail` | 标准 D01 记录批次，`[M, ...]` | 整个片段中的**第 $k-1$ 步转移记录**（即最后一次真实发生的动作转移） |
| `reward_sum` | `[M, 1]`，`float32` | 累计折算后的即时奖励标量 $R$ |
| `steps` | `[M]`，`int64` | 每个样本在遭遇物理终止前实际有效推进的动作步数 $k$（$1 \le k \le n$） |
| `discount` | `[M, 1]`，`float32` | 实际应用的衰减折扣因子 $\gamma^k$（当提前结束时 $k < n$，必须用 $\gamma^k$ 而不能套用 $\gamma^n$） |
| `bootstrap_mask` | `[M, 1]`，`float32` | 尾部自举开关。若尾部遭遇物理真终止则输出 0.0，超时截断或未结束输出 1.0 |

> [!CAUTION]
> **强化学习的核心细节：尾部记录 $\neq$ 尾部到达状态！**
> 假设我们在时刻 $t$ 启动了一个 $k=2$ 步的多步折算：
> - 第 1 条记录（偏移 0）描述动作转移：$s_t \to s_{t+1}$，产生奖励 $r_0$；
> - 第 2 条记录（偏移 1）描述动作转移：$s_{t+1} \to s_{t+2}$，产生奖励 $r_1$。
> 此时，第 2 条记录就是我们的 `tail`（对应索引 $k-1 = 1$）。
> 当我们在公式末尾计算未来预估价值 $Q(\dots)$ 时，自举的状态必须是执行完第二个动作后**最终到达的物理状态**，即 **`tail.next_*`（时刻 $t+2$）**！绝对不能误取 `tail` 自身的起始观测（时刻 $t+1$），否则就会少算一步物理演进。

## `returns.py`：逐函数职责与调用边界

| 函数或数据结构 | 参数与返回值 | 职责与设计约束 |
|---|---|---|
| `NStepReturnError` | 继承 `ValueError` | 专用异常类。当折扣因子 $\gamma \notin (0, 1]$、读取长度不匹配或有效前缀断裂时抛出 |
| `NStepBatch` | `@dataclass(frozen=True)` | 数据容器，严格封装上述六个字段；冻结类属性，但内部张量只读仍需调用方自觉遵守 |
| `accumulate_rewards` | `(rewards, boundary, gamma) -> (R, steps)` | 纯张量矢量化运算：输入 `rewards[M, n, 1]` 与终止边界掩码 `boundary[M, n]`，定位每行的截断点 $k$，输出加权累计回报 $R[M, 1]$ 与各样本步数 $k[M]$ |
| `select_tail` | `(sequence, steps) -> TransitionBatch` | 调用 `sequence.at(steps - 1)`，从各时序矩阵中准确定位并抽取出第 $k-1$ 步记录作为 `tail` |
| `build_n_step_batch` | `(sequence, gamma, n_steps) -> NStepBatch` | 顶层串联入口：核验参数、校验有效前缀完整性、调用奖励加权折算、提取 `first` 与 `tail`，并通过 D01 推导尾部自举开关 |

所有函数均为无副作用的纯计算逻辑，不改动输入的经验池张量，不修改模型参数。

## 如何精准定位截断步数 k 并进行折现加权

在连续序列中，某行样本可能在第 1 步摔倒，另一行可能在第 3 步超时，还有一行可能平稳走满了全部 $n$ 步。
`accumulate_rewards` 采用完全消除 Python 循环的高效 PyTorch 矢量化算法：

```python
has_boundary = boundary.any(dim=1)
first_boundary = torch.argmax(boundary.to(torch.int64), dim=1)
steps = torch.where(has_boundary, first_boundary + 1, torch.full_like(first_boundary, n))
```

1. `has_boundary`：检测某行样本在前 $n$ 步中是否出现了任何终止事件（`terminated | time_out`）；
2. `first_boundary`：通过 `argmax` 在时间轴上快速检索出**第一个**为 True 的索引位置（$0 \dots n-1$）；
   > 注意：当整行全为 False（没有终止）时，`argmax` 默认也会返回索引 0。因此必须通过 `torch.where` 结合 `has_boundary` 进行二重甄别：有终止时取 `first_boundary + 1`，无终止时取满额步数 $n$。
3. 权重掩码计算：对每一个时间偏移 $j \in [0, n)$，当 $j < k$ 时赋予衰减权重 $\gamma^j$，当 $j \ge k$ 时直接置为 0：
   ```python
   reward_sum = (rewards.squeeze(-1) * weights).sum(dim=1, keepdim=True)
   ```
   因为终止步本身是最后一次执行动作的结果，其即时奖励发生于 $j = k-1$，依然被包含在 $\gamma^{k-1}$ 的加权项中，随后发生的任何奖励都被完全屏蔽为 0。

### 经典手算案例验证（无边界正常走满）

在测试用例中，设定 $n=5, \gamma=0.9$，前 5 步即时奖励分别为 $1, 2, 3, 4, 5$：
$$R = 1 + 0.9 \times 2 + 0.9^2 \times 3 + 0.9^3 \times 4 + 0.9^4 \times 5 = 1 + 1.8 + 2.43 + 2.916 + 3.2805 = 11.4265$$
实际步数 $k = 5$，有效折扣率 $\text{discount} = 0.9^5 = 0.59049$，自举开关 $\text{mask} = 1.0$。
尾部记录 `tail` 为步号 4 的记录，自举状态由 `tail.next_*` 提供。当退化为 $n=1$ 时（单步情况），算法优雅输出 $R=r_0, k=1, \text{discount}=\gamma$。

## 真终止与纯截断：都截断累加，但自举开关完全不同

当遇到轮次结束边界时，无论是真终止还是超时截断，**即时奖励的累加都会立刻停止**（严防跨轮次奖励污染）。但它们在贝尔曼自举开关上呈现关键分歧：

测试中构造了极其严格的边界测试：设定 $n=5, \gamma=0.5$，每步即时奖励均为 1.0，而在新一轮的首步故意注入了极端干扰值 $10^6$：

| 首个结束事件发生的时机 | 实际有效步数 $k$ | 累计回报 $R$ 计算式 | 衰减折扣 $\gamma^k$ | 真终止（翻车）mask | 纯时间截断（超时）mask |
|---|---:|---|---:|---:|---:|
| **第 1 次动作后结束** | 1 | $1.0$ | $0.5^1 = 0.5$ | **0.0** | **1.0** |
| **第 2 次动作后结束** | 2 | $1.0 + 0.5 \times 1.0 = 1.5$ | $0.5^2 = 0.25$ | **0.0** | **1.0** |
| **第 5 次动作后结束** | 5 | $1 + 0.5 + \dots + 0.5^4 = 1.9375$ | $0.5^5 = 0.03125$ | **0.0** | **1.0** |

- **共同点**：两类结束都只累计到自身轮次末尾，绝不把下一轮的 $10^6$ 算入 $R$；
- **分水岭**：
  - **真终止（翻车摔倒）**：物理生命周期结束，`mask = 0.0`，完全舍弃末尾的 $Q$ 预估值；
  - **纯时间截断（超时到期）**：机器人运动状态完好，`mask = 1.0`，算法依然会以重置前真实到达的末状态 `tail.next_*` 为依据，接续最新 Q 网络对未来的预测！

## 数据不足为什么不能被视为一种结束

在 `build_n_step_batch` 中，有一段关键的前缀完整性防线：

```python
needed = torch.arange(n_steps, device=steps.device).unsqueeze(0) < steps.unsqueeze(1)
missing = (needed & ~sequence.valid).any(dim=1)
if bool(missing.any()):
    raise NStepReturnError(...)
```

代码根据计算出的步数 $k$，严格校验从索引 0 到 $k-1$ 的所有槽位在 D02 中是否被标记为 `valid`（连续有效）。
如果当前样本既没有遇到摔倒，也没有遇到超时，仅仅是因为经验池刚刚开跑、后继数据暂时不足（例如只写了 3 步），算法**坚决报错拦截，绝不私自将 $k$ 缩小为 3 来迁就数据缺失**！
因为如果随意见缝插针地缩小 $k$，就会人为扭曲训练样本的折扣因子分布，破坏强化学习的数学收敛性。

## 与 PPO、开源 SAC 的设计差异

1. **与原 PPO 的差异**：原版 PPO 在 `compute_returns` 中逆向推导广义优势估计（GAE），主要服务于短期策略梯度的方差折中；而 D03 是直接从可回放的长周期轨迹中，为 Q 价值网络构建前向多步训练目标。
2. **与开源 SAC 参考实现（`rsl_rl_sac`）的差异**：参考实现虽然也在遇到第一个结束时屏蔽奖励，但其 mask 计算公式会将双真冲突错算为 1；而 D03 引入了 D01 的互斥检查与真终止优先原则，并结合 D02 的连续性掩码过滤掉了所有无效跨代内存。

> [!NOTE]
> **关于离策略偏差（Off-Policy Bias）**：
> 强化学习理论指出，当 $n > 1$ 时，由于连续片段中间的过渡动作是过去旧策略生成的，较大的 $n$ 可能会引入轻微的离策略分布偏差。因此，在实践中 $n$ 并非越大越好（本项目默认候选 $n=5$），需要结合具体训练表现进行实验。

## `test_returns.py`：7 项测试与容差调试经验

测试脚本通过 `buffer_with_episode` 人工构造标准轮次与带有 $10^6$ 极端奖励的新轮次，测试覆盖全部关键分支：

| 测试类与具体用例名称 | 独立检验标准与覆盖逻辑 |
|---|---|
| `AccumulateTest.test_known_sequence_without_boundary` | 5 步正常递推：核对精确回报 11.4265、步数 5、折扣 0.59049，确认末状态指向步号 4 的 `next_*` |
| `AccumulateTest.test_n_equals_one_is_single_step` | 单步退化：验证 $n=1$ 时输出 $R=7.0, k=1, \text{discount}=0.99$ |
| `AccumulateTest.test_invalid_gamma` | 防御性测试：验证 $\gamma \le 0$ 时报错拒绝 |
| `BoundaryTest.test_boundaries_at_steps_1_2_n` | 组合测试：核对第 1/2/5 步分别发生真终止与纯截断共 6 种情形的回报、步数与 mask，确认排除下一轮 $10^6$ 干扰 |
| `MissingSuccessorTest.test_start_without_enough_successors` | 边界测试：验证未结束但步数不足时报错拒绝；补全步数后顺利通过 |
| `MissingSuccessorTest.test_length_must_match_n` | 维度校验：声明 $n=3$ 但序列长度传入 2 时报错拦截 |
| `SampledBatchTest.test_sampled_starts_from_mixed_buffer` | 综合实测：在经历多环境循环覆盖的复杂经验池中随机抽取 50 个样本，逐行比对手算 R、步数与 mask |

在开发调试该模块时，工程团队曾遭遇一次极具代表性的数值问题：
在执行包含 $10^6$ 干扰项的测试时，最初设定的断言绝对容差为 `1e-3` 导致测试偶发失败。深入排查发现：在 IEEE 754 标准下，单精度浮点数（`float32`）的尾数有效位仅约 24 位，在数值达到 $10^6$ 附近时，两个相邻可表示浮点数的最小间距（Machine Epsilon）约为 **0.0625**！此时要求 `1e-3` 的绝对精度已经违背了硬件浮点表示规律。随后测试改为与量级匹配的相对容差后，7 项测试完全稳定通过。

本地复现测试命令：
```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/n_step_returns/test_returns.py -v
```

## 阅读导航与后续模块

- 了解时序片段来源与环形队列机制：请参考 [D02 经验池写入与抽样](../replay_buffer/README_HUMAN.md)；
- 了解单步记录字段与自举掩码设计：请参考 [D01 经历记录契约](../transition_schema/README_HUMAN.md)；
- 了解提取出的特征如何进行标准化与窗口切片：请继续阅读 [D04 观测归一化](../normalization/README_HUMAN.md) 与 [D05 历史窗口](../history_windows/README_HUMAN.md)。
