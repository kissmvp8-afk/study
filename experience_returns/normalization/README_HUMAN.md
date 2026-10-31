# D04 · 观测归一化与统计管理｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 已完成讲解与润色 |
| 技术核对状态 | 已核对实现、8 项测试、D01 字段与 D02/D03 回放衔接，确认公式、计数与状态复制；未开展全面代码审核 |
| 语言润色状态 | 2026-10-31，Gemini 完成自然语言润色，面向具备 Python 基础、正在学习强化学习与机器人控制的读者，阐明了在线增量统计合并、重采样防污染隔离与双套观测标准化机制；严格保留全部技术信息、公式、张量形状、算法边界与验证限制 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现/测试来自 `e77bc66d7878831fb1b5eb19d616785c81f2d251`，读取时无后续源码变化 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 本目录 AI README、[normalization.py](normalization.py)、[test_normalization.py](test_normalization.py) 全文；D01～D03 实现与测试、D05 衔接测试；固定 PPO `normalizer.py` 全文及 `actor_critic.py` 的归一化路径。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 8 项记录见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。本次未重跑；人工数据，未接网络、仿真或训练 |

## 归一化为什么要区分“更新”和“使用”

在双足轮腿机器人的控制任务中，传感器返回的观测数据呈现出跨度极大的数值异构性：
- 机身姿态四元数与关节角度在 $[-3, 3]\text{ rad}$ 范围内变动；
- 轮子转速在 $[-30, 30]\text{ rad/s}$ 范围内高频旋转；
- 基座线加速度与角速度的瞬时峰值甚至能突破上百数值单位。

如果直接将这些物理量原始输入给深度神经网络，大尺度的通道将在梯度反向传播中占据压倒性优势，导致网络过度拟合于几个高幅值特征，难以敏锐感知姿态的微弱变化。
标准化的通用做法是**在线经验归一化（Empirical Normalization）**，通过实时追踪各通道的均值 $\mu$ 与总体方差 $\sigma^2$，将输入平滑投影到均值为 0、方差接近 1 的无量纲空间：
$$y = \frac{x - \mu}{\sqrt{\sigma^2} + \epsilon}, \quad \epsilon = 0.01$$

但在像 SAC 这样的 Off-Policy（离线策略）算法中，经验池中的历史样本会被成千上万次反复重抽样（Re-sampling）。
这引发了极其严重的架构问题：**如果每次从经验池取出训练批次时都去更新均值和方差，那些被频繁抽样的旧样本就会被重复统计成千上万次！** 其结果是整个系统的特征基准被抽样概率严重绑架，导致特征分布随回放频率发生不可逆的恶性漂移。

**因此，D04 确立了严格的职责解耦：**
1. **更新（`update_from_fresh_data`）**：仅在外部环境产生**全新物理采集数据**并准备写入经验池的单向流水线上执行，且每批新数据只统计一次；
2. **使用（`normalize`）**：回放训练、策略推理以及 Q 价值评价路径上，**仅以只读方式使用当前已固化的统计参数执行数学变换**，绝对不推进计数器或修改分布参数。

## 两套独立统计各管哪些字段

系统构建了两个相互隔离的归一化器实例：

| 归一化器对象 | 空间维度与统计边界 | 服务对象与设计原则 |
|---|---|---|
| `ObservationNormalizers.policy` | 特征维数 $D = 33$；逐帧统计与标准化 | 专用于策略历史 `policy_history`。由于同一通道在不同时间步具有相同的物理含义，同一套均值与方差参数统一作用于 10 帧历史的每一帧（无论张量形状为 `[B, 10, 33]`、`[B, 5, 33]` 还是单帧 `[B, 33]`） |
| `ObservationNormalizers.q` | 特征维数 $D = C$（平地预期 54，复杂地形预期 241） | 专用于 Q 价值网络观测。**正在学习的 Online Q 与缓慢更新的 Target Q 必须严格共享这同一套统计参数**，确保两网络在完全一致的特征尺度坐标系下对状态进行估值 |

> [!NOTE]
> **关于下游辅助学习的原始特征读取**：
> D01 和 D02 始终持久化保存未被修改的原始物理观测。D04 的归一化操作仅在数据流向下游网络时生成全新张量。
> 后续 N02 模块在提取机器人的真实线速度标签用于自监督回归监督时，**必须直接读取未经过 D04 归一化的原始 Q 观测字段**，并保留 E02 规范中约定的 2.0 倍线速度放大系数。

## `normalization.py`：逐类、逐函数职责与调用边界

| 符号名称 | 签名与参数定义 | 核心职责与状态变迁 |
|---|---|---|
| `NormalizationError` | 继承 `ValueError` | 归一化专用异常类。当维度不匹配、输入包含非有限数、恢复状态损坏时抛出 |
| `NORMALIZER_EPS` | `1e-2` (0.01) | 默认数值稳定常数，加在标准差开方结果之外（防止方差极小时除零） |
| `STAT_DTYPE` | `torch.float64` | 内部统计量（均值、方差、样本数）的底层存储类型，全程使用双精度浮点数以保证长期累积数值稳定 |
| `RunningNormalizer.__init__` | `(dim, eps=0.01, device='cpu')` | 检查参数为正；初始化 $\mu = \mathbf{0}, \sigma^2 = \mathbf{1}, N = 0$。未更新前的默认初始映射近似为 $x / 1.01$ |
| `RunningNormalizer.update_from_fresh_data` | `(x: torch.Tensor)` | 接收二维张量 `[N, D]`，校验数值与形状；在 `float64` 下执行增量总体均值与方差合并，累加样本计数 $N$；**修改内部状态** |
| `RunningNormalizer.normalize` | `(x: torch.Tensor) -> torch.Tensor` | 校验最后一维为 $D$，其余维度保持；读取当前统计输出新张量，并还原为输入数据类型；**绝对不改动内部状态** |
| `RunningNormalizer.export_statistics` | `() -> dict` | 导出包含 `mean`、`var`、`count`、`eps`、`dim` 的深拷贝状态字典，用于随训练检查点安全存档 |
| `RunningNormalizer.load_statistics` | `(state: dict)` | 从存档字典中安全加载并恢复所有统计指标，自动完成设备迁移与参数核验 |
| `select_fresh_samples` | `(batch: TransitionBatch) -> (policy, q)` | 纯函数：从新采集批次中精准提取出无重复、无填充的真实物理帧，返回待更新数据 |
| `ObservationNormalizers` | 顶层管理类 | 组合封装了 `policy` 与 `q` 两套独立的正规化器，提供统一的更新、归一化与序列化接口 |

## 统计指标如何按批增量合并（并行 Welford 算法）

在传统流式统计中，如果简单将两组样本的方差求平均，就会彻底丢失**“两组样本本身的均值差异所产生的离散度”**。
D04 采用了严格的**总体方差合并公式（Parallel Welford Algorithm）**：

设已有统计量包含 $n_a$ 个样本，均值为 $\mu_a$，方差为 $\sigma^2_a$；当前新到达的批次包含 $n_b$ 个样本，均值为 $\mu_b$，方差为 $\sigma^2_b$。
两组样本的均值差为 $\Delta = \mu_b - \mu_a$，合并后的总样本数为 $n = n_a + n_b$。

核心合并公式由两部分构成：
```python
delta = mean_b - self.mean
self.mean = self.mean + delta * (n_b / n)
self.var = (self.var * self.count + var_b * n_b + delta.square() * (self.count * n_b / n)) / n
self.count = n
```

1. **新均值**：以两组样本数量为权重的加权平均：$\mu_{\text{new}} = \mu_a + \Delta \times \frac{n_b}{n}$；
2. **新方差**：
   $$\sigma^2_{\text{new}} = \frac{n_a \sigma^2_a + n_b \sigma^2_b + \Delta^2 \times \frac{n_a n_b}{n}}{n}$$
   - 分子前两项代表组内的离均差平方和；
   - 分子第三项 $\Delta^2 \times \frac{n_a n_b}{n}$ 是至关重要的**组间方差补偿项**；
   - 分母使用总体大小 $n$（而非样本方差的 $n-1$），确保多批合并的统计量与全量数据一次性计算完全一致。

在首批数据到达时（$n_a = 0$），代码直接将初始化的 $\sigma^2 = 1$ 覆盖为实际批次方差，不将初始占位值当作虚拟样本引入偏差。

### 归一化计算细节

```python
return ((x.to(STAT_DTYPE) - mean) / (std + self.eps)).to(x.dtype)
```

- 计算在 `float64` 精度下进行，随后转换回输入的原生数据类型（如 `float32`）；
- 数值稳定常数加在标准差的外面：$\sqrt{\sigma^2} + 0.01$，而不是根号内部 $\sqrt{\sigma^2 + 0.01}$；
- 当某一通道为完全无变化的常数输入时（$\sigma^2 = 0$），输入减去均值结果为 0，分母为 $0.01$，输出精准为稳定的 0.0，杜绝了除零崩溃。

## 新采集数据如何做到“既不遗漏，也不重复”

如果简单粗暴地将每个批次的 10 帧历史直接展开喂给归一化器，由于相邻两步之间有 9 帧重叠，中间状态将被重复统计 10 次！
`select_fresh_samples` 依据强化学习的状态转移特性设计了一套精妙的**物理帧单次计数法则**：

```python
first = batch.history_length == 1
policy = torch.cat([batch.policy_history[first, -1], batch.next_policy_history[:, -1]], dim=0)
q = torch.cat([batch.q_observation[first], batch.next_q_observation], dim=0)
```

- 在一个连续运行的轮次中，动作执行后的下一状态 $s_{t+1}$，必然就是下一个时间步的当前状态 $s_t$。
- 因此，对于每一步动作转移，我们**只需要统计其动作执行后的最终到达帧（即 `next_*` 的最新帧）**！
- 唯一会被遗漏的，只有**整个轮次最开始的第一步起始观测（$s_0$）**。代码通过 `history_length == 1` 精准捕捉新轮次的第一条记录，并将其起始帧 $s_0$ 补齐。

### 算例印证：连续 12 步与重置场景

在单元测试中，构造了一个持续 12 次动作的人工环境（状态从 0 持续演进到 12）：
- 步号 0（起始步）：有效长度为 1，计入初始状态 0 和下一状态 1（新增 2 个样本）；
- 步号 1 到 11：仅计入各自的下一状态 $2 \dots 12$（新增 11 个样本）；
- 最终总计数：精确为 $2 + 11 = 13$ 个样本（对应物理状态 $0 \dots 12$），样本均值为 6.0，总体方差为 14.0。没有多算一个重复历史帧，也没有把开局的填充数据计入！

在轮次重置测试中（第 0 轮 3 步后摔倒，第 1 轮从状态 100 开始跑 2 步）：
统计集合精确包含 $\{0, 1, 2, 3\}$（含重置前末状态 3）以及 $\{100, 101, 102\}$（含新起点 100），总计 7 个样本。**重置前末状态与新起点各自被严格统计了一次，既没有丢失摔倒数据，也没有用新起点顶替末状态。**

## 与原 PPO、参考作者 SAC 的区别

1. **与原 PPO 的差异**：原版 PPO 的 `EmpiricalNormalization` 在前向传播 `forward()` 处于训练模式时，会自动调用 `update()`。这对于 On-Policy 单次计算是可行的，但放在 Off-Policy 经验回放中会造成致命的重复统计漂移。D04 将其彻底拆分为两个显式函数。
2. **与开源 SAC 参考实现（`rsl_rl_sac`）的差异**：参考实现在环境步中更新归一化器时，在环境发生真终止时使用了重置后的新观测进行更新，导致末状态未被计入统计；D04 通过 D01 契约显式保证了真实末状态与重置后新起点的双双入库。
3. **检查点恢复契约**：模型保存时，必须将两套归一化器的 `mean`、`var`、`count`、`eps`、`dim` 全部序列化保存。若只保存均值和方差而丢弃 `count`，后续恢复训练时的批合并权重就会彻底错乱。

## `test_normalization.py`：8 项单元测试验证

| 测试类与具体方法 | 覆盖的核心机制与边界断言 |
|---|---|
| `RunningNormalizerTest.test_known_statistics_and_batch_merge` | 4 个已知样本手算比对：验证单次全量更新与拆分多批次合并后的均值和方差完全相等 |
| `RunningNormalizerTest.test_normalize_does_not_change_statistics_or_input` | 只读契约校验：连续调用 5 次 `normalize`，断言内部统计指标和输入张量分毫不变 |
| `RunningNormalizerTest.test_export_and_restore` | 状态持久化核验：导出统计并还原，验证新旧对象对相同输入输出相同，且修改导出字典不产生副作用 |
| `RunningNormalizerTest.test_constant_tiny_variance_and_extreme_inputs` | 极端数值鲁棒性：验证常数通道、极小方差通道在 $\epsilon=0.01$ 下稳定输出，验证 $10^{30}$ 极大输入不发生溢出 |
| `RunningNormalizerTest.test_rejects_bad_updates` | 防御性测试：拦截包含 NaN 的更新数据与维度不符张量，确认内部计数仍为 0 |
| `FreshDataRuleTest.test_each_real_frame_counted_once` | 核心单次计数测试：12 次动作精确累加 13 个样本，严格排除 10 帧历史中的重叠旧帧与填充 |
| `FreshDataRuleTest.test_terminal_and_reset_states_each_counted_once` | 边界状态双统计测试：验证摔倒末状态与重置新起点各被准确计入一次 |
| `ReplayIntegrationTest.test_replayed_observations` | 回放全链路集成：经由 D02 写入并由 D03 读取后，多次执行归一化，确认样本计数严格锁定不变 |

本地复现测试命令：
```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/normalization/test_normalization.py -v
```

> [!WARNING]
> **验证边界说明**：
> 目前 8 项测试在 CPU 合成数据下通过。尚未在包含 Isaac Lab 物理引擎、GPU 显存环境或搭配真实神经网络内部归一化层（如 BatchNorm）的实际训练中评估性能。

## 阅读导航与后续模块

- 了解原始特征定义与真实下一观测结构：请参考 [D01 经历记录契约](../transition_schema/README_HUMAN.md)；
- 了解历史在时间轴上的切片与 Barlow Twins 辅助学习：请继续阅读 [D05 历史窗口与有效性](../history_windows/README_HUMAN.md)；
- 检查点持久化与生产模型导出规范：请参考 [T03 训练检查点](../../training_validation/checkpoints/README.md) 与 [T05 导出推理](../../training_validation/export_inference/README.md)。
