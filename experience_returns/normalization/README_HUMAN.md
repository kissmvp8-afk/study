# D04 · 观测归一化与统计管理｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 已完成讲解与技术复核，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 已复核总体方差、count 类型、两套统计、逐方法职责与 8 项既有测试内容；恢复调用方去重责任和 E02/E05 接入限制；只核对文档与源码，不开展全面代码审核 |
| 语言润色状态 | 2026-10-31，Gemini 已完成整体语言润色（`d29af80`）；后续技术修正已复核（`22862f1`）；按用户决定本轮不再进行局部润色，保留已有润色内容 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现/测试来自 `e77bc66d7878831fb1b5eb19d616785c81f2d251`，读取时无后续源码变化；本次实际读取 `d29af80b06d941a0d4d11c1be63c9ecbbb975640`，实现/测试与 `5b47469` 一致 |
| 待同步代码版本 | 无；本次仅修正文档，未修改实现或测试 |
| 实际阅读材料 | 本目录 AI README、[normalization.py](normalization.py)、[test_normalization.py](test_normalization.py) 全文；D01～D03 实现与测试、D05 衔接测试；固定 PPO `normalizer.py` 全文及 `actor_critic.py` 的归一化路径。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与工程验证证据)；本次另对照 E02、E05、N04、N05、T05 契约及 `5b47469`→`d29af80` 文档差异 |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 8 项记录见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。本次未重跑；人工数据，未接网络、仿真或训练；修正后检查本地链接/锚点、函数覆盖和代码片段，并核对修改范围，不重跑算法测试 |

## 归一化为什么要区分“更新”和“使用”

机器人的观测包含不同物理量，字段顺序、单位和已有缩放由 E02 定义，各通道的数值尺度可能不同。例如关节角度用 rad、角速度用 rad/s；姿态若用四元数表示，其分量没有 rad 单位。D04 不定义这些物理范围，只对传入的各通道分别维护统计并做变换。

如果直接将这些物理量原始输入给深度神经网络，大尺度的通道将在梯度反向传播中占据压倒性优势，导致网络过度拟合于几个高幅值特征，难以敏锐感知姿态的微弱变化。
标准化的通用做法是**在线经验归一化（Empirical Normalization）**，通过实时追踪各通道的均值 $\mu$ 与总体方差 $\sigma^2$，将输入平滑投影到均值为 0、方差接近 1 的无量纲空间：
$$y = \frac{x - \mu}{\sqrt{\sigma^2} + \epsilon}, \quad \epsilon = 0.01$$

SAC 是 Off-Policy（离策略）算法，允许使用旧策略采集的经历，并不等同于只使用固定离线数据。经验池会反复抽样，因此**若每次回放都更新统计，抽得多的旧样本就会被重复计数**，输入尺度会受到回放次数和抽样权重影响。

**因此，D04 确立了严格的职责解耦：**
1. **更新（`update_from_fresh_data`）**：仅在外部环境产生**全新物理采集数据**并准备写入经验池的单向流水线上执行，且每批新数据只统计一次；
2. **使用（`normalize`）**：回放训练、策略推理以及 Q 价值评价路径上，**仅以只读方式使用当前已固化的统计参数执行数学变换**，绝对不推进计数器或修改分布参数。

每批只更新一次是调用方责任：函数没有保存“已统计记录”的集合，不会自动拒绝重复批次或回放数据。再次调用 update 仍会再次计数；normalize 只保证这次变换不修改统计，并不表示训练期间统计永远冻结。

## 两套独立统计各管哪些字段

系统构建了两个相互隔离的归一化器实例：

| 归一化器对象 | 空间维度与统计边界 | 服务对象与设计原则 |
|---|---|---|
| `ObservationNormalizers.policy` | 特征维数 $D = 33$；逐帧统计与标准化 | 专用于策略历史 `policy_history`。由于同一通道在不同时间步具有相同的物理含义，同一套均值与方差参数统一作用于 10 帧历史的每一帧（无论张量形状为 `[B, 10, 33]`、`[B, 5, 33]` 还是单帧 `[B, 33]`） |
| `ObservationNormalizers.q` | 特征维数 $D = C$（平地预期 54，复杂地形预期 241） | 专用于 Q 价值网络观测。**正在学习的 Online Q 与缓慢更新的 Target Q 必须严格共享这同一套统计参数**，确保两网络在完全一致的特征尺度坐标系下对状态进行估值 |

这属于 D04，实际管理类名是复数 `ObservationNormalizers`。Q 侧每个通道各有自己的均值和方差，并非把 C 个数字混算一组标量统计。当前约定将 Q 可见的额外信息一并放入 C 维向量，尚无另建第三套特权信息归一化器的要求；E02 后续仍须确定字段顺序、缩放及实际 C。普通计数测试使用 C=4，回放衔接测试使用 C=6，均为人工维数，不是实际机器人接口。

> [!NOTE]
> **关于下游辅助学习的原始特征读取**：
> D01 和 D02 始终持久化保存未被修改的原始物理观测。D04 的归一化操作仅在数据流向下游网络时生成全新张量。
> 后续 N02 模块在提取机器人的真实线速度标签用于自监督回归监督时，**必须直接读取未经过 D04 归一化的原始 Q 观测字段**，并保留 E02 规范中约定的 2.0 倍线速度放大系数。

## `normalization.py`：逐类、逐函数职责与调用边界

| 符号名称 | 签名与参数定义 | 核心职责与状态变迁 |
|---|---|---|
| `NormalizationError` | 继承 `ValueError` | 归一化专用异常类。当维度不匹配、输入包含非有限数、恢复状态损坏时抛出 |
| `NORMALIZER_EPS` | `1e-2` (0.01) | 默认数值稳定常数，加在标准差开方结果之外（防止方差极小时除零） |
| `STAT_DTYPE` | `torch.float64` | 均值、方差张量及合并计算使用双精度，减小长期累积误差；`count` 单独保存为 Python 整数 |
| `RunningNormalizer.__init__` | `(dim, eps=0.01, device='cpu')` | 检查参数为正；初始化 $\mu = \mathbf{0}, \sigma^2 = \mathbf{1}, N = 0$。未更新前的默认初始映射近似为 $x / 1.01$ |
| `RunningNormalizer.update_from_fresh_data` | `(x: torch.Tensor) -> None` | 接收 `[N,D]`，拒绝形状不符和 NaN/inf；N=0 时不变。转换到统计设备和 float64，合并总体均值、方差与计数；**修改内部状态** |
| `RunningNormalizer.normalize` | `(x: torch.Tensor) -> torch.Tensor` | 校验最后一维为 $D$，其余维度保持；读取当前统计输出新张量，并还原为输入数据类型；**绝对不改动内部状态** |
| `RunningNormalizer.export_statistics` | `() -> dict` | 导出包含 `mean`、`var`、`count`、`eps`、`dim` 的深拷贝状态字典，用于随训练检查点安全存档 |
| `RunningNormalizer.load_statistics` | `(state: dict) -> None` | 检查维数、非负 count 与有限非负 var，将均值/方差复制到统计设备并恢复 count、eps；不是任意不可信状态的完整校验器 |
| `select_fresh_samples` | `(batch: TransitionBatch) -> (policy, q)` | 只读选择策略帧 `[N_p,33]` 与 Q 行 `[N_q,C]`；依赖上游长度与 `next_*` 正确，不检测重复批次或观测来源 |
| `ObservationNormalizers.__init__` | `(q_dim, eps=0.01, device='cpu')` | 构建 `policy`（33 维）与 `q`（C 维）两个 `RunningNormalizer` |
| `ObservationNormalizers.update_from_fresh_data` | `(batch: TransitionBatch) -> None` | 调用 `select_fresh_samples`，再分别更新两套统计；每批新记录只能调用一次 |
| `ObservationNormalizers.normalize_policy` | `(history: Tensor) -> Tensor` | 委托 `policy.normalize`，要求最后一维为 33；返回新张量，不增加 count |
| `ObservationNormalizers.normalize_q` | `(q_observation: Tensor) -> Tensor` | 委托 `q.normalize`，要求最后一维为 C；供在线和目标 Q 共用，不增加 count |
| `ObservationNormalizers.export_statistics` | `() -> dict` | 返回 `{'policy': 完整状态, 'q': 完整状态}`，统计张量均为副本；不直接写文件 |
| `ObservationNormalizers.load_statistics` | `(state: dict) -> None` | 分别调用两套统计的恢复方法；修改统计，不恢复网络参数或经验池 |

## 统计指标如何按批增量合并（Chan 并行方差公式）

在传统流式统计中，如果简单将两组样本的方差求平均，就会彻底丢失**“两组样本本身的均值差异所产生的离散度”**。
D04 使用**总体方差合并公式**。每个通道先分别求新批均值和总体方差：

```python
mean_b = x.mean(dim=0)
var_b = x.var(dim=0, unbiased=False)
```

`dim=0` 表示沿样本维统计，每个通道保留自己的结果；`unbiased=False` 使用分母 N，计算总体方差，不是分母 N-1 的无偏样本方差估计。

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

现有手算测试输入 `[[1,2],[3,4],[5,6],[7,10]]`，得到 mean=`[4,5.5]`、var=`[5,8.75]`、count=4。第一通道的平方偏差和为 $9+1+1+9=20$，所以总体方差为 $20/4=5$；一次更新四行，或先一行再三行，统计一致。

### 归一化计算细节

```python
return ((x.to(STAT_DTYPE) - mean) / (std + self.eps)).to(x.dtype)
```

- 计算在 `float64` 精度下进行，随后转换回输入的原生数据类型（如 `float32`）；
- 数值稳定常数加在标准差的外面：$\sqrt{\sigma^2} + 0.01$，而不是根号内部 $\sqrt{\sigma^2 + 0.01}$；
- 当某通道方差为 0 且输入等于均值时，分子为 0、分母为 0.01，输出为 0；eps 防止分母为零，不是输出裁剪或任意输入下都无溢出的保证。

上面手算例若输入 `[4+sqrt(5),5.5]`，输出为 `[sqrt(5)/(sqrt(5)+0.01),0]`，第一项接近但不严格等于 1。

## 新采集数据如何做到“既不遗漏，也不重复”

如果简单粗暴地将每个批次的 10 帧历史直接展开喂给归一化器，由于相邻两步之间有 9 帧重叠，中间状态将被重复统计 10 次！
`select_fresh_samples` 依据强化学习的状态转移特性设计了一套精妙的**物理帧单次计数法则**：

```python
first = batch.history_length == 1
policy = torch.cat([batch.policy_history[first, -1], batch.next_policy_history[:, -1]], dim=0)
q = torch.cat([batch.q_observation[first], batch.next_q_observation], dim=0)
```

- 在同一轮连续采集时，一条记录的下一状态 $s_{t+1}$，也会成为下一条记录的当前状态（仍是 $s_{t+1}$），不应再次计数。
- 因此，对于每一步动作转移，我们**只需要统计其动作执行后的最终到达帧（即 `next_*` 的最新帧）**！
- 唯一会被遗漏的，只有**整个轮次最开始的第一步起始观测（$s_0$）**。代码通过 `history_length == 1` 精准捕捉新轮次的第一条记录，并将其起始帧 $s_0$ 补齐。

### 算例印证：连续 12 步与重置场景

在单元测试中，构造了一个持续 12 次动作的人工环境（状态从 0 持续演进到 12）：
- 步号 0（起始步）：有效长度为 1，计入初始状态 0 和下一状态 1（新增 2 个样本）；
- 步号 1 到 11：仅计入各自的下一状态 $2 \dots 12$（新增 11 个样本）；
- 最终总计数：精确为 $2 + 11 = 13$ 个样本（对应物理状态 $0 \dots 12$），样本均值为 6.0，总体方差为 14.0。没有多算一个重复历史帧，也没有把开局的填充数据计入！

在轮次重置测试中（第 0 轮 3 步后摔倒，第 1 轮从状态 100 开始跑 2 步）：
统计集合精确包含 $\{0, 1, 2, 3\}$（含重置前末状态 3）以及 $\{100, 101, 102\}$（含新起点 100），总计 7 个样本。**重置前末状态与新起点各自被严格统计了一次，既没有丢失摔倒数据，也没有用新起点顶替末状态。**

这些结果依赖 E05 正确提供末状态与有效长度，以及采集循环只更新一次。函数不会根据填充值 -999 或传感器数值反推来源；编号或捕获时机错误，仍可能选错样本。

## 与原 PPO、参考作者 SAC 的区别

1. **与原 PPO 的差异**：原版 PPO 的 `EmpiricalNormalization.forward()` 在 training 模式下自动 `update()`。D04 保留逐帧统计与默认 eps，但把更新和使用拆开，避免回放时重复计数。
2. **与作者 SAC 的差异**：按本地文件头和 AI 规范的 `e0d243a` 版本记录，作者在环境步中用 `true_next_obs` 更新，其中真终止环境使用重置后观测、不计末状态。这里按 D01/E05 契约将末状态和下一轮首帧各统计一次；正确捕获仍由 E05 负责。本次未独立读取外部作者源码，不把来源记录当成 TITA 验证结果。
3. **检查点恢复契约**：模型保存时，必须将两套归一化器的 `mean`、`var`、`count`、`eps`、`dim` 全部序列化保存。若只保存均值和方差而丢弃 `count`，后续恢复训练时的批合并权重就会彻底错乱。

## `test_normalization.py`：8 项单元测试验证

`history_at(t,offset)` 返回 `[10,33]` 人工历史，真实帧靠右、其余填 `PAD=-999`。`episode_batch(env,episode,steps,offset,terminated_last)` 经 D01 构造整轮转移，Q 观测使用局部数据生成函数 `q(t)`，取 `[t+offset,2(t+offset),5,0]`；它不是 Q 网络。测试入口是 `unittest.main()`。

| 测试类与具体方法 | 覆盖的核心机制与边界断言 |
|---|---|
| `RunningNormalizerTest.test_known_statistics_and_batch_merge` | 4 个已知样本手算比对：验证单次全量更新与拆分多批次合并后的均值和方差完全相等 |
| `RunningNormalizerTest.test_normalize_does_not_change_statistics_or_input` | 只读契约校验：连续调用 5 次 `normalize`，断言内部统计指标和输入张量分毫不变 |
| `RunningNormalizerTest.test_export_and_restore` | 状态持久化核验：导出统计并还原，验证新旧对象对相同输入输出相同，且修改导出字典不产生副作用 |
| `RunningNormalizerTest.test_constant_tiny_variance_and_extreme_inputs` | 检查人工常数/极小方差通道和给定 $10^{30}$ 输入的输出有限，默认 eps=0.01；不代表任意输入都无溢出 |
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
