# D02 · 经验池写入与抽样｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 已完成讲解与润色 |
| 技术核对状态 | 已对照实际实现、9 项测试及 D01/D03 调用核对接口、形状、复制与覆盖规则；本次不扩展为全面代码审核 |
| 语言润色状态 | 2026-10-31，Gemini 完成自然语言润色，面向具备 Python 基础、正在学习强化学习与机器人控制的读者，阐明了多环境环形队列管理、覆盖代数追踪与 n 步连续片段检索机制；严格保留全部技术信息、公式、张量形状、槽位运算与验证限制 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现与测试来自 `3838fe4a0d00bf2e8e102edda39a58d5c21e9f0c`，读取时无后续源码变化 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根规范、上层及本目录 AI README；[buffer.py](buffer.py)、[test_buffer.py](test_buffer.py) 全文；D01 `schema.py`/测试、D03 `returns.py`/测试，D04/D05 衔接测试；固定 PPO `rollout_storage.py` 相关片段。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 D01～D05 共 55 项 CPU 检查已独立通过；本模块 9 项的既有结果见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 和 [AI 交接表](README.md#实现证据与文档交接)。本次只核对文档，不重跑算法测试；人工数据，未接入仿真或训练 |

## 要解决的问题

在基于 SAC（Soft Actor-Critic）算法的强化学习训练中，系统依赖经验回放池（Replay Buffer）来打破样本之间的时间自相关性。
然而，在多机器人并行仿真（如 Isaac Lab 中的数千个并行环境）背景下，经验池的设计远比“向单一数组追加数据”复杂得多：
1. **多环境数据交错写入**：几千个机器人在同一个时钟周期产生新数据，不同环境的轮次（Episode）时长各不相同，结束和重置的时机异步发生。
2. **容量满后的环形覆盖**：当经验池达到预设容量上限时，旧数据会被最新采集的数据覆盖。
3. **时序跳跃风险**：如果仅仅根据内存槽位的物理相邻关系来抽取后续动作，极易将“上一代旧数据”与“刚写入的新数据”错误地拼接在一起，甚至把已经结束的旧轮次与刚开始的新轮次缝合成一条虚假轨迹。

**D02 模块的核心职责，是在内存中维护一个按环境隔离的环形经验池。它接收 D01 标准的 `TransitionBatch` 记录，通过自增写入序号（`_write_seq`）严密追踪每个槽位的代数演进，精准判定物理时序的连续性，并为下游 D03 模块提供完全合法、同轮次且连续的 n 步时序片段。**

本模块只负责经验的高效留存、连续性校验与抽样读取；它不计算多步折扣回报（属于 D03）、不执行观测特征归一化（属于 D04），也不缓存会随网络学习改变的历史潜空间表征。

## 从 D01 写入，到 D03 读取

为了方便定量描述数据流，我们定义以下关键维度符号：
- $E$ (`num_envs`)：并行机器人的环境总数（例如 4096）；
- $K$ (`capacity_per_env`)：**每个环境分配的环形队列深度（槽位数）**；整个经验池的总容量为 $E \times K$；
- $C$ (`q_dim`)：Q 价值网络观测维数（平地预期 54，复杂地形预期 241）；
- $M$ (`batch_size`)：下游训练时随机抽取的起始样本数；
- $L$ (`length`)：从每个起点开始向前读取的时序步数（通常等于 n 步回报的探测跨度 $n$）。

### 数据在各阶段的张量形状与转换关系

| 处理环节 | 对应数据对象 | 字段形状与类型特征 | 业务含义与说明 |
|---|---|---|---|
| **D01 采集输入** | 单步转移批次 `TransitionBatch` | 包含 13 个字段，各字段批维均为 $B$：<br>策略历史 `[B, 10, 33]`、Q 观测 `[B, C]`、动作 `[B, 8]`、奖励 `[B, 1]` 等 | $B$ 为当前步参与写入的记录数（$1 \le B \le E$）。同一批内 `env_id` 必须严格唯一 |
| **池内底层存储** | 经验池预分配内部张量 | 每个字段扩展为 `[E, K, *字段形状]`：<br>如历史为 `[E, K, 10, 33]`，编号为 `[E, K]`，`_write_seq` 为 `[E, K]` | 在内存中为每个环境开辟固定的 $K$ 个槽位。**槽位索引（Slot）是物理内存地址，不等于动作步号** |
| **有效起点检索** | `(env_ids, slots)` | 形状均为 `[M]`（`int64`） | 抽样选中的 $M$ 个起始位置，代表能够合法延伸出有效片段的物理槽位 |
| **D02 连续片段读取** | `SequenceBatch.fields` | 13 个字段在第 1 维展开为长度 $L$：<br>如策略历史为 `[M, L, 10, 33]`，奖励为 `[M, L, 1]` | 从指定起点沿着环形槽位向前连读 $L$ 步所构成的轨迹矩阵 |
| | `SequenceBatch.valid` | 形状为 `[M, L]`（`bool`） | 逐项标记从起点到当前步是否在物理时序上完全连续、同轮次且未被覆盖 |
| **按偏移精准定点** | `SequenceBatch.at(offsets)` | 输入 `offsets`（`[M]`），返回标准的 `TransitionBatch` (`[M, ...]`) | 从每个时序片段中按偏移提取单步快照，恢复为标准的 D01 批次 |

整体调用流程呈现单向流水线：

```text
D01 make_transition (校验并生成干净张量)
           ↓
ReplayBuffer.append (写入对应环境的下一槽位，推进写入代数)
           ↓
ReplayBuffer.sample_starts (依据连续性逻辑，均匀抽取 M 个有效起点)
           ↓
ReplayBuffer.read_sequence (沿槽位环读取长度为 n 的连续时序矩阵 SequenceBatch)
           ↓
D03 build_n_step_batch (下游接收 SequenceBatch，折算 n 步折扣回报)
```

`append` 允许传入部分环境的数据（$B < E$），但在同一次调用中绝对不允许出现重复的 `env_id`（因为同一批内未定义同环境两条记录的先后时间）。多步连续数据必须分步多次写入。
在内存安全方面，写入时通过索引赋值复制，读取时通过张量高级索引（Advanced Indexing）复制。外部数据源的后续改写或读出结果的原地修改，都不会破坏池内存储。

## `buffer.py`：逐函数职责与设计约束

| 符号名称 | 函数签名 / 属性定义 | 核心职责与边界约束 |
|---|---|---|
| `ReplayBufferError` | 继承 `ValueError` | 经验池专用的异常类。当容量配置非正、抽样参数越界、多环境 ID 冲突或读取无效槽位时抛出 |
| `WRITE_SEQ_DTYPE` | `torch.int64` | 槽位写入序号的数据类型；数值为 -1 时代表该槽位为空、尚未写入任何数据 |
| `bytes_per_record` | `(q_dim: int) -> int` | 计算单条经历（含 13 个字段和 1 个写入序号）占用的静态内存字节数，不进行内存分配 |
| `estimate_storage_bytes` | `(num_envs, capacity_per_env, q_dim) -> int` | 预先估算经验池全量张量所需的内存字节数（$E \times K \times \text{单条字节}$），辅助配置显存与内存预算 |
| `ReplayBuffer.__init__` | `(num_envs, capacity_per_env, q_dim, device='cpu')` | 构造函数。检查 $E, K > 0$，一次性预先分配全部张量底座、代数矩阵 `_write_seq[E, K]` 与各环境写入计数器 `_writes[E]` |
| `ReplayBuffer.size` | 属性，返回 `int` | 当前经验池中已写入且尚未被丢弃的有效记录总条数（不等于有效 n 步起点的数量） |
| `ReplayBuffer.storage_bytes` | 属性，返回 `int` | 精确统计池内所有张量（含写指针）实际占用的显存/内存物理字节数 |
| `ReplayBuffer.append` | `(batch: TransitionBatch) -> torch.Tensor` | 校验批次数据，核对设备与环境编号唯一性；定位各环境的下一个环形槽位并存入数据，返回所占槽位索引 |
| `ReplayBuffer._continuity` | 内部方法，返回 `(present, linked, boundary)` | 计算 `[E, K]` 的布尔张量：当前槽位是否有数据、是否能合法链接到下一个物理槽位、当前步是否为轮次终止边界 |
| `ReplayBuffer.valid_starts` | `(n_steps: int) -> tuple[Tensor, Tensor]` | 核心检索算法。输入 $1 \le n \le K$，返回所有能够向前构成合法 n 步片段的 `(env_ids, slots)` 起点坐标 |
| `ReplayBuffer.sample_starts` | `(batch_size, n_steps, generator=None)` | 依托 `valid_starts` 检索候选集，进行有放回均匀随机抽样；支持传入 `torch.Generator` 以实现确定性复现 |
| `ReplayBuffer.read_sequence`| `(env_ids, slots, length: int) -> SequenceBatch` | 给定起点和读取长度 $L$，沿环形槽位提取连续矩阵，并输出 `valid` 连续性掩码 |
| `SequenceBatch.length` | 属性，返回 `int` | 当前时序矩阵的跨度长度 $L$（即 `valid.shape[1]`） |
| `SequenceBatch.at` | `(offsets: Sequence | Tensor) -> TransitionBatch` | 传入针对各采样子序列的相对时间步偏移 `[M]`，提取对应位置的单步快照，组装为 D01 标准记录输出 |

## 环形槽位为什么还需要写入序号

在环形缓冲区中，槽位的寻址逻辑非常经典：

```python
seq = self._writes[env]
slot = seq % self.capacity_per_env
```

`seq` 是该环境专属的**累计自增写入序号**，取模运算 `%` 仅仅决定数据存放在数组中的哪个物理格子（Slot）。
然而，仅靠槽位索引是无法区分时序的。假设某个环境的容量 $K=4$，连续写入步号 0 到 9 的数据，这 4 个物理格子的最终状态如下：

| 物理槽位（Slot） | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| **存放的步数（`step_index`）** | 8 | 9 | 6 | 7 |
| **对应的写入序号（`_write_seq`）** | 8 | 9 | 6 | 7 |

此时，真实的时间流动顺序是槽位 **`2 → 3 → 0 → 1`**（对应物理步数 $6 \to 7 \to 8 \to 9$）。
当我们从槽位 1 试图走向槽位 2 时，虽然在环形数组上 `(1 + 1) % 4 == 2` 恰好是下一个格子，但此时槽位 2 里的数据是上一代遗留下来的旧数据（序号 6，落后于当前序号 9）！如果把槽位 1 和槽位 2 连在一起，时序就会从步数 9 瞬间倒退回步数 6。

为了从根本上消除这种跨代拼接错误，`_continuity` 实施了严密的**四重连续性判据**：

```python
present = self._write_seq >= 0
linked = (
    present
    & torch.roll(present, shifts=-1, dims=1)
    & (torch.roll(self._write_seq, shifts=-1, dims=1) == self._write_seq + 1)
    & (torch.roll(episode, shifts=-1, dims=1) == episode)
    & (torch.roll(step, shifts=-1, dims=1) == step + 1)
)
```

1. **前后均已写入**：当前槽位与下一个槽位均包含有效数据；
2. **写入序号严格递增 1**：下一个格子的 `_write_seq` 必须恰好比当前大 1（彻底阻断槽位 1 跳回槽位 2 的跨代倒流）；
3. **轮次完全相同**：下一个格子的 `episode_id` 必须与当前一致（阻断跨轮次拼接）；
4. **步数严格自增 1**：下一个格子的 `step_index` 必须正好自增 1（阻断丢步或重复步）。

通过这套逻辑，合法跨越数组末尾的连续连接（如槽位 $3 \to 0$）被完整保留，而过期的、被覆盖的或新一轮的错误连接被全部切断。

## 有效起点不是“剩下几条就用几条”

下游算法在计算 n 步回报时，需要从起点开始连续向前看 $n$ 步。经验池在筛选合法起点时，必须遵守明确的动力学边界规则：

```python
link_j = torch.roll(linked, shifts=-j, dims=1)
ok &= ~open_ | link_j
open_ &= link_j & ~torch.roll(boundary, shifts=-(j + 1), dims=1)
```

算法通过循环推进两个布尔状态集合：
- `ok`：标记哪些槽位可以作为合格的起点；
- `open_`：标记哪些起点在当前步**尚未遇到物理终止**、因而必须要求后继步骤存在。

这带来了极为关键的工程判定：
- **情况 A（未遇到终止，但数据不足）**：在上文容量 $K=4$ 且没有结束的例子中，如果算法请求 $n=3$ 的片段，只有步数 6 和 7 属于有效起点；步数 8 和 9 后面缺乏足够的数据步，属于“未来数据暂缺”，**绝对不能因为只剩 2 条或 1 条就强行拿去训练**！
- **情况 B（中途遭遇物理终止）**：如果机器人在步数 7 发生了真终止（翻车），此时从步数 6 开始往后走，虽然物理上总共只有 2 步数据（步数 6 和 7），但因为其已经合法到达了任务终点，它依然被判定为**合法的 3 步起点**（在读取的序列中表现为 `valid = [True, True, False]`）。

读取出的序列中，`valid` 为 `False` 的位置可能是尚未写入的空槽、另一轮的数据或旧一代残留，绝对不能直接将整个矩阵丢进损失函数。调用方必须通过 D03 截取有效前缀，或通过 `SequenceBatch.at` 提取定点记录。

## 内存容量计算：高维历史占用了多少空间

许多强化学习项目在扩展到大规模并行仿真时遭遇内存溢出（OOM），原因在于未精细核算时序历史的存储体积。
单条经历包含当前与下一策略历史两份张量，每份张量形状为 `[10, 33]`（`float32`，每数字 4 字节）。仅历史观测这一项，单条记录就需要占用：
$$2 \times 10 \times 33 \times 4 = 2640 \text{ 字节}$$

结合 Q 观测（$2 \times C \times 4 = 8C$）、策略动作（$8 \times 4 = 32$）、奖励（$1 \times 4 = 4$）、两个布尔标记（2 字节）、五个 64 位整数元数据（$5 \times 8 = 40$）以及一个写入序号（8 字节），单条经历的静态内存公式为：
$$\text{bytes\_per\_record}(C) = 2640 + 8C + 32 + 4 + 2 + 40 + 8 = 2726 + 8C \text{ 字节}$$

- 当 $C = 54$（平地简单观测）时：每条记录约 **3158 字节**（约 3.08 KB）；
- 当 $C = 241$（复杂地形高度图）时：每条记录约 **4654 字节**（约 4.54 KB）。

假设系统配置 4096 个并行环境，每个环境存储 1221 步（总计约 500 万条经历）：
当 $C = 241$ 时，仅静态字段数据就将占用约 **23.3 GB 内存**（按十进制计算）。
函数 `estimate_storage_bytes` 与属性 `storage_bytes` 能帮助工程师在经验池构建之初精确预判显存与内存开销，防止训练中途发生崩溃。

## `test_buffer.py`：9 项单元测试的验证范围

测试脚本通过 `code_of(env, episode, step) = env * 10000 + episode * 100 + step` 这一数学编码，将环境号、轮次号和步数编号无损压缩为一个独立的浮点标记，并将其填充进奖励与传感器观测中，从而能以最高精度核验每一步的数据归属：

| 测试类与具体用例名称 | 核心验证逻辑与覆盖边界 |
|---|---|
| `WriteReadTest.test_round_trip_all_fields` | 写入两条记录，经 `read_sequence` 和 `at` 提取后比对全部 13 个 D01 字段，确认完全无损还原 |
| `WriteReadTest.test_copies_on_write_and_read` | 写入后原地篡改外部来源张量，读取后原地篡改读出张量；断言池内核心存储原封不动 |
| `OverwriteTest.test_wrapped_ring_keeps_only_valid_starts` | 在容量 4 的队列中连续压入 10 步数据，严密验证槽位绕回后仅暴露步数 6..9，并核对 n=3 和 n=4 的有效起点 |
| `OverwriteTest.test_sampled_sequences_are_consecutive_after_many_overwrites` | 在经历数千次覆盖后执行 64 次抽样，核验所有抽出的片段均满足同环境、同轮次且步数严格连续 |
| `MultiEnvAndBoundaryTest.test_interleaved_envs_only_join_same_env` | 多个环境以异步步调交错写入，确认时序链接绝不会跨环境串线 |
| `MultiEnvAndBoundaryTest.test_episode_boundaries` | 验证物理真终止和超时截断能使片段合法提前闭合，而新一轮未填满的起点被精准过滤 |
| `MultiEnvAndBoundaryTest.test_reproducible_sampling` | 验证固定随机种子下，`sample_starts` 能够产生 100% 确定性的抽样结果 |
| `CapacityAndErrorTest.test_bytes_follow_actual_fields` | 核对手算内存字节数公式，并验证包含写指针开销后的精确内存统计 |
| `CapacityAndErrorTest.test_rejects_bad_writes_and_reads` | 防御性测试：拦截重复环境号写入、拦截越界环境号、拦截空起点抽样、拦截无效偏移访问 |

复现测试时，可在仓库根目录执行：
```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/replay_buffer/test_buffer.py -v
```

> [!WARNING]
> **验证边界声明**：
> 本模块目前的 9 项测试已在 CPU 合成数据下全部通过。尚未在包含 Isaac Lab 物理引擎的真实训练循环中测试显存吞吐性能，也未验证高并发（数千环境实时交互）条件下的多线程与异步读写安全。

## 阅读导航与后续模块

- 深入理解输入的单步转移标准：请参考 [D01 经历记录契约](../transition_schema/README_HUMAN.md)；
- 查看抽样出的连续片段如何折算为 TD 目标：请继续阅读 [D03 连续多步回报](../n_step_returns/README_HUMAN.md)；
- 理解读出的观测特征如何进行标准化与窗口切片：请参考 [D04 观测归一化](../normalization/README_HUMAN.md) 与 [D05 历史窗口](../history_windows/README_HUMAN.md)。
