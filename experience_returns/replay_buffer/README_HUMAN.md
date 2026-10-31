# D02 · 经验池写入与抽样｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色；已补充 T03 保存恢复接口，保留原有效正文 |
| 技术核对状态 | 2026-10-31，Codex 复核本版本 buffer.py、test_buffer.py，补充 state_dict/load_state_dict、2 项 StateTest 与 T03 调用；既有写入/连续性/抽样讲解保留，本次不开展全面代码审核 |
| 语言润色状态 | 2026-10-31，Gemini 已完成整体语言润色（`d29af80`）；后续技术修正已复核（`22862f1`）；按用户决定本轮不再进行局部润色，保留已有润色内容 |
| 已核对代码版本 | 本次 `a07d7b22e5bc2f40c462c01cc3378f7bcf3fe0c7`（claude/training-core，PR #7），包含保存恢复接口提交 `f81a92a2720694b31460f9a9174f758ce9524a55`；与远端及验收版本一致。原正文基于 `7071e41`、后经 `d29af80` / `22862f1` 复核 |
| 待同步代码版本 | 无；T03 所需接口已同步，写入、连续性与抽样算法未改动 |
| 实际阅读材料 | 根规范、上层及本目录 AI README；[buffer.py](buffer.py)、[test_buffer.py](test_buffer.py) 全文；D01 `schema.py`/测试、D03 `returns.py`/测试，D04/D05 衔接测试；固定 PPO `rollout_storage.py` 相关片段。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与工程验证证据)；此前另对照 E02、E05、N04、N05、T05 契约及 `5b47469`→`d29af80` 文档差异；本轮实际阅读 T01/T02/T03 实现与测试及 AI 交接表 |
| 验证依据与限制 | 用户报告 D01～D05 共 55 项 CPU 检查已独立通过；本模块 9 项的既有结果见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 和 [AI 交接表](README.md#实现证据与文档交接)。本轮引用 PR #7 的 D02 11 项 CPU 通过记录（其中新增 StateTest 2 项），T03 另有 10 项恢复/流程检查；已在 CPU 假环境衔接，未接入 Isaac 仿真或机器人训练。本次不重跑算法测试，执行 verify.py --docs 并核对接口/片段/链接，结果见 [T 大模块记录](../../training_validation/README_HUMAN.md#阅读材料与证据) |

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
| **池内底层存储** | 经验池预分配内部张量 | 每个字段扩展为 `[E, K, *字段形状]`：<br>如历史为 `[E, K, 10, 33]`，编号为 `[E, K]`，`_write_seq` 为 `[E, K]` | 为每个环境分配固定的 $K$ 个槽位。**槽位索引（Slot）是数组中的位置编号，不等于动作步号或内存地址** |
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
| `bytes_per_record` | `(q_dim: int) -> int` | 复用 D01 `_expected_layout`，按各字段元素数与 dtype 字节数求和，再加一个写入序号；不分配整座经验池 |
| `estimate_storage_bytes` | `(num_envs, capacity_per_env, q_dim) -> int` | 返回 $E \times K \times \text{单条字节}$，不含逐环境写入计数器和分配器等额外开销 |
| `ReplayBuffer.__init__` | `(num_envs, capacity_per_env, q_dim, device='cpu')` | 构造函数。检查 $E, K > 0$，一次性预先分配全部张量底座、代数矩阵 `_write_seq[E, K]` 与各环境写入计数器 `_writes[E]` |
| `ReplayBuffer.size` | 属性，返回 `int` | 当前经验池中已写入且尚未被丢弃的有效记录总条数（不等于有效 n 步起点的数量） |
| `ReplayBuffer.storage_bytes` | 属性，返回 `int` | 精确统计池内所有张量（含写指针）实际占用的显存/内存物理字节数 |
| `ReplayBuffer.append` | `(batch: TransitionBatch) -> torch.Tensor` | 校验批次数据，核对设备与环境编号唯一性；定位各环境的下一个环形槽位并存入数据，返回所占槽位索引 |
| `ReplayBuffer._continuity` | 内部方法，返回 `(present, linked, boundary)` | 计算 `[E, K]` 的布尔张量：当前槽位是否有数据、是否能合法链接到下一个物理槽位、当前步是否为轮次终止边界 |
| `ReplayBuffer.valid_starts` | `(n_steps: int) -> tuple[Tensor, Tensor]` | 核心检索算法。输入 $1 \le n \le K$，返回所有能够向前构成合法 n 步片段的 `(env_ids, slots)` 起点坐标 |
| `ReplayBuffer.sample_starts` | `(batch_size, n_steps, generator=None)` | 依托 `valid_starts` 检索候选集，进行有放回均匀随机抽样；支持传入 `torch.Generator` 以实现确定性复现 |
| `ReplayBuffer.read_sequence`| `(env_ids, slots, length: int) -> SequenceBatch` | 给定起点和读取长度 $L$，沿环形槽位提取连续矩阵，并输出 `valid` 连续性掩码 |
| `ReplayBuffer.state_dict` | `() -> dict` | 返回 E/K/C、全部字段存储、写入序号和逐环境累计写入次数，张量均为副本，供 T03 保存 |
| `ReplayBuffer.load_state_dict` | `(state: dict) -> None` | 核对 E/K/C 与存储字段集合，再复制回已分配张量；修改当前经验池，不读取文件或恢复随机数 |
| `SequenceBatch` | 冻结数据类 | 保存 `fields`、`valid`、起点 `env_ids[M]` 与读出槽位 `slots[M,L]`；冻结属性不使内部张量只读 |
| `SequenceBatch.length` | 属性，返回 `int` | 当前时序矩阵的跨度长度 $L$（即 `valid.shape[1]`） |
| `SequenceBatch.at` | `(offsets: Tensor) -> TransitionBatch` | 输入各行相对偏移 `[M]`，检查越界与 `valid`，通过高级索引复制并组装单步记录；不再执行 D01 全字段校验 |

## 供 T03 使用的经验池保存与恢复

T03 的检查点是一次训练状态的文件。D02 只提供内存字典接口，由 [T03 checkpoint.py](../../training_validation/checkpoints/checkpoint.py) 的 `collect_training_state` 调用 `buffer.state_dict()`，`save_checkpoint` 负责写文件；恢复时 `restore_checkpoint` 调用 `buffer.load_state_dict(state["replay"])`。D02 自己不做文件操作。

`state_dict()` 除了保存原始 13 个字段，还必须保存覆盖代次和下一次写入位置：

```python
"storage": {name: tensor.clone() for name, tensor in self._storage.items()},
"write_seq": self._write_seq.clone(),
"writes": self._writes.clone(),
```

第一行复制全部预分配字段（包括尚未使用的槽位），第二行复制每槽位序号，第三行复制每环境累计写入次数。只存观测而漏掉后两项，恢复后可能覆盖错槽位或错误判断连续性。返回字典另含 `num_envs/capacity_per_env/q_dim`。这些张量与原池不共享存储，但完整复制会增加临时内存占用。

`load_state_dict(state)` 先要求 E/K/C 与当前对象相等，存储字段集合相同，否则抛 `ReplayBufferError`；然后逐字段 `copy_`，再复制 `_write_seq/_writes`，保持当前预分配存储。它是 `state_dict` 输出的配套恢复接口，不是任意外部字典的完整语义验证器。复制完成后，修改传入字典不影响池内容。

新增 `StateTest.test_round_trip_and_continued_writes` 的人工例子是 E=2、K=4、C=6：环境 0 写了 6 次，环境 1 写了 3 次。恢复后下一条分别落在槽位 `6%4=2` 和 `3%4=3`；n=1/2/3 的有效起点、读出字段与 valid 都与原池一致。测试还修改导出的奖励张量，确认已恢复池不受影响。`test_rejects_mismatched_shape` 则核对 K=4 的状态不能装入 K=5 的池。

经验池状态不包含抽样生成器。T02 持有的 replay 随机流由 T03 单独保存；“同池 + 同随机流”才支持下一批抽样复现。带池的 `exact_replay` 恢复已在同配置、CPU、同 PyTorch 版本下验证下一批学习；不带池的 `model_only` 恢复会新建空池。两者都不恢复环境现场，T03 将轮次编号加一，防止新采集与旧回合拼接。详见 [T03 人类版](../../training_validation/checkpoints/README_HUMAN.md)。

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
nxt = (torch.arange(self.capacity_per_env, device=self.device) + 1) % self.capacity_per_env
episode = self._storage["episode_id"]
step = self._storage["step_index"]
linked = (
    present
    & (self._write_seq[:, nxt] == self._write_seq + 1)
    & (episode[:, nxt] == episode)
    & (step[:, nxt] == step + 1)
)
```

1. **当前已写入**：`present` 排除当前空槽；下一槽位若为空，其序号为 -1，也不可能通过随后“序号加一”的判断；
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
- `open_`：标记哪些起点在当前步**尚未遇到真终止或纯时间截断**、因而仍需后继记录。

`link_j` 将第 j 条与下一条的连接标记移到起点位置；`~open_ | link_j` 表示只有仍开放的起点才需要这条连接。最后一行在连接断开或下一条到达结束边界时关闭 `open_`，已经合法结束的起点不再要求更多数据。

这带来了极为关键的工程判定：
- **情况 A（未遇到终止，但数据不足）**：在上文容量 $K=4$ 且没有结束的例子中，如果算法请求 $n=3$ 的片段，只有步数 6 和 7 属于有效起点；步数 8 和 9 后面缺乏足够的数据步，属于“未来数据暂缺”，**绝对不能因为只剩 2 条或 1 条就强行拿去训练**！
- **情况 B（合法提前结束）**：另取只写到步号 7、且 7 为真终止的情形。从步号 6 开始只有两条记录，但已经到达结束边界，仍是**合法的 3 步起点**，读出 `valid = [True, True, False]`。纯时间截断也允许提前闭合，后续自举开关由 D03 区分。

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

两者的范围不同：估算值不含 `_writes[E]`，`storage_bytes` 则计入这 $8E$ 字节；两者都不含分配器、临时计算张量与网络的额外开销。这里是按字段计算的容量预算，不是五百万条数据的实际性能测量。

## `test_buffer.py`：既有 9 项与新增 2 项保存恢复检查

测试脚本用 `code_of(env, episode, step) = env * 10000 + episode * 100 + step` 生成可辨识的人工标记。在现有测试的小整数范围内，可据此核对数据归属；该编码不是任意大编号的无损编码，也不是真实机器人观测。

测试辅助函数的调用关系如下：`make_records(entries, terminated, time_out, rewards, q_dim)` 按编号构造 13 个字段，再交给 D01 `make_transition`；`terminated` / `time_out` 集合指定批次中的行号。`write_episode(buffer, env, episode, steps, terminal_at, time_out_at)` 逐步调用它并 `append`；`start_steps(buffer, n)` 将 `valid_starts` 的槽位经 `read_sequence` 换算成 `(env, episode, step)` 列表，供断言使用。D03～D05 衔接测试也复用 `make_records`；默认人工 Q 维数为 6，不代表真实 C。

| 测试类与具体用例名称 | 核心验证逻辑与覆盖边界 |
|---|---|
| `WriteReadTest.test_round_trip_all_fields` | 写入两条记录，经 `read_sequence` 和 `at` 提取后比对全部 13 个 D01 字段，确认完全无损还原 |
| `WriteReadTest.test_copies_on_write_and_read` | 写入后原地篡改外部来源张量，读取后原地篡改读出张量；断言池内核心存储原封不动 |
| `OverwriteTest.test_wrapped_ring_keeps_only_valid_starts` | 在容量 4 的队列中连续压入 10 步数据，严密验证槽位绕回后仅暴露步数 6..9，并核对 n=3 和 n=4 的有效起点 |
| `OverwriteTest.test_sampled_sequences_are_consecutive_after_many_overwrites` | 两个环境各写入 23 次，每环境容量为 5；一次抽取 64 个起点，核对各片段有效前缀的环境、轮次与步数连续性 |
| `MultiEnvAndBoundaryTest.test_interleaved_envs_only_join_same_env` | 多个环境以异步步调交错写入，确认时序链接绝不会跨环境串线 |
| `MultiEnvAndBoundaryTest.test_episode_boundaries` | 验证物理真终止和超时截断能使片段合法提前闭合，而新一轮未填满的起点被精准过滤 |
| `MultiEnvAndBoundaryTest.test_reproducible_sampling` | 验证固定随机种子下，`sample_starts` 能够产生 100% 确定性的抽样结果 |
| `CapacityAndErrorTest.test_bytes_follow_actual_fields` | 核对手算内存字节数公式，并验证包含写指针开销后的精确内存统计 |
| `CapacityAndErrorTest.test_rejects_bad_writes_and_reads` | 防御性测试：拦截重复环境号写入、拦截越界环境号、拦截空起点抽样、拦截无效偏移访问 |

上表保留原 9 项；新增 `StateTest` 两项见前文保存恢复小节，目前共 11 项。脚本入口为 `unittest.main()`。通过结果来自既有 CPU 记录，本次只静态核对测试内容。

复现测试时，可在仓库根目录执行：
```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/replay_buffer/test_buffer.py -v
```

> [!WARNING]
> **验证边界声明**：
> 本模块目前的 11 项测试已由既有记录报告在 CPU 合成数据下全部通过，本轮未重跑。尚未在包含 Isaac Lab 物理引擎的真实训练循环中测试显存吞吐性能，也未验证高并发（数千环境实时交互）条件下的多线程与异步读写安全。

## 阅读导航与后续模块

- 深入理解输入的单步转移标准：请参考 [D01 经历记录契约](../transition_schema/README_HUMAN.md)；
- 查看抽样出的连续片段如何折算为 TD 目标：请继续阅读 [D03 连续多步回报](../n_step_returns/README_HUMAN.md)；
- 理解读出的观测特征如何进行标准化与窗口切片：请参考 [D04 观测归一化](../normalization/README_HUMAN.md) 与 [D05 历史窗口](../history_windows/README_HUMAN.md)；
- 查看何时写池、何时回放以及保存恢复：请继续阅读 [T02 调度](../../training_validation/training_loop/README_HUMAN.md) 和 [T03 检查点](../../training_validation/checkpoints/README_HUMAN.md)。
