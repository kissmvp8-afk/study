# D02 · 经验池写入与抽样｜AI 实施规范

状态：单元验证通过（纯数据范围）。已实现 [buffer.py](buffer.py) 与 [test_buffer.py](test_buffer.py)，11 项 CPU 检查通过（Python 3.11 + PyTorch；其中 2 项为 T03 所需的保存恢复接口）；使用人工数据，已接入 T02 假环境采集，未接入仿真，未进行训练或性能验证。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

保存和重复抽取过去的真实经历，同时保持时间关系和容量边界。

术语先说明：循环缓冲区：容量满后逐步覆盖最旧数据的存储方式。覆盖代次：用于区分旧槽位内容与新内容的标识。

## 与原 PPO 的关系

原 PPO 更新后清空近期数据；SAC 保存多批经历，反复使用。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/storage/replay_buffer.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/storage/replay_buffer.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | D01 的经历记录、容量、并行环境数、存放设备和抽样随机状态。 |
| 输出 | 单步样本或同一环境连续片段的索引，以及有效记录数、内存占用等统计。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [buffer.py](buffer.py)

| 符号 | 签名 | 职责、单位、异常与副作用 |
|---|---|---|
| `ReplayBufferError` | `ValueError` 子类 | 写入或读取不符合约定 |
| `bytes_per_record` | `(q_dim: int) -> int` | 一条记录的字节数：D01 全部字段 + 1 个 int64 写入序号。q_dim=54 时为 3158，241 时为 4654 |
| `estimate_storage_bytes` | `(num_envs, capacity_per_env, q_dim) -> int` | 分配前估算存储字节数。例：4096 环境 × 每环境 1221 条 ≈ 500 万条，q_dim=241 时约 23.3 GB。不提供默认容量 |
| `ReplayBuffer` | `(num_envs, capacity_per_env, q_dim, device="cpu")` | 构造时一次性分配存储；属性 `size`（已存条数）、`storage_bytes`（实际字节数） |
| `ReplayBuffer.append` | `(batch: TransitionBatch) -> Tensor[B]` | 先用 D01 `validate_transition` 校验；每条写到该 env_id 行的下一个槽位，返回槽位。一批内 env_id 不能重复且须在 `[0, num_envs)`，设备须一致。**复制**：索引赋值写入，之后修改来源不影响池内记录 |
| `ReplayBuffer.valid_starts` | `(n_steps) -> (env_ids[K], slots[K])` | n 步片段的全部有效起点：同一轮内连续存在记录，直到遇到真终止 / 纯截断，或凑满 n 条；既不够 n 条又没结束的起点暂不可用。n=1 时任何已写入记录都有效 |
| `ReplayBuffer.sample_starts` | `(batch_size, n_steps, generator=None) -> (env_ids, slots)` | 从有效起点有放回地均匀抽取；`generator` 为 CPU `torch.Generator`，同种子同内容时结果可复现；没有有效起点时报错 |
| `ReplayBuffer.read_sequence` | `(env_ids[M], slots[M], length) -> SequenceBatch` | 读取每个起点开始的 L 条记录。**复制**：高级索引返回新张量，修改读出结果不影响池内记录 |
| `ReplayBuffer.state_dict` / `load_state_dict` | `() -> dict` / `(state) -> None` | 为 T03 检查点补充的最小接口：导出形状参数、全部字段、写入序号与每环境写入次数（均为副本）；恢复时校验形状与字段，复制进已分配的存储。缺少写入序号与写入次数会把新旧记录错误连接 |
| `SequenceBatch` | 字段 `fields: dict[str, Tensor[M,L,...]]`、`valid: Tensor[M,L] bool`、`env_ids`、`slots`；属性 `length`；方法 `at(offsets[M]) -> TransitionBatch` | `valid[m,j]` 表示第 j 条与起点属于同一环境同一轮且连续未断，断开后全部为 False；`at` 取每行第 offsets[m] 条，指向无效记录时报错 |

### 存储布局与连续性

- **布局**：每个 D01 字段一块张量 `[num_envs, capacity_per_env, *字段形状]`，dtype 与 D01 相同。容量单位是“每个环境的记录条数”，总容量为 num_envs × capacity_per_env。
- **覆盖代次**：每个环境有独立写指针，第 w 次写入落在槽位 `w % capacity_per_env`，并把 w 记入 `_write_seq`。槽位 s 的后继是 s+1，当且仅当 `_write_seq[s+1] == _write_seq[s] + 1`、episode_id 相同、step_index 加一。覆盖后序号改变，旧起点不会接到新内容上。
- **不依据数组相邻**：时序只看写入序号与 D01 编号。
- **与 D01 的衔接**：直接导入 D01 的内部函数 `_expected_layout` 分配存储，保证布局只有一个来源。D01 没有公开这个函数，本次不改 D01；建议以后由 D01 正式公开。

调用方：D03 `build_n_step_batch` 使用 `SequenceBatch`；D04、D05 测试使用读出结果。采集循环（E05 / T02）尚未实现。

## 必须成立的规则

- 第一版存编码前完整历史，暂不缓存会随编码器训练变化的隐藏特征。
- 循环覆盖后不能把末尾旧记录和新开头错误连接。
- 缓存容量须按实际字段字节数估算；500 万条不是 TITA 的默认保证值。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：解释环形覆盖代次、同环境连续性检查、有效索引判定、内存估算以及不缓存过期隐藏特征的原因。

## 验收案例（已执行，纯数据范围）

全部使用人工数据：观测编码 `code = env*10000 + episode*100 + step`（当前观测填 code，真实下一观测填 code+0.5），奖励等于 code；每轮 step 从 0 开始，`history_length = min(step+1, 10)` 是测试约定。

| 案例 | 对应测试（[test_buffer.py](test_buffer.py)） | 结果 |
|---|---|---|
| D01 → D02 写入已知编号，读回逐字段一致；来源与读出结果被修改都不影响池内记录 | `WriteReadTest`（2 项） | 通过 |
| 容量 4 写 10 次：只剩 step 6..9；n=3 时只有 step 6、7 可作起点（8、9 缺后继，9 的“下一槽位”是更旧的 step 6） | `OverwriteTest.test_wrapped_ring_keeps_only_valid_starts` | 通过 |
| 多次覆盖后抽 64 个起点：全部凑满 3 条或在边界结束，编号连续、同环境同一轮 | `OverwriteTest.test_sampled_sequences_are_consecutive_after_many_overwrites` | 通过 |
| 多环境交错写入只拼同环境片段；真终止 / 纯截断处截断片段；固定种子重复抽样结果相同 | `MultiEnvAndBoundaryTest`（3 项） | 通过 |
| 字节数与手算一致；重复 env_id、越界、无有效起点、n 超过容量、读取无效记录等报错 | `CapacityAndErrorTest`（2 项） | 通过 |
| 保存恢复后 n=1/2/3 的有效起点、读取结果与下一次写入槽位都相同，导出结果是副本；形状不符时拒绝恢复 | `StateTest`（2 项，`f81a92a` 新增） | 通过 |

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/replay_buffer/test_buffer.py -v
```

## 故障定位

偶发巨大 Q 目标先检查是否读到跨环境或已覆盖记录；内存不足先输出分字段占用。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [D01 经历记录契约](../../experience_returns/transition_schema/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（纯数据范围）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [buffer.py](buffer.py)：`ReplayBuffer`（`append`、`valid_starts`、`sample_starts`、`read_sequence`、`state_dict`、`load_state_dict`）、`SequenceBatch`、`bytes_per_record`、`estimate_storage_bytes`；依赖 D01 `TransitionBatch`、`validate_transition`、`FIELD_NAMES`、`_expected_layout`。[test_buffer.py](test_buffer.py)：5 个测试类共 11 项，其中的 `make_records` 被 D03～D05 测试复用。调用方：D03、T02 `Trainer`（写入、抽样、读取）、T03 检查点（`state_dict` / `load_state_dict`） |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；存储布局 `[num_envs, capacity_per_env, ...]`；构造时分配存储，`append` 修改池内存储，其余方法只读 |
| 代码注释核对 | 已核对：存储布局、容量单位与覆盖代次（文件头、`_continuity`）；不依据数组相邻（文件头）；写入与读取的复制责任（`append`、`read_sequence`）；有效起点规则（`valid_starts`）；与作者全局写指针做法的差异（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU）；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/replay_buffer/test_buffer.py -v` → `Ran 9 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除，不影响结果 |
| 未验证项 | 采集循环逐步写入（E05 / T02）；GPU 设备上的存储与抽样速度；大容量下的实际内存占用；与 N05 训练批大小的配合；仿真、训练与性能 |
| 代码版本 | 代码与测试：`3838fe4a0d00bf2e8e102edda39a58d5c21e9f0c`（分支 `claude/experience-returns`，基于 `c06509e`）；为 T03 补充 `state_dict` / `load_state_dict` 与 `StateTest`：`f81a92a2720694b31460f9a9174f758ce9524a55`（分支 `claude/training-core`，未改动写入、连续性与抽样算法）。补充后 `test_buffer.py` 共 11 项，同一环境下 `Ran 11 tests … OK` |
| 本次改动及理由 | 首次实现 D02：作者用全局写指针、假定所有环境同步写入，不满足“按显式编号判断连续”；改为逐环境写指针 + 写入序号 + 编号检查 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；D01 `schema.py`；带注释的 `buffer.py`、`test_buffer.py`；作者 `rsl_rl_sac` 的 `replay_buffer.py`（`_insert`、`_generate_valid_indices`）。建议 Codex 重点解释 `_continuity` 中的覆盖代次、`valid_starts` 的逐步递推，以及读写两端的复制 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；面向有 Python 基础的读者，实际核对版本与限制见本目录 README_HUMAN.md |
