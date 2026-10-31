# D01 · 经历记录契约｜AI 实施规范

状态：单元验证通过（纯数据范围）。已实现 [schema.py](schema.py) 与 [test_schema.py](test_schema.py)，20 项 CPU 检查通过（Python 3.11 + PyTorch）；使用人工数据，未接入 E05 末状态捕获、D02 经验池或仿真，未进行训练或性能验证。

[人类版（待 Codex 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

定义跨环境接口、经验池和训练器共用的数据记录，避免同名字段含义不同。

术语先说明：经历记录：一次动作前后发生的事实。bootstrap mask：是否允许接上未来预测的开关。

## 与原 PPO 的关系

从原短期 PPO rollout 数据，改为可长期保存并重新计算价值的 SAC 真实经历。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真中验证。

核对结果：原 PPO 的 wrapper 把 `dones = terminated | truncated` 合并，并在 `np3o.py` 中把时间截断的价值预先加进奖励（`rewards += gamma * values * time_outs`）。D01 只存原始奖励，并分开保存两个标记。

## 作者实现参考

- [作者 SAC：`rsl_rl/storage/replay_buffer.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/storage/replay_buffer.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

作者实现存 `dones`（终止或截断）与 `bootstrap`（= time_outs），目标中使用 `bootstrap_mask = bootstrap + 1 - dones`（`sac.py`）。同一步既终止又到时，该式结果为 1，会接后续价值，与本项目“真终止优先”的约定不符。因此 D01 改为只存真终止与纯时间截断，mask 只由真终止推导，并拒绝两者同时为真。作者在 `process_env_step` 中用 `extras["time_outs_obs"]` 替换末状态，这一步属于 E05，不在 D01 中实现。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 当前/下一策略历史、当前/下一 Q 观测、动作、原始奖励、结束原因、环境与轮次编号。 |
| 输出 | 统一的记录对象和校验结果；动作必须与实际送入环境的策略输出处在相同坐标。 |

## 实际文件与接口

实现与验证都在本目录。数据容器为 PyTorch 张量，只依赖 `torch`；测试另外导入 E03 的 `contract.py`，用于核对动作维数。

依赖与安装：仓库目前没有统一的依赖文件（属于 T01）。本次验证使用 uv 临时环境，不修改系统 Python：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/transition_schema/test_schema.py -v
```

在已有 torch 的环境中（例如 Isaac Lab 环境），也可以直接运行 `python -B experience_returns/transition_schema/test_schema.py -v`，或先 `pip install torch`。测试不需要 GPU。

### [schema.py](schema.py)

| 符号 | 签名 / 字段 | 职责、单位、异常与副作用 |
|---|---|---|
| 常量 | `HISTORY_LENGTH=10`、`POLICY_OBS_DIM=33`、`ACTION_DIM=8`、`FIELD_NAMES` | 与根 README、E02、E03 的约定一致；测试核对 `ACTION_DIM == len(TITA_ACTION_LAYOUT)` |
| `TransitionSchemaError` | `ValueError` 子类 | 消息中写明出错的字段名与行号 |
| `TransitionBatch` | 冻结 dataclass，13 个 `torch.Tensor` 字段（见下表）；属性 `batch_size` | 一批 B 条记录，第 0 维为批维。冻结只阻止重新绑定字段，不阻止原地修改张量内容，下游不得原地修改 |
| `make_transition` | `(data: dict[str, torch.Tensor], q_dim: int) -> TransitionBatch` | **复制责任在此**：逐字段执行 `detach().clone()`（保持设备与类型，不做类型转换），之后调用 `validate_transition`。键必须恰好等于 `FIELD_NAMES`，缺失或多余（如 `reset_policy_history`、`bootstrap`）都会报错 |
| `validate_transition` | `(batch: TransitionBatch, q_dim: int) -> None` | 检查形状、类型、批维一致且 ≥ 1、同一设备、浮点字段有限、`terminated` 与 `time_out` 不同时为真、编号 ≥ 0、有效长度在 1..10，并且 `next_history_length == min(history_length + 1, 10)`；列出全部问题后抛 `TransitionSchemaError`；无副作用 |
| `derive_bootstrap_mask` | `(batch) -> torch.Tensor` | `[B,1] float32`，等于 `1 - terminated`：真终止为 0，正常步和纯时间截断为 1。不修改 batch |
| `describe_transition` | `(batch, row: int = 0) -> str` | 单条记录的摘要：编号、有效长度、策略动作、原始奖励、结束原因与 mask、形状和设备；`row` 越界时抛 `IndexError` |

### `TransitionBatch` 字段

“当前”指执行动作前的时刻 t；“真实下一”指动作执行后、任何重置发生之前的时刻 t+1。

| 字段 | 形状 / 类型 | 含义与单位 |
|---|---|---|
| `policy_history` / `next_policy_history` | `[B,10,33]` float32 | 当前与真实下一的策略历史，含当前帧，从旧到新；尺度与 E02 一致（未经 D04 归一化） |
| `q_observation` / `next_q_observation` | `[B,C]` float32 | 当前与真实下一的 Q 输入；C 由调用方以 `q_dim` 给出（平地预期 54，复杂地形预期 241），不推断、不截断或补齐 |
| `action` | `[B,8]` float32 | 实际送入环境的策略动作，策略动作坐标，无物理单位，顺序同 E03 |
| `reward` | `[B,1]` float32 | 环境原始奖励，不含价值补偿 |
| `terminated` | `[B,1]` bool | 真终止 |
| `time_out` | `[B,1]` bool | 纯时间截断，与 `terminated` 互斥 |
| `env_id` / `episode_id` / `step_index` | `[B]` int64，≥ 0 | 并行环境编号、该环境内的轮次编号、本轮内的动作序号；时序只看这些编号，不看数组相邻 |
| `history_length` / `next_history_length` | `[B]` int64，1..10 | 两段历史中属于本轮真实观测的帧数，其余为重置填充 |

不保存会随网络训练变化的历史编码特征；不保存重置后观测；不另存 bootstrap 开关（由 `derive_bootstrap_mask` 推导，因此无需一致性检查）。

与 E05 的分工：E05 负责在重置前捕获真实末状态，并把同一步“摔倒又到时”的情况只标为 `terminated`；D01 接收结果，拒绝两标记同时为真。mask 也只依赖 `terminated`，所以即使绕过校验，结果仍按真终止处理。

调用方：目前只有本目录的验证程序。计划由 E05 / T02 采集循环调用 `make_transition`，D02 写入经验池，D03 / N05 使用 `derive_bootstrap_mask`。这些模块尚未实现。

## 必须成立的规则

- reward 不包含采集时预先计算的价值补偿。
- true_next_observation 与用于下一次环境交互的 reset_observation 分开命名。
- 记录含 env_id、episode_id、时间序号；不以数组相邻推断时间相邻。

## 算法约定与实现提示

一条经历描述“在本轮哪个时刻，看到了什么，做了什么，真正到达了哪里”。下表是第一版字段要求，Claude 可以选择合适的数据容器，但不能省略含义。

| 字段含义 | 形状或类型 | 用途 |
|---|---|---|
| 当前与真实下一策略历史 | 每项 [B,10,33] | 用当前编码器重新解释旧经历 |
| 当前与真实下一 Q 观测 | 每项 [B,C] | 评价原动作与构造尾部目标 |
| 实际策略动作 | [B,8] | 保持与送入环境的动作坐标一致 |
| 原始奖励 | [B,1] | 多步奖励累计，不预加价值 |
| 真终止、纯时间截断 | 每项 [B,1] 布尔值 | 确定序列边界及是否接后续估值 |
| 环境编号、轮次编号、时间序号 | 整数 | 检查时序，防止跨环境、跨轮次拼接 |
| 当前与下一有效历史长度 | 整数，范围 1..10 | 区分真实历史与重置填充；辅助窗口需 6 个真实时刻 |

B 表示这一批记录的数量；C 表示 Q 观测的字段总数。真实速度监督标签从已定义的训练侧字段读取。是否允许后续估值由真终止标记推导，若另外存储开关则检查二者一致。重置后观测由采集循环用于下次动作，不能覆盖表中的真实下一观测。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：逐字段标注当前/真实下一状态、env/episode/时间编号、布尔标记和数组形状；说明谁负责复制与校验。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，纯数据范围）

全部使用人工数据：末状态填 7.0，重置状态填 -3.0；`q_dim=54` 只是示例。

| 案例 | 对应测试（[test_schema.py](test_schema.py)） | 结果 |
|---|---|---|
| 同一批含正常步、真终止、纯截断，mask 为 [1, 0, 1]；两标记同时为真时被拒绝，绕过校验时 mask 仍为 0 | `BootstrapMaskTest`（2 项） | 通过 |
| 保留传入的真实末状态而非重置状态；拒绝多余的 `reset_policy_history` 字段；把重置后历史填进 next 字段时，由有效长度检查发现 | `TrueNextObservationTest`（3 项） | 通过 |
| 创建后原地修改全部来源张量，记录不变，且与来源不共享存储 | `CopyResponsibilityTest` | 通过 |
| 编号与有效长度原样保存；长度越界、编号为负、历史被重复推进时报错 | `IdentityAndHistoryLengthTest`（3 项） | 通过 |
| 动作非 8 维、批维不一致、字段缺失、奖励 NaN/inf、类型不符、Q 维数不符（241 对 54，不截断）、历史形状错、不是张量、q_dim 非法，均指出具体字段 | `FieldErrorTest`（9 项） | 通过 |
| `ACTION_DIM` 与 E03 一致；摘要输出 | `ContractConsistencyTest`（2 项） | 通过 |

运行命令见上文“依赖与安装”。

## 故障定位

Q 输入动作与环境实际动作不一致时，沿动作坐标和后续裁剪处理查找。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E02 观测接口](../../environment_control/observation_interface/README.md)
- [E03 八维动作定义表](../../environment_control/action_contract/README.md)
- [E05 终止、截断与重置前观测](../../environment_control/episode_lifecycle/README.md)：末状态捕获与真终止优先由 E05 负责

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（纯数据范围）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [schema.py](schema.py)：`TransitionBatch`、`make_transition`、`validate_transition`、`derive_bootstrap_mask`、`describe_transition`、`TransitionSchemaError`；[test_schema.py](test_schema.py)：6 个测试类共 20 项，另导入 E03 `TITA_ACTION_LAYOUT`。目前唯一的调用方是验证程序 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”两张表；`make_transition` 会分配副本内存，其余函数无副作用 |
| 代码注释核对 | 已逐项核对本模块的注释重点：逐字段的当前 / 真实下一含义、编号、布尔标记与形状（`TransitionBatch` docstring）；复制责任（`make_transition`）；校验职责与 E05 分工（`validate_transition`、文件头）；真终止优先（`derive_bootstrap_mask`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU）；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/transition_schema/test_schema.py -v` → `Ran 20 tests … OK`。临时环境未装 numpy，torch 打印了一条 numpy 未安装的 UserWarning，不影响结果 |
| 未验证项 | E05 在 Isaac Lab 中于重置前捕获末状态（含 `[B,10,33]` 历史掩码展开）；实际 C 值（54 / 241）需 E02 从运行时观测组校验；E01/E05 编号来源（episode_id、step_index 由谁维护）；D02 写入与抽样；GPU 设备上的复制；仓库统一依赖配置（T01）；仿真、训练与性能 |
| 代码版本 | 基于 `759be40`（main，含 E03、E04）；提交号以本分支 `claude/d01-transition-schema` 的 PR 记录为准 |
| 本次改动及理由 | 首次实现 D01：用 torch 张量作为唯一容器，与环境和经验池一致；为满足真终止优先，以“真终止 + 纯截断”代替作者的 dones + bootstrap；由 `make_transition` 统一复制，防止来源缓冲区的原地修改污染记录 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；E02、E03、E05、D02、D03 的接口约定；带注释的 `schema.py`、`test_schema.py`；上述验证记录；原 PPO `np3o.py` 的 time_outs 奖励补偿与 `wrapper.py`，作者 `sac.py` 的 `process_env_step` 与 bootstrap_mask。建议 Codex 重点解释 `make_transition` 的复制、`validate_transition` 中 next_history_length 的规则与两标记互斥，以及 `derive_bootstrap_mask` 与作者公式的差别 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文 |
| 技术初稿 / 语言润色 | 待 Codex 技术初稿；其后由 Gemini 润色同一份人类版 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
