# E02 · 观测接口｜AI 实施规范

状态：代码已实现，CPU 单元验证通过（纯数据范围）。已实现 [observations.py](observations.py) 与 [test_observations.py](test_observations.py)，8 项检查通过；运行时观测组维数与实际数值待 Windows + Isaac 验证（由 E01 `run_sac_isaac.py --mode inspect` 核对）。未进行训练或性能验证。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

明确 Actor 与 Q 分别看到什么，让每一个观测数字的顺序、尺度和来源可查。

术语先说明：观测接口：模块之间对数据内容、排列和单位的约定。训练侧额外信息：仿真可提供、运行中的 Actor 未必能直接获得的信息。

## 与原 PPO 的关系

保留原策略侧 33 维观测及历史；Q 继承原 Critic 可见信息，但评价时还需要当前待评价动作。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块的字段契约与历史规则已实现并在 CPU 上验证，尚未在仿真中验证。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 环境输出的策略观测、Critic/priv/scanner 观测组、实际字段定义。 |
| 输出 | 编码前的策略历史 [B,10,33]、Q 的状态信息 [B,C]、当前真实速度监督标签与字段说明。B 是并行环境数，C 由运行时观测组确定。 |

## 实际文件与接口

实现与验证都在本目录。[observations.py](observations.py) 把 `../action_contract` 加入 `sys.path`，用 E03 的 `read_baseline_file` 读取固定快照；依赖 PyTorch，不依赖 Isaac。

字段（与快照 `rough_env_cfg.py` 的 `ObservationsCfg` 逐项比对，名称 / 函数 / 缩放 / 是否带噪声都一致）：

| 组 | 字段（维数，环境字段缩放） | 去向 |
|---|---|---|
| policy（33，带噪声） | base_ang_vel 3（×0.25）· projected_gravity 3 · velocity_commands 3（×(2, 2, 0.25)）· joint_pos 8（相对默认角，轮子置 0）· joint_vel 8（×0.05）· actions 8 | Actor 历史 [B,10,33]；**不含 base_lin_vel** |
| critic（36，无噪声） | base_lin_vel 3（×2.0）+ 上面 6 项 | Q 观测第 0～35 列；第 0～2 列是速度标签 |
| priv（18） | contact_state 2 · joint_kp_factor 8 · joint_kd_factor 8 | Q 观测第 36～53 列 |
| scanner（187，仅复杂地形） | height_scan 187（17×11） | Q 观测第 54～240 列 |

joint_pos / joint_vel 按 articulation 关节顺序（`joint_names=".*"`、`preserve_order=True`），由运行时读取，不一定等于 E03 动作顺序。

| 符号 | 签名 | 职责、形状、异常与副作用 |
|---|---|---|
| 常量 | `HISTORY_LENGTH=10`、`POLICY_FRAME_DIM=33`、`VELOCITY_LABEL_COLUMNS=(0, 3)`、`VELOCITY_LABEL_SCALE=2.0` | — |
| `ObservationError` | `ValueError` 子类 | 字段、形状或数值不符 |
| `ObservationField` | 冻结 dataclass：name、func、dim、scale、noisy、meaning | 一个观测项 |
| `POLICY_FIELDS` / `CRITIC_FIELDS` / `PRIV_FIELDS` / `SCANNER_FIELDS` / `GROUP_FIELDS` / `GROUP_CLASSES` | tuple / dict | 上表的字段定义与组名 ↔ 原配置类名 |
| `q_groups` | `(task) -> tuple[str, ...]` | 平地 ("critic", "priv")，复杂地形再加 "scanner"；未知任务报错 |
| `expected_q_dim` | `(task) -> int` | 由字段推出的预期 C：54 / 241 |
| `describe_observation_fields` | `(task) -> list[dict]` | 逐项列出 target、group、field、columns、scale、noisy、meaning |
| `parse_observation_config` / `baseline_observation_config` | `(env_cfg_source) -> dict` / `() -> dict` | AST 解析原配置的各观测组项（注释掉的项自然不在其中） |
| `validate_observations` | `(task, term_names, term_dims, policy_frame_dim=None) -> int` | 用 Isaac `observation_manager.active_terms` / `group_obs_term_dim` 逐项比较名称与维数（同维字段调换也能发现）；policy 每项必须是单帧；返回 C；列出全部问题后抛 `ObservationError` |
| `read_observation_groups` | `(observations, task, q_dim) -> (frame [B,33], q [B,C])` | 按 [critic \| priv \| scanner] 拼接（同原 PPO wrapper）；形状不符或含 NaN/inf 报错，**不截断、不补齐**；不克隆 |
| `velocity_label` / `velocity_label_to_mps` | `(q [B,C]) -> [B,3]` / `(label) -> [B,3]` | 速度标签 = Q 前 3 列（base_lin_vel × 2.0，不是 m/s）；除以 2.0 换回 m/s |
| `push_frame` | `(history [B,10,33], pushes [B], frame [B,33]) -> (新历史, 新计数)` | Isaac Lab 2.3.0 CircularBuffer 规则：从旧到新、最新帧在最后；计数为 0 的环境用该帧填满 10 个槽位；返回新张量，不改输入 |
| `valid_length` | `(pushes) -> [B]` | min(推送次数, 10)，即 D01 的 history_length |
| `PolicyHistory` | `(num_envs, device="cpu")`；`reset(env_ids)`、`commit(env_ids, history, pushes)` | 保存 history / pushes；`reset` 清零并把计数归零（原地）；`commit` 只写回指定行 |

为什么历史放在本模块：锁定的 Isaac Lab 2.3.0 在 `step` 中先重置结束的环境、再计算观测。要拿到含当前帧的重置前历史，只能在重置前调用一次 `compute(update_history=True)`，但这会让未结束环境在重置后的那次 compute 中再推进一次（重复当前帧）；`compute(update_history=False)` 又只返回不含当前帧的旧历史。因此 SAC 配置把 policy 组的 Isaac 历史长度设为 0（E01 `TitaSacFlatEnvCfg` / `TitaSacRoughEnvCfg`），由 `PolicyHistory` 维护，每个环境每步只推进一次。

两层缩放：×0.25、×2.0 等环境字段缩放已经包含在 Isaac 输出中，本模块不再缩放；D04 归一化是之后另做的统计标准化，也不撤销环境缩放。

调用方：E05 [lifecycle.py](../episode_lifecycle/lifecycle.py)（`PolicyHistory`、`push_frame`、`read_observation_groups`、`valid_length`）；E01 [metadata.py](../robot_environment/metadata.py)（`validate_observations`）。

## 必须成立的规则

- Actor 不直接接收仿真真实身体线速度；该信息可供 Q 和速度监督使用。
- 平地 C 预期为 54，复杂地形预期为 241，但必须从实际配置校验，禁止静默截断或补齐。
- 速度监督标签沿用原 base_lin_vel 的 2.0 缩放；不能把缩放后的预测直接当成米/秒。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：标注每组观测的形状、字段顺序和缩放；解释真实速度为何只进入 Q/监督标签，不能泄漏给 Actor。

## 验收案例

### CPU 已执行（[test_observations.py](test_observations.py)，不是仿真验证）

| 案例 | 对应测试 | 结果 |
|---|---|---|
| 四组字段的名称、函数、缩放、是否带噪声与固定快照 `8fe40aa` 逐项一致 | `FieldContractTest.test_matches_baseline_config` | 通过 |
| policy 33 维、C 为 54 / 241；策略输入没有 base_lin_vel；速度标签列 (0, 3)、缩放 2.0；平地无 height_scan | `FieldContractTest.test_dimensions_and_actor_inputs` | 通过 |
| 运行时字段核对：正确时返回 C；调换两个同维字段、priv 维数错误、policy 误用 Isaac 历史都报错 | `FieldContractTest.test_runtime_validation` | 通过 |
| 拼接顺序；241 维不会被截断成 54；三维 policy、NaN 都报错 | `GroupReadingTest.test_concatenation_and_no_padding` | 通过 |
| 速度标签与 m/s 换算 | `GroupReadingTest.test_velocity_label` | 通过 |
| 25 步含两次部分重置，与按 Isaac Lab 2.3.0 源码复写的 `ReferenceCircularBuffer` 逐步完全一致 | `HistoryTest.test_matches_reference_buffer_with_partial_resets` | 通过 |
| 首帧填满、从旧到新、有效长度 | `HistoryTest.test_order_fill_and_length` | 通过 |
| `push_frame` 返回新张量；`commit` 只写指定行 | `HistoryTest.test_push_returns_new_tensor_and_commit_only_selected_rows` | 通过 |

`ReferenceCircularBuffer` 按 Isaac Lab（BSD-3-Clause）`utils/buffers/circular_buffer.py` 的算法复写，作为独立对照。

### Windows + Isaac 待验证

- 平地和复杂地形分别由 E01 `inspect` 读取运行时 `group_obs_term_dim`，确认 C 与 policy 单帧；`interact` 确认形状与数值有限。

运行命令（仓库根目录，无需 Isaac Lab；需要本地有固定快照 `8fe40aa`）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/observation_interface/test_observations.py -v
```

## 故障定位

维数正确仍表现异常时，查字段顺序、缩放和坐标系；不要只重设网络输入长度。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E01 机器人与任务环境](../../environment_control/robot_environment/README.md)：SAC 配置关闭 policy 组 Isaac 历史；`inspect` 调用 `validate_observations`
- [E03 八维动作定义表](../action_contract/README.md)：`read_baseline_file`、`ENV_CFG_PATH`
- [E05 终止、截断与重置前观测](../episode_lifecycle/README.md)：使用 `PolicyHistory` 与 `read_observation_groups`
- [D04 观测归一化](../../experience_returns/normalization/README.md)：在本模块输出之后做统计标准化

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 代码已实现；CPU 单元验证通过（纯数据范围）；Isaac 运行时核对未执行 |
| 实际文件、函数签名与调用关系 | [observations.py](observations.py)（依赖 E03 `baseline_source`）；[test_observations.py](test_observations.py)：3 个测试类共 8 项。调用方：E05 `lifecycle.py`、E01 `metadata.py` |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；策略帧 [B,33]、历史 [B,10,33] float32、计数 [B] int64、Q [B,C]；速度标签为 m/s × 2.0；只有 `PolicyHistory.reset` / `commit` 原地修改 |
| 代码注释核对 | 已核对本模块注释重点：每组形状、字段顺序与缩放（文件头与字段表）；真实速度只进 Q / 标签（文件头、`CRITIC_FIELDS`）；两层缩放的区别；历史放在本模块的原因与 CircularBuffer 规则（文件头、`push_frame`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），Python 3.11.17：`uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/observation_interface/test_observations.py -v` → `Ran 8 tests … OK`；torch 2.7.0（CPU）下同样通过 |
| 未验证项 | Isaac 运行时的字段名称与维数、实际数值与噪声；articulation 关节顺序；仿真、训练与性能 |
| 代码版本 | `8a0ee22937937e9d08ad059e025d416f380d4294`（分支 `claude/environment-integration`），此后 `observations.py`、`test_observations.py` 未改动 |
| 本次改动及理由 | 首次实现 E02：字段契约与快照逐项比对；为了让 E05 在重置前取得末状态而不重复推进历史，把 10 帧历史从 Isaac 移到本模块并复刻 CircularBuffer 规则 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；E01、E05 AI 规范；带注释的 `observations.py`、`test_observations.py`；上述验证记录；Isaac Lab v2.3.0 `circular_buffer.py` 与 `observation_manager.py`。建议 Codex 重点解释：Q 观测的列区间与速度标签；`validate_observations` 为何逐项比较；历史为何不用 Isaac 的 CircularBuffer、`push_frame` 的首帧填满规则 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；已按 a12e69a 同步 PR #8 接入变化，材料与证据见本目录 README_HUMAN.md |
