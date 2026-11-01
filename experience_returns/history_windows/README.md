# D05 · 历史窗口与有效性｜AI 实施规范

状态：单元验证通过（纯数据范围）。已实现 [history.py](history.py) 与 [test_history.py](test_history.py)，11 项 CPU 检查通过（Python 3.11 + PyTorch）；使用人工数据，未接入网络或导出入口，未进行训练或性能验证。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

把环境维护的历史切成当前观测及两段相邻窗口，保证训练和运行读到同样的时间顺序。

术语先说明：窗口：连续若干帧的片段。填充历史：新一轮开始时为凑够输入长度而填入的记录，不代表已经经历过这些时间。

## 与原 PPO 的关系

保留原 10 帧缓冲与 5 帧编码窗口；明确辅助学习需要跨 6 个时刻的两个重叠窗口。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)
- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 包含当前帧、按从旧到新排列的历史 [B,10,33]，以及本轮有效历史长度。 |
| 输出 | 当前帧、截至当前的 5 帧、截至前一时刻的 5 帧，以及辅助损失有效样本标记。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [history.py](history.py)

| 符号 | 签名 | 职责、单位、异常与副作用 |
|---|---|---|
| `WINDOW_FRAMES` / `AUXILIARY_REQUIRED_FRAMES` | `5` / `6` | 编码器窗口帧数（原 PPO `bt_window=5`），以及两个窗口合起来覆盖的时刻数 |
| `HistoryWindows` | 冻结 dataclass：`current[B,33]`、`recent[B,5,33]`、`previous[B,5,33]` | 都是输入历史的切片视图（不复制），调用方不得原地修改 |
| `split_windows` | `(history[B,10,33]) -> HistoryWindows` | `current = h[:,9]`（t），`recent = h[:,5:10]`（t-4..t），`previous = h[:,4:9]`（t-5..t-1）。只切片，不推进历史、不追加当前帧；形状不符时报错 |
| `valid_history_mask` | `(history_length[B], required_frames=6) -> Tensor[B] bool` | `history_length >= required_frames`。当前历史传 `history_length`，下一历史传 `next_history_length`；不读取观测数值 |
| `validate_time_order` | `(env_id[M,L], episode_id[M,L], step_index[M,L]) -> None` | 每行同一环境、同一轮，step_index 逐条加一；列出跨环境、跨轮次、重复 / 跳跃的行号后报错 |

使用约定：
- 训练、评分与导出 / 运行入口都调用同一个 `split_windows`。
- D04 归一化是逐帧变换，先归一化再切片与先切片再归一化结果相同（测试已验证）。
- 辅助学习样本用 `valid_history_mask(..., 6)` 筛选。

原 PPO 导出问题的回归记录（供 T05）：原 `_InferenceWrapper` 把 `history[:, 1:]` 与 `current` 拼接后取最后 5 帧；如果传入的 `history` 已经包含当前帧，窗口会变成 t-3, t-2, t-1, t, t，当前帧出现两次。训练路径 `_split_policy_obs` 与 `split_windows` 一致。

调用方：[N01](../../networks_learning/history_encoder/encoder.py) 调用 `split_windows` 构造 Actor 输入；[T02](../../training_validation/training_loop/runner.py) 将窗口传给 N02 辅助学习；[T05](../../training_validation/export_inference/export.py) 复用同一推理特征路径。

## 必须成立的规则

- 输入历史由环境推进，本模块切片时不再次追加当前帧。
- Actor 窗口为 t-4 到 t；辅助另一窗口为 t-5 到 t-1。
- 新一轮填充规则与环境统一，辅助损失不把不同轮次的状态拼成一段。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：标注 t-4..t 与 t-5..t-1 的切片含义、当前帧已包含的前提、重置填充和有效样本判定。

## 验收案例（已执行，纯数据范围）

全部使用人工数据。原 PPO 的切片写法按 baseline/ppo 快照 `actor_critic.py` 复写，作为独立对照。

| 案例 | 对应测试（[test_history.py](test_history.py)） | 结果 |
|---|---|---|
| 输入帧编号 0..9：当前=9，新窗口=5..9，旧窗口=4..8 | `SplitWindowsTest.test_frames_0_to_9` | 通过 |
| 随机历史上与原 PPO `_split_policy_obs` + 决策窗口拼接逐元素相等 | `SplitWindowsTest.test_matches_ppo_training_split` | 通过 |
| 原 PPO 导出包装器在完整历史上得到 6,7,8,9,9（回归记录，供 T05） | `SplitWindowsTest.test_ppo_export_wrapper_on_full_history_repeats_current` | 通过 |
| 切片不修改、不推进历史，返回视图；形状错误时报错 | `SplitWindowsTest`（2 项） | 通过 |
| 有效长度 1/5/6/10 → 辅助有效 F/F/T/T；不依赖观测是否为零；越界报错 | `ValidityTest`（3 项） | 通过 |
| 编号连续时通过；跨环境、跨轮次、重复 / 跳跃时报出对应行 | `TimeOrderTest`（2 项） | 通过 |
| D02 读出片段的有效前缀编号首尾相接；D03 批次经 D04 归一化后切片，与先切片再归一化相同；辅助有效标记来自 `history_length` / `next_history_length`；原始记录不变 | `ReplayIntegrationTest` | 通过 |

训练内推理与导出一致性的后续检查已位于 [T05 test_export.py](../../training_validation/export_inference/test_export.py)，包括编号帧不重复当前帧的回归。本目录的原始验收仍只代表 D05 切片检查。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/history_windows/test_history.py -v
```

## 故障定位

导出后表现改变先检查是否把当前帧重复拼入；不要只比较张量总长度。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E02 观测接口](../../environment_control/observation_interface/README.md)
- [D01 经历记录契约](../../experience_returns/transition_schema/README.md)：`history_length` / `next_history_length` 与时间编号

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（纯数据范围）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [history.py](history.py)：`split_windows`、`HistoryWindows`、`valid_history_mask`、`validate_time_order`、`HistoryWindowError`、`WINDOW_FRAMES`、`AUXILIARY_REQUIRED_FRAMES`；依赖 D01 `HISTORY_LENGTH`、`POLICY_OBS_DIM`。[test_history.py](test_history.py)：4 个测试类共 11 项，衔接 D02、D03、D04 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`split_windows` 返回视图，其余函数只读 |
| 代码注释核对 | 已核对：索引与时刻的对应（文件头、`split_windows`）；不推进、不追加当前帧（`split_windows`）；按有效长度而非观测数值判断（`valid_history_mask`）；只用显式编号检查时序（`validate_time_order`）；与原 PPO 的关系（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU）；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/history_windows/test_history.py -v` → `Ran 11 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除，不影响结果 |
| 未验证项 | T05 导出入口与训练窗口的一致性；N01 / N02 的实际使用；E05 重置填充规则与 `history_length` 的来源；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`807ee636b1d40ff4be86ed27f9ce9ecd9f5afe2a`（分支 `claude/experience-returns`，基于 `c06509e`）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 D05：统一训练与运行的切片入口，并把原 PPO 导出包装器在完整历史上重复当前帧的现象记录为 T05 回归依据 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；D01 `schema.py`；带注释的 `history.py`、`test_history.py`；原 PPO `actor_critic.py` 的 `_split_policy_obs`、`BarlowTwinsActor.forward`、`_InferenceWrapper`。建议 Codex 重点解释三段切片的索引、辅助学习为什么需要 6 帧，以及原导出包装器重复当前帧的原因 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；面向有 Python 基础的读者，实际核对版本与限制见本目录 README_HUMAN.md |
