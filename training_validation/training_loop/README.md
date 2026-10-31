# T02 · 采集与学习调度｜AI 实施规范

状态：单元验证通过（CPU、**假环境**）。已实现 [fake_env.py](fake_env.py)、[runner.py](runner.py) 与 [test_runner.py](test_runner.py)，9 项检查通过；未接入 Isaac 仿真，结果不代表机器人训练成功。

[人类版（讲解与技术复核已完成，本轮不再润色）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

把已验证的小模块串起来，明确每一批经历怎样产生以及何时更新各网络。

术语先说明：调度：决定各模块在什么时候运行以及运行多少次。假环境：用于验证数据流程的可控简单环境，不代表真实仿真效果。

## 与原 PPO 的关系

原 PPO 是采集近期批次后训练并清空；新流程持续积累经验，分别更新 Q、Actor、alpha、目标 Q 与辅助编码器。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/runner.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/runner.py)
- [原 PPO：scripts/np3o/train.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/train.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已在 CPU 假环境中实现，尚未接入 Isaac 仿真。

## 作者实现参考

- [作者 SAC：`rsl_rl/runners/off_policy_runner.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/runners/off_policy_runner.py)
- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 已验证模块、完整配置、环境状态与可恢复的训练计数。 |
| 输出 | 新经历、更新后的模型、分项训练指标和进度记录。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可）；仓库尚无统一依赖文件。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。T02、T03 的测试需要本地有 baseline/ppo 快照（E03 从快照读取动作表）。

### [fake_env.py](fake_env.py)：假环境（不是 Isaac 仿真）

| 符号 | 签名 | 说明 |
|---|---|---|
| `EnvObservation` | policy_history[E,10,33]、q_observation[E,C]、history_length、env_id、episode_id、step_index | 下一次交互所用的观测与编号 |
| `EnvStep` | next_policy_history、next_q_observation、next_history_length、reward、terminated、time_out、observation | 重置前的真实下一观测与结束原因，以及重置后的续用观测 |
| `FakeLocomotionEnv` | `(num_envs, q_dim, max_episode_steps, termination_steps)`；`reset(first_episode_ids=None)`、`step(action[E,8])`；`received_actions` | 确定性人工动力学：f' = 0.8 f + 0.2 W a，奖励 1 − 0.1·mean(a²)，Q 观测前 3 列 = 2 × f[0:3]；同一步终止又到时只报真终止；重置帧为 −3 附近 |

这套 `reset` / `step` 接口就是 Trainer 与环境之间的约定。E01 / E05 接入 Isaac Lab 时实现同一接口：重置前保存真实末状态，维护 episode_id、step_index、history_length 和重置填充规则。Trainer 不重复实现这些逻辑。

### [runner.py](runner.py)

| 符号 | 签名 | 职责与副作用 |
|---|---|---|
| `TrainingCounters` | iterations、env_steps、transitions、critic_updates、actor_updates、alpha_updates、target_updates、auxiliary_updates、auxiliary_skipped、replay_samples、auxiliary_samples、learning_skipped_iterations | 分项计数 |
| `Trainer` | `(config: RunConfig, env, contract=None)` | 校验配置；`torch.manual_seed(seed)` 后构造 E04 边界与 u0、D04、D02、N01、N02 projector、N03、N04 及目标、N07、四个优化器（全部显式传入配置值）；三个独立 `torch.Generator`（collect / replay / learn）；调用 `env.reset()` |
| `Trainer.collect_steps` | `(num_steps) -> dict` | Actor 随机采样（预热期同样使用有界、温和初始化的 Actor）→ `env.step` → D01 `make_transition` → D04 `update_from_fresh_data`（只用这批新数据）→ D02 `append` |
| `Trainer.learn_from_replay` | `(num_critic_updates) -> dict` | 按作者顺序：D02 抽样 → D03 → N05 → 当前状态采样一次 → N07（每 `alpha_update_every` 次）→ N06 用新 alpha 与同一次采样（每 `actor_update_every` 次，默认 2，第 1、3、5… 次）→ N08（每 `target_update_every` 次）。没有 n 步有效起点时跳过并计数 |
| `Trainer.auxiliary_update` | `() -> dict` | 从经验池抽取单步记录 → D04 归一化 → D05 切片 → N02；速度标签取配置的列；有效历史不足时由 N02 跳过并计数 |
| `Trainer.train` | `(iterations, checkpoint_path=None, checkpoint_every=0, include_replay=True) -> list[dict]` | 每轮：采集 `steps_per_iteration` 步；预热结束后 Critic 更新 `critic_updates_per_iteration` 次、辅助学习 `auxiliary_updates_per_iteration` 次（按迭代计，与 Critic 次数无关）；可按间隔调用 T03 保存；日志出现非有限数值时报错 |

计数的含义：环境步是 `env.step` 的调用次数；transition 数 = 环境步 × 环境数；回放样本数 = Critic 更新数 × batch_size；辅助样本数是 N02 实际使用的有效样本数。

## 必须成立的规则

- 经验不足时先用有界 Actor 收集，不用全范围均匀乱采代替论文初始化。
- 辅助学习频率独立设置，不能因 Q 大量回放就无意重复训练编码器。
- 统计更新、梯度边界与目标更新顺序写清楚；调度层不复制底层公式。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释采集/回放计数、预热动作来源、各优化器顺序与频率，以及模式、统计和重置数据的交接。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，CPU 假环境）

全部在 `FakeLocomotionEnv` 上运行；配置为 T01 `cpu_debug_config`（人工限位）。

| 案例 | 对应测试（[test_runner.py](test_runner.py)） | 结果 |
|---|---|---|
| 配置实际生效：四个优化器学习率、初始 std、alpha、D04 eps、目标熵系数、经验池形状、三个网络的首层宽度、E04 中心进入 Actor 缓冲 | `ConfigurationAppliedTest.test_values_reach_modules` | 通过 |
| 拦截 `clip_grad_norm_`，所有调用的上限都是配置的 0.37；无结束片段的 discount 为 0.9³；回放样本数 = 4 × 7 | `ConfigurationAppliedTest.test_grad_norm_gamma_n_and_batch_size_are_used` | 通过 |
| tau=1 时目标 Q 等于在线 Q；tau=0 时目标保持不变 | `ConfigurationAppliedTest.test_tau_controls_target_following` | 通过 |
| 5 轮（预热 2 轮）：环境步 30、transition 90、Critic 15、Actor 8（ceil(15/2)）、alpha 15、慢更新 15、辅助 6（按迭代计）、回放样本 240；预热轮没有学习 | `ScheduleTest.test_counters_over_iterations` | 通过 |
| 经验池中记录的动作逐条等于假环境实际收到的动作；动作在 [L, U] 内；tanh(u) 的标准差 < 0.25，远小于全范围均匀随机的 0.577 | `ScheduleTest.test_actions_recorded_equal_actions_executed_and_are_mild` | 通过 |
| D04 计数 = 3 × (1 + 4)；回放学习与辅助学习之后计数不变 | `ScheduleTest.test_statistics_only_from_fresh_data` | 通过 |
| 真终止、纯截断，以及同一步二者同时发生（只记真终止）；末状态是重置前的真实帧，下一条记录为新一轮（step 0、长度 1、重置帧）；D03 在第 3 步截断 k=3：真终止 mask=0，纯截断 mask=1，尾部为重置前状态 | `EpisodeBoundaryTest` | 通过 |
| 拦截 N07 与 N06 的调用：顺序为 alpha、actor、alpha、alpha、actor、alpha；Actor 收到的 alpha 等于刚更新后的值（≠ 更新前），log π 与 alpha 更新是同一个张量对象 | `OrderAndFallbackTest.test_actor_uses_updated_alpha_and_same_sample` | 通过 |
| 只有 2 步、n=5 时学习被跳过并计数；有效历史不足 6 帧时辅助学习跳过，编码器不变 | `OrderAndFallbackTest.test_insufficient_experience_and_history` | 通过 |

调试记录：首次运行时 1 项失败，原因是测试把 float32 的 Actor 缓冲与 float64 的 E04 数值逐位比较；改为按 float32 精度比较后通过，实现无改动。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/training_loop/test_runner.py -v
```

## 故障定位

损失有变化而机器人不改善时，先查动作是否实际执行、数据是否最新和奖励分项。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [T01 版本、依赖与训练配置](../../training_validation/configuration/README.md)
- [E05 终止、截断与重置前观测](../../environment_control/episode_lifecycle/README.md)
- [D02 经验池写入与抽样](../../experience_returns/replay_buffer/README.md)
- [D03 连续多步回报](../../experience_returns/n_step_returns/README.md)
- [N02 速度监督与 Barlow Twins](../../networks_learning/auxiliary_learning/README.md)
- [N05 Q 的训练目标与更新](../../networks_learning/critic_learning/README.md)
- [N06 Actor 的参数更新](../../networks_learning/actor_learning/README.md)
- [N07 探索力度 alpha](../../networks_learning/entropy_temperature/README.md)
- [N08 目标 Q 的慢更新](../../networks_learning/target_updates/README.md)
- [T03 训练进度保存与恢复](../../training_validation/checkpoints/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、假环境、人工数据）；Isaac 仿真集成未开始 |
| 实际文件、函数签名与调用关系 | [fake_env.py](fake_env.py)：`FakeLocomotionEnv`、`EnvObservation`、`EnvStep`；[runner.py](runner.py)：`Trainer`、`TrainingCounters`；调用 E03 `load_baseline_action_contract`、E04 `derive_bounds` / `calibrate_initial_center`、D01 `make_transition`、D02、D03 `build_n_step_batch`、D04、D05 `split_windows`、N01～N08 的公开接口，T01 `validate_configuration`，以及按需调用 T03 `save_checkpoint`。[test_runner.py](test_runner.py)：5 个测试类共 9 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`collect_steps` 修改经验池、D04 统计、环境状态与 collect 生成器；`learn_from_replay` 修改在线 / 目标 Q、Actor、log_alpha、对应优化器与 replay / learn 生成器；`auxiliary_update` 修改编码器、projector（含 BN 统计）与辅助优化器 |
| 代码注释核对 | 已核对：采集 / 回放计数（文件头、`TrainingCounters`）；预热动作来源（`collect_steps`）；各优化器顺序与频率（文件头、`learn_from_replay`）；统计只随新数据更新、编码器模式由 N02 管理、重置数据由环境提供（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/training_loop/test_runner.py -v` → `Ran 9 tests … OK`（首次运行有 1 项因测试精度写法失败，见上文“调试记录”）。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | Isaac Lab 环境（E01 / E02 / E05）接入、自动重置时机与末状态捕获；小规模 Isaac 仿真持续训练与保存；真实奖励与观测尺度下的稳定性；GPU 性能；训练效果 |
| 代码版本 | `4bf81a5486a8895c8d6a04305b6b7a1702351c9e`（分支 `claude/training-core`，基于 PR #6 的已复核版本 `42c07a0`，并已合并 main `e43ee42`）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 T02：调度层只调用已有模块；按作者顺序组织 Critic、alpha、Actor、慢更新；辅助学习按迭代计频；用明确标注的假环境检查数据流程 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `fake_env.py`、`runner.py`、`test_runner.py`；作者 `off_policy_runner.py`（learn 循环、start_training）与 `sac.py`（update 顺序、update_step、policy_frequency）。建议 Codex 重点解释一次 Critic 更新内部的顺序、各计数的含义、预热期为何仍用 Actor，以及假环境与真实 E05 的边界 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；核对版本 a07d7b2，材料、证据及限制见本目录 README_HUMAN.md |
| 技术初稿 / 语言润色 | Codex 已完成中文讲解与技术核对；按本轮用户要求不再交 Gemini |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
