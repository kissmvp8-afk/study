# E05 · 终止、截断与重置前观测｜AI 实施规范

状态：代码已实现，CPU 单元验证通过（人工钩子范围）。已实现 [lifecycle.py](lifecycle.py) 与 [test_lifecycle.py](test_lifecycle.py)，10 项检查通过；真实钩子（E01 `TitaSacHooks`）未在 Isaac 中执行，待 Windows + Isaac 验证。未进行训练或性能验证。

[人类版（待 Codex 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

区分摔倒与人为时间结束，并记录动作真正到达的最后状态。

术语先说明：终止是本轮任务真正结束；截断是外部时间限制停止记录；重置是把环境恢复到新一轮初始状态。

## 与原 PPO 的关系

原 PPO 在采集时对超时奖励加价值补偿；SAC 改为保存真实奖励与重置前观测，训练时再估计后续价值。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)
- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块的步骤顺序与末状态捕获已实现并用人工钩子验证，尚未在仿真中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 物理步后的环境状态、真正终止标记、时间截断标记、观测管理器的历史状态。 |
| 输出 | 用于继续交互的重置后观测，以及用于经验池的正确末状态、终止/纯截断标记。 |

## 实际文件与接口

实现与验证都在本目录。[lifecycle.py](lifecycle.py) 导入 E02 的 `PolicyHistory`、`push_frame`、`read_observation_groups`、`valid_length`；依赖 PyTorch，不依赖 Isaac。仿真操作通过“钩子”对象注入：E01 [isaac_env.py](../robot_environment/isaac_env.py) 的 `TitaSacHooks` 用 Isaac 的 manager 实现，测试用人工的 `ScriptedSimulation`。

一步的顺序（`run_isaac_step`，复刻 Isaac Lab 2.3.0 `ManagerBasedRLEnv.step`，只在重置前插入一次观测计算）：

1. `apply_action_and_simulate(action)`：处理动作 → decimation 次物理步 → `episode_length_buf += 1`
2. `compute_terminations() -> (terminated, time_outs)`
3. `compute_rewards() -> [B]`：真实奖励，不含 PPO 的超时价值补偿
4. `compute_observations()`：**重置前观测**，克隆保存
5. `reset_envs(env_ids)`：只在有环境结束时调用
6. `after_reset()`：指令更新与 interval 事件
7. `compute_observations()`：**重置后观测**，下一次交互使用

| 符号 | 签名 / 字段 | 职责、形状、异常与副作用 |
|---|---|---|
| `EnvObservation` | 冻结 dataclass：policy_history [E,10,33]、q_observation [E,C]、history_length / env_id / episode_id / step_index [E] int64 | 下一次交互所用的观测与编号（**公共接口**，原在 T02 fake_env.py，现移到这里，字段与语义不变） |
| `EnvStep` | 冻结 dataclass：next_policy_history、next_q_observation、next_history_length、reward [E,1]、terminated / time_out [E,1] bool、observation | next_* 是重置前真实下一状态；terminated 与 time_out 互斥；observation 中结束的环境已是新一轮（**公共接口**） |
| `LifecycleError` | `RuntimeError` 子类 | 结束的环境缺少重置前观测，或重置的环境与结束标记不一致 |
| `RawStep` | dataclass：pre_reset、post_reset、reward、terminated、truncated、reset_env_ids | `run_isaac_step` 的原始结果（仍在仿真设备上） |
| `classify_endings` | `(terminated, truncated) -> (terminated [B,1], time_out [B,1])` | time_out = 到时且未真终止（真终止优先） |
| `run_isaac_step` | `(hooks, action) -> RawStep` | 按上面 7 步调用钩子；重置前观测逐组 `clone()` |
| `capture_final_observations` | `(history, frame_pre, q_pre) -> (末历史, 末计数, 末 Q)` | 末历史 = 当前历史推入重置前帧；返回新张量，不写回 history |
| `build_step_record` | `(history, ended, frame_pre, q_pre, frame_post, q_post) -> (next_history, next_length, next_q)` | 掩码按字段展开（历史 [B,1,1]、Q [B,1]）；未结束环境推入 frame_post；结束环境 next_* 取末状态，然后历史清零再推入 frame_post（填满、长度 1）；有环境结束却没有重置前观测时报错。副作用：推进 history，每个环境一次 |
| `EpisodeLifecycle` | `(hooks, task, q_dim, output_device="cpu")`；`reset(first_episode_ids=None) -> EnvObservation`、`step(action [E,8]) -> EnvStep` | T02 环境接口的实现。`reset` 调用 `hooks.reset_all()`，编号从 `first_episode_ids`（T03 恢复用）或 0 开始；`step` 把动作搬到 `hooks.device`、把结果搬到 `output_device`，核对被重置的环境 = 结束的环境，结束的环境 episode_id 加一、step_index 归零。属性 num_envs、q_dim 供 T02 Trainer 检查 |

钩子协议：属性 `num_envs`、`device`，方法 `reset_all() -> {组名: 张量}` 与上面 6 个方法。E01 `test_environment.py` 用 AST 检查 `TitaSacHooks` 覆盖了本文件调用的全部钩子。

数据来源：未结束环境的 next_* 与 observation 都来自第 7 步，第 4 步的结果对它们不使用；结束环境的 next_* 来自第 4 步，observation 来自第 7 步。两份数据各采样一次观测噪声。

设备：钩子在仿真设备上（例如 cuda:0），学习器在 CPU 上。每步有多次张量搬运：动作 CPU→仿真设备 1 次；仿真设备→CPU 按张量逐个搬，重置前、重置后两份观测各按观测组搬（平地 policy / critic / priv 共 3 个，复杂地形再加 scanner 共 4 个），另加 terminated、time_outs、被重置环境编号与奖励各 1 个，即平地 10 次、复杂地形 12 次；run_isaac_step 中的 nonzero 在 GPU 上还会引起一次同步。实际耗时未测量。

## 必须成立的规则

- 同一步摔倒又到时，按真正终止处理。
- 只在正确时间推进一次历史；不得因额外 compute 调用导致历史重复当前帧。
- 快照不能被重置时的原地修改覆盖；有截断却缺最后观测时必须报错。

## 算法约定与实现提示

一次正常流程是：执行动作 → 计算奖励和结束原因 → 保存本步真实末状态 → 对结束的环境重置 → 返回下一次交互所需观测。末状态必须同时包含 Actor 的历史和 Q 的训练侧信息。

Isaac Lab 的自动重置可能发生在标准 `step` 返回之前，因此只在外层读取返回观测未必能取到末状态。实现时检查锁定版本的执行顺序，在重置前保存数据，并避免额外观测计算重复推进历史或重采噪声。逐字段验证保存内容对应同一次物理步。

作者 `process_env_step` 中替换末状态的二维掩码不能直接用于 TITA 的 [B,10,33] 历史；掩码需按字段维度正确展开。先用明显不同的末状态与重置状态检查，再接仿真。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释重置前快照时机、真终止优先级、历史只推进一次、按维度展开掩码与复制内存的原因。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例

### CPU 已执行（[test_lifecycle.py](test_lifecycle.py)，人工钩子，不是仿真验证）

`ScriptedSimulation` 用标量状态 s 编码每个环境的观测（策略帧全为 s，Q 前 3 列为 2s），重置值约为 −100，与运动中的状态明显不同，并记录调用顺序。

| 案例 | 对应测试 | 结果 |
|---|---|---|
| 调用顺序为 simulate → terminations → rewards → observe → reset → after_reset → observe；无环境结束时不调用 reset | `OrderTest.test_step_order` | 通过 |
| 正常、单独终止、单独到时、同时发生：同时发生只记真终止 | `EndingTest.test_classification`、`EndingTest.test_simultaneous_terminal_and_time_out_in_steps` | 通过 |
| 强制一个环境终止：next_* 是重置前末状态（−96.5，不是重置值 −101），Q 速度列同为末状态，next 有效长度 4；observation 为新一轮（填满、长度 1、episode 加一、step 0）；其他环境的历史每步一帧、未受影响 | `FinalStateTest.test_final_state_and_partial_reset` | 通过 |
| 保存的末状态在之后 3 步后不变（快照不被覆盖） | `FinalStateTest.test_snapshot_not_overwritten_by_later_steps` | 通过 |
| 有环境结束却缺重置前观测 → 报错 | `FinalStateTest.test_missing_final_observation_is_an_error` | 通过 |
| 动作原样交给钩子（换算只在 Isaac 动作项做一次） | `FinalStateTest.test_actions_passed_unchanged` | 通过 |
| `first_episode_ids` 用于恢复后继续编号 | `FinalStateTest.test_first_episode_ids_for_resume` | 通过 |
| 接现有 T02 Trainer：经验池中终止记录的 next_* 为末状态、下一条为新一轮、记录的动作 = 执行的动作；D03 在终止处截断（步数 3、bootstrap 0）；能学习 | `TrainerIntegrationTest.test_collect_learn_and_true_final_state_in_buffer` | 通过 |
| T03 保存 / 恢复：exact_replay，恢复后 episode_id 加一、step_index 归零 | `TrainerIntegrationTest.test_checkpoint_restore_with_adapter` | 通过 |

### Windows + Isaac 待验证

- E01 `interact --steps 1005`：形状与数值有限；若有摔倒，next_* 与 observation 不同。1005 步**不保证**出现到时（环境可能先摔倒并重置，新一轮的计数从 0 开始）；是否覆盖了到时路径，以 `summary.json` 中 `interact.time_out` 计数是否大于 0 为准，`interact.terminated` 同理。
- E01 `train`：真实钩子下采集、学习、保存与恢复跑通。

运行命令（仓库根目录，无需 Isaac Lab）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/episode_lifecycle/test_lifecycle.py -v
```

## 故障定位

每到时间上限价值突变时，检查结束原因、最后观测来源和是否重复添加价值补偿。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E01 机器人与任务环境](../../environment_control/robot_environment/README.md)：`TitaSacHooks` 实现钩子协议
- [E02 观测接口](../../environment_control/observation_interface/README.md)：`PolicyHistory`、`push_frame`、`read_observation_groups`、`valid_length`
- [D01 经历记录契约](../../experience_returns/transition_schema/README.md)
- [D05 历史窗口与有效性](../../experience_returns/history_windows/README.md)
- [T02 训练调度](../../training_validation/training_loop/README.md)：`EnvObservation` / `EnvStep` 是 T02 Trainer、T04 评估与 T03 恢复共用的环境接口；`fake_env.py` 现从本模块导入这两个类

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 代码已实现；CPU 单元验证通过（人工钩子范围）；Isaac 钩子未执行 |
| 实际文件、函数签名与调用关系 | [lifecycle.py](lifecycle.py)（依赖 E02 `observations.py`）；[test_lifecycle.py](test_lifecycle.py)：4 个测试类共 10 项。调用方：E01 `run_sac_isaac.py`；T02 `fake_env.py` 从这里导入 `EnvObservation` / `EnvStep` |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`step` 推进钩子的仿真状态与本对象的历史和编号，返回的张量都是新张量 |
| 代码注释核对 | 已核对本模块注释重点：重置前快照时机与锁定版本顺序（文件头、`run_isaac_step`）；真终止优先（`classify_endings`）；历史只推进一次与数据来源（文件头、`build_step_record`）；按维度展开掩码（`build_step_record`）；复制内存的原因（`run_isaac_step` 的 clone、`capture_final_observations`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），Python 3.11.17：`uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/episode_lifecycle/test_lifecycle.py -v` → `Ran 10 tests … OK`；同时回归 T02 9 项、T03 10 项、T04 12 项、T05 6 项均通过；torch 2.7.0（CPU）下全部通过 |
| 未验证项 | 真实 Isaac 钩子的顺序与数值、重置后观测、GPU→CPU 搬运；仿真、训练与性能 |
| 代码版本 | `3d4c995b2730b3183614f4cb5b7a62edfa83f927`（分支 `claude/environment-integration`）；`794cd623de54c75ee3ed96ae6788fb72c5a5cebd` 只修改了 `lifecycle.py` 文件头的设备搬运说明（不改行为），`test_lifecycle.py` 未改动 |
| 本次改动及理由 | 首次实现 E05：锁定版本在 `step` 返回前就重置，因此拆成钩子并在重置前插入一次观测计算；把公共环境接口 `EnvObservation` / `EnvStep` 从 T02 fake_env.py 移到这里，供假环境与真实环境共用 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；E01、E02、T02 AI 规范；带注释的 `lifecycle.py`、`test_lifecycle.py`；上述验证记录；Isaac Lab v2.3.0 `manager_based_rl_env.py` 的 `step`。建议 Codex 重点解释：7 步顺序与原 `step` 的对应；`build_step_record` 中结束 / 未结束环境的数据来源与掩码展开；为何每个环境每步只推进一次历史；`EpisodeLifecycle.step` 的重置一致性检查与编号更新 |
| 人类版 | 待 Codex 编写（代码、注释与 CPU 验证记录已齐备） |
| 技术初稿 / 语言润色 | 待 Codex 技术初稿；其后由 Gemini 润色同一份人类版 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
