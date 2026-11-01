# E05 · 终止、截断与重置前观测｜人类阅读版

[AI 实施规范](README.md) · [环境大模块人类版](../README_HUMAN.md) · [项目人类版](../../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 对照实际实现、注释与测试核对职责、调用、形状、单位和关键片段；面向有 Python 基础的读者，本次不扩大为全面代码审核 |
| 语言润色状态 | 本轮由 Codex 完成讲解与技术核对，不再交 Gemini |
| 已核对代码版本 | 原实现 main `f6c8b03d`；Windows 路径修复 `373bb48d`；新增诊断及本轮证据随本次提交保存，未改算法 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | AGENTS.md、CODEX.md、根与环境大模块/本目录 AI README；[lifecycle.py](lifecycle.py)、[test_lifecycle.py](test_lifecycle.py) 全文；E01 TitaSacHooks、E02 历史、T02 采集、T03 恢复及 D01/D03 末状态约定；上游片段与依赖见[阅读记录](../README_HUMAN.md#阅读材料与验证依据) |
| 验证依据与限制 | 2026-10-31 Windows 真实平地 1 环境的加载、50/1005 步交互、短训练保存恢复及提前计数纯截断通过；历史 CPU 记录保留，本轮不重跑全仓。截断仅实际推进 0.04 s，不证明稳定运行 20 s；复杂地形、长期训练、性能与稳定行走未验证。证据见下方 Windows 验证记录 |

## Windows 验证记录（2026-10-31）

Windows 平地单环境最小闭环已验证；提前设置计数器的真实纯时间截断检查通过。复杂地形、长期训练、性能与稳定行走仍未验证。 [精简报告、命令与 JSON 证据](../robot_environment/validation/windows_2026-10-31/REPORT.md)。

真实截断诊断在正常 reset 后仅把 `episode_length_buf` 从 0 提前设为 998，保留正式 20 s / 1000 步上限和 `base_contact` 真终止项。随后 2 个策略步（8 个物理步、0.04 s）由原管理器给出 `terminated=False,time_out=True`，首次尝试通过。

E05 的 `next_*` 与同一次计算捕获的重置前末状态相等；`observation` 对应新回合，历史有效长度 1、步号 0、回合编号 0→1。真实记录经过 D01/D02/D03，mask=1、尾部仍属旧回合、排除新回合奖励；后续真实交互 3 步不覆盖已保存末状态。没有手工伪造结束标记，也没有关闭真实终止条件。

此前自然交互 1005 步有 40 次真终止、0 次超时；它没有覆盖超时，本次覆盖来自明确标注的提前计数诊断。短训练为 CPU 学习器 / GPU 仿真，3×8 步保存后恢复续训 1 轮；检查点不保存仿真现场，且续训后未再次保存模型。轮速显式 20 rad/s 仍为“首次仿真调试选择，尚未验证”。

启动使用 `--kit_args="--vulkan --/app/extensions/fsWatcherEnabled=false"`，诊断实际读取到热重载开关为 False。仍有图形接口/Aftermath 告警；未宣称消除所有图形兼容风险。T06 本身未执行这些仿真，因此其 `simulation` / `performance` 仍返回 `not_run`。

## 一次动作之后，环境可能需要返回两份状态

一轮交互叫 episode。真正终止表示任务已经结束，例如机身触地；时间截断表示达到人为时长上限，机器人并非因此失去未来。重置把环境放回下一轮初始状态。SAC（软演员—评论家算法）回放旧经历时，需要知道旧动作真正到达了哪里，不能把下一轮起点当成旧动作的结果。

E05 因此提供两份输出：`next_*` 用于经验池和末状态估值；`observation` 用于下一次选择动作。正常步两者相同，结束步通常不同。自举（bootstrap）指用价值预测补上未观察到的未来：真终止 mask=0，正常与纯时间截断 mask=1。时间截断仍在旧回合末状态估值，但不能继续累计新回合奖励。

原 PPO 会在采集路径对超时奖励加旧价值补偿；这里保留真实环境奖励，由 D03/N05 在回放时重新构造目标。E05 不调用 Q，也不把价值预加进 reward。

## `lifecycle.py` 中的数据对象

B/E 均表示并行环境数，C 是已核对的 Q 状态信息维数。浮点数据供 CPU 学习器使用，历史与奖励为 float32，编号与长度为 int64。

| 类 | 字段、职责与调用方 |
|---|---|
| `EnvObservation` | `policy_history[E,10,33]`、`q_observation[E,C]`、`history_length/env_id/episode_id/step_index[E]`；下一次交互的观测与编号 |
| `EnvStep` | `next_policy_history[E,10,33]`、`next_q_observation[E,C]`、`next_history_length[E]`、`reward[E,1]`、互斥 bool `terminated/time_out[E,1]`、下次交互的 `observation` |
| `RawStep` | 仿真设备上的 `pre_reset/post_reset` 观测组字典、`reward/terminated/truncated[B]`、`reset_env_ids[K]`；内部调度结果，尚未整理成学习器输入 |
| `LifecycleError` | `RuntimeError` 子类；缺末观测或重置对象与结束标记不一致时报告 |

前三个是冻结数据类，冻结只限制属性重新绑定，不让张量自动只读。`EnvObservation/EnvStep` 原定义在 T02 `fake_env.py`，现在统一放在此处，假环境导入复用；T02、T03、T04 使用的字段和语义未改变。

## 为什么拆开环境 step

锁定的 Isaac Lab 2.3.0 在 `step` 返回前已重置结束环境；简单包装返回观测无法取得旧回合末状态。E05 的 `run_isaac_step(hooks,action)` 按以下顺序调用“钩子”——钩子是由外部对象提供的一组具体操作：

1. `apply_action_and_simulate`：处理策略动作、推进物理步并更新步数。
2. `compute_terminations`：取得真终止与原始到时标记。
3. `compute_rewards`：取得未加价值补偿的奖励。
4. `compute_observations`：重置前计算观测并逐组 clone。
5. `reset_envs`：仅重置已经结束的环境。
6. `after_reset`：更新指令、执行 interval 事件（定时推扰等）。
7. 再次 `compute_observations`：取得下次交互观测。

E01 的 `TitaSacHooks` 提供真实操作；CPU 测试的 `ScriptedSimulation` 提供人工操作。顺序依据是[上游锁定版本的 step](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)，本轮阅读源码没有运行该环境。

末状态复制的关键行是：

```python
pre_reset = {name: value.clone() for name, value in hooks.compute_observations().items()}
```

每个组都复制，因为后续 reset 或观测管理器可能复用底层存储。只给字典换个名字不会复制张量，旧末状态可能被覆盖。`RawStep` 返回这些快照与第二次观测，后续再按环境选择数据来源。

## 结束标记与历史选择

`classify_endings(terminated,truncated)` 将标记整理为 `[B,1]`，真终止优先：

```python
terminated = terminated.reshape(-1, 1).bool()
return terminated, truncated.reshape(-1, 1).bool() & ~terminated
```

第一行统一形状/类型；第二行只保留“到时且未真终止”的 `time_out`。四种组合正常、真终止、纯到时、二者同时发生，对应 `(False,False)`、`(True,False)`、`(False,True)`、`(True,False)`。不能把 `terminated|time_out` 直接当作未来价值开关。

`capture_final_observations(history,frame_pre,q_pre)` 用 E02 `push_frame` 将重置前帧追加到旧历史，返回末历史、末推送计数、Q 副本，不写回历史管理器。`build_step_record(...)` 则组装 next_* 并推进持久历史：

| 环境是否结束 | 存入经历的 next_* | 下次交互历史 |
|---|---|---|
| 未结束 | 第 7 步的帧与 Q，旧历史推进一次 | 同一份按时间推进后的内容 |
| 已结束 | 第 4 步的帧与 Q，末历史包含旧回合最后一帧 | 历史清零后用第 7 步新回合首帧填满，有效长度 1 |

第 7 步是在指令/interval 更新后，正常步应使用它，使记录的下一状态与下次实际决策输入一致。对未结束环境，第 4 步的候选不进入经历；但两次观测计算仍会执行，不能据此声称观测噪声随机序列与原 PPO 完全相同。本次只核对同一次观测计算的实际捕获值，未核对与原 PPO 的噪声随机序列一致性。

按环境选择历史和 Q 的关键片段：

```python
next_history = torch.where(ended.reshape(-1, 1, 1), final_history, post_history)
next_pushes = torch.where(ended, final_pushes, post_pushes)
next_q = torch.where(ended.reshape(-1, 1), final_q, q_post)
```

历史有帧和通道两轴，掩码扩为 `[B,1,1]`；计数是一维，用 `[B]`；Q 有特征轴，用 `[B,1]`。若照搬二维观测的掩码到三维历史，可能广播失败或选择错行。此后仅未结束行 commit 正常历史，结束行 reset 后 commit 首帧填充，返回的 next 长度由旧回合计数计算。缺少重置前观测且有结束环境时，明确抛 `LifecycleError`。

## 用测试里的 −96.5 追踪一次真终止

人工环境 1 从 −101 开始，每步状态增加 `mean(action)+1`。动作八维全为 0.5，每步增加 1.5；第 3 次动作真终止：末帧为 `−101+3×1.5=−96.5`，Q 前三列为 −193，末历史有效长度 4（初始帧+3 步）。返回的 `next_*` 必须保留这些数。

同一次返回的 `observation` 已是新回合：历史全为 −101，长度 1、episode_id 加一、step_index=0。其他未结束环境的历史仍按每步一帧推进，不受这次局部重置影响。

| 用于 D03/N05 的情况 | 奖励停止在哪里 | mask 与后续估值 |
|---|---|---|
| 正常步，已凑齐 n 条 | 第 n 次动作后 | 1，接第 n 条 next_* 的预测 |
| 第 3 步真终止，n=5 | 第 3 步奖励 | 0，只有三步折扣奖励 |
| 第 3 步纯截断，n=5 | 第 3 步奖励 | 1，接重置前末状态预测，不接新回合起点 |

mask 的推导在 D01，累计在 D03，估值在 N05；本模块确保它们接到正确状态。D01 的有效历史长度检查只能发现部分不一致，无法仅凭长度证明观测确实来自 reset 前。

## `EpisodeLifecycle` 怎样接 T02/T03

- `__init__(hooks,task,q_dim,output_device='cpu')` 保存钩子/任务/C，构造 E02 历史和编号，暴露 `num_envs/q_dim` 给 Trainer 校验；构造本身不执行 reset。
- `_read(observations)` 将每个组搬到输出设备，再调用 E02 拼帧/Q；`_observation(q_post)` 克隆交互历史、Q 和编号，生成 `EnvObservation`。
- `reset(first_episode_ids=None)` 调 `hooks.reset_all()`，清历史并推入首帧；步号为 0，轮次号取传入值或 0，返回交互观测。有环境与内部状态修改。
- `step(action[E,8])` 断开采集动作梯度、搬到仿真设备，调 `run_isaac_step`，整理结束标记并核对 `reset_env_ids` 与结束集合一致；读取两份观测，调用 `build_step_record`；步号先加一，结束行轮次加一、步号归零，返回 `EnvStep`。

T02 先保留动作前的 `EnvObservation`，用它的旧编号建立 D01 记录；`step.observation` 的新编号只用于下一次交互。T03 恢复传入保存轮次号+1，例如 `[5,2,9]`，E05 从这些新轮次的第 0 步开始，不恢复旧环境现场，D02 因轮次不同不会接错旧记录。

当前学习器只支持 CPU；仿真可配置为 `cuda:0`，这不等于 GPU 学习器。每步动作 CPU→仿真设备；前后两次观测逐组搬回 CPU，再搬两个标记、重置编号和奖励：按当前调用计，平地有 10 次回传、复杂地形 12 次，另有 GPU `nonzero` 同步点。不能简写成“每步一次搬运”，实际耗时未测量。

## `test_lifecycle.py` 证明什么

`ScriptedSimulation` 的 `__init__` 保存人工状态和计划；`_obs` 构造 policy/critic/priv；`reset_all` 全部复位；`apply_action_and_simulate` 记录动作并推进标量；`compute_terminations/compute_rewards/compute_observations` 返回人工结果并记调用顺序；`reset_envs` 局部复位；`after_reset` 仅记事件。这不是机器人动力学。`make_lifecycle(**kwargs)` 组合此对象与平地 C=54 的适配器。

10 项测试的定位如下，方法均由 unittest 调用：

| 类 / 方法 | 验证范围 |
|---|---|
| `OrderTest.test_step_order` | 七步顺序；无结束时不调 reset |
| `EndingTest.test_classification`、`test_simultaneous_terminal_and_time_out_in_steps` | 四种标记组合、同一步真终止优先 |
| `FinalStateTest.test_final_state_and_partial_reset` | −96.5 末帧、Q 和长度、重置后新回合、其他环境不重复推进 |
| `test_snapshot_not_overwritten_by_later_steps` | 后续三步不覆盖已返回末历史 |
| `test_missing_final_observation_is_an_error` | 缺末观测拒绝 |
| `test_actions_passed_unchanged` | 策略动作不提前做物理换算 |
| `test_first_episode_ids_for_resume` | 恢复编号与首帧长度 |
| `TrainerIntegrationTest.test_collect_learn_and_true_final_state_in_buffer` | 接 T02/D02/D03，末状态正确、记录动作等于执行动作、k=3/mask=0，并能调用学习 |
| `test_checkpoint_restore_with_adapter` | T03 保存恢复后新轮次、步号归零 |

`TrainerIntegrationTest.make_trainer()` 用 CPU 调试配置覆盖 C=54，构造人工钩子训练器，不启动 Isaac。既有通过记录见 [AI 交接表](README.md#实现证据与文档交接)，本轮未运行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/episode_lifecycle/test_lifecycle.py -v
```

末状态跳到重置值先查 `pre_reset` clone 与选择掩码；历史重复查 E01 是否关闭 Isaac 历史、E02 是否只 commit 一次；恢复跨回合拼接查编号；截断价值突变查 mask、末状态和是否误加 PPO 补偿。真实钩子行为仍需 E01 的 `inspect/interact/train` 在 Windows + Isaac 中核对；`--steps 1005` 也不保证出现到时，因为可能提前摔倒重置，须看实际 `time_out` 计数。
