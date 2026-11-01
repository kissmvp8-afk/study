# T02 · 采集与学习调度｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色；保留有效正文并同步 PR #8 接入变化 |
| 技术核对状态 | 2026-10-31，Codex 按实际实现与测试同步资产、设备、公共环境接口或集成检查；本次是文档核对，不扩大代码审核 |
| 语言润色状态 | 保留已有中文讲解与历史润色记录；本轮由 Codex 技术同步，不再交 Gemini |
| 已核对代码版本 | 本轮 `a12e69a38c6b9b3a236473a2563e255a9a24e761`（claude/environment-integration，PR #8），fetch 后 git rev-parse HEAD 与已审查/远端版本一致；历史版本见正文与 Git 历史 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 本轮阅读 runner.py/fake_env.py 与已有 test_runner.py 相关断言、E05 lifecycle.py/test_lifecycle.py、E01 启动/填配置及 CPU 衔接测试；runner.py/test_runner.py 未改变；全局材料与本轮检查见[项目阅读记录](../../README_HUMAN.md#阅读材料与验证依据) |
| 验证依据与限制 | 既有 CPU 结果见本目录 AI 交接表；PR #8 记录全仓库 216 项测试（跳过真实资产 1 项）及 18 项 CPU 集成测试通过，本轮未重跑。只执行 verify.py --docs 与文档静态核对；真实 Isaac、GPU、机器人训练及性能未验证 |

## 把一个学习步骤变成可重复的训练流程

“采集”是策略与环境交互产生新经历；“回放”是再次抽取已经保存的经历；“迭代”是 T02 组织的一轮采集和若干次学习。它们的次数不同，不能把日志中的一次迭代误当作一次物理步或一次网络更新。

SAC（软演员—评论家算法）用 Actor 生成动作，用 Q 估计状态与动作的未来回报，并用温度 alpha 调节鼓励多样性的力度。底层公式在 N 模块实现，T02 负责调用顺序、频率和状态归属。与原 PPO 收集一批近期数据后做优势更新不同，SAC 的旧经历保留在经验池，多次抽取；当前 Actor 仍需重新生成用于策略学习和尾部估值的动作。

归一化指按均值和尺度调整输入；历史窗口是连续若干观测帧；梯度是损失对参数的变化率，决定优化方向。T02 只调度这些已有计算，细节分别见 D04、D05 和 N 模块。

当前环境是人工的 `FakeLocomotionEnv`。它帮助核对生命周期和调度，不模拟 TITA 动力学。代码能在 CPU 上计算有限损失，不能据此判断机器人会走。

## 两份观测，两个用途

公共数据类 `EnvObservation/EnvStep` 定义在 [E05 lifecycle.py](../../environment_control/episode_lifecycle/lifecycle.py)，[fake_env.py](fake_env.py) 导入复用；E05 `EpisodeLifecycle` 为 E01 真实钩子实现相同 reset/step 接口。E 是并行环境数，C 是 Q 观测维数；CPU 配置是 E=3、C=6。

| 对象 | 字段与含义 |
|---|---|
| `EnvObservation` | `policy_history[E,10,33]`、`q_observation[E,C]` 是下次交互所见；`history_length/env_id/episode_id/step_index[E]` 为有效帧数及编号。历史为 float32，编号为 int64 |
| `EnvStep` | `next_policy_history/next_q_observation/next_history_length` 是动作后、重置前的真实下一状态；`reward[E,1]`、`terminated/time_out[E,1]` 为原始奖励与互斥结束标记；`observation` 是下次交互所用，结束的环境已重置 |

例如第 3 次动作真正终止，记录仍保存这次动作到达的末历史；下次交互使用新一轮的历史、`step_index=0` 和 `history_length=1`。纯时间截断也这样分开保存，但 D03 保留末状态的未来价值；真终止则不接未来价值。D01 的长度检查只能发现部分错误，不能证明观测一定来自重置前；正确捕获由 E05 的重置前快照和 E01 钩子负责；代码已实现，真实仿真尚待核对。

## `FakeLocomotionEnv` 每个方法做什么

- `__init__(num_envs,q_dim,max_episode_steps,termination_steps)` 分配 10 帧历史、计数和固定混合矩阵。`termination_steps` 每项给该环境第几次动作真终止，`None` 表示不真终止；`received_actions` 留存实际收到的动作副本供测试使用。
- `_reset_frame()` 构造第 e 个环境的人工重置帧 `-3+0.01e`；`_reset_envs(mask)` 原地用它填满历史，长度设 1、步号设 0；它不自行更改轮次号。
- `_q(history)` 取最新帧，拼接 `2*f[:3]` 和后续列为 Q 观测。前 3 列仅模仿原 PPO 的速度尺度，不能称为真实速度测量；当前配置 C=6，不应把该人工构造推广到任意真实 C。
- `_observation()` 返回独立历史、长度和编号副本及计算出的 Q 输入，避免后续环境更新覆盖调用方保存的内容。
- `reset(first_episode_ids=None)` 重置全部环境并返回 `EnvObservation`；缺省轮次号为 0，T03 恢复时传入保存后递增的轮次号。它不恢复旧环境现场。
- `step(action[E,8])` 检查形状、断开梯度并转为 float32，推进人工动力学与历史，计算结束原因；先保存真实下一状态，再重置已结束环境，返回 `EnvStep`。输入为策略动作，无物理单位，没有执行真实 E03 执行器换算。

其动力学是 `f'=0.8f+0.2Wa`，奖励为 `1-0.1*mean(a²)`。第一帧全为 -3、动作全零时，下一帧全为 -2.4，奖励为 1；这只是便于辨认时间顺序的人工数字。

结束处理中的关键顺序：

```python
time_out = (self._step == self.max_episode_steps) & ~terminated  # 真终止优先
next_history, next_length = self._history.clone(), self._length.clone()
```

第一行在真终止的同一步排除到时标记；第二行在后续 `_reset_envs` 原地覆盖前复制末状态。若反过来复制，经验池会把新一轮的起点接到上一轮末尾，截断时的自举估值就错了。自举指用模型预测接上未观察到的未来回报。

## `runner.py`：谁持有状态，谁改动状态

`TrainingCounters` 是可变数据类，12 个计数均从 0 开始：迭代、环境步、经历条数、Critic/Actor/alpha/目标/辅助更新、辅助跳过、回放/辅助样本量、经验不足跳过的学习调用数。T03 保存它们，因为计数还决定预热和更新相位。

`Trainer.__init__(config,env,contract=None)` 校验 T01 配置与环境 E/C，默认从固定 PPO 快照读取 E03 动作表，经 E04 推出边界和初始中心，再构造 D02 经验池、D04 两套统计、编码器/projector/Actor/在线与目标双 Q/温度及四个优化器。所有可配置值显式传入，不依赖 N 模块的默认学习率或 tau。构造会调用 `torch.manual_seed` 和 `env.reset()`，有全局随机种子与环境重置副作用。当前这些模块构造在 CPU；尚未实现完整的 GPU 设备迁移。

三个独立 `torch.Generator` 分别控制采集、回放抽样和学习采样，种子为 `seed+1/+2/+3`。网络初始化使用全局种子。分离随机流后，多做一次采集不会直接消耗回放的随机序列；T03 会保存三条流的状态。

### `collect_steps(num_steps) -> dict`

每步按以下顺序运行，返回本次采集奖励均值、真终止数和纯截断数：

```text
当前原始历史 → D04 只读归一化 → D05 切片 → N01 编码得到 52 维特征
→ N03 随机有界动作 → env.step
→ D01 make_transition（当前 + 重置前 next_*，复制数据）
→ D04 update_from_fresh_data → D02 append
→ self.observation 改为 step.observation → 增加采集计数
```

采集在 `no_grad` 下，且编码器保持 eval，既不建立训练图也不隐式更新 BN（批归一化层的运行统计）。预热同样使用初始标准差 0.15 的有界 Actor，不换成覆盖全范围的均匀随机动作。记录中的 `action` 就是传给环境的策略动作。

D04 每批新数据更新一次，回放不重复统计。3 个环境连续采集 4 步、未重置时，每环境统计初始当前帧 1 次与真实下一帧 4 次，计数为 3×5=15；历史中的重叠旧帧不重复计数。

### `learn_from_replay(num_critic_updates) -> dict`

开始时若没有满足 n 步的合法起点，返回 `{"skipped": True}`，增加 `learning_skipped_iterations`，不会把缺未来数据当作回合结束。每次合法更新先由 D02 抽样和读序列，D03 构造多步回报，N05 计算目标并更新在线双 Q。批量 M=16 时，起点历史为 `[16,10,33]`、Q 观测 `[16,C]`，动作 `[16,8]`。

目标使用尾记录的 `next_*` 和当前 Actor 的新采样：`y=R+γ^k*mask*(min(Q_target)-alpha*logπ)`，真终止 mask=0，纯截断 mask=1；目标不带梯度。随后在起点状态上再采样一组动作，供温度和 Actor 共用。尾部采样与这次起点采样用途不同。

关键调度取自真实代码：

```python
update_index = self.counters.critic_updates  # 本次更新之前已完成的 Critic 更新数（作者 update_step）
```

它在更新前读取，从 0 开始。在线 Q 更新后 `critic_updates` 加 1，但 alpha/Actor 的判定仍用 `update_index`：

```python
if update_index % a.actor_update_every == 0:
    # Actor 损失在 alpha 更新之后组装，因此使用的是新 alpha（作者顺序）。
    loss = compute_actor_loss(self.critics, q_input, action, log_prob, current_alpha(self.temperature))
```

所以间隔 2 对应第 1、3、5…次 Critic 更新后，而非第 2、4、6…次。`action/log_prob` 在此之前只采样一次，N07 先更新 alpha，N06 再用新 alpha 组装 Actor 损失；N07 内部 detach 密度，只改温度，N06 保留动作梯度、冻结 Q 参数，只改 Actor。N01 特征已隔离，不随 Actor 损失训练。

目标 Q 是否慢更新用**更新后的** `critic_updates` 调用 `soft_update_due`，间隔 2 则在第 2、4…次发生。两种计数相位要分清；默认 alpha 与目标更新间隔都为 1。返回日志包含各损失均值、alpha 与本批折扣最大值，不是完整性能报告。

### `auxiliary_update() -> dict`

从 D02 按 n=1 抽辅助批量，取每段首记录；D04 归一化历史、D05 切两窗口，原始 Q 的配置标签列交 N02。辅助损失包含速度监督和 Barlow Twins（要求相邻窗口的同一特征一致、不同特征减少冗余）。N02 筛选有效历史≥6 帧且有效样本至少 2 个，才在 train 模式更新编码器和 projector 的参数/BN，结束恢复 eval。

池为空或有效历史不足时记 `auxiliary_skipped`；成功时记更新次数和实际有效样本数。它按迭代计频，与 Critic 次数无关，防止 Q 大量回放导致编码器也过量更新。

### `train(iterations, checkpoint_path=None, checkpoint_every=0, include_replay=True)`

返回每轮日志列表。先采集 `steps_per_iteration`，预热期结束才执行配置次数的 Critic 学习和辅助学习，然后增加迭代数、检查日志浮点数是否有限，按可选间隔调用 T03 保存检查点。出现非有限日志抛 `FloatingPointError`；有限也不能证明学习效果。

现有测试的调度例子：训练 5 轮、预热 2 轮、每轮 6 步、E=3，每个学习轮 5 次 Critic 和 2 次辅助尝试：

| 计数 | 计算 |
|---|---|
| `env_steps` / `transitions` | 5×6=30 次 `step` / 30×3=90 条记录 |
| `critic_updates` | (5−2)×5=15 |
| `actor_updates` | 第 1、3…15 次，共 8 |
| `alpha_updates` / `target_updates` | 默认各 15 |
| 辅助更新 + 辅助跳过 | (5−2)×2=6 次尝试 |
| `replay_samples` | 15×16=240；不含辅助抽样 |

## `test_runner.py` 与排错入口

测试辅助函数 `make_trainer(...)` 用调试配置构造假环境，可覆盖算法/采集参数；`records_of_env(trainer,env,count)` 在尚未覆盖的槽位中取记录供比对。

9 项既有测试按四类组织：`ConfigurationAppliedTest` 三项核对配置进入模块、梯度裁剪/γ/n/批量和 tau；`ScheduleTest` 三项核对次数、实际执行与记录动作一致且初始化温和、新数据统计不随回放增加；`EpisodeBoundaryTest.test_terminal_time_out_and_priority` 连 D01～D03 检查末状态、重置与终止优先；`OrderAndFallbackTest` 两项核对新 alpha 与同一次采样、经验/历史不足时跳过。详细断言见[测试文件](test_runner.py)，通过记录来自 [AI 交接表](README.md#实现证据与文档交接)，本轮未执行这些算法检查。
对应上文断言的精确方法名如下。测试方法由 unittest 调用，无业务参数，成功时返回 None；测试工厂与设置/清理方法的职责见上文。

| 测试类 | 方法定位（职责与预期见上文） |
|---|---|
| `ConfigurationAppliedTest` | `test_values_reach_modules`、`test_grad_norm_gamma_n_and_batch_size_are_used`、`test_tau_controls_target_following` |
| `ScheduleTest` | `test_counters_over_iterations`、`test_actions_recorded_equal_actions_executed_and_are_mild`、`test_statistics_only_from_fresh_data` |
| `OrderAndFallbackTest` | `test_actor_uses_updated_alpha_and_same_sample`、`test_insufficient_experience_and_history` |

需要复现时，在仓库根目录运行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/training_loop/test_runner.py -v
```

更新次数不对先对 `TrainingCounters` 与相位；一直跳过学习先看 D02 合法起点和 n，辅助跳过再看有效历史和样本数；统计膨胀先检查是否误在回放调用更新；截断价值异常先核对 `next_*` 是否是重置前数据。真实接入代码见 [E02 观测](../../environment_control/observation_interface/README_HUMAN.md) 与 [E05 生命周期](../../environment_control/episode_lifecycle/README_HUMAN.md)，已用人工钩子核对接口，尚未在 Isaac 运行，恢复语义继续读 [T03](../checkpoints/README_HUMAN.md)。

## 同一个 Trainer 怎样接入真实环境

[E01 启动入口](../../environment_control/robot_environment/run_sac_isaac.py) 先核对资产、动作、周期和运行时观测，再填 T01 的完整配置，构造 `EpisodeLifecycle(TitaSacHooks(env), task, q_dim)`，交给 `Trainer(config,lifecycle,contract)`。`runner.py` 的采集与学习算法没有因此改变；不同的是环境对象现在可来自真实钩子。

配置 `device` 固定为 cpu，E01 `--device` 控制仿真设备。E05 将动作搬到仿真设备，观测、奖励和标记搬回 CPU；不是 GPU 学习器，实际传输/同步开销未测量。train 模式还调用 T03 保存、构造新 Trainer、恢复并续一轮；环境重新 reset，不恢复旧物理现场。

这条接口已有 E05 `TrainerIntegrationTest` 两项与 E01 `RunConfigTest.test_trainer_runs_on_filled_config` 的 CPU 人工钩子检查，T06 已纳入集成入口。它们不执行 `TitaSacHooks` 的真实 Isaac 行为，不代表仿真或机器人训练通过。
