# E02 · 观测接口｜人类阅读版

[AI 实施规范](README.md) · [环境大模块人类版](../README_HUMAN.md) · [项目人类版](../../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 对照实际实现、注释与测试核对职责、调用、形状、单位和关键片段；面向有 Python 基础的读者，本次不扩大为全面代码审核 |
| 语言润色状态 | 本轮由 Codex 完成讲解与技术核对，不再交 Gemini |
| 已核对代码版本 | `a12e69a38c6b9b3a236473a2563e255a9a24e761`（PR #8，`claude/environment-integration`）；开始时 fetch 后用 `git rev-parse HEAD` 记录，与远端及已审查版本一致 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | AGENTS.md、CODEX.md、根与环境大模块/本目录 AI README；[observations.py](observations.py)、[test_observations.py](test_observations.py) 全文；E01 运行时校验、E05 生命周期、D01/D04/D05 与 N01 的输入约定；上游片段与依赖见[阅读记录](../README_HUMAN.md#阅读材料与验证依据) |
| 验证依据与限制 | 本目录 AI 交接表记录 8 项 CPU 测试通过（Python 3.11.17，torch 2.15.0 与 2.7.0 CPU）。本轮只阅读并引用既有结果，未重跑算法测试；本轮文档检查见根目录记录。真实 Isaac、GPU 搬运、机器人训练效果与性能均未验证 |

## 为什么 Actor 和 Q 不能看到同一份输入

Actor 是生成动作的策略网络，部署时只能使用可获得的观测；Q 网络估计给定状态和动作的未来回报，训练时可以利用仿真额外提供的信息。这个信息差异常称为非对称观测。E02 保留原 PPO 的策略侧字段和训练侧字段划分，SAC（软演员—评论家算法）则让 Q 额外评价指定动作。

身体真实线速度不直接送给 Actor，而用于 Q 和编码器的速度监督。监督指用已知真实值训练网络预测；N01 从历史估计速度。接触状态、随机化后的执行器增益、高度扫描也只进入训练侧 Q，不会因拼接方便而泄漏给 Actor。

“观测历史”是按时间排列的多帧输入，“归一化”是 D04 用统计均值和尺度调整输入。本模块先整理环境已经缩放过的字段，维护历史；它不重复执行字段缩放，也不做 D04 的统计归一化。

## 一帧 33 维，Q 为 54 或 241 维

B 是同批并行环境数，C 是 Q 状态信息维数。列区间均左闭右开。下表的物理单位描述缩放前的量，环境输出已乘所列系数。

| policy 帧列区间 | 字段 / 维数 | 物理含义与环境缩放 |
|---|---|---|
| `[0,3)` | `base_ang_vel` / 3 | 身体系角速度 rad/s，×0.25，带噪声 |
| `[3,6)` | `projected_gravity` / 3 | 身体系单位重力方向，无量纲，带噪声 |
| `[6,9)` | `velocity_commands` / 3 | 速度指令 vx/vy（m/s）、wz（rad/s），×(2,2,0.25)，不加噪声 |
| `[9,17)` | `joint_pos` / 8 | 相对默认角 rad，轮子两项置 0，×1，带噪声 |
| `[17,25)` | `joint_vel` / 8 | 关节速度 rad/s，×0.05，带噪声 |
| `[25,33)` | `actions` / 8 | 上一步策略动作，无物理单位，E03 动作顺序，不加噪声 |

`joint_pos/joint_vel` 使用 articulation（仿真中多连杆机器人对象）的关节顺序；它可能与 E03 动作顺序不同。不能为统一顺序擅自重排这些观测；E01 会同时记录两种顺序。

Q 观测按下表拼接，不是把 policy 的带噪声帧直接复制过去：

| Q 列区间 | 观测组 | 内容 |
|---|---|---|
| `[0,36)` | `critic` | 真实身体线速度 3 维（×2.0），再加与 policy 同类的 33 维字段，无噪声 |
| `[36,54)` | `priv` | 两轮接触状态 2 维（接触 +0.5，否则 −0.5）、刚度比例 8、阻尼比例 8 |
| `[54,241)` | `scanner`，仅复杂地形 | 17×11=187 个高度扫描值 |

所以平地 C=36+18=54，复杂地形 C=54+187=241。这是固定配置的预期，运行时还要逐项核对。E02 输出 `frame[B,33]` 与 `q[B,C]`；E05 用帧生成原始 `policy_history[B,10,33]`，D04 归一化后交 D05/N01；Q 状态信息则交 N04，同时另接 `[B,8]` 策略动作。

速度标签取 `q[:,0:3]`。测试标签 `[2,-1,0.4]` 表示实际速度 `[1,-0.5,0.2]` m/s；N01 的预测与标签在同一 2.0 缩放坐标比较，`velocity_label_to_mps` 才负责除以 2。环境字段 scale 与 D04 归一化是两层不同操作。

## `observations.py`：字段、读取与校验

`ObservationField` 是冻结数据类，保存 name、观测函数 func、维数 dim、scale、是否有噪声 noisy、含义 meaning。`POLICY_FIELDS/CRITIC_FIELDS/PRIV_FIELDS/SCANNER_FIELDS` 保存定义，`GROUP_FIELDS/GROUP_CLASSES` 关联组名和原配置类名。`ObservationError` 用于报告字段、形状或有限性错误。

| 函数 | 输入 → 输出、用途与副作用 |
|---|---|
| `q_groups(task)` | `isaac_flat/isaac_rough` → 按拼接顺序的组名；未知任务报错 |
| `expected_q_dim(task)` | 根据上述字段求和 → 54/241，供预期与检查使用，不从任意输入猜 C |
| `describe_observation_fields(task)` | → 每字段的目标、组、列区间、scale、噪声与含义字典列表；排查列偏移入口 |
| `parse_observation_config(env_cfg_source)` | 原配置文本 → 按声明顺序的组/字段/函数/缩放/噪声；用 AST（Python 语法树）读配置，不导入 Isaac，注释掉的项不会出现 |
| `baseline_observation_config()` | 通过 E03 `read_baseline_file` 读固定快照，再调用解析器；只读 Git 子进程 |
| `validate_observations(task,term_names,term_dims,policy_frame_dim=None)` | 用运行时字段名称和每项维数逐项比对，要求 policy 单帧，返回契约 C；列出不一致后报错。E01 `compare_with_baseline` 调用它 |
| `read_observation_groups(observations,task,q_dim)` | 环境组字典 → `frame[B,33],q[B,C]`，检查形状与 NaN/inf，不截断、不补齐、不重复缩放 |
| `velocity_label(q_observation)` | Q 输入 → 前三列视图，不复制，调用方不得原地修改 |
| `velocity_label_to_mps(label)` | 标签或速度预测 → 除以 2.0 的 m/s 值，新张量 |

关键拼接行是：

```python
q = torch.cat([observations[group] for group in q_groups(task)], dim=-1)
```

`q_groups` 决定组顺序，沿最后一维拼出固定列布局。若交换 critic 和 priv，总维数仍可能正确，但前 3 列就不再是速度标签。运行时校验因此还比较每个字段名与维数，测试专门调换两个同为 3 维的字段，证明只核对总长度不够。

返回的 frame 仍引用环境的 policy 张量，Q 拼接分配新内存；本函数不承担完整快照责任。E05 在 reset 前 clone 原始组，之后 D01 再对经历字段 `detach().clone()`。冻结数据类也不等于张量内容不能原地修改。

## 历史为什么移出 Isaac，以及怎样推进

锁定版本的环境先重置结束的环境，再返回观测。E05 要在重置前多计算一份观测；若仍由 Isaac 的历史缓冲自动推进，同一策略步可能被推入两次。因此 E01 将 policy 的 Isaac `history_length` 从 10 改为 0，仅返回单帧；E02 接手维护 10 帧历史。[Isaac Lab v2.3.0 的历史与观测处理](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/managers/observation_manager.py)可供核对。

`push_frame(history,pushes,frame)` 不改输入，返回新历史和新计数：

```python
shifted = torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)
first = (pushes == 0).reshape(-1, 1, 1)
filled = frame.unsqueeze(1).expand_as(history)
return torch.where(first, filled, shifted), pushes + 1
```

第一行丢弃最旧帧并在末尾追加当前帧；第二行把“该环境尚无本轮观测”的标记扩成可广播到帧与通道的形状；第三行生成首帧重复填满的候选；最后按环境选择首帧填充或正常移位，并增加计数。少掉首帧填充会改变原 PPO 输入起点，重复调用则会把一个时刻伪装成多个真实时刻。

`valid_length(pushes)` 返回 min(pushes,10)。填满 10 个槽位不表示已经观察了 10 帧：首帧填满时长度仍为 1。测试连续推入帧号 0～11，最终存 `[2,3,…,11]`、有效长度 10；计数 `[0,1,7,15]` 对应长度 `[0,1,7,10]`。0 只在尚未推入首帧的内部状态出现，D01 接收的经历长度要求 1～10。

`PolicyHistory.__init__(num_envs,device='cpu')` 分配 `history[B,10,33]` 和 int64 `pushes[B]`；`reset(env_ids)` 原地清零指定行；`commit(env_ids,history,pushes)` 只把指定行的候选写回。E05 利用“先构造候选、再选择写回”，让末历史快照与重置后的交互历史互不覆盖。

下游 D05 只切片：索引 9 是当前帧，5:10 是决策窗口，4:9 是辅助另一窗口。辅助学习要求两窗口覆盖的 6 个时刻都是真实帧；这里的 10 帧不是 D03 的未来 n 步回报。

## `test_observations.py` 的证据和排错

测试辅助 `runtime_terms(task)` 构造人工的运行时字段/维数；`ReferenceCircularBuffer.__init__/reset/append/buffer` 是按锁定上游算法复写的对照：分配环形缓冲、局部重置、推帧、旋转后返回从旧到新的副本。它不是启动真实 Isaac。

| 测试类 / 方法 | 核对内容 |
|---|---|
| `FieldContractTest.test_matches_baseline_config` | 名称、函数、scale、噪声与固定快照逐项一致 |
| `test_dimensions_and_actor_inputs` | 33/54/241、无真实线速度直达 Actor、标签和扫描列范围 |
| `test_runtime_validation` | 正确元数据通过，交换同维字段、priv 维数错、policy 含 Isaac 历史被拒绝 |
| `GroupReadingTest.test_concatenation_and_no_padding` | 人工 critic=1、priv=2、scanner=3 的拼接分段正确；错误 C、三维 policy、NaN 被拒绝 |
| `test_velocity_label` | 标签提取和物理速度换算 |
| `HistoryTest.test_matches_reference_buffer_with_partial_resets` | 25 步、两次部分重置与参考缓冲逐步一致 |
| `test_order_fill_and_length` | 首帧填充、帧号顺序、有效长度 |
| `test_push_returns_new_tensor_and_commit_only_selected_rows` | 推帧分配副本，commit 只改指定环境 |

既有 8 项结果来自 [AI 交接表](README.md#实现证据与文档交接)，本轮未重跑。复现入口如下，不需要 Isaac，但字段对照需要本地固定 PPO 快照对象：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/observation_interface/test_observations.py -v
```

维数错误先看 `validate_observations` 的逐项列表；数值量级错先区分字段 scale 与 D04；关节语义错查 articulation 顺序；重置后历史异常先查 pushes、E05 选择逻辑和是否重复推进。真实字段、噪声和机器人状态尚待 E01 `inspect/interact` 在 Isaac 中核对。
