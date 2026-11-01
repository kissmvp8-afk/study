# E01 · 机器人与任务环境｜人类阅读版

[AI 实施规范](README.md) · [环境大模块人类版](../README_HUMAN.md) · [项目人类版](../../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 对照实际实现、注释与测试核对职责、调用、形状、单位和关键片段；面向有 Python 基础的读者，本次不扩大为全面代码审核 |
| 语言润色状态 | 本轮由 Codex 完成讲解与技术核对，不再交 Gemini |
| 已核对代码版本 | 原实现 main `f6c8b03d`；Windows 路径修复 `373bb48d`；新增诊断及本轮证据随本次提交保存，未改算法 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | AGENTS.md、CODEX.md、根与环境大模块/本目录 AI README；[assets.py](assets.py)、[metadata.py](metadata.py)、[isaac_env.py](isaac_env.py)、[run_sac_isaac.py](run_sac_isaac.py)、[test_environment.py](test_environment.py) 全文；E02/E03/E04/E05、T01/T02/T03 相关实现与契约；上游片段与依赖见[阅读记录](../README_HUMAN.md#阅读材料与验证依据) |
| 验证依据与限制 | 2026-10-31 Windows 真实平地 1 环境的加载、50/1005 步交互、短训练保存恢复及提前计数纯截断通过；历史 CPU 记录保留，本轮不重跑全仓。截断仅实际推进 0.04 s，不证明稳定运行 20 s；复杂地形、长期训练、性能与稳定行走未验证。证据见下方 Windows 验证记录 |

## Windows 验证记录（2026-10-31）

Windows 平地单环境最小闭环已验证；提前设置计数器的真实纯时间截断检查通过。复杂地形、长期训练、性能与稳定行走仍未验证。 [精简报告、命令与 JSON 证据](validation/windows_2026-10-31/REPORT.md)。

真实截断诊断在正常 reset 后仅把 `episode_length_buf` 从 0 提前设为 998，保留正式 20 s / 1000 步上限和 `base_contact` 真终止项。随后 2 个策略步（8 个物理步、0.04 s）由原管理器给出 `terminated=False,time_out=True`，首次尝试通过。

E05 的 `next_*` 与同一次计算捕获的重置前末状态相等；`observation` 对应新回合，历史有效长度 1、步号 0、回合编号 0→1。真实记录经过 D01/D02/D03，mask=1、尾部仍属旧回合、排除新回合奖励；后续真实交互 3 步不覆盖已保存末状态。没有手工伪造结束标记，也没有关闭真实终止条件。

此前自然交互 1005 步有 40 次真终止、0 次超时；它没有覆盖超时，本次覆盖来自明确标注的提前计数诊断。短训练为 CPU 学习器 / GPU 仿真，3×8 步保存后恢复续训 1 轮；检查点不保存仿真现场，且续训后未再次保存模型。轮速显式 20 rad/s 仍为“首次仿真调试选择，尚未验证”。

启动使用 `--kit_args="--vulkan --/app/extensions/fsWatcherEnabled=false"`，诊断实际读取到热重载开关为 False。仍有图形接口/Aftermath 告警；未宣称消除所有图形兼容风险。T06 本身未执行这些仿真，因此其 `simulation` / `performance` 仍返回 `not_run`。

## 把 SAC 学习器接到同一个 TITA 任务

SAC（Soft Actor-Critic，软演员—评论家算法）迁移首先要保证机器人、奖励、指令、执行器和时间尺度可比较。执行器把关节目标转换为驱动力矩；URDF 是机器人连杆、关节与几何的描述文件；元数据是关节顺序、周期、维数等描述运行对象的信息。

E01 负责锁定资产、继承原 PPO 任务、核对实际环境，并把它接到 E05 的生命周期接口和 T02 学习器。代码已经实现；当前 CPU 测试核对纯数据计算、人工运行时元数据和源码结构，**现已完成 Windows 平地单环境的真实最小验证**。

## 四个实现文件怎样协作

```text
assets.py：URDF 路径、校验、关节与几何
       ↓                     固定 PPO 快照 → metadata.py：周期、指令与核对规则
isaac_env.py：继承配置、注册任务、创建 Isaac、读取 RuntimeMetadata
       ↓ compare_with_baseline → E02 字段检查 → C
       ↓ TitaSacHooks → E05 EpisodeLifecycle
run_sac_isaac.py：inspect / interact / train
       ↓ 填 T01 配置 → T02 Trainer → T03 保存、恢复、再训练
```

纯数据文件不导入 Isaac。`isaac_env.py` 必须在 `AppLauncher` 启动应用后导入；`run_sac_isaac.py` 将相关导入放进 `main`，避免 Kit（Isaac 应用运行环境）尚未启动就加载仿真模块。各目录通过 `sys.path` 互相导入，当前不是一套已打包安装的 study Python 包。

## `assets.py`：资产已锁定，运行时仍要校验

`ROBOT_ASSET` 固定 `DDTRobot/ddt_ros2_control` 的提交 `e7b32d9746df900680641ff2845ffb0b5ffe99ed`，URDF 路径为 `urdfs/tita_description/urdf/robot.urdf`，SHA-256 为 `0941b83136cd5c8c89bbb33165bff515a3a988d33319b67c8b06bfb9c71350ce`。SHA-256 是按文件字节计算的摘要，用来发现内容不同。对应记录同时在 [UPSTREAM.json](../../UPSTREAM.json) 与 T01 `TARGET_ENVIRONMENT`。

选择依据是固定 PPO 上游提交日期之前资产 main 的最新提交；这是本项目为可重复性作的选择，不是原 PPO 曾记录了该资产提交。本仓库没有复制 URDF/网格，外部资产仍需放在原 DDT_Lab 的预期目录中。已锁定也不等于已在仿真加载成功。

| 类 / 函数 | 输入 → 输出、职责与副作用 |
|---|---|
| `AssetError` | 资产缺失、摘要不符、URDF 结构不合预期的异常 |
| `UrdfJoint` | 冻结数据类，保存 name/type、父子连杆、位置上下限、力矩/速度上限、origin_xyz/rpy、axis；角度 rad，位置 m，速度 rad/s，力矩 N·m |
| `urdf_path(ddt_lab_root)` | → 原版约定下的文件路径，只拼路径，不验证存在 |
| `sha256_of(path)` | 读文件字节 → 摘要字符串 |
| `verify_urdf(path)` | 检查文件、摘要和相邻 meshes 目录；返回 `exact` 或仅换行转换一致的 `crlf_normalized`，不修改文件。它核对 URDF 字节和网格目录存在，不逐个校验网格内容 |
| `parse_urdf_joints(path)` | XML 文件 → 按文件顺序的名称到 `UrdfJoint` 字典，不执行模型；内部 `number` 读取可选限位数值，`_triple` 解析三元组 |
| `wheel_radius(path,wheel_joint)` | 轮关节的子连杆碰撞圆柱 → 半径 m，非圆柱时报错 |
| `soft_joint_limits(joints,names,factor)` | 硬限位与系数 → 每关节软限位，缺位置限位时报错 |
| `link_position(joints,link,joint_positions)` | 关节角字典 → 连杆原点在根坐标系的位置 m；未给的关节角按 0 |
| `wheel_half_track(joints,joint_positions)` | 两个轮心 y 距离绝对值的一半，单位 m，用给定姿态计算 |

`link_position` 沿父链取关节，再从根向目标连杆相乘变换。辅助 `_transform(rpy,xyz)` 构造含平移的 4×4 矩阵，旋转顺序为 Rz·Ry·Rx；`_axis_rotation(axis,angle)` 用 Rodrigues 公式构造绕轴旋转；`_matmul(a,b)` 做矩阵乘法。这里只用于该机器人几何核对，不是完整通用动力学引擎。

软限位的真实短代码是：

```python
mean, span = (joint.lower + joint.upper) / 2, joint.upper - joint.lower
limits[name] = (mean - 0.5 * span * factor, mean + 0.5 * span * factor)
```

第一行求中点和全宽，第二行将半宽乘 factor 后向两边展开。人工测试硬范围 `[−0.5,1.5]` rad，factor=0.9，得到 `[−0.4,1.4]` rad；把系数乘上下端点本身会改变中心。真实运行用原版 factor=0.9 的 URDF 推算，与 Isaac 读到的软限位对照。

轮子的位置范围不能当作速度范围；即使锁定 URDF 的轮关节是很宽的 revolute 限位，E04 也只消费六个位置控制关节的软限位。轮速另由用户指定工作范围。

Windows 检出可能把 LF 换成 CRLF。`verify_urdf` 仅在换回 LF 的摘要一致时接受并标记 `crlf_normalized`；其他内容差异仍拒绝。测试中真实文件的 `test_real_urdf` 使用原始摘要，需保留原始换行。

## `metadata.py`：把“预期”与“运行时读到”逐项比较

`RuntimeMetadata` 保存任务、环境数、仿真设备、articulation 关节顺序、默认角/软限位/速度上限、动作项、周期、回合上限、观测字段/维数和指令范围。运行时从第 0 个环境取这些共同配置，不是每个环境每步的观测。`MetadataError` 汇总不一致项。

| 函数 | 输入 → 输出、调用关系 |
|---|---|
| `_post_init_assignments(source,class_name)` | AST 读取 `__post_init__` 中的字面量赋值，非字面量跳过，供快照静态查询 |
| `baseline_timing(repo_dir)` | 固定 rough/flat 配置 → 物理步 0.005 s、decimation=4、策略步 0.02 s、回合 20 s/1000 步；确认平地没有改写这些项 |
| `baseline_command_ranges(repo_dir)` | 快照 → vx∈[−1,1] m/s、vy=0、wz∈[−1,1] rad/s；只读取这三项，不解析含 math.pi 的 heading 范围 |
| `required_wheel_speed(command_ranges,half_track,wheel_radius)` | 最大指令和几何 → 无滑动差速外侧轮速需求 rad/s；vy 非零时该模型不适用，报错 |
| `wheel_speed_ranges(limit,required,model_limit=30)` | 显式上限 → 两轮对称范围；None/非有限/小于需求或大于模型上限时报错，没有工作轮速默认值 |
| `compare_with_baseline(runtime,contract,timing,command_ranges,urdf_soft_limits=None,tol=1e-4)` | 核对动作展开顺序/类型/scale、受控关节存在、腿部默认角、周期/回合长、指令、可选 URDF 软限位，再调 E02 校验观测，返回 C；失败抛 `MetadataError` |
| `leg_soft_limits(runtime,contract)` | 只选六个位置项的运行时软限位 → E04 所需字典 |
| `fill_isaac_run_config(...,seed,steps_per_iteration,capacity_per_env,debug_sizes)` | 已核对元数据、C、轮速范围和用户规模 → 填齐 T01 `RunConfig` 并校验；只创建配置，不启动训练 |

动作项展开后按 E03 顺序比较，articulation 自身的关节顺序可以不同。不能为“让检查通过”按字母重排动作。`tol` 用于 float32 运行时值和 Python 浮点值比较。`compare_with_baseline` 是明确列项的核对，不保证任意环境属性都已验证。

时间上，一个策略动作保持 4 个物理步：`4×0.005=0.02 s`，即 50 Hz；`ceil(20/0.02)=1000`。D03 的 γ 按策略步折扣，不能改为每个 0.005 s 都算一次。

### 工作轮速下限和 20 rad/s 的身份

无滑动差速估算只作下限依据：

```python
vx = max(abs(v) for v in command_ranges["lin_vel_x"])
wz = max(abs(v) for v in command_ranges["ang_vel_z"])
return (vx + wz * half_track) / wheel_radius
```

前两行取最大速度指令幅值；转向使外侧轮多承担 `wz×半轮距` 的线速度，最后除以轮半径得到角速度。锁定 URDF 默认姿态下，半轮距约 0.28253 m、半径 0.0925 m，所以 `(1+1×0.28253)/0.0925≈13.87 rad/s`。代码用计算值校验，而非把四舍五入后的 13.87 写死。姿态、打滑和地形仍可能需要余量，公式不证明某个范围足以控制机器人。

首次仿真调试可考虑显式传入 **20 rad/s**，但它只是建议的调试起点，也是部分 CPU 测试的输入，**不是代码默认值，也不是已验证的 TITA 参数**。上限参考为模型 30 rad/s；启动入口实际使用运行时两轮速度上限的较小值约束范围。最终工作轮速仍需使用者决定并记入实验来源。

`fill_isaac_run_config` 把 C 和六腿软限位标为运行时来源，轮速标为用户设置；E、每轮步数和容量在 `debug_sizes=True` 时标为接入调试值。资产提交、SAC 任务 id、`sim_device` 写入 environment，学习器 `device='cpu'`。资产固定并未消除每次运行读取实际值的必要。

## `isaac_env.py`：真实钩子与平地单环境验证

`TitaSacFlatEnvCfg/TitaSacRoughEnvCfg` 分别继承原版任务，两个 `__post_init__` 都先调用父实现，再只赋值 `self.observations.policy.history_length=0`。场景、执行器、奖励、指令、终止、事件、课程与物理周期沿用原版。

- `register_sac_tasks()` 注册两个 `Tita-SAC-Velocity-*-v0` id，检查冲突并保留两个原 PPO id；有注册表副作用，可重复调用同一配置。
- `_flatten(value,prefix)` 展平嵌套配置；`sac_config_differences(task)` 比较新建默认 PPO/SAC 配置，差异必须恰为 policy history 10→0，否则报错。它不是比较用户改过的所有实验参数。
- `assert_no_isaac_history(cfg)` 检查组级以及生效的项级历史都关闭，避免 E05 两次 compute 重复推进。
- `make_sac_cfg(task,num_envs,seed,sim_device)` 新建配置、设置三项运行值并检查历史；`urdf_path_from_cfg(cfg)` 取原配置实际资产路径。
- `make_env(task,cfg)` 注册后调用 Gym 创建未包装的 `ManagerBasedRLEnv` 并检查类型，有真实仿真创建副作用。
- `read_runtime_metadata(env,task)` 读取机器人数据与 manager，动作项用 `find_joints(...,preserve_order=...)` 重新解析，用动作类区分位置/速度，检查解析数等于 action_dim，生成 `RuntimeMetadata`。这一步已在平地单环境实际核对，复杂地形仍待验证。

`TitaSacHooks.__init__(env,randomize_initial_episode_length=False)` 保存环境、E、设备及最近回合日志；启用 recorder（记录器）的分支未实现，构造时明确拒绝。其余方法对应 E05 所需协议：

| 方法 | 修改或读取的状态，为什么要这样做 |
|---|---|
| `reset_all()` | `env.reset()` 后可选打散 episode_length_buf，到时计数分散；默认关，原 PPO 训练入口曾打开这一选项 |
| `apply_action_and_simulate(action)` | `process_action` 做一次物理映射，然后 decimation 次 apply/write/sim.step/scene.update，增加环境计数 |
| `compute_terminations()` | 计算 termination_manager，复制 terminated/time_outs，避免后续 manager 原地改写 |
| `compute_rewards()` | `reward_manager.compute(dt=step_dt)` 后复制原始奖励，不加 PPO 的 γV 超时补偿 |
| `compute_observations()` | `compute(update_history=False)` 并保存 obs_buf；历史关闭后得到单帧，由 E02/E05 维护历史 |
| `reset_envs(env_ids)` | 记录 reset 调用、执行 `_reset_idx`、需要时重渲染，保存 extras 中的回合日志 |
| `after_reset()` | 更新指令，执行 interval 事件；下一次交互观测应在它之后计算 |

物理映射发生在：

```python
env.action_manager.process_action(action.to(env.device))
```

输入仍是无物理单位的八维策略动作，Isaac 动作项做腿部 `q0+0.25a`、轮速 `5a` 和目标 clip。若在此之前先用 E03 换算目标，就会重复缩放。之后一个策略周期内多次 `apply_action` 是保持同一控制目标，不是每次重新把 scale 乘一遍。

这条路径拆分了锁定版本的 step，使用 `_sim_step_counter/_reset_idx` 等私有成员。升级 Isaac Lab 必须重新核对；现有 AST 结构检查不能证明真实 manager、传感器、重置及物理推进都正确。

## `run_sac_isaac.py`：从核对到短训练和恢复

`parse_args()` 定义模式、任务和调试规模，AppLauncher 追加 `--device/--headless`；train 缺轮速时在启动应用前报错。`_write_json(path,data)` 创建父目录并写 UTF-8 JSON。`main(args)` 串起全部操作，返回 summary 字典，打印并写输出，正常结束关闭环境。脚本入口先 parse、启动应用，再调用 main，finally 关闭应用。

| 模式 | 实际顺序与输出 |
|---|---|
| `inspect` | E03 动作表 → SAC 配置 → 校验 URDF/几何/软限位 → 配置差异 → 创建环境 → 运行时元数据核对；写 `metadata.json` 和 `summary.json` |
| `interact` | 先 inspect，再构造 E05 并 reset，执行零策略动作；检查 next 历史/Q/奖励有限，累计终止/到时/奖励和并记形状与编号 |
| `train` | 先 inspect，再校验显式轮速、填 T01/写 `run_config.json` → E05/T02 训练 → T03 写 `checkpoint.pt` → 新 Trainer → restore → 再训练 1 轮 |

默认 E=16、迭代 3、交互 50 步；每轮步数/每环境容量未传时用 8/200，均为接入调试规模。两项都显式给出时规模来源标为用户设置；任一缺省则 `debug_sizes=True`。`--device` 指仿真设备（通常 cuda:0），学习器仍只能 CPU。实际逐张量搬运与同步成本未测量，不能把可创建大环境数等同于学习吞吐已验证。

train 模式每轮 8 步、3 轮只采集 24 次策略步；即使没有真终止，也远不到 1000 步的自然到时。interact 用 `--steps 1005` 只是增加覆盖到时的机会，可能先摔倒并多次重置；要看 `summary.json` 的 `interact.time_out>0` 才能说覆盖了到时路径。

T03 保存网络、优化器、统计、计数、随机流和经验池；恢复要求学习配置一致。新 Trainer 构造与 restore 都会 reset，旧关节/接触现场不保存，恢复后新轮次编号防止 D02 跨回合拼接。此入口演示保存后立刻恢复并续一轮，尚无命令行加载任意旧检查点的选项。

完成依赖与资产准备后，在仓库根目录可按以下顺序验证；**本轮执行了平地单环境变体，复杂地形未执行；实际命令与结果见 Windows 验证记录**。完整锁定版本安装与目录步骤见 [AI 运行说明](README.md#windows--nvidia-安装与验证步骤powershell)，不要把 CPU 测试环境的 torch 2.15.0 当作 Isaac 的安装组合。

```powershell
python environment_control\robot_environment\run_sac_isaac.py --task isaac_flat --mode inspect --headless
python environment_control\robot_environment\run_sac_isaac.py --task isaac_rough --mode inspect --headless
python environment_control\robot_environment\run_sac_isaac.py --task isaac_flat --mode interact --steps 1005 --headless
# 20 仅为显式选择的首次仿真调试建议，不是默认或已验证值
python environment_control\robot_environment\run_sac_isaac.py --task isaac_flat --mode train --wheel_speed_limit 20 --headless
```

## `test_environment.py` 的证据与排错

辅助函数 `write_fixture(directory,with_meshes=True)` 写人工 URDF/目录；`soft(limits,factor)` 独立算软限位；`make_runtime(contract,task,**changes)` 造人工元数据，并故意让 articulation 顺序不同于动作顺序。测试使用临时目录与 mock 校验摘要分支，不修改真实资产。

| 测试类 / 方法 | 检查内容与限制 |
|---|---|
| `AssetTest.test_parse_soft_limits_and_geometry` | 人工关节解析、软限位、continuous 缺限位、半径、转动后轮距 |
| `test_verify_reports_path_and_hash` | 缺路径、摘要不符、CRLF、缺 meshes |
| `test_paths_and_pinned_commit_consistent` | 路径与 E01/T01/UPSTREAM 三处锁定记录一致 |
| `test_real_urdf` | 需 TITA_URDF；真实文件摘要、关节、限位、几何及轮速下限。无变量时 skipped，不是通过 |
| `BaselineMetadataTest.test_timing_and_commands`、`test_wheel_speed` | 固定快照周期/指令、差速公式、无默认及非法轮速拒绝 |
| `RuntimeComparisonTest.test_matching_runtime_returns_q_dim`、`test_each_mismatch_is_reported` | 人工元数据逐项一致/报错；`setUpClass/compare` 只准备并复用比较输入 |
| `RunConfigTest.test_filled_config`、`test_trainer_runs_on_filled_config` | 填齐的 Isaac 模式配置接 CPU Trainer、拒绝 GPU 学习器字符串；`setUpClass/fill` 准备动作表与配置。后者用人工 ScriptedSimulation，非真实仿真 |
| `IsaacFileStructureTest.test_hooks_match_lifecycle_calls` | 钩子名称覆盖 E05 调用，`methods` 读 AST 方法集合 |
| `test_sac_cfg_changes_only_history` | AST 中只改历史，允许差异常量一致 |
| `test_task_ids_do_not_override_ppo` | SAC/PPO id 不冲突 |
| `test_launcher_imports_isaac_modules_after_app_start` | 顶层导入结构正确；`setUpClass` 解析两个文件，不启动应用 |

14 项既有 CPU 记录、设置真实资产后的记录和历史变异检查见 [AI 交接表](README.md#实现证据与文档交接)。本轮复用已安装环境与已核对的真实 URDF，不重跑全仓测试；历史 CPU 记录与本次真实仿真证据分开保存。需要复现 CPU 检查时：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/robot_environment/test_environment.py -v
```

加载失败先查 DDT_Lab 可编辑安装、资产路径/摘要和依赖版本；任务差异看 `sac_config_differences`；动作错位看 `action_terms` 与 articulation 两种顺序；截断错误转 E05；学习慢先区分仿真设备、CPU 学习器与搬运成本。真实 Isaac 行为、轮速余量和训练效果都是待验证项，CPU 测试不能替代它们。
