# E01 · 机器人与任务环境｜AI 实施规范

状态：Windows 平地单环境最小闭环已验证；提前设置计数器的真实纯时间截断检查通过。复杂地形、长期训练、性能与稳定行走仍未验证。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../README.md#项目维护规则)

## Windows 验证记录（2026-10-31）

Windows 平地单环境最小闭环已验证；提前设置计数器的真实纯时间截断检查通过。复杂地形、长期训练、性能与稳定行走仍未验证。 [精简报告、命令与 JSON 证据](validation/windows_2026-10-31/REPORT.md)。

真实截断诊断在正常 reset 后仅把 `episode_length_buf` 从 0 提前设为 998，保留正式 20 s / 1000 步上限和 `base_contact` 真终止项。随后 2 个策略步（8 个物理步、0.04 s）由原管理器给出 `terminated=False,time_out=True`，首次尝试通过。

E05 的 `next_*` 与同一次计算捕获的重置前末状态相等；`observation` 对应新回合，历史有效长度 1、步号 0、回合编号 0→1。真实记录经过 D01/D02/D03，mask=1、尾部仍属旧回合、排除新回合奖励；后续真实交互 3 步不覆盖已保存末状态。没有手工伪造结束标记，也没有关闭真实终止条件。

此前自然交互 1005 步有 40 次真终止、0 次超时；它没有覆盖超时，本次覆盖来自明确标注的提前计数诊断。短训练为 CPU 学习器 / GPU 仿真，3×8 步保存后恢复续训 1 轮；检查点不保存仿真现场，且续训后未再次保存模型。轮速显式 20 rad/s 仍为“首次仿真调试选择，尚未验证”。

启动使用 `--kit_args="--vulkan --/app/extensions/fsWatcherEnabled=false"`，诊断实际读取到热重载开关为 False。仍有图形接口/Aftermath 告警；未宣称消除所有图形兼容风险。T06 本身未执行这些仿真，因此其 `simulation` / `performance` 仍返回 `not_run`。

独立诊断脚本：[diagnose_timeout_isaac.py](diagnose_timeout_isaac.py)。`parse_args` 限定 headless / 最多三次尝试；`CapturingHooks` 只复制原管理器输出；`main` 核对 E01 元数据并串起 E05→D01/D02/D03，写 metadata、summary 和真实观测/经历 JSON。脚本不训练、不保存检查点；断言失败或未覆盖返回非零。

## 实施目标与术语

加载 TITA 并提供相同的平地/复杂地形任务，使 PPO 与 SAC 控制的是同一个对象。

术语先说明：执行器：将角度或速度目标转成力矩的控制模型。URDF：描述机器人部件、关节和几何形状的文件。

## 与原 PPO 的关系

沿用原模型、指令、奖励与执行器设置；新增 SAC 环境注册，训练方法改变不应暗中改变任务。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/flat_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/flat_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/assets/ddt_robot.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/assets/ddt_robot.py)
- [Isaac Lab v2.3.0：`envs/manager_based_rl_env.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)（`step` / `_reset_idx` 的执行顺序，`TitaSacHooks` 逐段复刻）

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。SAC 配置直接继承原版配置类，只改一项（见下），差异由 `sac_config_differences` 在运行时逐项比对。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 任务（`isaac_flat` / `isaac_rough`）、并行环境数、种子、仿真设备；原版 DDT_Lab 701a49c 安装及其目录下的 ddt_ros2_control e7b32d9；训练时另需用户给出的工作轮速上限。 |
| 输出 | E05 钩子（`TitaSacHooks`）与由它构造的 `EpisodeLifecycle`；运行时元数据 `RuntimeMetadata`（关节顺序、默认角、软限位、关节速度上限、动作项、周期、观测组维数、指令范围）；填好并通过校验的 T01 `RunConfig`。 |

## 资产锁定与选择依据

| 项目 | 值 |
|---|---|
| 仓库 | [DDTRobot/ddt_ros2_control](https://github.com/DDTRobot/ddt_ros2_control) |
| 提交 | `e7b32d9746df900680641ff2845ffb0b5ffe99ed`（2026-05-29） |
| 路径 | `urdfs/tita_description/urdf/robot.urdf`，网格在 `urdfs/tita_description/meshes` |
| SHA-256 | `0941b83136cd5c8c89bbb33165bff515a3a988d33319b67c8b06bfb9c71350ce` |
| 选择依据 | 原版 README 只要求克隆默认分支，未固定版本。取原版 DDT_Lab 701a49c（2026-06-26）之前 main 的最新提交，即原 PPO 当时会拿到的版本。该 URDF 在仓库中只被 4f5d962、e62e1aa（2026-01-08）、beb05df（2026-10-30）改过；e7b32d9 的文件与 e62e1aa 相同，beb05df 已改动 URDF，因此不用更新的版本 |
| 许可 | 仓库未声明许可证：本仓库只记录路径、提交与校验值，不复制 URDF 或网格 |
| 放置位置 | 与原版 `ddt_robot.py` 的 `DDT_MODEL_DIR` 一致：`<DDT_Lab 根目录>\ddt_ros2_control\urdfs\...`。运行时从 `cfg.scene.robot.spawn.asset_path` 取实际路径再校验 |

同一份记录在 [assets.py](assets.py) `ROBOT_ASSET`、根目录 [UPSTREAM.json](../../UPSTREAM.json) `robot_assets` 与 T01 `TARGET_ENVIRONMENT["robot_assets_commit"]`，测试核对三者一致。

## 实际文件与接口

纯数据部分（`assets.py`、`metadata.py`）只依赖标准库、PyTorch 与本仓库模块，在 CPU 上测试；Isaac 部分（`isaac_env.py`、`run_sac_isaac.py`）依赖 Isaac Lab，只能在 `AppLauncher` 启动后导入。

### [assets.py](assets.py)

| 符号 | 签名 / 字段 | 职责、单位、异常与副作用 |
|---|---|---|
| `ROBOT_ASSET` | dict：repository、url、commit、urdf_path、urdf_sha256、license、selection | 锁定版本记录 |
| `WHEEL_JOINTS` | `("joint_left_leg_4", "joint_right_leg_4")` | 速度控制的轮子关节 |
| `AssetError` | `RuntimeError` 子类 | 资产缺失、校验值不符或 URDF 不符合预期 |
| `UrdfJoint` | 冻结 dataclass：name、type、parent、child、lower、upper、effort、velocity、origin_xyz、origin_rpy、axis | 位置 rad，origin m / rad |
| `urdf_path` | `(ddt_lab_root: Path) -> Path` | 按 `DDT_MODEL_DIR` 约定拼出 URDF 路径（pathlib，Windows 可用） |
| `sha256_of` | `(path) -> str` | 文件字节的 SHA-256 |
| `verify_urdf` | `(path) -> str` | 文件存在、校验值一致、`../meshes` 存在。返回 `"exact"`；若只是 Windows git `core.autocrlf` 把 LF 换成 CRLF（换回后一致）返回 `"crlf_normalized"`。缺文件时报具体路径；内容不同报实际校验值 |
| `parse_urdf_joints` | `(path) -> dict[str, UrdfJoint]` | 按文件顺序解析全部关节 |
| `wheel_radius` | `(path, wheel_joint) -> float` | 轮子子连杆碰撞圆柱的半径（m）；TITA 为 0.0925 |
| `soft_joint_limits` | `(joints, names, factor) -> dict[str, (float, float)]` | Isaac Lab 2.3.0 公式：中点 ± 0.5 × 范围 × factor（rad）；无位置限位的关节报错 |
| `link_position` | `(joints, link, joint_positions) -> (x, y, z)` | 正运动学：URDF origin（R = Rz·Ry·Rx）与绕关节轴的 Rodrigues 旋转逐级相乘，返回连杆原点在根坐标系的位置（m） |
| `wheel_half_track` | `(joints, joint_positions) -> float` | 两轮心 y 距离的一半（m）；TITA 默认姿态下 0.28253 |

### [metadata.py](metadata.py)

| 符号 | 签名 / 字段 | 职责、单位、异常与副作用 |
|---|---|---|
| 常量 | `MODEL_WHEEL_VELOCITY_LIMIT=30.0`、`LEG_SOFT_LIMIT_FACTOR=0.9`、`SAC_TASK_IDS`、`PPO_TASK_IDS`、`DEBUG_WINDOWS` | 模型轮速上限参考（rad/s）、原版软限位系数、任务 id、调试值来源标签 |
| `MetadataError` | `ValueError` 子类 | 列出全部不一致 |
| `RuntimeMetadata` | 冻结 dataclass：task、num_envs、sim_device、joint_names、default_joint_pos、soft_joint_pos_limits、joint_vel_limits、action_terms、physics_dt、decimation、step_dt、max_episode_length、obs_term_names、obs_term_dims、command_ranges | 由 `isaac_env.read_runtime_metadata` 填写（第 0 号环境） |
| `baseline_timing` | `(repo_dir=根目录) -> dict` | 从固定快照 AST 读取 sim.dt=0.005、decimation=4、episode_length_s=20，推出 step_dt=0.02、max_episode_length=1000；确认平地配置未改写这三项 |
| `baseline_command_ranges` | `(repo_dir) -> dict` | lin_vel_x=(−1,1)、lin_vel_y=(0,0)、ang_vel_z=(−1,1)（heading 含 math.pi，不读） |
| `required_wheel_speed` | `(command_ranges, half_track, wheel_radius) -> float` | 无滑动差速外侧轮速 (max\|vx\| + max\|wz\| × 半轮距) / 轮半径（rad/s）；lin_vel_y 非零时报错 |
| `wheel_speed_ranges` | `(limit, required, model_limit=30.0) -> dict` | 用户给定上限 → 两轮对称 (−limit, limit)；None / 非有限 / 不在 [required, model_limit] 时报错。**没有默认值** |
| `compare_with_baseline` | `(runtime, contract, timing, command_ranges, urdf_soft_limits=None, tol=1e-4) -> int` | 动作项展开后的关节顺序 / 控制类型 / scale 等于 E03；8 个受控关节都在；默认角等于 E03；周期、回合长度、指令范围；URDF 推算的软限位；观测组（调用 E02 `validate_observations`）。返回运行时 C；任何不一致抛 `MetadataError` |
| `leg_soft_limits` | `(runtime, contract) -> dict` | 只取 6 个位置控制关节的运行时软限位（轮子的位置限位不进入 E04） |
| `fill_isaac_run_config` | `(runtime, contract, q_dim, wheel_ranges, required_speed, *, seed, steps_per_iteration, capacity_per_env, debug_sizes) -> RunConfig` | 在 T01 `isaac_config` 上填 q_dim、软限位、工作轮速、采集规模、种子，`num_envs` 取实际环境数；来源标为“运行时 / 用户设置 / Windows 接入调试值”；`environment` 记录资产提交、`sim_device`、SAC 任务 id；学习器 `device="cpu"`；最后 `validate_configuration` |

### [isaac_env.py](isaac_env.py)（需要 Isaac；平地单环境已验证）

| 符号 | 签名 | 职责 |
|---|---|---|
| `TitaSacFlatEnvCfg` / `TitaSacRoughEnvCfg` | 继承原版 `TitaFlatEnvCfg` / `TitaRoughEnvCfg` | 只设 `observations.policy.history_length = 0`；场景、DDT_TITA_CFG 执行器、指令、奖励、终止、事件、课程、周期全部沿用 |
| `register_sac_tasks` | `() -> None` | 注册 `Tita-SAC-Velocity-Flat-v0` / `Tita-SAC-Velocity-Rough-v0`（entry point `isaaclab.envs:ManagerBasedRLEnv`）；同名 id 已被别的配置占用时报错；确认原版 `DDT-Velocity-Flat-Tita-v0` / `DDT-Velocity-Rough-Tita-v0` 仍指向原配置类 |
| `sac_config_differences` | `(task) -> dict` | 两份默认配置 `to_dict()` 展开逐项比较；差异必须恰好是 `{"observations.policy.history_length": (10, 0)}`，否则报错 |
| `assert_no_isaac_history` | `(cfg) -> None` | 所有观测组 / 项都没有 Isaac 历史（E05 每步调用两次 compute，有历史会被重复推进） |
| `make_sac_cfg` | `(task, num_envs, seed, sim_device)` | 新建 SAC 配置并只设置这三项 |
| `urdf_path_from_cfg` | `(cfg) -> Path` | `cfg.scene.robot.spawn.asset_path` |
| `make_env` | `(task, cfg) -> ManagerBasedRLEnv` | `gym.make(SAC id, cfg=cfg).unwrapped` |
| `TitaSacHooks` | `(env, randomize_initial_episode_length=False)` | E05 钩子：`reset_all`（`env.reset`，可选按原 PPO 脚本打散回合计数）、`apply_action_and_simulate`（process_action → 4 次 apply_action / write / sim.step / scene.update → 计数加一）、`compute_terminations`（返回 terminated、time_outs 的副本）、`compute_rewards`（`reward_manager.compute(dt=step_dt)` 的副本，无超时补偿）、`compute_observations`（`compute(update_history=False)`）、`reset_envs`（recorder → `_reset_idx` → 可选重渲染 → recorder，保存 `extras["log"]`）、`after_reset`（指令更新、interval 事件）。启用 recorder 的环境直接拒绝 |
| `read_runtime_metadata` | `(env, task) -> RuntimeMetadata` | articulation 关节顺序、`default_joint_pos`、`soft_joint_pos_limits`、`joint_vel_limits`（第 0 号环境）；动作项按 JointAction 自身的 `find_joints(cfg.joint_names, preserve_order)` 重新解析，并按 `JointPositionAction` / `JointVelocityAction` 判断类型；`step_dt`、`physics_dt`、`max_episode_length`；`observation_manager.active_terms` / `group_obs_term_dim`；指令范围 |

不直接调用 `env.step()`：锁定版本在 `step` 返回前已经重置结束的环境，拿不到重置前末状态。复刻时用到私有成员 `_sim_step_counter` 与 `_reset_idx`，只对 v2.3.0 核对过，升级 Isaac Lab 必须重新核对。

### [run_sac_isaac.py](run_sac_isaac.py)（需要 Isaac；平地单环境已验证）

`python environment_control\robot_environment\run_sac_isaac.py --task {isaac_flat,isaac_rough} --mode {inspect,interact,train} [选项] --headless`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num_envs` | 16（接入调试值） | 并行环境数；原 PPO 为 4096 |
| `--seed` | 0 | 环境与 T01 种子 |
| `--steps` | 50 | interact 模式交互次数（每次 0.02 s）；回合上限 1000 步，所以至少要 1000 次以上才可能出现到时，但不保证（可能先摔倒重置），以输出的 `time_out` 计数为准 |
| `--wheel_speed_limit` | **无** | 工作轮速上限 rad/s；train 模式必填，启动 Isaac 前就检查 |
| `--iterations` | 3（接入调试值） | train 轮数 |
| `--steps_per_iteration` / `--capacity_per_env` | 未给出时用 8 / 200 并标为调试值 | 两者都给出时来源标为“用户设置” |
| `--randomize_initial_episode_length` | 关 | 原 PPO 脚本 `init_at_random_ep_len=True`；是否沿用需用户决定 |
| `--output_dir` | `logs\sac_isaac\<年-月-日_时-分-秒>` | 写 `metadata.json`、`summary.json`、`run_config.json`、`checkpoint.pt` |
| `--device`（AppLauncher） | `cuda:0` | **仿真**设备；学习器固定在 CPU |

三种模式：

1. `inspect`：E03 动作表（固定快照）→ SAC 配置 → URDF 校验 → URDF 软限位与轮速依据 → `sac_config_differences` → 创建环境 → `read_runtime_metadata` → `compare_with_baseline`，写 `metadata.json`。
2. `interact`：在 inspect 之后用 `EpisodeLifecycle(TitaSacHooks(env), task, q_dim)` 执行零动作（默认姿态、轮速 0），检查有限值，统计终止 / 到时 / 奖励和，记录观测形状与编号。
3. `train`：在 inspect 之后 `wheel_speed_ranges` → `fill_isaac_run_config` → `export_run_configuration` → T02 `Trainer(config, lifecycle, contract)` 训练 `--iterations` 轮 → T03 `save_checkpoint` → 新 `Trainer` → `restore_checkpoint` → 再训练 1 轮。

数据搬运：仿真张量在 `--device`（GPU）上；`EpisodeLifecycle.step` 把 CPU 上的动作搬到仿真设备，把重置前 / 后两份观测、奖励与标记搬回 CPU。每步有多次张量搬运：动作 CPU→仿真设备 1 次；仿真设备→CPU 按张量逐个搬，重置前、重置后两份观测各按观测组搬（平地 policy / critic / priv 共 3 个，复杂地形再加 scanner 共 4 个），另加 terminated、time_outs、被重置环境编号与奖励各 1 个，即平地 10 次、复杂地形 12 次；run_isaac_step 中的 nonzero 在 GPU 上还会引起一次同步。实际耗时未测量。学习器只实现了 CPU 版本：T01 `validate_configuration` 拒绝 `device != "cpu"`，只改配置字符串不会得到 GPU 学习器。4096 个环境时 CPU 学习器可能成为瓶颈，是否实现 GPU 学习器需要另行决定。

## 必须成立的规则

- 策略周期为 0.02 秒、物理步长为 0.005 秒，第一版沿用原配置（`baseline_timing` 从快照读取，`compare_with_baseline` 与运行时核对）。
- 机器人资源来自外部 ddt_ros2_control e7b32d9；运行前校验 SHA-256，不假装基线分支已包含 URDF。
- 不把原仓库未使用的 Cartpole 模板当成 TITA 配置；SAC 环境注册不覆盖 PPO 入口（`register_sac_tasks` 检查）。
- 运行时可读取的值（关节顺序、默认角、软限位、观测维数 C）一律取运行时值并与原版核对；工作轮速没有已验证依据，必须由用户显式给出，不能用模型 30 rad/s 代替。

## 工作轮速依据（待用户决定余量）

原版指令 vx ∈ [−1, 1] m/s、vy = 0、wz ∈ [−1, 1] rad/s。按无滑动差速运动学，外侧轮速 = (|vx| + |wz| × 半轮距) / 轮半径。锁定 URDF 在默认姿态（leg_1=0.1、leg_2=0.8、leg_3=−1.5 rad）下的半轮距为 0.28253 m，轮半径 0.0925 m，所以跟踪最大指令至少需要 **13.87 rad/s**。模型 `velocity_limit_sim = 30 rad/s` 是上限。13.87 只是下限依据：跟踪误差、腿部姿态变化（轮距随 leg_1 改变）、地形与打滑都要额外余量，但余量没有已验证的数值，因此由用户在 [13.87, 30] 内选定。若选 30，含义是“不另设工作范围”，应在实验记录中写明。

## 需要用户决定的参数

| 参数 | 现状 | 依据 / 约束 |
|---|---|---|
| 工作轮速上限 `--wheel_speed_limit` | 无默认值 | 见上节；程序只检查在 [13.87, 30] 内 |
| 正式训练规模：`--num_envs`、`--steps_per_iteration`、`--capacity_per_env`、`--iterations` | 只有接入调试值 16 / 8 / 200 / 3 | 原 PPO 4096 个环境；作者 `num_steps_per_env` 无默认值；容量按显存/内存预算（T01 `estimate_memory`）选择 |
| 开局是否打散回合计数 `--randomize_initial_episode_length` | 默认关 | 原 PPO 打开；关闭时所有环境在第 1000 步同时到时 |
| 学习器设备 | 只能 CPU | 是否实现 GPU 学习器需另立任务 |
| 种子 `--seed` | 0 | — |

## 代码注释要求

共同要求见 [代码注释规范](../../README.md#代码注释规范)。

本模块特别需要注释：标注控制周期与物理步长的关系、资产版本来源、环境注册及奖励配置沿用的范围。

## 验收案例

### CPU 已执行（[test_environment.py](test_environment.py)，不是仿真验证）

| 案例 | 对应测试 | 结果 |
|---|---|---|
| 人工 URDF：关节解析、软限位公式、continuous 关节无限位报错、轮半径、零位与转动后的半轮距（与解析式比较） | `AssetTest.test_parse_soft_limits_and_geometry` | 通过 |
| 缺文件报具体路径；校验值不符报错；CRLF 换行记为 `crlf_normalized`、内容改动仍报错；缺网格目录报错 | `AssetTest.test_verify_reports_path_and_hash` | 通过 |
| pathlib 路径拼接；`ROBOT_ASSET`、UPSTREAM.json、T01 三处资产提交一致 | `AssetTest.test_paths_and_pinned_commit_consistent` | 通过 |
| 真实 URDF（需 `TITA_URDF`）：SHA-256、8 个受控关节、腿部限位、轮半径 0.0925、半轮距 ≈ 0.2825、需求轮速 ≈ 13.86 | `AssetTest.test_real_urdf` | 设置后通过；未设置时跳过 |
| 快照周期 0.005 / 4 / 0.02 / 20 s / 1000 步；指令范围 | `BaselineMetadataTest.test_timing_and_commands` | 通过 |
| 轮速公式；lin_vel_y 非零报错；None、NaN、低于需求、高于上限都拒绝 | `BaselineMetadataTest.test_wheel_speed` | 通过 |
| 人工运行时元数据（关节顺序故意与动作顺序不同）：平地 C=54、复杂地形 C=241 | `RuntimeComparisonTest.test_matching_runtime_returns_q_dim` | 通过 |
| 动作顺序、scale、默认角、step_dt、回合长度、指令、软限位、观测维数、缺受控关节，逐项单独出错都被报告 | `RuntimeComparisonTest.test_each_mismatch_is_reported` | 通过 |
| 填好的配置：device=cpu、C、6 个腿部软限位、对称轮速、来源标签、环境记录；device 改为 cuda 被拒绝 | `RunConfigTest.test_filled_config` | 通过 |
| 填好的 Isaac 配置驱动现有 Trainer 两轮（钩子为 E05 测试的人工模拟） | `RunConfigTest.test_trainer_runs_on_filled_config` | 通过 |
| `TitaSacHooks` 覆盖 E05 调用的全部钩子；SAC 配置只赋值 history_length=0；任务 id 不与 PPO 重名；启动入口在顶层只导入 AppLauncher | `IsaacFileStructureTest` 4 项（AST 检查） | 通过 |

另做了 3 个变异检查（临时改代码后跑测试再恢复）：钩子改名、去掉默认角比较、去掉轮速上限检查，都被测试发现。

### Windows + Isaac 验收范围（平地单环境已验证，复杂地形待验证）

- `inspect`：加载 TITA 后读到 8 个受控关节，动作顺序 / scale / 默认角与 E03 一致，软限位与 URDF 推算一致；缺失资产时报告具体路径。
- `inspect`：平地 C=54、无 scanner 组；复杂地形 C=241、scanner 187 维进入 Q；policy 组每项为单帧。
- `inspect`：`sac_config_differences` 只有 history_length 10→0，即 PPO / SAC 的奖励、指令与物理周期一致。
- `interact`：形状 [B,10,33] / [B,C]、数值有限、编号随结束递增；`--steps 1005` 时检查 `time_out` 计数，大于 0 才说明到时路径被覆盖（机器人可能先摔倒重置，1005 步不保证出现到时）。
- `train`：短训练 → 保存 → 恢复 → 继续训练 1 轮跑通。只证明流程，不代表机器人学会行走。

运行命令（仓库根目录）：

```bash
# CPU（本仓库测试用的组合，Windows 上也可以用 Isaac 环境自带的 Python 直接运行）
uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/robot_environment/test_environment.py -v
```

## Windows + NVIDIA 安装与验证步骤（PowerShell）

版本按 Isaac Lab v2.3.0 官方 pip 安装说明（`docs/source/setup/installation/pip_installation.rst`）核对：Isaac Sim 5.1.0、Python 3.11，Windows（x86_64）对应 PyTorch 2.7.0 + CUDA 12.8（文档中的 2.9.0 + cu130 只用于 Linux aarch64）。**不要照抄本仓库 CPU 测试用的 torch 2.15.0**；本仓库代码另在 torch 2.7.0（CPU 版）上跑过全部测试。显卡驱动与系统要求以 Isaac Sim 5.1 官方要求为准。以下路径以 `D:\work` 为例，可替换。

```powershell
# 0. 允许长路径（管理员 PowerShell，执行一次后重启）
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# 1. Python 3.11 环境（conda 或 python -m venv 均可）
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip

# 2. Isaac Sim 5.1.0 与 PyTorch 2.7.0（CUDA 12.8）
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Isaac Lab v2.3.0（首次运行 Isaac Sim 需按提示接受 EULA）
git clone --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git D:\work\IsaacLab
cd D:\work\IsaacLab
.\isaaclab.bat --install
.\isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py --headless   # 确认 Isaac Lab 能启动

# 4. 原版 DDT_Lab 701a49c 与锁定资产 e7b32d9（关闭换行转换，保持 URDF 字节不变）
git clone https://github.com/DDTRobot/DDT_Lab.git D:\work\DDT_Lab
cd D:\work\DDT_Lab
git checkout 701a49c6e78dfb93a8b30145f0e7723b22921c0b
git clone -c core.autocrlf=false https://github.com/DDTRobot/ddt_ros2_control.git ddt_ros2_control
git -C ddt_ros2_control checkout e7b32d9746df900680641ff2845ffb0b5ffe99ed
(Get-FileHash ddt_ros2_control\urdfs\tita_description\urdf\robot.urdf -Algorithm SHA256).Hash
# 应为 0941B83136CD5C8C89BBB33165BFF515A3A988D33319B67C8B06BFB9C71350CE
python -m pip install -e source\ddt_lab
python scripts\list_envs.py   # 原版任务 DDT-Velocity-*-Tita-v0 应出现

# 5. 本仓库（需要 baseline/ppo 快照对象，供 E03/E02/E01 静态读取原版配置）
git clone https://github.com/kissmvp8-afk/study.git D:\work\study
cd D:\work\study
git checkout main
git cat-file -t 8fe40aa0ab3afdb8998cab1357f1f795d2cabc3c   # 应输出 commit；否则 git fetch origin baseline/ppo
```

验证（在 `D:\work\study`、已激活 `env_isaaclab`）：

```powershell
# a. CPU 测试（用 Isaac 环境里的 torch 2.7.0，并核对真实 URDF）
$env:TITA_URDF = "D:\work\DDT_Lab\ddt_ros2_control\urdfs\tita_description\urdf\robot.urdf"
python -B environment_control\robot_environment\test_environment.py -v

# b. 加载环境并核对元数据（平地、复杂地形各一次）
python environment_control\robot_environment\run_sac_isaac.py --task isaac_flat  --mode inspect --headless
python environment_control\robot_environment\run_sac_isaac.py --task isaac_rough --mode inspect --headless

# c. 少量交互；第二条运行超过 1000 步回合上限（约 20 s 仿真时间），可能出现到时，但不保证：
#    机器人可能先摔倒并重置。看输出 interact.time_out 是否大于 0 再判断到时路径是否被覆盖
python environment_control\robot_environment\run_sac_isaac.py --task isaac_flat --mode interact --steps 50 --headless
python environment_control\robot_environment\run_sac_isaac.py --task isaac_flat --mode interact --steps 1005 --headless

# d. 短训练 + 保存 / 恢复（<轮速> 由用户在 [13.87, 30] 内决定，例如依据上节自行选定）
python environment_control\robot_environment\run_sac_isaac.py --task isaac_flat --mode train --wheel_speed_limit <轮速> --headless
```

请把每条命令的完整输出与输出目录中的 `metadata.json`、`summary.json` 带回，用于填写下方交接表。若 `asset_check` 为 `crlf_normalized`，说明 ddt_ros2_control 被换行转换过（内容一致，但应按第 4 步重新克隆）。

## 故障定位

加载失败先查资源路径和依赖版本；运动差异先比较任务摘要，再分析算法。

- `找不到 TITA URDF`：DDT_Lab 是否以可编辑方式安装（`pip install -e source\ddt_lab`），ddt_ros2_control 是否在 DDT_Lab 根目录下。
- `URDF 校验值 … 不符`：ddt_ros2_control 是否检出 e7b32d9。
- `SAC 配置与原版的差异超出允许范围`：原版 DDT_Lab 是否为 701a49c；列出的差异项即被改动的配置。
- `运行时元数据与原版不一致`：按列出的项逐一核对（动作顺序、默认角、周期、观测组维数）。
- `ModuleNotFoundError: isaaclab_tasks` / `ddt_lab`：Isaac Lab 或 DDT_Lab 安装没有在当前 Python 环境中完成。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [T01 版本、依赖与训练配置](../../training_validation/configuration/README.md)：`isaac_config`、`validate_configuration`（学习器设备必须为 CPU）、`export_run_configuration`
- [E02 观测接口](../observation_interface/README.md)：`validate_observations`
- [E03 八维动作定义表](../action_contract/README.md)：`load_baseline_action_contract`、`validate_action_contract`
- [E04 动作边界与初始中心](../action_bounds/README.md)：消费本模块填入的软限位与工作轮速
- [E05 终止、截断与重置前观测](../episode_lifecycle/README.md)：`EpisodeLifecycle` 与钩子协议
- [T02 训练调度](../../training_validation/training_loop/README.md)、[T03 检查点](../../training_validation/checkpoints/README.md)：train 模式调用 `Trainer`、`save_checkpoint`、`restore_checkpoint`

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | Windows 平地单环境最小闭环已验证；提前设置计数器的真实纯时间截断检查通过。复杂地形、长期训练、性能与稳定行走仍未验证。 |
| 实际文件、函数签名与调用关系 | [assets.py](assets.py)、[metadata.py](metadata.py)（导入 E02 `validate_observations`、E03 `ActionContract`/`ControlType`/`baseline_source`、T01 `isaac_config`/`validate_configuration`）、[isaac_env.py](isaac_env.py)（导入原版 `TitaFlatEnvCfg`/`TitaRoughEnvCfg` 与 Isaac Lab）、[run_sac_isaac.py](run_sac_isaac.py)（串起 E03 → E01 → E05 → T01 → T02 → T03）、[test_environment.py](test_environment.py)（5 个测试类共 14 项，复用 E05 测试的 `ScriptedSimulation`） |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；角度 rad、轮速 rad/s、长度 m、时间 s；纯数据函数无副作用（`verify_urdf` 只读文件）；`TitaSacHooks` 原地推进 Isaac 环境；启动入口写输出目录 |
| 代码注释核对 | 已核对本模块注释重点：控制周期与物理步长（metadata.py 文件头、`baseline_timing`、`apply_action_and_simulate` 循环注释）；资产版本来源（assets.py 文件头与 `ROBOT_ASSET`）；环境注册与配置沿用范围（isaac_env.py 文件头、`register_sac_tasks`、`sac_config_differences`）；奖励沿用与无超时补偿（`compute_rewards`）；轮速依据（metadata.py 文件头、`wheel_speed_ranges`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0，arm64，无 NVIDIA 显卡），Python 3.11.17：`uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/robot_environment/test_environment.py -v` → `Ran 14 tests … OK (skipped=1)`；设置 `TITA_URDF` 指向下载的 e7b32d9 URDF 后 → `Ran 14 tests … OK`。全部 24 个测试文件在 torch 2.15.0 与 torch 2.7.0（CPU）下各跑一遍均通过。pyflakes 检查 isaac_env.py / run_sac_isaac.py 无未定义名称。T06 `verify.py --docs --unit --integration`（torch 2.15.0）→ documentation passed；unit passed，216 项、跳过 1（未设置 `TITA_URDF`）；cpu_integration passed，15 项；simulation / performance not_run。`794cd62` 把本模块 `RunConfigTest.test_trainer_runs_on_filled_config` 与 E05 `TrainerIntegrationTest` 纳入 T06 集成检查后，`verify.py --integration` → passed，18 项 |
| 未验证项 | 复杂地形、长期训练、性能/搬运吞吐、稳定行走、轮速余量；提前计数截断不证明自然运行满 20 秒 |
| 代码版本 | `f3a1c743b8e9992aa77f218e1ef649d638db9796`（分支 `claude/environment-integration`），T01 配套修改 `bd99b7b87b09b5fd1124ebdd9d842547910a9831` |
| 本次改动及理由 | 首次实现 E01：锁定并校验资产、从原版读取周期与指令、用运行时值核对原版与 E03、给出工作轮速依据但不替用户定余量、SAC 配置只关闭 Isaac 历史以配合 E05 的重置前捕获、提供 Windows 可用的接入入口 |
| 阅读重点与材料 | 根/大模块/本模块 AI 规范；E02、E03、E05、T01 AI 规范；带注释的四个实现文件与测试；上述验证记录；Isaac Lab v2.3.0 `manager_based_rl_env.py` 的 `step` / `_reset_idx`。重点说明：`TitaSacHooks` 与原 `step` 的逐段对应及“为何不调用 env.step”；`required_wheel_speed` 的差速公式和它只是下限；`compare_with_baseline` 核对哪些项；`sac_config_differences` 为何只允许 history_length 一项；Windows 安装中 autocrlf 与校验值的关系 |
| 人类版 | 已有讲解，见 [README_HUMAN.md](README_HUMAN.md)；版本与验证范围以该文件记录为准 |
