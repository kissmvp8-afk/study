# 环境与动作接口｜人类阅读版

[AI 实施规范](README.md) · [项目人类版](../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色；先完成小模块及动作说明同步，再写本概览 |
| 技术核对状态 | 2026-10-31，Codex 核对五模块分工、真实入口、历史/动作/末状态与训练接口；不开展全面代码审核 |
| 语言润色状态 | 本轮由 Codex 完成，不再交 Gemini |
| 已核对代码版本 | `a12e69a38c6b9b3a236473a2563e255a9a24e761`（claude/environment-integration，PR #8），fetch 后 git rev-parse HEAD 与远端及已审查版本一致 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根规则及五模块 AI README；E01 四实现/测试、E02/E05 实现/测试、E03/E04 实现与既有讲解，T01/T02/T03/T06 调用及必要 D/N 契约；详见文末 |
| 验证依据与限制 | 引用 E01/E02/E03/E04/E05 的 14/8/18/14/10 项既有 CPU 记录；E01 一项真实 URDF 检查未设变量时跳过。真实 Isaac 钩子、GPU、训练效果与性能未验证。本轮只运行文档检查，未重跑算法测试 |

## 同一台机器人，保持明确的数据含义

把 PPO 换为 SAC（软演员—评论家算法），需要改变学习方法，同时保持“每项观测是什么、每个动作控制谁、何时结束一轮”的含义可追踪。Actor 是生成动作的策略网络，Q 是估计状态和指定动作未来回报的网络；它们依赖本大模块输出，不能自行猜关节顺序或把重置状态当作末状态。

五个小模块代码均已实现，有 CPU 层证据。E01 也提供了真正调用 Isaac 的入口；目前尚未运行它。人工钩子是按计划产生状态和事件的测试对象，只能检查接口衔接，不能证明机器人接触、动力学或训练效果。

## 五个模块分别负责什么

| 小模块 | 入口与输出 | 下一步消费者 |
|---|---|---|
| [E01 机器人与任务环境](robot_environment/README_HUMAN.md) | 锁定/校验资产、继承并注册任务、读取运行时元数据、提供 TitaSacHooks 和启动入口 | E02 字段核对、E04 范围、E05 生命周期、T01/T02/T03 |
| [E02 观测接口](observation_interface/README_HUMAN.md) | 字段布局、单帧 `[B,33]`、Q `[B,C]`、策略历史 `[B,10,33]` 与长度 | E05 整理每步结果，D04/D05/N01 处理历史，N04 使用 Q 信息 |
| [E03 八维动作定义表](action_contract/README_HUMAN.md) | 索引到关节/位置或速度/默认值/scale/clip 的有序表 | E04 反推边界，D01/Q/导出统一动作坐标 |
| [E04 动作边界与初始中心](action_bounds/README_HUMAN.md) | 每维策略下上界、中心 b、半宽 c、初始内部中心 u0 | N03 有界动作映射与初始化，T02 构造策略 |
| [E05 终止、截断与重置前观测](episode_lifecycle/README_HUMAN.md) | EnvObservation/EnvStep，末状态快照、重置后的交互状态、编号 | T02 建 D01 记录，T03 恢复编号，T04 评价 |

建议先读 E03→E04 了解动作，再读 E02→E05 了解观测时间线，最后用 E01 把真实入口串起来。本概览只连接接口，类和函数逐项说明放在小模块中。

## Actor 看到的历史与 Q 看到的信息

B 表示并行环境数。策略一帧 33 维包括角速度、重力方向、指令、关节相对位置/速度和上一步动作；不含真实身体线速度。E02 保留 10 帧，从旧到新，最后一帧是当前。D05 切最近五帧 t−4…t 供编码器，辅助另一窗口为 t−5…t−1，两者合计六个时刻。

N01 从归一化历史估计速度 3 维和隐藏特征 16 维，再拼当前帧 33 维，Actor 输入为 `[速度3|隐藏16|当前33]` 共 52 维。初始历史重复填满首帧，但有效长度仍为 1；重复内容不能当成已经过了十个物理时刻。

Q 输入状态信息按 `[critic36|priv18|scanner187（仅复杂地形）]` 拼接，平地 C=54、复杂地形 C=241。真实身体线速度、接触/增益、扫描只进入 Q 或速度监督；Q 另外接收指定策略动作。环境 scale 已经作用在字段上，例如速度标签×2.0；D04 后续归一化是另一层操作。关节观测按 articulation 顺序，动作按 E03 顺序，两种顺序不能混用。

SAC 配置把 Isaac 内部 policy 历史从 10 改成 0，由 E02 维护历史，以便 E05 在 reset 前后各算观测而不重复推进。关闭的是 Isaac 的历史维护，不是取消 Actor 的 10 帧输入。

## 一步的两份状态怎样进入经验池

```text
策略动作 a → E01 process_action（一次物理映射）→ 4 次物理步
  → 终止/到时 → 原始奖励 → 重置前观测 clone
  → 结束环境 reset → 更新指令/interval 事件 → 下一次交互观测
  → E05 按环境选择 next_*，维护 E02 历史与 episode/step 编号
  → T02：D01 make_transition → D04 新状态统计 → D02 append
```

未结束环境的 next_* 与下一次交互观测都取最后一次计算；结束环境的 next_* 取 reset 前快照，交互观测则是新回合首帧。真终止优先于同一步到时；纯截断虽结束当前记录段，D03/N05 仍用旧回合末状态接未来估值。

E05 测试中，环境 1 从 −101 每步增加 1.5，第 3 步终止，经历末帧是 −96.5、长度 4；下一次交互帧是 −101、长度 1、轮次加一、步号归零。这些人工数便于发现混用，但不是 TITA 传感器值。

## 动作与资产已经接到哪里

策略动作 `[B,8]` 没有物理单位，位置项索引为 0/1/2/4/5/6，轮速项为 3/7。腿部物理目标 `q0+0.25a`（rad），轮速 `5a`（rad/s），再做原配置的目标 clip。E04 的策略边界来自六腿软限位和单独工作轮速，不是用宽的 clip 代替工作范围。N03 输出 a，E05 搬运它，E01/Isaac 动作项只换算一次。

资产已锁定为 ddt_ros2_control `e7b32d9746df900680641ff2845ffb0b5ffe99ed`。E01 验证 URDF 摘要与网格目录存在，从真实 articulation 读取软限位、关节与观测元数据，再与固定 PPO 配置及 URDF 推算核对；这些真实运行时检查仍待执行。

工作轮速没有默认值。默认姿态下轮半径约 0.0925 m、半轮距约 0.28253 m，最大指令的无滑动需求约 13.87 rad/s，模型上限参考为 30 rad/s。20 rad/s 只是建议的首次仿真调试值，必须显式传入，不是已验证参数；姿态、地形与打滑所需余量仍待验证。

## 从环境核对到训练、恢复

E01 `inspect` 创建环境、核对资产/配置差异/动作/周期/观测，输出 metadata；`interact` 在此基础上经 E05 执行零策略动作，记录形状、有限性和结束次数；`train` 再填 T01 配置，交给 T02 采集学习、T03 保存/恢复并续一轮。

当前学习器 device 只能 cpu；仿真设备由 E01 `--device` 单独选择，E05 负责动作、两份观测与结束信息的逐张量搬运。GPU 仿真不代表 GPU 学习器，吞吐与搬运成本尚未测量。默认 16 环境、每轮 8 步、容量 200、3 轮只是接入调试规模。

恢复保存的是学习状态与可选经验池，环境现场不保存。恢复时学习配置必须一致，E05 以保存轮次号+1 reset，防止旧回合与新采集拼接。CPU 人工钩子已经检查这条链；真实 train 入口尚无执行证据。

## 维护与排错入口

| 现象 | 先看 |
|---|---|
| 加载不了机器人 | E01 资产路径、摘要、meshes、DDT_Lab 安装和锁定版本 |
| 维数对但行为不对 | E02 逐字段顺序/缩放、articulation 与动作两种顺序 |
| 末状态突然跳成初始状态 | E05 clone 时机、结束掩码、next_* 与 observation 来源 |
| 历史重复当前帧 | E01 是否关闭 Isaac 历史，E02/E05 是否只提交一次 |
| 动作固定倍率错误 | E03 scale、是否先换算又送入动作项、E04 位置/速度选错字典 |
| 截断后回报异常 | 真终止优先、D01 mask、D03 尾状态、是否误加 PPO 补偿 |
| CPU 检查全过仍无法仿真 | E01 实际依赖、私有接口/运行时核对；结构测试不执行 Isaac |

升级 Isaac Lab 时，优先重核 step/reset 顺序、观测历史和私有 `_reset_idx/_sim_step_counter`；改变资产或动作/观测字段时，按 E01→E02/E03/E04→E05→D/N/T 追踪下游，再更新对应说明与证据。

## 阅读材料与验证依据

本轮实际阅读根 AGENTS.md、CODEX.md、README.md、UPSTREAM.json、环境大模块及五个小模块 AI README；E01 四个实现与 `test_environment.py`、E02 `observations.py/test_observations.py`、E05 `lifecycle.py/test_lifecycle.py`；E03/E04 既有实现、测试和人类版的未变部分与当前调用关系；T01 配置/测试、T02 runner/fake_env、T03 checkpoint、T06 verify/test_verify 相关代码与差异。必要 D01/D03/D04/D05 和 N01 等接口沿用同版本的已核对说明并对照实现；本轮不重新全面审核 D/N。

固定 PPO 配置从本地 `8fe40aa0ab3afdb8998cab1357f1f795d2cabc3c` 读取，重点看 rough/flat 环境、观测和动作。另阅读官方 Isaac Lab v2.3.0 的 [ManagerBasedRLEnv.step](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)、[CircularBuffer](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py) 和 [ObservationManager](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/managers/observation_manager.py) 的相关源码；这是源码阅读，不是安装或仿真验证。真实 URDF 的几何与历史测试结果引用交接记录，本轮未下载核验该资产。

现有 CPU 结果来自各 AI 交接表：E01 14 项（真实资产可选）、E02 8、E03 18、E04 14、E05 10；T06 新增 E05/E01 人工钩子集成后共 18 个测试通过。没有真实 Isaac/GPU/训练效果证据。本轮实际文档检查与未解决项统一见[根目录记录](../README_HUMAN.md#阅读材料与验证依据)。
