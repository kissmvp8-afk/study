# 环境与动作接口｜AI 实施规范

状态：五个小模块均已实现并通过 CPU 单元验证；E01 的 Isaac 接入代码（`isaac_env.py`、`run_sac_isaac.py`）尚未在仿真中运行，等待 Windows + NVIDIA 验证；本大模块未完成运行验证。

[人类讲解](README_HUMAN.md) · [返回项目总说明](../README.md)

## 模块职责与范围

负责机器人任务、观测与动作的物理含义，以及一轮结束前后的数据。

小模块保持平级，实现、测试、接口规范和人类讲解放在同一目录。维护规则统一见 [AGENTS.md](../AGENTS.md)。

## 小模块导航与职责

| 小模块 | 负责的功能 | 状态 |
|---|---|---|
| [E01 · 机器人与任务环境](robot_environment/README.md) | 加载 TITA 并提供相同的平地/复杂地形任务，使 PPO 与 SAC 控制的是同一个对象。 | CPU 单元验证通过（Isaac 接入未运行） |
| [E02 · 观测接口](observation_interface/README.md) | 明确 Actor 与 Q 分别看到什么，让每一个观测数字的顺序、尺度和来源可查。 | CPU 单元验证通过（未仿真） |
| [E03 · 八维动作定义表](action_contract/README.md) | 给八个策略输出逐一确定控制对象、类型、单位、默认值和缩放，这是第一个建议实施的小模块。 | 单元验证通过（纯数据，未仿真） |
| [E04 · 动作边界与初始中心](action_bounds/README.md) | 按每个动作的物理含义计算策略边界，并让初始动作映射到合适的姿态。 | 单元验证通过（纯数据，人工限位，未仿真） |
| [E05 · 终止、截断与重置前观测](episode_lifecycle/README.md) | 区分摔倒与人为时间结束，并记录动作真正到达的最后状态。 | CPU 单元验证通过（人工钩子，未仿真） |

## 边界与协作

不实现 Q/Actor 的参数更新，也不替经验池计算回报。

环境与动作接口提供物理含义和真实交互；经验与回报整理这些记录；网络与学习使用记录改进策略；训练与验证负责调度并提供证据。跨模块交流使用根 README 的统一数据约定。

## 与原算法的关系

原 PPO 固定代码保存在独立的 baseline/ppo 分支。每个小模块的 README 给出原始来源和具体差异。作者的 SAC 是参考实现，TITA 的混合动作与历史编码仍需适配，不能只替换算法名称。

## 当前证据

以下为实现阶段已入库的验证记录；后续实验进展见[首页](../README.md#当前状态)。

E03 有实现文件与 18 项无仿真单元检查，证据记录在 [E03 交接表](action_contract/README.md#实现证据与文档交接)；E04 有实现文件与 14 项无仿真单元检查，使用人工软限位与工作轮速，证据记录在 [E04 交接表](action_bounds/README.md#实现证据与文档交接)；E01 有 14 项 CPU 检查（[E01 交接表](robot_environment/README.md#实现证据与文档交接)），E02 有 8 项（[E02 交接表](observation_interface/README.md#实现证据与文档交接)），E05 有 10 项（[E05 交接表](episode_lifecycle/README.md#实现证据与文档交接)）。整个大模块尚无仿真运行、训练曲线或性能结果。后续证据由各小模块记录，本 README 维护整体进度和跨模块问题。

接入关系：E01 `run_sac_isaac.py` 依次调用 E03 动作表 → E01 资产校验与元数据核对（`compare_with_baseline` 用 `action_manager` 实际动作项核对 E03 顺序、scale 与默认值，用运行时 `soft_joint_pos_limits` 核对 URDF 推算）→ E02 观测组校验 → E05 `EpisodeLifecycle(TitaSacHooks)` → T01 填配置 → T02 / T03。SAC 配置只把 policy 组的 Isaac 历史长度改为 0，10 帧历史由 E02 维护，以便 E05 在重置前取得末状态而不重复推进。

跨模块待办：以上运行时核对尚未在 Isaac 中执行，需在 Windows + NVIDIA 上按 [E01 的步骤](robot_environment/README.md#windows--nvidia-安装与验证步骤powershell) 验证；工作轮速由用户在 [13.87, 30] rad/s 内决定，不能直接用模型的 30 rad/s 当工作范围而不加说明；学习器只实现了 CPU 版本。
