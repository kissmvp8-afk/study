# 训练与验证｜AI 实施规范

状态：框架与说明已建立；T01～T06 六个小模块都已完成 **CPU 实现**并通过单元检查（T02、T03 使用假环境）。Isaac 仿真、真实训练与性能验证都未开始，本大模块未完成运行验证。

[人类讲解](README_HUMAN.md) · [返回项目总说明](../README.md)

## 模块职责与范围

组织配置、采集和更新，保存进度，评价并导出模型，记录可复查证据。

小模块保持平级，实现、测试、接口规范和人类讲解放在同一目录。维护规则统一见 [AGENTS.md](../AGENTS.md)。

## 小模块导航与职责

| 小模块 | 负责的功能 | 状态 |
|---|---|---|
| [T01 · 版本、依赖与训练配置](configuration/README.md) | 把环境版本、算法候选设置、调试规模和正式实验设置集中记录。 | 单元验证通过（CPU；Isaac 配置由 E01 用运行时值填写，未仿真） |
| [T02 · 采集与学习调度](training_loop/README.md) | 把已验证的小模块串起来，明确每一批经历怎样产生以及何时更新各网络。 | 单元验证通过（CPU 假环境，未仿真） |
| [T03 · 训练进度保存与恢复](checkpoints/README.md) | 让中断后的实验可以有明确语义地恢复，并保留定位变化所需的信息。 | 单元验证通过（CPU 假环境，不保存仿真现场） |
| [T04 · PPO 与 SAC 对照评价](evaluation/README.md) | 用固定任务和可复查指标判断 SAC 是否学会控制，而非只看训练损失或一段视频。 | 单元验证通过（CPU，人工记录；没有真实对照结果） |
| [T05 · 模型导出与运行一致性](export_inference/README.md) | 让导出的 Actor 在相同观测与历史下，产生与训练内推理一致的动作。 | 单元验证通过（CPU，TorchScript；未经真机部署） |
| [T06 · 检查组织与维护证据](verification/README.md) | 统一收集各小模块的检查结果，并检查说明与代码是否同步。 | 单元验证通过（CPU；仿真与性能层级未执行） |

## 边界与协作

调度模块不重新实现底层动作、回报或网络公式。

环境与动作接口提供物理含义和真实交互；经验与回报整理这些记录；网络与学习使用记录改进策略；训练与验证负责调度并提供证据。跨模块交流使用根 README 的统一数据约定。

## 与原算法的关系

原 PPO 固定代码保存在独立的 baseline/ppo 分支。每个小模块的 README 给出原始来源和具体差异。作者的 SAC 是参考实现，TITA 的混合动作与历史编码仍需适配，不能只替换算法名称。

## 当前证据

以下为实现阶段已入库的验证记录；后续实验进展见[首页](../README.md#当前状态)。

六个小模块都有实现与 CPU 单元检查（需要 PyTorch），全部通过（Python 3.11.17 + torch 2.15.0）：T01 9 项、T02 9 项、T03 10 项、T04 12 项、T05 6 项、T06 6 项。

T02、T03 使用 [`FakeLocomotionEnv`](training_loop/fake_env.py)，这是确定性人工环境，**不是 Isaac 仿真**。这些检查证明“配置 → 采集 → 经验池 → 抽样学习 → 保存 → 恢复 → 导出 → 评价工具 → 检查入口”在 CPU 上走通，不代表机器人训练成功。证据分别记录在各小模块的“实现证据与文档交接”表中。

**“六个模块完成 CPU 实现”与“Isaac 仿真、训练、性能验证完成”是两件事；后者都还没有开始。** T04 只有评价工具，没有任何真实 PPO / SAC 结果；T05 只在 CPU 上验证了一致性，没有部署到真机；T06 的仿真与性能层级记为未执行。后续证据由各小模块记录，本 README 维护整体进度和跨模块问题。

已打通的 CPU 流程：

```text
T01 cpu_debug_config ──► T02 Trainer（构造时把每个配置值显式传给 E04 / D02 / D04 / N01～N08）
   collect_steps：有界 Actor 采集 → FakeLocomotionEnv.step → D01 → D04 统计（只用新数据）→ D02
   learn_from_replay：D02 → D03 → N05 → 采样一次 → N07 → N06（新 alpha，每 2 次）→ N08
   auxiliary_update：D02 → D04 → D05 → N02（按迭代计频）
   ──► T03 save_checkpoint（原子保存，可选经验池）──► restore_checkpoint（要求学习配置一致；exact_replay / model_only）
   ──► T05 export_policy（D04 → D05 → N01 → N03 确定性动作，TorchScript；输出策略坐标，E03 换算在运行端执行一次）
   ──► T04 run_evaluation / compute_metrics / write_report（证据不足时不发布性能结论）
T06 verify.py：文档 / 单元 / CPU 集成三层可执行；仿真 / 性能两层为 not_run
```

跨模块接口变化：D02 新增 `ReplayBuffer.state_dict` / `load_state_dict`，供 T03 保存经验池；写入、连续性与抽样算法没有改动。D02 人类版已同步保存恢复接口与验证说明（核对版本 a07d7b2，本轮不再润色）。T04～T06 没有修改任何已有模块的接口；T04 `run_evaluation` 使用 T02 的环境接口，T05 复用 N01 / N03 的推理函数，T06 只运行已有测试。

环境接入带来的变化（分支 `claude/environment-integration`）：T01 `isaac_config` 的学习器设备由 `cuda` 改为 `cpu`，`validate_configuration` 拒绝其他设备（学习器只实现了 CPU 版本），资产提交填入 e7b32d9（`bd99b7b`）；`EnvObservation` / `EnvStep` 的定义移到 E05 `lifecycle.py`，T02 `fake_env.py` 改为导入，字段与语义不变（`3d4c995`）；E01 `run_sac_isaac.py --mode train` 用真实环境调用 T02 `Trainer` 与 T03 保存恢复（尚未在 Isaac 中运行）。T01、T02 与本大模块人类版已同步（核对版本 a12e69a，本轮不再润色）。T06 在 `794cd62` 把 E05 `TrainerIntegrationTest`、E01 `RunConfigTest.test_trainer_runs_on_filled_config` 纳入 CPU 集成检查（人工钩子），仿真层说明改为“接入代码已实现，真实仿真未验证”，T06 人类版已同步新增集成检查与证据边界（a12e69a）。

T03 审核修复：恢复时学习配置必须与检查点一致（只有 name、sources、environment 可以不同），否则在修改 Trainer 之前报错，并提示用 `load_checkpoint_config` 重建。“换配置续训”不在本版本范围内。

假环境与真实环境的边界：`FakeLocomotionEnv` 的 `reset` / `step` 接口即 Trainer 与环境的约定。E01 / E02 / E05 已实现同一接口，下一步需在真实 Isaac Lab 中验证：
- 重置前保存真实末状态；
- 维护 episode_id、step_index、history_length 与重置填充规则；
- 真终止优先；
- 由 E02 确定 C 与速度标签列。

Isaac 配置目前仍有 5 个未确定项（见 T01）。
