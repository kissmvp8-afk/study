# T02 · 采集与学习调度

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

把已验证的小模块串起来，明确每一批经历怎样产生以及何时更新各网络。

术语先说明：调度：决定各模块在什么时候运行以及运行多少次。假环境：用于验证数据流程的可控简单环境，不代表真实仿真效果。

## 与原 PPO 的关系

原 PPO 是采集近期批次后训练并清空；新流程持续积累经验，分别更新 Q、Actor、alpha、目标 Q 与辅助编码器。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/runner.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/runner.py)
- [原 PPO：scripts/np3o/train.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/train.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/runners/off_policy_runner.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/runners/off_policy_runner.py)
- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 已验证模块、完整配置、环境状态与可恢复的训练计数。 |
| 输出 | 新经历、更新后的模型、分项训练指标和进度记录。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `runner.py` | `collect_steps` | 用温和初始化的当前 Actor 采集真实数据 |
| `runner.py` | `learn_from_replay` | 按配置调度各独立更新 |
| `runner.py` | `train` | 组织采集、辅助学习、保存与评价时机 |
| `test_runner.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 经验不足时先用有界 Actor 收集，不用全范围均匀乱采代替论文初始化。
- 辅助学习频率独立设置，不能因 Q 大量回放就无意重复训练编码器。
- 统计更新、梯度边界与目标更新顺序写清楚；调度层不复制底层公式。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 使用简单假环境走通采集—存储—取样—更新，检查次数和数据来源。
- 小规模 Isaac 仿真能持续重置、训练、保存；日志没有非有限数值。
- 新采集步数与重复训练样本数分开统计，方便比较耗时与数据效率。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

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

## 当前实现与验证记录

| 内容 | 当前状态 |
|---|---|
| README 设计与职责 | 已编写，等待实施时与实际代码保持同步 |
| 实现文件和真实函数说明 | 待 Claude 实现 |
| 关键代码逐行说明 | 待代码存在后填写；不得编造行号 |
| 单元验证命令与结果 | 待实现 |
| 仿真集成结果（如适用） | 未验证 |
| 修改记录 | 本次仅建立模块契约与验收要求 |
