# T02 · 采集与学习调度｜AI 实施规范

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[人类版（待 Gemini 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

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

实际实现和验证代码放在本目录，与本目录两份 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

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

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释采集/回放计数、预热动作来源、各优化器顺序与频率，以及模式、统计和重置数据的交接。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Gemini 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用；填写下方交接表，将人类版元数据标为待 Gemini 同步，保留其讲解正文。
6. 记录证据后更新实现状态；按根 GEMINI.md 把同一版本的规范、带注释代码、验证结果和变更说明交给 Gemini。人类版尚未同步和代码尚未验证须分别记录。

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

## 实现证据与 Gemini 交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已编写；实现时同步实际接口 |
| 实现状态 | 待 Claude 实现 |
| 实际文件、函数签名与调用关系 | 尚无实现；上方仅为计划 |
| 参数/返回值、形状、单位、异常和副作用 | 按输入输出约定实施后补充真实接口记录 |
| 代码注释核对 | 待代码存在后检查本模块注释重点 |
| 验证命令、环境与结果 | 未执行；尚无验证程序 |
| 未验证项 | 仿真、训练与性能均未验证 |
| 代码版本 | 尚无实现提交；不得填入虚构提交号 |
| 本次改动及理由 | 仅建立 AI 规范与人类版分工，未改算法实现 |
| Gemini 所需材料 | 根/大模块/本模块 AI 规范、依赖契约、实际带注释代码、验证文件与结果、上述代码版本和变更摘要 |
| 人类版 | 待 Gemini 编写；实际代码讲解待实现后补充 |

Claude 每次改动都需更新此表，列出应由 Gemini 重点解释的实际文件、函数和计算片段。实现不存在时，Gemini 只能写设计导读；实现后按 [GEMINI.md](../../GEMINI.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
