# N05 · Q 的训练目标与更新

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

用真实多步奖励和重新估计的后续价值教会两个 Q。

术语先说明：训练目标 y：依据真实奖励和后续估计得到的参考值，仍含预测成分。梯度：参数小幅改变对损失的影响。

## 与原 PPO 的关系

替代原 V 的 GAE 训练目标，使用论文多步与超时感知的 SAC 目标。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | D03 奖励和、有效 k、末状态和开关，当前 Actor、目标 Q、alpha。 |
| 输出 | 冻结的本次目标 y、两个 Q 的误差、在线 Q 更新及数值诊断。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `critic_update.py` | `compute_target` | 以当前 Actor 的尾部动作和较小目标 Q 构造 y |
| `critic_update.py` | `critic_loss` | 比较记录中旧动作的 Q 与 y |
| `critic_update.py` | `update_critics` | 只更新在线 Q 并记录目标量级 |
| `test_critic_update.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- y=折扣奖励和 + gamma^k × 开关 × (较小目标Q - alpha×下一动作对数密度)。
- 目标计算不反向更新 Actor、alpha、目标 Q；旧记录里的动作只用于当前 Q 误差。
- 没有末状态数据时不能偷偷用重置后状态或把超时变成失败。

## 计算含义与实现提示

先让当前 Actor 在尾部状态提出动作，再用两个目标 Q 中较小的评分接在实际奖励后面，得到 y。y 是本次用来学习的参考值，不是真实完整未来，也不在这次 Q 更新中继续求梯度。

两个在线 Q 评价经历中实际记录的旧动作，分别得到 q1 和 q2。损失可沿用作者形式：L_Q=0.5×平均[(q1−y)^2+(q2−y)^2]。它要求两个评分都向参考值靠近，只修改在线 Q 的参数。

尾部策略动作重新按当前网络生成；不能把记录中的旧动作当成当前策略在尾部选择的动作。真正终止的样本无需计算无效尾部的 Q；明确分支也避免把无效值乘零后继续传播。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 用固定数值的假评分器验证正常、终止、截断和 k<n 的目标。
- 损失下降的一次更新只改变在线 Q；目标副本此时保持不变。
- 对不存在的尾部值使用显式分支，避免 0×NaN 仍传播 NaN。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

Q 爆大先检查原始奖励量级、gamma^k、密度项及结束标记，再调整学习率。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [D03 连续多步回报](../../experience_returns/n_step_returns/README.md)
- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)
- [N04 双 Q 评分网络](../../networks_learning/q_networks/README.md)
- [N07 探索力度 alpha](../../networks_learning/entropy_temperature/README.md)
- [N08 目标 Q 的慢更新](../../networks_learning/target_updates/README.md)

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
