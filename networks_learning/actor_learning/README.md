# N06 · Actor 的参数更新

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

利用当前 Q 评价和探索要求，改进当前策略提出的动作。

术语先说明：冻结参数：本次不修改这些参数；它不等于禁止计算输入动作怎样影响输出评分。

## 与原 PPO 的关系

替代 PPO 的 ratio/裁剪目标；从历史观测重新采样当前策略动作，不能把旧动作当成模仿标签。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 回放中的状态、当前 Actor、在线双 Q、alpha、隔离梯度的历史特征。 |
| 输出 | Actor 目标、动作分布变化和更新后的动作网络参数。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `actor_update.py` | `compute_actor_loss` | 计算 alpha×log_prob - min(Q1,Q2) 的批次平均 |
| `actor_update.py` | `update_actor` | 利用 Q 对动作的变化趋势更新 Actor |
| `actor_update.py` | `report_policy_change` | 报告随机程度和动作贴边比例 |
| `test_actor_update.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 冻结 Q 参数，但保留 Q 对输入动作的梯度。
- Actor 更新不改变辅助编码器参数或其 BatchNorm 统计。
- 固定 alpha 的梯度边界；不把 Actor 损失拿去更新温度。

## 计算含义与实现提示

在回放提供的状态上重新采样动作，最小化 L_actor=平均[alpha×log_prob−min(Q1,Q2)]。log_prob 是这次新动作的对数密度；Q1、Q2 是在线评分。减去 Q 使高评分动作受到鼓励，密度项按 alpha 的力度保留多样性。

旧经历提供状态，新动作必须由当前 Actor 计算。这里不需要 PPO 的新旧策略概率比、优势值或裁剪公式。Q 的参数暂时冻结，但“动作改变会怎样影响 Q”这条计算关系必须保留，否则 Actor 学不到动作方向。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 构造已知峰值的假 Q，验证 Actor 更新有正确方向，而不是只检查损失有数值。
- Actor 更新后 Q、目标 Q、编码器的状态保持约定不变。
- 回放动作与新采样动作不同也可正常更新。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

Actor 完全不变时查是否把整个 Q 计算都禁止求梯度；旧行为被照抄时查误用模仿目标。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)
- [N04 双 Q 评分网络](../../networks_learning/q_networks/README.md)
- [N07 探索力度 alpha](../../networks_learning/entropy_temperature/README.md)

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
