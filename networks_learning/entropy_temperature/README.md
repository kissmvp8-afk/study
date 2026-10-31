# N07 · 探索力度 alpha

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

调节策略中鼓励动作多样性的力度，避免把探索简单理解成越随机越好。

术语先说明：熵：衡量动作分布多样性的量。alpha：训练中鼓励这种多样性的权重，不直接表示电机转动幅度。

## 与原 PPO 的关系

原 PPO 有固定熵系数；SAC 在策略与后续价值目标中使用熵，并可自动学习其权重。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 当前策略的对数密度、配置的目标熵、当前 log_alpha。 |
| 输出 | 正数 alpha、温度损失和实际熵相对目标的偏差。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `temperature.py` | `current_alpha` | 由内部参数得到正数探索权重 |
| `temperature.py` | `temperature_loss` | 以固定的策略密度信号训练探索权重 |
| `temperature.py` | `update_temperature` | 只更新 alpha 的内部参数 |
| `test_temperature.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 连续动作的熵数值可以为负，不据此判为错误。
- 目标熵与密度使用的动作坐标及缩放约定一致；保留作者配置作为显式实验起点。
- 此更新不能反向修改 Actor 或 Q。

## 计算含义与实现提示

使用 log_alpha 表示可训练参数，再取 alpha=exp(log_alpha)，自然保证 alpha 为正。作者的目标是 L_alpha=−平均[log_alpha×(log_prob+target_entropy)]；括号内数值在这次更新中当作固定数据。

直观检查：估计熵低于目标时增加 alpha，让保留多样性的要求更强；高于目标时减小 alpha。作者按 target_entropy=−target_entropy_scale×动作维数配置目标；TITA 的维数是 8。动作缩放改变密度的坐标，因此把所用尺度与目标一起记录，不能省略缩放修正后仍照搬所有数值。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 已知熵低于/高于目标的样例，验证 alpha 的调整方向。
- alpha 始终为正且数值有限；保存恢复后结果一致。
- 改变动作尺度时检查密度修正与目标熵约定，不能隐式改变探索语义。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

alpha 持续异常增长时先检查密度、目标熵和动作变换，再扩大或缩小学习率。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)

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
