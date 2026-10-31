# N06 · Actor 的参数更新｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [actor_update.py](actor_update.py) 与 [test_actor_update.py](test_actor_update.py)，4 项检查通过；未接入仿真或训练。

[人类版（待 Codex 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

利用当前 Q 评价和探索要求，改进当前策略提出的动作。

术语先说明：冻结参数：本次不修改这些参数；它不等于禁止计算输入动作怎样影响输出评分。

## 与原 PPO 的关系

替代 PPO 的 ratio/裁剪目标；从历史观测重新采样当前策略动作，不能把旧动作当成模仿标签。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 回放中的状态、当前 Actor、在线双 Q、alpha、隔离梯度的历史特征。 |
| 输出 | Actor 目标、动作分布变化和更新后的动作网络参数。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [actor_update.py](actor_update.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| `frozen_parameters` | 上下文管理器 `(module)` | 临时关闭参数的 `requires_grad`，退出时恢复；输入张量的梯度不受影响 |
| `make_actor_optimizer` | `(actor, lr=1e-3) -> Adam` | 只含 N03 Actor 参数（编码器、Q、log_alpha 都不在其中） |
| `compute_actor_loss` | `(critics, q_observation[B,C], action[B,8], log_prob[B,1], alpha) -> L` | 接收调用方用 `actor.sample_action_and_log_prob(features)` 在回放状态上**同一次采样**的 a 与 log π；L = 平均 [alpha·log π − min(Q1, Q2)]；前向时冻结 Q 参数，保留 Q 对动作的梯度 |
| `update_actor` | `(actor, optimizer, loss, max_grad_norm=1.0) -> dict` | 反传并更新 Actor |
| `report_policy_change` | `(actor, features, action, log_prob) -> dict` | 只读：平均内部 std、熵估计 −平均 log π、贴边比例（|tanh u| > 0.99） |

梯度边界：
- `features` 由 N01 在 `no_grad`、eval 下得到，不更新编码器参数或 BN 统计。
- alpha 以数值传入，不对 log_alpha 求导。
- 采样与更新顺序（作者 `sac.py`）：采样一次 → N07 用这次的 log π 更新 alpha → 用**更新后的** alpha 和同一次采样调用 `compute_actor_loss` → `update_actor`。采样放在调用方，正是为了让 Actor 损失能在温度更新之后组装。
- 作者每 2 次 Critic 更新做一次 Actor 更新（`policy_frequency=2`），具体频率由 T02 决定。

调用方：N08 测试中的完整更新衔接检查。

## 必须成立的规则

- 冻结 Q 参数，但保留 Q 对输入动作的梯度。
- Actor 更新不改变辅助编码器参数或其 BatchNorm 统计。
- 固定 alpha 的梯度边界；不把 Actor 损失拿去更新温度。

## 算法约定与实现提示

在回放提供的状态上重新采样动作，最小化 L_actor=平均[alpha×log_prob−min(Q1,Q2)]。log_prob 是这次新动作的对数密度；Q1、Q2 是在线评分。减去 Q 使高评分动作受到鼓励，密度项按 alpha 的力度保留多样性。

旧经历提供状态，新动作必须由当前 Actor 计算。这里不需要 PPO 的新旧策略概率比、优势值或裁剪公式。Q 的参数暂时冻结，但“动作改变会怎样影响 Q”这条计算关系必须保留，否则 Actor 学不到动作方向。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释为何重新采样、Actor 损失的两个方向、冻结 Q 参数但保留动作梯度，以及编码器/alpha 的隔离。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，CPU、人工数据）

Actor 输入特征与 Q 观测为随机人工数据；动作边界为人工对称区间 [−1, 1]。

| 案例 | 对应测试（[test_actor_update.py](test_actor_update.py)） | 结果 |
|---|---|---|
| 已知峰值的假 Q（−‖a − a*‖²）：150 次更新后确定性动作到 a* 的平方距离降到初始的 20% 以下；Q 的占位参数没有梯度，退出后恢复 `requires_grad` | `DirectionTest.test_moves_toward_peak` | 通过 |
| 只有 Actor 改变；真实 TwinQ、目标 Q、编码器（含 BN 统计）不变；Q 参数无梯度；log_alpha 无梯度 | `IsolationTest.test_only_actor_changes` | 通过 |
| alpha = 0 时损失只剩 −min Q，Actor 仍得到梯度，说明 Q 对动作的梯度没有被切断 | `IsolationTest.test_q_gradient_reaches_actor_without_entropy_term` | 通过 |
| 回放动作与新采样动作不同时可正常更新；报告的平均 std ≈ 0.15 | `IsolationTest.test_replayed_actions_differ_from_new_samples` | 通过 |

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/actor_learning/test_actor_update.py -v
```

## 故障定位

Actor 完全不变时查是否把整个 Q 计算都禁止求梯度；旧行为被照抄时查误用模仿目标。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)
- [N04 双 Q 评分网络](../../networks_learning/q_networks/README.md)
- [N07 探索力度 alpha](../../networks_learning/entropy_temperature/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [actor_update.py](actor_update.py)：`compute_actor_loss`、`update_actor`、`report_policy_change`、`make_actor_optimizer`、`frozen_parameters`；依赖 N04 `evaluate_q_pair`，使用 N03 Actor。[test_actor_update.py](test_actor_update.py)：2 个测试类共 4 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`update_actor` 只修改 Actor 参数与优化器；`compute_actor_loss` 临时改变并恢复 Q 参数的 `requires_grad` |
| 代码注释核对 | 已核对：重新采样的原因、损失的两个方向、冻结 Q 参数但保留动作梯度、编码器与 alpha 的隔离（文件头、`compute_actor_loss`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/actor_learning/test_actor_update.py -v` → `Ran 4 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | 真实 Q 下的训练方向与稳定性；Actor 更新频率（T02）；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`0b150dc5dc7dfb410485bf6a51197926a7021e7a`，审核后修改接口 `bd69772f7e9cb2f6207d025b5c7ce6d906184d13`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N06：沿用作者“冻结 Q 参数、保留动作梯度”的做法；编码器通过 N01 的 no_grad 特征隔离 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `actor_update.py`、`test_actor_update.py`；作者 `sac.py`（actor 更新与冻结 critic）。建议 Codex 重点解释为什么重新采样、`frozen_parameters` 如何保留动作梯度，以及 alpha=0 的对照测试 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文。人类版面向有 Python 基础的读者 |
| 技术初稿 / 语言润色 | 待 Codex 技术初稿；其后由 Gemini 润色同一份人类版 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
