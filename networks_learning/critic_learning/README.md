# N05 · Q 的训练目标与更新｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [critic_update.py](critic_update.py) 与 [test_critic_update.py](test_critic_update.py)，3 项检查通过；未接入仿真或训练。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../README.md#项目维护规则)

## 实施目标与术语

用真实多步奖励和重新估计的后续价值教会两个 Q。

术语先说明：训练目标 y：依据真实奖励和后续估计得到的参考值，仍含预测成分。梯度：参数小幅改变对损失的影响。

## 与原 PPO 的关系

替代原 V 的 GAE 训练目标，使用论文多步与超时感知的 SAC 目标。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | D03 奖励和、有效 k、末状态和开关，当前 Actor、目标 Q、alpha。 |
| 输出 | 冻结的本次目标 y、两个 Q 的误差、在线 Q 更新及数值诊断。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [critic_update.py](critic_update.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| `make_critic_optimizer` | `(critics, lr=1e-3) -> Adam` | 只含在线双 Q 参数 |
| `compute_target` | `(batch: NStepBatch, encoder, actor, target_critics, normalizers, alpha, generator=None) -> y[M,1]` | `no_grad` 下计算：y = R + γ^k·mask·(min 目标 Q(s_tail, a') − alpha·log π(a')) |
| `critic_loss` | `(critics, batch, target, normalizers) -> (L_Q, q1, q2)` | 在线双 Q 评价记录里的旧动作 `first.action`；L_Q = 0.5·[MSE(q1, y) + MSE(q2, y)] |
| `update_critics` | `(critics, optimizer, batch, target, normalizers, max_grad_norm=1.0) -> dict` | 只更新在线 Q；返回损失、q1/q2 均值、目标均值与绝对值最大值 |

计算目标时的约定：
- 直接使用 D03 `NStepBatch` 的 `reward_sum`、`discount`（γ^k）、`bootstrap_mask` 和 `tail.next_*`（重置前的真实状态）。
- s_tail 经 N01 `features_from_history` 编码；a' 由当前 Actor 重新采样，不用记录里的动作。
- 尾部 Q 观测用 D04 统计归一化，与在线 Q 一致。
- alpha 以数值传入。

两种动作：记录中的旧动作只用于在线 Q 的预测；尾部动作 a' 由当前 Actor 在 s_tail 上重新生成。

真终止样本（mask = 0）走显式分支：y = R，不计算尾部 Q 与密度，避免 0 × NaN 传播。

调用方：N08 测试中的完整更新衔接检查；正式调度由 T02 实现。

## 必须成立的规则

- y=折扣奖励和 + gamma^k × 开关 × (较小目标Q - alpha×下一动作对数密度)。
- 目标计算不反向更新 Actor、alpha、目标 Q；旧记录里的动作只用于当前 Q 误差。
- 没有末状态数据时不能偷偷用重置后状态或把超时变成失败。

## 算法约定与实现提示

先让当前 Actor 在尾部状态提出动作，再用两个目标 Q 中较小的评分接在实际奖励后面，得到 y。y 是本次用来学习的参考值，不是真实完整未来，也不在这次 Q 更新中继续求梯度。

两个在线 Q 评价经历中实际记录的旧动作，分别得到 q1 和 q2。损失可沿用作者形式：L_Q=0.5×平均[(q1−y)^2+(q2−y)^2]。它要求两个评分都向参考值靠近，只修改在线 Q 的参数。

尾部策略动作重新按当前网络生成；不能把记录中的旧动作当成当前策略在尾部选择的动作。真正终止的样本无需计算无效尾部的 Q；明确分支也避免把无效值乘零后继续传播。

## 代码注释要求

共同要求见 [代码注释规范](../../README.md#代码注释规范)。

本模块特别需要注释：解释旧动作与尾部新动作的不同用途、gamma^k、终止开关、熵项符号及目标停止求梯度的原因。

## 验收案例（已执行，CPU、人工数据）

经历为人工数据，经 D02 写入读取、D03 组合；假 Actor 恒输出 log π = −2，假目标 Q 恒为 5 和 7。

| 案例 | 对应测试（[test_critic_update.py](test_critic_update.py)） | 结果 |
|---|---|---|
| 正常（k=3）、真终止（k=2, mask=0）、纯截断（k=2, mask=1）的 y 与手算一致（软价值 = min(5,7) − 0.1×(−2) = 5.2）；只为 2 个非终止样本采样尾部动作；y 不带梯度 | `TargetTest.test_targets_by_case` | 通过 |
| 假目标 Q 返回 NaN：真终止样本 y 仍为 1.9，非终止样本为 NaN（对照） | `TargetTest.test_terminal_branch_blocks_nan` | 通过 |
| 一次更新后，同一 y 下损失下降；只有在线 Q 改变，目标 Q、Actor、编码器不变；计算目标时 Actor 没有梯度 | `UpdateTest.test_update_changes_only_online_q` | 通过 |

调试记录：首次运行时 `UpdateTest` 失败，损失从 1.2e5 升到 4.2e8。原因是人工观测编码数值约 2e4，而 D04 统计未更新，Q 输入未缩放，学习率 1e-2 下一步发散。实现本身无误；改为先用这批记录更新 D04 统计、使用默认学习率 1e-3 后通过。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/critic_learning/test_critic_update.py -v
```

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

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [critic_update.py](critic_update.py)：`compute_target`、`critic_loss`、`update_critics`、`make_critic_optimizer`；依赖 N01 `features_from_history`、N04 `evaluate_q_pair` / `list_critic_parameters`，使用 D03 `NStepBatch`、D04 `ObservationNormalizers`、N03 Actor。[test_critic_update.py](test_critic_update.py)：2 个测试类共 3 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`compute_target` 无梯度、只消耗 generator 随机状态；`update_critics` 只修改在线 Q 与优化器 |
| 代码注释核对 | 已核对：旧动作与尾部新动作的用途、γ^k、终止开关与显式分支、熵项符号、目标停止求梯度（文件头、`compute_target`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/critic_learning/test_critic_update.py -v` → `Ran 3 tests … OK`（首次运行有 1 项因测试数据未归一化而失败，见上文“调试记录”）。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | 真实尺度下的目标量级与稳定性；n=1 与 n=5 的对照；更新频率（T02）；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`e43b4ea2f227ed3392b1a0438fb36f63f6711365`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N05：按规范公式直接使用 D03 的结果；真终止样本走显式分支 |
| 阅读重点与材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `critic_update.py`、`test_critic_update.py`；D03 `returns.py`；作者 `sac.py`（target_q 与 critic 损失）。重点说明两种动作的用途、显式分支与 0×NaN，以及为什么 y 不求梯度 |
| 人类版 | 已有讲解，见 [README_HUMAN.md](README_HUMAN.md)；版本与验证范围以该文件记录为准 |
