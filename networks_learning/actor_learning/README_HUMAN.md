# N06 · Actor 的参数更新｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对同一次采样与新 alpha、冻结 Q 参数而保留动作梯度、编码器隔离及日志；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `0b150dc`，审核后接口修正 `bd69772` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[actor_update.py](actor_update.py)、[test_actor_update.py](test_actor_update.py) 全文；N01/N03/N04/N07 实现和测试、D04 只读归一化接口；N08 更新衔接测试；固定 PPO 策略损失/优化片段。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 4 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## 学习目标：改变当前策略，不模仿经验中的旧动作

经验回放提供过去遇到的状态；我们希望改进的是当前 Actor 在这些状态上的选择，因此重新采样动作。原 PPO 的概率比与裁剪目标不再使用，旧动作也不是模仿标签。N06 最小化：

$$L_{actor}=\operatorname{mean}[\alpha\log\pi(a\mid s)-\min(Q_1(s,a),Q_2(s,a))].$$

减 Q 使高评分动作更有利；密度项按 alpha 的权重保留分布多样性。动作及 log π 由 N03 生成，Q 为 N04 **在线**双 Q，而不是目标副本。作者 SAC 提供这一目标和冻结 Q 参数的参考做法，本项目进一步明确了编码器与温度的隔离。

## 先采样，再更新温度，再组装 Actor 损失

一次需要更新 Actor 的机会中，调用方遵守：

1. N01 从 `first.policy_history[B,10,33]` 生成无梯度 `[B,52]` 特征，编码器为 eval。
2. N03 **只采样一次**，得到相关联的 `action[B,8]` 和 `log_prob[B,1]`，两者保留 Actor 计算图。
3. N07 用该 log_prob 的 detach 版本更新 log_alpha。
4. 读取更新后的 alpha 数值，配同一次 action/log_prob 调用 `compute_actor_loss`，再 `update_actor`。

当前 `compute_actor_loss` 接收已采样结果，**不会自己再采样**，这是审核后接口的明确含义。温度更新通过 detach 使用采样值，不消耗 Actor 的计算图，所以不需要为了下一步 Actor 反传设置 `retain_graph=True`。若重新采样或沿用旧 alpha，就不再是当前衔接测试验证的顺序。

Q 侧输入为同一批状态经 D04 `normalize_q` 后的 `[B,C]`；动作是无量纲策略坐标。alpha 为普通数值，损失与 Q 使用当前奖励尺度。作者 `policy_frequency=2` 是参考起点，**本模块没有计数器决定何时更新**，正式频率由 T02 配置；N08 的一次衔接检查不是正式训练器。

## 文件与函数的责任

[actor_update.py](actor_update.py) 没有自定义网络类，使用 N03 Actor 与 N04 双 Q。[test_actor_update.py](test_actor_update.py) 用随机特征和人工峰值检查方向与隔离。

| 函数 | 输入、输出与副作用 |
|---|---|
| `frozen_parameters(module)` | 上下文管理器；保存各参数原始 requires_grad，临时关闭，退出时无论正常/异常都恢复；不切 train/eval 模式，也不清除已有 `.grad` |
| `make_actor_optimizer(actor,lr=0.001)` | 只为 N03 Actor 参数建 Adam，不含历史编码器、Q、log_alpha |
| `compute_actor_loss(critics,q_observation,action,log_prob,alpha)` | 接收对齐的 `[B,C]`、`[B,8]`、`[B,1]` 和新 alpha；临时冻结 Q 参数计算标量损失，返回连接 Actor 的计算图 |
| `update_actor(actor,optimizer,loss,max_grad_norm=1.0)` | 清 Actor 优化器梯度，反传、裁剪、step；返回 `actor_loss`，按约定只改 Actor 与优化器状态 |
| `report_policy_change(actor,features,action,log_prob)` | no_grad 下返回 `mean_std`、`entropy_estimate`、`near_bound_fraction`；不采样、不更新网络，也不自动比较前后两版策略 |

常量 `ACTOR_LEARNING_RATE=0.001`、`MAX_GRAD_NORM=1.0` 为作者默认起点；`NEAR_BOUND_THRESHOLD=0.99` 是诊断阈值。

## 冻结参数，不等于关闭整个 Q 的梯度

关键代码：

```python
with frozen_parameters(critics):
    q1, q2 = evaluate_q_pair(critics, q_observation, action)
return (alpha * log_prob - torch.min(q1, q2)).mean()
```

第一行只让 Q 的可训练权重在前向时不参与求导。第二行仍记录 Q 输出怎样随输入 action 改变，因此损失能沿 `Q→action→tanh→高斯参数→Actor` 传回。退出上下文虽恢复 Q 的标志，也不会把刚才未记录的参数梯度补回来。

如果误用 `torch.no_grad()` 包住 Q 前向，或对 action 执行 detach，评分对动作的这条梯度就断了。测试特意令 alpha=0，此时只剩 −min Q，Actor 仍应得到梯度，避免靠熵项掩盖错误。

编码器的隔离由 N01 的 no_grad/eval 特征入口实现，N06 不负责替任意输入自动隔离。alpha 应通过 N07 `current_alpha` 变成 float；把带图的 log_alpha 或其他网络参数塞入损失，会改变约定的梯度范围。优化器的参数归属也必须与上表一致。

## 日志怎样解读

`report_policy_change` 用：

```python
squashed = (action - actor.center) / actor.half_width
```

还原 tanh 坐标，然后计算所有样本、所有动作分量中 `abs(squashed)>0.99` 的比例；它不是“有任意一维贴边的样本比例”。`mean_std` 是内部高斯标准差的均值，不是物理轮速或关节角标准差；`entropy_estimate` 是所传采样的 −log π 均值。

函数会重新计算所给 features 的当前 std，但熵与贴边率使用传入的 action/log_prob。若在 step 后传入 step 前的采样，日志有不同时间来源，应明确记录，不能把它们全部称作更新后重新采样的指标。

## 测试与维护

测试辅助类 `PeakQ.__init__` 建一个检查冻结用的占位参数，把 `q1/q2` 都指向 `PeakQ.score`；`score(q_observation,action)` 返回 `−‖a−PEAK‖²+offset`，忽略状态，仅作已知最优点的人工评分器。`setup(seed)` 返回对称人工边界下的 Actor、随机 `[64,52]` 特征和 `[64,6]` Q 观测。

| 既有测试 | 验证内容 |
|---|---|
| `DirectionTest.test_moves_toward_peak` | 假 Q、学习率 0.003、150 次更新后，确定性动作到 PEAK 的平均平方距离低于初始 20%；Q 占位参数无梯度且标志恢复 |
| `IsolationTest.test_only_actor_changes` | 一次更新仅改 Actor，真实双 Q、目标、编码器含 BN 状态不变，log_alpha 无梯度 |
| `IsolationTest.test_q_gradient_reaches_actor_without_entropy_term` | alpha=0 仍可经 Q 得到 Actor 梯度 |
| `IsolationTest.test_replayed_actions_differ_from_new_samples` | 旧动作与新动作不同仍能更新，内部 std 日志接近初始 0.15 |

N08 另检查“新旧 alpha 会给出可区分损失，实际使用新值”。本文件入口为 `unittest.main()`；用户报告接口修正后的 N02/N06/N08 共 17 项再次通过，**本次没有重新运行**：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/actor_learning/test_actor_update.py -v
```

Actor 不变先查 action/log_prob 是否来自当前网络且保留图、Q 前向是否被 no_grad 包住，再查优化器参数；状态越界或 alpha 异常则回查 N03/N07。假峰值方向正确不等于真实 Q 训练稳定，实际调度、GPU、仿真和运动效果仍未验证。继续阅读 [N07](../entropy_temperature/README_HUMAN.md)、[N08](../target_updates/README_HUMAN.md)。
