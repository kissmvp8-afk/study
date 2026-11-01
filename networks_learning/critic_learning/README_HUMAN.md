# N05 · Q 的训练目标与更新｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对多步目标、末状态、动作重新采样、真终止分支和在线 Q 梯度范围；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `e43b4ea` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[critic_update.py](critic_update.py)、[test_critic_update.py](test_critic_update.py) 全文；D01/D02/D03 记录、有效片段与多步批次接口及实现片段；D04/D05 和 N01/N03/N04 全文或调用实现；E05 末状态契约。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 3 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## 学习目标：用真实奖励与尾部预测教会 Q

自举（bootstrap）是用网络预测补上尚未实际观察到的未来。D03 已累计最多 n 次动作的奖励，N05 再把当前策略在尾部的预计价值接上，形成 Q 的学习目标 y。y 含预测成分，不是真实完整未来。

原 PPO 通过 GAE 组织 V 与策略学习；这里使用 SAC 的多步、熵感知目标，训练 N04 两个在线 Q。作者 SAC 的目标形式是参考，TITA 的历史编码、混合动作与重置前观测按本项目契约接入。

## 输入与两种动作的时间含义

`batch` 是 D03 `NStepBatch`，M 为抽样起点数：

| 字段 | 形状、时间与用途 |
|---|---|
| `first.q_observation` / `first.action` | `[M,C]` / `[M,8]`，时刻 t 的状态与当时执行的旧策略动作，供在线 Q 预测 |
| `tail.next_policy_history` | `[M,10,33]`，第 k 次动作后 t+k 的真实历史，供当前编码器与 Actor |
| `tail.next_q_observation` | `[M,C]`，同一个 t+k 的 Q 观测，供目标 Q |
| `reward_sum=R` | `[M,1]`，原始奖励的累计折扣和，没有预加价值补偿 |
| `discount=γ^k`、`bootstrap_mask` | 各 `[M,1]`，实际 k 步折扣及是否允许接续未来 |

策略历史和 Q 观测分别经 D04 各自统计归一化，在线/目标 Q 共用当前 Q 统计。动作仍是无量纲策略坐标，Q 输出和目标遵循奖励数值尺度。

“尾部记录”是偏移 k−1 的转移，其 **next_*** 才是 t+k 状态。尾部新动作 a′ 必须由当前 Actor 在该状态上重新采样；不能拿 `first.action` 或末条记录的旧动作代替。旧动作的职责仅是让在线 Q 评价当时发生的选择。

## 公式与具体测试数字

设 m 为自举开关，目标是：

$$y=R+\gamma^k m\,[\min(\bar Q_1(s_{t+k},a'),\bar Q_2(s_{t+k},a'))-\alpha\log\pi(a'\mid s_{t+k})].$$

目标 Q 用横线区分；a′ 与 log π′ 来自同一次尾部采样。括号内称软价值，含熵项；必须用 N03 完整变换后的密度。alpha 从 N07 取为普通数值，不参与这里求导。

现有测试用 γ=0.9、n=3、每步奖励 1，假目标 Q 为 5 和 7，假 Actor 的 log π=−2，alpha=0.1，因此软价值为 `5−0.1×(−2)=5.2`：

| 情形 | k / mask | R | y |
|---|---|---:|---:|
| 正常走满三步 | 3 / 1 | 2.71 | `2.71+0.9³×5.2=6.5008` |
| 第二次动作后真终止 | 2 / 0 | 1.9 | 1.9 |
| 第二次动作后纯时间截断 | 2 / 1 | 1.9 | `1.9+0.9²×5.2=6.112` |

两类结束都停止累计下一轮奖励，只有纯截断继续从**重置前末状态**估值。这份状态的捕获仍由 E05 负责；N05 选用传入的 next_*，不能证明其来源正确。普通数据不足不构成合法提前结束，D02/D03 应在本模块之前拒绝。

## 文件与每个函数

[critic_update.py](critic_update.py) 没有自定义网络类，使用 N01/N03/N04 与 D03 数据；[test_critic_update.py](test_critic_update.py) 提供人工批次、假评分器与一次更新检查。

| 函数 | 输入输出、调用和副作用 |
|---|---|
| `make_critic_optimizer(critics,lr=0.001)` | 调用 N04 `list_critic_parameters`，创建只含在线双 Q 的 Adam |
| `compute_target(batch,encoder,actor,target_critics,normalizers,alpha,generator=None)` | 只为可自举行编码尾部、重新采样、评价目标 Q，返回不带梯度的 `[M,1]` y；消耗采样随机状态，不更新网络/统计 |
| `critic_loss(critics,batch,target,normalizers)` | 对 `first.action` 评价在线双 Q，返回 `(L_Q,q1,q2)`；target 应来自前一函数，接口不会自行把任意目标断开梯度 |
| `update_critics(critics,optimizer,batch,target,normalizers,max_grad_norm=1.0)` | 调用损失、清梯度、反传、裁剪并 step；按约定优化器只改在线 Q，返回损失、双 Q 均值、目标均值/最大绝对值 |

`CRITIC_LEARNING_RATE=0.001`、`MAX_GRAD_NORM=1.0` 参考作者默认值，是实验起点，不是已验证的机器人最优值。

## 为什么显式筛掉真终止，再计算尾部

实际代码先复制奖励部分，并计算哪些行可自举：

```python
target = batch.reward_sum.clone()
live = batch.bootstrap_mask.squeeze(1) > 0
```

这两行处在 `with torch.no_grad()` 内。clone 使写目标不改 D03 的 R；`live` 只选 mask=1 的正常/纯截断样本。接下来的编码、Actor 采样与目标 Q 前向均只索引 live 行，然后：

```python
soft_value = torch.min(q1, q2) - alpha * next_log_prob
target[live] = target[live] + batch.discount[live] * soft_value
```

第一行取小目标 Q 并加入熵奖励；第二行按各自 γ^k 接回目标。终止行保留原 R，不调用尾部网络。不能先算所有行再乘零：浮点运算中 `0×NaN` 仍是 NaN，会污染本应只有奖励的目标。

在线 Q 损失则为：

```python
return 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target)), q1, q2
```

两个预测分别拟合同一冻结 y，MSE 对批次取均值，两项再乘 0.5。反传只走在线双 Q；Actor、目标 Q、编码器不接这条目标梯度，编码器也保持 eval，不改变 BN。`update_critics` 报告的是 step 前计算的损失；若要比较更新后损失，必须用同一份冻结 y 重新评价，不能重采样目标后直接比较。

## 测试、历史失败记录与排错

`three_kinds_of_batches()` 经 D02/D03 构造上述三种情况。`FakeActor.__init__` 建行数日志，`FakeActor.sample_action_and_log_prob` 返回零动作、log π=−2 并记录调用行数；`FakeTargets.__init__(v1,v2)` 提供两个固定评分函数。它们只用于手算对照。

| 既有测试 | 覆盖范围 |
|---|---|
| `TargetTest.test_targets_by_case` | 三类 y、k、mask 与手算一致；只为两个非终止行采样；y 不带梯度 |
| `TargetTest.test_terminal_branch_blocks_nan` | 假目标 Q 返回 NaN 时，终止行仍是 1.9，正常/截断行确实读取 NaN；不是把所有坏值自动修好 |
| `UpdateTest.test_update_changes_only_online_q` | 固定目标下一次更新损失下降，仅在线 Q 状态变化，Actor/目标/编码器不变 |

AI 交接记录首次更新测试因人工观测约 2e4、D04 未设合适统计且学习率 0.01 而失败；改为先设置测试统计、使用默认 0.001 后通过。这是既有调试记录，不是本次重现。测试中的统计准备不能替代真实采集“每批新记录只统计一次”的规则。

入口为 `unittest.main()`；既有命令，本次未执行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/critic_learning/test_critic_update.py -v
```

Q 爆大先查 reward、目标绝对值、γ^k、mask、log π 坐标与观测尺度；重置附近异常先查 tail.next_*。三项 CPU 测试不证明真实环境稳定性，n 步偏差、调度频率、GPU 和训练效果仍待验证。下一步读 [N06 Actor 更新](../actor_learning/README_HUMAN.md)、[N07 温度](../entropy_temperature/README_HUMAN.md) 与 [N08 慢更新](../target_updates/README_HUMAN.md)。
