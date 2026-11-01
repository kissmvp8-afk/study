# N08 · 目标 Q 的慢更新｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对参数插值、缓冲复制、tau/频率、目标隔离及一次更新衔接顺序；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `7f671f3`，审核后衔接顺序/缓冲说明修正 `bd69772` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[targets.py](targets.py)、[test_targets.py](test_targets.py) 全文；N01～N07 实现与测试；E03/E04 动作和人工边界、D01～D05 契约与关键实现；T02 调度契约。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 5 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## 学习目标：让参考网络逐渐跟随，而不是同时剧烈变化

N05 训练在线双 Q 时，目标 y 又依赖 Q 的后续评分。若参考网络每次立即随在线值变化，目标也会快速移动。N04 因此先复制出独立目标双 Q，N08 再让目标参数缓慢跟随在线参数，这种参数插值也称 Polyak 平均。

平均的是每个参数，不是两个 Q 的评分，也不把在线 Q 反向改成目标值。原 PPO 的路径没有这套 SAC 目标双 Q；本模块参考作者默认 `tau=0.005`、每次 Critic 更新后跟随一次，作为实验起点。

## 输入、输出与一次数值更新

输入是结构对应的在线/目标模块、无单位比例 tau，以及调度计数。操作原地修改目标参数与缓冲，返回 None；在线网络不变。目标应已经由 N04 `initialize_targets` 复制、冻结梯度并设为 eval。

$$\theta_{target}^{new}=(1-\tau)\theta_{target}^{old}+\tau\theta_{online}.$$

例如单个参数目标旧值 2、在线值 10，tau=0.3 后为 `0.7×2+0.3×10=4.4`。tau=1 复制在线参数，tau=0 保留目标参数；默认0.005时本例只变为2.04。tau 与频率共同决定跟随速度，单记一个数不足以复现实验。

## 文件与每个接口

[targets.py](targets.py) 提供检查和更新函数；[test_targets.py](test_targets.py) 既测小网络公式，也串联 D/N 模块执行一次人工更新。

| 接口 | 输入输出、调用和副作用 |
|---|---|
| `TargetUpdateError` | 结构、共享存储、tau 或目标状态不符合约定时的 `ValueError` 子类 |
| `validate_target_structure(online,target)` | 参数/缓冲分别按对应名称核对键、形状、dtype 与存储；不对应或对应项共享时报错，只检查不更新 |
| `soft_update(online,target,tau=0.005)` | 检查 tau∈[0,1] 与结构，然后在 no_grad 下插值所有目标参数、复制所有目标缓冲 |
| `soft_update_due(critic_update_count,update_every=1)` | 按从1开始的在线 Q 更新次数取模，返回 bool；`update_every<1` 报错；不维护计数器或自动执行更新 |
| `check_target_state(online,target,optimizers=())` | 先查结构，再确认目标参数不求梯度、不在所给优化器内；返回参数张量个数与 `buffers_copied_verbatim` 名称；不替调用方切 eval 或修改优化器 |

`TAU=0.005`、`UPDATE_EVERY=1` 是默认值。每两次跟随的例子，输入计数1、2、3、4，`soft_update_due(...,2)` 返回 False、True、False、True；计数含义是 Critic 更新次数，不是仿真步或 Actor 步，正式调度由 T02 维护。

## 参数与缓冲为什么分开处理

参数通过梯度学习；缓冲是网络保存但不由优化器训练的状态，例如 BN 的运行均值、方差和整数批次数。真实代码为：

```python
with torch.no_grad():
    for name, parameter in target.named_parameters():
        parameter.lerp_(online_params[name], tau)
    for name, buffer in target.named_buffers():
        buffer.copy_(online_buffers[name])
```

第一行避免生成优化计算图。`lerp_` 按名称把每个目标参数向对应在线值移动 tau 比例；遍历 named_parameters 会包含注册的子模块，未来若加入 Q 地形编码层，须将它注册为 Q 的组成部分。缓冲则直接 copy，不乘 tau。

因此 **tau=0 只代表参数不变，缓冲仍会复制**。若在线 running_mean=4、批次数=7，即使 tau=0，目标对应缓冲也变为4和7。缓冲复制不是所有算法唯一正确的选择；浮点统计可以另定平均规则，当前项目统一复制，而整数计数不适合插值。

当前 N04 双 Q 没有 BN 等缓冲，D04 归一化统计位于外部，在线/目标共用同一份，不在这里做慢更新。以后加入新层应核对 `check_target_state` 列出的缓冲，明确当前复制约定是否仍适用。目标必须保持存储独立、冻结梯度、不在优化器中，不能用 `target=online` 代替复制。

## 一次人工更新串联了哪些模块

`UpdateCycleTest.test_one_cycle_changes_only_owned_state` 的数据来自 D02 `make_records`：两个环境各写12次，0号在 step_index=7 真终止后换轮，1号在 step_index=11 纯截断；奖励为1与0.5。每批先更新一次 D04，再写 D02，抽32个起点、每段3条，经 D03 得 n=3、γ=0.99 的训练批次。

动作表读取固定 E03 快照，边界与轮速来自 E04 人工数据。速度标签使用原始 Q 观测前3维，这是测试借用的原 PPO 布局，E02 尚未确认，不能当真实采集接口。

| 次序 | 操作 | 检查到改变的模块 |
|---|---|---|
| 1 | N05 用尾部新采样和目标 Q 算冻结 y，再更新 Critic | 仅在线双 Q |
| 2 | N01 生成当前特征，N03 采样一次 a/log π | 参数和缓冲均不变；随机状态会推进 |
| 3 | N07 对同一次 log π 更新温度 | 仅 Temperature |
| 4 | N06 用**新 alpha 与同一次 a/log π**更新 Actor | 仅 Actor |
| 5 | N08 慢更新 | 仅目标双 Q |
| 6 | N02 有效窗口辅助更新 | 仅编码器与 projector，含 BN 状态 |

第4步还有独立数值对照：分别用新旧 alpha 计算损失，确认两者差异大于1e-5，实际损失与新值结果一致。它防止接口虽然按顺序调用，损失却提前使用了旧 alpha。池内原始历史还与更新前副本比较，报告数值也检查为有限。

**这只是一次 CPU 人工更新衔接，不是 T02 训练器。** 每轮要做多少次 Critic、何时更新 Actor（作者参考 policy_frequency=2）、辅助学习频率和 n 的选择，仍由后续 T02 配置。一次检查不能推导长期训练稳定、仿真通过或控制性能。

## 全部测试及辅助函数

`SmallQ.__init__(with_batchnorm=False)` 构造人工 Linear 与可选 BN，仅供状态更新测试，无自定义 forward；`pair(with_batchnorm)` 返回独立在线/目标小网络并冻结目标参数。更新循环内部 `snapshot` lambda 复制各模块 state_dict；局部 `changed(before)` 对照副本，返回变化的模块名集合，不是生产调度接口。

| 既有测试 | 覆盖内容 |
|---|---|
| `SoftUpdateTest.test_tau_zero_one_and_middle` | 不带缓冲的小网络：tau=0/0.3/1 的参数结果与手算一致，在线值不变 |
| `SoftUpdateTest.test_shared_storage_and_structure_are_rejected` | 同对象共享、参数/缓冲名称不符、tau=1.5 被拒绝 |
| `SoftUpdateTest.test_buffers_are_copied_not_averaged` | tau=0.1 时 BN 均值直接复制为4、计数为7；检查结果列出缓冲。tau=0 仍复制的结论来自上述实际循环，并非另跑了新测试 |
| `SoftUpdateTest.test_target_state_and_frequency` | 优化器包含目标或目标仍求梯度时拒绝，按计数判断频率 |
| `UpdateCycleTest.test_one_cycle_changes_only_owned_state` | 上述完整数据/更新衔接、模块归属与新 alpha 对照 |

入口为 `unittest.main()`。用户报告修正后的 N02/N06/N08 共17项再次通过，原43项记录在 PR/AI交接中；本次只静态阅读，未重跑：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/target_updates/test_targets.py -v
```

目标突然同步跳变先查共享存储、tau和目标是否误进优化器；长期不动查 Critic 计数与调用频率。新增带缓冲层后查复制规则。未验证长期学习、GPU、仿真或真实训练效果。回到 [N04](../q_networks/README_HUMAN.md) 核对初始化，整体更新顺序见[大模块概览](../README_HUMAN.md)。
