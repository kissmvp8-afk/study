# N04 · 双 Q 评分网络｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对状态/动作拼接、双 Q 独立性、目标副本和训练侧观测边界；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `ef98631` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[critics.py](critics.py)、[test_critics.py](test_critics.py) 全文；E02/E03 契约与 E03 动作换算片段；D01/D04 的字段和归一化实现；N01/N03 全文及固定 PPO 的相关结构片段。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 5 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## 从“这个状态好吗”到“这个动作好吗”

原 PPO 的 V 网络评价状态；SAC 的 Q 网络还要接收一个指定动作，预测从该状态执行该动作后的长期收益。这个预测不是实时物理测量，也不是直接输出控制动作。

N04 创建两套独立 Q。后续 N05/N06 取两个评分的较小值，以减轻高估带来的误导；它不保证绝对保守或准确。每套在线 Q（被优化器训练的网络）还有一份目标 Q（为学习目标提供较慢变化的参考），由 N08 更新。

## 输入、输出与额外信息

| 数据 | 形状与含义 |
|---|---|
| `q_observation` | `[B,C]`，D04 `normalize_q` 后的训练侧状态信息 |
| `action` | `[B,8]`，与 N03/D01/E03 相同的无量纲策略动作，不是物理关节目标 |
| 单 Q 输出 | `[B,1]`，所用奖励尺度下的折扣收益预测 |
| 双 Q 输出 | `(q1,q2)`，两个 `[B,1]` 张量，不在本模块内部合并 |

C 在构造时固定，来自 E02 的实际观测组；54/241 只是平地/复杂地形预期值。Q 可以读取 Actor 看不到的仿真信息，例如真实身体线速度，不能因此将这些信息拼进 N01 的 Actor 输入。在线与目标 Q 都使用 D04 的**同一份当前 Q 统计**，而不是各自统计输入。

当前结构是 `(C+8)→512→256→128→1`，隐藏层用 ELU，沿用原 PPO Critic 的隐藏维数，增加了动作输入。第一版没有单独的地形编码层；若 E02 提供 187 维高度扫描，它们直接作为 C 的一部分进入 MLP。原 PPO 使用 scan_encoder 的做法并未在这里实现，后续若加入，应作为 Q 子模块同时纳入目标复制、慢更新和存档。

## 逐文件、逐接口

[critics.py](critics.py) 提供网络、评价入口及目标初始化；[test_critics.py](test_critics.py) 检查维数、动作参与和参数独立性。

| 接口 | 输入输出、调用及状态变化 |
|---|---|
| `QNetwork.__init__(q_dim,action_dim=8,hidden_dims=(512,256,128))` | 要求 q_dim>0，建立一个全连接网络；保存 C 与动作维数 |
| `QNetwork.forward(q_observation,action)` | 检查两者末维，按 `[Q 观测 \| 动作]` 拼接并返回评分；预期批次已经对齐，未重复执行 D01 完整校验 |
| `TwinQ.__init__(q_dim,action_dim=8,hidden_dims=(512,256,128))` | 分别构造 `q1`、`q2`，不是将同一个对象赋值两次；本类没有统一 forward |
| `evaluate_q_pair(critics,q_observation,action)` | 分别调用两个 Q，返回二者评分；是否记录梯度由外层上下文决定 |
| `initialize_targets(critics)` | deepcopy 完整在线双 Q，冻结目标参数，设为 eval；分配独立存储 |
| `list_critic_parameters(critics)` | 返回所有 `requires_grad=True` 的 `(名称,参数)` 列表，供 N05 构造优化器，不更新网络 |

`ACTION_DIM=8`、`CRITIC_HIDDEN_DIMS=(512,256,128)` 给出默认维数。当前 Q 没有 BN 或其他注册缓冲，归一化在 D04 外部完成；作者 SAC 的双 Q/目标副本是参考做法，TITA 的 C 与混合动作坐标以本项目契约为准。

## 为什么动作、目标复制与梯度上下文都重要

评价的一行核心代码：

```python
return self.net(torch.cat([q_observation, action], dim=-1))
```

动作沿最后一维接在状态之后。例如 C=54 时输入是 62 维；换动作就能改变 Q 输入，SAC 才能通过“动作怎样改变评分”训练 Actor。若漏动作，网络退化为只评价状态，N06 将失去这条动作梯度。

目标初始化的真实代码：

```python
target = copy.deepcopy(critics)
for parameter in target.parameters():
    parameter.requires_grad_(False)
return target.eval()
```

deepcopy 保证初始数值相同而存储独立；若直接 `target=critics`，训练在线 Q 会立即改目标。冻结参数让目标不参与梯度优化，eval 则明确推理模式；两者作用不同。以后目标由 N08 按名称更新，而不是加入 Adam。

同一个 `evaluate_q_pair` 在 N05 在线 Q 损失中需要对 Q 参数求梯度，在 N05 目标中处于 no_grad，在 N06 中冻结 Q 参数但保留对输入动作的梯度。本模块不统一加 no_grad，才能服务这三种调用。

## 既有测试与定位方法

`storages(module)` 收集参数存储地址以比较是否共享，不比较数值是否恰好相同。

| 既有测试 | 覆盖内容 |
|---|---|
| `EvaluationTest.test_different_actions_give_different_scores` | 同状态分别输入零动作/一动作，两个 Q 输出均有变化，形状 `[8,1]` |
| `EvaluationTest.test_q_dim_is_checked` | C=54/241 的人工输入可计算，错误 C 或七维动作被拒绝；不证明真实 E02 配置 |
| `IndependenceTest.test_twins_are_independent` | 双 Q 初值与存储独立，只优化 Q1 时 Q2 不变 |
| `IndependenceTest.test_targets_copy_values_not_storage` | 目标数值复制、独立存储、不求梯度且 eval；修改在线值不改目标 |
| `IndependenceTest.test_parameter_listing_and_no_sharing_with_actor_encoder` | 参数列表覆盖在线双 Q，与目标、N01、N03 不共享参数 |

测试入口为 `unittest.main()`。以下是既有复现命令，本次未执行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/q_networks/test_critics.py -v
```

不同动作评分相同先查动作拼接和尺度，再查梯度；目标跟着在线值立即跳变先查共享存储。实际 C、字段含义、地形编码是否需要、GPU 与训练评分质量仍待验证。继续阅读 [N05 Q 更新](../critic_learning/README_HUMAN.md)、[N06 Actor 更新](../actor_learning/README_HUMAN.md)、[N08 目标更新](../target_updates/README_HUMAN.md)。
