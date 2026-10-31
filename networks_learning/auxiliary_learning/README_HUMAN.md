# N02 · 速度监督与 Barlow Twins｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [Codex 写作约定](../../CODEX.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对速度标签、相关矩阵、常数特征、六帧筛选、BN 模式与参数归属；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `2e6c470`，审核后说明修正 `bd69772` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[auxiliary.py](auxiliary.py)、[test_auxiliary.py](test_auxiliary.py) 全文；N01 实现与测试；D04/D05 接口及实现片段、E02 速度标签契约；固定 PPO `barlow_twins_loss`、`off_diagonal` 与优化路径片段。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 8 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## 为什么编码器需要自己的学习目标

N01 为 Actor 生成特征时断开梯度，所以不能期待 SAC 的 Actor 损失顺便教会编码器。N02 使用两个辅助目标：**速度监督**直接比较预测与已知速度标签；**Barlow Twins（BT）**比较相邻两个历史窗口的表示，使对应通道保持一致，并减少不同通道的信息重复。

BT 不需要人为给 16 个隐藏特征逐个命名。它经一个只用于训练的转换网络 projector，把特征变为 64 维后计算相关性；运行 Actor 时不用 projector。速度头、隐藏头和主干仍是 N01 的同一个编码器。

原 PPO 把辅助损失并入总损失、共用优化器。这里保留速度误差与 BT 数值公式，但用独立优化器和独立调用管理辅助学习，避免 SAC 多次回放时无意反复更新编码器。频率仍由后续 T02 决定。

## 输入、输出与有效样本

调用方先对历史做 D04 归一化，再用 D05 切片。`update_encoder` 接收：

| 输入 | 形状与含义 |
|---|---|
| `windows.recent` / `previous` | `[B,5,33]`，分别截至 t 与 t−1，联合覆盖 t−5..t |
| `history_length` | `[B]`，对应当前历史的真实帧数，包含初始观测 |
| `velocity_label` | `[B,3]`，时刻 t 的真实身体线速度观测，已经乘 2.0，**未经 D04 归一化** |
| `encoder`、`projector`、`optimizer` | N01 编码器、训练转换网络及其专属优化器 |

只保留长度 ≥6 的行。比如长度 `[2,5,6,10]` 仅最后两行有效；这同时用于 BT 和速度监督。六帧门槛来自两个五帧窗口联合跨度，不是单个窗口长度。有效行少于 2 时又必须跳过，因为 train 模式的 BN 需要至少两条样本。

速度预测为 `[2,0,0]`、标签为 `[2,0,0]` 时 MSE=0，二者对应的物理速度均是 `[1,0,0] m/s`。若错误拿 `[1,0,0]` 作标签，优化目标本身就错了。原 PPO 用 Critic 观测前 3 维作为标签；当前 E02 尚未确认最终字段，接口因此要求显式传入，不自行猜切片。

正常返回日志 `skipped=False`、`valid_samples`、`loss`、`velocity_loss`、`on_diagonal`、`off_diagonal`；跳过时只返回前两项。修改的是编码器、projector 的参数、BN 运行统计及优化器状态，不修改 Actor、Q 或 alpha。

## 两个文件与全部实现接口

| [auxiliary.py](auxiliary.py) 中的接口 | 职责与调用关系 |
|---|---|
| `BarlowProjector.__init__()` | 新建 `16→64（无偏置 Linear、BN、ELU）→64（无偏置 Linear）→BN(affine=False)`，构造后 eval；末层无可学习仿射参数，但仍有运行统计 |
| `BarlowProjector.forward(hidden)` | `[N,16]→[N,64]`；train 时会更新自身 BN，eval 时只读统计 |
| `velocity_loss(predicted,label)` | 检查形状一致且最后一维为 3，返回逐元素平方误差的均值；不修改输入 |
| `_off_diagonal(matrix)` | 方阵 → 一维非对角元素，排除所有 i=j 项；供 BT 去重复项求和 |
| `barlow_loss(z1,z2,off_diagonal_weight=0.005)` | 接收 `[N,64]`，返回总 BT 损失、对角项、未加权的非对角项；预期输入已通过 projector |
| `make_auxiliary_optimizer(encoder,projector,lr=0.001)` | 创建只含二者参数的 Adam，不包含 Actor 动作网络、Q 或 log_alpha |
| `update_encoder(encoder,projector,optimizer,windows,history_length,velocity_label,max_grad_norm=1.0)` | 筛选 → 断开输入梯度 → train 前向两窗口 → BT+速度误差 → 反传、裁剪、step → eval；有效行不足则提前返回 |

常量 `PROJECTOR_DIM=64`、`OFF_DIAGONAL_WEIGHT=0.005`、`AUXILIARY_LEARNING_RATE=0.001`、`MAX_GRAD_NORM=1.0` 沿用原 PPO 相关配置作为起点；`MIN_VALID_SAMPLES=2` 是 BN 的最低批量要求。

## 相关矩阵为什么这样算

projector 在 train 模式下按批标准化各通道，随后：

```python
c = z1.T @ z2 / z1.shape[0]
on_diagonal = (torch.diagonal(c) - 1.0).pow(2).sum()
off_diagonal = _off_diagonal(c).pow(2).sum()
return on_diagonal + off_diagonal_weight * off_diagonal, on_diagonal, off_diagonal
```

第一行跨样本计算两个视图的通道相关矩阵 `[64,64]`，除 N，而不是除 64。第二行要求同一通道跨窗口相关性接近 1；第三行惩罚不同通道之间的重复相关性；最后用 0.005 降低非对角项权重。日志的 `off_diagonal` 是原始和，重建总损失时还须乘权重。

若输入隐藏特征全是常数，经 train 模式 projector 后各通道近零，c≈0，于是对角损失约为 `64×(0−1)²=64`，不是 0。**64 不是最大值**：若标准化后的两个视图对应通道反相关，c 的对角为 −1，对角项可达 `64×(−1−1)²=256`。这说明常数输出不能成为零损失解，不能据此保证真实训练永不塌缩。

总辅助损失还要加速度 MSE：`loss = bt + vel`。速度监督走 recent 的共享表示，BT 的第二视图才走 previous。把速度标签错配到 t−1，会改变所学习的预测时刻。

## 梯度与 BN 状态的更新边界

关键筛选发生在网络计算之前：

```python
valid = valid_history_mask(history_length, AUXILIARY_REQUIRED_FRAMES)
count = int(valid.sum())
if count < MIN_VALID_SAMPLES:
    return {"skipped": True, "valid_samples": count}
recent = windows.recent[valid].detach()
previous = windows.previous[valid].detach()
label = velocity_label[valid].detach()
```

前两行先确定样本；不足两条直接返回，连 train/eval 模式都不切换。后三行只取有效数据，再断开对输入及标签的梯度关系；**没有断开编码器参数的梯度**。因此无效行即便是 NaN，也不会先污染 BN 再被损失掩掉。

正常更新前把 encoder/projector 切为 train。recent 与 previous 各前向一次，共享主干、隐藏头及 projector 的 BN 因而各接触两批窗口；速度头复用 recent 表示。反传后把二者参数的总梯度范数裁剪到 1.0，再执行 Adam。`finally` 中恢复二者为 eval，即便计算中抛异常也执行模式恢复；这是恢复默认 eval，并非恢复任意调用前模式。

BN 的 `running_mean`、`running_var`、批次数也是模型状态。`no_grad` 本身不阻止它们在 train 前向中变化，因此采集、Critic 和 Actor 的特征入口必须遵守 N01 的 eval 约定。

## 测试验证了什么

[test_auxiliary.py](test_auxiliary.py) 的 `setup(batch,seed)` 建立随机窗口、速度标签、全长 10 的长度和优化器；`snapshot(*modules)` 复制参数/缓冲，`same(a,b)` 逐项比较。其他网络在本文件用占位模块检查隔离，真实网络衔接另见 N08。

| 既有测试 | 覆盖内容 |
|---|---|
| `LossTest.test_velocity_loss_zero_when_correct` | 正确速度标签得零 MSE，偏移后为正 |
| `LossTest.test_constant_features_are_not_perfect` | 常数特征对角项≈64；标准化随机独立特征的对角项接近零、总损失较小 |
| `LossTest.test_matches_ppo_formula` | 与测试内复写的原 PPO 原地运算公式比较三项结果 |
| `UpdateTest.test_only_encoder_and_projector_change` | 只改负责模块；优化器参数集合精确对应二者 |
| `UpdateTest.test_batchnorm_statistics_only_change_during_update` | eval 前向不改统计，辅助更新改统计，结束为 eval |
| `UpdateTest.test_insufficient_valid_history_is_skipped` | 0/1 条有效样本均跳过，参数、统计和优化器状态不变 |
| `UpdateTest.test_invalid_rows_are_not_used` | 三条无效行填 NaN，有效行计算仍有限 |
| `UpdateTest.test_repeated_updates_reduce_loss` | 同一人工批次更新 30 次损失下降，不是机器人训练效果 |

入口为 `unittest.main()`。既有交接还记录把带梯度张量的 `float(loss)` 日志改为 `.item()` 后警告消失。本次未重跑；复现命令为：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/auxiliary_learning/test_auxiliary.py -v
```

损失异常先分别看速度项、对角项、非对角项及有效样本数；BN 意外漂移先查模式与重复辅助调用；速度偏差查标签缩放和时刻。真实标签字段、辅助更新频率、GPU 与仿真/训练表现仍待后续验证。

返回 [N01 编码器](../history_encoder/README_HUMAN.md)，继续阅读 [N06 Actor 更新](../actor_learning/README_HUMAN.md)；历史有效性见 [D05](../../experience_returns/history_windows/README_HUMAN.md)。
