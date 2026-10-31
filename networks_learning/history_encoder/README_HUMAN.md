# N01 · 历史状态编码｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [Codex 写作约定](../../CODEX.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对窗口与展平顺序、52 维拼接、速度尺度、梯度及 BatchNorm 模式；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `7e7d34a` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[encoder.py](encoder.py)、[test_encoder.py](test_encoder.py) 全文；D04 `normalize_policy`、D05 `split_windows` 的契约和实际实现；固定 PPO 的 `MlpBarlowTwinsActor`、`mlp_batchnorm_factory` 相关片段。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 7 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## 学习目标：把过去五帧变成当前动作的依据

Actor 是接收观测并生成动作的策略网络。单帧观测未必足以描述运动趋势，因此本项目保留原 PPO 的历史编码器：用网络把最近五帧压缩为共享表示，再输出速度估计和隐藏特征。隐藏特征是网络学到的 16 个数字，没有逐维预先指定的物理解释。

N01 保留原 PPO 的层序与拼接顺序；替换 PPO 为 SAC，并不意味着让 Actor 直接读取仿真真实速度。真实速度只在 N02 中作为监督标签，运行时 Actor 使用本模块的预测。编码器的学习由 N02 管理，N01 本身没有优化器。

## 从十帧历史到 52 维输入

B 是批次记录数，当前时刻记为 t。D01 保存 `[B,10,33]` 的原始策略历史，含当前帧、从旧到新。先经 D04 逐通道归一化，再由 D05 切片：

| 数据 | 形状 | 时间或内容 |
|---|---|---|
| `windows.current` | `[B,33]` | t 的当前帧 |
| `windows.recent` | `[B,5,33]` | t−4..t，供本模块决策编码 |
| `windows.previous` | `[B,5,33]` | t−5..t−1，本模块决策不读取它，供 N02 辅助学习 |
| 共享表示 | 默认 `[B,64]` | 主干网络输出 |
| 速度估计 | `[B,3]` | 与 E02 标签同为 base_lin_vel 乘 2.0 后的数值 |
| 隐藏特征 | `[B,16]` | 非人工命名的状态表示 |
| Actor 输入 | `[B,52]` | **速度估计 3 → 隐藏特征 16 → 当前帧 33** |

52 是维数之和，不能据此任意交换顺序。速度预测若为 `[2,0,0]`，换回物理速度是 `[1,0,0] m/s`；送入 Actor 的仍是预测的缩放值。归一化后的当前帧是无量纲特征，16 维隐藏特征也没有逐项物理单位。

历史窗口按帧优先展平：若五帧所有通道分别填 5、6、7、8、9，165 维向量先是 33 个 5，再是 33 个 6，直到 33 个 9。把通道维和时间维先转置再展平，虽然仍有 165 个数，网络读到的含义却变了。

## 文件与每个接口怎样协作

[encoder.py](encoder.py) 定义网络与特征入口，[test_encoder.py](test_encoder.py) 用人工历史验证结构、时序和隔离。

| 类或函数 | 输入、输出与职责 |
|---|---|
| `HistoryEncoderError` | 形状或模式不符合约定时使用的 `ValueError` 子类 |
| `_mlp_batchnorm(input_dim, hidden_dims, out_dim)` | 返回 `nn.Sequential`；各隐藏层按 Linear → BatchNorm1d → ELU 排列，按需追加输出 Linear；构造新参数与统计缓冲 |
| `HistoryEncoder.__init__(backbone_dims=(128,64), latent_hidden_dim=32)` | 构建主干 `165→128→64`、隐藏头 `64→32→16` 和速度头 `64→3`，默认设为 eval；64 是默认主干末维，改变配置时两头随之接入 |
| `HistoryEncoder.encode_history(window)` | 检查 `[B,5,33]`，按帧展平后返回共享表示；直接调用时的 BN 行为取决于 train/eval 模式 |
| `HistoryEncoder.predict_velocity(representation)` | 共享表示 → `[B,3]`；只走速度头，不做单位换算 |
| `HistoryEncoder.hidden_features(representation)` | 共享表示 → `[B,16]`；只走隐藏头 |
| `build_actor_features(encoder, windows)` | 要求编码器为 eval；读取 recent 编码，再拼 current，返回不连接编码器计算图的 `[B,52]` |
| `features_from_history(encoder, normalizers, policy_history)` | 原始 `[B,10,33]` → D04 `normalize_policy` → D05 `split_windows` → 上一入口；用于当前状态与 N05 的尾部状态 |

常量 `VELOCITY_DIM=3`、`HIDDEN_FEATURE_DIM=16`、`ACTOR_FEATURE_DIM=52` 固定各部分大小。模块通过相邻目录的 `sys.path` 导入 D01/D04/D05，目前尚未形成统一 Python 包。

## 为什么 no_grad 与 eval 都需要

梯度是损失对参数变化的敏感程度，优化器据此改参数。BatchNorm（BN）则是网络内部的批标准化层：train 模式用当前批次统计并更新运行均值/方差，eval 使用保存的运行统计。它与 D04 对原始观测做的标准化是两套独立机制。

核心代码为：

```python
with torch.no_grad():
    representation = encoder.encode_history(windows.recent)
    velocity = encoder.predict_velocity(representation)
    hidden = encoder.hidden_features(representation)
    return torch.cat([velocity, hidden, windows.current], dim=-1)
```

第一行关闭这条特征路径的梯度记录；随后只编码 recent，两个头共用 representation，最后按 3/16/33 拼接。Actor 损失可训练 N03，却不能顺着这些特征更新编码器。**no_grad 不会自动把 BN 切为 eval**，因此函数在这段代码前还检查 `encoder.training`，若为 train 就报错。否则即便没有梯度，采集或回放前向仍可能改变 BN 统计。

`features_from_history` 只读取 D04 当前统计，不更新计数，也不推进历史。已归一化窗口可直接用 `build_actor_features`，避免再次归一化。编码器及 BN 的更新由 N02 在明确的辅助学习阶段完成。

## 测试覆盖与维护入口

测试辅助函数 `random_history(batch,seed)` 生成可复现的人工历史；`trained_like_encoder(seed)` 让主干 BN 接触一批随机数据后回到 eval，用来检查非默认统计，**不代表训练出了运动技能**。`ppo_mlp_batchnorm_factory` 与 `PpoEncoderReference.__init__` 复写固定 PPO 的编码结构，后者包含旧名称的三部分，供权重映射比较。

| 既有测试 | 验证内容 |
|---|---|
| `ShapeAndOrderTest.test_dimensions` | 64/3/16/52 维，Actor 输入最后 33 维等于 current |
| `ShapeAndOrderTest.test_repeatable_in_eval_mode` | 同输入、同参数重复输出相同 |
| `ShapeAndOrderTest.test_uses_recent_window_in_frame_order` | 与手工展平一致；反转帧或错用 previous 会改变输出 |
| `ShapeAndOrderTest.test_rejects_wrong_window_shape` | 拒绝六帧窗口 |
| `PpoEquivalenceTest.test_same_weights_same_outputs` | 按模块名前缀映射同一组权重，和测试内 PPO 结构输出在 1e-6 容差内一致；不是整套 PPO 与 SAC 的等价证明 |
| `GradientAndBatchNormTest.test_features_detached_and_statistics_unchanged` | 特征无梯度，参数与 BN 状态不变 |
| `GradientAndBatchNormTest.test_train_mode_is_rejected` | 特征入口拒绝 train 模式 |

测试入口是 `unittest.main()`。既有命令（本次未执行）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/history_encoder/test_encoder.py -v
```

速度看似翻倍先看 2.0 缩放；52 维数对但动作异常先看拼接/展平顺序、D04 是否重复归一化；Actor 更新后编码器统计变化先看 BN 模式。当前没有真实 TITA 编码质量、GPU、导出或训练效果证据。

下一步阅读 [N02 辅助学习](../auxiliary_learning/README_HUMAN.md) 与 [N03 动作分布](../actor_distribution/README_HUMAN.md)；窗口和原始字段分别见 [D05](../../experience_returns/history_windows/README_HUMAN.md)、[D01](../../experience_returns/transition_schema/README_HUMAN.md)。
