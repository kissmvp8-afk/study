# D05 · 历史窗口与有效性｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 已完成讲解与技术复核，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 已复核切片视图、有效帧数、编号检查与 11 项既有测试内容；修正动作次数/帧数偏差和未测性能表述，保留 T05 回归事项；只核对文档与源码，不开展全面代码审核 |
| 语言润色状态 | 2026-10-31，Gemini 已完成整体语言润色（`d29af80`）；后续技术修正已复核（`22862f1`）；按用户决定本轮不再进行局部润色，保留已有润色内容 |
| 已核对代码版本 | 初稿基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现/测试来自 `807ee636b1d40ff4be86ed27f9ce9ecd9f5afe2a`。文件头说明修正已包含于 `5b47469`；本次实际读取 `d29af80b06d941a0d4d11c1be63c9ecbbb975640`，源码与 `5b47469` 一致 |
| 待同步代码版本 | 无；本次仅修正文档，未修改实现或测试 |
| 实际阅读材料 | 本目录 AI README、[history.py](history.py)、[test_history.py](test_history.py) 全文；D01～D04 实现及测试；固定 PPO `actor_critic.py` 的 `_split_policy_obs`、`BarlowTwinsActor.forward`、`_InferenceWrapper`，T05 契约。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与工程验证证据)；本次另对照 E02、E05、N04、N05、T05 契约及 `5b47469`→`d29af80` 文档差异 |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 11 项记录见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。本次只核对文档/注释，不重跑算法测试；人工数据，未接网络、导出入口、仿真或训练；修正后检查本地链接/锚点、函数覆盖和代码片段，并核对修改范围，不重跑算法测试 |

## 已有十帧历史，为什么还要规定窗口

在机器人实机或物理仿真控制中，环境往往具有**部分可观测性（POMDP）**：仅凭单一时刻的传感器读数（如关节角度），算法无法确切得知关节的瞬时角速度与加速度，更无法感知传动机构的弹性形变与外力冲击。
为了赋予策略网络“感知时序变化趋势”的能力，系统在每次决策时都会保留一段连续的观测历史（D01 规定为 10 帧，形状 `[B, 10, 33]`）。

然而，策略网络并不会把这 10 帧数据一股脑全部塞入前馈层：
1. **主决策时序跨度**：沿用原 PPO 的窗口，取**最近的 5 帧**供历史编码器读取时序变化；这是一项保留的配置，当前测试没有比较不同窗口长度的训练效果；
2. **自监督辅助动力学学习（Barlow Twins）**：为了促使历史编码器学到稳健的机器人动力学隐空间表征，算法引入了自监督学习。它从 10 帧历史中同时切出**两段相邻且滑移了一步的 5 帧视窗**（主决策视窗与前置视窗），通过对比损失拉近两者的一致性并解耦特征通道。
3. **开局填充识别**：按现有测试约定，初始观测计为 1 帧，完成 2 次动作后已有 3 帧真实观测，10 帧中其余 7 个位置仍是填充。辅助学习应排除含填充的双窗口；具体重置填充方式尚待 E05 实现，D04 测试使用 -999，并不能据此断言实际环境重复初始帧或网络必然崩溃。

**D05 模块的核心职责，是通过共享输入存储的切片视图，将 10 帧策略历史分为当前帧、主决策视窗与前置辅助视窗；同时依据有效历史长度生成辅助学习掩码，并检查显式时间编号的连续性。**

本模块只进行张量切片与有效性校验；它不负责推移历史队列，不计算潜在特征编码，也不计算 Barlow Twins 损失。

## 输入输出与时间位置

设当前时刻为 $t$。历史充足时，10 帧索引 0 到 9 分别对应 $t-9\dots t$；开局不足的早期位置是填充，不代表已经经历过的物理时刻。

| 输出张量名称 | 切片代码表达 | 对应的物理时间点跨度 | 张量形状 | 核心用途与下游归宿 |
|---|---|---|---|---|
| `current` | `history[:, 9]` | $t$（单帧） | `[B, 33]` | 当前策略观测，供 Actor 决策；Q 的状态输入另由 `q_observation` 提供 |
| `recent` | `history[:, 5:10]` | $t-4 \dots t$（5 帧） | `[B, 5, 33]` | **主决策视窗**；供历史编码器输出隐空间状态表征，作为动作决策的核心依据 |
| `previous` | `history[:, 4:9]` | $t-5 \dots t-1$（5 帧） | `[B, 5, 33]` | **前置辅助视窗**；比主视窗前移 1 步，与 `recent` 组合构成自监督学习的双视角输入 |

### 联合跨度：为什么辅助学习必须要求 6 帧真实历史

我们用测试中的帧编号（从 0 到 9）直观展示三个张量的时序对齐：

```text
完整输入历史： 0   1   2   3   4   5   6   7   8   9  (共 10 帧，9 为当前时刻 t)
previous 视窗：                 4   5   6   7   8      (覆盖 t-5 .. t-1)
recent 视窗：                       5   6   7   8   9  (覆盖 t-4 .. t)
current 单帧：                                      9  (仅时刻 t)
```

- `recent` 视窗覆盖了时刻 $5, 6, 7, 8, 9$ 共 5 帧；
- `previous` 视窗覆盖了时刻 $4, 5, 6, 7, 8$ 共 5 帧；
- 两段视窗中间重叠了 4 帧（时刻 $5, 6, 7, 8$），二者联合起来**总共覆盖了 6 个互不相同的物理时间点（时刻 4 到 9）**！

> [!IMPORTANT]
> **深入理解为什么是 6 帧而不是 5 帧**：
> 如果当前轮次只有 5 帧真实观测（包含初始帧，对应已完成 4 次动作），真实数据只占历史索引 $5\dots9$，索引 4 仍是填充。
> 此时 `recent` 视窗虽然已经满额，但 `previous` 视窗的头部依然包含填充脏数据！
> 因此，自监督辅助学习（Barlow Twins）必须要求当前轮次的有效真实帧数 $\ge 6$（常量 `AUXILIARY_REQUIRED_FRAMES = 6`），才能确保两段视窗内的所有帧全部来自真实的物理交互。

## `history.py`：逐函数职责与调用边界

| 符号名称 | 签名与字段定义 | 核心职责与边界说明 |
|---|---|---|
| `HistoryWindowError` | 继承 `ValueError` | 历史切片专用异常类。在输入形状错误、有效长度越界或时序断裂时抛出 |
| `WINDOW_FRAMES` | `5` | 主决策视窗与辅助视窗的标准帧数 |
| `AUXILIARY_REQUIRED_FRAMES`| `6` | 两个 5 帧视窗联合所需的物理离散时间点数量（即 $\text{WINDOW\_FRAMES} + 1$） |
| `HistoryWindows` | `@dataclass(frozen=True)` | 封装输出张量的数据类，包含 `current`、`recent`、`previous` 三个字段；**张量均为输入数据的纯切片视图（View），不发生内存拷贝** |
| `split_windows` | `(history: Tensor) -> HistoryWindows` | 校验输入为三维且末两维为 `(10, 33)`，切出三个时序张量并封装返回；不改变输入，不追加新帧 |
| `valid_history_mask` | `(history_length, required_frames=6) -> Tensor` | 按契约接收有效长度向量 `[B]`，检查长度与门槛在 1..10 后，返回 `[B]` bool：`history_length >= required_frames`；默认门槛为 6，不读取观测内容 |
| `validate_time_order` | `(env_id, episode_id, step_index) -> None` | 接收形状相同的 `[M,L]` 编号，校验每行同环境、同轮次且步数逐条加一；列出出错行，L=1 时无相邻步需要检查 |

切片不复制观测数据，但仍会创建张量视图对象；当前只有人工 CPU 检查，没有耗时、GPU 显存或部署性能测量。

## 关键切片为什么不追加当前帧

在实现切片函数时，核心代码非常精简：

```python
end = HISTORY_LENGTH  # 索引 end-1 为时刻 t
return HistoryWindows(
    current=history[:, end - 1],
    recent=history[:, end - WINDOW_FRAMES : end],
    previous=history[:, end - WINDOW_FRAMES - 1 : end - 1],
)
```

`end=10` 确定历史右边界；`end-1=9` 取当前帧。`end-WINDOW_FRAMES=5` 到右端点 10 之前取索引 5..9；前置窗口两端各向左移 1，取 4..8。三个字段交给 `HistoryWindows` 封装，原历史不推进。

初学者可能会问：“为什么不像某些教程那样，先取过去的 4 帧，再用 `torch.cat` 把当前帧拼进去？”
**因为按照 D01 的数据契约，输入的 10 帧历史本身就已经包含了当前时刻 $t$（正是第 9 帧）！**
因此，直接向左截取 5 帧（`[:, 5:10]`）就已经完美覆盖了从时刻 $t-4$ 到时刻 $t$。如果画蛇添足地再次调用 `cat` 追加当前帧，就会导致当前时刻被重复计算两次，彻底破坏物理动力学的均匀时序间隔。

由于切片返回共享存储的张量视图，原地修改（如 `recent.add_()`）也会修改输入历史。若要原地加工独立副本，应先 `.clone()`；普通返回新张量的计算不必为了只读切片额外复制。冻结 `HistoryWindows` 属性同样不能禁止修改张量内容。

## 原 PPO 训练路径一致，原导出路径的缺陷分析

初稿通过 `git show` 读取原 PPO 固定快照 `8fe40aa`，对照以下三处切片；未启动 PPO 或部署模型：

1. **原训练数据切分（`_split_policy_obs`）**：
   从完整历史中提取 `current = obs[:, -1]` 与 `hist = obs[:, -6:-1]`（对应时刻 $t-5 \dots t-1$）；
2. **原训练网络前向（`BarlowTwinsActor.forward`）**：
   将 `hist[:, 1:]`（时刻 $t-4 \dots t-1$）与 `current` 拼接，合成主视窗（时刻 $t-4 \dots t$）；同时直接使用 `hist` 作为辅助视窗。**D05 的切片结果与原 PPO 的训练路径在数学与维度上完全一致**。
3. **原模型导出包装器（`_InferenceWrapper`）中的历史错位**：
   若向包装器传入已经包含当前帧的完整历史，它会执行 `history[:, 1:]`、重新拼接 `current`，再取最后 5 帧。输入帧号 $0\dots9$ 时，得到 **`6,7,8,9,9`**，而训练窗口为 `5,6,7,8,9`。这个结论有“输入已含当前帧”的前提，不是对所有既有部署调用方式的实测结论。

> [!NOTE]
> **保留既有工程缺陷记录，留待 T05 闭环**：
> 单元测试中的 `test_ppo_export_wrapper_on_full_history_repeats_current` 是一项专门的**回归警示测试**，用以验证原版导出包装器在面对完整历史时的重复缺陷。
> D05 严格保持与正确训练切片的一致性；后续在实现 T05 导出推理模块时，必须统一采用 D05 的无重复视窗契约，避免实机推理时再次引入当前帧重叠。

该测试按原包装器写法复现切片，并未运行 ONNX 或实机。当前没有证据说明旧 PPO 权重需要重复帧作为兼容要求；固定 baseline 保留原样。若 T05 需兼容既有部署产物，应先核对其调用端历史是否包含当前帧，再明确接口，不能据“旧权重”三个字默认保留错位。当前实现入口名为 `split_windows`。

## 有效长度与时间编号分别能证明什么

系统在时间轴上设置了两道严密的质量验证关卡：

### 1. 有效长度掩码（`valid_history_mask`）

核心判断代码只有一行：
```python
return history_length >= required_frames
```
- 输入测试有效长度 `[1, 5, 6, 10]`，当 `required_frames=6` 时，准确输出 `[False, False, True, True]`。
- **边界说明**：该函数不读取传感器数值内容，真实观测可以全零，填充也可能非零。它依赖 E05 维护的有效历史帧数，帧数包含初始观测，不等于已执行动作次数。D03 起始样本看 `first.history_length`，尾部下一历史看 `tail.next_history_length`。长度合法或门槛通过不能证明内容来自重置前。

### 2. 连续时间步校验（`validate_time_order`）

在从经验池抽出的序列矩阵中，校验函数执行向量化时序审查：
```python
bad = (step_index[:, 1:] != step_index[:, :-1] + 1).any(dim=1)
```
- 检查每一行内部的 `env_id` 是否相同，不同行可以来自不同环境；
- 检查每一行内部的 `episode_id` 是否相同，不要求整个批次属于同一轮；
- 检查步数编号 `step_index` 是否严格满足相邻两列步号递增 1（例如 $7 \to 8 \to 9$ 通过，出现重复 $0 \to 1 \to 1$ 或跳跃 $0 \to 2 \to 3$ 立即报警拦截）。

这仍只是编号检查，不能证明传感器内容或末状态捕获时机正确。对含无效后缀的 D02 片段，应像衔接测试那样逐行传入有效前缀，不能把另一轮的后缀一起要求为同轮连续。

## 与 D04 的无缝配合：同一统计下切片先后严格等价

D04 的 `normalize_policy` 是在张量的最后一维（33 个通道）上应用的逐帧固定变换。
因此，在统计参数保持冻结的前提下：
- **路径 A**：先对 10 帧输入执行 D04 归一化，再由 D05 切出 `recent` 窗口；
- **路径 B**：先由 D05 从原始历史中切出 `recent` 窗口（5 帧），再交给 D04 执行归一化。

两条路径在数学与计算精度上是**逐元素完全等价的**。在系统流水线中，推荐在网络前向入口处集中切片与归一化，保持逻辑紧凑清晰。

## `test_history.py`：11 项单元测试的覆盖范围

`numbered_history(batch_size=2)` 生成 `[B,10,33]` 人工历史，第 i 帧全部填 i；`frame_ids(window)` 读取第一个样本、每帧的第 0 通道，返回便于断言的编号列表。测试入口为 `unittest.main()`。

| 测试类与具体用例名称 | 核心验证逻辑与断言标准 |
|---|---|
| `SplitWindowsTest.test_frames_0_to_9` | 采用自然帧号 $0 \dots 9$ 验证：`current` 精确为 9，`recent` 精确为 $5 \dots 9$，`previous` 精确为 $4 \dots 8$，张量形状完全吻合 |
| `SplitWindowsTest.test_matches_ppo_training_split` | 随机输入比对：验证与原 PPO 训练切片逻辑在浮点精度下逐元素 100% 相等 |
| `SplitWindowsTest.test_ppo_export_wrapper_on_full_history_repeats_current` | 回归警示测试：重现并记录原版导出包装器在完整输入下产生重复帧 `6, 7, 8, 9, 9` 的历史问题 |
| `SplitWindowsTest.test_does_not_advance_or_copy_history` | CPU 视图检查：连续调用结果相同、输入不变；比对 `current` 与输入第 9 帧的数据地址，未进行 GPU 显存测量 |
| `SplitWindowsTest.test_rejects_wrong_shape` | 传入 `[2,9,33]`，验证形状错误被拒绝；实现也要求末维为 33，但本用例没有单独传入错误特征维数 |
| `ValidityTest.test_auxiliary_needs_six_real_frames` | 联合跨度测试：验证 5 帧能满足决策但不能满足辅助学习，确立 6 帧门槛的必要性 |
| `ValidityTest.test_does_not_use_observation_values` | 逻辑纯粹性测试：验证合法全零观测依然被判定为有效，确认不从数值瞎猜填充 |
| `ValidityTest.test_rejects_out_of_range` | 防御性测试：拦截有效长度为 0 或需求帧数大于 10 的非法参数 |
| `TimeOrderTest.test_consecutive_passes` | 连续时序测试：验证同环境同轮次的步序（如 $7, 8, 9$）顺利过关 |
| `TimeOrderTest.test_detects_problems` | 时序断裂拦截：精准定位并报错跨环境、跨轮次、步序重复或跳跃的具体样本行 |
| `ReplayIntegrationTest.test_windows_on_replayed_batch` | 人工 CPU 数据衔接：串联 D02、D03、D04，核对切片/归一化先后等价、有效长度标记与原始数据不变；不代表网络、仿真或导出链路打通 |

本地复现测试命令：
```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/history_windows/test_history.py -v
```

> [!WARNING]
> **验证边界声明**：
> 目前 11 项测试已在 CPU 合成数据下全部通过。尚未在包含真实 Barlow Twins 神经网络训练、Isaac Lab 物理引擎重置填充维护、实机 ONNX 模型导出以及高频部署推理闭环中开展最终验证。

## 阅读导航与后续模块

- 了解 10 帧历史与有效长度定义：请参考 [D01 经历记录契约](../transition_schema/README_HUMAN.md)；
- 了解历史观测的在线标准化机制：请参考 [D04 观测归一化](../normalization/README_HUMAN.md)；
- 了解时序特征供给与神经网络编码器：请参考 [N01 历史编码器](../../networks_learning/history_encoder/README.md) 与 [N02 辅助动力学学习](../../networks_learning/auxiliary_learning/README.md)；
- 了解模型导出推理中的视窗修复规范：请参考 [T05 导出推理](../../training_validation/export_inference/README.md)。
