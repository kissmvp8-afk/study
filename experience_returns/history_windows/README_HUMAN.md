# D05 · 历史窗口与有效性｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 已完成讲解与润色 |
| 技术核对状态 | 已核对切片、视图、有效长度、时间编号与 11 项测试；本次修正文件头对原导出包装器的错误说明，不修改算法 |
| 语言润色状态 | 2026-10-31，Gemini 完成自然语言润色，面向具备 Python 基础、正在学习强化学习与机器人控制的读者，阐明了时序切片视图、Barlow Twins 双窗口联合跨度与有效历史长度校验机制；严格保留全部技术信息、切片索引、时间顺序、历史记录与验证限制 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；算法与测试来自 `807ee636b1d40ff4be86ed27f9ce9ecd9f5afe2a`。本次工作区仅另修正 `history.py` 模块说明，执行语句保持不变，提交后由 Git 历史定位该注释版本 |
| 待同步代码版本 | 无；本页已反映指定文件头修正 |
| 实际阅读材料 | 本目录 AI README、[history.py](history.py)、[test_history.py](test_history.py) 全文；D01～D04 实现及测试；固定 PPO `actor_critic.py` 的 `_split_policy_obs`、`BarlowTwinsActor.forward`、`_InferenceWrapper`，T05 契约。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 11 项记录见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。本次只核对文档/注释，不重跑算法测试；人工数据，未接网络、导出入口、仿真或训练 |

## 已有十帧历史，为什么还要规定窗口

在机器人实机或物理仿真控制中，环境往往具有**部分可观测性（POMDP）**：仅凭单一时刻的传感器读数（如关节角度），算法无法确切得知关节的瞬时角速度与加速度，更无法感知传动机构的弹性形变与外力冲击。
为了赋予策略网络“感知时序变化趋势”的能力，系统在每次决策时都会保留一段连续的观测历史（D01 规定为 10 帧，形状 `[B, 10, 33]`）。

然而，策略网络并不会把这 10 帧数据一股脑全部塞入前馈层：
1. **主决策时序跨度**：网络选用**最近的 5 帧**数据作为主决策视窗，既能捕捉高频动态趋势，又能避免更早的历史引入滞后噪声；
2. **自监督辅助动力学学习（Barlow Twins）**：为了促使历史编码器学到稳健的机器人动力学隐空间表征，算法引入了自监督学习。它从 10 帧历史中同时切出**两段相邻且滑移了一步的 5 帧视窗**（主决策视窗与前置视窗），通过对比损失拉近两者的一致性并解耦特征通道。
3. **开局填充识别**：当机器人刚刚启动（如才走了 2 步），10 帧历史中有 8 帧是用初始状态重复填充的占位数据。如果把大量填充数据强行喂给辅助学习网络，会导致表征崩溃。

**D05 模块的核心职责，是在不发生显存复制的前提下，将 10 帧策略历史精准切分为当前帧、主决策视窗与前置辅助视窗；同时依据有效历史长度生成自监督学习可用性掩码，并提供时序连续性严格核验。**

本模块只进行张量切片与有效性校验；它不负责推移历史队列，不计算潜在特征编码，也不计算 Barlow Twins 损失。

## 输入输出与时间位置

设当前时刻为 $t$。在 D01 标准中，10 帧历史的索引从 0 到 9，分别对应物理时间点 $t-9 \dots t$：

| 输出张量名称 | 切片代码表达 | 对应的物理时间点跨度 | 张量形状 | 核心用途与下游归宿 |
|---|---|---|---|---|
| `current` | `history[:, 9]` | $t$（单帧） | `[B, 33]` | 当前最新观测帧；供 Actor 直接前馈或 Q 网络状态评估直接使用 |
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
> 如果机器人当前轮次刚刚走了 5 步（有效长度为 5，真实数据只存在于时刻 $5 \dots 9$），那么时刻 4 依然是开局重置时的填充占位符。
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
| `valid_history_mask` | `(history_length, required_frames=6) -> Tensor` | 输入有效长度标量，校验合法性后返回布尔掩码 `[B]`（当且仅当 $\text{length} \ge 6$ 时为 True） |
| `validate_time_order` | `(env_id, episode_id, step_index)` | 纯时序核验工具：接收形状为 `[M, L]` 的元数据矩阵，严格校验每行样本是否在同环境、同轮次内步数严格连续逐增 1 |

所有切片操作均属于 PyTorch 的原生内存视图机制，耗时仅微秒级，不占用额外显存。

## 关键切片为什么不追加当前帧

在实现切片函数时，核心代码非常精简：

```python
end = HISTORY_LENGTH  # 10
current = history[:, end - 1]
recent = history[:, end - WINDOW_FRAMES : end]
previous = history[:, end - WINDOW_FRAMES - 1 : end - 1]
```

初学者可能会问：“为什么不像某些教程那样，先取过去的 4 帧，再用 `torch.cat` 把当前帧拼进去？”
**因为按照 D01 的数据契约，输入的 10 帧历史本身就已经包含了当前时刻 $t$（正是第 9 帧）！**
因此，直接向左截取 5 帧（`[:, 5:10]`）就已经完美覆盖了从时刻 $t-4$ 到时刻 $t$。如果画蛇添足地再次调用 `cat` 追加当前帧，就会导致当前时刻被重复计算两次，彻底破坏物理动力学的均匀时序间隔。

由于切片返回的是底层共享内存的张量视图（View），若下游模块违规对这些张量执行原地加减操作（如 `recent.add_()`），就会顺带破坏输入的原始历史张量。调用方若需要对特征进行二次加工，必须显式调用 `.clone()`。

## 原 PPO 训练路径一致，原导出路径的缺陷分析

在审查原版 PPO 快照代码（提交 `8fe40aa`）时，工程团队深入比对了三处历史切片逻辑：

1. **原训练数据切分（`_split_policy_obs`）**：
   从完整历史中提取 `current = obs[:, -1]` 与 `hist = obs[:, -6:-1]`（对应时刻 $t-5 \dots t-1$）；
2. **原训练网络前向（`BarlowTwinsActor.forward`）**：
   将 `hist[:, 1:]`（时刻 $t-4 \dots t-1$）与 `current` 拼接，合成主视窗（时刻 $t-4 \dots t$）；同时直接使用 `hist` 作为辅助视窗。**D05 的切片结果与原 PPO 的训练路径在数学与维度上完全一致**。
3. **原模型导出包装器（`_InferenceWrapper`）中的历史错位**：
   在原版实机部署包装器中，代码对包含当前帧的 10 帧历史错误地执行了 `history[:, 1:]` 并重新拼接 `current`，最终提取 5 帧。若输入为帧号 $0 \dots 9$，原包装器输出的帧号竟然是 **`6, 7, 8, 9, 9`**（导致当前时刻 9 被重复记录了两次）！

> [!NOTE]
> **保留既有工程缺陷记录，留待 T05 闭环**：
> 单元测试中的 `test_ppo_export_wrapper_on_full_history_repeats_current` 是一项专门的**回归警示测试**，用以验证原版导出包装器在面对完整历史时的重复缺陷。
> D05 严格保持与正确训练切片的一致性；后续在实现 T05 导出推理模块时，必须统一采用 D05 的无重复视窗契约，避免实机推理时再次引入当前帧重叠。

## 有效长度与时间编号分别能证明什么

系统在时间轴上设置了两道严密的质量验证关卡：

### 1. 有效长度掩码（`valid_history_mask`）

核心判断代码只有一行：
```python
return history_length >= required_frames
```
- 输入测试有效长度 `[1, 5, 6, 10]`，当 `required_frames=6` 时，准确输出 `[False, False, True, True]`。
- **边界说明**：该函数不读取传感器数值内容（传感器读数本身可能合法全零，填充数据也可能非零）。它完全信任上游 E05 维护的真实步数计数。在计算 D03 样本时，起始样本看 `first.history_length`，尾部样本必须看 `tail.next_history_length`，切勿混淆。

### 2. 连续时间步校验（`validate_time_order`）

在从经验池抽出的序列矩阵中，校验函数执行向量化时序审查：
```python
bad = (step_index[:, 1:] != step_index[:, :-1] + 1).any(dim=1)
```
- 检查每一行样本的环境编号 `env_id` 是否全局单一；
- 检查轮次编号 `episode_id` 是否全局单一；
- 检查步数编号 `step_index` 是否严格满足相邻两列步号递增 1（例如 $7 \to 8 \to 9$ 通过，出现重复 $0 \to 1 \to 1$ 或跳跃 $0 \to 2 \to 3$ 立即报警拦截）。

## 与 D04 的无缝配合：同一统计下切片先后严格等价

D04 的 `normalize_policy` 是在张量的最后一维（33 个通道）上应用的逐帧固定变换。
因此，在统计参数保持冻结的前提下：
- **路径 A**：先对 10 帧输入执行 D04 归一化，再由 D05 切出 `recent` 窗口；
- **路径 B**：先由 D05 从原始历史中切出 `recent` 窗口（5 帧），再交给 D04 执行归一化。

两条路径在数学与计算精度上是**逐元素完全等价的**。在系统流水线中，推荐在网络前向入口处集中切片与归一化，保持逻辑紧凑清晰。

## `test_history.py`：11 项单元测试的覆盖范围

| 测试类与具体用例名称 | 核心验证逻辑与断言标准 |
|---|---|
| `SplitWindowsTest.test_frames_0_to_9` | 采用自然帧号 $0 \dots 9$ 验证：`current` 精确为 9，`recent` 精确为 $5 \dots 9$，`previous` 精确为 $4 \dots 8$，张量形状完全吻合 |
| `SplitWindowsTest.test_matches_ppo_training_split` | 随机输入比对：验证与原 PPO 训练切片逻辑在浮点精度下逐元素 100% 相等 |
| `SplitWindowsTest.test_ppo_export_wrapper_on_full_history_repeats_current` | 回归警示测试：重现并记录原版导出包装器在完整输入下产生重复帧 `6, 7, 8, 9, 9` 的历史问题 |
| `SplitWindowsTest.test_does_not_advance_or_copy_history` | 视图共享校验：连续调用两次输出相同，核对底层内存地址，确认不发生显存复制与状态篡改 |
| `SplitWindowsTest.test_rejects_wrong_shape` | 维度防御测试：拦截 9 帧输入（如 `[2, 9, 33]`）或非 33 维特征输入 |
| `ValidityTest.test_auxiliary_needs_six_real_frames` | 联合跨度测试：验证 5 帧能满足决策但不能满足辅助学习，确立 6 帧门槛的必要性 |
| `ValidityTest.test_does_not_use_observation_values` | 逻辑纯粹性测试：验证合法全零观测依然被判定为有效，确认不从数值瞎猜填充 |
| `ValidityTest.test_rejects_out_of_range` | 防御性测试：拦截有效长度为 0 或需求帧数大于 10 的非法参数 |
| `TimeOrderTest.test_consecutive_passes` | 连续时序测试：验证同环境同轮次的步序（如 $7, 8, 9$）顺利过关 |
| `TimeOrderTest.test_detects_problems` | 时序断裂拦截：精准定位并报错跨环境、跨轮次、步序重复或跳跃的具体样本行 |
| `ReplayIntegrationTest.test_windows_on_replayed_batch` | 经验池全链路打通：与 D02、D03、D04 串联运行，验证回放批次的切片、归一化先后等价性与原始数据不变性 |

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
