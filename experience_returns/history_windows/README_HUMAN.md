# D05 · 历史窗口与有效性｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 技术初稿完成，待 Gemini 润色 |
| 技术核对状态 | 已核对切片、视图、有效长度、时间编号与 11 项测试；本次修正文件头对原导出包装器的错误说明，不修改算法 |
| 语言润色状态 | 待 Gemini 润色，尚未润色 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；算法与测试来自 `807ee636b1d40ff4be86ed27f9ce9ecd9f5afe2a`。本次工作区仅另修正 `history.py` 模块说明，执行语句保持不变，提交后由 Git 历史定位该注释版本 |
| 待同步代码版本 | 无；本页已反映指定文件头修正 |
| 实际阅读材料 | 本目录 AI README、[history.py](history.py)、[test_history.py](test_history.py) 全文；D01～D04 实现及测试；固定 PPO `actor_critic.py` 的 `_split_policy_obs`、`BarlowTwinsActor.forward`、`_InferenceWrapper`，T05 契约。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 11 项记录见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。本次只核对文档/注释，不重跑算法测试；人工数据，未接网络、导出入口、仿真或训练 |

## 已有十帧历史，为什么还要规定窗口

环境给出包含当前帧的十帧历史 `[B,10,33]`。历史编码器不一定把十帧全部吃进去：当前设计用五帧作决策窗口，辅助学习再取前移一帧的另一个五帧窗口。窗口就是连续片段；有效长度则说明其中多少帧属于本轮真实观测，其余为重置填充。

D05 只做切片和检查，不推进历史、不追加当前帧、不计算编码或辅助损失。Barlow Twins 是后续辅助学习方法，使用相邻两段窗口形成两种视图；这里仅准备输入，尚未实现其网络与损失。

## 输入输出与时间位置

设当前时刻为 t，索引 0..9 对应 `t-9..t`，输入已有当前帧：

| 输出字段 | 切片 | 时刻 | 形状 |
|---|---|---|---|
| `current` | `history[:,9]` | t | `[B,33]` |
| `recent` | `history[:,5:10]` | t-4..t | `[B,5,33]` |
| `previous` | `history[:,4:9]` | t-5..t-1 | `[B,5,33]` |

用测试中的帧编号表示：

```text
完整历史：0 1 2 3 4 5 6 7 8 9
previous：        4 5 6 7 8
recent：            5 6 7 8 9
current：                   9
```

Python 切片右端不包含在内，所以 `4:9` 取 4..8。两个五帧窗口重叠四帧，合起来覆盖 **六个不同的时刻** 4..9，不是五个，也不是十个。因此辅助学习需要至少六帧真实历史。

`split_windows` 接收原始或已归一化历史，形状必须为 `[B,10,33]`；不检查内容是否真的来自正确时刻。输入应由 D01/E02/E05 提供有效的字段与长度。

## `history.py`：逐函数职责

| 符号 | 输入、输出与限制 |
|---|---|
| `HistoryWindowError` | 历史形状、有效长度或时间编号不符时的异常，无自定义方法 |
| `WINDOW_FRAMES=5`、`AUXILIARY_REQUIRED_FRAMES=6` | 编码窗口帧数、两窗口联合需要的真实时刻数，后者由前者+1 得到 |
| `HistoryWindows` | 冻结数据类，保存 `current`、`recent`、`previous` 三个张量；全是输入的视图，不分配独立观测副本 |
| `split_windows(history)` | 检查三维布局和末两维 `(10,33)`，返回上述三段；不改变输入、不增加历史帧 |
| `valid_history_mask(history_length, required_frames=6)` | 输入 `[B]` 有效长度，检查长度和要求均在 1..10，返回 `[B] bool` 的 `length>=required`；不接收观测，因此不会根据数值猜填充 |
| `validate_time_order(env_id, episode_id, step_index)` | 输入形状相同的 `[M,L]` 编号，检查每行同环境、同轮、步号逐条+1；正常返回 None，失败给出问题行号。L=1 没有相邻连接可比较，不证明编号或观测来源整体正确 |

除返回视图外没有历史状态更新；这些函数不调用 D04 或网络。当前生产网络与 T05 尚未接入，实际衔接体现在测试中。

## 关键切片为什么不追加当前帧

实际执行代码为：

```python
current=history[:, end - 1],
recent=history[:, end - WINDOW_FRAMES : end],
previous=history[:, end - WINDOW_FRAMES - 1 : end - 1],
```

`end=HISTORY_LENGTH=10`。第一行取末帧；第二行取 `[10-5,10)`，即 5..9；第三行把两端都前移一位，取 `[4,9)`，即 4..8。没有 `cat` 追加帧，因为历史已经包含 t，再追加一次就会重复当前时刻。

三段是普通切片视图，与输入共享存储。测试连续调用两次得到相同窗口，输入不变，并核对当前帧存储地址；这证明函数没有推进历史，也没有另存一份。调用方若原地修改视图，会影响输入；冻结数据类只禁止重绑字段，不使张量只读。需要独立副本时由调用方显式复制。

## 原 PPO 训练一致，原导出路径并不一致

本次实际读取的固定 PPO 快照 `8fe40aa` 中：

1. `_split_policy_obs` 从完整历史取 `current=obs[:,-1]` 和 `hist=obs[:,-6:-1]`，即 t 与 t-5..t-1。
2. `BarlowTwinsActor.forward` 把 `hist[:,1:]` 与 current 拼接，得到 t-4..t；D05 的 recent 与它一致，previous 与 hist 一致。
3. `_InferenceWrapper` 却对**完整历史**执行 `history[:,1:]`，再拼 current，最后取五帧。帧号输入 0..9 会得到 `6,7,8,9,9`，而非 `5,6,7,8,9`。

这就是文件头本次修正的含义：**本模块与原 PPO 训练路径一致；原导出包装器在输入含当前帧的完整历史时会重复当前帧，该问题已记录，留待 T05 处理。**

`test_ppo_export_wrapper_on_full_history_repeats_current` 是对既有问题的回归记录，确认它出现，不能解读成导出行为已正确或已修复。当前没有新的导出入口，也未做模型输出等价验证；后续 T05 应采用一致的窗口契约并验证完整推理路径。此处属于保留 TITA/PPO 历史编码的适配，不是作者 SAC 已验证的 TITA 历史方案。

## 有效长度与时间编号分别能证明什么

默认辅助标记只有这行：

```python
return history_length >= required_frames
```

测试长度 `[1,5,6,10]` 得 `[False,False,True,True]`；若仅要求五帧，则为 `[False,True,True,True]`。五帧已够 recent，但 previous 还可能含填充，所以不能拿“决策窗口够长”代替“两窗口都有效”。

真实观测可以全零，填充也可能非零；本函数根本不接收观测内容。标记依赖上游提供的真实帧数，不证明窗口内容或重置来源正确。处理 D03 时，当前输入用 `first.history_length`；尾部状态要用 `tail.next_history_length`，不能误用 tail 当前长度。

时间顺序检查的关键一行是：

```python
bad = (step_index[:, 1:] != step_index[:, :-1] + 1).any(dim=1)
```

逐对比较相邻步号是否恰好+1，再标记有问题的样本行；另外比较每行环境号、轮次号是否都与第一列一致。例如环境 `[2,2,2]`、轮次 `[5,5,5]`、步号 `[7,8,9]` 通过；步号 `[0,1,1]` 为重复，`[0,2,3]` 为跳跃，跨环境或跨轮次也拒绝。

用于 D02 输出时，只把 `valid` 的有效前缀交给时间检查，无效尾部可能是别轮或空槽。D02 还有覆盖序号判断，D05 的编号检查不代替它；编号连续也不能独立证明观测内容一致。

## 与 D04 配合：同一统计下切片先后等价

D04 在最后的 33 个通道上逐帧变换，且 `normalize_policy` 不更新统计。因此对**同一份固定统计**：先归一化十帧再切 recent，与先切 recent 再归一化逐元素相同。若两次之间更新了统计，就不满足这个前提。

回放测试走 D01→D02→D03，用 D04 归一化 `first.policy_history` 后切片；也检查 `tail.next_policy_history` 的 previous 形状为 `[40,5,33]`，当前/下一有效标记分别来自各自长度。原始记录保持不变。这是人工 CPU 衔接，不是编码网络、仿真或导出集成验证。

## `test_history.py`：逐项检查

`numbered_history(batch_size=2)` 生成 `[B,10,33]`、第 i 帧各通道都为 i 的人工输入；`frame_ids(window)` 取第 0 个样本、第 0 个通道读出帧号。两者帮助直接辨认时间位置。

| 测试类/方法 | 预期 |
|---|---|
| `SplitWindowsTest.test_frames_0_to_9` | current=9、recent=5..9、previous=4..8，形状正确 |
| `SplitWindowsTest.test_matches_ppo_training_split` | 随机历史上与原训练切片及决策拼接逐元素一致 |
| `SplitWindowsTest.test_ppo_export_wrapper_on_full_history_repeats_current` | 原导出写法得到 6,7,8,9,9，保留问题记录 |
| `SplitWindowsTest.test_does_not_advance_or_copy_history` | 不推进/修改历史，返回视图 |
| `SplitWindowsTest.test_rejects_wrong_shape` | `[2,9,33]` 被拒绝 |
| `ValidityTest.test_auxiliary_needs_six_real_frames` | 长度 1/5/6/10 与要求五帧、六帧的差别 |
| `ValidityTest.test_does_not_use_observation_values` | 长度 10 的全零真实观测仍有效，长度 3 不够辅助学习 |
| `ValidityTest.test_rejects_out_of_range` | 长度 0、要求 11 帧均拒绝 |
| `TimeOrderTest.test_consecutive_passes` | 同环境同轮的 7,8,9 及单条输入通过 |
| `TimeOrderTest.test_detects_problems` | 准确定位跨环境、跨轮次、步号重复/跳跃行 |
| `ReplayIntegrationTest.test_windows_on_replayed_batch` | 经 D02～D04 衔接后核对有效前缀、窗口、归一化先后及原始记录不变 |

测试入口为 `unittest.main()`。11 项 CPU 通过记录来自原验证，本次未重跑。需要复现可在根目录执行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/history_windows/test_history.py -v
```

导出表现不同时先查输入历史是否已含当前帧、是否重复拼接，再看归一化统计；辅助样本异常先查有效长度与轮次编号。E05 的填充规则、N01/N02 的真实使用、T05 导出一致性、GPU、仿真、训练与性能都还未验证。

## 阅读导航与润色交接

[D01](../transition_schema/README_HUMAN.md) 定义历史与长度，[D02](../replay_buffer/README_HUMAN.md)、[D03](../n_step_returns/README_HUMAN.md) 提供起点和尾部，[D04](../normalization/README_HUMAN.md) 提供逐帧只读变换；[T05 AI 契约](../../training_validation/export_inference/README.md) 记录导出后续工作。

Gemini 保留三个切片、四帧重叠/六时刻联合、视图共享存储、长度来源限制，以及“训练一致、导出问题尚待处理”的区别。
