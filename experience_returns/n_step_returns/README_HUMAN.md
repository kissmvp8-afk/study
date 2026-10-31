# D03 · 连续多步回报｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 技术初稿完成，待 Gemini 润色 |
| 技术核对状态 | 已核对实际函数、7 项测试、D02 有效片段规则和 D01 终止开关；未开展全面代码审核 |
| 语言润色状态 | 待 Gemini 润色，尚未润色 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现/测试来自 `c3b2fe09b752cc8f0ee713d674af5b057c675558`，读取时无后续源码变化 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 本目录 AI README、[returns.py](returns.py)、[test_returns.py](test_returns.py) 全文；D01/D02 实现与测试，D04/D05 衔接测试；固定 PPO `rollout_storage.py` 的 `compute_returns`。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 7 项原结果见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。人工数据，无目标 Q、仿真或训练；本次未重跑算法测试 |

## n 步回报解决什么问题

单步记录只保存一次动作后的奖励。多步回报把同一环境、同一轮的若干次奖励加在一起，让起点学习到较后才发生的结果。“折扣”用 γ 的幂降低较远奖励的权重；“尾部”是这些动作之后接价值预测的位置。

设最多使用 n 步，实际使用 k 步（1≤k≤n）：

```text
R = r0 + γr1 + … + γ^(k-1)r(k-1)
后续 N05 的目标形式 = R + γ^k × mask × V(尾部状态)
```

r0 是起点动作**之后**收到的奖励。D03 只构造 R、k、折扣和尾部数据，不调用 Q 网络；V 的具体 SAC 计算留给 N05。历史窗口中的“过去 5 帧”与回报中的“之后 5 次动作”是不同时间方向，不能共用含义。

## 输入与输出：尾部记录不等于尾部状态

输入是 D02 `SequenceBatch`：M 个起点，每个读取 n 条记录；奖励为 `[M,n,1]`，结束标记及 `valid` 为 `[M,n]`。n 必须与读取长度相同，γ 在 `(0,1]`。

| `NStepBatch` 字段 | 形状及用途 |
|---|---|
| `first` | D01 `[M,...]` 起点记录；Q 评价 `first.q_observation`、`first.action`，编码器使用其策略历史 |
| `tail` | 第 k-1 条记录，仍是 D01 `[M,...]`；目标状态使用它的 `next_policy_history`、`next_q_observation`、`next_history_length` |
| `reward_sum` | `[M,1] float32`，累计的 R |
| `steps` | `[M] int64`，每个样本自己的 k |
| `discount` | `[M,1] float32`，γ^k，不总是 γ^n |
| `bootstrap_mask` | `[M,1] float32`，尾部真终止为 0，其余为 1 |

例如起点在 t，取 k=2 条转移：第一条描述 `t→t+1`，第二条描述 `t+1→t+2`。`tail` 是第二条记录，但估值状态必须取 `tail.next_*`，即 t+2，而不是它的当前观测 t+1。

## `returns.py`：逐函数职责

| 符号 | 输入、输出、调用与边界 |
|---|---|
| `NStepReturnError` | γ、长度或有效前缀不符合约定时的异常，无自定义方法 |
| `NStepBatch` | 冻结数据类，保存上表六项；没有计算方法，冻结不使内部张量只读 |
| `accumulate_rewards(rewards, boundary, gamma)` | 接收 `[M,n,1]` 奖励、`[M,n] bool` 结束边界与 γ，返回 `(R[M,1], k[M])`；包括首个结束步的奖励，屏蔽之后奖励。接口预期由 D02 提供正确布局，不在这里重新执行 D01 全字段校验 |
| `select_tail(sequence, steps)` | 调用 `sequence.at(steps-1)` 返回每行尾部记录；无效位置会由 D02 抛 `ReplayBufferError`，不会替换末状态内容 |
| `build_n_step_batch(sequence, gamma, n_steps)` | 检查读取长度与 n 一致，构造有效结束标记，调用累计函数，确认需要的 0..k-1 条都有效，再调用 `select_tail`、D01 `derive_bootstrap_mask`，组合 `NStepBatch` |

这些函数不修改输入、经验池或网络参数。模块通过 `sys.path` 导入 D01/D02；当前实际使用者是测试及其他模块的纯数据衔接检查，N05 尚未接入。

## 如何找到 k，再给奖励加权

`accumulate_rewards` 先找首个边界：

```python
has_boundary = boundary.any(dim=1)
first_boundary = torch.argmax(boundary.to(torch.int64), dim=1)  # 无边界时为 0，下面用 has_boundary 区分
steps = torch.where(has_boundary, first_boundary + 1, torch.full_like(first_boundary, n))
```

第一行确认每行是否真的有结束。第二行把真假转成 0/1，`argmax` 找第一个 1；全零时也返回位置 0，不能把它误当成第一步结束。第三行据 `has_boundary` 决定取“首边界索引+1”还是完整 n，避免这个歧义。

随后给每个偏移 j 构造权重：j<k 用 γ^j，j≥k 用零，再沿时间维求和：

```python
reward_sum = (rewards.squeeze(-1) * weights).sum(dim=1, keepdim=True)
```

`squeeze(-1)` 去掉奖励末尾的单列，与 `[M,n]` 权重对齐；求和只消去时间维，`keepdim=True` 保留 `[M,1]`，方便后续与折扣、mask 相乘。边界步属于 j=k-1，所以它的奖励不会被遗漏。

现有测试的无结束例子为奖励 `1,2,3,4,5`、γ=0.9、n=5：

```text
R = 1 + 0.9×2 + 0.81×3 + 0.729×4 + 0.6561×5 = 11.4265
k = 5，discount = 0.9^5 = 0.59049，mask = 1
tail.step_index = 4，尾部状态来自该条记录的 next_*
```

n=1 的测试用首奖励 7、γ=0.99，输出 R=7、k=1、discount=0.99，退化为单步，而不是把折扣也去掉。

## 真终止与纯截断：都停止累加，开关不同

边界测试使用 n=5、γ=0.5、每步奖励 1，并在下一轮放入 `1e6` 奖励作为污染探针。

| 首个结束在第几次动作后 | k | R | γ^k | 真终止 mask | 纯时间截断 mask |
|---|---:|---:|---:|---:|---:|
| 第 1 次 | 1 | 1 | 0.5 | 0 | 1 |
| 第 2 次 | 2 | 1.5 | 0.25 | 0 | 1 |
| 第 5 次 | 5 | 1.9375 | 0.03125 | 0 | 1 |

两种结束都不允许把下一轮的 `1e6` 加进 R。真终止不接未来；纯时间截断允许从**旧轮次重置前末状态**接估值。mask=1 不是跨轮次继续取奖励的许可；末状态由 E05 捕获，D03 只是选对传入记录，不能证明其观测来源。

## 数据不足为什么不是一种结束

`build_n_step_batch` 把结束标记与 `sequence.valid` 相与，只承认有效记录上的边界。得到 k 后，执行：

```python
needed = torch.arange(n_steps, device=steps.device).unsqueeze(0) < steps.unsqueeze(1)
missing = (needed & ~sequence.valid).any(dim=1)
```

第一行逐行标出前 k 条必须存在的记录；第二行检查这些位置是否有无效项。缺少就报错，不把 k 改小来迁就缺数据。

测试只写 3 条、未结束，却请求 n=5：D02 返回零个有效起点，强行读 5 条再构造 D03 也会报错。补写 step 3、4 后才有一个五步起点。相反，若第 2 条已经合法结束，则 k=2，不需要等后面三条。这与 D02 的 `valid_starts(n)` 是同一条可用性规则。

折扣最终按各自 k 计算：

```python
discount=(gamma ** steps.to(torch.float64)).to(torch.float32).unsqueeze(1),
```

先用 float64 求幂，再回到训练字段的 float32 和 `[M,1]`；不能统一使用 γ^n，否则提前结束样本的未来权重会出错。

## 与 PPO、作者 SAC 的关系

原 PPO 的 `compute_returns` 反向递推广义优势估计（GAE），用于近期策略数据；D03 直接从可回放片段构造 Q 学习的 n 步奖励部分，两者都利用多步信息，目标与公式不同。

按本地实现文件头和 AI 规范，作者 SAC 也在第一个结束之后屏蔽奖励、按结束位置选择下一观测；本项目沿用这一思路，但有效起点改用 D02 的显式编号和覆盖序号，mask 调用 D01 的真终止规则。作者参考版本为 `e0d243a`，本次未独立读取外部作者源码，不把这些说明扩展成作者对 TITA 的验证结论。

“离策略”表示片段中间的动作由过去的策略产生。未经校正的多步目标可能与当前策略不完全一致，所以 n 更大不自动更好；n=1 可作排查对照，n=5 只是论文候选，训练效果尚未验证。

## `test_returns.py`：逐项验证与边界

`buffer_with_episode(rewards, terminal_at, time_out_at, next_episode_steps)` 经 D01/D02 构造第 0 轮及奖励为 `HUGE=1e6` 的下一轮；`n_step_from_episode_start(buffer,n,gamma)` 找第 0 轮 step 0 这个有效起点，再读取、构造 D03。记录复用 D02 测试的 `make_records`，属于人工数据，不是真实机器人轨迹。

| 测试类/方法 | 独立预期 |
|---|---|
| `AccumulateTest.test_known_sequence_without_boundary` | 手算 11.4265、k=5、折扣 0.59049，尾部为 step 4 的 next 状态 |
| `AccumulateTest.test_n_equals_one_is_single_step` | R=7、k=1、折扣 0.99 |
| `AccumulateTest.test_invalid_gamma` | γ=0 报错 |
| `BoundaryTest.test_boundaries_at_steps_1_2_n` | 第 1/2/5 步×真终止/纯截断共六种，检查 R、k、折扣、尾部与 mask，排除下一轮大奖励 |
| `MissingSuccessorTest.test_start_without_enough_successors` | 未结束但仅有三步不可构造五步目标，补写后可用 |
| `MissingSuccessorTest.test_length_must_match_n` | 读两条却声明 n=3，报错 |
| `SampledBatchTest.test_sampled_starts_from_mixed_buffer` | 多环境、多轮、循环覆盖后抽 50 个起点，逐行核对手算 R、编号和 mask |

入口为 `unittest.main()`。原交接记录了首次 `SampledBatchTest` 因容差过严失败：float32 在 `1e6` 附近的间隔约 0.0625，比绝对容差 `1e-3` 大；改为随数值量级的容差后七项通过。这是既有调试记录，不是本次重现结果。

需要复现时，在根目录运行既有命令；本次不执行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/n_step_returns/test_returns.py -v
```

当前证据只覆盖人工 CPU 数据及 D02→D03 衔接，未覆盖 GPU、目标 Q 计算、离策略偏差影响、仿真、训练或性能。重置附近异常时依次查 k、`tail.next_*`、两种结束标记及原始奖励，不能只看 n。

## 阅读导航与润色交接

[D02](../replay_buffer/README_HUMAN.md) 解释连续起点与覆盖；[D01](../transition_schema/README_HUMAN.md) 解释字段和 mask；[D04](../normalization/README_HUMAN.md)、[D05](../history_windows/README_HUMAN.md) 处理 `first.policy_history`、`tail.next_policy_history` 等输入。N05 尚未实现，不可把本页目标示意当作可运行 SAC 更新。

Gemini 只改善表达，保留奖励从动作后开始、k-1 与 next_* 的区别、γ 的幂、六种边界数字、失败记录及未验证项。
