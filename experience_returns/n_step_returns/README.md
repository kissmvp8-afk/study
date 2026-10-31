# D03 · 连续多步回报｜AI 实施规范

状态：单元验证通过（纯数据范围）。已实现 [returns.py](returns.py) 与 [test_returns.py](test_returns.py)，7 项 CPU 检查通过（Python 3.11 + PyTorch）；使用人工数据并经 D02 读取，未接入 N05 目标 Q，未进行训练或性能验证。

[人类版（讲解与技术复核已完成，本轮不再润色）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

累计连续几步的真实奖励，输出用于 Q 目标的奖励部分和正确的尾部状态。

术语先说明：n 步表示使用最多 n 次连续交互；折扣使越远的奖励权重越小；尾部是这段经历之后接预测的位置。

## 与原 PPO 的关系

用 SAC 的 n 步目标取代原 PPO 的 GAE 目标；两者都用多步信息，但计算与用途不同。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/rollout_storage.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/storage/replay_buffer.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/storage/replay_buffer.py)
- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 同环境按时间排列的经历、折扣 gamma、最大步数 n。 |
| 输出 | 折扣奖励和、有效步数 k、尾部观测、是否接后续价值的开关；本模块不调用 Q。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [returns.py](returns.py)

| 符号 | 签名 | 职责、单位、异常与副作用 |
|---|---|---|
| `NStepReturnError` | `ValueError` 子类 | 输入不符合约定 |
| `accumulate_rewards` | `(rewards[M,n,1], boundary[M,n] bool, gamma) -> (R[M,1], k[M] int64)` | R = r0 + γ·r1 + … + γ^(k-1)·r(k-1)。k 为首个边界位置 + 1，没有边界时为 n；边界之后（新一轮）的奖励权重为 0。gamma 须在 (0, 1] |
| `select_tail` | `(sequence: SequenceBatch, steps[M]) -> TransitionBatch` | 取每行第 k-1 条记录；它的 `next_*` 是 k 次动作后的真实状态（纯截断时是重置前末状态） |
| `build_n_step_batch` | `(sequence, gamma, n_steps) -> NStepBatch` | 检查 `sequence.length == n_steps`，且起点到第 k-1 条全部有效；否则报“数据不足不能当作结束”。mask 调用 D01 `derive_bootstrap_mask(tail)` |
| `NStepBatch` | `first, tail: TransitionBatch`；`reward_sum[M,1]`、`steps[M]`、`discount[M,1]=γ^k`、`bootstrap_mask[M,1]` | Q 评价 `first.q_observation`、`first.action`；目标用 `tail.next_*`，并按 R + γ^k · mask · V(尾部) 组合（V 由 N05 计算，本模块不调用 Q） |

与 D02 的规则统一：D02 `valid_starts(n)` 只返回“凑满 n 条或在边界结束”的起点；`build_n_step_batch` 用同一个 `SequenceBatch.valid` 再检查一次，任何不满足的起点都会报错，而不是被截短。n=1 时 k 恒为 1，退化为单步。

调用方：目前只有本目录与 D04、D05 的测试。N05 Critic 学习尚未实现。

## 必须成立的规则

- 不足 n 步时按实际 k 使用 gamma^k。
- 真正终止丢弃尾部价值；纯截断保留末状态估值，但不继续加新一轮奖励。
- 5 帧观测是过去输入，5 步回报是后来结果，不能共用同一个窗口含义。

## 算法约定与实现提示

实际收集到 k 步时，奖励部分为 R=r0+gamma×r1+…+gamma^(k-1)×r(k-1)。r0 是第一个动作后的奖励；gamma 是折扣比例；k 不超过配置的 n。尾部对应这 k 次动作之后的状态，未来预测由 N05 接上。

先遇到真终止或时间截断就停止累计；普通数据尾部还没有未来经历时，第一版等后继写入后再抽取，不能悄悄把缺数据当成任务结束。发生提前结束时则按实际 k 输出。

回放中的中间动作由过去的策略产生，未经校正的多步目标可能存在偏差。因此把 n=1 作为排查对照，n=5 作为论文候选；不能认定步数越多越好。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释累计索引、gamma 的幂、实际 k、尾部选择和停止条件；说明缺少后继为何不等于真正终止。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，纯数据范围）

全部使用人工数据，经 D01 `make_transition` 创建，经 D02 写入和读取。

| 案例 | 对应测试（[test_returns.py](test_returns.py)） | 结果 |
|---|---|---|
| 无结束奖励 1..5、γ=0.9、n=5：R=11.4265（手算），k=5，γ^5=0.59049，mask=1，尾部是 step 4 | `AccumulateTest.test_known_sequence_without_boundary` | 通过 |
| n=1 退化为单步：R=r0，k=1，discount=γ | `AccumulateTest.test_n_equals_one_is_single_step` | 通过 |
| 在第 1、2、n 步分别放置真终止和纯截断（共 6 种）：R、k、γ^k 与手算一致；真终止 mask=0，纯截断 mask=1；尾部是第 0 轮末条记录的重置前状态；下一轮的 1e6 奖励未进入 R | `BoundaryTest.test_boundaries_at_steps_1_2_n` | 通过 |
| 只有 3 步、未结束：n=5 时没有有效起点，强行构造会报错；补写后继后即可使用 | `MissingSuccessorTest`（2 项） | 通过 |
| D02 → D03：多环境、多轮、下一轮首步大额奖励，抽 50 个起点全部可构造，R、k、尾部编号与 mask 都和手算一致 | `SampledBatchTest` | 通过 |
| γ 不合法时报错 | `AccumulateTest.test_invalid_gamma` | 通过 |

调试记录：第一次运行时 `SampledBatchTest` 失败。原因是测试容差 1e-3 小于 float32 在 1e6 附近的最小间隔（约 0.06），实现本身无误；改为相对容差后通过。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/n_step_returns/test_returns.py -v
```

## 故障定位

损失在重置附近异常时检查是否用固定 n 替代有效 k，或把重置后奖励接进旧轮次。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [D01 经历记录契约](../../experience_returns/transition_schema/README.md)
- [D02 经验池写入与抽样](../../experience_returns/replay_buffer/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（纯数据范围）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [returns.py](returns.py)：`accumulate_rewards`、`select_tail`、`build_n_step_batch`、`NStepBatch`、`NStepReturnError`；依赖 D02 `SequenceBatch`、D01 `TransitionBatch`、`derive_bootstrap_mask`。[test_returns.py](test_returns.py)：4 个测试类共 7 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；全部函数无副作用，读取结果本身已是 D02 返回的副本 |
| 代码注释核对 | 已核对：R 的公式与 γ^k（文件头、`accumulate_rewards`）；遇到真终止或纯截断就停止、下一轮奖励权重为 0（`accumulate_rewards`）；尾部为重置前状态（`select_tail`）；数据不足不当作结束（`build_n_step_batch`）；与作者实现的关系与离策略偏差说明（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU）；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/n_step_returns/test_returns.py -v` → `Ran 7 tests … OK`（首次运行有 1 项因测试容差失败，见上文“调试记录”）。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除，不影响结果 |
| 未验证项 | N05 用目标 Q 组合目标；n=5 与 n=1 的训练对照；离策略多步偏差的影响；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`c3b2fe09b752cc8f0ee713d674af5b057c675558`（分支 `claude/experience-returns`，基于 `c06509e`）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 D03：沿用作者“首个结束之后屏蔽奖励、按首个结束取尾部”的思路，但起点有效性改用 D02 的显式编号，mask 由真终止推导 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；D01 `schema.py`；D02 `buffer.py`；带注释的 `returns.py`、`test_returns.py`；作者 `replay_buffer.py` 的 `_generate_batch` 与 `sac.py` 的目标组合。建议 Codex 重点解释 `accumulate_rewards` 中权重的构造、`build_n_step_batch` 的有效性检查，以及尾部为何取第 k-1 条记录的 `next_*` |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；面向有 Python 基础的读者，实际核对版本与限制见本目录 README_HUMAN.md |
| 技术初稿 / 语言润色 | Gemini 整体润色已完成，Codex 后续技术修正已复核；按用户决定本轮不再局部润色 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
