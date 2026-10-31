# T04 · PPO 与 SAC 对照评价｜AI 实施规范

状态：单元验证通过（CPU、人工记录）。已实现 [evaluate.py](evaluate.py) 与 [test_evaluate.py](test_evaluate.py)，12 项检查通过；真实 PPO / SAC 对照尚未运行，**没有任何性能结论**。

[人类版（讲解与技术复核已完成，本轮不再润色）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

用固定任务和可复查指标判断 SAC 是否学会控制，而非只看训练损失或一段视频。

术语先说明：对照实验：在明确相同的条件下比较两种方法。随机起点：训练中的随机初始化和采样序列。

## 与原 PPO 的关系

保留原 PPO 固定版本作为对照；增加 SAC 的重复训练与相同条件评价。

- [原 PPO：scripts/np3o/play.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/play.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/runner.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/runner.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块已在 CPU 上实现，尚未接入 Isaac 仿真或真实训练结果。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 模型检查点、同一任务设置、评价指令序列、不同随机起点和资源预算。 |
| 输出 | 速度误差、摔倒率、稳定时长、动作平滑性、力矩与耗时内存报告，附视频及配置。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可）；各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [evaluate.py](evaluate.py)

| 符号 | 签名 | 职责与说明 |
|---|---|---|
| `EvaluationEpisode` | actions[T,8]、end_reason、commanded_velocity[T,3]、measured_velocity[T,3]、joint_torques[T,8]（后三项可为 None） | 一个 episode 的逐步记录。end_reason 为 `terminated`（真终止，计为摔倒）、`time_out`（正常到时）或 `unfinished`（评价结束时仍在运行） |
| `EvaluationRecord` | provenance、episodes、control_dt=0.02、interaction_steps、wall_time_s、peak_memory_bytes | 一次评价运行；provenance 必须包含 algorithm、model_source、config_name、code_commit、training_seed |
| `MetricValue` | value、unit、scope、status（measured / missing）、samples | 缺失时 value 为 None、status 为 missing，不用 0 冒充 |
| `validate_record` | `(record) -> None` | 检查结束原因与各信号形状 |
| `compute_metrics` | `(record) -> dict[str, MetricValue]` | 指标与口径见下表 |
| `run_evaluation` | `(policy, env, num_steps, provenance, signals=None, control_dt=0.02) -> EvaluationRecord` | 在 T02 `reset` / `step` 接口的环境上运行确定性策略；`signals(step)` 提供速度指令、真实速度、力矩，未提供时为缺失 |
| `distinct_training_seeds` | `(records) -> list` | 有效（非空）且互不相同的 training_seed；重复种子的多份记录只算一个随机起点 |
| `check_budget` | `(baseline, candidate, budget) -> dict` | 返回 rule、values、comparable、reason。交互步数：两组全部记录相等；耗时：最大值 ≤ 最小值 × (1 + `WALL_TIME_RELATIVE_TOLERANCE`)，容差 0.05 为本项目暂定约定（未经验证）。缺少预算时不可比 |
| `summarize_runs` | `(records) -> dict` | 多次运行的各次值、均值与总体标准差；任一次缺失则为 missing；`training_seeds` 为 `distinct_training_seeds` 的结果 |
| `compare_runs` | `(baseline, candidate, budget, thresholds=None) -> dict` | budget 为 `interaction_steps` 或 `wall_time`。以下任一条件成立时，仍展示各指标差异，但 `conclusion_allowed=False`、全部 verdict 为 None，并在 `conclusion_blockers` 中写明原因：任一组不同的有效训练随机起点少于 3、缺少来源、`check_budget` 不可比、阈值未固定。结果包含 `budget_check` |
| `write_report` | `(records, path) -> dict` | 写 JSON 报告：样本量、指标（缺失写 null）、波动、来源；缺来源，或不同的有效训练随机起点少于 3（与 `compare_runs` 同一口径，不按记录条数）时，`performance_conclusion` 以“不发布”开头 |

### 指标与统计口径

| 指标 | 单位 | 口径 |
|---|---|---|
| `linear_velocity_error` | m/s | ‖(vx, vy)_cmd − (vx, vy)_meas‖₂，对全部 episode 的全部步平均（按步加权） |
| `yaw_rate_error` | rad/s | \|wz_cmd − wz_meas\|，同上 |
| `fall_rate` | 比例 | 真终止 episode 数 / 全部 episode 数；正常到时与 unfinished 都不算摔倒，但计入分母 |
| `time_out_count` / `unfinished_count` | 个 | 单独报告，便于判断评价时长是否足够 |
| `stable_duration` | s | 每个 episode 的步数 × 控制周期，按 episode 平均 |
| `time_to_fall` | s | 摔倒 episode 的平均倒地时间；没有摔倒时为 missing |
| `action_rate` | 策略动作单位² | episode 内相邻步 Σ(a_t − a_{t−1})² 的平均（与原 PPO action_rate 奖励项同形式），不跨 episode 求差 |
| `torque_abs_mean` / `torque_squared_mean` | N·m / N²·m² | 按步与关节平均 |
| `wall_time` / `peak_memory` / `interaction_steps` | s / 字节 / 步 | 记录值，缺失为 missing |

只要有一个 episode 缺少某项信号，该指标就整体标为 missing，不对部分 episode 求平均。

对照与报告的口径：
- 随机起点按 `distinct_training_seeds` 计数：同一个种子的多份记录只算一个起点。
- 所选预算必须可比：交互步数要求完全相等；耗时允许 5% 的相对差异，这是暂定约定，未经验证，会记录在结果的 `budget_check.rule` 中。
- 不满足条件时仍展示差异，只是不判定。

## 必须成立的规则

- 验收阈值在获得 PPO 波动范围后、正式 SAC 比较前固定。
- 建议至少三个不同训练随机起点；演示一次成功不等于性能验证。
- 以各自原生仿真推理先比较，导出差异由 T05 单独检查。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释评价指标的分母与统计范围、随机起点、时间截断和摔倒的区别，以及对照设置固定的原因。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，CPU）

全部使用人工记录与独立手算；`run_evaluation` 的检查使用 T02 假环境。

| 案例 | 对应测试（[test_evaluate.py](test_evaluate.py)） | 结果 |
|---|---|---|
| 线速度误差 = 1/6 m/s、偏航误差 = 0.35/6 rad/s（按步加权，手算） | `MetricTest.test_velocity_errors_step_weighted` | 通过 |
| 真终止、正常到时、未完成各 1 个：摔倒率 1/3（到时不算摔倒）；稳定时长 (2+1+3)×0.02/3 s；倒地时间 0.04 s | `MetricTest.test_end_reasons_fall_rate_and_durations` | 通过 |
| 动作平滑性 (8+2+0)/3，不跨 episode 求差 | `MetricTest.test_action_rate_within_episodes_only` | 通过 |
| 缺少力矩与内存时为 missing（value 为 None）；提供后与手算一致；任一 episode 缺速度信号时整体为 missing | `MetricTest.test_missing_signals_are_marked_not_zero` | 通过 |
| 未知结束原因被拒绝 | `MetricTest.test_rejects_unknown_end_reason` | 通过 |
| 报告：缺 model_source、只有 1 次运行时“不发布”，JSON 中缺失写 null；3 个随机起点且来源齐全时不再阻断 | `ReportAndComparisonTest.test_report_blocks_conclusion_without_provenance_or_seeds` | 通过 |
| 对照：阈值未固定时不判定；固定阈值、3 个随机起点、预算齐全时给出判定；随机起点不足或缺预算时阻断 | `ReportAndComparisonTest.test_compare_requires_fixed_thresholds_seeds_and_budget` | 通过 |
| 回归：两组交互步数相差 1000 倍（1e3 对 1e6）时不判定，verdict 全为 None，差异照常展示 | `BudgetAndSeedRegressionTest.test_interaction_steps_differ_by_1000x` | 通过 |
| 回归：耗时 12.5 s 对 20.0 s 超出 5% 容差时不判定，容差写入 `budget_check.rule`；12.5 s 对 13.0 s（4%）在容差内，可以判定 | `BudgetAndSeedRegressionTest.test_wall_time_outside_tolerance` | 通过 |
| 回归：同一个种子的 3 份记录只算 1 个随机起点，报告“不发布”，对照阻断 | `BudgetAndSeedRegressionTest.test_three_records_with_same_seed_are_one_seed` | 通过 |
| 回归：预算相同、3 个不同种子的正常案例仍可判定，报告不阻断 | `BudgetAndSeedRegressionTest.test_matching_budget_and_three_seeds_still_pass` | 通过 |
| 假环境评价：结束原因为 2 个真终止、1 个到时、2 个未完成；速度指标在假环境中为 missing，提供 signals 后为 measured | `RunEvaluationTest` | 通过 |

调试记录：首次运行时 1 项失败。原因是人工记录为 float32 张量（0.7、0.4 不能精确表示），9 位小数的比较过严；改为 6 位，实现无改动。

这些测试只证明指标计算与报告规则，不包含任何真实 PPO / SAC 结果。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/evaluation/test_evaluate.py -v
```

## 故障定位

奖励提高但运动变差时检查奖励分项、姿态与力矩视频；不立刻认定算法更优。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [T01 版本、依赖与训练配置](../../training_validation/configuration/README.md)
- [T02 采集与学习调度](../../training_validation/training_loop/README.md)
- [T03 训练进度保存与恢复](../../training_validation/checkpoints/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU）；Isaac 仿真、真实训练与性能验证未开始 |
| 实际文件、函数签名与调用关系 | [evaluate.py](evaluate.py)：见上表；`run_evaluation` 使用 T02 环境接口（`reset` / `step`）。[test_evaluate.py](test_evaluate.py)：4 个测试类共 12 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`write_report` 写文件，`run_evaluation` 调用环境；其余函数只读 |
| 代码注释核对 | 已核对：指标的分母与统计范围、时间截断与摔倒的区别、随机起点与阈值固定的原因（文件头、`compute_metrics`、`compare_runs`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/evaluation/test_evaluate.py -v` → `Ran 12 tests … OK`（首次实现时为 8 项，其中 1 项曾因测试精度失败，见“调试记录”；预算与随机起点修复后新增 4 项回归）。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | Isaac 仿真中的 PPO 与 SAC 评价运行（速度指令、真实速度、力矩信号来自 E01 / E02）；评价指令序列与时长；PPO 波动范围与验收阈值；视频记录；真实耗时与内存 |
| 代码版本 | `a8a44fc77b387a9d853e4e290843133fadfeb2cc`，预算与随机起点修复 `8ead2dff4b33ed8229ddcc956a567f856d237219`（分支 `claude/training-core`，PR #7）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 T04：只提供评价数据接口、指标、报告与对照规则；证据不足时阻断性能结论。审核修复：原 `compare_runs` 只检查预算有没有值，现要求所选预算可比（交互步数相等；耗时在 5% 暂定容差内）；原 `write_report` 按记录条数判断随机起点，现与 `compare_runs` 一样按不同的有效 training_seed 计数 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `evaluate.py`、`test_evaluate.py`；原 PPO `play.py`。建议 Codex 重点解释各指标的口径、摔倒与到时的区别、缺失标记，以及阻断结论的条件 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；核对版本 a07d7b2，材料、证据及限制见本目录 README_HUMAN.md |
| 技术初稿 / 语言润色 | Codex 已完成中文讲解与技术核对；按本轮用户要求不再交 Gemini |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
