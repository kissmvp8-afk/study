# T04 · PPO 与 SAC 对照评价｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex 写作约定](../../CODEX.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 对照实际实现、注释与测试核对职责、调用、形状、单位、代码片段和边界；本次是文档核对，不是全面代码审核 |
| 语言润色状态 | 本轮由 Codex 完成中文讲解与技术核对，不另交 Gemini |
| 已核对代码版本 | `a07d7b22e5bc2f40c462c01cc3378f7bcf3fe0c7`（`claude/training-core`，PR #7）；读取时与用户验收版本、远端分支一致，无后续源码差异 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | AGENTS.md、CODEX.md、根及本大模块/本目录 AI README；[evaluate.py](evaluate.py)、[test_evaluate.py](test_evaluate.py) 全文；T02 环境接口、T03 模型来源与 T05 确定性动作接口；依赖与来源详见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 本目录 AI 交接表记录 12 项 CPU 检查通过（Python 3.11.17 + torch 2.15.0），本次阅读测试、引用既有结果，未重跑算法测试。本轮执行的文档检查见大模块记录；人工数据/假环境不代表 Isaac 仿真、GPU、机器人训练或性能验证 |

## 评价的是控制结果，不是损失大小

PPO 与 SAC（软演员—评论家算法）的损失定义不同，数值不能直接比较。T04 把一段评价中的逐步记录转成有单位、有分母、有来源的指标，再检查能否发布对照判定。当前只有人工记录和 CPU 假环境检查，没有真实 PPO/SAC 对照结果。

一次 episode 指从开始到结束的一轮交互；训练随机种子决定训练初始化和采样的起点；墙钟时间指现实中经过的秒数。多个训练种子用于观察训练结果的波动，同一模型多拍几个视频不能增加训练种子数。

## `evaluate.py` 保存什么数据

| 对象 / 属性 | 输入内容、输出与用途 |
|---|---|
| `EvaluationEpisode` | 一个 episode 的 `actions[T,8]` 和 `end_reason`；可选 `commanded_velocity/measured_velocity[T,3]`、`joint_torques[T,8]` |
| `EvaluationEpisode.steps` | 返回动作张量的 T，即本 episode 实际记录的步数 |
| `EvaluationRecord` | 一次评价的来源字典、episode 元组、控制周期（默认 0.02 s）、可选训练交互预算 `interaction_steps`、训练墙钟 `wall_time_s`、峰值字节 `peak_memory_bytes` |
| `MetricValue` | 单个指标的 `value/unit/scope/status/samples`，让数值、单位、口径、样本量可追溯；`missing(unit,scope)` 返回值为 None、样本数 0 的缺失标记 |

速度每行是身体（偏航对齐）坐标系的 `(vx,vy,wz)`，前两项 m/s，第三项 rad/s；力矩为 N·m；动作是 E03 顺序的策略坐标，无物理单位。这里的速度信号必须是对应的物理量，不能直接把乘过 2.0 的 Q 速度标签当作 m/s。

来源必须有 `algorithm/model_source/config_name/code_commit/training_seed`。结束原因只有 `terminated`、`time_out`、`unfinished`。当前评价口径把 `terminated` 计为摔倒；后续接入其他真终止原因时需先确认这一口径。到时和评价截止时仍运行都不计为摔倒，但会单列数量。

## 从一次 rollout 到指标报告

rollout 指连续执行策略并记录结果；确定性策略指固定参数、相同观测得到相同动作，不额外采样噪声。`run_evaluation(policy,env,num_steps,provenance,signals=None,control_dt=0.02)` 调用 T02 的 `reset/step`，在 `no_grad` 下运行策略，逐环境积累数据；内部 `close(env_index,reason)` 把该环境积累的列表堆叠成一个 episode，再清空列表。先判断真终止，再判断到时，评价截止时仍有数据的部分记 `unfinished`。

策略接口是 `policy(observation)->[E,8]`，应返回确定性动作。若直接用 T05 模型，要适配为 `lambda obs: exported(obs.policy_history)`，因为 T05 接收张量而非整个 `EnvObservation`。

`signals(step)` 可提供三类物理信号；没有提供则保留 None。假环境默认没有速度指令、真实速度、力矩，不能用零补成“测量为零”。`run_evaluation` 返回的记录也没有自动计时或填写训练预算，`interaction_steps/wall_time_s/peak_memory_bytes` 仍为 None；调用方需另行记录真实训练资源并组装完整 `EvaluationRecord`。训练交互预算不是本次评价收集了多少步。

## 指标的分母决定数字的意义

`validate_record(record)` 检查结束原因、动作 `[T,8]`（T≥1）和可选信号的同长形状，错误抛 `ValueError`。`compute_metrics(record)` 先校验，返回指标名到 `MetricValue` 的字典，不改原记录。辅助函数 `_all_present(episodes,name)` 要求非空且每个 episode 都具备该信号；任一段缺失就把整项标为 missing，不选择性计算有数据的部分。

[test_evaluate.py](test_evaluate.py) 的 `manual_record` 构造 2 步真终止、1 步到时、3 步未完成，共 6 步。它用于讲清口径，不是机器人实测。

| 指标 | 这组人工数据怎样计算 |
|---|---|
| `linear_velocity_error` | 六步二维速度差模长为 `0.5,0,0.5,0,0,0`，平均 1/6 m/s；不是先对三个 episode 平均再等权平均 |
| `yaw_rate_error` | 绝对偏航差总计 0.25+0.1=0.35，除以 6，约 0.05833 rad/s |
| `fall_rate` | 真终止 1 / 全部 episode 3 = 1/3；分母包含 unfinished，需同时看 `unfinished_count=1` 和 `time_out_count=1` |
| `stable_duration` | 每个 episode 的时长等权平均：`(2+1+3)*0.02/3=0.04 s`；不承诺未完成段未来还能稳定多久 |
| `time_to_fall` | 只对真终止段平均，这里为 `2*0.02=0.04 s`；没有摔倒时为 missing，不是 0 秒 |
| `action_rate` | 段内动作差平方和有三个值 8、2、0，平均 10/3，单位为策略动作单位²；没有除以 dt，不是物理角加速度 |
| `torque_abs_mean` | 可选人工力矩 2、−1、0 N·m，分别持续 2、1、3 步，8 关节等权后为 `(2*16+1*8)/48=5/6 N·m` |
| `torque_squared_mean` | 同一数据为 `(4*16+1*8)/48=1.5 N²·m²` |
| `wall_time/peak_memory/interaction_steps` | 使用记录值，单位分别为 s/字节/步；缺失就 missing |

速度误差的关键行：

```python
cmd = torch.cat([ep.commanded_velocity for ep in eps]).double()
meas = torch.cat([ep.measured_velocity for ep in eps]).double()
```

拼接所有 episode 的全部步，再转 float64 计算，保证长段按步数自然获得较大权重。线速度取前两列的差的二范数，再平均；偏航取第三列绝对差，再平均。若改成逐段等权平均，就改变了指标口径。

动作平滑性直接用：

```python
diffs = [(ep.actions[1:] - ep.actions[:-1]).double().pow(2).sum(dim=1) for ep in eps if ep.steps > 1]
```

每段分别配对相邻步，八维平方后求和，最后拼接全部差分再平均。单步 episode 没有差分；若先把所有动作拼接再差分，就会把重置跳变也当成控制抖动。该形式沿用原 PPO 的 `action_rate` 奖励项，但此处报告正的变化量，不是训练奖励权重。

## 多次运行怎样汇总与比较

- `distinct_training_seeds(records)` 返回有效且互不相同的 `training_seed`，排除 None 和空字符串，按字符串排序。例如 `[7,7,7]` 只有 1 个有效起点，种子 0 是有效的。
- `missing_provenance(record)` 返回缺少的必需来源字段。
- `summarize_runs(records)` 对每项列出各运行值、运行间均值和**总体标准差**（`pstdev`）；只有一次时 std=None，任一次指标缺失则整项 missing。运行之间等权，不再按 episode 总步数加权；重复种子的记录仍参与数值汇总，种子去重只用于证据门槛。
- `check_budget(baseline,candidate,budget)` 返回 `rule/values/comparable/reason`。比较的是两组**全部记录**，不只比两组均值。
- `compare_runs(baseline,candidate,budget,thresholds=None)` 返回差值、可比性和判定阻断原因，无文件写入。只接受 `interaction_steps` 或 `wall_time` 预算。
- `_metric_to_dict(metric)` 把指标变为 JSON 基本字段；`write_report(records,path)` 用它写 JSON 并返回报告，有文件写入副作用，缺失数值写 null。

预算规则中最关键的是：

```python
comparable = len(set(flat)) == 1
```

这是交互步预算分支：两组每份记录都必须相等，1000 步对 100 万步不可比。耗时分支是：

```python
comparable = min(flat) > 0 and max(flat) <= min(flat) * (1 + WALL_TIME_RELATIVE_TOLERANCE)
```

要求正时间，最大值不超过最小值的 1.05 倍。测试中 12.5 s 对 13.0 s 相差 4%，可比；12.5 s 对 20 s 不可比。**5% 是项目暂定约定，未经性能验证，不是论文结论**，并写进结果 `budget_check.rule`。

`compare_runs` 在任一组少于 3 个不同有效训练种子、缺来源、预算不可比、阈值未固定时，仍展示可计算差异，但 `conclusion_allowed=False`、verdict 全为 None。阈值应在获得 PPO 波动范围后、正式 SAC 对照前固定。判定用 `candidate_mean-baseline_mean≤阈值`，例如摔倒率允许增量 0；只有 thresholds 中列出的且有数据的指标才得到判定，不能把 `conclusion_allowed=True` 当作所有指标都通过。

`write_report` 只检查来源与种子门槛；门槛足够时写“可在固定阈值下用 compare_runs 判定”，它本身不校验两组预算、不宣布 SAC 更优。

## 测试验证到了哪一步

[test_evaluate.py](test_evaluate.py) 的 `manual_record(...)` 是主要人工工厂；`BudgetAndSeedRegressionTest.group(...)` 构造多个种子，`assert_blocked(...)` 核对阻断结果，普通临时目录由测试自行管理。

12 项既有测试包括 `MetricTest` 五项（速度、结束与时长、动作差分、信号缺失、未知结束原因）、`ReportAndComparisonTest` 两项（来源/种子、阈值与预算门槛）、`BudgetAndSeedRegressionTest` 四项（1000 倍预算差、耗时容差、重复种子、正常条件仍可判定）、`RunEvaluationTest` 一项（2 个环境跑 7 次 step 得到 14 条，2 真终止/1 到时/2 未完成）。本轮只核对其代码与既有记录，未运行这些算法检查。
对应上文断言的精确方法名如下。测试方法由 unittest 调用，无业务参数，成功时返回 None；测试工厂与设置/清理方法的职责见上文。

| 测试类 | 方法定位（职责与预期见上文） |
|---|---|
| `MetricTest` | `test_velocity_errors_step_weighted`、`test_end_reasons_fall_rate_and_durations`、`test_action_rate_within_episodes_only`、`test_missing_signals_are_marked_not_zero`、`test_rejects_unknown_end_reason` |
| `ReportAndComparisonTest` | `test_report_blocks_conclusion_without_provenance_or_seeds`、`test_compare_requires_fixed_thresholds_seeds_and_budget` |
| `BudgetAndSeedRegressionTest` | `test_interaction_steps_differ_by_1000x`、`test_wall_time_outside_tolerance`、`test_three_records_with_same_seed_are_one_seed`、`test_matching_budget_and_three_seeds_still_pass` |
| `RunEvaluationTest` | `test_fake_environment_rollout` |

需要复现时，在仓库根目录运行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/evaluation/test_evaluate.py -v
```

指标缺失先看每段信号是否齐全；速度量级错先查单位与 2.0 缩放；摔倒率异常先看结束原因和 unfinished 数量；没有判定先读 `conclusion_blockers`，再检查种子、预算与阈值。真实任务、指令序列、评价时长、PPO 波动和阈值仍未验证。导出的一致性由 [T05](../export_inference/README_HUMAN.md) 单独负责，不能用本模块的人工对照代替部署验证。
