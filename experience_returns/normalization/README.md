# D04 · 观测归一化与统计管理｜AI 实施规范

状态：单元验证通过（纯数据范围）。已实现 [normalization.py](normalization.py) 与 [test_normalization.py](test_normalization.py)，8 项 CPU 检查通过（Python 3.11 + PyTorch）；使用人工数据，未接入网络或采集循环，未进行训练或性能验证。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../README.md#项目维护规则)

## 实施目标与术语

把不同量级的输入调整到适合网络学习的尺度，并明确什么时候更新统计。

术语先说明：归一化：用数据均值与尺度调整输入。NaN：计算结果不是有效数值的标记。

## 与原 PPO 的关系

原归一化层在 training forward 内自动累计；SAC 反复回放旧数据，必须把更新统计与使用统计分开。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/normalizer.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/normalizer.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 新采集的策略/训练侧观测，以及当前均值、方差、计数。 |
| 输出 | 使用固定统计计算的网络输入，以及可以保存恢复的完整统计状态。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [normalization.py](normalization.py)

| 符号 | 签名 | 职责、单位、异常与副作用 |
|---|---|---|
| `NORMALIZER_EPS` | `1e-2` | 加在标准差上：y = (x − mean)/(√var + eps)。与原 PPO `normalizer.py`、作者 `EmpiricalNormalization` 的默认值相同 |
| `RunningNormalizer` | `(dim, eps=1e-2, device="cpu")`；属性 `mean`、`var`（float64）、`count`（int） | 单个输入对象的统计。初始 mean=0、var=1、count=0 |
| `RunningNormalizer.update_from_fresh_data` | `(x[N,D]) -> None` | 按批合并（Chan 并行方差公式，总体方差）；拒绝 NaN/inf 与形状不符；N=0 时不变。修改统计 |
| `RunningNormalizer.normalize` | `(x[...,D]) -> Tensor` | 只读统计，返回新张量，dtype 与输入相同 |
| `RunningNormalizer.export_statistics` / `load_statistics` | `() -> dict` / `(state) -> None` | 完整状态 `mean, var, count, eps, dim`；导出与恢复都复制，不共享存储；维数不符、count 为负、var 非法时报错 |
| `select_fresh_samples` | `(batch: TransitionBatch) -> (policy[N_p,33], q[N_q,C])` | 选出计入统计的样本（规则见下）；只读 |
| `ObservationNormalizers` | `(q_dim, eps=1e-2, device="cpu")`；属性 `policy`（33 维）、`q`（C 维） | `update_from_fresh_data(batch)`、`normalize_policy(history)`、`normalize_q(q)`、`export_statistics()`、`load_statistics(state)` |

### 字段、统计对象与计数规则

| 统计 | 维数 | 用于 | 变换 |
|---|---|---|---|
| `policy` | 33（逐帧） | Actor / 历史编码器：`policy_history`、`next_policy_history` 的每一帧 | 同一组统计作用于 [B,10,33] 的每一帧（与原 PPO 共用 `obs_normalizer` 一致）；逐帧变换，与 D05 切片先后无关 |
| `q` | C | online Q 的 `q_observation` 与 target Q 的尾部 `next_q_observation` | 两者都必须调用同一个 `normalize_q` 和同一份当前统计 |

速度监督标签（N02）直接读取**未归一化**的原始 Q 观测字段（E02 约定的 2.0 缩放），本模块不处理它。

计入统计的新采集数据（每个真实状态只计一次）：

1. 每条新记录的真实下一观测最新一帧 `next_policy_history[:, -1]` 与 `next_q_observation`。这覆盖了每次动作后到达的状态，包括真终止 / 纯截断时重置前的末状态。
2. 一轮第一条记录（`history_length == 1`）的当前观测 `policy_history[:, -1]` 与 `q_observation`。重置后观测正是这一条的当前观测，只在这里计一次。
3. 不计入：历史中的旧帧（之前已作为最新帧计入）、重置填充帧、从经验池回放出的记录（回放只调用 normalize）。

采集循环对每批新记录只调用一次 `update_from_fresh_data`。不归一化奖励，也不改写 D01 / D02 中保存的原始经历。

与作者的差异：作者在 `process_env_step` 中用 `true_next_obs` 更新统计；其中真终止环境的 `true_next_obs` 是重置后观测，末状态不计入。这里末状态与重置后状态各计一次。

调用方：[T02 Trainer](../../training_validation/training_loop/runner.py) 创建并随新采集数据更新统计；[N01](../../networks_learning/history_encoder/encoder.py) 读取策略统计，Q 学习使用 `normalize_q`；[T03](../../training_validation/checkpoints/checkpoint.py) 保存恢复统计，[T05](../../training_validation/export_inference/export.py) 固化推理统计。

## 必须成立的规则

- 重复回放同一数据不得重复增加统计计数。
- Actor 与 Q 各自的字段尺度明确；online/target Q 使用一致的输入变换。
- 仿真步、末状态、重置状态谁被统计一次须在实现 README 中列出，防止重复计数。

## 代码注释要求

共同要求见 [代码注释规范](../../README.md#代码注释规范)。

本模块特别需要注释：解释统计更新与只读归一化的区别、计数含义、稳定常数及回放不能重复累计统计的原因。

## 验收案例（已执行，纯数据范围）

全部使用人工数据。`FreshDataRuleTest` 构造帧值等于时间步的一轮：第 s 条记录的下一历史正是第 s+1 条的当前历史，重置填充帧填 −999。

| 案例 | 对应测试（[test_normalization.py](test_normalization.py)） | 结果 |
|---|---|---|
| 已知 4 行数据：均值 [4, 5.5]、总体方差 [5, 8.75]（手算），一次更新与分两批更新结果相同 | `RunningNormalizerTest.test_known_statistics_and_batch_merge` | 通过 |
| 同一批数据 normalize 5 次：统计（含计数）完全不变，输入不变，输出相同 | `RunningNormalizerTest.test_normalize_does_not_change_statistics_or_input` | 通过 |
| 保存恢复后同样输入结果一致，计数与 eps 一致；修改导出结果不影响原统计；维数不符时拒绝恢复 | `RunningNormalizerTest.test_export_and_restore` | 通过 |
| 常数通道输出 0、极小方差与 1e30 量级输入不产生 NaN；记录的 eps 为 1e-2 | `RunningNormalizerTest.test_constant_tiny_variance_and_extreme_inputs` | 通过 |
| NaN 或形状不符的更新被拒绝，计数不变 | `RunningNormalizerTest.test_rejects_bad_updates` | 通过 |
| 12 次动作只计入真实帧 0..12 各一次：count=13、均值 6、方差 14；重叠帧和填充帧都不计入 | `FreshDataRuleTest.test_each_real_frame_counted_once` | 通过 |
| 真终止末状态（帧 3）与下一轮重置后观测（帧 100）各计一次 | `FreshDataRuleTest.test_terminal_and_reset_states_each_counted_once` | 通过 |
| D02 → D03 回放 3 次后计数不变；归一化 `first.policy_history`、`tail.next_q_observation`、`first.q_observation`，原始记录与池内记录都不变 | `ReplayIntegrationTest` | 通过 |

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/normalization/test_normalization.py -v
```

## 故障定位

同一输入在不更新参数时输出漂移，先查统计或 BatchNorm 状态是否仍被更新。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E02 观测接口](../../environment_control/observation_interface/README.md)
- [D01 经历记录契约](../../experience_returns/transition_schema/README.md)：计数规则依赖 `history_length` 与 next_* 字段

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（纯数据范围）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [normalization.py](normalization.py)：`RunningNormalizer`、`ObservationNormalizers`、`select_fresh_samples`、`NORMALIZER_EPS`、`NormalizationError`；依赖 D01 `TransitionBatch`、`POLICY_OBS_DIM`。[test_normalization.py](test_normalization.py)：3 个测试类共 8 项，衔接 D02、D03 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`update_from_fresh_data`、`load_statistics` 修改统计，其余只读；统计以 float64 保存，输出与输入 dtype 相同 |
| 代码注释核对 | 已核对：变换公式与 eps 的位置和来源（文件头、`NORMALIZER_EPS`）；两套统计与 online/target 一致（`ObservationNormalizers`）；计数规则与末状态 / 重置状态的处理（文件头、`select_fresh_samples`）；normalize 只读（`normalize`）；与原 PPO / 作者的差异（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU）；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/normalization/test_normalization.py -v` → `Ran 8 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除，不影响结果 |
| 未验证项 | 真实观测尺度下的统计；N01 / N04 的接入方式（统计作为网络缓冲区还是外部对象）；T03 检查点保存；与 N02 BatchNorm 的交互；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`e77bc66d7878831fb1b5eb19d616785c81f2d251`（分支 `claude/experience-returns`，基于 `c06509e`）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 D04：原 PPO 在 forward 中自动更新统计，SAC 回放时会重复计数，因此拆分更新与使用；明确每个真实状态只计一次 |
| 阅读重点与材料 | 根/大模块/本模块 AI 规范；D01 `schema.py`；带注释的 `normalization.py`、`test_normalization.py`；原 PPO `normalizer.py`、`actor_critic.py`（`_normalize`）；作者 `modules/normalization.py` 与 `sac.py` 的 `process_env_step`。重点说明并行方差合并、`select_fresh_samples` 的计数规则，以及为什么回放只调用 normalize |
| 人类版 | 已有讲解，见 [README_HUMAN.md](README_HUMAN.md)；版本与验证范围以该文件记录为准 |
