# N08 · 目标 Q 的慢更新｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [targets.py](targets.py) 与 [test_targets.py](test_targets.py)，5 项检查通过（含网络与学习的完整更新衔接检查）；未接入仿真或训练。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../README.md#项目维护规则)

## 实施目标与术语

让提供后续估值的参考网络逐渐跟随在线 Q，减缓训练目标的变化。

术语先说明：tau：目标参数每次吸收在线参数的比例。缓冲状态：如统计均值等保存于网络内、但不由梯度优化的数据。

## 与原 PPO 的关系

PPO 原路径没有这一套 SAC 目标 Q；新增两个在线 Q 各自的慢更新副本。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/models/sac_mlp_model.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/models/sac_mlp_model.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 在线 Q 参数、目标 Q 参数、跟随比例 tau，以及明确的统计状态管理约定。 |
| 输出 | 慢更新后的目标 Q；在线 Q 不被本操作修改。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [targets.py](targets.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| 常量 | `TAU=0.005`、`UPDATE_EVERY=1` | 作者默认：每次 Critic 更新后慢更新一次 |
| `validate_target_structure` | `(online, target) -> None` | 参数与缓冲按名称、形状、dtype 一一对应，且不共享存储；否则报错 |
| `soft_update` | `(online, target, tau=0.005) -> None` | 参数：target ← (1 − tau)·target + tau·online（`lerp_`）；缓冲：原样复制（tau=0 时也复制）；不修改在线网络；tau 须在 [0, 1] |
| `soft_update_due` | `(critic_update_count, update_every=1) -> bool` | 第几次在线 Q 更新之后执行慢更新（从 1 计） |
| `check_target_state` | `(online, target, optimizers=()) -> dict` | 检查结构、无共享、目标参数不求梯度且不在优化器中；返回原样复制的缓冲名称 |

tau 与频率的分工：tau 决定每次跟随的比例，`update_every` 决定多久跟随一次，两者都要记录才能复现实验。

非参数缓冲状态：
- 当前 N04 双 Q 没有缓冲；观测归一化由 D04 负责，在线和目标 Q 共用同一份统计与同一个 `normalize_q`。
- 如果以后加入带缓冲的层，缓冲原样复制、不做平均，这是本项目的明确约定：浮点运行统计也可以另定平均规则，但整数计数不能插值，统一复制最简单且与在线网络一致。`check_target_state` 会列出这些缓冲，提醒确认这条规则。

调用方：N08 测试中的完整更新衔接检查；正式调度由 T02 实现。

## 必须成立的规则

- 这是平均参数，不是平均两次评分。
- 目标参数不参与梯度优化；地形编码层等可训练部分不可遗漏。
- 非参数状态逐项规定复制或共享规则，不对所有缓冲数据盲目做平均。

## 算法约定与实现提示

每次跟随更新使用：目标参数的新值=(1−tau)×目标参数的旧值+tau×在线参数的当前值。tau 较小会让参考网络变化更慢。刚开始先完整复制在线 Q，再按训练配置慢更新。

更新频率与 tau 共同决定跟随速度；记录“每多少次在线 Q 更新跟随一次”。只写 tau 而漏掉频率不足以复现实验。

## 代码注释要求

共同要求见 [代码注释规范](../../README.md#代码注释规范)。

本模块特别需要注释：解释 tau 与更新频率、初始化完整复制、目标不参与优化及缓冲统计的单独处理。

## 验收案例（已执行，CPU、人工数据）

| 案例 | 对应测试（[test_targets.py](test_targets.py)） | 结果 |
|---|---|---|
| tau=0 时目标参数不变（缓冲仍会复制）；tau=0.3 与手算 0.7·target + 0.3·online 一致；tau=1 完整复制；在线网络不变 | `SoftUpdateTest.test_tau_zero_one_and_middle` | 通过 |
| 目标与在线指向同一存储、结构不一致、tau 越界时报错 | `SoftUpdateTest.test_shared_storage_and_structure_are_rejected` | 通过 |
| BatchNorm 缓冲原样复制（不是 0.1×4），计数复制，并在检查结果中列出 | `SoftUpdateTest.test_buffers_are_copied_not_averaged` | 通过 |
| 目标参数出现在优化器中或仍求梯度时报错；频率函数正确 | `SoftUpdateTest.test_target_state_and_frequency` | 通过 |
| **网络与学习的完整更新衔接检查**：D01/D02 人工经历 → D03（n=3）→ D04 / D05 → N01 → N03 / N04，按作者 `sac.py` 的顺序执行 Critic 更新 → 采样一次 → 温度更新 → 用新 alpha 与同一次采样更新 Actor → 目标慢更新，再做一次辅助学习；每一步改变的模块恰好是 {critics}、∅（采样）、{temperature}、{actor}、{targets}、{encoder, projector}；独立计算确认 Actor 损失使用更新后的 alpha（新旧 alpha 的损失差 > 1e-5，可区分）；池内原始经历不变；所有日志为有限数 | `UpdateCycleTest.test_one_cycle_changes_only_owned_state` | 通过 |

衔接检查的数据说明：动作表来自 E03 原版快照；软限位与工作轮速是 E04 测试中标注的人工数据；速度标签取 Q 观测前 3 维，这是原 PPO 的布局，E02 尚未确认。这只是人工数据下的更新衔接检查，不是 T02 训练器，也不代表训练效果。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/target_updates/test_targets.py -v
```

## 故障定位

目标突然随在线网络同步跳变时查共享参数或误用 tau；长期不动时查调度计数。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N04 双 Q 评分网络](../../networks_learning/q_networks/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [targets.py](targets.py)：`validate_target_structure`、`soft_update`、`soft_update_due`、`check_target_state`、`TargetUpdateError`。[test_targets.py](test_targets.py)：2 个测试类共 5 项；`UpdateCycleTest` 调用 E03、E04、D02～D05、N01～N07 的实际接口 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`soft_update` 只修改目标网络 |
| 代码注释核对 | 已核对：tau 与更新频率、完整复制初始化（N04）、目标不参与优化、缓冲的单独处理（文件头、`soft_update`、`check_target_state`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/target_updates/test_targets.py -v` → `Ran 5 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | 长期训练中的目标稳定性；Q 若加入带缓冲的层时规则是否合适；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`7f671f38d31e6a52c8f54acc07440cb2356021e0`，审核后修改衔接顺序与缓冲说明 `bd69772f7e9cb2f6207d025b5c7ce6d906184d13`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N08：按名称覆盖全部参数，明确缓冲复制规则与更新频率；并加入大模块的一次完整更新衔接检查 |
| 阅读重点与材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `targets.py`、`test_targets.py`；N04 `critics.py`；作者 `sac_mlp_model.py`（`soft_update_target_networks`）。重点说明 lerp 公式、为什么缓冲复制而不平均，以及衔接检查中每一步只改哪些状态 |
| 人类版 | 已有讲解，见 [README_HUMAN.md](README_HUMAN.md)；版本与验证范围以该文件记录为准 |
