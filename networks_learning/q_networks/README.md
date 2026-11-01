# N04 · 双 Q 评分网络｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [critics.py](critics.py) 与 [test_critics.py](test_critics.py)，5 项检查通过；未接入仿真或训练。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

分别评价给定状态下的指定动作，供价值学习与策略学习使用。

术语先说明：Q 是状态和动作的预期价值；双 Q 的较小值减少单个评分过高带来的误导，不能保证绝对准确。

## 与原 PPO 的关系

原 V 只评价状态；新 Q 把当前待评价动作作为额外输入，并保留两套独立参数。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/models/sac_mlp_model.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/models/sac_mlp_model.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | Q 可见的状态信息、同一坐标下的 8 维策略动作。 |
| 输出 | 两个各为 [B,1] 的 Q 评分及其对应目标网络结构。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [critics.py](critics.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| `QNetwork` | `(q_dim, action_dim=8, hidden_dims=(512,256,128))`；`forward(q_observation[B,C], action[B,8]) -> [B,1]` | 输入按 [Q 观测 \| 动作] 拼接，经 ELU MLP 得到评分；前向校验 C 与动作维数 |
| `TwinQ` | `(q_dim, ...)`；子模块 `q1`、`q2` | 两套独立构造、独立初始化的 Q，不共享层 |
| `evaluate_q_pair` | `(critics, q_observation, action) -> (q1, q2)` | 同一批状态与动作分别评价 |
| `initialize_targets` | `(critics) -> TwinQ` | `deepcopy` 完整复制参数与缓冲；参数 `requires_grad=False`、eval 模式 |
| `list_critic_parameters` | `(critics) -> list[(名称, 参数)]` | 在线双 Q 的全部可训练参数，即 Critic 优化器的参数集合 |

数据约定：
- `q_observation` 为 D04 `normalize_q` 后的 Q 观测。C 由 E02 运行时观测组给出（平地预期 54、复杂地形预期 241），构造时固定，本模块不指定最终值。
- 动作是策略动作坐标，与 E03 / D01 / N03 一致。
- 网络不含 BatchNorm 等缓冲；在线和目标 Q 共用同一份 D04 统计。

第一版没有地形编码层：187 维高度扫描直接作为 Q 观测的一部分。原 PPO 先用 128/64/32 的 `scan_encoder` 处理高度扫描；如果之后加入，必须作为 `QNetwork` 的子模块，这样才会自动包含在目标副本、N08 慢更新和保存恢复中。

调用方：N05、N06、N08。

## 必须成立的规则

- 两个 Q 参数独立；Actor 历史编码器不与 Q 隐式共享可训练层。
- Q 可以用仿真额外信息，但 Actor 的输入契约保持不变。
- 若添加地形编码层，它也必须包含在目标副本与保存恢复中。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：标注 Q 状态和动作的拼接、双网络参数独立性、训练侧信息边界与地形编码的保存/目标副本责任。

## 验收案例（已执行，CPU、人工数据）

| 案例 | 对应测试（[test_critics.py](test_critics.py)） | 结果 |
|---|---|---|
| 同状态、不同动作得到不同评分 | `EvaluationTest.test_different_actions_give_different_scores` | 通过 |
| C=54 与 C=241（E02 预期值，仅作维数示例）都可构造；54 维网络拒绝 241 维输入与 7 维动作 | `EvaluationTest.test_q_dim_is_checked` | 通过 |
| Q1、Q2 初值不同、存储不重叠；只优化 Q1 时 Q2 不变 | `IndependenceTest.test_twins_are_independent` | 通过 |
| 目标副本数值相同、存储独立、不求梯度、eval；修改在线值不影响目标 | `IndependenceTest.test_targets_copy_values_not_storage` | 通过 |
| 参数列表覆盖在线双 Q 全部参数、不含目标；与 N01 编码器、N03 Actor 不共享存储 | `IndependenceTest.test_parameter_listing_and_no_sharing_with_actor_encoder` | 通过 |

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/q_networks/test_critics.py -v
```

## 故障定位

Q 对所有动作几乎相同，检查动作是否实际拼入、数值尺度及梯度。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E02 观测接口](../../environment_control/observation_interface/README.md)
- [E03 八维动作定义表](../../environment_control/action_contract/README.md)
- [D04 观测归一化与统计管理](../../experience_returns/normalization/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [critics.py](critics.py)：`QNetwork`、`TwinQ`、`evaluate_q_pair`、`initialize_targets`、`list_critic_parameters`。[test_critics.py](test_critics.py)：2 个测试类共 5 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`initialize_targets` 分配新存储，其余无副作用 |
| 代码注释核对 | 已核对：Q 状态与动作的拼接、双网络独立性、训练侧信息边界、地形编码层的保存与目标副本责任（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/q_networks/test_critics.py -v` → `Ran 5 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | E02 确认的实际 C 与字段；是否需要高度扫描编码层；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`ef98631024ce4e28f6966a445dc39c710aee349c`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N04：两套独立 Q 与完整复制的目标副本；暂不加地形编码层，避免在 C 未确认前固定结构 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `critics.py`、`test_critics.py`；作者 `sac_mlp_model.py`（`SACCriticModel`、目标网络）；原 PPO `actor_critic.py`（critic 与 `scan_encoder`）。建议 Codex 重点解释两个 Q 的独立性、目标副本的冻结，以及地形编码层的扩展约定 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文。人类版面向有 Python 基础的读者 |
