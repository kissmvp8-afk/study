# N07 · 探索力度 alpha｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [temperature.py](temperature.py) 与 [test_temperature.py](test_temperature.py)，4 项检查通过；未接入仿真或训练。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

调节策略中鼓励动作多样性的力度，避免把探索简单理解成越随机越好。

术语先说明：熵：衡量动作分布多样性的量。alpha：训练中鼓励这种多样性的权重，不直接表示电机转动幅度。

## 与原 PPO 的关系

原 PPO 有固定熵系数；SAC 在策略与后续价值目标中使用熵，并可自动学习其权重。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 当前策略的对数密度、配置的目标熵、当前 log_alpha。 |
| 输出 | 正数 alpha、温度损失和实际熵相对目标的偏差。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [temperature.py](temperature.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| 常量 | `INITIAL_ALPHA=0.05`、`TARGET_ENTROPY_SCALE=1.0`、`TEMPERATURE_LEARNING_RATE=1e-3` | 作者默认值，作为显式实验起点 |
| `TargetEntropy` | 冻结 dataclass：`action_coordinate`、`squashed_equivalent`、`log_half_width_sum` | 目标熵及其坐标换算 |
| `target_entropy` | `(half_width, target_entropy_scale=1.0) -> TargetEntropy` | 策略动作坐标下的目标 = −scale × 8（作者约定）；单位区间等价值 = 该目标 − Σ log c |
| `Temperature` | `(initial_alpha=0.05)`；参数 `log_alpha` | 只含一个可训练参数，alpha = exp(log_alpha) 恒为正 |
| `current_alpha` | `(temperature) -> float` | 不带梯度的 alpha 数值，供 N05、N06 使用 |
| `make_temperature_optimizer` | `(temperature, lr=1e-3) -> Adam` | 只含 log_alpha |
| `temperature_loss` | `(temperature, log_prob, target) -> 标量` | −平均 [log_alpha × (log π.detach() + target)] |
| `update_temperature` | `(temperature, optimizer, log_prob, target) -> dict` | 只更新 log_alpha；返回损失、alpha、熵估计与熵差（负数表示偏低） |

目标熵的坐标约定：
- N03 的 log π 是在策略动作坐标 a 下计算的，已经包含 −Σ log c。作者的 log π 也减去了 `log_action_range`，坐标一致，所以沿用作者的 −1.0 × 8 = −8。
- 在单位区间坐标 y = tanh(u) 下，等价目标为 −8 − Σ log c。E04 的半宽改变时，这个等价值随之改变，也就是说探索语义会变。`target_entropy` 同时返回这两个值，供配置与日志记录。
- 如果希望探索语义不随动作尺度变化，应当固定单位区间的目标，再加上 Σ log c 换算回动作坐标。

调用方：N08 测试中的完整更新衔接检查。

## 必须成立的规则

- 连续动作的熵数值可以为负，不据此判为错误。
- 目标熵与密度使用的动作坐标及缩放约定一致；保留作者配置作为显式实验起点。
- 此更新不能反向修改 Actor 或 Q。

## 算法约定与实现提示

使用 log_alpha 表示可训练参数，再取 alpha=exp(log_alpha)，自然保证 alpha 为正。作者的目标是 L_alpha=−平均[log_alpha×(log_prob+target_entropy)]；括号内数值在这次更新中当作固定数据。

直观检查：估计熵低于目标时增加 alpha，让保留多样性的要求更强；高于目标时减小 alpha。作者按 target_entropy=−target_entropy_scale×动作维数配置目标；TITA 的维数是 8。动作缩放改变密度的坐标，因此把所用尺度与目标一起记录，不能省略缩放修正后仍照搬所有数值。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：解释 log_alpha 保证正值、目标熵的坐标约定、detach 的边界，以及熵偏低/偏高时的调整方向。

## 验收案例（已执行，CPU、人工数据）

| 案例 | 对应测试（[test_temperature.py](test_temperature.py)） | 结果 |
|---|---|---|
| 熵估计（−20）低于目标（−8）时 alpha 增大，高于目标（30）时减小，熵差符号与之对应 | `DirectionTest.test_alpha_direction` | 通过 |
| 学习率 0.5 下连续 200 次压低 alpha，结果仍为正的有限数；`state_dict` 保存恢复后 alpha 与损失一致 | `DirectionTest.test_alpha_positive_finite_and_restorable` | 通过 |
| 温度损失不反传到 Actor，只有 log_alpha 得到梯度 | `BoundaryTest.test_no_gradient_to_actor` | 通过 |
| 同一网络与噪声，半宽从 1 变为 2：log π 恰好减少 8·log 2；动作坐标目标仍为 −8，单位区间等价目标从 −8 变为 −8 − 8·log 2，熵差也相应改变 | `BoundaryTest.test_action_scale_shifts_density_and_squashed_target` | 通过 |

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/entropy_temperature/test_temperature.py -v
```

## 故障定位

alpha 持续异常增长时先检查密度、目标熵和动作变换，再扩大或缩小学习率。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [temperature.py](temperature.py)：`Temperature`、`current_alpha`、`temperature_loss`、`update_temperature`、`make_temperature_optimizer`、`target_entropy`、`TargetEntropy`。[test_temperature.py](test_temperature.py)：2 个测试类共 4 项，用 N03 Actor 检查尺度关系 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`update_temperature` 只修改 log_alpha 与优化器 |
| 代码注释核对 | 已核对：log_alpha 保证正值、目标熵的坐标约定与换算、detach 边界、熵偏低 / 偏高时的调整方向（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/entropy_temperature/test_temperature.py -v` → `Ran 4 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | TITA 实际边界下 Σ log c 的取值与目标熵是否合适（需要训练对照）；auto_alpha 关闭时的固定 alpha（T01 配置）；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`b186639cdd5884c6631e093b04e8d9a6c898c281`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N07：目标熵沿用作者配置，并明确它与 N03 密度处于同一坐标，同时给出单位区间的等价值，不再无解释地照搬 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `temperature.py`、`test_temperature.py`；N03 `actor.py`；作者 `sac.py`（log_alpha、alpha 损失、target_entropy）。建议 Codex 重点解释两种坐标下的熵差 Σ log c，以及为什么调整方向由 log π + target 的符号决定 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文。人类版面向有 Python 基础的读者 |
