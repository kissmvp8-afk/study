# N03 · SAC 动作分布｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [actor.py](actor.py) 与 [test_actor.py](test_actor.py)，7 项检查通过；未接入仿真或训练。

[人类版（待 Codex 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

从 52 维状态描述产生有界动作及其密度，支持温和开局与确定性运行。

术语先说明：正态分布：用中心和分散程度描述随机数。tanh：将数值平滑压到 -1 与 1 之间的函数。对数密度：便于计算的动作密度表示。

## 与原 PPO 的关系

原 PPO 使用无界正态采样及不随状态变化的标准差；SAC 学习随状态变化的中心和随机程度，并进行有界变换。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/models/sac_mlp_model.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/models/sac_mlp_model.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 当前状态特征、E04 每维边界与初始中心、标准差的数值限制。 |
| 输出 | 8 维策略动作、变换后的动作对数密度、确定性动作与分布统计。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [actor.py](actor.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| 常量 | `LOG_STD_MIN=-20`、`LOG_STD_MAX=2`（作者）、`INITIAL_STD=0.15`（规范）、`MEAN_HEAD_INIT_STD=1e-3`（作者）、`ACTOR_HIDDEN_DIMS=(512,256,128)`（原 PPO） | 数值来源见代码注释 |
| `log_one_minus_tanh_squared` | `(u) -> log(1−tanh²u)` | 精确恒等式 `2(log2 − u − softplus(−2u))`；|u| 很大时仍为有限值（作者用 `+1e-6` 近似，这里不采用） |
| `squash_and_log_prob` | `(mean, log_std, noise, center, half_width) -> (a[B,8], log π[B,1])` | u = mu + std·ε，a = b + c·tanh(u)；log π = Σ[−ε²/2 − log_std − log(2π)/2] − Σ log(1−tanh²u) − Σ log c |
| `SquashedGaussianActor` | `(bounds: ActionBounds, initial_center, feature_dim=52, hidden_dims=(512,256,128), initial_std=0.15)` | 52→512→256→128（ELU）→ 均值头 8 + log_std 头 8；E04 的 b、c 存为**缓冲**（不训练、随模型保存）；均值头权重 ~N(0,1e-3)、偏置 = E04 的 u0；log_std 头权重 0、偏置 log 0.15 |
| `.distribution_parameters` | `(features[B,52]) -> (mu, log_std)` | log_std 限制在 [−20, 2] |
| `.sample_action_and_log_prob` | `(features, noise=None, generator=None) -> (a, log π)` | 重参数化采样，a 与 log π 来自同一次采样且带梯度；`noise` 用于复现 |
| `.deterministic_action` | `(features) -> a` | u = mu，走同一套 tanh 与缩放；不是随机分布的数学均值 |

与 E04 的衔接：直接接收 `derive_bounds` 的 `ActionBounds` 和 `calibrate_initial_center` 的 u0，不重新计算边界。

调用方：N05（尾部新动作）、N06（Actor 损失）、N07 / N08 测试。

## 必须成立的规则

- 初始标准差 0.15 是内部随机尺度；物理动作中心由 E04 校准。
- log probability 包含压缩变换与各维缩放的修正，并按动作维度相加。
- 不把压缩后确定性输出称为该随机分布严格的数学均值；运行不调用 Q 搜索动作。

## 算法约定与实现提示

网络先给出中心 mu 和标准差 sigma；标准差越大，随机样本通常越分散。取标准正态噪声 epsilon，算 u=mu+sigma×epsilon，再算 a=b+c×tanh(u)。固定 epsilon 后，这条计算路径仍能追踪参数改变怎样影响动作，用于训练 Actor。

动作变换会挤压或拉伸数值范围，密度也必须同步改变：最终对数密度 = 各维正态对数密度之和 − 各维 log(1−tanh(u)^2) 之和 − 各维 log(c) 之和。实现时采用数值稳定写法，并说明用于稳定计算的近似及误差范围。

确定性运行令 u=mu，仍执行同一套压缩和缩放。初始化需同时规定均值输出头的很小权重、E04 校准的中心以及 std=0.15 的输出；只缩小标准差、保留很大的随机均值不能保证温和开局。Q 学习和 Actor 学习使用的动作及密度必须来自同一次采样。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释可求梯度的采样路径、tanh 和每维缩放的密度修正、log_std 限制、初始中心和确定性输出。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，CPU、人工数据）

动作表来自 E03 原版快照；软限位与工作轮速复用 E04 测试中标注的**人工数据**（非对称）。

| 案例 | 对应测试（[test_actor.py](test_actor.py)） | 结果 |
|---|---|---|
| 特征放大 1e3、1e6 倍时动作仍在 [L, U] 内、密度有限；u = ±60 时稳定写法有限（朴素写法为 −inf）；中等 u 时两种写法一致 | `TransformTest.test_extreme_inputs_stay_in_bounds_with_finite_density` | 通过 |
| 固定噪声：动作与对数密度和 float64 下朴素公式（Normal.log_prob − log(1−tanh²) − log c）的结果一致（1e-4） | `TransformTest.test_density_matches_independent_formula` | 通过 |
| 确定性动作等于零噪声采样 | `TransformTest.test_deterministic_uses_same_transform` | 通过 |
| log_std 被限制在 [−20, 2] | `TransformTest.test_log_std_is_clamped` | 通过 |
| 初始内部 std 为 0.15 | `InitializationTest.test_initial_std` | 通过 |
| 初始确定性动作 |a − a0| < 0.05，E03 换算后的物理目标靠近默认姿态；对照：作者式零偏置在非对称维落到区间中点 b=0.4 | `InitializationTest.test_initial_deterministic_target_near_default` | 通过 |
| 采样路径把梯度传到均值头、log_std 头和隐藏层；b、c 是缓冲而非参数 | `GradientTest` | 通过 |

调试记录：初稿中有一条恒为真的断言，还会对 0 取对数，运行前已改成“朴素写法在 u=−60 处为 −inf”的对照；另删掉了一个未使用的 import，并重跑确认。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/test_actor.py -v
```

## 故障定位

动作总贴边时检查初始化、scale 和标准差；alpha 异常先检查密度坐标与变换修正。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N01 历史状态编码](../../networks_learning/history_encoder/README.md)
- [E04 动作边界与初始中心](../../environment_control/action_bounds/README.md)
- [D04 观测归一化与统计管理](../../experience_returns/normalization/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [actor.py](actor.py)：`SquashedGaussianActor`、`squash_and_log_prob`、`log_one_minus_tanh_squared`；依赖 E04 `ActionBounds`、N01 `ACTOR_FEATURE_DIM`。[test_actor.py](test_actor.py)：3 个测试类共 7 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；全部方法无副作用（采样时消耗 generator 的随机状态） |
| 代码注释核对 | 已核对：可求梯度的采样路径、tanh 与各维缩放的密度修正、稳定写法、log_std 限制、初始中心与确定性输出（文件头、`squash_and_log_prob`、构造函数） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/actor_distribution/test_actor.py -v` → `Ran 7 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | TITA 实际边界（E04 人工数据）下的探索幅度；训练中贴边比例；导出（T05）；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`a7a469ef3c21411a52920f523cb3b5488ee44258`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N03：沿用作者的 tanh 变换与各维缩放，均值偏置改用 E04 校准的 u0；密度修正改用精确恒等式，避免 1e-6 近似 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `actor.py`、`test_actor.py`；E04 `bounds.py`；作者 `sac_mlp_model.py`（初始化、`sample_action_logp`）。建议 Codex 重点解释重参数化、三项密度修正的来源、精确恒等式，以及 u0 如何保证温和开局 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文。人类版面向有 Python 基础的读者 |
| 技术初稿 / 语言润色 | 待 Codex 技术初稿；其后由 Gemini 润色同一份人类版 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
