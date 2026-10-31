# N02 · 速度监督与 Barlow Twins｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [auxiliary.py](auxiliary.py) 与 [test_auxiliary.py](test_auxiliary.py)，8 项检查通过；未接入仿真或训练。

[人类版（待 Codex 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

为历史编码器提供直接速度标签和无逐项标签的特征学习要求。

术语先说明：辅助学习：帮助状态表示学习的额外目标。BatchNorm：使用一批数据的统计调整中间特征的网络层。

## 与原 PPO 的关系

沿用原速度误差与相邻窗口相关性目标，辅助更新频率与大量 SAC 回放更新分开管理。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 有效的相邻历史窗口、缩放一致的真实速度标签、N01 编码器与训练用转换模块。 |
| 输出 | 速度误差、一致性误差、去重复误差、辅助更新后的编码器参数和可读日志。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [auxiliary.py](auxiliary.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| 常量 | `PROJECTOR_DIM=64`、`OFF_DIAGONAL_WEIGHT=5e-3`、`AUXILIARY_LEARNING_RATE=1e-3`、`MAX_GRAD_NORM=1.0`、`MIN_VALID_SAMPLES=2` | 前四项取自原 PPO 配置，作为起点；最后一项是 train 模式下 BatchNorm 的最低样本数 |
| `BarlowProjector` | `()`；`forward(hidden[N,16]) -> [N,64]` | 训练用 64 维转换：16→64（无偏置, BN, ELU）→64（无偏置）→ `BatchNorm1d(64, affine=False)`；构造后为 eval；不进入 Actor 输入 |
| `velocity_loss` | `(predicted[N,3], label[N,3]) -> 标量` | 均方误差；预测与标签都是 2.0 缩放后的值 |
| `barlow_loss` | `(z1[N,64], z2[N,64], off_diagonal_weight=5e-3) -> (总损失, 对角项, 非对角项)` | `c = z1ᵀz2/N`，Σ(c_ii−1)² + 0.005·Σ_{i≠j} c_ij²，与原 PPO 一致 |
| `make_auxiliary_optimizer` | `(encoder, projector, lr=1e-3) -> Adam` | 独立优化器，只含编码器与 projector 的参数 |
| `update_encoder` | `(encoder, projector, optimizer, windows, history_length[B], velocity_label[B,3], max_grad_norm=1.0) -> dict` | 按 D05 `valid_history_mask(..., 6)` 筛选；有效样本 < 2 时跳过，不改任何状态；否则 train 模式前向两个窗口、计算 L_bt + L_vel、更新，**结束后恢复 eval**。返回 `skipped`、`valid_samples` 与各项损失 |

BatchNorm 统计何时更新：只在 `update_encoder` 内的两次 train 前向中更新（与原 PPO 相同，一次辅助更新会前向两个窗口）。采集、Critic、Actor 更新都在 eval 下进行，不会改动统计。

速度标签：时刻 t 的原始（未归一化）训练侧身体线速度，即乘过 2.0 的 base_lin_vel，由调用方显式传入。原 PPO 取 critic 观测前 3 维，E02 尚未确认该布局。

调用方：目前是本目录测试与 N08 测试中的完整更新衔接检查。辅助更新的频率由 T02 决定，与 SAC 回放更新分开。

## 必须成立的规则

- 同维相关性趋向 1，不同维趋向 0；避免只要求相同而出现全常数输出。
- 原 off-diagonal 权重 0.005 可作为起点；保持速度标签与预测的相同缩放。
- 辅助学习开启 BatchNorm 更新的时机明确；采集和 Actor 优化不隐式修改这些统计。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释相关性矩阵对角/非对角项、速度监督尺度、有效窗口筛选，以及独立优化器与 BatchNorm 模式。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，CPU、人工数据）

| 案例 | 对应测试（[test_auxiliary.py](test_auxiliary.py)） | 结果 |
|---|---|---|
| 速度预测正确时平方误差为 0 | `LossTest.test_velocity_loss_zero_when_correct` | 通过 |
| 常数特征：经 projector 后相关矩阵为 0，对角项 = 64（最大），不会被误认为完美表示；对照“两窗口一致且各维独立”的特征损失接近 0 | `LossTest.test_constant_features_are_not_perfect` | 通过 |
| 与原 PPO 损失写法（`c.div_`、`diagonal.add_(-1).pow_(2)`）一致 | `LossTest.test_matches_ppo_formula` | 通过 |
| 辅助更新只改编码器与 projector；优化器参数集合恰好等于二者；Actor 动作头、Q、alpha 的占位模块不变 | `UpdateTest.test_only_encoder_and_projector_change` | 通过 |
| eval 前向不改 BN 统计，辅助更新改变统计，结束后恢复 eval | `UpdateTest.test_batchnorm_statistics_only_change_during_update` | 通过 |
| 有效历史不足（0 条、1 条）时跳过：参数、BN 统计、优化器状态都不变 | `UpdateTest.test_insufficient_valid_history_is_skipped` | 通过 |
| 无效行（长度 < 6）的窗口填 NaN，损失与参数仍为有限值，说明它们没有参与计算 | `UpdateTest.test_invalid_rows_are_not_used` | 通过 |
| 同一批数据重复更新 30 次，损失下降 | `UpdateTest.test_repeated_updates_reduce_loss` | 通过 |

调试记录：第一次运行时 8 项都通过，但 torch 对 `float(loss)` 给出“张量仍带梯度”的警告；改用 `.item()` 后重跑，输出中已无警告。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/test_auxiliary.py -v
```

## 故障定位

辅助损失降低但控制变差，分别看速度误差、特征变化幅度及辅助更新频率，不只看总损失。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N01 历史状态编码](../../networks_learning/history_encoder/README.md)
- [D05 历史窗口与有效性](../../experience_returns/history_windows/README.md)
- [E02 观测接口](../../environment_control/observation_interface/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [auxiliary.py](auxiliary.py)：`BarlowProjector`、`velocity_loss`、`barlow_loss`、`make_auxiliary_optimizer`、`update_encoder`；依赖 N01 `HistoryEncoder`、D05 `valid_history_mask`。[test_auxiliary.py](test_auxiliary.py)：2 个测试类共 8 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`update_encoder` 修改编码器与 projector 的参数、BN 统计和优化器状态，跳过时不修改 |
| 代码注释核对 | 已核对：相关矩阵对角 / 非对角项与常数特征（文件头）；速度标签尺度（文件头、`velocity_loss`）；有效窗口筛选与跳过规则（`update_encoder`）；独立优化器与 BN 模式（文件头、`update_encoder`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/auxiliary_learning/test_auxiliary.py -v` → `Ran 8 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | E02 确认速度标签所在字段；辅助更新频率与 SAC 更新次数之比（T02）；真实数据上的表示质量；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`2e6c470770fe22458a9412178144d9410c97cf81`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N02：原 PPO 把辅助损失加入 PPO 总损失并共用一个优化器；SAC 中改为独立优化器与独立更新，并明确 BN 更新时机与有效样本 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `auxiliary.py`、`test_auxiliary.py`；原 PPO `actor_critic.py`（`barlow_twins_loss`、`off_diagonal`）与 `np3o.py`（辅助损失并入总损失）。建议 Codex 重点解释相关矩阵、常数特征为何得到最大损失、有效样本筛选与跳过，以及 train / eval 切换 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文。人类版面向有 Python 基础的读者 |
| 技术初稿 / 语言润色 | 待 Codex 技术初稿；其后由 Gemini 润色同一份人类版 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
