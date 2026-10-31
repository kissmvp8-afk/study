# N01 · 历史状态编码｜AI 实施规范

状态：单元验证通过（CPU、人工数据）。已实现 [encoder.py](encoder.py) 与 [test_encoder.py](test_encoder.py)，7 项检查通过；未接入仿真或训练。

[人类版（待 Codex 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

从最近 5 帧中提取共享表示，再输出三维速度估计与 16 维隐藏特征。

术语先说明：隐藏特征：网络学到的数字描述，未必能逐个解释成具体物理量。前向计算：给定输入和参数，算出输出。

## 与原 PPO 的关系

保留原历史编码功能，不因换成 SAC 就让 Actor 直接使用真实速度。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)
- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真或训练中验证。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | D05 提供的窗口与 D04 固定统计下的观测；每帧 33 个数。 |
| 输出 | 3 维缩放后的身体线速度估计、16 维隐藏特征；合并当前 33 维观测得到 52 维 Actor 输入。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可），与 D01～D05 相同；仓库尚无统一依赖文件（T01）。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [encoder.py](encoder.py)

| 符号 | 签名 | 职责、形状、副作用 |
|---|---|---|
| `VELOCITY_DIM` / `HIDDEN_FEATURE_DIM` / `ACTOR_FEATURE_DIM` | `3` / `16` / `52` | Actor 输入 = [速度估计 3 \| 隐藏特征 16 \| 当前帧 33]，顺序同原 PPO `actor_input` |
| `HistoryEncoder` | `(backbone_dims=(128, 64), latent_hidden_dim=32)`；子模块 `backbone`、`latent_head`、`velocity_head` | 层序与原 PPO `MlpBarlowTwinsActor` 相同：165→128（BN, ELU）→64（BN, ELU）；64→32（BN, ELU）→16；64→3。**构造后为 eval 模式** |
| `HistoryEncoder.encode_history` | `(window[B,5,33]) -> [B,64]` | 帧优先展平为 [B,165]（t-4 帧在前），形状不符时报错 |
| `HistoryEncoder.predict_velocity` / `hidden_features` | `([B,64]) -> [B,3]` / `-> [B,16]` | 速度估计与 E02 标签同为 2.0 缩放后的值（除以 2.0 才是 m/s）；隐藏特征没有逐项物理含义 |
| `build_actor_features` | `(encoder, windows: HistoryWindows) -> [B,52]` | 用 t-4..t 窗口编码；在 `no_grad` 下计算，结果不连接编码器计算图；编码器处于 train 模式时报错，防止隐式更新 BN 统计 |
| `features_from_history` | `(encoder, normalizers, policy_history[B,10,33]) -> [B,52]` | D04 `normalize_policy` → D05 `split_windows` → `build_actor_features`；当前状态与尾部状态都走这条路径 |

梯度与 BN 边界：本模块不创建优化器。编码器参数和 BN 统计只由 N02 更新，N02 只在更新期间切到 train，结束后恢复 eval。

调用方：N02（编码器子模块）、N05（尾部特征）、N06 / N08 衔接检查（当前状态特征）。

## 必须成立的规则

- 本模块定义网络前向计算，不自行决定优化器更新。
- 16 个隐藏特征没有逐个人工指定的物理意义。
- 策略更新时保持编码器梯度隔离；BatchNorm 的模式管理另由辅助学习明确处理。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：标注历史展平顺序、3/16/52 维的组成和速度缩放；解释编码器梯度与 BatchNorm 统计管理边界。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，CPU、人工数据）

| 案例 | 对应测试（[test_encoder.py](test_encoder.py)） | 结果 |
|---|---|---|
| 维数为 64 / 3 / 16 / 52，最后 33 维即当前帧 | `ShapeAndOrderTest.test_dimensions` | 通过 |
| eval 模式下同参数、同输入重复输出一致 | `ShapeAndOrderTest.test_repeatable_in_eval_mode` | 通过 |
| 帧顺序：与手动帧优先展平一致；帧顺序反转或误用 t-5..t-1 时输出不同 | `ShapeAndOrderTest.test_uses_recent_window_in_frame_order` | 通过 |
| 与原 PPO 编码器（按 baseline 层序复写）加载同一组权重：`load_state_dict` 键与形状完全对应，52 维输出在 1e-6 内一致 | `PpoEquivalenceTest` | 通过 |
| Actor 特征不带梯度、不改变参数与 BN 统计；train 模式被拒绝；窗口形状错误被拒绝 | `GradientAndBatchNormTest`（2 项）、`ShapeAndOrderTest.test_rejects_wrong_window_shape` | 通过 |

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/test_encoder.py -v
```

## 故障定位

估计速度看似翻倍时查标签缩放；Actor 输入不匹配时查拼接顺序。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [D04 观测归一化与统计管理](../../experience_returns/normalization/README.md)
- [D05 历史窗口与有效性](../../experience_returns/history_windows/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、人工数据）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [encoder.py](encoder.py)：`HistoryEncoder`、`build_actor_features`、`features_from_history`；依赖 D04 `ObservationNormalizers`（只读）、D05 `split_windows`、D01 `POLICY_OBS_DIM`。[test_encoder.py](test_encoder.py)：3 个测试类共 7 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`build_actor_features` 与 `features_from_history` 无副作用 |
| 代码注释核对 | 已核对：展平顺序、3/16/52 的组成与速度缩放（文件头）；梯度隔离与 BN 模式边界（文件头、`build_actor_features`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/history_encoder/test_encoder.py -v` → `Ran 7 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | 真实观测尺度下的编码；N02 训练后的表示质量；导出（T05）；GPU；仿真、训练与性能 |
| 代码版本 | 代码与测试：`7e7d34a63ddbe20201ffbdf37e9923c50c1bcc73`（分支 `claude/networks-learning`，基于 `5b47469`，即 PR #5 头部）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 N01：保留原 PPO 编码器结构，使旧权重可以直接对应；把 Actor 梯度隔离与 BN 模式写成可检查的约束 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `encoder.py`、`test_encoder.py`；原 PPO `actor_critic.py`（`MlpBarlowTwinsActor`）与 `common_modules.py`（`mlp_batchnorm_factory`）。建议 Codex 重点解释展平顺序、52 维拼接，以及为什么特征在 `no_grad`、eval 下计算 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文。人类版面向有 Python 基础的读者 |
| 技术初稿 / 语言润色 | 待 Codex 技术初稿；其后由 Gemini 润色同一份人类版 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
