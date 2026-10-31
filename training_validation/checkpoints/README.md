# T03 · 训练进度保存与恢复｜AI 实施规范

状态：单元验证通过（CPU、假环境）。已实现 [checkpoint.py](checkpoint.py) 与 [test_checkpoint.py](test_checkpoint.py)，10 项检查通过；只支持按检查点原有学习配置恢复；不保存仿真现场。

[人类版（待 Codex 编写）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

让中断后的实验可以有明确语义地恢复，并保留定位变化所需的信息。

术语先说明：检查点：保存训练进度的一组数据。原子性保存：先完整写好，再替换正式文件，减少中断留下半文件的风险。

## 与原 PPO 的关系

原 PPO 保存网络与优化器；SAC 增加双 Q/目标 Q、多个优化器、alpha、辅助模块及回放状态约定。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/runner.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/runner.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已在 CPU 假环境中实现，尚未接入 Isaac 仿真。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)
- [作者 SAC：`rsl_rl/runners/off_policy_runner.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/runners/off_policy_runner.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 所有网络、优化器、归一化统计及计数、随机状态、训练配置与可选经验池。 |
| 输出 | 带格式版本的检查点、恢复报告和一致性检查结果。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可）；仓库尚无统一依赖文件。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。T02、T03 的测试需要本地有 baseline/ppo 快照（E03 从快照读取动作表）。

### [checkpoint.py](checkpoint.py)

| 符号 | 签名 | 职责与副作用 |
|---|---|---|
| `CHECKPOINT_FORMAT` / `CHECKPOINT_VERSION` | `"tita-sac-checkpoint"` / `1` | 格式标记与版本；没有格式标记的文件（例如原 PPO 模型）一律拒绝 |
| `DESCRIPTIVE_FIELDS` | `name`、`sources`、`environment` | 恢复时唯一允许与检查点不同的配置字段（只用于描述）；其余配置（algorithm、network、collection、task、mode、seed、device）必须逐字段一致 |
| `collect_training_state` | `(trainer, include_replay) -> dict` | 汇总全部训练状态（见下表），只读 |
| `save_checkpoint` | `(trainer, path, include_replay=True) -> Path` | 先写同目录临时文件并 fsync，再 `os.replace`；失败时删除临时文件，旧检查点不变 |
| `restore_checkpoint` | `(trainer, path) -> ResumeReport` | `torch.load(weights_only=True)`；依次校验格式、版本、学习配置逐字段一致、E04 边界与 E03 关节顺序，全部通过后才修改 Trainer。配置不同时报错，列出全部差异，并提示用 `load_checkpoint_config` 重建 Trainer。恢复后环境 `reset`，episode_id 从保存时加一 |
| `ResumeReport` | mode、replay_restored、environment_state_restored（恒为 False）、counters、saved_versions | 恢复结果；返回时学习配置已校验与检查点一致 |
| `describe_resume_mode` | `(report) -> str` | `exact_replay`：在学习配置一致（已校验）的条件下，下一批回放与学习可复现（已在 CPU、同一 PyTorch 版本下验证），采集从新一轮开始；`model_only`：经验池为空，需要重新积累，不是精确接续 |
| `load_checkpoint_config` | `(path) -> RunConfig` | 读取检查点中的生效配置，用于按原配置重建 Trainer |

### 检查点内容

| 类别 | 内容 |
|---|---|
| 网络 | N01 编码器（含 BN 运行统计）、N02 projector、N03 Actor（含 E04 的 b、c 缓冲）、N04 在线与目标双 Q、N07 log_alpha |
| 优化器 | critic、actor、alpha、auxiliary 四个优化器的状态 |
| 统计 | D04 两套统计的 mean、var、count、eps |
| 调度与随机数 | T02 全部计数（决定 Actor 的更新相位与预热）、collect / replay / learn 三个生成器的状态 |
| 来源 | 生效配置、版本信息、E04 动作边界、E03 关节顺序 |
| 可选 | D02 经验池（字段、写入序号、写入次数，使用 D02 新增的 `state_dict` 接口） |
| 不保存 | 环境（仿真）现场：中断时未结束的那一轮不会接着进行；新一轮从 step 0 开始，D02 按显式编号判断连续性，不会与旧经历拼接 |

作者 `SAC.save` 只保存 Actor、Critic、优化器和 alpha，外加迭代号；没有保存经验池、随机数状态和 `update_step` 计数。

## 必须成立的规则

- 不保存经验池时明确要求再次积累经历，不能称为精确接续。
- 原 PPO 的 V 参数不能直接当作 SAC 的 Q 参数加载。
- 保存输入统计计数与历史编码相关状态，不能只保存权重和偏置。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释检查点版本、原子保存、全部训练状态、经验池缺失的恢复限制，以及 PPO V 不能直接加载为 Q。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，CPU 假环境）

Trainer 使用 T01 `cpu_debug_config` 与 T02 `FakeLocomotionEnv`。

| 案例 | 对应测试（[test_checkpoint.py](test_checkpoint.py)） | 结果 |
|---|---|---|
| 训练 3 轮并做辅助学习后保存，恢复进新 Trainer：同一组固定输入的确定性动作、Q1、Q2、目标 Q1、log_alpha、D04 均值 / 方差 / 计数、编码器 BN 运行均值、四个优化器的 exp_avg、全部计数都完全相等；模式为 exact_replay | `RestoreTest.test_outputs_statistics_and_counters_match` | 通过 |
| 带经验池恢复：原 Trainer 与恢复后的 Trainer 各再学习 3 次并辅助学习 1 次，全部参数逐位相等，计数相等，下一批抽样的起点相同 | `RestoreTest.test_exact_replay_reproduces_next_learning` | 通过 |
| 不带经验池：模式为 model_only，经验池为空，说明中含“不是精确接续”；确定性动作一致；学习先被跳过，重新采集后可以继续 | `RestoreTest.test_model_only_requires_new_experience` | 通过 |
| 恢复后环境开始新一轮（episode_id = 保存时 + 1，step 0，长度 1）；中断时未结束的旧记录不会与新写入的记录连接 | `RestoreTest.test_environment_restarts_without_joining_old_experience` | 通过 |
| 学习配置不同（actor_lr、normalizer_eps、gamma、工作轮速、Actor 网络维数，各一例）时，在修改 Trainer 之前拒绝：报错列出对应字段并提示 `load_checkpoint_config`；Actor 参数、优化器学习率、D04 eps、计数都保持构造时的值 | `CompatibilityTest.test_rejects_learning_config_differences_before_changing_state` | 通过 |
| 用 `load_checkpoint_config` 重建（只改描述性的 name）时可以恢复 | `CompatibilityTest.test_rebuilding_from_checkpoint_config_is_accepted` | 通过 |
| 检查点中保存的动作边界被篡改时拒绝 | `CompatibilityTest.test_rejects_tampered_action_bounds` | 通过 |
| 原 PPO 式的模型文件、版本号不符的检查点被拒绝；可以读回检查点中的配置 | `CompatibilityTest.test_rejects_other_formats_and_versions` | 通过 |
| 写入中途失败：旧检查点字节不变，没有残留临时文件 | `CompatibilityTest.test_atomic_save_keeps_previous_file` | 通过 |
| 一次 CPU 流程：训练 4 轮、每 2 轮保存一次 → 按检查点配置重建 → 恢复 → 再训练 2 轮，计数接续（iterations 6，Critic 次数增加 8） | `FullFlowTest.test_train_save_restore_continue` | 通过 |

调试记录：首次运行时 6 项报错、2 项失败。一处是测试的导入顺序问题；另一处是 T01 `version_report` 写入了 `TorchVersion` 对象，`weights_only` 读取按设计拒绝了它。将其改为普通 str 后，T03 的 8 项和 T01 的 9 项全部通过。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/checkpoints/test_checkpoint.py -v
```

## 故障定位

恢复后动作改变先比较归一化、边界、历史顺序和加载模式，再查训练随机性。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [D02 经验池写入与抽样](../../experience_returns/replay_buffer/README.md)
- [D04 观测归一化与统计管理](../../experience_returns/normalization/README.md)
- [N01 历史状态编码](../../networks_learning/history_encoder/README.md)
- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)
- [N04 双 Q 评分网络](../../networks_learning/q_networks/README.md)
- [N07 探索力度 alpha](../../networks_learning/entropy_temperature/README.md)
- [N08 目标 Q 的慢更新](../../networks_learning/target_updates/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、假环境、人工数据）；Isaac 仿真集成未开始 |
| 实际文件、函数签名与调用关系 | [checkpoint.py](checkpoint.py)：`save_checkpoint`、`restore_checkpoint`、`describe_resume_mode`、`collect_training_state`、`load_checkpoint_config`、`ResumeReport`、`CheckpointError`；读取 T02 `Trainer` 的属性（encoder、projector、actor、critics、targets、temperature、optimizers、normalizers、counters、generators、buffer、observation、bounds、contract、config、env），调用 D02 `state_dict` / `load_state_dict`、D04 `export_statistics` / `load_statistics`、T01 `config_to_dict` / `version_report`。[test_checkpoint.py](test_checkpoint.py)：3 个测试类共 10 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`save_checkpoint` 写文件；`restore_checkpoint` 修改 Trainer 的全部训练状态并调用 `env.reset` |
| 代码注释核对 | 已核对：检查点版本与格式标记、原子保存、全部训练状态、经验池缺失的恢复限制、原 PPO V 不能加载为 Q、仿真现场未保存（文件头、`restore_checkpoint`、`describe_resume_mode`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/checkpoints/test_checkpoint.py -v` → `Ran 10 tests … OK`（首次实现时为 8 项，问题见上文“调试记录”；审核修复后新增 2 项、改写 1 项）。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | Isaac 环境的现场保存与恢复（不在本模块范围内，目前也没有承诺）；大容量经验池的保存耗时与文件大小；跨设备（GPU ↔ CPU）恢复；异常中断后的实际恢复演练 |
| 代码版本 | `90b7d21ce6cbde0d9375e7b5e835c00cb122536c`，审核修复 `d213a24425ab9588fa4e77e64fb87106116dd401`；依赖 D02 新增接口 `f81a92a2720694b31460f9a9174f758ce9524a55`（分支 `claude/training-core`，基于 PR #6 的已复核版本 `42c07a0`，并已合并 main `e43ee42`）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 T03：在作者保存内容的基础上补齐经验池、随机数与调度计数；明确两种恢复语义，以及不恢复仿真现场。审核修复（Codex 复核 `13382f3`）：原实现允许恢复时使用不同的学习配置，但优化器状态与 D04 统计会带回旧的学习率与 eps，导致配置记录与实际运行不一致，再次保存时仍不一致；改 gamma 后仍报告“下一批学习可复现”也不成立。现改为只支持按检查点原有学习配置恢复：不一致时在修改 Trainer 之前报错并列出差异，提示用 `load_checkpoint_config` 重建 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `checkpoint.py`、`test_checkpoint.py`；T02 `runner.py`；D02 `buffer.py` 的 `state_dict`；作者 `sac.py` 的 `save` / `load`。建议 Codex 重点解释原子保存、两种恢复语义、结构检查，以及为什么恢复后旧经历不会与新一轮拼接 |
| 人类版 | 待 Codex 编写；本次只更新了元数据，未写正文 |
| 技术初稿 / 语言润色 | 待 Codex 技术初稿；其后由 Gemini 润色同一份人类版 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
