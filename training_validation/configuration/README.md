# T01 · 版本、依赖与训练配置｜AI 实施规范

状态：单元验证通过（CPU）。已实现 [config.py](config.py) 与 [test_config.py](test_config.py)，9 项检查通过；`isaac_config` 本身仍有未确定项、不能直接运行，由 E01 用运行时元数据与用户设置填齐（尚未在 Isaac 中运行）。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../README.md#项目维护规则)

## 实施目标与术语

把环境版本、算法候选设置、调试规模和正式实验设置集中记录。

术语先说明：学习率：每次参数更新的步幅设置。版本固定：记录确定的代码提交与依赖版本，使他人能够重复实验。

## 与原 PPO 的关系

增加 SAC 专用配置和安装入口；原 PPO 的固定版本独立保存在 baseline/ppo。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py)
- [原 PPO：scripts/np3o/train.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/train.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已在 CPU 假环境中实现，尚未接入 Isaac 仿真。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)
- [作者 SAC：`rsl_rl/runners/off_policy_runner.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/runners/off_policy_runner.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | UPSTREAM.json、可用 GPU 内存、任务选择、调试或正式模式。 |
| 输出 | 可序列化的完整配置、版本报告、安装/运行说明及内存估算。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可）；仓库尚无统一依赖文件。各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。T02、T03 的测试需要本地有 baseline/ppo 快照（E03 从快照读取动作表）。

### [config.py](config.py)

| 符号 | 签名 / 内容 | 职责与说明 |
|---|---|---|
| `AlgorithmConfig` | gamma、n_steps、tau、target_update_every、actor/critic/alpha/auxiliary 学习率、initial_alpha、target_entropy_scale、initial_std、max_grad_norm、batch_size、critic_updates_per_iteration、actor_update_every、alpha_update_every、auxiliary_updates_per_iteration、auxiliary_batch_size、normalizer_eps | SAC 与辅助学习的全部数值；更新频率以“次 Critic 更新”或“轮迭代”计 |
| `NetworkConfig` | 编码器 128/64 + 32、Actor 与 Q 512/256/128 | 原 PPO TITA 维数 |
| `CollectionConfig` | num_envs、steps_per_iteration、warmup_iterations、capacity_per_env | 采集规模与经验池容量（容量单位：每个环境的记录条数） |
| `TaskConfig` | task、q_dim、velocity_label_columns、joint_pos_limits、wheel_speed_ranges、action_limits_artificial、initial_targets | 任务、Q 维数与物理边界 |
| `RunConfig` | name、mode、seed、device、上述四组、environment、sources | 一次运行的完整配置；`sources` 为“字段路径 → 来源” |
| `AUTHOR_ALGORITHM` / `PAPER_ALGORITHM` | 作者默认 / 论文候选 | 论文候选：gamma 0.97、n 5、tau 0.003、std 0.15、actor/Q lr 2e-4、alpha lr 2e-5；其余为作者默认 |
| `cpu_debug_config(seed=0)` | 名称 `cpu-debug-fake-env`，模式 `cpu_debug` | 论文候选的算法值 + 缩小规模（3 环境、每轮 8 步、容量 64、批 16）+ **人工限位**，只用于假环境 |
| `isaac_config(task="isaac_flat" \| "isaac_rough")` | 名称 `isaac_flat-sac` / `isaac_rough-sac`，模式 `isaac_sim` | q_dim、关节软限位、工作轮速、每轮采集步数、容量都为 None（未确定）；学习器 `device="cpu"`；`environment` 记录资产提交 e7b32d9，`sim_device` 为 None（创建环境时由 E01 填写）。E01 `fill_isaac_run_config` 用运行时值与用户设置填齐后才能运行 |
| `undetermined_fields` | `(config) -> list[str]` | 值为 None 的字段路径 |
| `validate_configuration` | `(config, for_run=True) -> None` | 检查来源齐全、数值范围、模式与任务一致、人工限位不进入仿真配置、**学习器设备必须为 `cpu`**（只实现了 CPU 学习器，改字符串不等于支持 GPU）、速度标签列与 q_dim、n_steps ≤ 容量；`for_run=True` 时拒绝未确定项 |
| `estimate_memory` | `(config, q_dim=None, capacity_per_env=None) -> dict` | 按 D01 字段形状与 dtype 估算 D02 经验池：总字节数与按字段降序的分项；未确定时必须显式给出场景值。**范围只含经验池张量** |
| `check_memory_budget` | `(estimate, available_bytes) -> None` | 超出预算时报错并指出占用最多的字段，不自动缩小规模 |
| `version_report` | `(repo_root) -> dict` | git 提交、是否有已跟踪文件的改动、Python 与 torch 版本、UPSTREAM.json 中的 PPO / SAC 来源与资产版本（e7b32d9） |
| `config_to_dict` / `config_from_dict` | | 与 JSON 基本类型互转 |
| `export_run_configuration` / `load_run_configuration` | `(config, path, repo_root)` / `(path)` | 导出生效配置、来源、未确定项、版本与经验池估算；加载时校验格式版本 |

配置如何生效：T02 `Trainer` 构造时把每个值显式传给对应模块，不依赖各模块的默认常量。T02 测试逐项检查了学习率、初始 std、alpha、eps、网络维数、容量、梯度裁剪、gamma / n、批大小、tau 和各项频率。

未纳入配置的固定值：N02 的非对角权重 0.005（原 PPO）、N03 的 log_std 范围 [−20, 2]（作者）。它们没有对应的调用参数，保持各自模块中的常量。

## 必须成立的规则

- 原基线目标环境为 Isaac Sim 5.1、Isaac Lab 2.3.0、Python 3.11；实际兼容性需安装验证。
- 论文候选：gamma=0.97、n=5、tau=0.003、初始 std=0.15、actor/Q lr=2e-4、alpha lr=2e-5；所有参数可查。
- 工程入口配置可放根目录；实现与验证仍放所属小模块，避免新增第五个业务大模块。

## 代码注释要求

共同要求见 [代码注释规范](../../README.md#代码注释规范)。

本模块特别需要注释：区分原版值、论文候选值与 TITA 适配值；解释参数校验、版本固定及内存估算的字段依据。

## 验收案例（已执行，CPU 假环境）

| 案例 | 对应测试（[test_config.py](test_config.py)） | 结果 |
|---|---|---|
| 调试与仿真配置名称不同、模式不同；调试配置可运行；论文候选与作者默认的取值和来源说明正确 | `PresetTest.test_names_modes_and_sources`、`test_paper_and_author_values` | 通过 |
| Isaac 配置列出 5 个未确定项并拒绝运行；学习器设备为 cpu；资产提交为 e7b32d9 | `PresetTest.test_isaac_config_refuses_to_run_with_undetermined_fields` | 通过 |
| 人工限位放进 Isaac 配置时被拒绝 | `PresetTest.test_artificial_limits_rejected_for_simulation` | 通过 |
| gamma=0、n_steps 超过容量、速度标签超出 q_dim、缺少来源、模式与任务不符、device 为 cuda 时报错 | `PresetTest.test_invalid_combinations` | 通过 |
| 导出再加载后配置完全相等；版本记录中的 git 提交等于 `git rev-parse HEAD` | `ExportTest`（2 项） | 通过 |
| 经验池估算与 D02 `estimate_storage_bytes` 一致，指出占用最多的字段（策略历史）；预算不足时报错 | `MemoryTest.test_estimate_matches_buffer_and_names_largest_part` | 通过 |
| Isaac 场景必须显式给出 q_dim 与容量；241 维、4096×1221 条时为 4096×1221×4654 字节（≈ 23.28 GB） | `MemoryTest.test_isaac_scenario_requires_explicit_values` | 通过 |

调试记录：T03 首次运行时发现 `version_report` 中的 `torch.__version__` 是 `TorchVersion` 对象，检查点无法以 `weights_only` 读取；已改为普通 str（`90b7d21`），并重跑 T01 的 9 项，全部通过。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/configuration/test_config.py -v
```

## 故障定位

同一命令表现不同先比较版本、资产提交、配置快照与随机起点。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- 先阅读根 README 和本大模块 README；本单元可以先用人工构造的元数据独立实现。

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU、假环境、人工数据）；Isaac 仿真集成未开始 |
| 实际文件、函数签名与调用关系 | [config.py](config.py)：见上表；依赖 D01 `_expected_layout`（与 D02 相同的复用方式）、D02 `bytes_per_record`、`WRITE_SEQ_DTYPE`。[test_config.py](test_config.py)：3 个测试类共 9 项。调用方：T02 `Trainer`、T03 检查点 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`export_run_configuration` 写文件，其余函数无副作用 |
| 代码注释核对 | 已核对：原版值 / 论文候选 / TITA 适配 / 调试值的区分（文件头、`sources`）；参数校验（`validate_configuration`）；版本固定（`version_report`）；内存估算的字段依据与范围（`estimate_memory`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/configuration/test_config.py -v` → `Ran 9 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | Isaac Sim 5.1 / Isaac Lab 2.3.0 下的安装与兼容性；真实 q_dim、限位、轮速、每轮步数与容量（由 E01 在运行时填写，尚未在 Isaac 中运行）；GPU 显存实际占用（估算只含经验池）；GPU 学习器（未实现） |
| 代码版本 | `124d442e8c85feb1370088d6bb6d4032e506f0ac`；`version_report` 的修正在 `90b7d21ce6cbde0d9375e7b5e835c00cb122536c`（分支 `claude/training-core`，基于 PR #6 的已复核版本 `42c07a0`，并已合并 main `e43ee42`）；学习器设备改为 CPU 并锁定资产提交在 `bd99b7b87b09b5fd1124ebdd9d842547910a9831`（分支 `claude/environment-integration`）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 T01：配置逐字段记录来源，未确定项保持 None 并拒绝运行，人工限位只允许出现在调试配置。`bd99b7b`：`isaac_config` 原写 `device="cuda"`，但学习器只在 CPU 上实现，改为 `cpu` 并增加校验，仿真设备另记在 `environment["sim_device"]`；E01 锁定资产后填入 `robot_assets_commit` |
| 阅读重点与材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `config.py`、`test_config.py`；作者 `sac.py` / `off_policy_runner.py` 的默认值；原 PPO `np3o_cfg.py`。重点说明三类来源、未确定项如何阻止运行，以及内存估算的范围 |
| 人类版 | 已有讲解，见 [README_HUMAN.md](README_HUMAN.md)；版本与验证范围以该文件记录为准 |
