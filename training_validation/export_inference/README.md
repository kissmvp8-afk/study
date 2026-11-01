# T05 · 模型导出与运行一致性｜AI 实施规范

状态：单元验证通过（CPU）。已实现 [export.py](export.py) 与 [test_export.py](test_export.py)，6 项检查通过；导出格式为 TorchScript；CPU 一致性不代表真机部署通过。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

让导出的 Actor 在相同观测与历史下，产生与训练内推理一致的动作。

术语先说明：导出：把训练框架中的模型转换成便于运行的文件。推理：使用固定参数算动作，不进行训练更新。

## 与原 PPO 的关系

保留历史编码的运行链，导出 SAC 的确定性动作及其压缩缩放；Q 与训练用 BT 转换模块不进入动作执行链。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)
- [原 PPO：scripts/np3o/play.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/play.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块已在 CPU 上实现，尚未接入 Isaac 仿真或真实训练结果。

## 作者实现参考

- [作者 SAC：`rsl_rl/models/sac_mlp_model.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/models/sac_mlp_model.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 固定模式的 Actor/编码器、输入统计、历史契约、动作映射和目标运行格式。 |
| 输出 | 导出模型、输入输出说明、同输入对比报告和 50 Hz 控制周期约定。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可）；各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [export.py](export.py)

| 符号 | 签名 | 职责与说明 |
|---|---|---|
| `InferencePolicy` | `(encoder, actor, normalizers)`；`forward(policy_history[B,10,33]) -> action[B,8]` | 深拷贝并冻结编码器、Actor 与 D04 统计；forward 直接调用 N01 `features_from_history`（D04 → D05 → N01）与 N03 `deterministic_action` |
| `build_inference_pipeline` | `(trainer) -> InferencePolicy` | 由 T02 Trainer（或恢复后的 Trainer）构造；只读 Trainer |
| `export_metadata` | `(trainer) -> dict` | 输入输出说明、E03 关节顺序 / 类型 / 单位 / 默认目标 / scale / clip、E04 区间、控制周期 0.02 s、统计计数、配置名、版本、排除项与验证范围 |
| `export_policy` | `(trainer, path, example_batch=4) -> dict` | `torch.jit.trace`（`check_trace=True`）后 `torch.jit.save` 到 path，并写出 `path.with_suffix('.json')` 元数据 |
| `load_exported_policy` | `(path) -> ScriptModule` | `torch.jit.load` 后设为 eval |
| `compare_outputs` | `(reference, exported, histories, atol=1e-6) -> dict` | 只比较确定性动作：最大绝对差、是否在容限内、样本数 |

### 使用方法与接口

```python
metadata = export_policy(trainer, "policy.pt")      # 同时写出 policy.json
policy = load_exported_policy("policy.pt")
action = policy(policy_history)                     # [B,10,33] 原始历史（含当前帧，从旧到新）→ [B,8] 策略动作
targets = action_to_targets(contract, action[0].tolist())  # E03：物理目标只在运行端换算一次
```

- **输入**：原始（未归一化）的策略观测历史，含当前帧、从旧到新；归一化在模型内部用冻结统计完成。
- **输出**：策略动作坐标，E03 顺序，位于 E04 区间 [L, U] 内。**不包含物理换算**；物理目标 = default_target + scale × a，再按 clip 裁剪，由运行端执行一次（Isaac Lab 中即环境动作项）。如果导出模型内部再换算一次，就会重复缩放。
- **历史对齐**：只接收一份已含当前帧的历史，不再追加当前帧。原 PPO `_InferenceWrapper` 的重复当前帧问题有回归测试。
- **不进入动作执行链**：Q、目标 Q、Barlow projector、随机采样、温度。

格式选择与已知警告：选用 TorchScript，原因是它只依赖 PyTorch，与原 PPO 的导出方式一致，并且在 Isaac Lab 2.3 对应的 torch 版本中可用。
- 在 torch 2.15 中，`torch.jit.trace/save/load` 会给出“已弃用，建议改用 torch.export”的 FutureWarning；`torch.export` 可作为后续选项。
- trace 时，各模块的输入形状检查会产生 TracerWarning，被记录为常量。窗口长度与帧维由契约固定，批大小仍是动态的（测试用了与 trace 示例不同的批大小）。

## 必须成立的规则

- 历史输入已含当前帧时不能再次追加；原 _InferenceWrapper 存在这一静态不一致，需要明确处理。
- 比较同样的确定性动作，不拿随机采样结果作逐元素等值比较。
- 仿真导出通过不等于真机验证；物理单位、关节顺序和运行周期仍需确认。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：解释训练和导出历史对齐、确定性动作的统一映射、输入统计冻结和误差容限；保留原历史重复问题的回归说明。

## 验收案例（已执行，CPU）

模型来自 T02 Trainer 在假环境上训练 3 轮并做 2 次辅助学习的结果（cpu_debug_config，人工限位）。

| 案例 | 对应测试（[test_export.py](test_export.py)） | 结果 |
|---|---|---|
| 训练内推理、eager 管线、导出并重新加载后的模型三条路径一致（批大小 7，与 trace 示例的 4 不同）；重复调用结果相同 | `ExportConsistencyTest.test_three_paths_match` | 通过 |
| 带帧编号的历史：导出模型与训练内推理一致；把已含当前帧的历史再拼一次当前帧（原 PPO `_InferenceWrapper` 的情形），动作随之改变（回归） | `ExportConsistencyTest.test_frame_numbered_history_is_not_duplicated` | 通过 |
| 导出后再采集（D04 统计更新）：导出模型输出不变，训练内推理随之改变 | `ExportConsistencyTest.test_statistics_frozen_at_export` | 通过 |
| 极端输入（×1e4）时动作在 [L, U] 内；重置历史与极端输入下 eager 与导出模型都一致 | `ExportConsistencyTest.test_bounds_and_reset_history` | 通过 |
| 元数据写入并可读回；关节顺序 = E03 `TITA_ACTION_LAYOUT`；输出坐标为 policy_action，不含物理换算；控制周期 0.02 s；导出输出经 E03 换算一次，等于训练内动作经 E03 换算一次；轮速项物理目标 = 5 × a | `InterfaceTest.test_metadata_and_single_physical_conversion` | 通过 |
| 导出模块只含编码器与 Actor 参数（不含 Q、projector），不求梯度、eval 模式 | `InterfaceTest.test_q_and_projector_not_exported` | 通过 |

调试记录：首次运行时测试文件因导入路径缺少配置目录而无法启动；补上路径后 6 项全部通过，实现无改动。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/export_inference/test_export.py -v
```

## 故障定位

导出后异常时先定位输入与变换差异，不先改奖励重训。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)
- [D04 观测归一化与统计管理](../../experience_returns/normalization/README.md)
- [D05 历史窗口与有效性](../../experience_returns/history_windows/README.md)
- [E03 八维动作定义表](../../environment_control/action_contract/README.md)
- [E04 动作边界与初始中心](../../environment_control/action_bounds/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU）；Isaac 仿真、真实训练与性能验证未开始 |
| 实际文件、函数签名与调用关系 | [export.py](export.py)：`InferencePolicy`、`build_inference_pipeline`、`export_metadata`、`export_policy`、`load_exported_policy`、`compare_outputs`；调用 N01 `features_from_history`、N03 `deterministic_action`、T01 `version_report`，读取 T02 Trainer 的 encoder、actor、normalizers、contract、bounds、config。[test_export.py](test_export.py)：2 个测试类共 6 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`export_policy` 写 .pt 与 .json 两个文件；管线只读，不修改 Trainer |
| 代码注释核对 | 已核对：训练与导出历史对齐及原 PPO 重复当前帧的回归说明、确定性动作的统一映射、输入统计冻结、误差容限、物理换算接口（文件头、`InferencePolicy`、`compare_outputs`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/export_inference/test_export.py -v` → `Ran 6 tests … OK`。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除 |
| 未验证项 | 真机或仿真中的部署运行；目标运行端（C++ / ONNX 等）的加载；物理单位、关节顺序与 50 Hz 周期在真机上的确认；GPU 上的数值差异；torch.export 格式 |
| 代码版本 | `82d6e8371daad31e2069d9b726014c0446af2c4c`（分支 `claude/training-core`，PR #7）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 T05：导出路径复用训练内推理代码，从结构上避免两条入口切片不一致；输出保持策略坐标，物理换算只在运行端执行一次 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `export.py`、`test_export.py`；N01 `encoder.py`、N03 `actor.py`、D05 `history.py`；原 PPO `actor_critic.py` 的 `_InferenceWrapper`。建议 Codex 重点解释为什么复用训练路径、统计如何冻结、输出与 E03 换算的分工，以及原 PPO 重复当前帧的问题 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；核对版本 a07d7b2，材料、证据及限制见本目录 README_HUMAN.md |
