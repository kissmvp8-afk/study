# E04 · 动作边界与初始中心｜AI 实施规范

状态：单元验证通过（纯数据范围）。已实现 [bounds.py](bounds.py) 与 [test_bounds.py](test_bounds.py)，14 项无仿真检查通过；软限位与工作轮速仍是调用方传入的人工数据，未接入仿真或 Actor，未进行训练或性能验证。

[人类版（已完成讲解与润色）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

按每个动作的物理含义计算策略边界，并让初始动作映射到合适的姿态。

术语先说明：软限位：比机械极限收紧的角度范围。半宽：上下界距离的一半，用于把有界采样换算成各动作自己的范围。

## 与原 PPO 的关系

PPO 的宽动作裁剪不能直接作为 SAC 分布的物理边界；作者的统一位置缩放也需要适配 TITA 的轮速动作。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/assets/ddt_robot.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/assets/ddt_robot.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真中验证。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | E03 动作表、实际关节软限位、配置的轮速工作范围、希望的初始物理目标。 |
| 输出 | 每维下界、上界、区间中心、半宽，以及映射后落在默认目标附近的初始分布中心。 |

## 实际文件与接口

实现与验证都在本目录。[bounds.py](bounds.py) 把 `../action_contract` 加入 `sys.path`，直接导入 E03 的 `ActionContract`、`ControlType`。只用标准库，语法兼容 Python 3.11。

| 符号 | 签名 / 字段 | 职责、单位、异常与副作用 |
|---|---|---|
| `ActionBoundsError` | `ValueError` 子类 | 推导、校验或校准失败时抛出，消息写明索引与关节名 |
| `ActionBounds` | 冻结 dataclass：`joint_names, lower, upper, center, half_width`，均为长度等于动作维数的 tuple | 策略动作坐标（无物理单位）下的 L、U、b=(U+L)/2、c=(U-L)/2，顺序与 E03 一致 |
| `derive_bounds` | `(contract: ActionContract, joint_pos_limits: Mapping[str, tuple[float, float]], wheel_speed_ranges: Mapping[str, tuple[float, float]]) -> ActionBounds` | 位置项 L/U = (软限位 - q0)/scale（rad）；速度项 L/U = (工作轮速 - 默认轮速)/scale（rad/s）。按 E03 的控制类型选择查哪个字典，速度项不会读取轮子的角度限位。缺少某维范围或 scale ≤ 0 时抛错，最后调用 `validate_bounds`；无副作用 |
| `validate_bounds` | `(contract, bounds) -> None` | 检查关节顺序与动作表一致、L/U 有限、c > 0、L < 0 < U（默认目标严格在内部）；列出全部问题后抛 `ActionBoundsError` |
| `calibrate_initial_center` | `(contract, bounds, initial_targets: Sequence[float] \| None = None) -> tuple[float, ...]` | a0 = (目标 - default_target)/scale，u0 = atanh((a0 - b)/c)；`None` 表示使用默认目标（a0 = 0）。a0 不严格位于 (L, U) 内，或长度不符时抛错；不夹紧，无副作用 |

调用方：目前只有本目录的验证程序。计划由 N03 动作分布读取 `center`、`half_width` 作为 tanh 映射的 b、c，并把 u0 作为均值输出层的初始偏置；该模块尚未实现。

软限位、工作轮速都由调用方显式传入，本模块不内置 TITA 数值。模型 `velocity_limit_sim=30 rad/s` 不会被自动当作工作轮速。

## 必须成立的规则

- 位置界为 (q_soft_min-q0)/scale 与 (q_soft_max-q0)/scale；不能用绝对值掩盖默认姿态在限位外的问题。
- 轮子单独配置工作轮速；模型 30 rad/s 是上限参考，不是已验证的真机工作规格。
- 非对称边界的内部零均值映射到区间中点，未必是零动作；校准中心是 TITA 适配并记录差异。

## 算法约定与实现提示

先把物理目标换回策略动作的坐标，再设置分布。记下界为 L、上界为 U，区间中心 b=(U+L)/2，半宽 c=(U-L)/2；Actor 最后输出 a=b+c×tanh(u)。u 是网络内部的未压缩数值，a 才交给环境。

若希望初始确定性动作是 a0，则内部中心满足 tanh(u0)=(a0-b)/c。只有 a0 严格在区间内部时，才能得到有限的 u0；在边界上或越界时应显式报告并调整配置，不能静默夹紧后声称精确满足默认姿态。当前 TITA 的 scale 都为正，实现时明确检查这一前提。

作者的 `_compute_action_scaling` 从关节位置限位和一个全局动作缩放推导边界。TITA 的两项轮速控制必须单独处理，不能直接照搬这个函数。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释由物理限位反推动作边界、轮速独立配置、非对称范围的中心校准，以及不可达初始目标的拒绝条件。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Codex 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Codex 编写。已有讲解受影响时标为待 Codex 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 CODEX.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Codex。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（已执行，纯数据范围）

测试中的软限位与工作轮速都是**人工数据**，不是 TITA 实际限位；八维案例的默认目标与 scale 来自 E03 由原版快照生成的动作表。

| 案例 | 对应测试（[test_bounds.py](test_bounds.py)） | 结果 |
|---|---|---|
| q0=0.8、软限位 [0.3, 1.3]、scale=0.25 → [-2, 2]，b=0，c=2，u0=0 | `SingleJointTest.test_readme_example_symmetric_bounds` | 通过 |
| 非对称 [0.5, 1.3] → [-1.2, 2]，b=0.4，c=1.6。未校准（u=0）时目标为 0.9；校准后经 a=b+c·tanh(u0) 和 E03 `action_to_targets` 回到 0.8 / 1.0 / 0.6 | `SingleJointTest.test_asymmetric_center_returns_to_desired_target` | 通过 |
| 八维混合：腿部用软限位与 0.25，轮子用工作轮速 [-10, 20] rad/s 与 5，得 [-2, 4]（不是轮子角度限位的换算，也不是按全局 0.25 得到的 [-40, 80]） | `EightDimMixedTest.test_bounds_per_control_type` | 通过 |
| 八维默认目标与混合初始目标的往返：映射后与期望值一致；非对称维 u0 ≠ 0，对称维 u0 = 0 | `EightDimMixedTest` 其余 2 项 | 通过 |
| 默认姿态越界或恰在限位上、缺少轮速范围（只给了角度限位也不行）、无限限位、空区间、scale ≤ 0、初始目标在边界上或越界、长度不符、边界与动作表不匹配 | `FailureTest` 共 9 项 | 通过 |

运行命令（仓库根目录，无需 Isaac Lab）：

```bash
python3 -B environment_control/action_bounds/test_bounds.py -v
python3 -B environment_control/action_contract/test_contract.py -v   # 依赖的 E03
```

`EightDimMixedTest` 需要本地有固定快照 `8fe40aa`，缺少时会跳过这 3 项并给出原因。

只校准确定性动作：随机采样 a=b+c·tanh(u0+σε) 的平均值在非对称区间下一般不等于 a0，本模块不声称两者相同，也没有测试采样分布。

## 故障定位

开局撞限位先查看映射后的物理目标分布，不只查看网络内部均值。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E03 八维动作定义表](../../environment_control/action_contract/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 人类版初稿与润色

对应代码完成后，Codex 按根 CODEX.md 读取真实文件、函数、注释及验证证据，编写本目录 README_HUMAN.md 的技术初稿；Gemini 再按根 GEMINI.md 润色同一份文件。Gemini 保留代码片段、符号、单位、公式、边界条件和验证结论，技术疑问交回 Codex 核对。

## 实现证据与文档交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（纯数据范围）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [bounds.py](bounds.py)：`derive_bounds`、`validate_bounds`、`calibrate_initial_center`、`ActionBounds`、`ActionBoundsError`，依赖 E03 [contract.py](../action_contract/contract.py)；[test_bounds.py](test_bounds.py)：3 个测试类共 14 项，另用到 E03 的 `action_to_targets`、`build_action_contract`、`load_baseline_action_contract`。目前唯一的调用方是验证程序 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；边界与 u0 都是长度等于动作维数的 tuple，处于策略动作坐标、无单位；输入的软限位单位为 rad，工作轮速为 rad/s；全部函数无副作用 |
| 代码注释核对 | 已逐项核对本模块的注释重点：物理限位反推（`derive_bounds` docstring）、轮速独立配置（按控制类型选字典，不读取轮子角度限位）、非对称中心校准（`calibrate_initial_center`）、不可达目标的拒绝（`validate_bounds` 的 L<0<U 检查与校准中的开区间检查），以及与作者 `_compute_action_scaling` 的三处差异（文件头） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），Python 3.14.5；`python3 -B environment_control/action_bounds/test_bounds.py -v` → `Ran 14 tests … OK`；E03 `test_contract.py` → `Ran 18 tests … OK`；两份文件在 3.11 下语法解析通过，本机没有 3.11 解释器，未实际运行 |
| 未验证项 | TITA 实际软限位（需要 URDF 与 Isaac Lab 的 `soft_joint_pos_limits`，ddt_ros2_control 版本未锁定）；TITA 工作轮速的取值（需人工确定，不能直接用 30 rad/s）；N03 Actor 接入；随机采样动作的分布与开局姿态；Python 3.11 实际运行；仿真、训练与性能 |
| 代码版本 | 实现与验证代码：`4f0ad6e6ea3a06a786f50f120196bfdedcda51f2`（基于 `752ebbd`，main，含 E03），此后 `bounds.py`、`test_bounds.py` 没有改动。人类版讲解在其后的文档提交中，均在分支 `claude/e04-action-bounds`（PR #3） |
| 本次改动及理由 | 首次实现 E04：作者的绝对值写法和全局 scale 无法处理默认姿态越界与 TITA 轮速，因此改为每维带符号反推，并校准非对称区间下的初始确定性动作 |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；E03 AI 规范与 `contract.py`；带注释的 `bounds.py`、`test_bounds.py`；上述验证记录；作者 `sac.py` 中 `_compute_action_scaling` 与 `sac_mlp_model.py` 中 tanh 映射的固定版本。建议 Codex 重点解释 `derive_bounds` 中按控制类型选择范围、`validate_bounds` 中 L<0<U 的含义，以及 `calibrate_initial_center` 中 atanh 反推与“只校准确定性动作”的限制 |
| 人类版 | 已完成讲解与润色；状态与核对范围以 [README_HUMAN.md](README_HUMAN.md) 的元数据为准 |
| 技术初稿 / 语言润色 | Codex 已依据 `4f0ad6e` 完成技术初稿与核对，Gemini 已完成语言润色；两者都不包含仿真验证或重跑测试 |

Claude 每次改动都需更新此表，列出应由 Codex 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Codex 写作并保留占位；代码和注释完成、验证记录齐备后，按 [CODEX.md](../../CODEX.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
