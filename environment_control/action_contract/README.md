# E03 · 八维动作定义表｜AI 实施规范

状态：单元验证通过（纯数据范围）。已实现 [contract.py](contract.py)、[baseline_source.py](baseline_source.py) 与 [test_contract.py](test_contract.py)，18 项无仿真检查通过；未在 Isaac Lab 中加载环境核对，未进行训练、仿真或性能验证。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

给八个策略输出逐一确定控制对象、类型、单位、默认值和缩放，这是第一个建议实施的小模块。

术语先说明：rad 是弧度，表示角度；rad/s 是每秒转过的弧度，表示转速。默认偏移是动作值为零时对应的控制目标。

## 与原 PPO 的关系

保留 PPO 的物理动作含义；SAC 可以换采样方式，但同一个动作编号仍控制同一个关节。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/assets/ddt_robot.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/assets/ddt_robot.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配已实现纯数据部分，尚未在仿真中验证。

动作执行语义来自 Isaac Lab，不在原版快照内。实现时阅读了 [Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0)（tag 提交 `3c6e67bb5c7ada942a6d1884ab69338f57596f77`）的 `envs/mdp/actions/joint_actions.py`、`actions_cfg.py`、`managers/action_manager.py`，确认三点：

1. `JointAction.process_actions`：`processed = raw * scale + offset`，若配置了 clip，再对 `processed`（物理目标）裁剪。
2. `use_default_offset=True` 时，`JointPositionAction` 的 offset 为 `default_joint_pos`，`JointVelocityAction` 的 offset 为 `default_joint_vel`。
3. `ActionManager._prepare_terms` 按 `cfg.__dict__` 的声明顺序遍历动作项，并依次拼接成动作向量。

这是阅读源码得到的结论，本机未安装 Isaac Lab，未运行验证。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 按环境动作项声明顺序排列的 `ActionTermSpec`（关节名称、控制类型、scale、是否使用默认偏移、clip），以及模型默认关节位置/速度（`init_state.joint_pos` / `joint_vel`，键可为正则）。 |
| 输出 | `ActionContract`：八项 `ActionEntry`。腿部目标 q*=q0+0.25a（rad），轮速目标 w*=0+5a（rad/s），再按原配置裁剪到 [-100, 100]。策略动作 a 无物理单位。 |

## 实际文件与接口

实现与验证都在本目录，与两份 README 并列。本目录不是 Python 包，调用方需把本目录加入 `sys.path` 后再执行 `import contract`；验证程序与 `baseline_source.py` 已这样处理。全部代码只用标准库，语法兼容 Python 3.11。

### [contract.py](contract.py)：动作表数据结构与规则

| 符号 | 签名 / 字段 | 职责、单位、异常与副作用 |
|---|---|---|
| `ActionContractError` | `ValueError` 子类 | 构建/校验失败；消息中逐条列出全部问题 |
| `ControlType` | 枚举 `POSITION`（`JointPositionActionCfg`, `rad`）、`VELOCITY`（`JointVelocityActionCfg`, `rad/s`）；属性 `cfg_class_name`、`unit`；`from_cfg_class_name(name) -> ControlType` | 由 Isaac Lab 配置类名识别类型；遇到力矩、相对位置等不支持的类时抛 `ActionContractError` |
| `TITA_ACTION_LAYOUT` | `tuple[tuple[str, ControlType], ...]`，8 项 | 固定原版的顺序与类型（只固定这两项，不固定 scale 与默认值）；作为 `validate_action_contract` 的默认期望 |
| `ActionTermSpec` | 冻结 dataclass：`term_name, control_type, joint_names: tuple[str,...], scale: float, use_default_offset=True, offset=0.0, clip=None, preserve_order=False` | 一条动作项配置；关节名为精确名称（不支持正则）；缺省值与 Isaac Lab v2.3.0 `JointActionCfg` 一致 |
| `ActionEntry` | 冻结 dataclass：`index, term_name, joint_name, control_type, unit, default_target, scale, clip_low=-inf, clip_high=inf` | 动作向量第 `index` 维的含义；`target = clip(a*scale + default_target)` |
| `ActionContract` | 冻结 dataclass：`entries: tuple[ActionEntry,...]`；`len()`；属性 `joint_names`、`scales`、`default_targets`；方法 `indices_of(control_type) -> tuple[int,...]` | 按列取值，供下游（如 E04）直接转成张量，避免下游重新拼接顺序 |
| `resolve_joint_values` | `(patterns: Mapping[str,float], joint_names: Iterable[str]) -> dict[str,float]` | 正则整串匹配（与 Isaac Lab 相同），把 `{".*": 0.0}` 这类写法展开到具体关节；无匹配或多重匹配时抛 `ActionContractError` |
| `build_action_contract` | `(terms: Sequence[ActionTermSpec], default_joint_pos: Mapping[str,float], default_joint_vel: Mapping[str,float]) -> ActionContract` | 按 terms 顺序展开，索引从 0 连续编号；位置项的偏移取默认关节角，速度项取默认关节速度；多关节项若 `preserve_order=False` 会报错（真实顺序取决于 URDF，不能猜）；不做 TITA 布局校验；无副作用 |
| `validate_action_contract` | `(contract, expected_layout=TITA_ACTION_LAYOUT) -> None` | 检查索引连续、关节不重名、单位与类型一致、scale 有限且非零、默认值有限、clip 上下限顺序；逐索引比对期望布局（重排或按字母序都会被拒绝）；`expected_layout=None` 时只查内部一致性；有问题抛 `ActionContractError` |
| `action_to_targets` | `(contract, action: Sequence[float]) -> tuple[float,...]` | 单条动作换算为物理目标，公式同 Isaac Lab；长度不符或出现 NaN/inf 时抛错；不处理批维，不计算力矩 |
| `describe_action_contract` | `(contract) -> str` | 生成 Markdown 表：序号/索引、动作项、关节、类型、默认目标、缩放、单位、物理目标裁剪 |

### [baseline_source.py](baseline_source.py)：从固定原版快照读取输入

| 符号 | 签名 | 职责、异常与副作用 |
|---|---|---|
| 常量 | `BASELINE_SNAPSHOT_COMMIT`、`BASELINE_TREE`、`ENV_CFG_PATH`、`FLAT_ENV_CFG_PATH`、`ROBOT_CFG_PATH`、`REPO_ROOT` | 与 `UPSTREAM.json` 的 `ppo.snapshot_commit` / `ppo.tree` 一致 |
| `read_baseline_file` | `(path: str, repo_dir=REPO_ROOT, commit=BASELINE_SNAPSHOT_COMMIT) -> str` | 执行 `git show <commit>:<path>`，并核对快照树哈希；提交不在本地、路径不存在或树哈希不符时抛 `RuntimeError`；只读调用 git 子进程 |
| `parse_action_terms` | `(env_cfg_source: str, class_name="ActionsCfg") -> list[ActionTermSpec]` | 用 ast 静态解析，按类体声明顺序返回，值为 `None` 的项跳过；遇到非字面量参数、不支持的动作类或非标量 scale 时抛 `ActionContractError` |
| `parse_init_state` | `(robot_cfg_source: str, cfg_name="DDT_TITA_CFG") -> tuple[dict, dict]` | 返回 `(joint_pos, joint_vel)` 两个正则→数值字典（rad，rad/s） |
| `assert_actions_not_modified` | `(source: str, filename: str) -> None` | 源码中出现 `self.actions` 访问时报错：子类 `__post_init__` 若改写动作，静态读取的结果就不可信 |
| `load_baseline_action_contract` | `(repo_dir=REPO_ROOT) -> ActionContract` | 读取 rough/flat/robot 三个文件，确认 rough 与 flat 都没有改写动作，然后建立动作表（不调用 validate） |
| 命令行 | `python3 -B environment_control/action_contract/baseline_source.py` | 校验并打印由原版配置生成的动作表 |

调用方：本目录验证程序、E01 运行时核对、T02 Trainer 与 T05 导出等已使用；E04 [bounds.py](../action_bounds/bounds.py) 已直接使用 `ActionContract`：逐项读取 `control_type`、`default_target`、`scale`、`joint_name`，并读取 `joint_names`、`default_targets`。D01、Q 网络与导出模块已实现，沿用同一索引顺序；接入代码已有 CPU 证据，真实 Isaac 核对尚未执行。

## 必须成立的规则

- 按动作项顺序映射，不按关节名字字母排序。
- 位置默认值是关节默认角度；轮子速度默认值是零，不能误用轮子角度。
- 本模块说明含义，不生成随机动作、不计算 Q、不决定最终边界。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：标注八维动作索引、关节顺序、rad 与 rad/s 的区别、默认偏移和 scale；解释零动作为何不等于零力矩。

## 验收案例（已执行，纯数据范围）

| 案例 | 对应测试（[test_contract.py](test_contract.py)） | 结果 |
|---|---|---|
| 零动作对应腿部默认姿态及零轮速，不能解释为零力矩 | `AcceptanceCaseTest.test_zero_action_gives_default_posture_and_zero_wheel_speed` | 通过 |
| 仅第 1 个动作设为 0.4：左腿关节 1 的目标由 0.1 变为 0.2 rad，其余不变 | `AcceptanceCaseTest.test_first_action_moves_left_leg_1_from_0_1_to_0_2_rad` | 通过 |
| 仅第 4 个动作设为 0.4：左轮目标为 2 rad/s，不能写成 2 m/s | `AcceptanceCaseTest.test_fourth_action_gives_left_wheel_2_rad_per_s` | 通过 |
| 原版配置生成的表与手写参考表逐项一致（关节、类型、单位、默认值、scale、clip） | `BaselineSnapshotTest.test_matches_reference_table` 等 4 项 | 通过 |
| 按动作项顺序而非字母序（原版顺序恰好等于字母序，因此用打乱的顺序检查） | `RuleTest.test_order_follows_terms_not_alphabetical` | 通过 |
| 轮速偏移取默认速度而非轮子角度（原版两者都是 0，因此用人工元数据区分） | `RuleTest.test_wheel_offset_uses_default_velocity_not_angle` | 通过 |
| 拒绝遗漏、重名、轮子标成位置控制、单位写成 m/s、零 scale、歧义输入、非法动作 | `RuleTest` 中其余 9 项 | 通过 |

运行命令（仓库根目录，无需 Isaac Lab；`-B` 避免在 exFAT 盘上生成 `__pycache__`）：

```bash
python3 -B environment_control/action_contract/test_contract.py -v
python3 -B environment_control/action_contract/baseline_source.py
```

`BaselineSnapshotTest` 需要本地有 git 并存在快照提交 `8fe40aa`（可用 `git fetch origin baseline/ppo` 获取），缺少时这 4 项会被跳过并给出原因，其余 14 项不依赖 Git。

测试能否发现错误也做了检查：在 scratchpad 副本中分别注入两个错误，(a) 速度项偏移改用默认关节角，(b) 构建后按关节名字母排序。(a) 被 2 项测试捕获，(b) 被 1 项测试捕获。其中 (a) 在原版对照测试中**不会**暴露，因为原版两者都是 0，所以人工元数据案例是必需的。

## 故障定位

轮子突然过快先查索引、单位和 scale；左右对应异常先查关节名称映射。可运行 `baseline_source.py` 打印当前动作表与原版比对；若上游配置改为非字面量参数，或在 `__post_init__` 中改写 `self.actions`，解析会直接报错，此时需要改为在 Isaac Lab 中从 `env.action_manager` 读取实际动作项来核对。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- 先阅读根 README 和本大模块 README；本单元可以先用人工构造的元数据独立实现。

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（纯数据范围）；集成验证未开始 |
| 实际文件、函数签名与调用关系 | [contract.py](contract.py)：`build_action_contract`、`validate_action_contract`、`action_to_targets`、`describe_action_contract`、`resolve_joint_values` 及数据类；[baseline_source.py](baseline_source.py)：`read_baseline_file`、`parse_action_terms`、`parse_init_state`、`assert_actions_not_modified`、`load_baseline_action_contract`；[test_contract.py](test_contract.py)：3 个测试类共 18 项。该条调用范围为首次实现时记录，当前调用关系见上方接口说明 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”两张表；动作为长度 8 的一维序列，无物理单位；目标单位为 rad 或 rad/s；只读调用 git 子进程，没有其他副作用 |
| 代码注释核对 | 已逐项核对本模块的注释重点：索引与顺序（`TITA_ACTION_LAYOUT`、`parse_action_terms`）、rad 与 rad/s 的区别（`ControlType`）、默认偏移（`build_action_contract` 速度分支）、scale/clip（`ActionEntry`、`action_to_targets`）、零动作不等于零力矩（`action_to_targets` docstring 与零动作测试） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），Python 3.14.5，仓库位于 exFAT 外置盘；`python3 -B environment_control/action_contract/test_contract.py -v` → `Ran 18 tests … OK`；`baseline_source.py` 打印的表与下方原配置表一致；两项错误注入均被捕获；三份文件在 3.11 下语法解析通过（`ast.parse(feature_version=(3,11))`），本机没有 3.11 解释器，未实际运行 |
| 未验证项 | Isaac Lab 实际加载后的 `ActionManager` 动作顺序与默认值（E01 `compare_with_baseline` 已实现该核对，尚未在 Isaac 中运行）；URDF 关节表（E01 已锁定 ddt_ros2_control e7b32d9，CPU 测试确认 8 个受控关节都在 URDF 中）；执行器力矩行为；Python 3.11 实际运行；仿真、训练与性能 |
| 代码版本 | 实现与验证代码：`9fdac4e5055989085eafd6b3c988c40d9df61ca9`（基于 `c6078f4`；本目录 3 个 .py 文件，以及本 README、本目录 README_HUMAN.md 元数据、大模块 README 状态、根 README 状态句），此后 3 个 .py 文件没有改动。人类版讲解：`ef7a5f0a7de22da413bc737cbab322e86e3c1f9e`。均在分支 `claude/e03-action-contract`（PR #2） |
| 本次改动及理由 | 首次实现 E03：用固定快照的原配置作为唯一的数值来源（静态解析，不手抄数值），按 Isaac Lab v2.3.0 的动作语义建表；校验规则对应 README 的“必须成立的规则” |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；带注释的三份代码；上述验证命令与输出；Isaac Lab v2.3.0 的三处语义依据（见“与原 PPO 的关系”）；建议 Codex 重点解释 `build_action_contract` 速度分支的偏移来源、`validate_action_contract` 为何逐索引比较、`action_to_targets` 中先换算再裁剪的顺序，以及 `parse_action_terms` 为何以声明顺序作为动作顺序 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；已按 a12e69a 同步 PR #8 接入变化，材料与证据见本目录 README_HUMAN.md |


## 已核对的原配置动作表

第一列同时列出人阅读的第几个数与从零开始的程序索引。下表是 `python3 -B environment_control/action_contract/baseline_source.py` 由快照 `8fe40aa`（树 `88bb121`）实际生成的输出，与测试中的手写参考表一致。原版 rough 与 flat 任务共用这张表（flat 配置不改写动作）。实际加载后的配置仍待 E01 在 Isaac Lab 中核对。

| 序号 / 索引 | 动作项 | 关节名称 | 类型 | 默认目标 | 缩放 | 物理目标单位 | 物理目标裁剪 |
|---|---|---|---|---:|---:|---|---|
| 1 / 0 | `joint_pos_0` | `joint_left_leg_1` | 角度 | 0.1 | 0.25 | rad | [-100, 100] |
| 2 / 1 | `joint_pos_1` | `joint_left_leg_2` | 角度 | 0.8 | 0.25 | rad | [-100, 100] |
| 3 / 2 | `joint_pos_2` | `joint_left_leg_3` | 角度 | -1.5 | 0.25 | rad | [-100, 100] |
| 4 / 3 | `joint_vel_3` | `joint_left_leg_4` | 轮速 | 0 | 5 | rad/s | [-100, 100] |
| 5 / 4 | `joint_pos_4` | `joint_right_leg_1` | 角度 | 0.1 | 0.25 | rad | [-100, 100] |
| 6 / 5 | `joint_pos_5` | `joint_right_leg_2` | 角度 | 0.8 | 0.25 | rad | [-100, 100] |
| 7 / 6 | `joint_pos_6` | `joint_right_leg_3` | 角度 | -1.5 | 0.25 | rad | [-100, 100] |
| 8 / 7 | `joint_vel_7` | `joint_right_leg_4` | 轮速 | 0 | 5 | rad/s | [-100, 100] |

腿部默认目标来自 `DDT_TITA_CFG.init_state.joint_pos`。轮速默认目标来自 `init_state.joint_vel={".*": 0.0}`，与轮子角度（恰好也是 0.0）无关。裁剪作用在物理目标上，±100 rad 或 ±100 rad/s 在正常范围内不起作用，不是 SAC 的动作边界（见 E04）。

零动作不等于零力矩：腿部执行器 `DCMotorCfg` 的 stiffness=60、damping=1.5，关节偏离默认角时仍会朝默认姿态出力；轮子 `ImplicitActuatorCfg` 的 stiffness=0、damping=0.5，零动作的效果是把轮速拉向 0，相当于制动。
