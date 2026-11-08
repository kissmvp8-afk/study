# T01 · 版本、依赖与训练配置｜人类阅读版

## SAC-R1-20261107 设备扩展（本地未执行 Torch）

`RunConfig.device` 现在可为 `cpu` 或 `cuda[:index]`，只指学习器。`validate_configuration(for_run=True)` 对不可用 CUDA/越界设备报错，不隐式回退；其他后端拒绝。`for_run=False` 可记录尚未执行的 CUDA 配置。CPU 仍是默认值，E01 原启动入口仍默认 CPU。

T02 已补齐网络、学习批次与动作的显式搬运，不能仅凭字符串断言 GPU 验证通过。经验池留在 CPU，仿真设备另存 `environment.sim_device`。新增针对性验证在 T02 `device_validation.py`，实际服务器结果应单独归档；这里没有把历史 CPU 结果改写为新 GPU 通过证据。`test_invalid_combinations` 的非法后端改为 `meta`，不再依赖机器是否有 CUDA。


[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色；保留有效正文并同步 PR #8 接入变化 |
| 技术核对状态 | 2026-10-31，Codex 按实际实现与测试同步资产、设备、公共环境接口或集成检查；本次是文档核对，不扩大代码审核 |
| 语言润色状态 | 保留已有中文讲解与历史润色记录；本轮由 Codex 技术同步，不再交 Gemini |
| 已核对代码版本 | 本轮 `a12e69a38c6b9b3a236473a2563e255a9a24e761`（claude/environment-integration，PR #8），fetch 后 git rev-parse HEAD 与已审查/远端版本一致；历史版本见正文与 Git 历史 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 本轮阅读 config.py/test_config.py、E01 fill_isaac_run_config 及其测试、UPSTREAM.json；对照 bd99b7b 的设备和资产差异，保留原配置与内存讲解；全局材料与本轮检查见[项目阅读记录](../../README_HUMAN.md#阅读材料与验证依据) |
| 验证依据与限制 | 既有 CPU 结果见本目录 AI 交接表；PR #8 记录全仓库 216 项测试（跳过真实资产 1 项）及 18 项 CPU 集成测试通过，本轮未重跑。只执行 verify.py --docs 与文档静态核对；真实 Isaac、GPU、机器人训练及性能未验证 |

## 为什么先把配置说清楚

一条 SAC（Soft Actor-Critic，软演员—评论家算法）训练曲线，只有同时知道算法参数、采集规模、输入维数和代码版本，才有可比较的含义。T01 把这些设置放进一个 `RunConfig`，再逐字段记录来源。它不启动机器人，也不替你挑选“最优参数”。

Actor 是生成动作的策略网络；Q 网络估计给定状态与动作的未来回报；alpha 是鼓励策略多样性的温度系数。归一化是用观测的均值和尺度调整网络输入。

“超参数”指学习率、折扣率等由实验者指定的设置；“随机种子”指定伪随机序列的起点；“预热”指先收集经历、暂不学习。原 PPO 使用近期数据计算优势并更新策略；这里需要额外明确经验池容量、多步回报和 SAC 各次更新的频率。

## 数字从哪里来

[config.py](config.py) 用来源字符串区分作者默认、论文候选、原 PPO、TITA 暂定适配、CPU 调试和未确定项。`sources` 是字段路径到来源说明的字典，例如 `algorithm.gamma`，不能只写一句“参数来自论文”覆盖整份配置。

| 设置 | `AUTHOR_ALGORITHM` | `PAPER_ALGORITHM` / CPU 调试的算法起点 | 含义 |
|---|---:|---:|---|
| `gamma` | 0.998 | 0.97 | 每个策略周期的未来奖励折扣 |
| `n_steps` | 1 | 5 | 最多累计几步奖励 |
| `tau` | 0.005 | 0.003 | 目标 Q 每次向在线 Q 靠近的比例 |
| Actor / Q 学习率 | 0.001 | 0.0002 | 每次优化的步幅设置 |
| alpha 学习率 | 0.001 | 0.00002 | 探索温度的优化步幅 |
| 内部初始标准差 | 1.0 | 0.15 | 高斯采样在 tanh 压缩前的尺度 |
| Actor 更新间隔 | 2 | 2 | 按 Critic 更新计频，具体相位见 T02 |

`PAPER_ALGORITHM` 通过 `replace(AUTHOR_ALGORITHM, ...)` 只替换七项候选值。其余设置仍保留逐字段来源；例如辅助学习率来自原 PPO，辅助更新次数是本项目暂定值。候选值只是实验起点，尚未证明适合 TITA。作者版本登记为 `e0d243a`，论文出处见 [AI 规范](README.md)；本轮核对的是本地配置与交接记录，没有重新复现论文。

`cpu_debug_config(seed=0)` 再把规模缩成 3 个环境、每轮 8 步、预热 1 轮、每环境容量 64、学习批量 16、每轮 4 次 Critic 更新、辅助批量 16。Q 观测 `C=6` 来自人工环境，不是 E02 的真实机器人观测。人工腿部范围分别为 `[-0.3,0.7]`、`[0,1.6]`、`[-2.4,-0.6]` rad，左右相同；人工轮速范围为 `[-10,20]` rad/s。这些数不能写入真实任务充当 TITA 限位。

## 配置对象怎样进入训练器

`config.py` 中的五个冻结数据类按职责分组。冻结限制属性重新赋值，内部字典仍可修改；实验中应把配置当作固定记录使用。

| 类 | 输入、保存内容与下游 |
|---|---|
| `AlgorithmConfig` | 折扣、多步数、四种学习率、初始温度/标准差、目标熵倍率、梯度裁剪、批量和更新频率；T02 显式传给 D03、D04、N01～N08 的对应入口 |
| `NetworkConfig` | 编码器骨干 `128/64`、潜特征隐藏层 `32`、Actor/Q 隐藏层 `512/256/128`，沿用原 PPO 结构；N01 的输出 3+16 维不由这里重新定义 |
| `CollectionConfig` | 环境数 E、每轮步数、预热轮数、每环境容量 K；总容量为 E×K，不能把 K 当作全池条数 |
| `TaskConfig` | 任务名、Q 维数 C、速度标签列、位置/轮速范围、是否人工数据、希望的初始物理目标；E04 用范围推导策略边界 |
| `RunConfig` | 上述四组、名称、模式、种子、设备、环境版本和来源；整体交给 T02 `Trainer` |
| `ConfigurationError` | 配置检查专用异常，消息列出问题；调用方据此定位字段 |

速度标签列 `(0,3)` 表示左闭右开三列。原 PPO 的身体线速度观测乘过 2.0，标签因此不是直接的 m/s；Isaac 接入时仍需 E02 确认列顺序。位置目标单位 rad，轮速目标单位 rad/s，策略动作本身无物理单位。`initial_targets=None` 明确表示使用 E03 默认目标，不能当作“忘填配置”。

`RunConfig.device` 明确指学习器设备，现在支持 `cpu` 或可用 `cuda[:index]`，`validate_configuration` 对其他后端或不可用设备报错。E01 的 `--device` 则指仿真设备，记在 `environment["sim_device"]`；E05 搬运数据连接两者。仿真用 GPU 不代表网络和经验池在 GPU 上学习。

## `config.py` 的函数与关键计算

- `cpu_debug_config(seed)` 返回可执行的 CPU 人工流程配置；`isaac_config(task, seed)` 返回 `isaac_flat` 或 `isaac_rough` 的待补齐配置。未知任务抛 `ConfigurationError`，两者不启动训练。
- `_field_paths(config)` 列出需说明来源的字段（不含 `name/mode/environment/sources`，人工标记也不单列来源）；`undetermined_fields(config)` 返回其中值为 `None` 的路径，排除有明确语义的 `task.initial_targets`。
- `validate_configuration(config, for_run=True)` 返回 `None` 或抛异常。它检查来源、折扣/比例/频率/正值、模式与任务、人工限位、学习器设备合法且可用、三列标签和 `n_steps≤K`。`for_run=False` 允许保存未确定的规划配置；`True` 还拒绝未确定项。它不是完整的物理配置验收器。
- `estimate_memory(config, q_dim=None, capacity_per_env=None)` 返回记录数、C、K、总字节与降序字段明细；两个可选参数只用于明确估算场景，不修改配置。
- `check_memory_budget(estimate, available_bytes)` 超预算时指出最大字段并报错，不自动缩小实验，以免记录的规模与实际运行不符。

经验池估算的核心是：

```python
records = config.collection.num_envs * capacity
parts = {name: records * math.prod(shape) * torch.empty((), dtype=dtype).element_size()
         for name, (shape, dtype) in _expected_layout(q_dim).items()}
parts["write_seq"] = records * torch.empty((), dtype=WRITE_SEQ_DTYPE).element_size()
total = sum(parts.values())
```

第一行用 E×K 得到条数；第二段直接复用 D01 的字段形状和类型，每字段的元素数乘每元素字节，避免维护另一份估算表；第三行补入 D02 每槽位的 int64 写入序号；最后求和。遗漏两份 `[10,33]` 历史会严重低估容量。单条为 `2726+8C` 字节；人工 C=6 时，3×64 条约占 532608 字节。

测试还用 C=241、E=4096、K=1221 估算约 23.28 GB（十进制）。这只是经验池字段与写入序号；未计 `_writes[E]` 的 8E 字节、网络、优化器、激活、缓存分配器和 Isaac Sim 自身占用。`storage_bytes` 才包含池内写指针；估算通过不能保证整次训练放得下。

保存配置和记录版本的函数如下：

| 函数 | 输入 → 输出、调用与副作用 |
|---|---|
| `version_report(repo_root)` | 仓库路径 → Git 提交、已跟踪文件是否有改动、Python/torch 版本、PPO/SAC 固定来源和资产版本；内部 `git(*args)` 只读调用子进程。torch 版本转为普通字符串，供 T03 的 `weights_only` 读取 |
| `config_to_dict(config)` | 数据类 → JSON 基本类型，tuple 变 list；无文件写入 |
| `config_from_dict(data)` | 字典 → 五类配置对象；内部 `tuples(d,keys)` 恢复网络维数、标签列和初始目标等 tuple，物理范围的值也恢复 tuple |
| `export_run_configuration(config,path,repo_root)` | 写 JSON 并返回记录，包含格式版本 1、生效配置、未确定项、版本和内存估算；估算条件不足时记录原因，仍可导出规划配置 |
| `load_run_configuration(path)` | 读上述 JSON → `RunConfig`；格式版本不符时报错。加载不等于可运行，运行前仍需校验 |

配置导出文件与 T03 检查点不同：前者记录设置，后者还包含网络、优化器、统计、随机数和可选经验池。

## Isaac 配置哪些事尚未确定

`isaac_config` 仍有五项 `None`：`task.q_dim`、`task.joint_pos_limits`、`task.wheel_speed_ranges`、`collection.steps_per_iteration`、`collection.capacity_per_env`。平地 C=54、复杂地形 C=241 都只是 E02 预期，需运行时核对。轮子模型上限也不能直接当作工作轮速。

资产提交已在 `TARGET_ENVIRONMENT` 锁定为 `e7b32d9746df900680641ff2845ffb0b5ffe99ed`，并与 UPSTREAM.json/E01 一致；不再是待锁定项。`sim_device=None` 在创建环境时由 E01 填写，也不属于 `undetermined_fields` 的五项。目标 Isaac Sim 5.1、Isaac Lab 2.3.0、Python 3.11 尚无本轮安装/运行证据。CPU 调试配置可用于人工流程；Isaac 模板经 E01 填齐才可交给学习器，真实运行未验证。

## 测试怎样支持这些说明

[test_config.py](test_config.py) 有 9 项既有测试：`PresetTest` 的五项核对命名/来源、候选与作者值、五项未确定、拒绝人工限位进入仿真配置和非法组合（含拒绝 cuda 学习器）；资产与版本断言已改为锁定提交；`ExportTest` 的两项核对往返/版本和拒绝其他格式；`MemoryTest` 的两项核对 D02 字节公式、超预算报错和必须显式给出估算场景。它们用标准 `unittest.main()` 入口，没有自定义测试数据工厂。值是否真的进入优化器由 T02 测试继续验证。

对应上文断言的精确方法名如下。测试方法由 unittest 调用，无业务参数，成功时返回 None；测试工厂与设置/清理方法的职责见上文。

| 测试类 | 方法定位（职责与预期见上文） |
|---|---|
| `PresetTest` | `test_names_modes_and_sources`、`test_paper_and_author_values`、`test_isaac_config_refuses_to_run_with_undetermined_fields`、`test_artificial_limits_rejected_for_simulation`、`test_invalid_combinations` |
| `ExportTest` | `test_round_trip_and_versions`、`test_rejects_other_format` |
| `MemoryTest` | `test_estimate_matches_buffer_and_names_largest_part`、`test_isaac_scenario_requires_explicit_values` |

需要复现时在仓库根目录运行（本轮未执行）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/configuration/test_config.py -v
```

配置报错先看异常中的完整字段路径；内存不足先看 `by_field_bytes`，再把训练额外占用算入预算；恢复时学习率“被改回去”先看 [T03 恢复规则](../checkpoints/README_HUMAN.md)。下一步按 [T02](../training_loop/README_HUMAN.md) 看配置怎样实际驱动流程。

## 从 Isaac 模板到本次生效配置

[E01 fill_isaac_run_config](../../environment_control/robot_environment/metadata.py) 先以 `isaac_config` 为模板，用已核对的运行时 C、六腿软限位、实际环境数填值；用户显式工作轮速、每轮步数和每环境容量补齐剩余字段，再校验并导出。`sources` 区分运行时、用户值和接入调试值。默认入口规模 16 环境、8 步/轮、容量 200、训练 3 轮用于接入排查，不能当作正式训练方案。

现有 `config.py` 文件头还保留“UPSTREAM.json 尚未锁定”旧注释，与当前 `TARGET_ENVIRONMENT` 和 UPSTREAM.json 不符；本文已按实际常量更正。该文字残留不影响配置取值，本轮按范围未修改源码。
