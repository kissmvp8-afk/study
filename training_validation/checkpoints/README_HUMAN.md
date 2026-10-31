# T03 · 训练进度保存与恢复｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex 写作约定](../../CODEX.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 对照实际实现、注释与测试核对职责、调用、形状、单位、代码片段和边界；本次是文档核对，不是全面代码审核 |
| 语言润色状态 | 本轮由 Codex 完成中文讲解与技术核对，不另交 Gemini |
| 已核对代码版本 | `a07d7b22e5bc2f40c462c01cc3378f7bcf3fe0c7`（`claude/training-core`，PR #7）；读取时与用户验收版本、远端分支一致，无后续源码差异 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | AGENTS.md、CODEX.md、根及本大模块/本目录 AI README；[checkpoint.py](checkpoint.py)、[test_checkpoint.py](test_checkpoint.py) 全文；T01 配置序列化、T02 状态与计数、D02 保存恢复、D04 统计导出/加载实现；依赖与来源详见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 本目录 AI 交接表记录 10 项 CPU 检查通过（Python 3.11.17 + torch 2.15.0），本次阅读测试、引用既有结果，未重跑算法测试。本轮执行的文档检查见大模块记录；人工数据/假环境不代表 Isaac 仿真、GPU、机器人训练或性能验证 |

## 保存的不只是 Actor 权重

检查点（checkpoint）是在某个时刻保存的一组训练状态。Actor 是生成动作的策略网络，Critic 是估计回报的 Q 网络；优化器还记着过去的梯度统计，归一化器记着观测的均值、方差和样本数。只保存权重，继续训练时的输入、步幅、抽样与更新相位都可能改变。

T03 因此围绕 T02 `Trainer` 保存完整学习状态，并明确区分“接着回放学习”与“接着原环境采集”。本版本不保存环境现场，恢复后一律从新回合采集。它也不能把原 PPO 的状态价值 V 网络直接装成 SAC（软演员—评论家算法）的动作价值 Q 网络：Q 还需要动作输入，结构和意义都不同。

## 文件里的对象与函数

[checkpoint.py](checkpoint.py) 的格式标识为 `tita-sac-checkpoint`，版本为 1。`CheckpointError` 表示格式或训练结构不兼容。`MODULE_NAMES` 列出六组网络；`DESCRIPTIVE_FIELDS` 限定恢复时允许变化的描述字段。

| 接口 | 输入 → 输出、调用关系与副作用 |
|---|---|
| `ResumeReport` | 冻结数据类，保存 `mode`、`replay_restored`、始终为 False 的 `environment_state_restored`、恢复计数和原版本；供日志解释恢复含义 |
| `_bounds_record(trainer)` | 将 E04 的关节顺序、上下界、中心、半宽变为基本类型字典，供保存与兼容性比较；只读 |
| `collect_training_state(trainer,include_replay)` | 汇总下表内容为字典；不改 Trainer，不写文件。网络 `state_dict` 的张量可能是引用，D02/统计导出是副本，不能把该字典长期当作与训练隔离的快照 |
| `save_checkpoint(trainer,path,include_replay=True)` | 汇总后原子写入，返回 `Path`；创建父目录、写临时文件并替换正式文件 |
| `_learning_config_differences(saved,current)` | 两份配置字典 → 差异路径列表；跳过三项描述字段，其余比较后报告检查点值和当前值 |
| `restore_checkpoint(trainer,path)` | 读取、校验并恢复到新构造的 Trainer，返回 `ResumeReport`；修改网络、优化器、统计、计数、随机流和经验池，并调用环境 reset |
| `describe_resume_mode(report)` | 报告 → 中文说明字符串，无状态修改；用于把恢复边界写进日志 |
| `load_checkpoint_config(path)` | 只读出检查点的 `RunConfig`，便于重建 Trainer；检查格式标识，完整版本及兼容性检查仍由 `restore_checkpoint` 执行 |

各目录以 `sys.path` 导入相邻模块，不是可独立安装的 Python 包；恢复示例应从仓库根目录按当前目录结构使用。

## 保存哪些状态，为什么缺一不可

| 内容 | 具体对象与原因 |
|---|---|
| 六组网络 | `encoder/projector/actor/critics/targets/temperature`；包括编码器 BN（批归一化）的运行统计、Actor 的 E04 中心/半宽、在线/目标双 Q 和 `log_alpha` |
| 四个优化器 | `critic/actor/alpha/auxiliary` 的状态，包括学习率与 Adam 梯度动量；恢复这些才是继续优化 |
| D04 观测统计 | policy 33 维与 Q 的 C 维，各自 mean/var/count/eps/dim；否则相同原始观测会变成不同网络输入 |
| T02 进度 | 全部 `TrainingCounters`；Critic 次数决定 Actor 更新相位，迭代数决定预热和保存时机 |
| 随机数 | `collect/replay/learn` 三个生成器的状态；回放起点与策略噪声都需要接续 |
| 配置与来源 | 生效配置、Git/Python/torch/固定上游版本、E04 边界和 E03 关节顺序，防止结构相同但动作意义变了 |
| 可选经验池 | D02 全部字段存储、写入序号、逐环境写入次数和 E/K/C；序号决定合法连续性和下一写槽位 |
| 新回合编号 | `next_episode_ids`，用于恢复后避免把新采集接到旧回合 |

历史仍按原始 `[B,10,33]` 保存，动作仍是 `[B,8]` 的策略坐标；保存检查点不会把经历转换成物理目标。作者 SAC 的保存项与本项目补充项的来源说明见 [AI 规范](README.md)；本轮实际核对本地实现，没有新增跨框架检查点兼容能力。

## 恢复必须使用相同学习配置

恢复优化器会带回旧学习率，加载 D04 会带回旧 eps。如果当前配置写 Actor 学习率 0.0007，而文件里的优化器仍是 0.0002，放行就会产生“记录的是新值、实际用的是旧值”的实验。

因此恢复先核对格式/版本，再执行：

```python
problems = _learning_config_differences(state["config"], config_to_dict(trainer.config))
if problems:
```

第一行收集学习配置差异，第二行进入报错分支，在加载任何网络或优化器前拒绝。允许不同的只有 `name`、`sources`、`environment`；算法、网络、任务、采集规模乃至 seed、device、mode 都要求一致。随后还独立核对 E04 边界与 E03 顺序，才依次加载。

推荐路径是 `config=load_checkpoint_config(path)` → 用该配置与匹配环境构造 `Trainer` → `restore_checkpoint(trainer,path)`。这不是换参数微调接口；改配置续训属于后续需求。学习配置不一致时的提前拒绝已由测试覆盖；不要把它扩写成“任意损坏文件恢复失败都会自动回滚”的保证。

读取用 `torch.load(..., weights_only=True)`，只接受相应张量和基本类型，拒绝本版本不支持的 pickle 内容。无格式标识的 PPO 文件会明确报错。

## 两种恢复方式，各能接续到哪里

| 保存方式 / 恢复 `mode` | 恢复后的经验池 | 能保证的范围 |
|---|---|---|
| `include_replay=True` / `exact_replay` | 原经验池、覆盖序号和计数都在 | 相同学习配置、CPU、同一 PyTorch 版本下，下一批回放与学习已由既有测试复现；不保证继续原环境轨迹 |
| `include_replay=False` / `model_only` | 新的空池 | 网络、优化器、归一化统计、计数和随机流仍恢复；需要重新收集合法经历后才能学习，不是精确接续 |

`model_only` 这个名字不表示“只加载 Actor 权重”。它保留旧的迭代数和更新计数，不把预热轮数重新归零；经验不足时由 T02 的合法起点检查跳过学习，直到补齐样本。

环境重启的关键代码是保存时的：

```python
"next_episode_ids": (trainer.observation.episode_id + 1).tolist(),
```

以及恢复时的：

```python
trainer.observation = trainer.env.reset(first_episode_ids=torch.tensor(state["next_episode_ids"], dtype=torch.int64))
```

第一行给每个环境预留一个新轮次号，第二行以这些号 reset，而不是恢复关节、接触或随机化现场。例如保存时轮次 `[2,4,3]`，恢复从 `[3,5,4]` 开始，步号为 0、有效历史为 1。D02 要求同轮次且步号递增，旧回合未完成的尾记录不会与新采集拼接。恢复后再采集，结果自然可能与未中断运行不同。

## 原子保存保护的是什么

原子替换指正式路径在替换时切换为新文件，避免训练中断留下半个正式检查点。核心操作为：

```python
torch.save(state, handle)
handle.flush()
os.fsync(handle.fileno())
os.replace(tmp, path)
```

第一行写同目录临时文件；第二行把 Python 缓冲交给系统；第三行要求同步文件内容；第四行才替换正式路径。若前面失败，异常分支删除临时文件、保留旧正式文件。这个保证针对保存过程，不是环境现场恢复或所有硬件故障的保证。

## `test_checkpoint.py` 的证据与定位方法

测试辅助函数：`new_trainer(config)` 构造 CPU 假环境；`trained_trainer()` 跑 3 轮并补 3 次辅助更新，使优化器/BN 偏离初值；`probe(trainer)` 用固定输入读取动作、Q、alpha、统计；`params(trainer)` 复制状态供逐项比较。`RestoreTest`、`CompatibilityTest` 的 `setUp/tearDown` 只管理临时文件。

10 项既有测试覆盖：`RestoreTest` 四项检查输出/优化器/计数一致、接着三次回放和一次辅助更新仍一致、不含池需新经历、新回合不接旧记录；`CompatibilityTest` 五项检查配置差异在修改前拒绝、原配置重建可用、边界被改后拒绝、其他格式/版本拒绝、模拟写入中断保留旧文件；`FullFlowTest.test_train_save_restore_continue` 检查训练 4 轮按间隔保存，恢复后再跑 2 轮到 6 轮、Critic 增加 8 次。

它们是 CPU 人工环境的既有结果，详见 [AI 交接表](README.md#实现证据与文档交接)，本轮未重跑。
对应上文断言的精确方法名如下。测试方法由 unittest 调用，无业务参数，成功时返回 None；测试工厂与设置/清理方法的职责见上文。

| 测试类 | 方法定位（职责与预期见上文） |
|---|---|
| `RestoreTest` | `test_outputs_statistics_and_counters_match`、`test_exact_replay_reproduces_next_learning`、`test_model_only_requires_new_experience`、`test_environment_restarts_without_joining_old_experience` |
| `CompatibilityTest` | `test_rejects_learning_config_differences_before_changing_state`、`test_rebuilding_from_checkpoint_config_is_accepted`、`test_rejects_tampered_action_bounds`、`test_rejects_other_formats_and_versions`、`test_atomic_save_keeps_previous_file` |

需要复现时，在仓库根目录运行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/checkpoints/test_checkpoint.py -v
```

恢复报错先看格式/版本和差异字段，再核对 E03 顺序/E04 范围；输出不同先查编码器 BN 与 D04 统计；抽样不同查 D02 序号和三个生成器；采集不同先确认是否只是正常的环境重启。继续阅读 [D02 保存接口](../../experience_returns/replay_buffer/README_HUMAN.md#供-t03-使用的经验池保存与恢复) 与 [T05 导出](../export_inference/README_HUMAN.md)。
