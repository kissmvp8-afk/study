# T06 · 检查组织与维护证据｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex 写作约定](../../CODEX.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色；保留有效正文并同步 PR #8 接入变化 |
| 技术核对状态 | 2026-10-31，Codex 按实际实现与测试同步资产、设备、公共环境接口或集成检查；本次是文档核对，不扩大代码审核 |
| 语言润色状态 | 保留已有中文讲解与历史润色记录；本轮由 Codex 技术同步，不再交 Gemini |
| 已核对代码版本 | 原实现 main `f6c8b03d`；Windows 路径修复 `373bb48d`；新增诊断及本轮证据随本次提交保存，未改算法 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 本轮阅读 verify.py/test_verify.py、794cd62 差异、E05 TrainerIntegrationTest 与 E01 RunConfigTest 的实际测试及 AI 运行记录；检查器测试本身未改动；全局材料与本轮检查见[项目阅读记录](../../README_HUMAN.md#阅读材料与验证依据) |
| 验证依据与限制 | 2026-10-31 Windows 真实平地 1 环境的加载、50/1005 步交互、短训练保存恢复及提前计数纯截断通过；历史 CPU 记录保留，本轮不重跑全仓。截断仅实际推进 0.04 s，不证明稳定运行 20 s；复杂地形、长期训练、性能与稳定行走未验证。证据见下方 Windows 验证记录 |

## Windows 验证记录（2026-10-31）

Windows 平地单环境最小闭环已验证；提前设置计数器的真实纯时间截断检查通过。复杂地形、长期训练、性能与稳定行走仍未验证。 [精简报告、命令与 JSON 证据](../../environment_control/robot_environment/validation/windows_2026-10-31/REPORT.md)。

提前计数截断仅实际推进 0.04 秒，不证明稳定运行满 20 秒。T06 入口本身未执行仿真，simulation/performance 仍为 not_run；详细计数、命令、告警和数据边界见上述证据。

## 五层证据回答五个不同的问题

单元测试针对局部行为；集成测试检查模块之间的数据传递；回归测试把曾出现的错误固定为可重复案例。T06 组织已有检查，不另写一个 SAC 算法来验证另一个算法。

| 层级 | 当前入口与证据 | 能回答的问题 |
|---|---|---|
| `documentation` 文档 | 链接、模块状态与文件、失效计划表检查 | 文件与说明是否在结构上同步；不证明公式、单位或全部文字正确 |
| `unit` 单元 | 各小模块已有 `test_*.py` | 给定人工输入下，局部计算和边界是否符合断言 |
| `cpu_integration` CPU 集成 | 从已有测试选择跨模块案例 | 假环境结束原因、回报、历史、动作、网络更新、保存恢复能否正确衔接 |
| `simulation` 仿真 | 当前 `not_run` | 本检查器没有执行仿真，仍记 not_run；Windows 平地单环境和提前计数纯截断的外部证据见 E01 验证记录 |
| `performance` 性能 | 当前 `not_run` | 需要真实 PPO/SAC 训练对照与 T04 证据，现阶段没有结论 |

运行前三级通过也不能替代后两级。“损失有限”只是数值检查结果，“机器人训练成功”需要独立的环境与任务证据。当前没有仿真或性能层的检查可供入口运行。

## `verify.py` 的输入与输出

输入是仓库路径、各模块 Markdown、实现与测试文件，以及命令行选择的层级；输出为终端分层摘要，可选 JSON，退出码为 0 或 1。它只使用 Python 标准库；`--docs` 不导入 torch。运行算法测试的子进程才需要相应依赖，并使用同一个解释器。

常量 `BIG_MODULES` 指四个大模块，`ROOT_DOCUMENTS` 指根入口，`INTEGRATION_CHECKS` 把跨模块风险关联到已有测试，`NOT_RUN_LEVELS` 固定说明当前仿真/性能未执行的原因。这里只记录可执行证据，不根据 README 中的“通过”代填运行结果。

### 文档发现与结构核对

- `small_modules(repo_root=REPO_ROOT)` 返回四大模块下含 `README.md` 的子目录，按路径排序；当前共 24 个小模块。
- `_markdown_files(repo_root)` 枚举根文档及四大模块中的 `README*.md`，排除 `._` 文件。
- `check_documentation(repo_root)` 返回问题字典列表，空列表表示本项通过。读取相对链接时去掉锚点，检查文件或目录存在；跳过外部 URL、邮件与纯锚点。再检查小模块 AI README 的 `状态：` 行与 `.py`/`test_*.py` 是否一致，已实现时不应残留“计划文件与逐函数职责”表。

它不会验证网络链接是否可访问、锚点是否真的存在、函数签名、公式或实现正确性；这些仍要人工核对。源码文件存在也不等于实现完整。

### 运行已有测试并解析结果

| 函数 | 输入 → 输出与副作用 |
|---|---|
| `_parse_unittest(output,returncode)` | unittest 文本及退出码 → 状态、测试数、失败/错误/跳过数；没有 `Ran N tests` 行时为 error，并保留末行信息 |
| `run_test_file(repo_root,test_file,test_names=(),timeout=600)` | 路径、可选测试类/方法 → 解析后的结果字典；用子进程执行 `[sys.executable,'-B',文件,*测试名]`，默认超时 600 s |
| `collect_module_results(repo_root,modules=None)` | 模块列表或全仓库 → 每个测试文件结果列表；没有测试文件的模块记 not_run，不凭空记通过 |
| `check_integration(repo_root,checks=INTEGRATION_CHECKS)` | 风险/文件/测试名条目 → 集成结果列表；文件不存在记 not_run，否则复用 `run_test_file` |

这些函数会实际运行测试；本次写作只使用 `--docs`，没有触发它们。标准输出/错误被捕获后解析，具体复现失败可直接运行结果中的文件及 `tests_selected`；超时等子进程异常需看终端错误，不能误称测试通过。

`INTEGRATION_CHECKS` 当前十项选择，原八项为：T02 的 `EpisodeBoundaryTest`；D03 的 `BoundaryTest`；D05 的 `SplitWindowsTest`；T05 重复当前帧回归；E03 的 `AcceptanceCaseTest`；T05 单次物理换算；N08 的 `UpdateCycleTest`；T03 的 `FullFlowTest` 与精确回放恢复用例。新增两项是 E05 `test_lifecycle.py` 的 `TrainerIntegrationTest`（2 个测试），以及 E01 `test_environment.py` 的 `RunConfigTest.test_trainer_runs_on_filled_config`（1 个测试）。前者检查末状态进入经验池、D03 回报与 T03 恢复；后者检查运行时元数据填出的 Isaac 模式配置可驱动 Trainer。两者都用 `ScriptedSimulation` 人工钩子，没有运行真实 Isaac。当前共 10 项检查选择、展开 18 个测试。

### 汇总状态与命令行

`_level(results)` 把同层结果聚合，None 表示未选择；排除 not_run 后，有 failed/error 则该层 failed，否则已执行项 passed，同时保留跳过和未执行数量。`build_summary(documentation,unit,integration)` 生成五层字典；文档参数是问题列表，空列表通过，None 未选。仿真/性能直接按 `NOT_RUN_LEVELS` 记 not_run。

结果状态需要连同计数阅读：

| 状态 | 准确含义 |
|---|---|
| `passed` | 子进程正常退出且解析结果没有失败/错误；跳过另记，不代表每项测试都执行过 |
| `failed` | 已得到测试汇总，但有失败/错误或退出码不成功 |
| `error` | 未得到标准 unittest 汇总，例如导入阶段失败 |
| `not_run` | 没有测试文件、层级未选或尚无可执行检查 |

`main(argv=None)` 解析 `--docs/--unit/--integration/--json`，调用上述函数、打印摘要，可写 JSON，并返回退出码。两个关键判定为：

```python
run_all = not (args.docs or args.unit or args.integration)
```

没有选任何层级就运行文档、单元和 CPU 集成；只写 `--json` 也会运行三层。因此本轮必须显式带 `--docs`。多个层级选项可以组合。

```python
return 1 if any(summary[level]["status"] == "failed" for level in ("documentation", "unit", "cpu_integration")) else 0
```

只要前三层中有失败就返回 1；尚未执行仿真/性能不会令命令失败。退出码 0 因此不是“全项目验证完成”。脚本末尾 `sys.exit(main())` 把这个返回值交给命令行。

## 怎样运行而不混淆证据

仓库根目录只做文档检查，本轮实际采用这个标准库入口：

```bash
python3 -B training_validation/verification/verify.py --docs
```

需要将来复现完整前三层时，使用带 torch 的环境（以下命令本轮未执行）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/verify.py
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/verify.py --integration --json report.json
```

T06 自身的测试入口为 `training_validation/verification/test_verify.py -v`，使用相同带 torch 的解释器可复现；它与实际运行全仓库检查是两种证据。

## `test_verify.py` 验证检查器是否有用

辅助 `write(path,text)` 在临时目录写去缩进的人工文件。6 项既有测试分成四类：

- `DocumentationTest.test_detects_documentation_problems` 故意造断链、状态与文件冲突、残留计划表，核对都被发现；`test_lists_all_small_modules` 核对实际 24 目录。
- `ResultCollectionTest.test_statuses` 构造通过/跳过、失败/错误、导入失败和无测试文件，核对解析口径。`Ran` 的总数包含跳过，跳过另计，不应再把总数全部当通过数。
- `IntegrationTest.test_detects_injected_history_misalignment` 在临时副本将 D05 决策窗口错位一帧，同一既有集成测试失败；`test_missing_test_file_is_not_run` 核对缺文件不冒充通过。
- `SummaryTest.test_levels` 核对五层分开、集成失败保留、仿真与性能 not_run。

通过记录来自本目录 [AI 交接表](README.md#实现证据与文档交接)，本轮没有重跑这 6 项或完整算法检查。它们不验证 Windows 路径、CI 接入、真实仿真与性能；实际执行的文档检查结果见[大模块证据记录](../README_HUMAN.md#阅读材料与证据)。

断链先看结果的 `file/target`；状态错误看 AI README 的 `状态：` 与文件清单；error 先检查依赖和导入，failed 再按测试名定位断言。修复真实跨模块故障时，回归测试应放回所属模块，再由这里选择组织，不在检查器中复制算法。

PR #8 的 AI 记录在 `794cd623de54c75ee3ed96ae6788fb72c5a5cebd` 后执行 `verify.py --integration`：18 个测试通过、跳过 0，仿真/性能仍 not_run；`test_verify.py` 6 项也通过。这是引用既有结果，本轮只运行 `--docs`，不会借此生成新的单元或集成通过记录。
