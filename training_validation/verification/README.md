# T06 · 检查组织与维护证据｜AI 实施规范

状态：单元验证通过（CPU）。已实现 [verify.py](verify.py) 与 [test_verify.py](test_verify.py)，6 项检查通过；仿真与性能层级目前没有可执行的检查，记为未执行。

[人类讲解](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md) · [维护规则](../../AGENTS.md)

## 实施目标与术语

统一收集各小模块的检查结果，并检查说明与代码是否同步。

术语先说明：回归检查：把曾经发生的错误变成固定测试，防止以后修改时再次出现。集成检查：把多个模块连起来核对数据传递。

## 与原 PPO 的关系

新增可复查的维护流程；各模块自己的验证代码仍与其 README 和实现放在同一目录。

- [原 PPO：scripts/np3o/train.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/train.py)
- [原 PPO：scripts/np3o/play.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/play.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块已在 CPU 上实现，尚未接入 Isaac 仿真或真实训练结果。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 模块状态、验证记录、README 链接、接口变更和评价报告。 |
| 输出 | 区分文档检查、单元检查、仿真集成与性能评价的验收清单。 |

## 实际文件与接口

依赖：PyTorch（CPU 即可）；各目录不是 Python 包，模块之间通过把相邻目录加入 `sys.path` 互相导入。

### [verify.py](verify.py)

| 符号 | 签名 | 职责与说明 |
|---|---|---|
| `small_modules` | `(repo_root) -> list[Path]` | 四个大模块下含 README.md 的小模块目录（共 24 个） |
| `check_documentation` | `(repo_root) -> list[dict]` | 返回问题列表：根目录与各模块 README / README_HUMAN 中断开的相对链接（去掉 #锚点后检查；外部链接与纯锚点不检查）；小模块“状态：”行与实现 / 测试文件不一致；已实现模块残留“计划文件与逐函数职责”表 |
| `run_test_file` | `(repo_root, test_file, test_names=(), timeout=600) -> dict` | 用当前解释器在子进程中运行已有测试文件（可指定类或方法），解析 unittest 输出 |
| `collect_module_results` | `(repo_root, modules=None) -> list[dict]` | 运行每个小模块的 test_*.py；没有测试文件时为 not_run |
| `INTEGRATION_CHECKS` / `check_integration` | `(repo_root, checks) -> list[dict]` | 跨模块风险 → 已有测试：结束原因与 n 步回报（T02、D03）、历史切片（D05）、导出不重复当前帧（T05）、动作单位与换算（E03、T05）、网络一次完整更新（N08）、训练保存恢复流程（T03）、E05 重置前末状态接 Trainer 与保存恢复（`TrainerIntegrationTest`）、E01 填写的 Isaac 配置驱动 Trainer（`RunConfigTest.test_trainer_runs_on_filled_config`）。后两项的环境钩子是人工模拟，只检查接口衔接，不是仿真检查；共 10 项检查、18 个测试 |
| `build_summary` | `(documentation, unit, integration) -> dict` | 五个层级：documentation、unit、cpu_integration、simulation（not_run：E01/E02/E05 接入代码已实现，真实仿真未验证）、performance（not_run：未进行 PPO / SAC 对照） |
| `main` | 命令行 `--docs` / `--unit` / `--integration` / `--json path` | 不带选项时运行前三层；任一已执行层级失败时退出码为 1 |

测试状态的区别：
- **passed**：全部通过，跳过数单独记录。
- **failed**：有失败或错误。
- **error**：没有得到 unittest 结果，例如导入失败。
- **not_run**：没有测试文件，或该层级尚无检查。

跳过的测试既不算通过，也不算失败。

本模块不重新实现算法，只组织运行各模块已有的测试。修复真实故障后，应在对应模块加入回归测试；如果它覆盖的是跨模块风险，再加入 `INTEGRATION_CHECKS`。

运行方式（仓库根目录；子进程使用同一个装有 torch 的解释器）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/verify.py              # 文档 + 单元 + CPU 集成
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/verify.py --docs       # 只做文档检查
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/verify.py --integration --json report.json
```

## 必须成立的规则

- 文档链接通过不等于代码正确；损失有限不等于机器人会走。
- 修复真实故障后增加能重现该故障的回归检查。
- Claude 必须同步 AI 规范、代码注释与验证证据，并记录人类版待 Codex 同步；不保留失效的计划接口。

## 代码注释要求

共同要求见 [AGENTS.md](../../AGENTS.md#代码与验证)。

本模块特别需要注释：解释不同验证层级的证据来源、状态与文件一致性检查，以及测试失败/跳过/未执行的区别。

## 验收案例（已执行，CPU）

| 案例 | 对应测试（[test_verify.py](test_verify.py)） | 结果 |
|---|---|---|
| 临时仓库中故意制造的断开相对链接、声称已实现却无实现与测试、待实现却有实现文件、已实现却残留计划表，都被发现；外部链接与锚点不误报 | `DocumentationTest.test_detects_documentation_problems` | 通过 |
| 实际仓库共 24 个小模块 | `DocumentationTest.test_lists_all_small_modules` | 通过 |
| 解析通过（含 1 项跳过）、失败（1 失败 1 错误）、导入错误、没有测试文件四种情况 | `ResultCollectionTest.test_statuses` | 通过 |
| D05 切片检查在实际仓库通过；把经验与回报模块复制到临时目录并将决策窗口错位一帧后，同一集成检查失败 | `IntegrationTest.test_detects_injected_history_misalignment` | 通过 |
| 测试文件不存在时为 not_run | `IntegrationTest.test_missing_test_file_is_not_run` | 通过 |
| 五个层级分开汇总；仿真与性能为 not_run；集成失败体现在 cpu_integration 层 | `SummaryTest.test_levels` | 通过 |

调试记录：变异测试初稿只复制了 D01、D05，`test_history.py` 导入 D02～D04 时失败，被正确判为 error 而不是 failed；改为复制整个经验与回报模块后，得到预期的 failed。

运行命令（仓库根目录）：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/test_verify.py -v
```

## 故障定位

出现无法解释的修复时，要求先记录失败案例，再比较修复前后的检查结果。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [T04 PPO 与 SAC 对照评价](../../training_validation/evaluation/README.md)
- [T05 模型导出与运行一致性](../../training_validation/export_inference/README.md)

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 实现证据与文档交接

以下保留实现时的版本与验证记录；后续项目进展见[首页](../../README.md#当前状态)。

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已同步实际接口（见“实际文件与接口”） |
| 实现状态 | 单元验证通过（CPU）；Isaac 仿真、真实训练与性能验证未开始 |
| 实际文件、函数签名与调用关系 | [verify.py](verify.py)：见上表；以子进程运行各模块已有的 test_*.py。[test_verify.py](test_verify.py)：4 个测试类共 6 项 |
| 参数/返回值、形状、单位、异常和副作用 | 见“实际文件与接口”；`main` 可写 JSON；检查本身只读仓库，运行测试时由各测试自行使用临时目录 |
| 代码注释核对 | 已核对：不同验证层级的证据来源、状态与文件一致性检查、失败 / 跳过 / 未执行的区别（文件头、`_parse_unittest`、`build_summary`） |
| 验证命令、环境与结果 | 2026-10-31，macOS（Darwin 25.4.0），uv 临时环境 Python 3.11.17 + torch 2.15.0（CPU），在独立 worktree 中运行；`uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/test_verify.py -v` → `Ran 6 tests … OK`（首次运行有 1 项因变异测试复制范围不足而得到 error，见“调试记录”）。临时环境未装 numpy，torch 打印的 numpy UserWarning 已从记录中去除。`794cd62` 之后（分支 `claude/environment-integration`，同一环境）：`test_verify.py -v` → `Ran 6 tests … OK`；`verify.py --docs` → documentation passed；`verify.py --integration` → cpu_integration passed，10 项检查、测试 18 项、跳过 0（新增的 E05 / E01 两项各自通过），simulation / performance not_run |
| 未验证项 | 仿真集成检查与性能评价层级（目前没有可执行的检查）；CI 集成；Windows 等其他平台上的路径处理 |
| 代码版本 | `c963821163e6b14955c21b987a0972e3216f30aa`（分支 `claude/training-core`，PR #7）；纳入 E01 / E05 集成检查与仿真层说明更新在 `794cd623de54c75ee3ed96ae6788fb72c5a5cebd`（分支 `claude/environment-integration`，PR #8）；文档同步在其后的提交中 |
| 本次改动及理由 | 首次实现 T06：用一个入口组织已有测试与文档检查，分层记录证据，未执行项明确标出。`794cd62`：E01 / E05 已有接 Trainer 的 CPU 检查（人工钩子），加入 `INTEGRATION_CHECKS`；仿真层说明由“尚未接入”改为“接入代码已实现，真实仿真未验证”，仍为 not_run |
| 技术初稿所需材料 | 根/大模块/本模块 AI 规范；依赖模块的实际代码；带注释的 `verify.py`、`test_verify.py`；各模块 AI README 的验收表。建议 Codex 重点解释五个层级的区别、状态一致性规则，以及为什么不重新实现算法 |
| 人类版 | 讲解与技术复核已完成，本轮不再润色；已按 a12e69a 同步 PR #8 接入变化，材料与证据见本目录 README_HUMAN.md |
