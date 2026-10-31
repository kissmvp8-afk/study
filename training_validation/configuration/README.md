# T01 · 版本、依赖与训练配置

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

把环境版本、算法候选设置、调试规模和正式实验设置集中记录。

术语先说明：学习率：每次参数更新的步幅设置。版本固定：记录确定的代码提交与依赖版本，使他人能够重复实验。

## 与原 PPO 的关系

增加 SAC 专用配置和安装入口；原 PPO 的固定版本独立保存在 baseline/ppo。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/agents/np3o_cfg.py)
- [原 PPO：scripts/np3o/train.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/train.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)
- [作者 SAC：`rsl_rl/runners/off_policy_runner.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/runners/off_policy_runner.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | UPSTREAM.json、可用 GPU 内存、任务选择、调试或正式模式。 |
| 输出 | 可序列化的完整配置、版本报告、安装/运行说明及内存估算。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `config.py` | `validate_configuration` | 检查参数组合、动作类型与依赖约束 |
| `config.py` | `estimate_memory` | 按记录字段和容量估算经验池占用 |
| `config.py` | `export_run_configuration` | 保存本次真实使用的配置 |
| `test_config.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 原基线目标环境为 Isaac Sim 5.1、Isaac Lab 2.3.0、Python 3.11；实际兼容性需安装验证。
- 论文候选：gamma=0.97、n=5、tau=0.003、初始 std=0.15、actor/Q lr=2e-4、alpha lr=2e-5；所有参数可查。
- 工程入口配置可放根目录；实现与验证仍放所属小模块，避免新增第五个业务大模块。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 调试与正式配置具有不同明确名称，不把调试结果记录为正式复现。
- 实际资源不足时显示哪一部分占用最多，不静默减小实验规模。
- 配置导出再加载后关键字段一致。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

同一命令表现不同先比较版本、资产提交、配置快照与随机起点。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- 先阅读根 README 和本大模块 README；本单元可以先用人工构造的元数据独立实现。

依赖链接表示必须理解的数据约定，不表示必须同时实现所有相关模块。上游接口改变时，检查本模块的输入说明和验证案例。

## 当前实现与验证记录

| 内容 | 当前状态 |
|---|---|
| README 设计与职责 | 已编写，等待实施时与实际代码保持同步 |
| 实现文件和真实函数说明 | 待 Claude 实现 |
| 关键代码逐行说明 | 待代码存在后填写；不得编造行号 |
| 单元验证命令与结果 | 待实现 |
| 仿真集成结果（如适用） | 未验证 |
| 修改记录 | 本次仅建立模块契约与验收要求 |
