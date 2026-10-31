# T03 · 训练进度保存与恢复｜AI 实施规范

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[人类版（等待代码完成）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

让中断后的实验可以有明确语义地恢复，并保留定位变化所需的信息。

术语先说明：检查点：保存训练进度的一组数据。原子性保存：先完整写好，再替换正式文件，减少中断留下半文件的风险。

## 与原 PPO 的关系

原 PPO 保存网络与优化器；SAC 增加双 Q/目标 Q、多个优化器、alpha、辅助模块及回放状态约定。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/runner.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/runner.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)
- [作者 SAC：`rsl_rl/runners/off_policy_runner.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/runners/off_policy_runner.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 所有网络、优化器、归一化统计及计数、随机状态、训练配置与可选经验池。 |
| 输出 | 带格式版本的检查点、恢复报告和一致性检查结果。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本目录两份 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `checkpoint.py` | `save_checkpoint` | 原子性保存模型、统计和来源信息 |
| `checkpoint.py` | `restore_checkpoint` | 校验兼容性并恢复状态 |
| `checkpoint.py` | `describe_resume_mode` | 区分完整恢复与重新积累经验的续训 |
| `test_checkpoint.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 不保存经验池时明确要求再次积累经历，不能称为精确接续。
- 原 PPO 的 V 参数不能直接当作 SAC 的 Q 参数加载。
- 保存输入统计计数与历史编码相关状态，不能只保存权重和偏置。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：解释检查点版本、原子保存、全部训练状态、经验池缺失的恢复限制，以及 PPO V 不能直接加载为 Q。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Gemini 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Gemini 编写。已有讲解受影响时标为待 Gemini 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 GEMINI.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Gemini。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（尚未执行）

- 保存前后同输入的确定性动作一致，Q 和 alpha 一致。
- 恢复随机状态和经验池后，可复现下一批抽样；仅模型恢复模式有明确提示。
- 维数、动作边界或文件版本不匹配时拒绝静默加载。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

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

## 实现证据与 Gemini 交接

| 项目 | 当前内容 |
|---|---|
| AI 规范 | 已编写；实现时同步实际接口 |
| 实现状态 | 待 Claude 实现 |
| 实际文件、函数签名与调用关系 | 尚无实现；上方仅为计划 |
| 参数/返回值、形状、单位、异常和副作用 | 按输入输出约定实施后补充真实接口记录 |
| 代码注释核对 | 待代码存在后检查本模块注释重点 |
| 验证命令、环境与结果 | 未执行；尚无验证程序 |
| 未验证项 | 仿真、训练与性能均未验证 |
| 代码版本 | 尚无实现提交；不得填入虚构提交号 |
| 本次改动及理由 | 仅建立 AI 规范与人类版分工，未改算法实现 |
| Gemini 所需材料 | 根/大模块/本模块 AI 规范、依赖契约、实际带注释代码、验证文件与结果、上述代码版本和变更摘要 |
| 人类版 | 等待代码完成；当前不编写正文 |

Claude 每次改动都需更新此表，列出应由 Gemini 重点解释的实际文件、函数和计算片段。对应代码未完成时，不启动 Gemini 写作并保留占位；代码和注释完成、验证记录齐备后，按 [GEMINI.md](../../GEMINI.md) 编写或同步 [README_HUMAN.md](README_HUMAN.md)。
