# E01 · 机器人与任务环境｜AI 实施规范

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[人类版（等待代码完成）](README_HUMAN.md) · [返回大模块](../README.md) · [返回总说明](../../README.md)

## 实施目标与术语

加载 TITA 并提供相同的平地/复杂地形任务，使 PPO 与 SAC 控制的是同一个对象。

术语先说明：执行器：将角度或速度目标转成力矩的控制模型。URDF：描述机器人部件、关节和几何形状的文件。

## 与原 PPO 的关系

沿用原模型、指令、奖励与执行器设置；新增 SAC 环境注册，训练方法改变不应暗中改变任务。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/flat_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/flat_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/assets/ddt_robot.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/assets/ddt_robot.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 版本锁定信息、机器人描述文件路径、任务类型、并行环境数。 |
| 输出 | 可被训练器调用的仿真环境，以及实际关节名称、默认姿态、限位、控制周期元数据。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本目录两份 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `environment.py` | `build_environment` | 按配置创建平地或复杂地形环境 |
| `environment.py` | `read_robot_metadata` | 读取真实关节名称、限位和执行器信息 |
| `environment.py` | `describe_task` | 输出任务、奖励、周期与版本摘要 |
| `test_environment.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 策略周期为 0.02 秒、物理步长为 0.005 秒，第一版沿用原配置。
- 机器人资源来自外部 ddt_ros2_control；运行前固定资源版本，不假装基线分支已包含 URDF。
- 不把原仓库未使用的 Cartpole 模板当成 TITA 配置；SAC 环境注册不覆盖 PPO 入口。

## 代码注释要求

执行[根目录代码注释规范](../../README.md#代码注释规范)。为每个实现文件、类、函数和方法写准确的中文说明；在关键计算或条件分支旁解释原因、前提和错误后果。

本模块特别需要注释：标注控制周期与物理步长的关系、资产版本来源、环境注册及奖励配置沿用的范围。

自检时把这些注释逐项对照实际代码和下方验收案例；修改行为时同次修改注释。Gemini 依据带注释的代码撰写人类版，代码中的解释责任仍属于 Claude。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，同时写 docstring 和关键代码注释；必要跨模块改动同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新本 AI 版的真实文件链接、函数签名、参数/返回值、单位、异常和副作用，并填写下方交接表。首次实现未完成时，人类版保持等待代码完成；完成后标为待 Gemini 编写。已有讲解受影响时标为待 Gemini 同步，保留正文。
6. 记录证据后更新实现状态；对应代码和注释完成后，按根 GEMINI.md 把同一版本的实际代码、规范、验证记录和变更说明交给 Gemini。人类版同步和代码验证状态分别记录，不把文档编写完成当成验证通过。

## 验收案例（尚未执行）

- 加载 TITA 后读到 8 个受控关节；缺失资产应报告具体路径。
- 平地任务无地形扫描输入；复杂地形的扫描信息供 Q 使用。
- 同一配置的 PPO/SAC 任务摘要中，奖励、指令和物理周期一致。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

加载失败先查资源路径和依赖版本；运动差异先比较任务摘要，再分析算法。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [T01 版本、依赖与训练配置](../../training_validation/configuration/README.md)

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
