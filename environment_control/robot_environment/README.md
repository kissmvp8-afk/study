# E01 · 机器人与任务环境

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

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

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

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

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

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

## 当前实现与验证记录

| 内容 | 当前状态 |
|---|---|
| README 设计与职责 | 已编写，等待实施时与实际代码保持同步 |
| 实现文件和真实函数说明 | 待 Claude 实现 |
| 关键代码逐行说明 | 待代码存在后填写；不得编造行号 |
| 单元验证命令与结果 | 待实现 |
| 仿真集成结果（如适用） | 未验证 |
| 修改记录 | 本次仅建立模块契约与验收要求 |
