# E02 · 观测接口

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

明确 Actor 与 Q 分别看到什么，让每一个观测数字的顺序、尺度和来源可查。

术语先说明：观测接口：模块之间对数据内容、排列和单位的约定。训练侧额外信息：仿真可提供、运行中的 Actor 未必能直接获得的信息。

## 与原 PPO 的关系

保留原策略侧 33 维观测及历史；Q 继承原 Critic 可见信息，但评价时还需要当前待评价动作。

- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 环境输出的策略观测、Critic/priv/scanner 观测组、实际字段定义。 |
| 输出 | 编码前的策略历史 [B,10,33]、Q 的状态信息 [B,C]、当前真实速度监督标签与字段说明。B 是并行环境数，C 由运行时观测组确定。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `observations.py` | `read_observation_groups` | 提取策略侧与训练侧观测 |
| `observations.py` | `describe_observation_fields` | 记录每项的顺序、单位和尺度 |
| `observations.py` | `validate_observations` | 检查维数、数值和字段来源 |
| `test_observations.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- Actor 不直接接收仿真真实身体线速度；该信息可供 Q 和速度监督使用。
- 平地 C 预期为 54，复杂地形预期为 241，但必须从实际配置校验，禁止静默截断或补齐。
- 速度监督标签沿用原 base_lin_vel 的 2.0 缩放；不能把缩放后的预测直接当成米/秒。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 调换两个同维字段也应被字段契约检测到，而不是只检查总长度。
- 策略输入中没有真实线速度；训练侧监督标签来自明确字段。
- 平地和复杂地形分别验证观测组及历史长度。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

维数正确仍表现异常时，查字段顺序、缩放和坐标系；不要只重设网络输入长度。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E01 机器人与任务环境](../../environment_control/robot_environment/README.md)

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
