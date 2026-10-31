# E05 · 终止、截断与重置前观测

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

区分摔倒与人为时间结束，并记录动作真正到达的最后状态。

术语先说明：终止是本轮任务真正结束；截断是外部时间限制停止记录；重置是把环境恢复到新一轮初始状态。

## 与原 PPO 的关系

原 PPO 在采集时对超时奖励加价值补偿；SAC 改为保存真实奖励与重置前观测，训练时再估计后续价值。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py)
- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/np3o.py)
- [原 PPO：source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/algorithms/sac.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 物理步后的环境状态、真正终止标记、时间截断标记、观测管理器的历史状态。 |
| 输出 | 用于继续交互的重置后观测，以及用于经验池的正确末状态、终止/纯截断标记。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `lifecycle.py` | `classify_endings` | 真正终止优先，得到可继续估值的纯截断标记 |
| `lifecycle.py` | `capture_final_observations` | 在重置前捕获策略历史与 Q 观测 |
| `lifecycle.py` | `build_step_record` | 分开交互下一状态与经验记录下一状态 |
| `test_lifecycle.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 同一步摔倒又到时，按真正终止处理。
- 只在正确时间推进一次历史；不得因额外 compute 调用导致历史重复当前帧。
- 快照不能被重置时的原地修改覆盖；有截断却缺最后观测时必须报错。

## 计算含义与实现提示

一次正常流程是：执行动作 → 计算奖励和结束原因 → 保存本步真实末状态 → 对结束的环境重置 → 返回下一次交互所需观测。末状态必须同时包含 Actor 的历史和 Q 的训练侧信息。

Isaac Lab 的自动重置可能发生在标准 `step` 返回之前，因此只在外层读取返回观测未必能取到末状态。实现时检查锁定版本的执行顺序，在重置前保存数据，并避免额外观测计算重复推进历史或重采噪声。逐字段验证保存内容对应同一次物理步。

作者 `process_env_step` 中替换末状态的二维掩码不能直接用于 TITA 的 [B,10,33] 历史；掩码需按字段维度正确展开。先用明显不同的末状态与重置状态检查，再接仿真。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 正常步、单独摔倒、单独到时、两种结束同时发生，分别检查是否接后续价值。
- 强制一组环境到时，重置前后状态明显不同，经验池必须存重置前版本。
- 其他未结束环境的观测和历史不能被这次重置影响。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

每到时间上限价值突变时，检查结束原因、最后观测来源和是否重复添加价值补偿。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [E01 机器人与任务环境](../../environment_control/robot_environment/README.md)
- [E02 观测接口](../../environment_control/observation_interface/README.md)
- [D01 经历记录契约](../../experience_returns/transition_schema/README.md)
- [D05 历史窗口与有效性](../../experience_returns/history_windows/README.md)

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
