# T05 · 模型导出与运行一致性

状态：待实现。当前只有说明；下面的文件名和函数名是设计约定，尚无实现或测试程序。未进行算法训练、仿真或性能验证。

[返回大模块](../README.md) · [返回总说明](../../README.md)

## 先让人看懂

让导出的 Actor 在相同观测与历史下，产生与训练内推理一致的动作。

术语先说明：导出：把训练框架中的模型转换成便于运行的文件。推理：使用固定参数算动作，不进行训练更新。

## 与原 PPO 的关系

保留历史编码的运行链，导出 SAC 的确定性动作及其压缩缩放；Q 与训练用 BT 转换模块不进入动作执行链。

- [原 PPO：source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py)
- [原 PPO：scripts/np3o/play.py](https://github.com/DDTRobot/DDT_Lab/blob/701a49c6e78dfb93a8b30145f0e7723b22921c0b/scripts/np3o/play.py)

原版快照可在 [baseline/ppo](https://github.com/kissmvp8-afk/study/tree/baseline/ppo) 查看。论文是算法依据；本模块对 TITA 的适配是待实现、待验证的设计。

## 作者实现参考

- [作者 SAC：`rsl_rl/models/sac_mlp_model.py`](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/models/sac_mlp_model.py)

这是固定版本的参考来源；只迁移本模块需要的行为，TITA 的动作类型和历史维数以本仓库契约为准。

## 输入与输出契约

| 项目 | 约定 |
|---|---|
| 输入 | 固定模式的 Actor/编码器、输入统计、历史契约、动作映射和目标运行格式。 |
| 输出 | 导出模型、输入输出说明、同输入对比报告和 50 Hz 控制周期约定。 |

## 计划文件与逐函数职责

实际实现和验证代码放在本目录，与本 README 并列。未来允许本目录增加必要的辅助文件，不能把实现散落到不相关的小模块。

| 计划文件 | 计划函数/方法 | 负责的功能 |
|---|---|---|
| `export.py` | `build_inference_pipeline` | 按 D05 组装唯一的历史输入路径 |
| `export.py` | `export_policy` | 导出包含归一化与动作变换的推理模型 |
| `export.py` | `compare_outputs` | 核对训练内与导出路径的动作差异 |
| `test_export.py` | 验证案例对应的测试 | 检查本模块规则，不重复实现主算法来证明自己 |

这些名字目前不可作为已有 API 导入。Claude 实现后必须用真实文件、函数签名及相对链接替换本表的计划信息。

## 必须成立的规则

- 历史输入已含当前帧时不能再次追加；原 _InferenceWrapper 存在这一静态不一致，需要明确处理。
- 比较同样的确定性动作，不拿随机采样结果作逐元素等值比较。
- 仿真导出通过不等于真机验证；物理单位、关节顺序和运行周期仍需确认。

## Claude 实施顺序

1. 先阅读根 README、根 CLAUDE.md、本大模块 README、本文件和下面依赖的契约。
2. 对照固定原 PPO 来源，确认输入、输出和改动范围。依赖可用明确的假数据验证，但不得把假数据结果称为仿真验证。
3. 按上表职责逐个实现函数，保持一次修改对应一个可解释行为；必要跨模块改动要同步双方契约并说明影响。
4. 在本目录实现下列验证案例，记录实际命令、依赖条件、输出结果及失败案例。
5. 更新 README：用人能读懂的语言解释每个真实文件与函数，关键公式或分支逐行说明“输入是什么、这一行做什么、为什么需要、删错会怎样”。
6. 记录本次修改和证据后再更新状态。不能仅因代码能导入就标记整个模块已完成。

## 验收案例（尚未执行）

- 带帧编号的历史经过两条入口切片一致。
- 相同观测、统计、参数在约定误差内输出相同 8 维动作。
- 边界附近动作、重置后的历史和模型重载分别检查。

运行命令：待实际文件与项目依赖配置建立后，由 Claude 写入可复制执行的命令。当前不能宣称本目录存在可运行测试。

## 故障定位

导出后异常时先定位输入与变换差异，不先改奖励重训。

遇到错误时记录输入摘要、预期与实际输出、相关配置和复现步骤。修复后把该案例加入本目录的验证代码。

## 依赖与跨模块边界

- [N03 SAC 动作分布](../../networks_learning/actor_distribution/README.md)
- [D04 观测归一化与统计管理](../../experience_returns/normalization/README.md)
- [D05 历史窗口与有效性](../../experience_returns/history_windows/README.md)
- [E03 八维动作定义表](../../environment_control/action_contract/README.md)
- [E04 动作边界与初始中心](../../environment_control/action_bounds/README.md)

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
