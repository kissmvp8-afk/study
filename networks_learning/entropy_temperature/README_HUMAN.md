# N07 · 探索力度 alpha｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对温度梯度、调整方向、动作坐标目标熵与单位区间换算；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `b186639` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[temperature.py](temperature.py)、[test_temperature.py](test_temperature.py) 全文；N03 密度实现与测试、N06 接口、N08 当前状态采样/温度/Actor 衔接测试；E04 半宽与 D04/N01 特征契约。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 4 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## alpha 调节的是损失权重，不是电机幅度

熵衡量动作分布的多样性。连续动作使用微分熵，可用当前策略样本的 `−log π` 均值估计；它依赖坐标，数值允许为负。N07 学习正数 alpha，控制 N05/N06 中熵项相对 Q 评分的权重，不直接设定关节角或轮速。

原 PPO 的熵系数是配置值；这里参考作者 SAC，通过目标熵自动更新温度。本模块只拥有一个可训练参数 `log_alpha`，不更新 Actor、Q 或编码器。

## 输入、输出与调用顺序

`log_prob[B,1]` 来自 N03 在回放当前状态上的**同一次采样**，是在策略动作 a 坐标下、包含 tanh 与半宽修正的联合对数密度。`target` 是同坐标下的目标熵 float；温度没有批维参数，整个批次共用一个 alpha。

一次 Actor 更新的顺序是：采样 action/log_prob → `update_temperature` → `current_alpha` 取新数值 → N06 用新 alpha 和原 action/log_prob 更新 Actor。N07 对 log_prob 执行 detach，所以这一步只把它当数据，不消费 Actor 的采样计算图，也不需要重新采样。

更新返回 `alpha_loss`、**更新后的** `alpha`、`entropy_estimate` 和 `entropy_gap=估计熵−目标熵`。后两项来自这次传入的样本，不是温度更新后重新抽出的分布评估。

## 文件、数据类与全部函数

[temperature.py](temperature.py) 负责参数化、坐标换算与更新，[test_temperature.py](test_temperature.py) 检查方向、隔离和尺度变化。

| 接口 | 输入输出与副作用 |
|---|---|
| `TargetEntropy` | 冻结数据类，保存动作坐标目标 `action_coordinate`、单位区间等价目标 `squashed_equivalent`、`log_half_width_sum=Σlog(c)`，无自定义计算方法 |
| `target_entropy(half_width,target_entropy_scale=1.0)` | 读取 E04 正半宽，按维数返回上述三项；只计算，不改变网络/边界 |
| `Temperature.__init__(initial_alpha=0.05)` | 要求初值为正，创建标量可训练参数 `log_alpha=log(initial_alpha)`；本类没有 forward |
| `current_alpha(temperature)` | 先 detach，再 exp、item，返回不带梯度的 float，供 N05/N06 使用 |
| `make_temperature_optimizer(temperature,lr=0.001)` | 创建只含 log_alpha 的 Adam |
| `temperature_loss(temperature,log_prob,target)` | 返回只连接 log_alpha 的标量损失，Actor 的 log_prob 在内部断开梯度 |
| `update_temperature(temperature,optimizer,log_prob,target)` | 清梯度、反传、step 后报告；只修改 log_alpha 及其优化器状态 |

`INITIAL_ALPHA=0.05`、`TARGET_ENTROPY_SCALE=1.0`、`TEMPERATURE_LEARNING_RATE=0.001` 参考作者默认值，仍是实验起点。数学上 `exp(log_alpha)>0`；浮点表示仍有范围限制，有限次测试不能证明无限训练期间始终有限正值。

## 目标熵 −8 属于哪个坐标

E04 的动作映射是 `a=b+c*y`，其中 `y=tanh(u)` 在单位区间 (−1,1)。若半宽为 c，则：

$$\log\pi_a=\log\pi_y-\sum_i\log c_i,\qquad H_a=H_y+\sum_i\log c_i.$$

本项目保留作者配置：**动作 a 坐标下目标为 −1×8=−8**。既然 N03 的 log π 已含 `−Σlog(c)`，温度损失必须使用 `TargetEntropy.action_coordinate`；同一目标换到单位区间后才是：

$$H_y^{target}=-8-\sum_i\log c_i.$$

真实计算：

```python
log_sum = sum(math.log(c) for c in half_width)
action_target = -target_entropy_scale * len(half_width)
return TargetEntropy(action_target, action_target - log_sum, log_sum)
```

第一行累计八维缩放的对数；第二行保留动作坐标的 −8 约定；第三行同时记录等价值。不能把“保留 −8”误写成“所有半宽下单位区间的探索要求都相同”。

现有测试把八维半宽全部从 1 改为 2，网络权重与噪声保持一致。新 log π 比旧值少 `8log2≈5.545`；动作坐标目标仍为 −8，而单位区间等价目标从 −8 变为约 −13.545。同样的内部高斯分布在 a 坐标下熵更大，熵差因而改变。如果以后希望固定单位区间目标，须反过来加 `Σlog(c)` 换算到动作坐标；当前实现没有采用这一替代约定。

## 损失符号为什么决定正确方向

实际损失：

```python
return -(temperature.log_alpha * (log_prob.detach() + target)).mean()
```

detach 只断开 Actor 路径，log_alpha 仍可求导。若估计熵 H=−平均 log π，括号均值就是 `target−H`；损失对 log_alpha 的导数为 `H−target`。

测试中 H=−20、目标−8，熵差为−12，梯度下降提高 log_alpha，从而提高 alpha，增强后续 Actor 目标中保留多样性的权重。H=30 时熵差为38，更新方向相反。alpha 更新并没有立刻改变 Actor 分布，Actor 还需在下一步使用新权重优化。

N05 构造冻结 Q 目标时读取当时的 alpha；在 N08 当前更新顺序中，该目标先于这次温度更新。N06 的 Actor 损失则明确读取温度更新后的新值，两处使用时间不同。

## 既有检查与排错

`artificial_bounds(half_width)` 生成八维对称人工边界，供 N03 尺度测试；它不是真实机器人工作范围。

| 既有测试 | 验证内容 |
|---|---|
| `DirectionTest.test_alpha_direction` | 人工 log π=20/−30 时，熵低/高于−8，alpha 分别增/减，日志熵差符号一致 |
| `DirectionTest.test_alpha_positive_finite_and_restorable` | 给定输入与 lr=0.5，连续 200 次更新后 alpha 正且有限；state_dict 恢复后值与损失相同，不代表完整训练检查点已接入 |
| `BoundaryTest.test_no_gradient_to_actor` | 真实 N03 采样的 log π 经温度损失反传，只有 log_alpha 得到梯度 |
| `BoundaryTest.test_action_scale_shifts_density_and_squashed_target` | 半宽加倍的 log π、两坐标目标及熵差变化符合公式 |

入口为 `unittest.main()`。既有复现命令，本次未执行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/entropy_temperature/test_temperature.py -v
```

alpha 异常先核对 log π 完整修正、所用坐标、`action_coordinate` 与熵差，再看学习率；不要因熵为负就判错。E04 真实边界尚未确定，−8 是否适合实际任务、固定 alpha 配置、GPU 与训练表现均未验证。与 [N03](../actor_distribution/README_HUMAN.md)、[N06](../actor_learning/README_HUMAN.md) 一起阅读，更新衔接见 [N08](../target_updates/README_HUMAN.md)。


## 设备内温度标量迁移（2026-11-08）
参考官方实现：https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py
仅将目标值与Actor损失计算使用的alpha保留为设备上的无梯度标量张量。
原float日志接口、更新顺序、目标熵、SAC损失、HIM编码器及Barlow辅助目标保持不变。
尚未证明性能或控制质量提升。
外置诊断工具统一部署到 /root/rivermind-data/tita-check/sac-alpha-migration-tools/。
待执行命令（此处不代表已通过）：
- python /root/rivermind-data/tita-check/sac-alpha-migration-tools/test_alpha_tensor.py --study-root STUDY --device cpu
- python /root/rivermind-data/tita-check/sac-alpha-migration-tools/test_alpha_tensor.py --study-root STUDY --device cuda:0
- python training_validation/verification/verify.py --docs
- python /root/rivermind-data/tita-check/sac-alpha-migration-tools/run_alpha_two_iterations.py --help
试跑完整参数由部署者记录到运行目录；128环境，warmup=0仅作兼容诊断，两轮各16次critic与16次辅助更新尝试。
实际命令、结果和错误须在执行后补记，不可将文档结构检查当成学习测试通过。


### 本次实际验证结果
CPU/CUDA等价测试通过；N07原4项、T02原9项回归通过；文档检查通过。128环境两轮真实更新完成。进一步同seed单次64轮配对最终权重完全相同，600步冻结评价日志完全相同，均未过速度跟踪门槛；不能声称控制改善。学习阶段单次观测少11.76%，存在顺序/缓存/温度混杂，不构成稳定提速结论。完整证据位于服务器 sac-alpha-pilot-20261108/export-001；每轮日志已保存。
