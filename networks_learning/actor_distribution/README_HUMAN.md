# N03 · SAC 动作分布｜人类阅读版

[AI 实施规范](README.md) · [返回大模块](../README_HUMAN.md) · [项目维护规则](../../README.md#项目维护规则)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成 |
| 技术核对状态 | 2026-10-31，Codex 实际阅读实现与测试，核对重参数化、边界映射、完整密度修正、初始化和确定性动作；本轮为文档写作，不重复全面代码审核 |
| 语言润色状态 | 本轮不再另交 Gemini |
| 已核对代码版本 | PR #6 分支 `claude/networks-learning` 的 `42c07a025f16fe41251c3e306a18313ce1704f02`，与用户已复核版本一致；实现来源 `a7a469e` |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根 AGENTS.md、CODEX.md、README.md，大模块及本目录 AI README、原人类版占位；[actor.py](actor.py)、[test_actor.py](test_actor.py) 全文；N01 全文；E03 动作换算、E04 `bounds.py` 与 `test_bounds.py` 的人工范围/校准案例；D04 接口与实现片段。全局来源与阅读边界见[大模块阅读记录](../README_HUMAN.md#阅读材料与验证边界) |
| 验证依据与限制 | 本模块 7 项 CPU 检查的既有记录见 [AI 交接表](README.md#实现证据与文档交接)及 [PR #6](https://github.com/kissmvp8-afk/study/pull/6)；用户报告先前 43 项通过，修正后 N02/N06/N08 共 17 项再次通过。本次只做文档静态检查，不重新运行算法测试；人工数据，未验证仿真、GPU 或真实训练效果 |

## 学习目标：输出动作，也输出动作的密度

SAC（Soft Actor-Critic，软演员—评论家算法）在改进动作评分的同时保留一定动作多样性。N03 因此既要给出八维动作，也要计算该动作的**概率密度**；连续变量的密度不是“精确取到某个点的概率”，它依赖坐标和尺度，甚至可以大于 1。后续损失使用其自然对数 `log_prob`，即 log π。

原 PPO 使用无界高斯动作与不随状态变化的标准差；本模块让均值、标准差随 52 维状态特征变化，再通过 tanh 得到有界动作。tanh 是把实数平滑压入 (−1,1) 的函数，随后用 E04 边界映射到各维策略动作区间。

## 三种坐标不要混用

| 数据 | 形状 | 含义 |
|---|---|---|
| `features` | `[B,52]` | N01 的隔离梯度特征，不包含真实速度标签 |
| `mean`、`log_std`、噪声 ε | `[B,8]` | 内部高斯坐标 u 的均值、标准差对数、标准正态噪声 |
| `center=b`、`half_width=c` | `[8]` | E04 给出的策略动作区间中心、半宽，按批广播 |
| `action=a` | `[B,8]` | `b+c*tanh(u)`，无物理单位，与 D01 保存和 Q 评价的动作同坐标 |
| `log_prob` | `[B,1]` | a 坐标下八维联合对数密度 |

最后由 E03 把腿部策略动作换成 `q*=q0+0.25a`（rad）、轮速换成 `w*=5a`（rad/s），再做物理目标裁剪。N03 不执行这层物理换算，也不把动作改成力矩。八维顺序仍是 E03 的顺序，轮速在索引 3、7，其他六维是位置。

## 文件、类与函数

[actor.py](actor.py) 定义分布与网络，[test_actor.py](test_actor.py) 比较边界、密度、初始化和梯度。

| 接口 | 作用与副作用 |
|---|---|
| `log_one_minus_tanh_squared(u)` | 返回稳定的 log(1−tanh²u)，形状与 u 相同，保留梯度 |
| `squash_and_log_prob(mean,log_std,noise,center,half_width)` | 给定同一组噪声，返回动作及完整 log π，不抽取新随机数 |
| `SquashedGaussianActor.__init__(bounds,initial_center,feature_dim=52,hidden_dims=(512,256,128),initial_std=0.15)` | 创建 ELU 主干与两个八维输出头；`initial_center` 接收 E04 的内部 u0，不是区间中心 b；b、c 用 `register_buffer` 保存，随模型迁移/存档，不在优化器参数中 |
| `SquashedGaussianActor.distribution_parameters(features)` | 主干前向后返回 mean、已裁剪的 log_std，各 `[B,8]` |
| `SquashedGaussianActor.sample_action_and_log_prob(features,noise=None,generator=None)` | 有显式 noise 则复用，否则抽标准正态噪声；动作与密度来自同一次采样，保留 Actor 梯度；抽样会推进随机数状态 |
| `SquashedGaussianActor.deterministic_action(features)` | 令 u=mean，输出同一套 tanh/边界映射后的动作；不搜索 Q，也不创建额外随机样本 |

`ACTOR_HIDDEN_DIMS=(512,256,128)` 沿用原 PPO；`LOG_STD_MIN=-20`、`LOG_STD_MAX=2` 和 `MEAN_HEAD_INIT_STD=0.001` 参考作者 SAC；`INITIAL_STD=0.15` 是本项目规范起点。`_HALF_LOG_2PI` 是高斯密度的常数项。作者差异以本地来源注释/AI 记录为依据，本次未独立拉取外部 SAC 源码。

## 重参数化与三项密度计算

重参数化是把随机性显式写成 ε，再使输出对网络参数可导。固定同一个 ε 时，均值或标准差改变仍能连续地改变动作：

```python
u = mean + log_std.exp() * noise
action = center + half_width * torch.tanh(u)
gaussian = -0.5 * noise.pow(2) - log_std - _HALF_LOG_2PI
log_prob = (gaussian - log_one_minus_tanh_squared(u) - torch.log(half_width)).sum(dim=-1, keepdim=True)
```

第一行将 log_std 指数化为正标准差，再生成内部高斯样本。第二行压缩并缩放，得到 E04 范围内的策略动作。第三行用已知 ε 写高斯对数密度，避免再用 `(u−mean)/std` 恢复噪声。

最后一行应用变量变换：

$$\log\pi(a)=\sum_i\log\mathcal N(u_i;\mu_i,\sigma_i)-\sum_i\log(1-\tanh^2u_i)-\sum_i\log c_i.$$

减去 tanh 导数的对数，补偿压缩带来的密度变化；再减去 log c，补偿区间拉伸。对八个动作维度求和并保留 `[B,1]`，才能与 Q 标量相配。漏掉任一项都会改变 N05/N06 的熵项和 N07 的温度调节；log π 也不能改在物理目标坐标计算后仍沿用原目标熵。

直接计算 `log(1-tanh(u)**2)` 在大 |u| 时会因浮点饱和得到 log(0)。实际函数使用：

```python
return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
```

softplus(x)=log(1+exp(x))，这里是同一数学量的稳定恒等式，不是加 1e-6 的近似。测试覆盖 u=±60 和给定极端特征，不能据此宣称任意无限大输入都安全。数学上有限 u 对应开区间；浮点 tanh 可能舍入到端点，现有边界测试使用闭区间包含判断。

## 为什么同时校准中心和随机尺度

E04 的单维人工例子：默认角 0.8 rad、scale=0.25，范围 `[0.5,1.3] rad` 反推 L=−1.2、U=2、b=0.4、c=1.6。若内部均值初始化为 0，确定性动作是 0.4，物理目标为 0.9 rad；E04 校准 `u0=atanh((0−0.4)/1.6)=atanh(−0.25)`，就能使确定性动作回到 0、目标回到 0.8 rad。**这些是人工数据，不是真实 TITA 限位。**

构造函数把 mean_head 偏置设为 u0，权重设为标准差 0.001 的小随机数，所以网络初始确定性输出接近期望，而不是对所有特征严格相等。log_std_head 权重为零、偏置为 log(0.15)，故初始内部标准差为 0.15；它不是“每个关节角随机 0.15 rad”，还要经过 tanh、c 与 E03 scale。

确定性输出 `b+c*tanh(mean)` 等于零噪声采样，但非线性变换后的随机动作期望通常不是它。E04 校准的是确定性中心，不保证非对称范围下随机动作平均值恰好为默认动作。

## 既有测试与排错

测试的 `make_actor(seed,initial_center)` 固定初始化，`features(batch,seed)` 生成随机 52 维输入。`CONTRACT` 来自 E03 固定 PPO 快照，`BOUNDS`/`U0` 由 E04 **人工**腿部范围与工作轮速推导，未读取真实模型限位。

| 既有测试 | 验证范围 |
|---|---|
| `TransformTest.test_extreme_inputs_stay_in_bounds_with_finite_density` | 特征放大 1e3/1e6 后动作在界内且密度有限；稳定公式在 ±60 有限、中等输入与朴素式一致 |
| `TransformTest.test_density_matches_independent_formula` | 固定噪声，与 float64 的 Normal.log_prob−tanh 修正−log c 独立公式比较 |
| `TransformTest.test_deterministic_uses_same_transform` | 确定性动作等于零噪声采样 |
| `TransformTest.test_log_std_is_clamped` | 过大/过小输出限制到 2/−20 |
| `InitializationTest.test_initial_std` | 内部初始 std=0.15 |
| `InitializationTest.test_initial_deterministic_target_near_default` | 校准后策略动作偏差 <0.05，E03 物理目标偏差 <0.05×scale；零偏置对照落在非对称区间中点附近 |
| `GradientTest.test_reparameterized_gradients` | 梯度到均值头、标准差头、主干；边界是缓冲，不是可训练参数 |

入口为 `unittest.main()`。AI 交接还记录运行前修正了一条恒真且含 log(0) 的断言，本次没有重新运行。既有命令：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/actor_distribution/test_actor.py -v
```

贴边动作多先查 u0、std、b/c 和 E03 scale；alpha 异常先核对三项密度及坐标。真实边界、部署导出、GPU 和训练探索效果均未验证。继续阅读 [N06](../actor_learning/README_HUMAN.md)、[N07](../entropy_temperature/README_HUMAN.md)，边界来源见 [E04](../../environment_control/action_bounds/README_HUMAN.md)。
