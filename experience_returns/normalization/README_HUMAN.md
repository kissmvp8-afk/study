# D04 · 观测归一化与统计管理｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 技术初稿完成，待 Gemini 润色 |
| 技术核对状态 | 已核对实现、8 项测试、D01 字段与 D02/D03 回放衔接，确认公式、计数与状态复制；未开展全面代码审核 |
| 语言润色状态 | 待 Gemini 润色，尚未润色 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现/测试来自 `e77bc66d7878831fb1b5eb19d616785c81f2d251`，读取时无后续源码变化 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 本目录 AI README、[normalization.py](normalization.py)、[test_normalization.py](test_normalization.py) 全文；D01～D03 实现与测试、D05 衔接测试；固定 PPO `normalizer.py` 全文及 `actor_critic.py` 的归一化路径。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 55 项 CPU 检查独立通过；本模块 8 项记录见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 与 [AI 交接表](README.md#实现证据与文档交接)。本次未重跑；人工数据，未接网络、仿真或训练 |

## 归一化为什么要区分“更新”和“使用”

观测各通道的数值尺度可能差很多。归一化用均值和标准差重新表达输入，让网络不必只因某个通道的数字较大就过度关注它。均值是样本平均，方差是偏离均值的平方平均，标准差是方差的平方根。

D04 使用：

```text
y = (x - mean) / (sqrt(var) + eps)，eps 默认 0.01
```

统计更新会改变 mean、var、count；使用统计做变换则不应改变它们。SAC 反复抽取旧经历，若每次回放都重新计数，抽得多的旧状态会被重复统计，经验池的抽样行为就悄悄改变了输入尺度。

因此当前接口明确分工：新采集的每批数据只 `update_from_fresh_data` 一次；回放、评分、推理只 `normalize`。这是调用方责任，不是函数内维护了“已经统计过哪些记录”的集合。把同一批再次传给 update，仍会再次计数。

## 两套统计各管哪些字段

| 统计对象 | 维数、输入与用途 |
|---|---|
| `ObservationNormalizers.policy` | D=33，逐帧统计；同一组参数作用于 `[B,10,33]` 的每一帧，也能处理 `[B,33]` 或 `[B,5,33]` |
| `ObservationNormalizers.q` | D=C；当前 `first.q_observation` 与目标尾部 `tail.next_q_observation` 共用同一个 `normalize_q` 和同一份当前统计 |

“online Q”是正在学习的 Q，“target Q”是缓慢跟随的目标网络；网络参数不同，不意味着输入应使用两套不同统计。两者用相同变换才能对同一坐标下的状态估值。

D01/D02 继续保存原始历史和 Q 观测，D04 返回新张量，不覆盖记录、不归一化奖励或动作。N02 的速度监督标签应读取**未经过 D04 归一化**的 Q 原字段，并保留 E02 约定的 2.0 缩放；它不等于已经换回米/秒。

## `normalization.py`：逐类、逐函数职责

| 符号 | 输入、输出与状态变化 |
|---|---|
| `NormalizationError` | 更新、归一化或恢复不符合当前约定时的异常，无自定义方法 |
| `NORMALIZER_EPS`、`STAT_DTYPE` | 分别为 `1e-2`、`torch.float64`；eps 加在标准差外，统计用双精度存储 |
| `RunningNormalizer.__init__(dim, eps=0.01, device='cpu')` | 要求维数和 eps 为正；创建 `mean[dim]=0`、`var[dim]=1`、`count=0`，mean/var 为 float64。尚未更新时输出约为 x/1.01，不是严格恒等变换 |
| `RunningNormalizer.update_from_fresh_data(x)` | 输入 `[N,D]`，检查形状与有限值；N=0 不改变状态。转换到统计设备/float64，合并总体均值、方差与计数；会修改状态 |
| `RunningNormalizer.normalize(x)` | 要求最后一维为 D，其他轴保留；只读统计，返回新张量，dtype 与 x 相同。数据路径预期使用 D01 的浮点观测；此方法不重复完整记录校验 |
| `RunningNormalizer.export_statistics()` | 返回含 `mean,var,count,eps,dim` 的字典；mean/var 是副本，可供后续检查点保存，不直接写文件 |
| `RunningNormalizer.load_statistics(state)` | 从该状态恢复，检查维数、非负 count 及有限非负 var；转到统计设备/float64 并复制，恢复 count 和 eps。此接口并非任意不可信文件的完整校验器 |
| `select_fresh_samples(batch)` | 只读一批新 D01 记录，返回策略帧 `[N_p,33]`、Q 行 `[N_q,C]`，具体选择规则见下；不按观测数值判断来源 |
| `ObservationNormalizers.__init__(q_dim, eps, device)` | 分别构建 33 维 policy 与 C 维 q 的 `RunningNormalizer` |
| `ObservationNormalizers.update_from_fresh_data(batch)` | 调用 `select_fresh_samples`，再更新两套统计；每批新记录只调用一次 |
| `ObservationNormalizers.normalize_policy(history)` | 委托 policy.normalize，支持最后一维为 33 的观测，不增加 count |
| `ObservationNormalizers.normalize_q(q_observation)` | 委托 q.normalize，供当前 Q 和目标 Q 共用，不增加 count |
| `ObservationNormalizers.export_statistics()` | 返回 `{'policy': 完整状态, 'q': 完整状态}`，两套统计都要保存 |
| `ObservationNormalizers.load_statistics(state)` | 分别恢复 policy、q 状态；修改统计，不恢复网络参数或经验池 |

所有实现都在本文件，依赖 torch 与 D01，通过 `sys.path` 导入相邻模块；N01/N04/T03 的生产接入仍未完成。

## 统计怎样按批合并

假设已经统计 n_a 个样本，新批有 n_b 个，两个均值之差为 delta。只平均两份方差会漏掉“两个群体均值不同”带来的波动，因此必须包含组间差项。

实际关键计算为：

```python
mean_b = x.mean(dim=0)
var_b = x.var(dim=0, unbiased=False)
```

每个通道独立求均值和**总体方差**，分母是 N，不是样本方差的 N-1。之后令 n=n_a+n_b、delta=mean_b-mean_a，旧统计不为空时：

```python
self.mean = self.mean + delta * (n_b / n)
self.var = (self.var * self.count + var_b * n_b + delta.square() * (self.count * n_b / n)) / n
```

均值按样本数量加权；方差分子中的前两项是两组各自的离均差平方和，第三项补上组间均值差。最后除总数量，避免把“一批 1 条”和“一批 3 条”当成等权两批。首次 n_a=0 时直接采用新批统计，初始 var=1 不作为虚构样本参与。

现有测试数据是：

```text
[[1,2], [3,4], [5,6], [7,10]]
mean = [4,5.5]，var = [5,8.75]，count = 4
```

一次输入四行，或先一行再三行，得到相同的手算统计。第一通道的平方偏差和为 9+1+1+9=20，总体方差 20/4=5。

输出计算为：

```python
return ((x.to(STAT_DTYPE) - mean) / (std + self.eps)).to(x.dtype)
```

先在 float64 中相减、除以标准差加 eps，再转回输入 dtype。这里是 `sqrt(var)+0.01`，不是 `sqrt(var+0.01)`。测试把第一通道输入设为 `4+sqrt(5)`，输出为 `sqrt(5)/(sqrt(5)+0.01)`，接近但不等于 1；第二通道输入均值 5.5，输出 0。常数通道若输入等于均值，也是 0；eps 防止除零，不是输出裁剪或任意量级下的数值安全保证。

## 新数据如何只计一次

选择样本的全部关键代码是：

```python
first = batch.history_length == 1
policy = torch.cat([batch.policy_history[first, -1], batch.next_policy_history[:, -1]], dim=0)
q = torch.cat([batch.q_observation[first], batch.next_q_observation], dim=0)
```

第一行用有效历史长度辨认一轮首条记录；第二行加上首条记录的当前最新帧，以及**每条**记录的真实下一最新帧；第三行对 Q 采用相同规则。它不把十帧历史展开成十倍样本，因为重叠旧帧在之前已经统计过；也不把填充计为真实状态。

现有人工例子让第 t 帧的各通道都等于 t，填充为 -999：

| 记录中的动作 | 当前最新帧 | 下一最新帧 | 本条新增统计 |
|---|---:|---:|---|
| 第 0 次 | 0 | 1 | 当前 0 和下一 1 |
| 第 1 次 | 1 | 2 | 只加下一 2 |
| … | … | … | … |
| 第 11 次 | 11 | 12 | 只加下一 12 |

12 次动作最终统计帧值 0..12 共 13 个样本，均值 6、总体方差 14。count 不是动作数 12，也不是完整历史位置数 120。测试还核对 Q 均值为 `[6,12,5,0]`。

另一项测试在第 0 轮三次动作后真终止，下一轮从帧值 100 开始执行两次：统计集合为 `0,1,2,3,100,101,102`，count=7。3 是重置前末状态，100 是新一轮首帧，两者各计一次，不是用后者替换前者。

这依赖 E05 正确提供 `next_*`、有效长度以及调用方只统计每批新数据一次。函数不会检测重复批次，也不能从值 -999 判断填充；标签错了，计数规则也会错。把回放数据传给 update 不会被自动拒绝，所以回放路径应只使用 normalize。

## 与 PPO、作者 SAC 的区别及恢复状态

实际读取的 PPO `EmpiricalNormalization.forward` 在 training 模式下自动 `update`；其 Actor 对当前帧和历史帧共用一个观测归一化器。D04 保留逐帧同统计和默认 eps，但把更新从变换中拆开，避免回放重复计数。

按本地文件头及 AI 规范的作者版本记录，作者 SAC 在环境步中用 `true_next_obs` 更新，真终止环境的该值是重置后观测、不包含末状态；这里显式把重置前末状态和下一轮初始观测分别计入。这是本项目的选择，本次未独立读取外部作者源码，不表示作者已验证过 TITA 上的效果。

检查点必须保存两套状态的均值、方差、**count、eps、dim**。只保存均值与方差会使恢复后的后续合并权重不对，漏掉 eps 也会改变相同输入的输出。导出和恢复都复制统计张量；测试用 count=40、eps=0.001 恢复后，同输入输出一致，修改导出字典的 mean 不影响原对象或恢复对象。T03 还未接通，导出字典不等于已经写好完整训练检查点。

## `test_normalization.py`：逐项验证

`history_at(t,offset)` 返回 `[10,33]` 人工历史，真实帧靠右、填充为 `PAD=-999`；`episode_batch(env,episode,steps,offset,terminated_last)` 经 D01 构造整轮人工转移，Q 取 `[t+offset,2(t+offset),5,0]`。其中局部函数 `q` 只是这个测试数据生成规则，不是 Q 网络。

| 测试类/方法 | 覆盖内容 |
|---|---|
| `RunningNormalizerTest.test_known_statistics_and_batch_merge` | 四行手算统计，一次更新与拆批一致 |
| `RunningNormalizerTest.test_normalize_does_not_change_statistics_or_input` | 连续五次 normalize，计数、统计和输入不变，输出相同 |
| `RunningNormalizerTest.test_export_and_restore` | 恢复 count/eps/统计，验证副本和维数不符的拒绝 |
| `RunningNormalizerTest.test_constant_tiny_variance_and_extreme_inputs` | 常数与极小方差人工通道、1e30 输入的有限输出，默认 eps=0.01；不代表任意输入无溢出 |
| `RunningNormalizerTest.test_rejects_bad_updates` | NaN 和形状错误的更新报错，count 仍为 0 |
| `FreshDataRuleTest.test_each_real_frame_counted_once` | 12 次动作只计 0..12，排除重复旧帧与填充 |
| `FreshDataRuleTest.test_terminal_and_reset_states_each_counted_once` | 末状态 3、新轮初始状态 100 各计一次 |
| `ReplayIntegrationTest.test_replayed_observations` | D02→D03 回放三次仅做 normalize，计数和原始记录保持不变 |

测试入口为 `unittest.main()`。八项 CPU 通过是已有证据，本次只静态核对；未验证真实观测尺度、GPU、网络 BatchNorm（网络内部另一种归一化层）交互、仿真或训练效果。复现已有检查可在根目录运行：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/normalization/test_normalization.py -v
```

同输入输出漂移时，先看是否误调用 update、统计是否已变、online/target 是否共用统计；计数过大先查重叠历史和回放重复计数。记录变动则查是否把归一化结果原地写回 D01/D02。

## 阅读导航与润色交接

[D01](../transition_schema/README_HUMAN.md) 规定原始字段；[D02](../replay_buffer/README_HUMAN.md)、[D03](../n_step_returns/README_HUMAN.md) 保留可回放输入；[D05](../history_windows/README_HUMAN.md) 切片可在同一份固定统计的逐帧变换前后进行。实际网络、采集循环和检查点接入仍是后续工作。

Gemini 保留总体方差、eps 位置、计数示例、两套统计与调用方去重责任，不把“只读 normalize”润色成“自动识别回放”。
