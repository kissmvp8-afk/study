# T05 · 模型导出与运行一致性｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex 写作约定](../../CODEX.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 讲解与技术复核已完成，本轮不再润色 |
| 技术核对状态 | 2026-10-31，Codex 对照实际实现、注释与测试核对职责、调用、形状、单位、代码片段和边界；本次是文档核对，不是全面代码审核 |
| 语言润色状态 | 本轮由 Codex 完成中文讲解与技术核对，不另交 Gemini |
| 已核对代码版本 | `a07d7b22e5bc2f40c462c01cc3378f7bcf3fe0c7`（`claude/training-core`，PR #7）；读取时与用户验收版本、远端分支一致，无后续源码差异 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | AGENTS.md、CODEX.md、根及本大模块/本目录 AI README；[export.py](export.py)、[test_export.py](test_export.py) 全文；N01 encoder.py、N03 actor.py、D04 normalization.py、D05 history.py、E03/E04 动作实现；固定 PPO 快照的 actor_critic.py（训练切片及 _InferenceWrapper）；依赖与来源详见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 本目录 AI 交接表记录 6 项 CPU 检查通过（Python 3.11.17 + torch 2.15.0），本次阅读测试、引用既有结果，未重跑算法测试。本轮执行的文档检查见大模块记录；人工数据/假环境不代表 Isaac 仿真、GPU、机器人训练或性能验证 |

## 导出模型必须带走输入处理

Actor 是生成动作的策略网络。导出指把训练中的推理计算保存成可重新加载的模型；推理是用固定参数算动作，不做学习。T05 导出的对象包含观测归一化、历史切片、历史编码和 Actor 的确定性动作。如果只保存 Actor 的权重，而运行端少做归一化或错取一帧，网络虽能执行，动作含义也会改变。

SAC（软演员—评论家算法）训练中会随机采样动作；这里导出确定性路径，即把内部高斯分布的均值直接经 tanh 和 E04 边界映射。它不是随机动作经非线性变换后的数学期望。Q、目标 Q、辅助学习 projector、温度 alpha 和随机采样均不进入执行链。

当前输出格式为 TorchScript trace（用示例记录张量运算图），保存 `.pt` 与同名 `.json` 元数据；尚无 ONNX、C++ 或真机运行实现。与原 PPO 一样采用 PyTorch 导出工具，但 SAC 的有界动作和单输入历史接口按本项目契约定义。

## 输入到输出的一条路径

```text
原始 policy_history[B,10,33]（含当前帧，从旧到新）
  → D04 用冻结的 33 维统计逐帧归一化
  → D05 current=9、recent=5:10
  → N01 展平 5×33=165，编码后拼成 [速度估计3 | 隐特征16 | 当前帧33]=52
  → N03 deterministic_action：a=b+c*tanh(mu)
  → action[B,8]（E03 顺序、策略动作坐标）
```

B 是一次输入的样本数。输入 float32，必须是**未经 D04 归一化**的历史；在运行端再归一化会重复变换。输入也不是全部传感器的任意排列，33 维各字段与尺度仍须遵守 E02。编码器速度预测沿用 2.0 缩放，内部不应被当作直接 m/s。

历史 `[0,1,…,9]` 的决策窗口应是 `[5,6,7,8,9]`。固定 PPO `_split_policy_obs` 先取 `[4,5,6,7,8]`，再去最旧帧并拼当前帧，得到相同结果；原 `_InferenceWrapper` 如果接收已含当前帧的完整历史，再追加 current，会得到 `[6,7,8,9,9]`。T05 只接收一份完整历史并复用 N01→D05，消除新 SAC 导出路径中的重复；它没有修改固定 PPO 快照，也没有证明旧 PPO 权重的实机兼容性。

## `export.py` 逐接口说明

`InferencePolicy` 是一个 `nn.Module`，`__init__(encoder,actor,normalizers)` 构造冻结副本；`forward(policy_history)` 返回动作 `[B,8]`，直接复用 N01 和 N03：

```python
return self.actor.deterministic_action(features_from_history(self.encoder, self.normalizers, policy_history))
```

先由 `features_from_history` 做归一化、切片和 52 维特征，再计算确定性动作，不在导出代码里重写一套索引或公式。这样以后修改公共路径时，训练与导出可一起接受测试。

冻结的真实操作为：

```python
self.encoder = copy.deepcopy(encoder).eval()
self.actor = copy.deepcopy(actor).eval()
self.normalizers = copy.deepcopy(normalizers)  # D04 统计的冻结副本（不是 nn.Module，trace 时作为常量写入）
for parameter in self.parameters():
    parameter.requires_grad_(False)
```

前两行复制参数和缓冲，并让编码器 BN 使用已有运行统计；第三行复制 D04 的均值、方差、计数与 eps；最后关闭网络参数求导。D04 不是 `nn.Module`，trace 时统计作为常量进入图。后续 Trainer 继续采集更新统计或训练网络，不会改变这份导出副本。这里既要 `eval` 又要复制，单独关闭梯度不能阻止 BN 在 train 模式更新统计。

| 函数 | 输入 → 输出、调用与副作用 |
|---|---|
| `build_inference_pipeline(trainer)` | Trainer → `InferencePolicy`，只读训练器，分配冻结副本；也可用于 T03 恢复后的 Trainer |
| `export_metadata(trainer)` | Trainer → 字典：输入输出形状/类型/坐标、E03 各关节类型/单位/default/scale/clip、E04 上下界、0.02 s 周期、D04 计数、配置名、版本和验证范围；只读 |
| `export_policy(trainer,path,example_batch=4)` | 构造管线，以非零 `[4,10,33]` 示例 trace，保存模型与 `path.with_suffix('.json')`，返回元数据；创建目录并写两个文件 |
| `load_exported_policy(path)` | `.pt` 路径 → eval 模式的 `ScriptModule`，读取文件，不更新训练状态 |
| `compare_outputs(reference,exported,histories,atol=1e-6)` | 两个张量输入接口和同一历史 → 最大绝对差、容限、是否达标、B；`no_grad` 下只比较确定性动作 |

核心导出步骤：

```python
traced = torch.jit.trace(pipeline, example, check_trace=True)
```

它用示例追踪，并做 trace 自带的核对；一次示例核对不能代替重置历史、极端输入和不同批量的独立测试。默认误差容限 `1e-6` 比的是策略动作坐标中的最大绝对误差，不是 rad 或 rad/s 误差。

已有验证环境中 torch 2.15 会给出 TorchScript 弃用提示；形状校验也会产生条件被追踪为常量的 TracerWarning。历史长度 10 与每帧 33 固定，测试证明示例 B=4 导出后 B=7 能用，不能据此声称任意输入形状都支持。GPU 与其他运行格式未验证。

## 物理换算只在运行端执行一次

输出 action 无物理单位，索引 0/1/2/4/5/6 是腿位置，3/7 是轮速。运行端按 E03 做 `default_target+scale*a`，再执行物理目标 clip：位置用 rad，轮速用 rad/s。这一步不在导出模型内。

例如轮速项默认 0 rad/s、scale=5，输出 a=0.4 时目标为 2 rad/s；不能把 2 再交给同一动作项缩放成 10 rad/s。腿位置项 scale=0.25，a=0.4 表示在默认角基础上增加 0.1 rad。Isaac 接入时应由环境动作项执行一次换算，不应先用 E03 函数换成物理目标后又送入会再次换算的动作项。

使用导出模型的接口是：

```python
metadata = export_policy(trainer, "policy.pt")
policy = load_exported_policy("policy.pt")
action = policy(policy_history)
```

这是调用示例，需调用方已构造或恢复 `trainer` 并维护正确原始历史；不是本轮运行记录。`policy.json` 明确写 `coordinate="policy_action"`、`physical_conversion_included=False`，用于运行端核对。0.02 s 对应 50 Hz，是控制接口要求；导出文件并不会自动调度时钟，也没有证明实时性能。

## `test_export.py` 验证与排错

测试工厂 `trained_trainer()` 在假环境训练 3 轮并做 2 次辅助更新；`training_path(trainer,history)` 用固定训练内路径算确定性动作；`frame_history(batch=3)` 给历史加可识别帧号及通道/批次偏移。`ExportConsistencyTest.setUpClass/tearDownClass` 共享并清理临时导出文件。

6 项既有 CPU 测试分别核对：

- `ExportConsistencyTest.test_three_paths_match`：训练内、普通 PyTorch 管线、重新加载的模型一致，B=7 不同于示例 4，重复调用确定。
- `test_frame_numbered_history_is_not_duplicated`：帧编号历史与训练一致，故意重复当前帧会改变动作。
- `test_statistics_frozen_at_export`：继续采集改变 Trainer 统计，已导出模型输出不变。
- `test_bounds_and_reset_history`：×1e4 极端输入仍在 E04 范围内，重置历史及极端输入下路径一致。
- `InterfaceTest.test_metadata_and_single_physical_conversion`：元数据读回一致、E03 顺序/0.02 s/策略坐标正确，导出与训练动作各换算一次后相同。
- `test_q_and_projector_not_exported`：网络参数只来自编码器和 Actor，关闭参数求导且为 eval。

本轮未重跑上述测试；既有通过记录见 [AI 交接表](README.md#实现证据与文档交接)。需要复现时：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/export_inference/test_export.py -v
```

动作对不上先用同一原始历史与同一冻结统计比较，依次查重复归一化、重复当前帧、BN 模式、E04 映射；物理目标差 5 倍或 0.25 倍时查重复缩放。CPU 数值一致不能替代 E01/E02/E05 接入、真实关节顺序、物理单位、GPU 和控制周期验证；证据分层继续读 [T06](../verification/README_HUMAN.md)。
