"""D01 经历记录契约：定义环境采集、经验池与训练器共用的一批经历记录。

一条经历描述“在本轮哪个时刻，看到了什么，做了什么，得到多少奖励，真正到达了哪里，为何结束”。

职责：
    - TransitionBatch：一批记录的字段、形状、类型、单位与时间含义。
    - make_transition：创建记录的入口，负责复制来源张量并校验。
    - validate_transition：检查字段齐全、形状与批维一致、类型、数值有限、编号与有效历史长度。
    - derive_bootstrap_mask：由真终止标记推导是否接后续价值（真终止 0，其余 1）。
    - describe_transition：生成单条记录的人工核查摘要。

不负责：
    - 不从仿真捕获末状态、不重置环境（E05 负责在重置前保存真实下一观测，并把同一步“摔倒又到时”
      处理为真终止后再交给本模块）。
    - 不写入经验池、不抽样（D02），不累计多步奖励（D03），不切历史窗口（D05），不归一化（D04），
      不保存会随网络训练变化的历史编码特征（第一版只存编码前的完整历史，由当前编码器重新解释）。

数据容器：PyTorch 张量（torch.Tensor），与 Isaac Lab 环境输出及后续经验池、网络一致；不提供其他后端。
测试只需 CPU。

与原 PPO / 作者 SAC 的关系：
    - 原 PPO（NP3O）把 time_outs 的价值预先加进奖励（rewards += gamma * values * time_outs），
      SAC 的经历会被长期保存、用新网络重新估值，因此这里只存环境原始奖励，不预加价值补偿。
    - 作者 SAC（rsl_rl_sac e0d243a，BSD-3-Clause）存 dones（终止或截断）与 bootstrap（time_outs）两个标记，
      目标中用 bootstrap + 1 - dones 作为是否接后续价值；同一步既终止又到时，该式会得到 1。
      本项目约定真终止优先，因此只存“真终止”和“纯时间截断”，开关只由真终止推导，并拒绝两者同时为真。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

HISTORY_LENGTH = 10  # 策略历史帧数，包含当前帧，从旧到新排列（根 README 统一约定）
POLICY_OBS_DIM = 33  # 每帧策略观测维数（E02）
ACTION_DIM = 8  # 策略动作维数，顺序与 E03 TITA_ACTION_LAYOUT 一致


class TransitionSchemaError(ValueError):
    """记录不符合契约；消息中写明出错的字段名。"""


@dataclass(frozen=True)
class TransitionBatch:
    """一批经历记录（B 条），每个字段第 0 维都是批维 B，第 i 行属于同一条经历。

    时间含义：第 i 条记录描述环境 env_id[i] 在第 episode_id[i] 轮、本轮第 step_index[i] 次动作的前后。
    “当前”指执行该动作前的时刻 t；“真实下一”指该动作执行后、任何重置发生之前的时刻 t+1。

    字段：
        policy_history: [B,10,33] float32。时刻 t 的策略观测历史，含当前帧，从旧到新。
        next_policy_history: [B,10,33] float32。动作后真实到达的历史（重置前）。结束的环境随后会被重置，
            重置后观测只用于下一次交互，不能填入这里，否则会把新一轮的状态接到上一轮的末尾。
        q_observation: [B,C] float32。时刻 t 的 Q 输入（训练侧信息，含 Actor 看不到的字段）；C 由调用方给出。
        next_q_observation: [B,C] float32。动作后真实到达的 Q 输入（重置前）。
        action: [B,8] float32。实际送入环境的策略动作，策略动作坐标、无物理单位；
            不换算成关节角或轮速（换算见 E03 action_to_targets），Q 评价的必须是同一坐标的动作。
        reward: [B,1] float32。环境返回的原始奖励，不含任何价值补偿。
        terminated: [B,1] bool。真终止（如机身触地），之后没有未来价值可接。
        time_out: [B,1] bool。纯时间截断：只因到达轮次时长上限而结束，机器人仍可继续，未来价值应保留。
            与 terminated 互斥；同一步两者都发生时，E05 应只标 terminated。
        env_id: [B] int64。并行环境编号，≥ 0。
        episode_id: [B] int64。该环境内的轮次编号，≥ 0；重置后加一，用于防止跨轮次拼接。
        step_index: [B] int64。本轮内的动作序号，≥ 0。数组中相邻的两行不代表时间相邻，时序只看这些编号。
        history_length: [B] int64，1..10。policy_history 中属于本轮真实观测的帧数，其余为重置填充。
        next_history_length: [B] int64，1..10。next_policy_history 的有效帧数，应为 min(history_length+1, 10)。

    冻结只阻止重新绑定字段，不阻止对张量内容的原地修改；下游读取时不得原地修改这些张量。
    """

    policy_history: torch.Tensor
    next_policy_history: torch.Tensor
    q_observation: torch.Tensor
    next_q_observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    time_out: torch.Tensor
    env_id: torch.Tensor
    episode_id: torch.Tensor
    step_index: torch.Tensor
    history_length: torch.Tensor
    next_history_length: torch.Tensor

    @property
    def batch_size(self) -> int:
        """记录条数 B。"""
        return int(self.reward.shape[0])


FIELD_NAMES = tuple(f.name for f in fields(TransitionBatch))


def _expected_layout(q_dim: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """各字段除批维外的形状与类型。q_dim 即 C。"""
    history = ((HISTORY_LENGTH, POLICY_OBS_DIM), torch.float32)
    q = ((q_dim,), torch.float32)
    flag = ((1,), torch.bool)
    index = ((), torch.int64)
    return {
        "policy_history": history,
        "next_policy_history": history,
        "q_observation": q,
        "next_q_observation": q,
        "action": ((ACTION_DIM,), torch.float32),
        "reward": ((1,), torch.float32),
        "terminated": flag,
        "time_out": flag,
        "env_id": index,
        "episode_id": index,
        "step_index": index,
        "history_length": index,
        "next_history_length": index,
    }


def make_transition(data: dict[str, torch.Tensor], q_dim: int) -> TransitionBatch:
    """由一组来源张量创建记录：先检查字段名，再逐字段复制，最后校验。

    复制责任在这里：每个字段都执行 detach().clone()，因此采集循环之后原地修改来源张量
    （例如环境重置时覆盖观测缓冲区），不会改变已创建的记录。复制保持原设备与类型，不做类型转换，
    类型不符由 validate_transition 报错，避免把 float64 或整型奖励悄悄转换掉。

    参数：
        data: 字段名 → 张量，必须恰好包含 FIELD_NAMES 中的全部字段；多出的键（如 reset_observation、
            bootstrap）会被拒绝，防止把重置后观测或另存的开关混进记录。
        q_dim: Q 观测维数 C，由调用方按实际观测组给出（平地预期 54、复杂地形预期 241，见 E02）；
            不从数据推断，也不截断或补齐。
    返回：TransitionBatch（字段为独立副本）。
    异常：TransitionSchemaError：字段缺失、多余、不是张量，或 validate_transition 不通过。
    副作用：分配与来源同等大小的新显存/内存。
    """
    missing = [name for name in FIELD_NAMES if name not in data]
    unexpected = [name for name in data if name not in FIELD_NAMES]
    if missing or unexpected:
        raise TransitionSchemaError(f"字段不匹配：缺失 {missing}，多余 {unexpected}")
    copies: dict[str, torch.Tensor] = {}
    for name in FIELD_NAMES:
        value = data[name]
        if not isinstance(value, torch.Tensor):
            raise TransitionSchemaError(f"字段 {name} 必须是 torch.Tensor，实际为 {type(value).__name__}")
        copies[name] = value.detach().clone()
    batch = TransitionBatch(**copies)
    validate_transition(batch, q_dim)
    return batch


def validate_transition(batch: TransitionBatch, q_dim: int) -> None:
    """检查一批记录是否符合契约，列出全部问题后一次报错。

    检查项：
        - 形状与类型：见 TransitionBatch 字段说明；所有字段批维 B 相同且 B ≥ 1，位于同一设备。
        - 浮点字段（历史、Q 观测、动作、奖励）全部为有限数。
        - terminated 与 time_out 不能同时为真：真终止优先由 E05 处理，这里只接收已区分的结果。
        - env_id、episode_id、step_index ≥ 0。
        - history_length、next_history_length 在 1..10，且 next = min(current + 1, 10)：
          真实下一观测与当前观测属于同一轮、只晚一步，历史只推进一帧。若下一历史被重置后观测替代，
          其有效长度会变成 1，这一项会报错。
    参数：
        batch: 待检查的记录。
        q_dim: Q 观测维数 C，必须为正整数。
    异常：TransitionSchemaError。
    副作用：无。
    """
    if not isinstance(q_dim, int) or q_dim <= 0:
        raise TransitionSchemaError(f"q_dim 必须为正整数，实际为 {q_dim!r}")
    problems: list[str] = []
    batch_sizes: dict[str, int] = {}
    devices: set[torch.device] = set()
    for name, (shape, dtype) in _expected_layout(q_dim).items():
        value = getattr(batch, name)
        if not isinstance(value, torch.Tensor):
            problems.append(f"{name}：缺失或不是 torch.Tensor（实际为 {type(value).__name__}）")
            continue
        if value.dim() != len(shape) + 1 or tuple(value.shape[1:]) != shape:
            problems.append(f"{name}：形状应为 {_shape_text(shape)}，实际为 {list(value.shape)}")
        else:
            batch_sizes[name] = int(value.shape[0])
        if value.dtype != dtype:
            problems.append(f"{name}：类型应为 {dtype}，实际为 {value.dtype}")
        elif value.is_floating_point() and not bool(torch.isfinite(value).all()):
            problems.append(f"{name}：含 NaN 或 inf")
        devices.add(value.device)
    if problems:
        # 形状或类型已经不对时，后续逐行检查没有意义，先报告这些问题。
        raise TransitionSchemaError("经历记录校验失败：\n- " + "\n- ".join(problems))

    if len(set(batch_sizes.values())) != 1:
        problems.append(f"批维不一致：{batch_sizes}")
    elif next(iter(batch_sizes.values())) < 1:
        problems.append("批维 B 必须 ≥ 1")
    if len(devices) != 1:
        problems.append(f"字段位于不同设备：{sorted(str(d) for d in devices)}")
    if problems:
        raise TransitionSchemaError("经历记录校验失败：\n- " + "\n- ".join(problems))

    both = (batch.terminated & batch.time_out).squeeze(1)
    if bool(both.any()):
        problems.append(f"terminated 与 time_out 在行 {_rows(both)} 同时为真；真终止优先应由 E05 处理后只标 terminated")
    for name in ("env_id", "episode_id", "step_index"):
        bad = getattr(batch, name) < 0
        if bool(bad.any()):
            problems.append(f"{name}：行 {_rows(bad)} 为负数")
    for name in ("history_length", "next_history_length"):
        value = getattr(batch, name)
        bad = (value < 1) | (value > HISTORY_LENGTH)
        if bool(bad.any()):
            problems.append(f"{name}：行 {_rows(bad)} 不在 1..{HISTORY_LENGTH}")
    expected_next = torch.clamp(batch.history_length + 1, max=HISTORY_LENGTH)
    bad = batch.next_history_length != expected_next
    if bool(bad.any()):
        problems.append(
            f"next_history_length：行 {_rows(bad)} 不等于 min(history_length+1, {HISTORY_LENGTH})，"
            "检查是否填入了重置后观测或历史被重复推进"
        )
    if problems:
        raise TransitionSchemaError("经历记录校验失败：\n- " + "\n- ".join(problems))


def _shape_text(shape: tuple[int, ...]) -> str:
    """把除批维外的形状写成 [B,...] 文本，如 (10, 33) → "[B,10,33]"，() → "[B]"。"""
    return "[" + ",".join(["B", *map(str, shape)]) + "]"


def _rows(mask: torch.Tensor) -> list[int]:
    """把布尔掩码转换成出错行号列表，用于报错信息。"""
    return torch.nonzero(mask.reshape(-1)).reshape(-1).tolist()


def derive_bootstrap_mask(batch: TransitionBatch) -> torch.Tensor:
    """推导是否接后续价值：真终止为 0，正常步和纯时间截断为 1。

    只看 terminated：真终止之后没有未来，接上未来价值会虚增回报；纯时间截断是人为结束，
    机器人在末状态仍可继续，因此保留未来价值，由 next_* 字段（重置前的真实末状态）估值。
    由于只依赖 terminated，即使上游误把两个标记同时置真，结果仍是 0（真终止优先）。
    返回：[B,1] float32，与 reward 同设备，可直接乘到目标 Q 上；不修改 batch。
    """
    return (~batch.terminated).to(torch.float32)


def describe_transition(batch: TransitionBatch, row: int = 0) -> str:
    """生成第 row 条记录的多行摘要，供人工核查编号、长度、动作、奖励与结束原因。

    异常：IndexError：row 超出 [0, B)。
    """
    if not 0 <= row < batch.batch_size:
        raise IndexError(f"row={row} 超出批大小 {batch.batch_size}")
    if bool(batch.terminated[row, 0]):
        reason = "真终止"
    elif bool(batch.time_out[row, 0]):
        reason = "纯时间截断"
    else:
        reason = "未结束"
    action = ", ".join(f"{v:.4g}" for v in batch.action[row].tolist())
    return "\n".join(
        [
            f"记录 {row}/{batch.batch_size}：env_id={int(batch.env_id[row])} episode_id={int(batch.episode_id[row])} "
            f"step_index={int(batch.step_index[row])}",
            f"  有效历史长度：当前 {int(batch.history_length[row])} → 下一 {int(batch.next_history_length[row])}（共 {HISTORY_LENGTH} 帧）",
            f"  策略动作（策略坐标，无单位）：[{action}]",
            f"  原始奖励：{float(batch.reward[row, 0]):.6g}",
            f"  结束原因：{reason}；bootstrap mask={float(derive_bootstrap_mask(batch)[row, 0]):g}",
            f"  形状：policy_history {list(batch.policy_history.shape)}，q_observation {list(batch.q_observation.shape)}，"
            f"设备 {batch.reward.device}",
        ]
    )
