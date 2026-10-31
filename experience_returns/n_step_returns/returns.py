"""D03 连续多步回报：从 D02 读出的同一轮连续片段，构造 Q 目标所需的奖励部分与尾部状态。

实际收集到 k 步（k ≤ n）时：
    R = r0 + gamma*r1 + ... + gamma^(k-1)*r(k-1)
    目标 = R + gamma^k * bootstrap_mask * V(尾部)，其中 V(尾部) 由 N05 用目标 Q 计算，本模块不调用 Q。
r0 是起点动作之后的奖励；尾部是这 k 次动作之后真实到达的状态，即第 k-1 条记录的 next_* 字段。

职责：
    - accumulate_rewards：按 gamma 的幂累计到首个结束边界或 n 步，返回 R 与 k。
    - select_tail：取每个片段第 k-1 条记录（含重置前的真实下一观测）。
    - build_n_step_batch：检查片段有效，组合起点记录、R、k、gamma^k、尾部与 bootstrap mask。

规则：
    - 真终止或纯时间截断都停止累计：之后的记录属于新一轮，其奖励不能计入上一轮。
    - 真终止 mask=0（丢弃尾部价值）；纯截断 mask=1，尾部使用重置前的真实状态估值。
      mask 直接调用 D01 derive_bootstrap_mask，与单步规则一致。
    - 既未凑满 n 步、又未遇到边界的起点不可用；这类起点由 D02 valid_starts 排除，
      这里再检查一次，发现时报错而不是把数据不足当成结束。
    - n=1 时 k 恒为 1，R=r0，gamma^k=gamma，退化为单步目标。

与作者 SAC 的关系（rsl_rl_sac e0d243a，BSD-3-Clause）：作者在 replay_buffer._generate_batch 中用
dones（终止或截断）的累积乘积屏蔽首个结束之后的奖励，并按首个结束位置取 next_obs，思路相同；
有效起点则只靠写指针判断。这里改用 D02 的显式编号连续性，并用真终止 / 纯截断两个标记推导 mask。
回放中的中间动作来自过去的策略，未经校正的多步目标可能有偏差；n=1 作为排查对照，n=5 是论文候选值。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "transition_schema"))
sys.path.insert(0, str(_ROOT / "replay_buffer"))

from buffer import SequenceBatch  # noqa: E402
from schema import TransitionBatch, derive_bootstrap_mask  # noqa: E402


class NStepReturnError(ValueError):
    """多步回报的输入不符合约定。"""


@dataclass(frozen=True)
class NStepBatch:
    """M 个起点的 n 步训练样本。

    first: 起点记录 [M, ...]（TransitionBatch）；Q 评价 first.q_observation 与 first.action，
        Actor / 编码器使用 first.policy_history。
    tail: 第 k-1 条记录 [M, ...]；目标使用 tail.next_policy_history、tail.next_q_observation、
        tail.next_history_length（重置前的真实状态）。
    reward_sum: [M,1] float32，R。
    steps: [M] int64，实际步数 k（1..n）。
    discount: [M,1] float32，gamma^k。
    bootstrap_mask: [M,1] float32，尾部真终止为 0，否则为 1。
    """

    first: TransitionBatch
    tail: TransitionBatch
    reward_sum: torch.Tensor
    steps: torch.Tensor
    discount: torch.Tensor
    bootstrap_mask: torch.Tensor


def accumulate_rewards(
    rewards: torch.Tensor, boundary: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 gamma 的幂累计奖励，直到首个结束边界（含该步）或片段末尾。

    参数：
        rewards: [M, n, 1] float32，片段内各步的原始奖励。
        boundary: [M, n] bool，该步是否为真终止或纯截断。
        gamma: 折扣比例，0 < gamma ≤ 1。
    返回：(R [M,1] float32, k [M] int64)。k = 首个边界的位置 + 1，没有边界时为 n。
    副作用：无。
    """
    if not 0.0 < gamma <= 1.0:
        raise NStepReturnError(f"gamma={gamma} 必须在 (0, 1]")
    n = rewards.shape[1]
    has_boundary = boundary.any(dim=1)
    first_boundary = torch.argmax(boundary.to(torch.int64), dim=1)  # 无边界时为 0，下面用 has_boundary 区分
    steps = torch.where(has_boundary, first_boundary + 1, torch.full_like(first_boundary, n))
    offsets = torch.arange(n, device=rewards.device)
    # 第 j 步的权重：j < k 时为 gamma^j，否则为 0，后者把边界之后（新一轮）的奖励排除在外。
    weights = torch.where(offsets < steps.unsqueeze(1), gamma ** offsets.to(rewards.dtype), torch.zeros((), dtype=rewards.dtype))
    reward_sum = (rewards.squeeze(-1) * weights).sum(dim=1, keepdim=True)
    return reward_sum, steps


def select_tail(sequence: SequenceBatch, steps: torch.Tensor) -> TransitionBatch:
    """取每个片段第 steps-1 条记录作为尾部（其 next_* 字段是 k 次动作后的真实状态）。

    纯截断时这一条的 next_* 是 E05 在重置前保存的末状态，而不是重置后状态；
    这里只是选取，不做替换。异常：ReplayBufferError：尾部记录无效。
    """
    return sequence.at(steps - 1)


def build_n_step_batch(sequence: SequenceBatch, gamma: float, n_steps: int) -> NStepBatch:
    """由 D02 read_sequence(..., length=n_steps) 的结果构造 n 步样本。

    检查：片段长度等于 n_steps；从起点到第 k-1 条全部有效（同一轮且连续）。
    若某起点在凑满 n 步或遇到边界之前就断开，说明它不是 D02 valid_starts(n_steps) 中的起点，报错。
    返回：NStepBatch。副作用：无（读取结果本身已是 D02 返回的副本）。
    """
    if sequence.length != n_steps:
        raise NStepReturnError(f"片段长度 {sequence.length} 与 n_steps={n_steps} 不一致")
    fields = sequence.fields
    boundary = (fields["terminated"] | fields["time_out"]).squeeze(-1) & sequence.valid
    reward_sum, steps = accumulate_rewards(fields["reward"], boundary, gamma)
    needed = torch.arange(n_steps, device=steps.device).unsqueeze(0) < steps.unsqueeze(1)
    missing = (needed & ~sequence.valid).any(dim=1)
    if bool(missing.any()):
        raise NStepReturnError(
            f"行 {torch.nonzero(missing).reshape(-1).tolist()} 在凑满 {n_steps} 步或遇到结束之前缺少同一轮的连续记录"
            "（后继尚未写入或已被覆盖）；数据不足不能当作结束，请用 D02 valid_starts / sample_starts 选择起点"
        )
    tail = select_tail(sequence, steps)
    return NStepBatch(
        first=sequence.at(torch.zeros_like(steps)),
        tail=tail,
        reward_sum=reward_sum,
        steps=steps,
        discount=(gamma ** steps.to(torch.float64)).to(torch.float32).unsqueeze(1),
        bootstrap_mask=derive_bootstrap_mask(tail),
    )
