"""N05 Q 的训练目标与更新：用 D03 的多步奖励和当前 Actor 在尾部的新动作构造 y，只更新在线双 Q。

目标（D03 NStepBatch 的字段直接使用）：
    y = R + gamma^k × mask × (min(Q1_target, Q2_target)(s_tail, a') − alpha × log π(a' | s_tail))
    R = reward_sum，gamma^k = discount，mask = bootstrap_mask（真终止 0，正常与纯截断 1）
    s_tail 为 tail.next_*（第 k 次动作后、重置前的真实状态），a'、log π 为当前 Actor 在 s_tail 上的新采样。
熵项符号：−alpha × log π 是“软价值”中的熵奖励（log π 越小、越随机，价值越高），与作者 next_state_entropy 相同。

两种动作的用途不同：
    - 记录中的旧动作 first.action 只用于当前 Q 的预测 q1、q2（评价当时实际执行的动作）。
    - 尾部动作 a' 必须由当前 Actor 重新生成；不能用记录里的动作代替当前策略在尾部的选择。

梯度边界：y 在 no_grad 下计算，不反传到 Actor、编码器、alpha 或目标 Q；本次损失只修改在线 Q。
真终止分支：mask=0 的样本 y = R，不计算尾部 Q 与密度（显式分支，避免 0 × NaN 仍得到 NaN）。
损失：L_Q = 0.5 × [平均 (q1 − y)² + 平均 (q2 − y)²]（作者形式）。Adam 学习率 1e-3、梯度裁剪 1.0 沿用作者默认值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "networks_learning" / "history_encoder"))
sys.path.insert(0, str(_ROOT / "networks_learning" / "q_networks"))

from critics import TwinQ, evaluate_q_pair, list_critic_parameters  # noqa: E402
from encoder import features_from_history  # noqa: E402

CRITIC_LEARNING_RATE = 1e-3  # 作者 critic_learning_rate
MAX_GRAD_NORM = 1.0  # 作者 max_grad_norm


def make_critic_optimizer(critics: TwinQ, lr: float = CRITIC_LEARNING_RATE) -> torch.optim.Optimizer:
    """在线双 Q 的 Adam 优化器；不包含目标副本、Actor、编码器或 alpha。"""
    return torch.optim.Adam([p for _, p in list_critic_parameters(critics)], lr=lr)


def compute_target(batch, encoder, actor, target_critics, normalizers, alpha: float, generator=None) -> torch.Tensor:
    """构造本次冻结的目标 y [M,1]（不带梯度）。

    参数：
        batch: D03 NStepBatch。
        encoder / actor: N01 HistoryEncoder（eval）与 N03 SquashedGaussianActor，用于尾部的新动作。
        target_critics: N04 initialize_targets 得到的目标双 Q。
        normalizers: D04 ObservationNormalizers，尾部策略历史与 Q 观测按当前统计归一化（与在线 Q 一致）。
        alpha: 当前温度（数值，来自 N07，不参与本次求导）。
    """
    with torch.no_grad():
        target = batch.reward_sum.clone()
        live = batch.bootstrap_mask.squeeze(1) > 0
        if bool(live.any()):
            tail = batch.tail
            features = features_from_history(encoder, normalizers, tail.next_policy_history[live])
            next_action, next_log_prob = actor.sample_action_and_log_prob(features, generator=generator)
            q1, q2 = evaluate_q_pair(target_critics, normalizers.normalize_q(tail.next_q_observation[live]), next_action)
            soft_value = torch.min(q1, q2) - alpha * next_log_prob
            # live 行的 mask 为 1，这里乘 discount = gamma^k 即完成 gamma^k × mask × soft_value。
            target[live] = target[live] + batch.discount[live] * soft_value
    return target


def critic_loss(critics: TwinQ, batch, target: torch.Tensor, normalizers) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在线双 Q 评价记录中的旧动作，返回 (L_Q, q1, q2)。target 应为 compute_target 的结果（无梯度）。"""
    q1, q2 = evaluate_q_pair(critics, normalizers.normalize_q(batch.first.q_observation), batch.first.action)
    return 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target)), q1, q2


def update_critics(
    critics: TwinQ, optimizer: torch.optim.Optimizer, batch, target: torch.Tensor, normalizers, max_grad_norm: float = MAX_GRAD_NORM
) -> dict:
    """一次在线 Q 更新，返回数值诊断（损失、两个 Q 与目标的均值、目标绝对值最大值）。

    副作用：只修改 critics 的参数与 optimizer 状态。
    """
    loss, q1, q2 = critic_loss(critics, batch, target, normalizers)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_([p for _, p in list_critic_parameters(critics)], max_grad_norm)
    optimizer.step()
    return {
        "critic_loss": loss.item(),
        "q1_mean": q1.mean().item(),
        "q2_mean": q2.mean().item(),
        "target_mean": target.mean().item(),
        "target_abs_max": target.abs().max().item(),
    }
