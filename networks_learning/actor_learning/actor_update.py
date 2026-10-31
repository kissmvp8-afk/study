"""N06 Actor 的参数更新：在回放状态上重新采样动作，最小化 L_actor = 平均 [alpha × log π(a|s) − min(Q1, Q2)(s, a)]。

为什么重新采样：旧经历只提供状态 s；被改进的是当前策略，因此 a 与 log π 必须由当前 Actor 重新计算。
不需要 PPO 的新旧概率比、优势或裁剪。
两个方向：−min(Q) 推动动作朝 Q 评分更高的方向移动；alpha × log π 惩罚过于集中的分布，保留多样性。

梯度边界：
    - Q：前向期间临时把在线双 Q 参数设为 requires_grad=False（结束后恢复），参数不会得到梯度，
      但 Q 对输入动作 a 的计算关系保留，梯度沿 a → tanh → u → Actor 参数传回（作者同样冻结 Q 参数）。
    - 编码器：Actor 输入 features 由 N01 build_actor_features 在 no_grad、eval 下得到，不连接编码器，
      也不改变其 BatchNorm 统计。本模块只接收已计算好的 features。
    - alpha：以数值传入（N07 current_alpha 的值），不对 log_alpha 求导；温度只由 N07 的损失更新。
同一次采样的 log π 也提供给 N07（与作者相同），保证 Actor 与温度更新看到同一组动作密度。
Adam 学习率 1e-3、梯度裁剪 1.0 沿用作者默认值。作者每 policy_frequency=2 次 Critic 更新做一次 Actor 更新，
频率由训练调度（T02）决定。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "networks_learning" / "q_networks"))

from critics import evaluate_q_pair  # noqa: E402

ACTOR_LEARNING_RATE = 1e-3  # 作者 actor_learning_rate
MAX_GRAD_NORM = 1.0  # 作者 max_grad_norm
NEAR_BOUND_THRESHOLD = 0.99  # |tanh(u)| 超过该值视为贴近边界（距边界不到半宽的 1%）


@contextmanager
def frozen_parameters(module: nn.Module):
    """临时关闭 module 全部参数的 requires_grad，退出时恢复原值；不影响输入张量的梯度。"""
    flags = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def make_actor_optimizer(actor: nn.Module, lr: float = ACTOR_LEARNING_RATE) -> torch.optim.Optimizer:
    """Actor（N03 网络）参数的 Adam 优化器；不包含编码器、Q 或 log_alpha。"""
    return torch.optim.Adam(actor.parameters(), lr=lr)


def compute_actor_loss(actor, critics, features: torch.Tensor, q_observation: torch.Tensor, alpha: float, generator=None):
    """返回 (L_actor, 新动作 a [B,8], log π [B,1])。

    features: [B,52]，回放状态经 D04 / D05 / N01 得到（无梯度）；q_observation: [B,C]，已 D04 normalize_q。
    a 与 log π 带 Actor 的梯度；Q 参数在前向时被冻结，因而不会累积梯度。
    """
    action, log_prob = actor.sample_action_and_log_prob(features, generator=generator)
    with frozen_parameters(critics):
        q1, q2 = evaluate_q_pair(critics, q_observation, action)
    loss = (alpha * log_prob - torch.min(q1, q2)).mean()
    return loss, action, log_prob


def update_actor(actor, optimizer: torch.optim.Optimizer, loss: torch.Tensor, max_grad_norm: float = MAX_GRAD_NORM) -> dict:
    """反传 L_actor 并更新 Actor 参数。副作用：只修改 actor 参数与 optimizer 状态。"""
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    optimizer.step()
    return {"actor_loss": loss.item()}


def report_policy_change(actor, features: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor) -> dict:
    """报告随机程度与贴边比例（只读）：平均内部标准差、−log π 的均值（熵的单样本估计）、贴近边界的动作比例。"""
    with torch.no_grad():
        _, log_std = actor.distribution_parameters(features)
        squashed = (action - actor.center) / actor.half_width  # = tanh(u)
        return {
            "mean_std": log_std.exp().mean().item(),
            "entropy_estimate": (-log_prob).mean().item(),
            "near_bound_fraction": (squashed.abs() > NEAR_BOUND_THRESHOLD).float().mean().item(),
        }
