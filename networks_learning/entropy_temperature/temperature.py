"""N07 探索力度 alpha：只更新 log_alpha，使策略熵的估计向目标熵靠近。

参数化：可训练参数是 log_alpha，alpha = exp(log_alpha) 自然为正；初值 alpha=0.05（作者默认）。
损失（作者形式）：L_alpha = −平均 [log_alpha × (log π + target_entropy)]，括号内在本次更新中当作固定数据（detach），
因此该损失不会反传到 Actor 或 Q。
调整方向：熵估计 H ≈ −log π。H 低于目标时 log π + target > 0，梯度下降使 log_alpha 增大，加强保留多样性；
H 高于目标时 alpha 减小。连续动作的熵可以为负，不据此判为错误。

目标熵的坐标约定（本项目明确规定）：
    - N03 的 log π 是策略动作坐标 a = b + c × tanh(u) 下的密度，已包含 −Σ log c 的缩放修正。
      作者 rsl_rl_sac 同样在 log π 中减去 log_action_range，并令 target_entropy = −target_entropy_scale × 动作维数，
      因此两者坐标一致：这里沿用作者配置 target_entropy_scale = 1.0，即 a 坐标下的目标熵 −8，作为显式实验起点。
    - 同一目标在单位区间坐标 y = tanh(u) ∈ (−1, 1)^8 下相当于 target − Σ log c（H_a = H_y + Σ log c）。
      E04 的半宽 c 改变时，若 a 坐标的目标保持 −8，单位区间上的探索要求会随之改变；target_entropy 同时返回
      这个等价值，供配置与日志记录。若希望探索语义不随动作尺度变化，应固定单位区间的目标并按 + Σ log c 换算。
Adam 学习率 1e-3 沿用作者默认值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

INITIAL_ALPHA = 0.05  # 作者 alpha 初值
TARGET_ENTROPY_SCALE = 1.0  # 作者 target_entropy_scale
TEMPERATURE_LEARNING_RATE = 1e-3  # 作者 alpha_learning_rate


@dataclass(frozen=True)
class TargetEntropy:
    """目标熵及其坐标换算。

    action_coordinate: 与 N03 log π 同一坐标（策略动作 a）下的目标，温度损失使用这一项。
    squashed_equivalent: 单位区间坐标 y = tanh(u) 下的等价目标 = action_coordinate − Σ log c。
    log_half_width_sum: Σ log c，二者之差。
    """

    action_coordinate: float
    squashed_equivalent: float
    log_half_width_sum: float


def target_entropy(half_width: Sequence[float], target_entropy_scale: float = TARGET_ENTROPY_SCALE) -> TargetEntropy:
    """按作者约定 −scale × 动作维数 给出 a 坐标下的目标，并换算单位区间的等价值。half_width 为 E04 的 c。"""
    log_sum = sum(math.log(c) for c in half_width)
    action_target = -target_entropy_scale * len(half_width)
    return TargetEntropy(action_target, action_target - log_sum, log_sum)


class Temperature(nn.Module):
    """只含一个可训练参数 log_alpha。"""

    def __init__(self, initial_alpha: float = INITIAL_ALPHA) -> None:
        super().__init__()
        if not initial_alpha > 0.0:
            raise ValueError(f"initial_alpha={initial_alpha} 必须为正")
        self.log_alpha = nn.Parameter(torch.tensor(math.log(initial_alpha)))


def current_alpha(temperature: Temperature) -> float:
    """当前 alpha 的 Python 标量（正数）；用于日志及兼容接口，不带梯度。"""
    return temperature.log_alpha.detach().exp().item()


def make_temperature_optimizer(temperature: Temperature, lr: float = TEMPERATURE_LEARNING_RATE) -> torch.optim.Optimizer:
    """只含 log_alpha 的 Adam 优化器。"""
    return torch.optim.Adam([temperature.log_alpha], lr=lr)


def temperature_loss(temperature: Temperature, log_prob: torch.Tensor, target: float) -> torch.Tensor:
    """L_alpha = −平均 [log_alpha × (log π + target)]；log π 先 detach，不向 Actor 求导。"""
    return -(temperature.log_alpha * (log_prob.detach() + target)).mean()


def update_temperature(temperature: Temperature, optimizer: torch.optim.Optimizer, log_prob: torch.Tensor, target: float) -> dict:
    """一次温度更新。log_prob 应来自 N06 同一次采样；target 为 TargetEntropy.action_coordinate。

    返回：alpha 损失、更新后的 alpha、熵估计 −平均 log π，以及熵估计与目标之差（负数表示熵偏低）。
    副作用：只修改 log_alpha 与 optimizer 状态。
    """
    loss = temperature_loss(temperature, log_prob, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    entropy = (-log_prob.detach()).mean().item()
    return {"alpha_loss": loss.item(), "alpha": current_alpha(temperature), "entropy_estimate": entropy, "entropy_gap": entropy - target}


def current_alpha_tensor(temperature: Temperature) -> torch.Tensor:
    """学习计算使用不带梯度的设备标量；日志仍使用原float接口。"""
    return temperature.log_alpha.detach().exp()
