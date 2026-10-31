"""N03 SAC 动作分布：由 52 维 Actor 输入产生 8 维有界策略动作及其对数密度。

变换（E04 约定）：
    mu, log_std = 网络(features)；log_std 限制在 [-20, 2]（作者 SACActorModel 的默认值）
    u = mu + exp(log_std) × epsilon，epsilon ~ N(0, I)          （重参数化：固定 epsilon 时 u 对参数可求导）
    a = b + c × tanh(u)                                          （b、c 为 E04 各维区间中心与半宽，a 是策略动作坐标）
对数密度（策略动作坐标 a 下，按 8 维相加）：
    log π(a) = Σ log N(u; mu, std) − Σ log(1 − tanh(u)²) − Σ log c
其中 log N(u; mu, std) = −epsilon²/2 − log_std − log(2π)/2，直接用 epsilon 计算，避免 (u−mu)/std 的除法误差；
log(1 − tanh(u)²) = 2 × (log 2 − u − softplus(−2u)) 是精确恒等式，|u| 很大时也不会出现 log(0)。
作者实现用 log(1 − tanh² + 1e-6) 近似，|u| 较大时会截断在 log(1e-6)，这里不采用该近似，因此没有近似误差。

确定性动作：u = mu，执行同一套 tanh 与缩放；它不是随机动作分布的数学均值（tanh 是非线性的）。

初始化（温和开局）：
    - 均值头权重 ~ N(0, 1e-3)（作者），偏置 = E04 calibrate_initial_center 求出的 u0，
      使初始确定性动作 ≈ E04 希望的初始动作 a0（对称区间且 a0=0 时 u0=0，与作者一致）。
    - log_std 头权重 = 0，偏置 = log(0.15)：初始内部标准差为 0.15（规范值，作者默认 1.0）。
    只缩小标准差而保留很大的随机均值不能保证温和开局，因此两者同时规定。
隐藏层 512/256/128、ELU 沿用原 PPO actor_hidden_dims。b、c 以不可训练的缓冲保存，随模型保存恢复。

不负责：历史编码（N01）、损失与优化（N06）、温度（N07）。运行时不调用 Q 搜索动作。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "environment_control" / "action_bounds"))
sys.path.insert(0, str(_ROOT / "networks_learning" / "history_encoder"))

from bounds import ActionBounds  # noqa: E402
from encoder import ACTOR_FEATURE_DIM  # noqa: E402

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0  # 作者 SACActorModel 默认
INITIAL_STD = 0.15  # 规范：内部初始标准差
MEAN_HEAD_INIT_STD = 1e-3  # 作者：均值头权重的初始化尺度
ACTOR_HIDDEN_DIMS = (512, 256, 128)  # 原 PPO actor_hidden_dims
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def log_one_minus_tanh_squared(u: torch.Tensor) -> torch.Tensor:
    """精确且数值稳定的 log(1 − tanh(u)²) = 2 × (log 2 − u − softplus(−2u))。"""
    return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


def squash_and_log_prob(
    mean: torch.Tensor, log_std: torch.Tensor, noise: torch.Tensor, center: torch.Tensor, half_width: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """给定 mu、log_std、噪声 epsilon 与 E04 的 b、c，返回 (策略动作 a [B,8], 对数密度 [B,1])。公式见文件头。"""
    u = mean + log_std.exp() * noise
    action = center + half_width * torch.tanh(u)
    gaussian = -0.5 * noise.pow(2) - log_std - _HALF_LOG_2PI
    log_prob = (gaussian - log_one_minus_tanh_squared(u) - torch.log(half_width)).sum(dim=-1, keepdim=True)
    return action, log_prob


class SquashedGaussianActor(nn.Module):
    """52 → 512 → 256 → 128（ELU）→ 均值头 8 + log_std 头 8，并带 E04 的 b、c 缓冲。

    参数：
        bounds: E04 derive_bounds 的结果（策略动作坐标下的 L、U、b、c）。
        initial_center: E04 calibrate_initial_center 的结果 u0（长度 8）。
    """

    def __init__(
        self,
        bounds: ActionBounds,
        initial_center: Sequence[float],
        feature_dim: int = ACTOR_FEATURE_DIM,
        hidden_dims: Sequence[int] = ACTOR_HIDDEN_DIMS,
        initial_std: float = INITIAL_STD,
    ) -> None:
        super().__init__()
        action_dim = len(bounds.center)
        if len(initial_center) != action_dim:
            raise ValueError(f"初始中心长度 {len(initial_center)} 与动作维数 {action_dim} 不符")
        self.action_dim = action_dim
        self.register_buffer("center", torch.tensor(bounds.center, dtype=torch.float32))
        self.register_buffer("half_width", torch.tensor(bounds.half_width, dtype=torch.float32))
        layers: list[nn.Module] = []
        previous = feature_dim
        for width in hidden_dims:
            layers += [nn.Linear(previous, width), nn.ELU()]
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(previous, action_dim)
        self.log_std_head = nn.Linear(previous, action_dim)
        with torch.no_grad():
            nn.init.normal_(self.mean_head.weight, mean=0.0, std=MEAN_HEAD_INIT_STD)
            self.mean_head.bias.copy_(torch.tensor(initial_center, dtype=torch.float32))
            nn.init.zeros_(self.log_std_head.weight)
            self.log_std_head.bias.fill_(math.log(initial_std))

    def distribution_parameters(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features [B,52] → (mu [B,8], log_std [B,8])，log_std 已限制在 [-20, 2]。"""
        hidden = self.trunk(features)
        return self.mean_head(hidden), self.log_std_head(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample_action_and_log_prob(
        self, features: torch.Tensor, noise: torch.Tensor | None = None, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """重参数化采样：返回同一次采样的 (动作 a [B,8], 对数密度 [B,1])，二者都保留对 Actor 参数的梯度。

        noise 为 [B,8] 的标准正态噪声，给定时用于复现；否则用 generator 抽取。
        """
        mean, log_std = self.distribution_parameters(features)
        if noise is None:
            noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype).to(mean.device)
        return squash_and_log_prob(mean, log_std, noise, self.center, self.half_width)

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        """确定性动作 a = b + c × tanh(mu)，与采样使用同一套变换。"""
        mean, _ = self.distribution_parameters(features)
        return self.center + self.half_width * torch.tanh(mean)
