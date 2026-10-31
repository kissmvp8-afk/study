"""N02 速度监督与 Barlow Twins：用辅助目标训练 N01 历史编码器，独立于 SAC 的 Actor / Q / alpha 更新。

损失（与原 PPO MlpBarlowTwinsActor.barlow_twins_loss 相同）：
    z1 = projector(latent_head(backbone(t-4..t)))，z2 = projector(latent_head(backbone(t-5..t-1)))，各 [N,64]
    c  = z1ᵀ z2 / N                                  （64×64 跨窗口相关矩阵；projector 末层 BatchNorm 使各维零均值单位方差）
    L_bt  = Σ_i (c_ii − 1)² + 0.005 × Σ_{i≠j} c_ij²
    L_vel = 平均 (velocity_head(backbone(t-4..t)) − 速度标签)²
    L = L_bt + L_vel
对角项趋向 1：同一维在相邻两个窗口上一致；非对角项趋向 0：不同维互不重复。只要求“一致”时，全常数输出也能满足，
非对角项和 BatchNorm 标准化让常数特征的 c 为 0，对角项损失为 64（每一维都偏离 1），远离完美表示的 0，因此不会被误认为完美表示。
（64 不是上界：两窗口反相关时 c_ii = −1，对角项可达 4 × 64 = 256。）

速度标签：与 E02 / 原 PPO 相同，取时刻 t 的训练侧原始（未归一化）身体线速度观测，即乘过 2.0 的 base_lin_vel；
预测与标签在同一尺度上比较。标签由调用方显式传入，本模块不推断它在 Q 观测中的位置。

有效样本：只用 D05 valid_history_mask(history_length, 6) 为真的样本，保证两个窗口覆盖的 t-5..t 都是本轮真实帧；
速度监督使用同一批样本。有效样本少于 2 条时跳过本次更新（train 模式下 BatchNorm 至少需要 2 条），
不切换模式、不改参数、不改 BN 统计，并在结果中报告跳过。

优化器与 BatchNorm：
    - 独立 Adam 优化器，只包含编码器（backbone、latent_head、velocity_head）与训练用 projector 的参数；
      学习率 1e-3、梯度裁剪 1.0 沿用原 PPO 配置，作为起点。辅助更新频率由训练调度（T02）决定，与 SAC 回放更新分开。
    - 编码器与 projector 平时为 eval；只在 update_encoder 内切到 train，前向两个窗口时更新 BN 运行统计，
      结束后无论成功与否都恢复 eval。采集、Critic、Actor 更新都不会改动这些统计。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "networks_learning" / "history_encoder"))
sys.path.insert(0, str(_ROOT / "experience_returns" / "history_windows"))

from encoder import HIDDEN_FEATURE_DIM, VELOCITY_DIM, HistoryEncoder  # noqa: E402
from history import AUXILIARY_REQUIRED_FRAMES, HistoryWindows, valid_history_mask  # noqa: E402

PROJECTOR_DIM = 64  # 原 PPO bt_projector_dims=[64]
OFF_DIAGONAL_WEIGHT = 5e-3  # 原 PPO barlow_twins_loss 的 weight
AUXILIARY_LEARNING_RATE = 1e-3  # 原 PPO learning_rate 初值
MAX_GRAD_NORM = 1.0  # 原 PPO max_grad_norm
MIN_VALID_SAMPLES = 2  # train 模式的 BatchNorm 需要至少 2 条样本


class BarlowProjector(nn.Module):
    """训练用 64 维转换（只在辅助学习中使用，不进入 Actor 输入）。结构同原 PPO projector，构造后为 eval。

    16 → Linear(无偏置) 64 → BN → ELU → Linear(无偏置) 64 → BatchNorm1d(64, affine=False)
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_FEATURE_DIM, PROJECTOR_DIM, bias=False),
            nn.BatchNorm1d(PROJECTOR_DIM),
            nn.ELU(),
            nn.Linear(PROJECTOR_DIM, PROJECTOR_DIM, bias=False),
            nn.BatchNorm1d(PROJECTOR_DIM, affine=False),
        )
        self.eval()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def velocity_loss(predicted: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """速度估计与标签的均方误差；二者都是 [N,3]、同为 2.0 缩放后的值。"""
    if predicted.shape != label.shape or predicted.shape[-1] != VELOCITY_DIM:
        raise ValueError(f"速度预测 {list(predicted.shape)} 与标签 {list(label.shape)} 形状应同为 [N,3]")
    return torch.mean((predicted - label) ** 2)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """方阵的全部非对角元素（一维），写法同原 PPO off_diagonal。"""
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, off_diagonal_weight: float = OFF_DIAGONAL_WEIGHT):
    """跨窗口相关性损失，返回 (总损失, 对角项, 非对角项)。z1、z2 为 projector 输出 [N,64]。"""
    c = z1.T @ z2 / z1.shape[0]
    on_diagonal = (torch.diagonal(c) - 1.0).pow(2).sum()
    off_diagonal = _off_diagonal(c).pow(2).sum()
    return on_diagonal + off_diagonal_weight * off_diagonal, on_diagonal, off_diagonal


def make_auxiliary_optimizer(encoder: HistoryEncoder, projector: BarlowProjector, lr: float = AUXILIARY_LEARNING_RATE):
    """辅助学习的独立 Adam 优化器：只含编码器与 projector 的参数，不含 Actor 动作头、Q 或 alpha。"""
    return torch.optim.Adam([*encoder.parameters(), *projector.parameters()], lr=lr)


def update_encoder(
    encoder: HistoryEncoder,
    projector: BarlowProjector,
    optimizer: torch.optim.Optimizer,
    windows: HistoryWindows,
    history_length: torch.Tensor,
    velocity_label: torch.Tensor,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> dict:
    """一次辅助更新：筛选有效样本 → train 模式前向两个窗口 → L_bt + L_vel → 更新编码器与 projector → 恢复 eval。

    参数：
        windows: D05 split_windows 的结果，输入历史应已经 D04 normalize_policy；[B,...]。
        history_length: [B]，与 windows 同一时刻的有效历史长度（当前历史用 D01 history_length）。
        velocity_label: [B,3]，时刻 t 的原始速度监督标签（2.0 缩放）。
    返回：{"skipped", "valid_samples", 以及未跳过时的 "loss"、"velocity_loss"、"on_diagonal"、"off_diagonal"} 浮点日志。
    副作用：修改编码器与 projector 的参数和 BN 运行统计、优化器状态；跳过时不修改任何状态。
    """
    valid = valid_history_mask(history_length, AUXILIARY_REQUIRED_FRAMES)
    count = int(valid.sum())
    if count < MIN_VALID_SAMPLES:
        return {"skipped": True, "valid_samples": count}
    recent = windows.recent[valid].detach()
    previous = windows.previous[valid].detach()
    label = velocity_label[valid].detach()
    encoder.train()
    projector.train()
    try:
        representation = encoder.encode_history(recent)
        z1 = projector(encoder.hidden_features(representation))
        z2 = projector(encoder.hidden_features(encoder.encode_history(previous)))
        bt, on_diagonal, off_diagonal = barlow_loss(z1, z2)
        vel = velocity_loss(encoder.predict_velocity(representation), label)
        loss = bt + vel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*encoder.parameters(), *projector.parameters()], max_grad_norm)
        optimizer.step()
    finally:
        encoder.eval()
        projector.eval()
    return {
        "skipped": False,
        "valid_samples": count,
        "loss": loss.item(),
        "velocity_loss": vel.item(),
        "on_diagonal": on_diagonal.item(),
        "off_diagonal": off_diagonal.item(),
    }
