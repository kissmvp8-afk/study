"""N01 历史状态编码：从最近 5 帧（t-4..t）提取共享表示，输出 3 维速度估计与 16 维隐藏特征。

结构（与原 PPO MlpBarlowTwinsActor 相同，TITA 配置 bt_mlp_encoder_dims=[128,64]、bt_latent_encoder_dims=[32,16]、ELU）：
    backbone:      165 → Linear 128 → BN → ELU → Linear 64 → BN → ELU      （共享表示 64 维）
    latent_head:   64  → Linear 32 → BN → ELU → Linear 16                  （隐藏特征 16 维，无逐项物理含义）
    velocity_head: 64  → Linear 3                                           （缩放后的身体线速度估计）
展平顺序：窗口 [B,5,33] 按帧优先展平为 [B,165]，即 t-4 帧的 33 个数在前、t 帧在最后（与原 PPO reshape(b,-1) 相同）。

Actor 输入（52 维）= [速度估计 3 | 隐藏特征 16 | 当前帧 33]，顺序与原 PPO actor_input 一致。
速度估计与 E02 的速度监督标签同尺度：原 base_lin_vel 观测乘 2.0，单位不是 m/s，需除以 2.0 才是米每秒。

梯度与 BatchNorm 边界：
    - 本模块只定义前向，不创建优化器；编码器参数只由 N02 辅助学习更新。
    - build_actor_features 在 no_grad 下计算并返回与计算图断开的特征，Actor / Critic 的损失不会反传到编码器
      （原 PPO forward 同样在 no_grad 下计算编码器）。
    - 编码器默认处于 eval 模式：BatchNorm 使用已保存的运行统计、不更新。只有 N02 在辅助更新期间切换到 train，
      结束后恢复 eval；build_actor_features 若发现编码器处于 train 模式会报错，防止采集或 Actor 更新隐式改动统计。

不负责：归一化（D04）、切窗口（D05）、辅助损失与优化（N02）、动作分布（N03）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "experience_returns" / "history_windows"))
sys.path.insert(0, str(_ROOT / "experience_returns" / "normalization"))
sys.path.insert(0, str(_ROOT / "experience_returns" / "transition_schema"))

from history import WINDOW_FRAMES, HistoryWindows, split_windows  # noqa: E402
from schema import POLICY_OBS_DIM  # noqa: E402

VELOCITY_DIM = 3
HIDDEN_FEATURE_DIM = 16
ACTOR_FEATURE_DIM = VELOCITY_DIM + HIDDEN_FEATURE_DIM + POLICY_OBS_DIM  # 52


class HistoryEncoderError(ValueError):
    """输入形状或模式不符合约定。"""


def _mlp_batchnorm(input_dim: int, hidden_dims: list[int], out_dim: int | None) -> nn.Sequential:
    """按原 PPO mlp_batchnorm_factory 的层序构造：每个隐藏层为 Linear → BatchNorm1d → ELU，可选输出 Linear。"""
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ELU()]
        previous = width
    if out_dim is not None:
        layers.append(nn.Linear(previous, out_dim))
    return nn.Sequential(*layers)


class HistoryEncoder(nn.Module):
    """5 帧历史编码器（backbone + latent_head + velocity_head）。构造后处于 eval 模式。"""

    def __init__(self, backbone_dims: tuple[int, ...] = (128, 64), latent_hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = WINDOW_FRAMES * POLICY_OBS_DIM
        self.backbone = _mlp_batchnorm(self.input_dim, list(backbone_dims), None)
        self.latent_head = _mlp_batchnorm(backbone_dims[-1], [latent_hidden_dim], HIDDEN_FEATURE_DIM)
        self.velocity_head = nn.Linear(backbone_dims[-1], VELOCITY_DIM)
        self.eval()

    def encode_history(self, window: torch.Tensor) -> torch.Tensor:
        """窗口 [B,5,33]（已经 D04 归一化）→ 共享表示 [B,64]。按帧优先展平，不改变帧顺序。"""
        if window.dim() != 3 or tuple(window.shape[1:]) != (WINDOW_FRAMES, POLICY_OBS_DIM):
            raise HistoryEncoderError(f"窗口形状应为 [B,{WINDOW_FRAMES},{POLICY_OBS_DIM}]，实际为 {list(window.shape)}")
        return self.backbone(window.reshape(window.shape[0], self.input_dim))

    def predict_velocity(self, representation: torch.Tensor) -> torch.Tensor:
        """共享表示 [B,64] → 速度估计 [B,3]（与监督标签同为 2.0 缩放后的值）。"""
        return self.velocity_head(representation)

    def hidden_features(self, representation: torch.Tensor) -> torch.Tensor:
        """共享表示 [B,64] → 隐藏特征 [B,16]。"""
        return self.latent_head(representation)


def build_actor_features(encoder: HistoryEncoder, windows: HistoryWindows) -> torch.Tensor:
    """用 t-4..t 窗口编码，并拼接成 Actor 输入 [B,52] = [速度估计 3 | 隐藏特征 16 | 当前帧 33]。

    windows 来自 D05 split_windows，输入历史应已经 D04 normalize_policy。
    在 no_grad 下计算，返回张量不连接编码器的计算图（Actor 梯度隔离）；不修改编码器参数或 BN 统计。
    异常：HistoryEncoderError：编码器处于 train 模式。
    """
    if encoder.training:
        raise HistoryEncoderError("编码器处于 train 模式，BatchNorm 会更新统计；计算 Actor 特征前应调用 encoder.eval()")
    with torch.no_grad():
        representation = encoder.encode_history(windows.recent)
        velocity = encoder.predict_velocity(representation)
        hidden = encoder.hidden_features(representation)
        return torch.cat([velocity, hidden, windows.current], dim=-1)


def features_from_history(encoder: HistoryEncoder, normalizers, policy_history: torch.Tensor) -> torch.Tensor:
    """原始策略历史 [B,10,33] → Actor 输入 [B,52]：D04 normalize_policy → D05 split_windows → build_actor_features。

    normalizers 为 D04 ObservationNormalizers；只读其统计。训练（当前状态与尾部状态）与运行都走这一条路径。
    """
    return build_actor_features(encoder, split_windows(normalizers.normalize_policy(policy_history)))
