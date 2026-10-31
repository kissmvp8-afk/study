"""N01 历史状态编码的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/history_encoder/test_encoder.py -v

数据说明：输入为随机或帧编号构造的人工历史；不使用真实 TITA 观测。
原 PPO 编码器按 baseline/ppo 快照 common_modules.mlp_batchnorm_factory 与 MlpBarlowTwinsActor 的层序在本文件中复写，
作为独立对照（只复写编码相关的三部分）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experience_returns" / "history_windows"))
sys.path.insert(0, str(ROOT / "experience_returns" / "normalization"))

from encoder import (  # noqa: E402
    ACTOR_FEATURE_DIM,
    HistoryEncoder,
    HistoryEncoderError,
    build_actor_features,
    features_from_history,
)
from history import split_windows  # noqa: E402
from normalization import ObservationNormalizers  # noqa: E402


def ppo_mlp_batchnorm_factory(activation, input_dims, out_dims, hidden_dims):
    """原 PPO common_modules.mlp_batchnorm_factory（bias=True、last_act=False 的情形）的复写。"""
    layers = [nn.Linear(input_dims, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), activation]
    for i in range(len(hidden_dims) - 1):
        layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.BatchNorm1d(hidden_dims[i + 1]), activation]
    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    return layers


class PpoEncoderReference(nn.Module):
    """原 PPO MlpBarlowTwinsActor 中 mlp_encoder / latent_layer / vel_layer 的复写（TITA 配置）。"""

    def __init__(self) -> None:
        super().__init__()
        self.mlp_encoder = nn.Sequential(*ppo_mlp_batchnorm_factory(nn.ELU(), 33 * 5, None, [128, 64]))
        self.latent_layer = nn.Sequential(*ppo_mlp_batchnorm_factory(nn.ELU(), 64, 16, [32]))
        self.vel_layer = nn.Linear(64, 3)


def random_history(batch: int = 6, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, 10, 33, generator=torch.Generator().manual_seed(seed))


def trained_like_encoder(seed: int = 0) -> HistoryEncoder:
    """随机初始化并让 BN 统计不再是默认值（模拟辅助学习之后的状态），返回 eval 模式的编码器。"""
    torch.manual_seed(seed)
    encoder = HistoryEncoder()
    encoder.train()
    with torch.no_grad():
        encoder.encode_history(torch.randn(64, 5, 33) * 2.0 + 0.5)
    encoder.eval()
    return encoder


class ShapeAndOrderTest(unittest.TestCase):
    """维数、帧顺序与重复输出。"""

    def test_dimensions(self) -> None:
        encoder = trained_like_encoder()
        windows = split_windows(random_history())
        representation = encoder.encode_history(windows.recent)
        self.assertEqual(representation.shape, (6, 64))
        self.assertEqual(encoder.predict_velocity(representation).shape, (6, 3))
        self.assertEqual(encoder.hidden_features(representation).shape, (6, 16))
        features = build_actor_features(encoder, windows)
        self.assertEqual(features.shape, (6, 52))
        self.assertEqual(ACTOR_FEATURE_DIM, 52)
        self.assertTrue(torch.equal(features[:, 19:], windows.current))  # 最后 33 维是当前帧

    def test_repeatable_in_eval_mode(self) -> None:
        encoder = trained_like_encoder()
        windows = split_windows(random_history())
        self.assertTrue(torch.equal(build_actor_features(encoder, windows), build_actor_features(encoder, windows)))

    def test_uses_recent_window_in_frame_order(self) -> None:
        encoder = trained_like_encoder()
        history = random_history()
        windows = split_windows(history)
        manual = encoder.backbone(history[:, 5:10].reshape(6, 165))  # t-4..t，帧优先展平
        self.assertTrue(torch.allclose(encoder.encode_history(windows.recent), manual))
        reversed_frames = encoder.encode_history(windows.recent.flip(1))
        self.assertFalse(torch.allclose(encoder.encode_history(windows.recent), reversed_frames))
        older = encoder.encode_history(windows.previous)  # 误用 t-5..t-1 时结果不同
        self.assertFalse(torch.allclose(encoder.encode_history(windows.recent), older))

    def test_rejects_wrong_window_shape(self) -> None:
        with self.assertRaises(HistoryEncoderError):
            HistoryEncoder().encode_history(torch.zeros(2, 6, 33))


class PpoEquivalenceTest(unittest.TestCase):
    """与原 PPO 编码器使用相同权重和历史，输出一致。"""

    def test_same_weights_same_outputs(self) -> None:
        encoder = trained_like_encoder(seed=3)
        reference = PpoEncoderReference()
        mapping = {"backbone": "mlp_encoder", "latent_head": "latent_layer", "velocity_head": "vel_layer"}
        state = {
            name.replace(prefix, mapping[prefix], 1): value
            for name, value in encoder.state_dict().items()
            for prefix in mapping
            if name.startswith(prefix + ".")
        }
        reference.load_state_dict(state)  # 键与形状完全一致才能加载成功
        reference.eval()
        history = random_history(seed=4)
        windows = split_windows(history)
        # 原 PPO forward：full = cat(hist[:, 1:], current)，actor_input = cat(vel, z, current)
        current, hist = history[:, -1], history[:, -6:-1]
        full = torch.cat([hist[:, 1:], current.unsqueeze(1)], dim=1)
        latent = reference.mlp_encoder(full.reshape(6, -1))
        expected = torch.cat([reference.vel_layer(latent), reference.latent_layer(latent), current], dim=-1)
        self.assertTrue(torch.allclose(build_actor_features(encoder, windows), expected, atol=1e-6))


class GradientAndBatchNormTest(unittest.TestCase):
    """Actor 特征不连接编码器计算图，也不改变 BN 统计。"""

    def test_features_detached_and_statistics_unchanged(self) -> None:
        encoder = trained_like_encoder()
        buffers = {k: v.clone() for k, v in encoder.state_dict().items()}
        history = random_history()
        features = features_from_history(encoder, ObservationNormalizers(6), history)
        self.assertFalse(features.requires_grad)
        for name, value in encoder.state_dict().items():
            self.assertTrue(torch.equal(value, buffers[name]), name)

    def test_train_mode_is_rejected(self) -> None:
        encoder = HistoryEncoder()
        encoder.train()
        with self.assertRaisesRegex(HistoryEncoderError, "train 模式"):
            build_actor_features(encoder, split_windows(random_history()))


if __name__ == "__main__":
    unittest.main()
