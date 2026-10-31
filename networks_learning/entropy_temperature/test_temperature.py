"""N07 探索力度 alpha 的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/entropy_temperature/test_temperature.py -v

数据说明：log π 为人工构造的常数或由 N03 Actor 在随机特征上采样；动作边界为人工对称区间（半宽 1 与 2）。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "networks_learning" / "actor_distribution"))
sys.path.insert(0, str(ROOT / "environment_control" / "action_bounds"))

from actor import SquashedGaussianActor  # noqa: E402
from bounds import ActionBounds  # noqa: E402
from temperature import (  # noqa: E402
    Temperature,
    current_alpha,
    make_temperature_optimizer,
    target_entropy,
    temperature_loss,
    update_temperature,
)


def artificial_bounds(half_width: float) -> ActionBounds:
    return ActionBounds(tuple(f"j{i}" for i in range(8)), (-half_width,) * 8, (half_width,) * 8, (0.0,) * 8, (half_width,) * 8)


class DirectionTest(unittest.TestCase):
    """熵低于 / 高于目标时 alpha 的调整方向。"""

    def test_alpha_direction(self) -> None:
        target = -8.0
        for log_prob_value, should_increase in ((20.0, True), (-30.0, False)):  # 熵估计 -20 < -8；30 > -8
            with self.subTest(log_prob=log_prob_value):
                temperature = Temperature()
                before = current_alpha(temperature)
                report = update_temperature(temperature, make_temperature_optimizer(temperature), torch.full((32, 1), log_prob_value), target)
                self.assertEqual(report["alpha"] > before, should_increase)
                self.assertEqual(report["entropy_gap"] < 0, should_increase)

    def test_alpha_positive_finite_and_restorable(self) -> None:
        temperature = Temperature()
        optimizer = make_temperature_optimizer(temperature, lr=0.5)
        for _ in range(200):
            update_temperature(temperature, optimizer, torch.full((8, 1), -1e4), -8.0)  # 熵远高于目标，持续减小 alpha
        alpha = current_alpha(temperature)
        self.assertTrue(alpha > 0.0 and math.isfinite(alpha))
        restored = Temperature()
        restored.load_state_dict(temperature.state_dict())
        self.assertEqual(current_alpha(restored), alpha)
        log_prob = torch.randn(8, 1)
        self.assertTrue(torch.equal(temperature_loss(restored, log_prob, -8.0), temperature_loss(temperature, log_prob, -8.0)))


class BoundaryTest(unittest.TestCase):
    """温度损失不反传到 Actor；目标熵与动作尺度的坐标约定。"""

    def test_no_gradient_to_actor(self) -> None:
        actor = SquashedGaussianActor(artificial_bounds(1.0), [0.0] * 8)
        _, log_prob = actor.sample_action_and_log_prob(torch.randn(16, 52))
        temperature = Temperature()
        temperature_loss(temperature, log_prob, -8.0).backward()
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        self.assertIsNotNone(temperature.log_alpha.grad)

    def test_action_scale_shifts_density_and_squashed_target(self) -> None:
        # 同一网络与噪声，半宽从 1 变为 2：log π 恰好减少 8·log 2；a 坐标目标仍为 −8（作者约定），
        # 单位区间等价目标从 −8 变为 −8 − 8·log 2，探索语义随动作尺度改变，因此必须记录。
        torch.manual_seed(0)
        actor_1 = SquashedGaussianActor(artificial_bounds(1.0), [0.0] * 8)
        actor_2 = SquashedGaussianActor(artificial_bounds(2.0), [0.0] * 8)
        actor_2.load_state_dict(actor_1.state_dict() | {"center": actor_2.center, "half_width": actor_2.half_width})
        features, noise = torch.randn(16, 52), torch.randn(16, 8)
        _, log_prob_1 = actor_1.sample_action_and_log_prob(features, noise=noise)
        _, log_prob_2 = actor_2.sample_action_and_log_prob(features, noise=noise)
        self.assertTrue(torch.allclose(log_prob_1 - log_prob_2, torch.full((16, 1), 8 * math.log(2.0)), atol=1e-4))
        t1, t2 = target_entropy([1.0] * 8), target_entropy([2.0] * 8)
        self.assertEqual((t1.action_coordinate, t2.action_coordinate), (-8.0, -8.0))
        self.assertAlmostEqual(t1.squashed_equivalent, -8.0)
        self.assertAlmostEqual(t2.squashed_equivalent, -8.0 - 8 * math.log(2.0))
        # 同一分布、同一目标下，单位区间熵相同，但 a 坐标的熵估计相差 8·log 2，alpha 的调整幅度因此不同。
        gap_1 = (-log_prob_1).mean().item() - t1.action_coordinate
        gap_2 = (-log_prob_2).mean().item() - t2.action_coordinate
        self.assertAlmostEqual(gap_2 - gap_1, 8 * math.log(2.0), places=3)


if __name__ == "__main__":
    unittest.main()
