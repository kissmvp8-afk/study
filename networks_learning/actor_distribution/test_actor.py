"""N03 SAC 动作分布的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/actor_distribution/test_actor.py -v

数据说明：动作表来自 E03 由固定快照生成的原版表（需要本地有 baseline/ppo 快照）；软限位与工作轮速复用 E04 测试中
明确标注的**人工数据**（ARTIFICIAL_JOINT_POS_LIMITS / ARTIFICIAL_WHEEL_SPEED_RANGES），不是 TITA 实际限位。
密度的独立参考：float64 下按 Normal.log_prob − log(1 − tanh²) − log c 的朴素公式逐项计算。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.distributions import Normal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for path in ("environment_control/action_bounds", "environment_control/action_contract"):
    sys.path.insert(0, str(ROOT / path))

from actor import INITIAL_STD, SquashedGaussianActor, log_one_minus_tanh_squared, squash_and_log_prob  # noqa: E402
from baseline_source import load_baseline_action_contract  # noqa: E402
from bounds import calibrate_initial_center, derive_bounds  # noqa: E402
from contract import action_to_targets  # noqa: E402
from test_bounds import ARTIFICIAL_JOINT_POS_LIMITS, ARTIFICIAL_WHEEL_SPEED_RANGES  # noqa: E402

CONTRACT = load_baseline_action_contract()
BOUNDS = derive_bounds(CONTRACT, ARTIFICIAL_JOINT_POS_LIMITS, ARTIFICIAL_WHEEL_SPEED_RANGES)  # 人工限位，非对称
U0 = calibrate_initial_center(CONTRACT, BOUNDS)  # 希望初始目标 = 默认目标


def make_actor(seed: int = 0, initial_center=U0) -> SquashedGaussianActor:
    torch.manual_seed(seed)
    return SquashedGaussianActor(BOUNDS, initial_center)


def features(batch: int = 16, seed: int = 1) -> torch.Tensor:
    return torch.randn(batch, 52, generator=torch.Generator().manual_seed(seed))


class TransformTest(unittest.TestCase):
    """边界、极端输入与密度公式。"""

    def test_extreme_inputs_stay_in_bounds_with_finite_density(self) -> None:
        actor = make_actor()
        lower, upper = torch.tensor(BOUNDS.lower), torch.tensor(BOUNDS.upper)
        for scale in (1e3, 1e6):
            action, log_prob = actor.sample_action_and_log_prob(features() * scale, generator=torch.Generator().manual_seed(0))
            self.assertTrue(bool(((action >= lower) & (action <= upper)).all()))
            self.assertTrue(bool(torch.isfinite(log_prob).all()))
        u = torch.tensor([[-60.0, -20.0, 0.0, 20.0, 60.0, 1.0, -1.0, 5.0]])
        self.assertTrue(bool(torch.isfinite(log_one_minus_tanh_squared(u)).all()))
        self.assertTrue(bool(torch.isinf(torch.log(1 - torch.tanh(u[:, :1]) ** 2)).all()))  # 对照：朴素写法在 u=-60 处为 -inf
        # 中等 u 时稳定写法与朴素写法一致。
        moderate = torch.linspace(-3, 3, 13, dtype=torch.float64)
        self.assertTrue(torch.allclose(log_one_minus_tanh_squared(moderate), torch.log(1 - torch.tanh(moderate) ** 2)))

    def test_density_matches_independent_formula(self) -> None:
        actor = make_actor()
        x = features()
        noise = torch.randn(16, 8, generator=torch.Generator().manual_seed(5))
        action, log_prob = actor.sample_action_and_log_prob(x, noise=noise)
        mean, log_std = (t.double() for t in actor.distribution_parameters(x))
        u = mean + log_std.exp() * noise.double()
        c = torch.tensor(BOUNDS.half_width, dtype=torch.float64)
        b = torch.tensor(BOUNDS.center, dtype=torch.float64)
        reference = (Normal(mean, log_std.exp()).log_prob(u) - torch.log(1 - torch.tanh(u) ** 2) - torch.log(c)).sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(log_prob.double(), reference, atol=1e-4))
        self.assertTrue(torch.allclose(action.double(), b + c * torch.tanh(u), atol=1e-5))

    def test_deterministic_uses_same_transform(self) -> None:
        actor = make_actor()
        x = features()
        sampled, _ = actor.sample_action_and_log_prob(x, noise=torch.zeros(16, 8))
        self.assertTrue(torch.equal(actor.deterministic_action(x), sampled))

    def test_log_std_is_clamped(self) -> None:
        actor = make_actor()
        with torch.no_grad():
            actor.log_std_head.bias.copy_(torch.tensor([50.0, -50.0, 0, 0, 0, 0, 0, 0]))
        _, log_std = actor.distribution_parameters(features())
        self.assertTrue(bool((log_std[:, 0] == 2.0).all()) and bool((log_std[:, 1] == -20.0).all()))


class InitializationTest(unittest.TestCase):
    """温和开局：std=0.15，初始确定性物理目标靠近默认姿态。"""

    def test_initial_std(self) -> None:
        _, log_std = make_actor().distribution_parameters(features())
        self.assertTrue(torch.allclose(log_std.exp(), torch.full((16, 8), INITIAL_STD)))

    def test_initial_deterministic_target_near_default(self) -> None:
        x = features(64)
        calibrated = make_actor().deterministic_action(x)
        # a0 = 0（默认目标）；均值头权重很小，偏差远小于 0.05（策略动作单位）。
        self.assertLess(calibrated.abs().max().item(), 0.05)
        for row in calibrated[:4]:
            targets = action_to_targets(CONTRACT, row.tolist())
            for got, want, entry in zip(targets, CONTRACT.default_targets, CONTRACT.entries):
                self.assertLess(abs(got - want), 0.05 * entry.scale)
        # 对照：作者式零偏置（u≈0）在非对称维落到区间中点 b，而不是默认姿态。
        uncalibrated = make_actor(initial_center=[0.0] * 8).deterministic_action(x)
        self.assertAlmostEqual(uncalibrated[:, 0].mean().item(), BOUNDS.center[0], delta=0.05)  # b = 0.4，偏离 0
        self.assertGreater(abs(BOUNDS.center[0]), 0.3)


class GradientTest(unittest.TestCase):
    """采样路径保留 Actor 训练所需的梯度。"""

    def test_reparameterized_gradients(self) -> None:
        actor = make_actor()
        action, log_prob = actor.sample_action_and_log_prob(features(), generator=torch.Generator().manual_seed(2))
        (action.sum() + log_prob.sum()).backward()
        for name in ("mean_head.weight", "log_std_head.weight", "trunk.0.weight"):
            grad = dict(actor.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0.0, name)
        self.assertNotIn("center", dict(actor.named_parameters()))  # b、c 是缓冲，不被优化
        self.assertIn("half_width", dict(actor.named_buffers()))


if __name__ == "__main__":
    unittest.main()
