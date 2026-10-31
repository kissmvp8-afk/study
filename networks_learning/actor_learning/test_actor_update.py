"""N06 Actor 参数更新的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/actor_learning/test_actor_update.py -v

数据说明：Actor 输入特征、Q 观测为随机人工数据；动作边界为人工对称区间 [-1, 1]（非 TITA 实际边界）。
“已知峰值的假 Q”为 Q(a) = −‖a − a*‖²，a* 为人工指定的目标动作。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for path in ("networks_learning/actor_distribution", "networks_learning/q_networks", "networks_learning/history_encoder",
             "environment_control/action_bounds"):
    sys.path.insert(0, str(ROOT / path))

from actor import SquashedGaussianActor  # noqa: E402
from actor_update import compute_actor_loss, make_actor_optimizer, report_policy_change, update_actor  # noqa: E402
from bounds import ActionBounds  # noqa: E402
from critics import TwinQ, initialize_targets  # noqa: E402
from encoder import HistoryEncoder  # noqa: E402

ARTIFICIAL_BOUNDS = ActionBounds(tuple(f"j{i}" for i in range(8)), (-1.0,) * 8, (1.0,) * 8, (0.0,) * 8, (1.0,) * 8)
PEAK = torch.tensor([0.5, -0.4, 0.3, 0.6, -0.2, 0.1, -0.5, 0.4])


class PeakQ(torch.nn.Module):
    """双 Q 都等于 −‖a − a*‖²（与状态无关），含一个占位参数以检查冻结。"""

    def __init__(self) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(1))
        self.q1 = self.score
        self.q2 = self.score

    def score(self, q_observation, action):
        return -((action - PEAK) ** 2).sum(-1, keepdim=True) + self.offset


def setup(seed: int = 0):
    torch.manual_seed(seed)
    actor = SquashedGaussianActor(ARTIFICIAL_BOUNDS, [0.0] * 8)
    features = torch.randn(64, 52, generator=torch.Generator().manual_seed(seed + 1))
    return actor, features, torch.randn(64, 6, generator=torch.Generator().manual_seed(seed + 2))


class DirectionTest(unittest.TestCase):
    """已知峰值的假 Q：更新方向正确。"""

    def test_moves_toward_peak(self) -> None:
        actor, features, q_obs = setup()
        critics = PeakQ()
        optimizer = make_actor_optimizer(actor, lr=3e-3)
        start = (actor.deterministic_action(features) - PEAK).pow(2).sum(-1).mean().item()
        for step in range(150):
            action, log_prob = actor.sample_action_and_log_prob(features, generator=torch.Generator().manual_seed(step))
            loss = compute_actor_loss(critics, q_obs, action, log_prob, alpha=1e-3)
            update_actor(actor, optimizer, loss)
        end = (actor.deterministic_action(features) - PEAK).pow(2).sum(-1).mean().item()
        self.assertLess(end, 0.2 * start)
        self.assertIsNone(critics.offset.grad)  # Q 参数被冻结，未累积梯度
        self.assertTrue(critics.offset.requires_grad)  # 退出后恢复


class IsolationTest(unittest.TestCase):
    """Actor 更新只改 Actor；Q 对动作的梯度保留。"""

    def test_only_actor_changes(self) -> None:
        actor, features, q_obs = setup()
        critics = TwinQ(6)
        targets = initialize_targets(critics)
        encoder = HistoryEncoder()
        log_alpha = torch.nn.Parameter(torch.tensor(-3.0))
        frozen = {name: {k: v.clone() for k, v in m.state_dict().items()} for name, m in
                  (("critics", critics), ("targets", targets), ("encoder", encoder))}
        actor_before = {k: v.clone() for k, v in actor.state_dict().items()}
        action, log_prob = actor.sample_action_and_log_prob(features)
        loss = compute_actor_loss(critics, q_obs, action, log_prob, alpha=log_alpha.exp().item())
        update_actor(actor, make_actor_optimizer(actor), loss)
        self.assertTrue(any(not torch.equal(v, actor_before[k]) for k, v in actor.state_dict().items()))
        for name, module in (("critics", critics), ("targets", targets), ("encoder", encoder)):
            for k, v in module.state_dict().items():
                self.assertTrue(torch.equal(v, frozen[name][k]), f"{name}.{k}")
        self.assertTrue(all(p.grad is None for p in critics.parameters()))
        self.assertIsNone(log_alpha.grad)
        self.assertTrue(all(p.requires_grad for p in critics.parameters()))

    def test_q_gradient_reaches_actor_without_entropy_term(self) -> None:
        # alpha=0 时损失只剩 −min(Q)，若 Q 对动作的梯度被切断，Actor 将得不到任何梯度。
        actor, features, q_obs = setup()
        action, log_prob = actor.sample_action_and_log_prob(features)
        loss = compute_actor_loss(TwinQ(6), q_obs, action, log_prob, alpha=0.0)
        loss.backward()
        self.assertGreater(actor.mean_head.weight.grad.abs().sum().item(), 0.0)

    def test_replayed_actions_differ_from_new_samples(self) -> None:
        actor, features, q_obs = setup()
        replayed = torch.rand(64, 8) * 2 - 1  # 回放中的旧动作（不参与 Actor 损失）
        action, log_prob = actor.sample_action_and_log_prob(features)
        loss = compute_actor_loss(TwinQ(6), q_obs, action, log_prob, alpha=0.05)
        self.assertFalse(torch.allclose(action, replayed))
        update_actor(actor, make_actor_optimizer(actor), loss)
        report = report_policy_change(actor, features, action.detach(), log_prob.detach())
        self.assertAlmostEqual(report["mean_std"], 0.15, delta=0.01)
        self.assertGreaterEqual(report["near_bound_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
