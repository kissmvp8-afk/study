"""N04 双 Q 评分网络的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/q_networks/test_critics.py -v

数据说明：Q 观测与动作为随机人工数据；C=54 / 241 只是 E02 README 中的预期值，用于检查维数校验，
不代表已确认的 TITA 观测组。参数独立性的检查使用真实的 N01 编码器与 N03 Actor 实例。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "networks_learning" / "history_encoder"))
sys.path.insert(0, str(ROOT / "networks_learning" / "actor_distribution"))
sys.path.insert(0, str(ROOT / "environment_control" / "action_bounds"))

from actor import SquashedGaussianActor  # noqa: E402
from critics import QNetwork, TwinQ, evaluate_q_pair, initialize_targets, list_critic_parameters  # noqa: E402
from encoder import HistoryEncoder  # noqa: E402
from bounds import ActionBounds  # noqa: E402

Q_DIM = 54


def storages(module: torch.nn.Module) -> set[int]:
    return {p.untyped_storage().data_ptr() for p in module.parameters()}


class EvaluationTest(unittest.TestCase):
    """动作参与评分；维数校验。"""

    def test_different_actions_give_different_scores(self) -> None:
        torch.manual_seed(0)
        critics = TwinQ(Q_DIM)
        q_obs = torch.randn(4, Q_DIM).repeat(2, 1)
        actions = torch.cat([torch.zeros(4, 8), torch.ones(4, 8)])
        q1, q2 = evaluate_q_pair(critics, q_obs, actions)
        self.assertEqual(q1.shape, (8, 1))
        self.assertFalse(torch.allclose(q1[:4], q1[4:]))
        self.assertFalse(torch.allclose(q2[:4], q2[4:]))

    def test_q_dim_is_checked(self) -> None:
        for c in (54, 241):  # E02 的平地 / 复杂地形预期值，仅作维数示例
            critics = TwinQ(c)
            self.assertEqual(evaluate_q_pair(critics, torch.zeros(2, c), torch.zeros(2, 8))[0].shape, (2, 1))
        with self.assertRaisesRegex(ValueError, "\\[B,54\\]"):
            QNetwork(54)(torch.zeros(2, 241), torch.zeros(2, 8))
        with self.assertRaises(ValueError):
            QNetwork(54)(torch.zeros(2, 54), torch.zeros(2, 7))


class IndependenceTest(unittest.TestCase):
    """两个 Q、目标副本、编码器与 Actor 之间不共享参数。"""

    def test_twins_are_independent(self) -> None:
        torch.manual_seed(1)
        critics = TwinQ(Q_DIM)
        self.assertFalse(torch.equal(critics.q1.net[0].weight, critics.q2.net[0].weight))
        self.assertTrue(storages(critics.q1).isdisjoint(storages(critics.q2)))
        q2_before = [p.clone() for p in critics.q2.parameters()]
        optimizer = torch.optim.SGD(critics.q1.parameters(), lr=0.1)
        critics.q1(torch.randn(8, Q_DIM), torch.randn(8, 8)).pow(2).mean().backward()
        optimizer.step()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(q2_before, critics.q2.parameters())))

    def test_targets_copy_values_not_storage(self) -> None:
        critics = TwinQ(Q_DIM)
        targets = initialize_targets(critics)
        for (name, online), (_, target) in zip(critics.named_parameters(), targets.named_parameters()):
            self.assertTrue(torch.equal(online, target), name)
            self.assertFalse(target.requires_grad)
        self.assertTrue(storages(critics).isdisjoint(storages(targets)))
        self.assertFalse(targets.training)
        with torch.no_grad():
            critics.q1.net[0].weight.add_(1.0)
        self.assertFalse(torch.equal(critics.q1.net[0].weight, targets.q1.net[0].weight))

    def test_parameter_listing_and_no_sharing_with_actor_encoder(self) -> None:
        critics = TwinQ(Q_DIM)
        listed = {id(p) for _, p in list_critic_parameters(critics)}
        self.assertEqual(listed, {id(p) for p in critics.parameters()})
        self.assertTrue(listed.isdisjoint({id(p) for p in initialize_targets(critics).parameters()}))
        bounds = ActionBounds(tuple(f"j{i}" for i in range(8)), (-1.0,) * 8, (1.0,) * 8, (0.0,) * 8, (1.0,) * 8)  # 人工边界
        others = [HistoryEncoder(), SquashedGaussianActor(bounds, [0.0] * 8)]
        for module in others:
            self.assertTrue(storages(critics).isdisjoint(storages(module)))


if __name__ == "__main__":
    unittest.main()
