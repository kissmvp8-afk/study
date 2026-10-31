"""N05 Q 的训练目标与更新的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/critic_learning/test_critic_update.py -v

数据说明：经历为人工数据（D02 测试的 make_records），经 D02 写入读取、D03 组合成 NStepBatch；
固定数值的假 Actor 与假目标 Q 用于手算 y；动作边界为人工对称区间 [-1, 1]。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for path in (
    "networks_learning/history_encoder",
    "networks_learning/actor_distribution",
    "networks_learning/q_networks",
    "environment_control/action_bounds",
    "experience_returns/replay_buffer",
    "experience_returns/n_step_returns",
    "experience_returns/normalization",
):
    sys.path.insert(0, str(ROOT / path))

from actor import SquashedGaussianActor  # noqa: E402
from bounds import ActionBounds  # noqa: E402
from buffer import ReplayBuffer  # noqa: E402
from critic_update import compute_target, critic_loss, make_critic_optimizer, update_critics  # noqa: E402
from critics import TwinQ, initialize_targets  # noqa: E402
from encoder import HistoryEncoder  # noqa: E402
from normalization import ObservationNormalizers  # noqa: E402
from returns import build_n_step_batch  # noqa: E402
from test_buffer import Q_DIM, make_records  # noqa: E402

GAMMA, N = 0.9, 3
ARTIFICIAL_BOUNDS = ActionBounds(tuple(f"j{i}" for i in range(8)), (-1.0,) * 8, (1.0,) * 8, (0.0,) * 8, (1.0,) * 8)


def three_kinds_of_batches():
    """环境 0：5 步未结束（k=3, mask=1）；环境 1：第 2 步真终止（k=2, mask=0）；环境 2：第 2 步纯截断（k=2, mask=1）。奖励全为 1。"""
    buffer = ReplayBuffer(num_envs=3, capacity_per_env=8, q_dim=Q_DIM)
    for step in range(5):
        entries, terminated, time_out = [(0, 0, step)], set(), set()
        if step < 2:
            entries += [(1, 0, step), (2, 0, step)]
            if step == 1:
                terminated, time_out = {1}, {2}
        buffer.append(make_records(entries, terminated=terminated, time_out=time_out, rewards={i: 1.0 for i in range(len(entries))}))
    sequence = buffer.read_sequence(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]), N)
    return build_n_step_batch(sequence, GAMMA, N)


class FakeActor:
    """固定输出：动作全 0，log π = -2；记录被调用时的行数。"""

    def __init__(self) -> None:
        self.rows: list[int] = []

    def sample_action_and_log_prob(self, features, generator=None):
        self.rows.append(features.shape[0])
        return torch.zeros(features.shape[0], 8), torch.full((features.shape[0], 1), -2.0)


class FakeTargets:
    """q1 恒为 v1，q2 恒为 v2。"""

    def __init__(self, v1: float, v2: float) -> None:
        self.q1 = lambda q, a: torch.full((q.shape[0], 1), v1)
        self.q2 = lambda q, a: torch.full((q.shape[0], 1), v2)


class TargetTest(unittest.TestCase):
    """固定评分器下的 y：正常、终止、截断、k<n。"""

    def test_targets_by_case(self) -> None:
        batch = three_kinds_of_batches()
        self.assertEqual(batch.steps.tolist(), [3, 2, 2])
        self.assertEqual(batch.bootstrap_mask.reshape(-1).tolist(), [1.0, 0.0, 1.0])
        alpha = 0.1
        actor = FakeActor()
        y = compute_target(batch, HistoryEncoder(), actor, FakeTargets(5.0, 7.0), ObservationNormalizers(Q_DIM), alpha)
        soft = 5.0 - alpha * (-2.0)  # min(5, 7) − alpha × log π = 5.2
        expected = [
            1 + 0.9 + 0.81 + 0.9**3 * soft,  # 正常，k=3
            1 + 0.9,  # 真终止，k=2：只有奖励
            1 + 0.9 + 0.9**2 * soft,  # 纯截断，k=2：保留尾部软价值
        ]
        for got, want in zip(y.reshape(-1).tolist(), expected):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(actor.rows, [2])  # 只为 2 个非终止样本计算尾部动作
        self.assertFalse(y.requires_grad)

    def test_terminal_branch_blocks_nan(self) -> None:
        batch = three_kinds_of_batches()
        y = compute_target(batch, HistoryEncoder(), FakeActor(), FakeTargets(float("nan"), float("nan")), ObservationNormalizers(Q_DIM), 0.1)
        self.assertAlmostEqual(y[1, 0].item(), 1.9, places=5)  # 真终止样本不受尾部 NaN 影响
        self.assertTrue(bool(torch.isnan(y[[0, 2]]).all()))  # 对照：非终止样本确实读取了尾部值


class UpdateTest(unittest.TestCase):
    """一次更新只改在线 Q，损失下降。"""

    def test_update_changes_only_online_q(self) -> None:
        torch.manual_seed(0)
        batch = three_kinds_of_batches()
        encoder, critics = HistoryEncoder(), TwinQ(Q_DIM)
        actor = SquashedGaussianActor(ARTIFICIAL_BOUNDS, [0.0] * 8)
        targets = initialize_targets(critics)
        normalizers = ObservationNormalizers(Q_DIM)
        # 人工观测编码数值很大（约 2e4），先像采集时一样用新数据更新 D04 统计，否则 Q 输入未经缩放。
        normalizers.update_from_fresh_data(batch.first)
        normalizers.update_from_fresh_data(batch.tail)
        frozen = [m.state_dict() for m in (encoder, actor, targets)]
        frozen = [{k: v.clone() for k, v in d.items()} for d in frozen]
        online_before = {k: v.clone() for k, v in critics.state_dict().items()}
        y = compute_target(batch, encoder, actor, targets, normalizers, alpha=0.05, generator=torch.Generator().manual_seed(1))
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        optimizer = make_critic_optimizer(critics)
        before = critic_loss(critics, batch, y, normalizers)[0].item()
        report = update_critics(critics, optimizer, batch, y, normalizers)
        after = critic_loss(critics, batch, y, normalizers)[0].item()
        self.assertAlmostEqual(report["critic_loss"], before, places=5)
        self.assertLess(after, before)
        self.assertTrue(any(not torch.equal(v, online_before[k]) for k, v in critics.state_dict().items()))
        for module, state in zip((encoder, actor, targets), frozen):
            for k, v in module.state_dict().items():
                self.assertTrue(torch.equal(v, state[k]), k)


if __name__ == "__main__":
    unittest.main()
