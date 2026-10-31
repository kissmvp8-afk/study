"""D04 观测归一化与统计管理的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/normalization/test_normalization.py -v

数据说明：全部为人工数据。EpisodeData 构造“帧值等于时间步”的一轮：第 t 帧的 33 个数都等于 t，
重置填充帧填 -999（若被计入统计会明显拉偏均值），且第 s 条记录的下一历史正好是第 s+1 条的当前历史，
与真实采集的时间关系一致。期望统计在注释中手算。
不覆盖：真实观测尺度、GPU、与网络 BatchNorm 的交互。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "replay_buffer"))
sys.path.insert(0, str(ROOT / "n_step_returns"))
sys.path.insert(0, str(ROOT / "transition_schema"))

from buffer import ReplayBuffer  # noqa: E402
from normalization import (  # noqa: E402
    NORMALIZER_EPS,
    NormalizationError,
    ObservationNormalizers,
    RunningNormalizer,
    select_fresh_samples,
)
from returns import build_n_step_batch  # noqa: E402
from schema import make_transition  # noqa: E402
from test_buffer import make_records  # noqa: E402

Q_DIM = 4
PAD = -999.0


def history_at(t: int, offset: float = 0.0) -> torch.Tensor:
    """第 t 帧时的 [10,33] 历史：真实帧 max(0,t-9)..t 的值等于帧号（加 offset），其余为填充值 PAD。"""
    frames = torch.full((10,), PAD)
    real = torch.arange(max(0, t - 9), t + 1, dtype=torch.float32) + offset
    frames[10 - real.numel():] = real
    return frames.reshape(10, 1).expand(10, 33).clone()


def episode_batch(env: int, episode: int, steps: int, offset: float = 0.0, terminated_last: bool = False):
    """一轮 steps 次动作的全部记录（一批），帧值 = 时间步 + offset；Q 观测 = [t+offset, 2(t+offset), 常数 5, 0]。"""
    rows = []
    for s in range(steps):
        rows.append((s, history_at(s, offset), history_at(s + 1, offset)))
    b = steps
    q = lambda t: torch.tensor([t + offset, 2 * (t + offset), 5.0, 0.0])  # noqa: E731
    terminated = torch.zeros(b, 1, dtype=torch.bool)
    terminated[-1] = terminated_last
    return make_transition(
        {
            "policy_history": torch.stack([r[1] for r in rows]),
            "next_policy_history": torch.stack([r[2] for r in rows]),
            "q_observation": torch.stack([q(s) for s in range(b)]),
            "next_q_observation": torch.stack([q(s + 1) for s in range(b)]),
            "action": torch.zeros(b, 8),
            "reward": torch.zeros(b, 1),
            "terminated": terminated,
            "time_out": torch.zeros(b, 1, dtype=torch.bool),
            "env_id": torch.full((b,), env, dtype=torch.int64),
            "episode_id": torch.full((b,), episode, dtype=torch.int64),
            "step_index": torch.arange(b, dtype=torch.int64),
            "history_length": torch.clamp(torch.arange(b) + 1, max=10),
            "next_history_length": torch.clamp(torch.arange(b) + 2, max=10),
        },
        Q_DIM,
    )


class RunningNormalizerTest(unittest.TestCase):
    """统计值、合并、只读使用、保存恢复与数值稳定。"""

    def test_known_statistics_and_batch_merge(self) -> None:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 10.0]])
        # 手算：均值 [4, 5.5]；总体方差 [(9+1+1+9)/4, (12.25+2.25+0.25+20.25)/4] = [5, 8.75]。
        whole = RunningNormalizer(2)
        whole.update_from_fresh_data(x)
        split = RunningNormalizer(2)
        split.update_from_fresh_data(x[:1])
        split.update_from_fresh_data(x[1:])
        for norm in (whole, split):
            self.assertTrue(torch.allclose(norm.mean, torch.tensor([4.0, 5.5], dtype=torch.float64)))
            self.assertTrue(torch.allclose(norm.var, torch.tensor([5.0, 8.75], dtype=torch.float64)))
            self.assertEqual(norm.count, 4)
        y = whole.normalize(torch.tensor([[4.0 + 5.0**0.5, 5.5]]))
        self.assertAlmostEqual(y[0, 0].item(), 5.0**0.5 / (5.0**0.5 + NORMALIZER_EPS), places=5)
        self.assertEqual(y[0, 1].item(), 0.0)

    def test_normalize_does_not_change_statistics_or_input(self) -> None:
        norm = RunningNormalizer(3)
        norm.update_from_fresh_data(torch.randn(50, 3, generator=torch.Generator().manual_seed(0)))
        before = norm.export_statistics()
        x = torch.randn(8, 3, generator=torch.Generator().manual_seed(1))
        x_copy = x.clone()
        outputs = [norm.normalize(x) for _ in range(5)]
        after = norm.export_statistics()
        self.assertEqual(after["count"], before["count"])
        self.assertTrue(torch.equal(after["mean"], before["mean"]) and torch.equal(after["var"], before["var"]))
        self.assertTrue(torch.equal(x, x_copy))
        self.assertTrue(all(torch.equal(o, outputs[0]) for o in outputs))

    def test_export_and_restore(self) -> None:
        norm = RunningNormalizer(3, eps=1e-3)
        norm.update_from_fresh_data(torch.randn(40, 3, generator=torch.Generator().manual_seed(2)))
        state = norm.export_statistics()
        restored = RunningNormalizer(3)
        restored.load_statistics(state)
        x = torch.randn(6, 3, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(restored.normalize(x), norm.normalize(x)))
        self.assertEqual((restored.count, restored.eps), (40, 1e-3))
        state["mean"].fill_(100.0)  # 导出结果是副本，修改它不影响原统计
        self.assertTrue(torch.equal(restored.normalize(x), norm.normalize(x)))
        with self.assertRaises(NormalizationError):
            RunningNormalizer(4).load_statistics(norm.export_statistics())

    def test_constant_tiny_variance_and_extreme_inputs(self) -> None:
        norm = RunningNormalizer(3)
        data = torch.stack([torch.full((20,), 5.0), 1.0 + 1e-9 * torch.arange(20.0), torch.linspace(-1, 1, 20)], dim=1)
        norm.update_from_fresh_data(data)
        y = norm.normalize(torch.tensor([[5.0, 1.0, 0.0]]))
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertEqual(y[0, 0].item(), 0.0)  # 常数通道：(5-5)/(0+eps) = 0
        self.assertTrue(bool(torch.isfinite(norm.normalize(torch.tensor([[1e30, -1e30, 1e30]]))).all()))
        self.assertEqual(norm.export_statistics()["eps"], NORMALIZER_EPS)

    def test_rejects_bad_updates(self) -> None:
        norm = RunningNormalizer(2)
        with self.assertRaisesRegex(NormalizationError, "NaN"):
            norm.update_from_fresh_data(torch.tensor([[1.0, float("nan")]]))
        with self.assertRaisesRegex(NormalizationError, "形状"):
            norm.update_from_fresh_data(torch.zeros(3, 5))
        self.assertEqual(norm.count, 0)


class FreshDataRuleTest(unittest.TestCase):
    """哪些新数据计入统计：每个真实状态一次，不计重叠帧与填充帧。"""

    def test_each_real_frame_counted_once(self) -> None:
        # 12 次动作：真实帧 0..12（第 0 帧来自首条记录的当前观测，1..12 来自每条记录的真实下一观测）。
        # 13 个值的均值 6，总体方差 (13^2-1)/12 = 14；填充值 -999 和重叠的旧帧都不应出现。
        batch = episode_batch(env=0, episode=0, steps=12)
        policy, q = select_fresh_samples(batch)
        self.assertEqual(sorted(policy[:, 0].tolist()), [float(t) for t in range(13)])
        normalizers = ObservationNormalizers(Q_DIM)
        normalizers.update_from_fresh_data(batch)
        self.assertEqual(normalizers.policy.count, 13)
        self.assertTrue(torch.allclose(normalizers.policy.mean, torch.full((33,), 6.0, dtype=torch.float64)))
        self.assertTrue(torch.allclose(normalizers.policy.var, torch.full((33,), 14.0, dtype=torch.float64)))
        self.assertEqual(normalizers.q.count, 13)
        self.assertTrue(torch.allclose(normalizers.q.mean, torch.tensor([6.0, 12.0, 5.0, 0.0], dtype=torch.float64)))

    def test_terminal_and_reset_states_each_counted_once(self) -> None:
        # 第 0 轮 3 步后真终止：帧 0..3（帧 3 为重置前末状态）；第 1 轮从帧值 100 开始（重置后观测）跑 2 步：帧 100..102。
        normalizers = ObservationNormalizers(Q_DIM)
        normalizers.update_from_fresh_data(episode_batch(0, 0, 3, terminated_last=True))
        normalizers.update_from_fresh_data(episode_batch(0, 1, 2, offset=100.0))
        expected = torch.tensor([0, 1, 2, 3, 100, 101, 102], dtype=torch.float64)
        self.assertEqual(normalizers.policy.count, 7)
        self.assertAlmostEqual(normalizers.policy.mean[0].item(), expected.mean().item(), places=9)
        self.assertAlmostEqual(normalizers.policy.var[0].item(), expected.var(unbiased=False).item(), places=6)


class ReplayIntegrationTest(unittest.TestCase):
    """D01/D02/D03 读出的观测按约定归一化，回放不增加计数，原始记录不变。"""

    def test_replayed_observations(self) -> None:
        buffer = ReplayBuffer(num_envs=2, capacity_per_env=8, q_dim=6)
        normalizers = ObservationNormalizers(6)
        for step in range(6):
            fresh = make_records([(0, 0, step), (1, 2, step)], terminated={0} if step == 5 else set())
            normalizers.update_from_fresh_data(fresh)  # 每批新数据只更新一次
            buffer.append(fresh)
        counts = (normalizers.policy.count, normalizers.q.count)
        env_ids, slots = buffer.sample_starts(32, 2, generator=torch.Generator().manual_seed(5))
        for _ in range(3):  # 重复回放同一批
            batch = build_n_step_batch(buffer.read_sequence(env_ids, slots, 2), 0.99, 2)
            raw_first = batch.first.policy_history.clone()
            raw_tail_q = batch.tail.next_q_observation.clone()
            current = normalizers.normalize_policy(batch.first.policy_history)
            target_q = normalizers.normalize_q(batch.tail.next_q_observation)
            online_q = normalizers.normalize_q(batch.first.q_observation)
            self.assertEqual(current.shape, (32, 10, 33))
            self.assertEqual(target_q.shape, online_q.shape)
            self.assertTrue(torch.equal(batch.first.policy_history, raw_first))
            self.assertTrue(torch.equal(batch.tail.next_q_observation, raw_tail_q))
        self.assertEqual((normalizers.policy.count, normalizers.q.count), counts)
        # 池内原始记录不受影响。
        again = buffer.read_sequence(env_ids, slots, 2)
        self.assertTrue(torch.equal(again.fields["policy_history"][:, 0], batch.first.policy_history))


if __name__ == "__main__":
    unittest.main()
