"""D03 连续多步回报的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/n_step_returns/test_returns.py -v

数据说明：全部为人工数据，经 D01 make_transition 创建、D02 ReplayBuffer 写入与读取（衔接检查）。
记录构造复用 D02 测试中的 make_records（观测编码 code = env*10000 + episode*100 + step）。
期望的折扣和在注释中手算写出，不调用被测函数自证。
不覆盖：N05 目标 Q 计算、多步目标的离策略偏差。
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
sys.path.insert(0, str(ROOT / "transition_schema"))

from buffer import ReplayBuffer  # noqa: E402
from returns import NStepReturnError, accumulate_rewards, build_n_step_batch  # noqa: E402
from test_buffer import Q_DIM, code_of, make_records  # noqa: E402

HUGE = 1.0e6  # 下一轮放置的大额奖励，若被计入上一轮会非常明显


def buffer_with_episode(rewards: list[float], terminal_at: int | None = None, time_out_at: int | None = None,
                        next_episode_steps: int = 0) -> ReplayBuffer:
    """在环境 0 写入第 0 轮（给定奖励与结束位置），随后写入第 1 轮若干步，其奖励全部为 HUGE。"""
    buffer = ReplayBuffer(num_envs=1, capacity_per_env=16, q_dim=Q_DIM)
    for step, reward in enumerate(rewards):
        buffer.append(
            make_records(
                [(0, 0, step)],
                terminated={0} if step == terminal_at else set(),
                time_out={0} if step == time_out_at else set(),
                rewards={0: reward},
            )
        )
    for step in range(next_episode_steps):
        buffer.append(make_records([(0, 1, step)], rewards={0: HUGE}))
    return buffer


def n_step_from_episode_start(buffer: ReplayBuffer, n: int, gamma: float):
    """从第 0 轮 step 0 起读取 n 条并构造样本（该起点须是 D02 的有效起点）。"""
    env_ids, slots = buffer.valid_starts(n)
    starts = buffer.read_sequence(env_ids, slots, 1)
    pick = (starts.fields["episode_id"][:, 0] == 0) & (starts.fields["step_index"][:, 0] == 0)
    assert int(pick.sum()) == 1, "第 0 轮 step 0 应是有效起点"
    return build_n_step_batch(buffer.read_sequence(env_ids[pick], slots[pick], n), gamma, n)


class AccumulateTest(unittest.TestCase):
    """无边界序列的折扣和与 n=1 退化。"""

    def test_known_sequence_without_boundary(self) -> None:
        # 1 + 0.9*2 + 0.81*3 + 0.729*4 + 0.6561*5 = 11.4265；k=5，gamma^5 = 0.59049。
        batch = n_step_from_episode_start(buffer_with_episode([1, 2, 3, 4, 5, 6]), n=5, gamma=0.9)
        self.assertAlmostEqual(batch.reward_sum.item(), 11.4265, places=4)
        self.assertEqual(batch.steps.tolist(), [5])
        self.assertAlmostEqual(batch.discount.item(), 0.59049, places=6)
        self.assertEqual(batch.bootstrap_mask.item(), 1.0)
        self.assertEqual(int(batch.first.step_index), 0)
        self.assertEqual(int(batch.tail.step_index), 4)  # 尾部是第 5 次动作所在记录
        self.assertEqual(batch.tail.next_q_observation[0, 0].item(), code_of(0, 0, 4) + 0.5)

    def test_n_equals_one_is_single_step(self) -> None:
        batch = n_step_from_episode_start(buffer_with_episode([7, 8]), n=1, gamma=0.99)
        self.assertEqual(batch.reward_sum.item(), 7.0)
        self.assertEqual(batch.steps.tolist(), [1])
        self.assertAlmostEqual(batch.discount.item(), 0.99, places=6)
        self.assertEqual(int(batch.tail.step_index), 0)

    def test_invalid_gamma(self) -> None:
        with self.assertRaises(NStepReturnError):
            accumulate_rewards(torch.zeros(1, 2, 1), torch.zeros(1, 2, dtype=torch.bool), 0.0)


class BoundaryTest(unittest.TestCase):
    """在第 1、2、n 步放置真终止或纯截断；下一轮的大额奖励不得进入上一轮。"""

    def test_boundaries_at_steps_1_2_n(self) -> None:
        n, gamma = 5, 0.5
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        for position in (1, 2, n):  # 第 position 步结束，即 step_index = position-1
            for kind in ("terminated", "time_out"):
                with self.subTest(position=position, kind=kind):
                    buffer = buffer_with_episode(
                        rewards[:position],
                        terminal_at=position - 1 if kind == "terminated" else None,
                        time_out_at=position - 1 if kind == "time_out" else None,
                        next_episode_steps=n,
                    )
                    batch = n_step_from_episode_start(buffer, n, gamma)
                    expected = sum(gamma**j for j in range(position))  # 1 + 0.5 + ... 共 position 项
                    self.assertAlmostEqual(batch.reward_sum.item(), expected, places=6)
                    self.assertLess(batch.reward_sum.item(), HUGE)
                    self.assertEqual(batch.steps.tolist(), [position])
                    self.assertAlmostEqual(batch.discount.item(), gamma**position, places=6)
                    self.assertEqual(batch.bootstrap_mask.item(), 0.0 if kind == "terminated" else 1.0)
                    # 尾部仍是第 0 轮的末条记录，next_* 为重置前的真实状态（code+0.5），不是下一轮的观测。
                    self.assertEqual(int(batch.tail.episode_id), 0)
                    self.assertEqual(int(batch.tail.step_index), position - 1)
                    self.assertEqual(batch.tail.next_policy_history[0, -1, 0].item(), code_of(0, 0, position - 1) + 0.5)


class MissingSuccessorTest(unittest.TestCase):
    """尚缺后继的起点不可采样，也不能当作结束。"""

    def test_start_without_enough_successors(self) -> None:
        buffer = buffer_with_episode([1, 2, 3])  # 未结束，只有 3 步
        env_ids, slots = buffer.valid_starts(5)
        self.assertEqual(env_ids.numel(), 0)
        sequence = buffer.read_sequence(torch.tensor([0]), torch.tensor([0]), 5)
        with self.assertRaisesRegex(NStepReturnError, "数据不足不能当作结束"):
            build_n_step_batch(sequence, 0.9, 5)
        # 后继写入之后即可使用。
        for step in range(3, 5):
            buffer.append(make_records([(0, 0, step)], rewards={0: 1.0}))
        self.assertEqual(buffer.valid_starts(5)[0].numel(), 1)

    def test_length_must_match_n(self) -> None:
        buffer = buffer_with_episode([1, 2, 3])
        with self.assertRaisesRegex(NStepReturnError, "不一致"):
            build_n_step_batch(buffer.read_sequence(torch.tensor([0]), torch.tensor([0]), 2), 0.9, 3)


class SampledBatchTest(unittest.TestCase):
    """D02 → D03：抽样片段全部可构造，且与手算一致。"""

    def test_sampled_starts_from_mixed_buffer(self) -> None:
        n, gamma = 3, 0.9
        buffer = ReplayBuffer(num_envs=2, capacity_per_env=6, q_dim=Q_DIM)
        for step in range(14):
            # 环境 0：每 4 步一轮，第 3 步真终止；环境 1：一直不结束。奖励 = 1，下一轮第 0 步奖励为 HUGE。
            buffer.append(
                make_records(
                    [(0, step // 4, step % 4), (1, 0, step)],
                    terminated={0} if step % 4 == 3 else set(),
                    rewards={0: HUGE if step % 4 == 0 and step > 0 else 1.0, 1: 1.0},
                )
            )
        env_ids, slots = buffer.sample_starts(50, n, generator=torch.Generator().manual_seed(7))
        batch = build_n_step_batch(buffer.read_sequence(env_ids, slots, n), gamma, n)
        for m in range(50):
            k = int(batch.steps[m])
            first_reward = HUGE if int(batch.first.env_id[m]) == 0 and int(batch.first.step_index[m]) == 0 and int(batch.first.episode_id[m]) > 0 else 1.0
            expected = first_reward + sum(gamma**j for j in range(1, k))
            # 奖励按 float32 存储：1e6 附近的最小间隔约 0.06，因此容差按数值大小取相对值。
            self.assertAlmostEqual(batch.reward_sum[m].item(), expected, delta=max(1e-4, abs(expected) * 1e-6))
            self.assertEqual(int(batch.tail.episode_id[m]), int(batch.first.episode_id[m]))
            self.assertEqual(int(batch.tail.step_index[m]), int(batch.first.step_index[m]) + k - 1)
            self.assertEqual(batch.bootstrap_mask[m].item(), 0.0 if bool(batch.tail.terminated[m]) else 1.0)


if __name__ == "__main__":
    unittest.main()
