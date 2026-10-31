"""D05 历史窗口与有效性的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/history_windows/test_history.py -v

数据说明：全部为人工数据。帧编号历史中第 i 帧的 33 个数都等于 i（0..9），便于直接读出窗口包含哪些时刻；
衔接检查中的记录来自 D02 测试的 make_records，并经 D02 写入读取、D03 组合、D04 归一化。
原 PPO 切片写法按 baseline/ppo 快照 actor_critic.py 复写，作为独立对照。
不覆盖：T05 导出入口（尚未实现）、N01/N02 网络。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
for module in ("replay_buffer", "n_step_returns", "normalization", "transition_schema"):
    sys.path.insert(0, str(ROOT / module))

from buffer import ReplayBuffer  # noqa: E402
from history import (  # noqa: E402
    AUXILIARY_REQUIRED_FRAMES,
    HistoryWindowError,
    split_windows,
    valid_history_mask,
    validate_time_order,
)
from normalization import ObservationNormalizers  # noqa: E402
from returns import build_n_step_batch  # noqa: E402
from test_buffer import make_records  # noqa: E402


def numbered_history(batch_size: int = 2) -> torch.Tensor:
    """[B,10,33]，第 i 帧全部等于 i。"""
    return torch.arange(10, dtype=torch.float32).reshape(1, 10, 1).expand(batch_size, 10, 33).clone()


def frame_ids(window: torch.Tensor) -> list[int]:
    """读出窗口中每一帧的编号（取第 0 个样本、第 0 个通道）。"""
    return [int(v) for v in window[0, ..., 0].reshape(-1).tolist()]


class SplitWindowsTest(unittest.TestCase):
    """切片位置、与原 PPO 一致、不推进历史。"""

    def test_frames_0_to_9(self) -> None:
        windows = split_windows(numbered_history())
        self.assertEqual(int(windows.current[0, 0]), 9)
        self.assertEqual(frame_ids(windows.recent), [5, 6, 7, 8, 9])
        self.assertEqual(frame_ids(windows.previous), [4, 5, 6, 7, 8])
        self.assertEqual(windows.current.shape, (2, 33))
        self.assertEqual(windows.recent.shape, (2, 5, 33))

    def test_matches_ppo_training_split(self) -> None:
        obs = torch.randn(4, 10, 33, generator=torch.Generator().manual_seed(0))
        current = obs[:, -1, :]  # 原 PPO _split_policy_obs
        hist = obs[:, -5 - 1 : -1, :]
        full = torch.cat([hist[:, 1:, :], current.unsqueeze(1)], dim=1)  # 原 BarlowTwinsActor.forward 的决策窗口
        windows = split_windows(obs)
        self.assertTrue(torch.equal(windows.current, current))
        self.assertTrue(torch.equal(windows.previous, hist))
        self.assertTrue(torch.equal(windows.recent, full))

    def test_ppo_export_wrapper_on_full_history_repeats_current(self) -> None:
        # 回归记录（供 T05）：原 _InferenceWrapper 把 history[:, 1:] 与 current 拼接后取最后 5 帧。
        # 若传入的 history 已包含当前帧（环境历史正是如此），窗口变成 6,7,8,9,9，当前帧出现两次。
        history = numbered_history(1)
        current = history[:, -1]
        full = torch.cat([history[:, 1:], current.unsqueeze(1)], dim=1)
        self.assertEqual(frame_ids(full[:, -5:]), [6, 7, 8, 9, 9])
        self.assertNotEqual(frame_ids(full[:, -5:]), frame_ids(split_windows(history).recent))

    def test_does_not_advance_or_copy_history(self) -> None:
        history = numbered_history()
        before = history.clone()
        first = split_windows(history)
        second = split_windows(history)
        self.assertTrue(torch.equal(history, before))
        self.assertTrue(torch.equal(first.recent, second.recent))
        self.assertEqual(first.current.data_ptr(), history[:, 9].data_ptr())  # 视图，不另存一份

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(HistoryWindowError, "\\[B,10,33\\]"):
            split_windows(torch.zeros(2, 9, 33))


class ValidityTest(unittest.TestCase):
    """有效性只看有效历史长度。"""

    def test_auxiliary_needs_six_real_frames(self) -> None:
        lengths = torch.tensor([1, 5, 6, 10])
        self.assertEqual(AUXILIARY_REQUIRED_FRAMES, 6)
        self.assertEqual(valid_history_mask(lengths).tolist(), [False, False, True, True])
        self.assertEqual(valid_history_mask(lengths, required_frames=5).tolist(), [False, True, True, True])

    def test_does_not_use_observation_values(self) -> None:
        # 真实观测全为零但长度 10：有效；填充为非零但长度 3：无效。函数不接收观测，也就无法按数值判断。
        real_zeros_length = torch.tensor([10])
        padded_nonzero_length = torch.tensor([3])
        self.assertTrue(bool(valid_history_mask(real_zeros_length)))
        self.assertFalse(bool(valid_history_mask(padded_nonzero_length)))

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(HistoryWindowError):
            valid_history_mask(torch.tensor([0]))
        with self.assertRaises(HistoryWindowError):
            valid_history_mask(torch.tensor([3]), required_frames=11)


class TimeOrderTest(unittest.TestCase):
    """用显式编号检查重复与跨轮次。"""

    def test_consecutive_passes(self) -> None:
        validate_time_order(torch.tensor([[2, 2, 2]]), torch.tensor([[5, 5, 5]]), torch.tensor([[7, 8, 9]]))
        validate_time_order(torch.tensor([[1]]), torch.tensor([[0]]), torch.tensor([[3]]))

    def test_detects_problems(self) -> None:
        env = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]])
        episode = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]])
        step = torch.tensor([[0, 1, 1], [0, 2, 3], [4, 5, 6], [0, 1, 2]])
        with self.assertRaises(HistoryWindowError) as caught:
            validate_time_order(env, episode, step)
        message = str(caught.exception)
        self.assertIn("跨环境：行 [3]", message)
        self.assertIn("跨轮次：行 [2]", message)
        self.assertIn("重复或跳跃）：行 [0, 1]", message)


class ReplayIntegrationTest(unittest.TestCase):
    """D01/D02/D03 读出的当前与下一观测，经 D04、D05 处理，原始记录不变。"""

    def test_windows_on_replayed_batch(self) -> None:
        buffer = ReplayBuffer(num_envs=2, capacity_per_env=12, q_dim=6)
        normalizers = ObservationNormalizers(6)
        for step in range(9):
            fresh = make_records([(0, 0, step), (1, 3, step)], time_out={1} if step == 8 else set())
            normalizers.update_from_fresh_data(fresh)
            buffer.append(fresh)
        env_ids, slots = buffer.sample_starts(40, 3, generator=torch.Generator().manual_seed(11))
        sequence = buffer.read_sequence(env_ids, slots, 3)
        for m in range(40):  # 有效前缀的编号必须首尾相接
            k = int(sequence.valid[m].sum())
            validate_time_order(*(sequence.fields[n][m : m + 1, :k] for n in ("env_id", "episode_id", "step_index")))
        batch = build_n_step_batch(sequence, 0.99, 3)
        raw = batch.first.policy_history.clone()
        normalized = normalizers.normalize_policy(batch.first.policy_history)
        windows = split_windows(normalized)
        # 归一化是逐帧变换，先归一化再切片与先切片再归一化结果相同。
        self.assertTrue(torch.equal(windows.recent, normalizers.normalize_policy(split_windows(raw).recent)))
        self.assertTrue(torch.equal(batch.first.policy_history, raw))
        # 辅助有效标记：当前历史用 history_length，尾部下一历史用 next_history_length（人工数据中 = min(step+1,10)）。
        current_valid = valid_history_mask(batch.first.history_length)
        tail_valid = valid_history_mask(batch.tail.next_history_length)
        self.assertTrue(torch.equal(current_valid, batch.first.step_index + 1 >= 6))
        self.assertTrue(torch.equal(tail_valid, batch.tail.step_index + 2 >= 6))
        self.assertEqual(split_windows(batch.tail.next_policy_history).previous.shape, (40, 5, 33))


if __name__ == "__main__":
    unittest.main()
