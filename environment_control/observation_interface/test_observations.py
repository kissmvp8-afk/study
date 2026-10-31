"""E02 观测接口的验证程序（unittest + PyTorch CPU，无需 Isaac Lab）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/observation_interface/test_observations.py -v

依据：
    - 字段常量与固定快照 8fe40aa 的 rough_env_cfg.py ObservationsCfg 逐项比对（需要本地有 baseline/ppo 快照）；
    - 历史推进规则与 Isaac Lab 2.3.0 utils/buffers/circular_buffer.py 的 append / reset / buffer 行为比对：
      本文件的 ReferenceCircularBuffer 按该源码的算法复写（Isaac Lab，BSD-3-Clause），作为独立对照。
运行时观测（group_obs_term_dim、实际数值）需要在 Isaac 环境中核对，这里用人工构造的张量。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from observations import (  # noqa: E402
    CRITIC_FIELDS,
    GROUP_FIELDS,
    POLICY_FIELDS,
    ObservationError,
    PolicyHistory,
    baseline_observation_config,
    describe_observation_fields,
    expected_q_dim,
    push_frame,
    read_observation_groups,
    valid_length,
    validate_observations,
    velocity_label,
    velocity_label_to_mps,
)


class ReferenceCircularBuffer:
    """Isaac Lab 2.3.0 CircularBuffer 的算法复写（只保留 append / reset / buffer）。"""

    def __init__(self, max_len: int, batch_size: int) -> None:
        self.max_length, self.batch_size = max_len, batch_size
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long)
        self._pointer, self._buffer = -1, None

    def reset(self, batch_ids) -> None:
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor) -> None:
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype)
        self._pointer = (self._pointer + 1) % self.max_length
        self._buffer[self._pointer] = data
        first = self._num_pushes == 0
        if torch.any(first):
            self._buffer[:, first] = data[first]
        self._num_pushes += 1

    @property
    def buffer(self) -> torch.Tensor:
        buf = torch.roll(self._buffer.clone(), shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, 0, 1)


def runtime_terms(task: str):
    """模拟 Isaac observation_manager.active_terms / group_obs_term_dim 的正确返回（人工构造）。"""
    groups = ("policy", "critic", "priv") + (("scanner",) if task == "isaac_rough" else ())
    return ({g: [f.name for f in GROUP_FIELDS[g]] for g in groups}, {g: [(f.dim,) for f in GROUP_FIELDS[g]] for g in groups})


class FieldContractTest(unittest.TestCase):
    """字段名称、顺序、缩放、噪声与原配置一致；Actor 不含真实线速度。"""

    def test_matches_baseline_config(self) -> None:
        parsed = baseline_observation_config()
        for group, fields in GROUP_FIELDS.items():
            with self.subTest(group=group):
                expected = [{"name": f.name, "func": f.func, "scale": f.scale, "noisy": f.noisy} for f in fields]
                self.assertEqual(parsed[group], expected)

    def test_dimensions_and_actor_inputs(self) -> None:
        self.assertEqual(sum(f.dim for f in POLICY_FIELDS), 33)
        self.assertEqual((expected_q_dim("isaac_flat"), expected_q_dim("isaac_rough")), (54, 241))
        self.assertNotIn("base_lin_vel", [f.name for f in POLICY_FIELDS])
        rows = {(r["target"], r["field"]): r for r in describe_observation_fields("isaac_rough")}
        self.assertEqual(rows[("q_observation", "base_lin_vel")]["columns"], (0, 3))
        self.assertEqual(rows[("q_observation", "base_lin_vel")]["scale"], 2.0)
        self.assertEqual(rows[("q_observation", "height_scan")]["columns"], (54, 241))
        self.assertNotIn(("q_observation", "height_scan"), {(r["target"], r["field"]) for r in describe_observation_fields("isaac_flat")})
        self.assertEqual(CRITIC_FIELDS[0].name, "base_lin_vel")

    def test_runtime_validation(self) -> None:
        for task in ("isaac_flat", "isaac_rough"):
            names, dims = runtime_terms(task)
            self.assertEqual(validate_observations(task, names, dims, 33), expected_q_dim(task))
        names, dims = runtime_terms("isaac_flat")
        swapped = dict(names, policy=[names["policy"][1], names["policy"][0], *names["policy"][2:]])  # 两个 3 维字段调换
        with self.assertRaisesRegex(ObservationError, "观测组 policy 字段不一致"):
            validate_observations("isaac_flat", swapped, dims)
        wrong_dim = dict(dims, priv=[(4,), (8,), (8,)])  # 例如接触体被解析成 4 个
        with self.assertRaisesRegex(ObservationError, "观测组 priv"):
            validate_observations("isaac_flat", names, wrong_dim)
        with_history = dict(dims, policy=[(10, d[0]) for d in dims["policy"]])  # 误用 Isaac 历史
        with self.assertRaisesRegex(ObservationError, "单帧"):
            validate_observations("isaac_flat", names, with_history)


class GroupReadingTest(unittest.TestCase):
    """Q 观测拼接顺序、维数校验与速度标签。"""

    def test_concatenation_and_no_padding(self) -> None:
        b = 4
        obs = {"policy": torch.randn(b, 33), "critic": torch.full((b, 36), 1.0), "priv": torch.full((b, 18), 2.0),
               "scanner": torch.full((b, 187), 3.0)}
        frame, q = read_observation_groups(obs, "isaac_rough", 241)
        self.assertTrue(torch.equal(frame, obs["policy"]))
        self.assertEqual((q[:, :36].unique().item(), q[:, 36:54].unique().item(), q[:, 54:].unique().item()), (1.0, 2.0, 3.0))
        flat_frame, flat_q = read_observation_groups(obs, "isaac_flat", 54)  # 平地不拼 scanner
        self.assertEqual(flat_q.shape, (b, 54))
        with self.assertRaisesRegex(ObservationError, "\\[4,54\\]"):
            read_observation_groups(obs, "isaac_rough", 54)  # 241 维不会被截断成 54
        with self.assertRaisesRegex(ObservationError, "Isaac 历史长度 0"):
            read_observation_groups(dict(obs, policy=torch.randn(b, 10, 33)), "isaac_flat", 54)
        bad = dict(obs, critic=obs["critic"].clone())
        bad["critic"][0, 0] = float("nan")
        with self.assertRaisesRegex(ObservationError, "NaN"):
            read_observation_groups(bad, "isaac_flat", 54)

    def test_velocity_label(self) -> None:
        q = torch.zeros(2, 54)
        q[:, :3] = torch.tensor([[2.0, -1.0, 0.4], [0.0, 0.0, 0.0]])  # base_lin_vel × 2.0
        self.assertTrue(torch.equal(velocity_label(q), q[:, :3]))
        self.assertTrue(torch.allclose(velocity_label_to_mps(velocity_label(q))[0], torch.tensor([1.0, -0.5, 0.2])))


class HistoryTest(unittest.TestCase):
    """历史推进与 Isaac CircularBuffer 一致：从旧到新、重置后首帧填满、每步只推进一次。"""

    def test_matches_reference_buffer_with_partial_resets(self) -> None:
        b = 3
        reference = ReferenceCircularBuffer(10, b)
        history = PolicyHistory(b)
        generator = torch.Generator().manual_seed(0)
        for step in range(25):
            if step in (4, 13):
                ids = torch.tensor([1]) if step == 4 else torch.tensor([0, 2])
                reference.reset(ids)
                history.reset(ids)
            frame = torch.randn(b, 33, generator=generator)
            reference.append(frame)
            new, pushes = push_frame(history.history, history.pushes, frame)
            history.commit(torch.arange(b), new, pushes)
            self.assertTrue(torch.equal(history.history, reference.buffer), f"第 {step} 步不一致")
            self.assertTrue(torch.equal(history.pushes, reference._num_pushes))

    def test_order_fill_and_length(self) -> None:
        history = PolicyHistory(1)
        for t in range(12):
            new, pushes = push_frame(history.history, history.pushes, torch.full((1, 33), float(t)))
            history.commit(torch.tensor([0]), new, pushes)
            if t == 0:
                self.assertEqual(history.history[0, :, 0].tolist(), [0.0] * 10)  # 首帧填满
        self.assertEqual(history.history[0, :, 0].tolist(), [float(t) for t in range(2, 12)])  # 从旧到新，最新帧在最后
        self.assertEqual(valid_length(history.pushes).tolist(), [10])
        self.assertEqual(valid_length(torch.tensor([0, 1, 7, 15])).tolist(), [0, 1, 7, 10])

    def test_push_returns_new_tensor_and_commit_only_selected_rows(self) -> None:
        history = PolicyHistory(2)
        new, pushes = push_frame(history.history, history.pushes, torch.ones(2, 33))
        self.assertNotEqual(new.data_ptr(), history.history.data_ptr())
        history.commit(torch.tensor([1]), new, pushes)
        self.assertEqual((history.pushes.tolist(), history.history[0].abs().sum().item()), ([0, 1], 0.0))


if __name__ == "__main__":
    unittest.main()
