"""D01 经历记录契约的验证程序（unittest + PyTorch CPU，无需 Isaac Lab、仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/transition_schema/test_schema.py -v
（或在已安装 torch 的环境中：python -B experience_returns/transition_schema/test_schema.py -v）

数据说明：全部为人工构造的数据。观测用容易辨认的常数填充（如末状态 7.0、重置状态 -3.0），
不代表真实 TITA 观测；Q 观测维数用 54（E02 平地预期值）只作为示例。
不覆盖：E05 在仿真中捕获末状态、D02 写入与抽样、D03 多步累计。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "environment_control" / "action_contract"))

from contract import TITA_ACTION_LAYOUT  # noqa: E402
from schema import (  # noqa: E402
    ACTION_DIM,
    FIELD_NAMES,
    TransitionBatch,
    TransitionSchemaError,
    derive_bootstrap_mask,
    describe_transition,
    make_transition,
    validate_transition,
)

Q_DIM = 54


def artificial_data(batch_size: int = 3, q_dim: int = Q_DIM) -> dict[str, torch.Tensor]:
    """构造一批人工来源数据：第 0 行正常步，第 1 行真终止，第 2 行纯时间截断（B=3 时）。

    当前历史填 1.0，真实下一历史填 7.0，便于识别；编号与长度取互不相同的值，便于检查是否原样保存。
    """
    b = batch_size
    terminated = torch.zeros(b, 1, dtype=torch.bool)
    time_out = torch.zeros(b, 1, dtype=torch.bool)
    if b >= 3:
        terminated[1] = True
        time_out[2] = True
    history_length = torch.tensor([3, 10, 7, 1, 9][:b], dtype=torch.int64)
    return {
        "policy_history": torch.full((b, 10, 33), 1.0),
        "next_policy_history": torch.full((b, 10, 33), 7.0),
        "q_observation": torch.full((b, q_dim), 2.0),
        "next_q_observation": torch.full((b, q_dim), 8.0),
        "action": torch.linspace(-1.0, 1.0, b * 8).reshape(b, 8),
        "reward": torch.tensor([[0.5], [-1.0], [0.25], [0.0], [1.0]][:b]),
        "terminated": terminated,
        "time_out": time_out,
        "env_id": torch.tensor([0, 5, 2, 3, 4][:b], dtype=torch.int64),
        "episode_id": torch.tensor([11, 4, 7, 0, 1][:b], dtype=torch.int64),
        "step_index": torch.tensor([2, 37, 999, 0, 8][:b], dtype=torch.int64),
        "history_length": history_length,
        "next_history_length": torch.clamp(history_length + 1, max=10),
    }


class BootstrapMaskTest(unittest.TestCase):
    """bootstrap mask：真终止 0，正常步与纯截断 1。"""

    def test_mixed_batch(self) -> None:
        batch = make_transition(artificial_data(), Q_DIM)
        mask = derive_bootstrap_mask(batch)
        self.assertEqual(mask.shape, (3, 1))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask.reshape(-1).tolist(), [1.0, 0.0, 1.0])

    def test_terminal_priority_without_validation(self) -> None:
        # 即使绕过校验把两个标记同时置真，mask 也只看真终止，结果为 0。
        data = artificial_data()
        data["time_out"][1] = True
        with self.assertRaisesRegex(TransitionSchemaError, "同时为真"):
            make_transition(data, Q_DIM)
        unchecked = TransitionBatch(**data)
        self.assertEqual(derive_bootstrap_mask(unchecked)[1, 0].item(), 0.0)


class TrueNextObservationTest(unittest.TestCase):
    """记录保存的是传入的真实末状态，而不是重置后状态。"""

    def test_keeps_final_state_not_reset_state(self) -> None:
        data = artificial_data()
        reset_history = torch.full((3, 10, 33), -3.0)  # 采集循环中重置后的观测，只用于下一次交互
        batch = make_transition(data, Q_DIM)
        # 结束的第 1、2 行也必须保存 7.0（末状态），不能是 -3.0。
        self.assertTrue(bool((batch.next_policy_history == 7.0).all()))
        self.assertFalse(bool((batch.next_policy_history == reset_history).any()))
        self.assertTrue(bool((batch.next_q_observation == 8.0).all()))

    def test_reset_observation_cannot_be_passed_as_extra_field(self) -> None:
        data = artificial_data()
        data["reset_policy_history"] = torch.full((3, 10, 33), -3.0)
        with self.assertRaisesRegex(TransitionSchemaError, "多余 \\['reset_policy_history'\\]"):
            make_transition(data, Q_DIM)

    def test_reset_history_in_next_field_is_caught_by_length(self) -> None:
        # 若把重置后历史（有效长度 1）误填进 next_*，下一有效长度与 min(当前+1, 10) 不符。
        data = artificial_data()
        data["next_policy_history"][1] = -3.0
        data["next_history_length"][1] = 1
        with self.assertRaisesRegex(TransitionSchemaError, "next_history_length：行 \\[1\\]"):
            make_transition(data, Q_DIM)


class CopyResponsibilityTest(unittest.TestCase):
    """make_transition 复制来源张量，来源的后续原地修改不影响记录。"""

    def test_in_place_source_changes_do_not_leak(self) -> None:
        data = artificial_data()
        batch = make_transition(data, Q_DIM)
        for name in FIELD_NAMES:
            value = data[name]
            if value.dtype == torch.bool:
                value.logical_not_()
            else:
                value.add_(100)  # 模拟环境重置或下一步计算覆盖缓冲区
        validate_transition(batch, Q_DIM)
        self.assertTrue(bool((batch.next_policy_history == 7.0).all()))
        self.assertEqual(batch.reward.reshape(-1).tolist(), [0.5, -1.0, 0.25])
        self.assertEqual(batch.terminated.reshape(-1).tolist(), [False, True, False])
        self.assertEqual(batch.env_id.tolist(), [0, 5, 2])
        for name in FIELD_NAMES:
            self.assertNotEqual(getattr(batch, name).data_ptr(), data[name].data_ptr(), name)


class IdentityAndHistoryLengthTest(unittest.TestCase):
    """编号与有效历史长度按约定保存和校验。"""

    def test_identifiers_saved_as_given(self) -> None:
        batch = make_transition(artificial_data(5), Q_DIM)
        self.assertEqual(batch.batch_size, 5)
        self.assertEqual(batch.env_id.tolist(), [0, 5, 2, 3, 4])
        self.assertEqual(batch.episode_id.tolist(), [11, 4, 7, 0, 1])
        self.assertEqual(batch.step_index.tolist(), [2, 37, 999, 0, 8])
        # 当前 10 帧已满时下一仍为 10；当前 1 帧时下一为 2。
        self.assertEqual(batch.history_length.tolist(), [3, 10, 7, 1, 9])
        self.assertEqual(batch.next_history_length.tolist(), [4, 10, 8, 2, 10])

    def test_rejects_out_of_range_lengths_and_negative_ids(self) -> None:
        for name, row, value, pattern in (
            ("history_length", 0, 0, "history_length：行 \\[0\\] 不在 1..10"),
            ("history_length", 2, 11, "history_length：行 \\[2\\] 不在 1..10"),
            ("env_id", 1, -1, "env_id：行 \\[1\\] 为负数"),
            ("episode_id", 0, -2, "episode_id：行 \\[0\\] 为负数"),
            ("step_index", 2, -5, "step_index：行 \\[2\\] 为负数"),
        ):
            with self.subTest(field=name, value=value):
                data = artificial_data()
                data[name][row] = value
                if name == "history_length":
                    data["next_history_length"] = torch.clamp(data["history_length"] + 1, 1, 10)
                with self.assertRaisesRegex(TransitionSchemaError, pattern):
                    make_transition(data, Q_DIM)

    def test_rejects_history_advanced_twice(self) -> None:
        data = artificial_data()
        data["next_history_length"][0] = 5  # 当前 3，正确应为 4
        with self.assertRaisesRegex(TransitionSchemaError, "next_history_length：行 \\[0\\]"):
            make_transition(data, Q_DIM)


class FieldErrorTest(unittest.TestCase):
    """错误信息指出具体字段。"""

    def assert_field_error(self, data: dict, pattern: str, q_dim: int = Q_DIM) -> None:
        with self.assertRaisesRegex(TransitionSchemaError, pattern):
            make_transition(data, q_dim)

    def test_action_not_8_dim(self) -> None:
        data = artificial_data()
        data["action"] = torch.zeros(3, 7)
        self.assert_field_error(data, "action：形状应为 \\[B,8\\]，实际为 \\[3, 7\\]")

    def test_batch_size_mismatch(self) -> None:
        data = artificial_data()
        data["reward"] = torch.zeros(2, 1)
        self.assert_field_error(data, "批维不一致：.*'reward': 2")

    def test_missing_field(self) -> None:
        data = artificial_data()
        del data["time_out"]
        self.assert_field_error(data, "缺失 \\['time_out'\\]")

    def test_non_finite_reward(self) -> None:
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad):
                data = artificial_data()
                data["reward"][2, 0] = bad
                self.assert_field_error(data, "reward：含 NaN 或 inf")

    def test_wrong_dtype(self) -> None:
        data = artificial_data()
        data["reward"] = data["reward"].double()
        self.assert_field_error(data, "reward：类型应为 torch.float32")
        data = artificial_data()
        data["terminated"] = data["terminated"].float()
        self.assert_field_error(data, "terminated：类型应为 torch.bool")

    def test_q_dim_mismatch_is_not_padded(self) -> None:
        # 241 维数据配 54 的契约：报错，不截断。
        self.assert_field_error(artificial_data(q_dim=241), "q_observation：形状应为 \\[B,54\\]")

    def test_history_shape(self) -> None:
        data = artificial_data()
        data["policy_history"] = torch.zeros(3, 5, 33)
        self.assert_field_error(data, "policy_history：形状应为 \\[B,10,33\\]")

    def test_not_a_tensor(self) -> None:
        data = artificial_data()
        data["env_id"] = [0, 1, 2]
        self.assert_field_error(data, "env_id 必须是 torch.Tensor")

    def test_invalid_q_dim(self) -> None:
        self.assert_field_error(artificial_data(), "q_dim 必须为正整数", q_dim=0)


class ContractConsistencyTest(unittest.TestCase):
    """与上游契约一致，以及摘要输出。"""

    def test_action_dim_matches_e03(self) -> None:
        self.assertEqual(ACTION_DIM, len(TITA_ACTION_LAYOUT))

    def test_describe_transition(self) -> None:
        batch = make_transition(artificial_data(), Q_DIM)
        text = describe_transition(batch, 1)
        self.assertIn("env_id=5 episode_id=4 step_index=37", text)
        self.assertIn("当前 10 → 下一 10", text)
        self.assertIn("结束原因：真终止；bootstrap mask=0", text)
        self.assertIn("纯时间截断；bootstrap mask=1", describe_transition(batch, 2))
        with self.assertRaises(IndexError):
            describe_transition(batch, 3)


if __name__ == "__main__":
    unittest.main()
