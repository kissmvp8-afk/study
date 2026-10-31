"""T05 模型导出与运行一致性的验证程序（unittest + PyTorch CPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/export_inference/test_export.py -v

数据说明：模型来自 T02 Trainer 在假环境上训练几轮的结果（cpu_debug_config，人工限位）；历史为人工构造。
这里比较的是 CPU 上训练内推理、eager 导出管线、TorchScript 模型与重新加载后的模型，不代表真机部署通过。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for path in ("training_validation/configuration", "training_validation/training_loop", "networks_learning/history_encoder",
             "environment_control/action_contract"):
    sys.path.insert(0, str(ROOT / path))

from config import cpu_debug_config  # noqa: E402
from contract import TITA_ACTION_LAYOUT, action_to_targets  # noqa: E402
from encoder import features_from_history  # noqa: E402
from export import build_inference_pipeline, compare_outputs, export_policy, load_exported_policy  # noqa: E402
from fake_env import FakeLocomotionEnv  # noqa: E402
from runner import Trainer  # noqa: E402


def trained_trainer() -> Trainer:
    config = cpu_debug_config(seed=5)
    trainer = Trainer(config, FakeLocomotionEnv(3, 6, 9, (5, None, 7)))
    trainer.train(3)
    for _ in range(2):
        trainer.auxiliary_update()  # 让编码器 BN 统计偏离初值
    return trainer


def training_path(trainer: Trainer, history: torch.Tensor) -> torch.Tensor:
    """训练内推理：与 T02 采集、N05/N06 相同的特征路径 + 确定性动作。"""
    with torch.no_grad():
        return trainer.actor.deterministic_action(features_from_history(trainer.encoder, trainer.normalizers, history))


def frame_history(batch: int = 3) -> torch.Tensor:
    """第 i 帧的 33 个数随帧号变化（带帧编号的历史），不同样本也不同。"""
    frames = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1) * 0.1
    channels = torch.linspace(-1, 1, 33).reshape(1, 1, 33)
    offsets = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1) * 0.05
    return frames + channels + offsets


class ExportConsistencyTest(unittest.TestCase):
    """训练内推理、eager 管线、导出与重新加载后的输出一致。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.trainer = trained_trainer()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "policy.pt"
        cls.metadata = export_policy(cls.trainer, cls.path)
        cls.pipeline = build_inference_pipeline(cls.trainer)
        cls.loaded = load_exported_policy(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_three_paths_match(self) -> None:
        histories = torch.randn(7, 10, 33, generator=torch.Generator().manual_seed(0))  # 批大小与 trace 示例（4）不同
        expected = training_path(self.trainer, histories)
        self.assertTrue(torch.equal(self.pipeline(histories), expected))
        report = compare_outputs(self.pipeline, self.loaded, histories)
        self.assertTrue(report["within_tolerance"], report)
        self.assertTrue(torch.equal(self.loaded(histories), self.loaded(histories)))  # 确定性

    def test_frame_numbered_history_is_not_duplicated(self) -> None:
        history = frame_history()
        exported = self.loaded(history)
        self.assertTrue(compare_outputs(lambda h: training_path(self.trainer, h), self.loaded, history)["within_tolerance"])
        # 回归（原 PPO _InferenceWrapper）：把已含当前帧的历史再拼一次当前帧，窗口变为 6,7,8,9,9，动作随之改变。
        duplicated = torch.cat([history[:, 1:], history[:, -1:]], dim=1)
        self.assertFalse(torch.allclose(self.loaded(duplicated), exported, atol=1e-6))

    def test_statistics_frozen_at_export(self) -> None:
        trainer = trained_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.pt"
            export_policy(trainer, path)
            loaded = load_exported_policy(path)
            history = frame_history()
            before = loaded(history)
            trainer.collect_steps(5)  # 新数据更新 D04 统计
            self.assertTrue(torch.equal(loaded(history), before))
            self.assertFalse(torch.allclose(training_path(trainer, history), before, atol=1e-6))

    def test_bounds_and_reset_history(self) -> None:
        lower, upper = torch.tensor(self.trainer.bounds.lower), torch.tensor(self.trainer.bounds.upper)
        extreme = torch.randn(5, 10, 33, generator=torch.Generator().manual_seed(1)) * 1e4
        action = self.loaded(extreme)
        self.assertTrue(bool(((action >= lower - 1e-5) & (action <= upper + 1e-5)).all()))
        reset = torch.full((2, 10, 33), -3.0)  # 新一轮：历史由重置帧填满（假环境的填充规则）
        self.assertTrue(compare_outputs(self.pipeline, self.loaded, reset)["within_tolerance"])
        self.assertTrue(compare_outputs(self.pipeline, self.loaded, extreme)["within_tolerance"])


class InterfaceTest(unittest.TestCase):
    """元数据、动作顺序与物理换算接口。"""

    def test_metadata_and_single_physical_conversion(self) -> None:
        trainer = trained_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            metadata = export_policy(trainer, path)
            self.assertEqual(json.loads(path.with_suffix(".json").read_text(encoding="utf-8")), metadata)
            loaded = load_exported_policy(path)
        self.assertEqual([j["joint_name"] for j in metadata["joints"]], [name for name, _ in TITA_ACTION_LAYOUT])
        self.assertEqual(metadata["output"]["coordinate"], "policy_action")
        self.assertFalse(metadata["output"]["physical_conversion_included"])
        self.assertEqual(metadata["control_period_s"], 0.02)
        history = frame_history(1)
        exported = loaded(history)[0].tolist()
        # 导出输出经 E03 换算一次 = 训练内确定性动作经 E03 换算一次。
        expected = action_to_targets(trainer.contract, training_path(trainer, history)[0].tolist())
        for got, want in zip(action_to_targets(trainer.contract, exported), expected):
            self.assertAlmostEqual(got, want, places=5)
        # 导出输出仍在策略坐标（E04 区间内），不是物理目标：轮速项的物理目标 = 5 × a。
        self.assertAlmostEqual(action_to_targets(trainer.contract, exported)[3], 5.0 * exported[3], places=5)

    def test_q_and_projector_not_exported(self) -> None:
        trainer = trained_trainer()
        pipeline = build_inference_pipeline(trainer)
        exported_params = sum(p.numel() for p in pipeline.parameters())
        expected = sum(p.numel() for p in trainer.encoder.parameters()) + sum(p.numel() for p in trainer.actor.parameters())
        self.assertEqual(exported_params, expected)
        self.assertFalse(any(p.requires_grad for p in pipeline.parameters()))
        self.assertFalse(pipeline.encoder.training)


if __name__ == "__main__":
    unittest.main()
