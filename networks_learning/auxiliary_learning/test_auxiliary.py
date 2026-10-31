"""N02 速度监督与 Barlow Twins 的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/auxiliary_learning/test_auxiliary.py -v

数据说明：历史、速度标签与有效长度均为人工数据；速度标签只是形状正确的随机数，不代表真实 TITA 速度。
原 PPO 损失写法（c.div_、diagonal.add_(-1).pow_(2)）在本文件中复写作为独立对照。
Actor 动作头、Q 与 alpha 在本文件中用独立的占位模块代表，完整网络的衔接见 N08 测试中的更新循环。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "networks_learning" / "history_encoder"))
sys.path.insert(0, str(ROOT / "experience_returns" / "history_windows"))

from auxiliary import (  # noqa: E402
    BarlowProjector,
    barlow_loss,
    make_auxiliary_optimizer,
    update_encoder,
    velocity_loss,
)
from encoder import HistoryEncoder  # noqa: E402
from history import split_windows  # noqa: E402


def setup(batch: int = 32, seed: int = 0):
    """返回 (encoder, projector, optimizer, windows, lengths, label)；全部有效长度为 10。"""
    torch.manual_seed(seed)
    encoder, projector = HistoryEncoder(), BarlowProjector()
    optimizer = make_auxiliary_optimizer(encoder, projector)
    history = torch.randn(batch, 10, 33, generator=torch.Generator().manual_seed(seed + 1))
    label = torch.randn(batch, 3, generator=torch.Generator().manual_seed(seed + 2))
    return encoder, projector, optimizer, split_windows(history), torch.full((batch,), 10), label


def snapshot(*modules: nn.Module) -> list[torch.Tensor]:
    """参数与缓冲（含 BN 运行统计）的副本。"""
    return [t.detach().clone() for m in modules for t in m.state_dict().values()]


def same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


class LossTest(unittest.TestCase):
    """速度误差与相关性目标的数值性质。"""

    def test_velocity_loss_zero_when_correct(self) -> None:
        label = torch.randn(5, 3)
        self.assertEqual(velocity_loss(label.clone(), label).item(), 0.0)
        self.assertGreater(velocity_loss(label + 0.1, label).item(), 0.0)

    def test_constant_features_are_not_perfect(self) -> None:
        # 常数特征经 projector 末层 BatchNorm（train 模式）后为 0，相关矩阵为 0，对角项损失 = 64（不是上界，反相关时可达 256）。
        projector = BarlowProjector()
        projector.train()
        z = projector(torch.full((16, 16), 3.0))
        total, on_diagonal, _ = barlow_loss(z, z)
        self.assertAlmostEqual(on_diagonal.item(), 64.0, places=3)
        self.assertGreaterEqual(total.item(), 64.0 - 1e-3)
        # 对照：两窗口完全一致、各维独立的标准化特征，对角项接近 0。
        good = torch.randn(4096, 64, generator=torch.Generator().manual_seed(0))
        good = (good - good.mean(0)) / good.std(0, unbiased=False)
        self.assertLess(barlow_loss(good, good)[1].item(), 1e-6)
        self.assertLess(barlow_loss(good, good)[0].item(), 1.0)

    def test_matches_ppo_formula(self) -> None:
        z1, z2 = torch.randn(10, 64), torch.randn(10, 64)
        c = z1.T @ z2
        c.div_(10)
        on = torch.diagonal(c).add_(-1).pow_(2).sum()
        off = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
        total, on_diagonal, off_diagonal = barlow_loss(z1, z2)
        self.assertTrue(torch.allclose(on_diagonal, on))
        self.assertTrue(torch.allclose(off_diagonal, off, rtol=1e-5))
        self.assertTrue(torch.allclose(total, on + 5e-3 * off, rtol=1e-5))


class UpdateTest(unittest.TestCase):
    """独立优化器、BatchNorm 时机、参数归属与有效样本筛选。"""

    def test_only_encoder_and_projector_change(self) -> None:
        encoder, projector, optimizer, windows, lengths, label = setup()
        actor_head, q_net, log_alpha = nn.Linear(52, 16), nn.Linear(62, 1), nn.Parameter(torch.tensor(-3.0))
        others = snapshot(actor_head, q_net) + [log_alpha.detach().clone()]
        before = snapshot(encoder, projector)
        report = update_encoder(encoder, projector, optimizer, windows, lengths, label)
        self.assertFalse(report["skipped"])
        self.assertFalse(same(before, snapshot(encoder, projector)))
        self.assertTrue(same(others, snapshot(actor_head, q_net) + [log_alpha.detach()]))
        optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimizer_params, {id(p) for p in [*encoder.parameters(), *projector.parameters()]})

    def test_batchnorm_statistics_only_change_during_update(self) -> None:
        encoder, projector, optimizer, windows, lengths, label = setup()
        stats = [encoder.backbone[1].running_mean.clone()]
        with torch.no_grad():
            encoder.encode_history(windows.recent)  # eval 前向不改统计
        self.assertTrue(torch.equal(encoder.backbone[1].running_mean, stats[0]))
        update_encoder(encoder, projector, optimizer, windows, lengths, label)
        self.assertFalse(torch.equal(encoder.backbone[1].running_mean, stats[0]))
        self.assertFalse(encoder.training or projector.training)  # 更新后恢复 eval

    def test_insufficient_valid_history_is_skipped(self) -> None:
        encoder, projector, optimizer, windows, _, label = setup(batch=4)
        before = snapshot(encoder, projector)
        for lengths in (torch.tensor([1, 3, 5, 5]), torch.tensor([6, 2, 3, 4])):  # 0 条、1 条有效
            with self.subTest(lengths=lengths.tolist()):
                report = update_encoder(encoder, projector, optimizer, windows, lengths, label)
                self.assertTrue(report["skipped"])
                self.assertTrue(same(before, snapshot(encoder, projector)))
                self.assertFalse(encoder.training)
        self.assertEqual(len(optimizer.state), 0)  # 优化器从未执行 step

    def test_invalid_rows_are_not_used(self) -> None:
        # 无效行（长度 < 6）的窗口填 NaN：若被使用，损失会是 NaN。
        encoder, projector, optimizer, windows, lengths, label = setup(batch=8)
        history = torch.randn(8, 10, 33)
        history[:3] = float("nan")
        lengths = torch.tensor([2, 4, 5, 6, 7, 8, 9, 10])
        report = update_encoder(encoder, projector, optimizer, split_windows(history), lengths, label)
        self.assertEqual(report["valid_samples"], 5)
        self.assertTrue(torch.isfinite(torch.tensor(report["loss"])))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in encoder.parameters()))

    def test_repeated_updates_reduce_loss(self) -> None:
        encoder, projector, optimizer, windows, lengths, label = setup(batch=64)
        losses = [update_encoder(encoder, projector, optimizer, windows, lengths, label)["loss"] for _ in range(30)]
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
