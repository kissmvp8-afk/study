"""N08 目标 Q 慢更新的验证程序，以及网络与学习大模块的一次完整更新衔接检查（unittest + PyTorch CPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B networks_learning/target_updates/test_targets.py -v

数据说明：
    - 慢更新检查用人工小网络与手算平均值。
    - UpdateCycleTest 从 D01/D02 人工经历（D02 测试的 make_records）出发，经 D03 组合、D04 归一化、D05 切片，
      依次执行 Critic、Actor、温度、目标慢更新与辅助学习更新各一次，并在每一步检查只有负责的参数和统计改变。
      动作表来自 E03 原版快照，软限位与工作轮速是 E04 测试中标注的人工数据；速度标签取 Q 观测前 3 维，
      这是原 PPO 的字段布局（base_lin_vel 为 critic 观测第一项），E02 尚未确认。
    这只是人工数据下的更新衔接检查，不是 T02 训练器，也不代表训练效果。
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
for path in (
    "networks_learning/history_encoder", "networks_learning/auxiliary_learning", "networks_learning/actor_distribution",
    "networks_learning/q_networks", "networks_learning/critic_learning", "networks_learning/actor_learning",
    "networks_learning/entropy_temperature", "environment_control/action_bounds", "environment_control/action_contract",
    "experience_returns/replay_buffer", "experience_returns/n_step_returns", "experience_returns/normalization",
    "experience_returns/history_windows",
):
    sys.path.insert(0, str(ROOT / path))

from targets import TargetUpdateError, check_target_state, soft_update, soft_update_due, validate_target_structure  # noqa: E402


class SmallQ(nn.Module):
    """人工小网络：一个线性层 + 可选 BatchNorm（用于检查缓冲规则）。"""

    def __init__(self, with_batchnorm: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.norm = nn.BatchNorm1d(2) if with_batchnorm else None


def pair(with_batchnorm: bool = False):
    torch.manual_seed(0)
    online, target = SmallQ(with_batchnorm), SmallQ(with_batchnorm)
    for p in target.parameters():
        p.requires_grad_(False)
    return online, target


class SoftUpdateTest(unittest.TestCase):
    """公式、覆盖范围与存储独立。"""

    def test_tau_zero_one_and_middle(self) -> None:
        online, target = pair()
        o, t = online.linear.weight.clone(), target.linear.weight.clone()
        soft_update(online, target, 0.0)
        self.assertTrue(torch.equal(target.linear.weight, t))
        soft_update(online, target, 0.3)
        self.assertTrue(torch.allclose(target.linear.weight, 0.7 * t + 0.3 * o))  # 手算平均
        soft_update(online, target, 1.0)
        self.assertTrue(torch.equal(target.linear.weight, o) and torch.equal(target.linear.bias, online.linear.bias))
        self.assertTrue(torch.equal(online.linear.weight, o))  # 在线网络不被修改

    def test_shared_storage_and_structure_are_rejected(self) -> None:
        online, _ = pair()
        with self.assertRaisesRegex(TargetUpdateError, "共享存储"):
            soft_update(online, online, 0.5)
        with self.assertRaisesRegex(TargetUpdateError, "名称不一致"):
            validate_target_structure(online, SmallQ(with_batchnorm=True))
        with self.assertRaises(TargetUpdateError):
            soft_update(*pair(), tau=1.5)

    def test_buffers_are_copied_not_averaged(self) -> None:
        online, target = pair(with_batchnorm=True)
        with torch.no_grad():
            online.norm.running_mean.fill_(4.0)
            online.norm.num_batches_tracked.fill_(7)
        soft_update(online, target, 0.1)
        self.assertTrue(torch.equal(target.norm.running_mean, torch.full((2,), 4.0)))  # 不是 0.1×4
        self.assertEqual(int(target.norm.num_batches_tracked), 7)
        self.assertIn("norm.running_mean", check_target_state(online, target)["buffers_copied_verbatim"])

    def test_target_state_and_frequency(self) -> None:
        online, target = pair()
        optimizer = torch.optim.Adam(online.parameters())
        self.assertEqual(check_target_state(online, target, (optimizer,))["parameters"], 2)
        with self.assertRaisesRegex(TargetUpdateError, "优化器"):
            check_target_state(online, target, (torch.optim.SGD(target.parameters(), lr=0.1),))
        target.linear.weight.requires_grad_(True)
        with self.assertRaisesRegex(TargetUpdateError, "求梯度"):
            check_target_state(online, target)
        self.assertEqual([soft_update_due(i, 2) for i in range(1, 5)], [False, True, False, True])
        self.assertTrue(all(soft_update_due(i) for i in range(1, 4)))


class UpdateCycleTest(unittest.TestCase):
    """D01/D02/D03 → D04/D05 → N01 → N03/N04 → N05/N06/N07/N08/N02 的一次完整更新。"""

    def test_one_cycle_changes_only_owned_state(self) -> None:
        from actor import SquashedGaussianActor
        from actor_update import compute_actor_loss, make_actor_optimizer, update_actor
        from auxiliary import BarlowProjector, make_auxiliary_optimizer, update_encoder
        from baseline_source import load_baseline_action_contract
        from bounds import calibrate_initial_center, derive_bounds
        from buffer import ReplayBuffer
        from critic_update import compute_target, make_critic_optimizer, update_critics
        from critics import TwinQ, initialize_targets
        from encoder import HistoryEncoder, features_from_history
        from history import split_windows
        from normalization import ObservationNormalizers
        from returns import build_n_step_batch
        from temperature import Temperature, current_alpha, make_temperature_optimizer, target_entropy, update_temperature
        from test_bounds import ARTIFICIAL_JOINT_POS_LIMITS, ARTIFICIAL_WHEEL_SPEED_RANGES
        from test_buffer import Q_DIM, make_records

        torch.manual_seed(0)
        contract = load_baseline_action_contract()
        bounds = derive_bounds(contract, ARTIFICIAL_JOINT_POS_LIMITS, ARTIFICIAL_WHEEL_SPEED_RANGES)
        # 经验：两个环境各 12 步，环境 0 在第 7 步真终止后开始新一轮，环境 1 在第 11 步纯截断。
        buffer, normalizers = ReplayBuffer(2, 16, Q_DIM), ObservationNormalizers(Q_DIM)
        for step in range(12):
            episode0, step0 = (0, step) if step <= 7 else (1, step - 8)
            fresh = make_records([(0, episode0, step0), (1, 0, step)], terminated={0} if step == 7 else set(),
                                 time_out={1} if step == 11 else set(), rewards={0: 1.0, 1: 0.5})
            normalizers.update_from_fresh_data(fresh)  # 每批新数据只更新一次统计
            buffer.append(fresh)
        env_ids, slots = buffer.sample_starts(32, 3, generator=torch.Generator().manual_seed(1))
        batch = build_n_step_batch(buffer.read_sequence(env_ids, slots, 3), gamma=0.99, n_steps=3)
        raw_before = buffer.read_sequence(env_ids, slots, 3).fields["policy_history"].clone()

        encoder, projector = HistoryEncoder(), BarlowProjector()
        actor = SquashedGaussianActor(bounds, calibrate_initial_center(contract, bounds))
        critics = TwinQ(Q_DIM)
        targets = initialize_targets(critics)
        temperature = Temperature()
        optimizers = {
            "critic": make_critic_optimizer(critics), "actor": make_actor_optimizer(actor),
            "alpha": make_temperature_optimizer(temperature), "aux": make_auxiliary_optimizer(encoder, projector),
        }
        modules = {"encoder": encoder, "projector": projector, "actor": actor, "critics": critics, "targets": targets,
                   "temperature": temperature}
        snapshot = lambda: {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in modules.items()}  # noqa: E731

        def changed(before) -> set[str]:
            now = snapshot()
            return {n for n in modules if any(not torch.equal(now[n][k], before[n][k]) for k in now[n])}

        # 1) Critic：目标 y 用尾部新动作与目标 Q，只更新在线 Q。
        before = snapshot()
        y = compute_target(batch, encoder, actor, targets, normalizers, current_alpha(temperature), torch.Generator().manual_seed(2))
        critic_report = update_critics(critics, optimizers["critic"], batch, y, normalizers)
        self.assertEqual(changed(before), {"critics"})

        # 2) Actor：回放状态上重新采样；只更新 Actor。
        before = snapshot()
        features = features_from_history(encoder, normalizers, batch.first.policy_history)
        q_input = normalizers.normalize_q(batch.first.q_observation)
        actor_loss, _, log_prob = compute_actor_loss(actor, critics, features, q_input, current_alpha(temperature),
                                                     torch.Generator().manual_seed(3))
        update_actor(actor, optimizers["actor"], actor_loss)
        self.assertEqual(changed(before), {"actor"})

        # 3) 温度：使用同一次采样的 log π；只更新 log_alpha。
        before = snapshot()
        alpha_report = update_temperature(temperature, optimizers["alpha"], log_prob, target_entropy(bounds.half_width).action_coordinate)
        self.assertEqual(changed(before), {"temperature"})

        # 4) 目标慢更新：只改目标 Q。
        before = snapshot()
        soft_update(critics, targets, tau=0.005)
        self.assertEqual(changed(before), {"targets"})
        check_target_state(critics, targets, tuple(optimizers.values()))

        # 5) 辅助学习：只改编码器与 projector（含 BN 统计）；速度标签取原始 Q 观测前 3 维（原 PPO 布局，E02 待确认）。
        before = snapshot()
        windows = split_windows(normalizers.normalize_policy(batch.first.policy_history))
        aux_report = update_encoder(encoder, projector, optimizers["aux"], windows, batch.first.history_length,
                                    batch.first.q_observation[:, :3])
        self.assertFalse(aux_report["skipped"])
        self.assertEqual(changed(before), {"encoder", "projector"})

        # 原始经历不变；所有日志为有限数。
        self.assertTrue(torch.equal(buffer.read_sequence(env_ids, slots, 3).fields["policy_history"], raw_before))
        values = [*critic_report.values(), actor_loss.item(), *alpha_report.values(), aux_report["loss"]]
        self.assertTrue(all(torch.isfinite(torch.tensor(float(v))) for v in values))


if __name__ == "__main__":
    unittest.main()
