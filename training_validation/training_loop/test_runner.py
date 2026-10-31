"""T02 采集与学习调度的验证程序（unittest + PyTorch CPU，使用假环境，无需 Isaac Lab 或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/training_loop/test_runner.py -v

数据说明：环境为 fake_env.FakeLocomotionEnv（人工动力学，非 Isaac 仿真）；配置为 T01 cpu_debug_config
（人工限位）；动作表来自 E03 原版快照（需要本地有 baseline/ppo 快照）。
这些检查证明调度与数据流程，不代表 Isaac 仿真验证或机器人训练成功。
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import runner  # noqa: E402
from config import cpu_debug_config  # noqa: E402
from fake_env import FakeLocomotionEnv  # noqa: E402
from returns import build_n_step_batch  # noqa: E402
from runner import Trainer  # noqa: E402


def make_trainer(termination_steps=(None, None, None), max_episode_steps=50, algorithm=None, collection=None, seed=0):
    """以 CPU 调试配置构造 Trainer；algorithm / collection 为需要覆盖的字段字典。"""
    config = cpu_debug_config(seed=seed)
    if algorithm:
        config = replace(config, algorithm=replace(config.algorithm, **algorithm))
    if collection:
        config = replace(config, collection=replace(config.collection, **collection))
    env = FakeLocomotionEnv(config.collection.num_envs, config.task.q_dim, max_episode_steps, termination_steps)
    return Trainer(config, env)


def records_of_env(trainer: Trainer, env: int, count: int):
    """读取环境 env 的前 count 次写入（要求容量足够、尚未覆盖），返回 SequenceBatch。"""
    return trainer.buffer.read_sequence(torch.tensor([env]), torch.tensor([0]), count)


class ConfigurationAppliedTest(unittest.TestCase):
    """T01 配置的每类设置都实际传到对应模块。"""

    def test_values_reach_modules(self) -> None:
        trainer = make_trainer(algorithm={"actor_lr": 3e-4, "critic_lr": 4e-4, "alpha_lr": 5e-5, "auxiliary_lr": 6e-4,
                                          "initial_alpha": 0.2, "initial_std": 0.3, "normalizer_eps": 0.05,
                                          "target_entropy_scale": 0.5})
        lrs = {name: opt.param_groups[0]["lr"] for name, opt in trainer.optimizers.items()}
        self.assertEqual(lrs, {"actor": 3e-4, "critic": 4e-4, "alpha": 5e-5, "auxiliary": 6e-4})
        self.assertAlmostEqual(trainer.temperature.log_alpha.exp().item(), 0.2, places=6)
        self.assertTrue(torch.allclose(trainer.actor.log_std_head.bias.exp(), torch.full((8,), 0.3)))
        self.assertEqual((trainer.normalizers.policy.eps, trainer.normalizers.q.eps), (0.05, 0.05))
        self.assertEqual(trainer.target_entropy.action_coordinate, -4.0)  # −0.5 × 8
        self.assertEqual((trainer.buffer.num_envs, trainer.buffer.capacity_per_env, trainer.buffer.q_dim), (3, 64, 6))
        network = trainer.config.network
        self.assertEqual(trainer.actor.trunk[0].out_features, network.actor_hidden_dims[0])
        self.assertEqual(trainer.critics.q1.net[0].out_features, network.critic_hidden_dims[0])
        self.assertEqual(trainer.encoder.backbone[0].out_features, network.encoder_backbone_dims[0])
        # 人工限位经 E04 得到的区间中心进入 Actor 缓冲（缓冲为 float32，按 float32 精度比较）。
        self.assertTrue(torch.allclose(trainer.actor.center, torch.tensor(trainer.bounds.center, dtype=torch.float32)))

    def test_grad_norm_gamma_n_and_batch_size_are_used(self) -> None:
        trainer = make_trainer(algorithm={"max_grad_norm": 0.37, "gamma": 0.9, "n_steps": 3, "batch_size": 7})
        trainer.collect_steps(12)
        seen = []
        original = torch.nn.utils.clip_grad_norm_
        with mock.patch("torch.nn.utils.clip_grad_norm_", side_effect=lambda params, max_norm, *a, **k: seen.append(max_norm) or original(params, max_norm, *a, **k)):
            report = trainer.learn_from_replay(4)
            trainer.auxiliary_update()
        self.assertTrue(seen and all(v == 0.37 for v in seen))
        self.assertAlmostEqual(report["discount_max"], 0.9**3, places=6)  # 无结束的片段：gamma^n
        self.assertEqual(trainer.counters.replay_samples, 4 * 7)

    def test_tau_controls_target_following(self) -> None:
        for tau, expect_equal in ((1.0, True), (0.0, False)):
            with self.subTest(tau=tau):
                trainer = make_trainer(algorithm={"tau": tau})
                initial = [p.clone() for p in trainer.targets.parameters()]
                trainer.collect_steps(10)
                trainer.learn_from_replay(2)
                same_as_online = all(torch.equal(o, t) for o, t in zip(trainer.critics.parameters(), trainer.targets.parameters()))
                unchanged = all(torch.equal(i, t) for i, t in zip(initial, trainer.targets.parameters()))
                self.assertEqual(same_as_online, expect_equal)
                self.assertEqual(unchanged, not expect_equal)


class ScheduleTest(unittest.TestCase):
    """多轮采集与学习：次数与数据来源。"""

    def test_counters_over_iterations(self) -> None:
        trainer = make_trainer(algorithm={"critic_updates_per_iteration": 5, "auxiliary_updates_per_iteration": 2},
                               collection={"warmup_iterations": 2, "steps_per_iteration": 6})
        logs = trainer.train(5)
        c = trainer.counters
        self.assertEqual((c.iterations, c.env_steps, c.transitions), (5, 30, 90))
        self.assertEqual(c.critic_updates, 3 * 5)  # 预热 2 轮后才学习
        self.assertEqual(c.actor_updates, math.ceil(15 / 2))  # 第 1、3、5… 次 Critic 更新后（作者 update_step % 2 == 0）
        self.assertEqual((c.alpha_updates, c.target_updates), (15, 15))
        self.assertEqual(c.auxiliary_updates + c.auxiliary_skipped, 3 * 2)  # 按迭代计，与 Critic 更新次数无关
        self.assertEqual(c.replay_samples, 15 * 16)
        self.assertNotIn("learn", logs[0])
        self.assertIn("learn", logs[2])

    def test_actions_recorded_equal_actions_executed_and_are_mild(self) -> None:
        trainer = make_trainer()
        trainer.collect_steps(10)  # 预热式采集：只用 Actor，不学习
        for env in range(3):
            stored = records_of_env(trainer, env, 10).fields["action"][0]
            executed = torch.stack([a[env] for a in trainer.env.received_actions])
            self.assertTrue(torch.equal(stored, executed))
        actions = torch.stack(trainer.env.received_actions).reshape(-1, 8)
        lower, upper = torch.tensor(trainer.bounds.lower), torch.tensor(trainer.bounds.upper)
        self.assertTrue(bool(((actions >= lower) & (actions <= upper)).all()))
        squashed = (actions - trainer.actor.center) / trainer.actor.half_width  # = tanh(u)
        # 温和初始化（std 0.15）：tanh(u) 的标准差远小于全范围均匀随机的 1/√3 ≈ 0.577。
        self.assertLess(squashed.std().item(), 0.25)

    def test_statistics_only_from_fresh_data(self) -> None:
        trainer = make_trainer()
        trainer.collect_steps(4)
        # 每个环境：首帧（history_length=1 的当前观测）1 次 + 每次动作后的真实下一观测 4 次。
        self.assertEqual(trainer.normalizers.policy.count, 3 * (1 + 4))
        counts = (trainer.normalizers.policy.count, trainer.normalizers.q.count)
        trainer.learn_from_replay(3)
        trainer.auxiliary_update()
        self.assertEqual((trainer.normalizers.policy.count, trainer.normalizers.q.count), counts)


class EpisodeBoundaryTest(unittest.TestCase):
    """真终止、纯截断与二者同时发生时的末状态、历史与回报衔接（由假环境按 E05 契约提供）。"""

    def test_terminal_time_out_and_priority(self) -> None:
        trainer = make_trainer(termination_steps=(3, None, 5), max_episode_steps=5)
        trainer.collect_steps(7)
        env0, env1, env2 = (records_of_env(trainer, e, 7).fields for e in range(3))
        self.assertEqual(env0["terminated"][0, :, 0].tolist(), [False, False, True, False, False, True, False])
        self.assertEqual(env1["time_out"][0, :, 0].tolist(), [False] * 4 + [True, False, False])
        self.assertTrue(bool(env2["terminated"][0, 4, 0]) and not bool(env2["time_out"][0, 4, 0]))  # 同步发生：真终止优先
        # 末状态是重置前的真实帧（不是 −3 附近的重置帧）；下一条记录属于新一轮。
        self.assertGreater(env0["next_policy_history"][0, 2, -1, 0].item(), -2.0)
        self.assertEqual((int(env0["episode_id"][0, 3]), int(env0["step_index"][0, 3]), int(env0["history_length"][0, 3])), (1, 0, 1))
        self.assertAlmostEqual(env0["policy_history"][0, 3, -1, 0].item(), -3.0, places=5)
        # D03：环境 0 从 step 0 起 n=5 在第 3 步真终止（k=3, mask=0）；环境 1 从 step 2 起在第 3 步纯截断（k=3, mask=1）。
        sequence = trainer.buffer.read_sequence(torch.tensor([0, 1]), torch.tensor([0, 2]), 5)
        batch = build_n_step_batch(sequence, 0.97, 5)
        self.assertEqual(batch.steps.tolist(), [3, 3])
        self.assertEqual(batch.bootstrap_mask.reshape(-1).tolist(), [0.0, 1.0])
        self.assertTrue(torch.equal(batch.tail.next_policy_history[1], env1["next_policy_history"][0, 4]))


class OrderAndFallbackTest(unittest.TestCase):
    """作者顺序（新 alpha、同一次采样）与经验不足时的处理。"""

    def test_actor_uses_updated_alpha_and_same_sample(self) -> None:
        trainer = make_trainer(algorithm={"alpha_lr": 0.05})  # 放大 alpha 的变化，便于区分新旧值
        trainer.collect_steps(12)
        events = []
        real_temperature, real_actor_loss = runner.update_temperature, runner.compute_actor_loss

        def spy_temperature(temperature, optimizer, log_prob, target):
            before = runner.current_alpha(temperature)
            result = real_temperature(temperature, optimizer, log_prob, target)
            events.append(("alpha", before, runner.current_alpha(temperature), log_prob))
            return result

        def spy_actor_loss(critics, q_observation, action, log_prob, alpha):
            events.append(("actor", alpha, log_prob))
            return real_actor_loss(critics, q_observation, action, log_prob, alpha)

        with mock.patch.object(runner, "update_temperature", spy_temperature), mock.patch.object(runner, "compute_actor_loss", spy_actor_loss):
            trainer.learn_from_replay(4)
        kinds = [e[0] for e in events]
        self.assertEqual(kinds, ["alpha", "actor", "alpha", "alpha", "actor", "alpha"])  # Actor 在第 1、3 次
        for i, event in enumerate(events):
            if event[0] == "actor":
                _, before, after, sampled = events[i - 1]
                self.assertEqual(event[1], after)
                self.assertNotEqual(event[1], before)
                self.assertIs(event[2], sampled)  # 与 alpha 更新同一次采样

    def test_insufficient_experience_and_history(self) -> None:
        trainer = make_trainer()
        trainer.collect_steps(2)  # n=5 且尚无结束：没有有效起点
        self.assertEqual(trainer.learn_from_replay(3), {"skipped": True})
        self.assertEqual((trainer.counters.critic_updates, trainer.counters.learning_skipped_iterations), (0, 1))
        encoder_before = {k: v.clone() for k, v in trainer.encoder.state_dict().items()}
        report = trainer.auxiliary_update()  # 有效历史最多 2 帧，不足 6 帧
        self.assertTrue(report["skipped"])
        self.assertEqual(trainer.counters.auxiliary_skipped, 1)
        self.assertTrue(all(torch.equal(v, encoder_before[k]) for k, v in trainer.encoder.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
