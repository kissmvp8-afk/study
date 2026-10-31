"""E05 终止、截断与重置前观测的验证程序（unittest + PyTorch CPU，无需 Isaac Lab）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/episode_lifecycle/test_lifecycle.py -v

数据说明：ScriptedSimulation 是**人工的**钩子实现，用标量状态 s 编码每个环境的观测（策略帧全为 s、Q 观测前 3 列为 2s），
重置值约为 −100，与运动中的状态明显不同；它记录调用顺序，用来检查 run_isaac_step 是否按锁定 Isaac Lab 的顺序执行。
它不模拟 TITA 动力学，这里的结果不是 Isaac 仿真验证；真实钩子（E01 TitaSacEnv）需在 Windows + Isaac 环境中验证。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for path in ("environment_control/observation_interface", "training_validation/configuration", "training_validation/training_loop",
             "training_validation/checkpoints", "experience_returns/n_step_returns"):
    sys.path.insert(0, str(ROOT / path))

from lifecycle import EpisodeLifecycle, LifecycleError, PolicyHistory, build_step_record, classify_endings, run_isaac_step  # noqa: E402
from observations import push_frame  # noqa: E402

RESET_BASE = -100.0


class ScriptedSimulation:
    """人工钩子：状态 s[B]；动作使 s 增加 mean(a)；按计划真终止，到 max_steps 时到时。"""

    def __init__(self, num_envs=3, max_steps=6, terminate_at=(None, 3, None), device="cpu") -> None:
        self.num_envs, self.device, self.max_steps = num_envs, torch.device(device), max_steps
        self.terminate_at = terminate_at
        self.s = torch.zeros(num_envs)
        self.steps = torch.zeros(num_envs, dtype=torch.int64)
        self.calls: list[str] = []
        self.actions: list[torch.Tensor] = []

    def _obs(self):
        s = self.s.reshape(-1, 1)
        critic = torch.cat([2.0 * s.expand(-1, 3), s.expand(-1, 33)], dim=1)  # 前 3 列模仿 base_lin_vel × 2.0
        priv = torch.arange(self.num_envs, dtype=torch.float32).reshape(-1, 1).expand(-1, 18).clone()
        return {"policy": s.expand(-1, 33).clone(), "critic": critic, "priv": priv}

    def reset_all(self):
        self.calls.append("reset_all")
        self.s = RESET_BASE - torch.arange(self.num_envs, dtype=torch.float32)
        self.steps.zero_()
        return self._obs()

    def apply_action_and_simulate(self, action):
        self.calls.append("simulate")
        self.actions.append(action.clone())
        self.s = self.s + action.mean(dim=1) + 1.0
        self.steps += 1

    def compute_terminations(self):
        self.calls.append("terminations")
        terminated = torch.tensor([t is not None and int(self.steps[i]) == t for i, t in enumerate(self.terminate_at)])
        return terminated, self.steps == self.max_steps

    def compute_rewards(self):
        self.calls.append("rewards")
        return self.s.clone()

    def compute_observations(self):
        self.calls.append("observe")
        return self._obs()

    def reset_envs(self, env_ids):
        self.calls.append("reset")
        self.s[env_ids] = RESET_BASE - env_ids.float()
        self.steps[env_ids] = 0

    def after_reset(self):
        self.calls.append("after_reset")


def make_lifecycle(**kwargs) -> tuple[EpisodeLifecycle, ScriptedSimulation]:
    sim = ScriptedSimulation(**kwargs)
    return EpisodeLifecycle(sim, "isaac_flat", 54), sim


class OrderTest(unittest.TestCase):
    """调用顺序与锁定 Isaac Lab 2.3.0 一致，重置前插入一次观测计算。"""

    def test_step_order(self) -> None:
        sim = ScriptedSimulation(terminate_at=(1, None, None))
        sim.reset_all()
        sim.calls.clear()
        raw = run_isaac_step(sim, torch.zeros(3, 8))
        self.assertEqual(sim.calls, ["simulate", "terminations", "rewards", "observe", "reset", "after_reset", "observe"])
        self.assertEqual(raw.reset_env_ids.tolist(), [0])
        self.assertNotEqual(raw.pre_reset["policy"][0, 0].item(), raw.post_reset["policy"][0, 0].item())
        sim.calls.clear()
        sim.terminate_at = (None, None, None)  # 终止计划按“本轮第几步”计，清空后下一步没有环境结束
        run_isaac_step(sim, torch.zeros(3, 8))  # 没有环境结束时不调用重置
        self.assertEqual(sim.calls, ["simulate", "terminations", "rewards", "observe", "after_reset", "observe"])


class EndingTest(unittest.TestCase):
    """正常、真终止、纯截断、二者同时发生。"""

    def test_classification(self) -> None:
        terminated, time_out = classify_endings(torch.tensor([False, True, False, True]), torch.tensor([False, False, True, True]))
        self.assertEqual(terminated.reshape(-1).tolist(), [False, True, False, True])
        self.assertEqual(time_out.reshape(-1).tolist(), [False, False, True, False])  # 同时发生时只记真终止

    def test_simultaneous_terminal_and_time_out_in_steps(self) -> None:
        lifecycle, _ = make_lifecycle(max_steps=3, terminate_at=(None, 3, 2))
        lifecycle.reset()
        flags = [lifecycle.step(torch.zeros(3, 8)) for _ in range(3)]
        self.assertEqual(flags[1].terminated.reshape(-1).tolist(), [False, False, True])
        last = flags[2]
        self.assertEqual(last.terminated.reshape(-1).tolist(), [False, True, False])
        self.assertEqual(last.time_out.reshape(-1).tolist(), [True, False, False])  # 环境 1 同时终止与到时：只记终止


class FinalStateTest(unittest.TestCase):
    """经验记录保存重置前的真实末状态；其他环境不受影响；历史每步只推进一次。"""

    def test_final_state_and_partial_reset(self) -> None:
        lifecycle, sim = make_lifecycle(terminate_at=(None, 3, None))
        observation = lifecycle.reset()
        frames = [observation.policy_history[:, -1, 0].clone()]
        for step in range(1, 4):
            action = torch.full((3, 8), 0.5)
            result = lifecycle.step(action)
            frames.append(result.next_policy_history[:, -1, 0].clone())
        # 环境 1 在第 3 步真终止：next_* 是重置前状态（−101 + 3×1.5 = −96.5），不是重置值 −101。
        self.assertAlmostEqual(result.next_policy_history[1, -1, 0].item(), -101.0 + 3 * 1.5)
        self.assertAlmostEqual(result.next_q_observation[1, 0].item(), 2 * (-101.0 + 3 * 1.5))  # 速度列同为末状态
        self.assertEqual(int(result.next_history_length[1]), 4)  # 重置前的有效长度 = 3 + 1
        # observation 是新一轮：重置帧填满、长度 1、episode 加一、step 归零。
        self.assertTrue(bool((result.observation.policy_history[1] == -101.0).all()))
        self.assertEqual((int(result.observation.history_length[1]), int(result.observation.episode_id[1]),
                          int(result.observation.step_index[1])), (1, 1, 0))
        # 未结束的环境 0：历史是每步一帧的序列（未被重复推进，也没有被环境 1 的重置影响）。
        expected = torch.tensor([RESET_BASE] * 7 + [RESET_BASE + 1.5 * k for k in (1, 2, 3)])
        self.assertTrue(torch.allclose(result.observation.policy_history[0, :, 0], expected))
        self.assertTrue(torch.equal(result.next_policy_history[0], result.observation.policy_history[0]))
        self.assertEqual((int(result.observation.history_length[0]), int(result.observation.step_index[0])), (4, 3))

    def test_snapshot_not_overwritten_by_later_steps(self) -> None:
        lifecycle, _ = make_lifecycle(terminate_at=(None, 2, None))
        lifecycle.reset()
        lifecycle.step(torch.zeros(3, 8))
        ended = lifecycle.step(torch.zeros(3, 8))
        saved = ended.next_policy_history.clone()
        for _ in range(3):
            lifecycle.step(torch.ones(3, 8))
        self.assertTrue(torch.equal(ended.next_policy_history, saved))

    def test_missing_final_observation_is_an_error(self) -> None:
        history = PolicyHistory(2)
        with self.assertRaisesRegex(LifecycleError, "重置前观测"):
            build_step_record(history, torch.tensor([True, False]), None, None, torch.zeros(2, 33), torch.zeros(2, 54))

    def test_actions_passed_unchanged(self) -> None:
        lifecycle, sim = make_lifecycle()
        lifecycle.reset()
        action = torch.linspace(-2, 2, 24).reshape(3, 8)  # 策略坐标；物理换算由 Isaac 动作项执行一次
        lifecycle.step(action)
        self.assertTrue(torch.equal(sim.actions[0], action))

    def test_first_episode_ids_for_resume(self) -> None:
        lifecycle, _ = make_lifecycle()
        observation = lifecycle.reset(first_episode_ids=torch.tensor([5, 2, 9]))
        self.assertEqual((observation.episode_id.tolist(), observation.step_index.tolist(), observation.history_length.tolist()),
                         ([5, 2, 9], [0, 0, 0], [1, 1, 1]))


class TrainerIntegrationTest(unittest.TestCase):
    """EpisodeLifecycle 与现有 T02 Trainer、D02 回放、D03 回报和 T03 恢复接口衔接（钩子为人工模拟）。"""

    def make_trainer(self):
        from config import cpu_debug_config
        from runner import Trainer

        config = cpu_debug_config(seed=1)
        # Q 维数取平地预期 54（模拟钩子的观测按此构造）；其余沿用 CPU 调试配置。
        config = replace(config, task=replace(config.task, q_dim=54))
        lifecycle, sim = make_lifecycle(max_steps=6, terminate_at=(None, 3, None))
        return Trainer(config, lifecycle), sim

    def test_collect_learn_and_true_final_state_in_buffer(self) -> None:
        from returns import build_n_step_batch

        trainer, sim = self.make_trainer()
        trainer.collect_steps(6)
        sequence = trainer.buffer.read_sequence(torch.tensor([1]), torch.tensor([0]), 5)
        fields = sequence.fields
        self.assertTrue(bool(fields["terminated"][0, 2, 0]))
        final = fields["next_policy_history"][0, 2, -1, 0].item()
        self.assertGreater(final, -101.0)  # 重置前末状态（有动作推进），不是重置值 −101
        self.assertAlmostEqual(fields["policy_history"][0, 3, -1, 0].item(), -101.0)  # 下一条记录是新一轮
        self.assertTrue(torch.equal(fields["action"][0, :3], torch.stack([a[1] for a in sim.actions[:3]])))
        batch = build_n_step_batch(sequence, 0.97, 5)
        self.assertEqual((batch.steps.tolist(), batch.bootstrap_mask.reshape(-1).tolist()), ([3], [0.0]))
        report = trainer.learn_from_replay(2)
        self.assertFalse(report["skipped"])

    def test_checkpoint_restore_with_adapter(self) -> None:
        from checkpoint import restore_checkpoint, save_checkpoint

        trainer, _ = self.make_trainer()
        trainer.train(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(trainer, Path(tmp) / "ckpt.pt")
            restored, _ = self.make_trainer()
            report = restore_checkpoint(restored, path)
        self.assertEqual(report.mode, "exact_replay")
        self.assertTrue(torch.equal(restored.observation.episode_id, trainer.observation.episode_id + 1))
        self.assertEqual(restored.observation.step_index.tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
