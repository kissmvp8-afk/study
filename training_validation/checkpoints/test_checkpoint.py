"""T03 训练进度保存与恢复的验证程序（unittest + PyTorch CPU，使用 T02 假环境）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/checkpoints/test_checkpoint.py -v

数据说明：Trainer 使用 T01 cpu_debug_config（人工限位）与 T02 FakeLocomotionEnv（人工动力学，非 Isaac 仿真）；
动作表来自 E03 原版快照。这里验证的是训练状态的保存恢复语义，不是仿真现场的恢复。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "training_validation" / "training_loop"))
sys.path.insert(0, str(ROOT / "networks_learning" / "history_encoder"))

from checkpoint import CheckpointError, describe_resume_mode, load_checkpoint_config, restore_checkpoint, save_checkpoint  # noqa: E402
from config import cpu_debug_config  # noqa: E402
from encoder import features_from_history  # noqa: E402
from fake_env import FakeLocomotionEnv  # noqa: E402
from runner import Trainer  # noqa: E402

TERMINATION = (5, None, 7)


def new_trainer(config=None) -> Trainer:
    config = config or cpu_debug_config(seed=3)
    env = FakeLocomotionEnv(config.collection.num_envs, config.task.q_dim, 9, TERMINATION)
    return Trainer(config, env)


def trained_trainer() -> Trainer:
    """训练 3 轮（含预热、学习与辅助学习），使网络、优化器、统计和 BN 统计都偏离初值。"""
    trainer = new_trainer()
    trainer.train(3)
    for _ in range(3):  # 有效历史足够后再做几次辅助学习
        trainer.auxiliary_update()
    return trainer


def probe(trainer: Trainer) -> dict:
    """同一组固定输入下的确定性动作、两个 Q、alpha 与统计（只读）。"""
    history = torch.linspace(-1, 1, 4 * 10 * 33).reshape(4, 10, 33)
    q_obs = torch.linspace(-1, 1, 4 * 6).reshape(4, 6)
    with torch.no_grad():
        features = features_from_history(trainer.encoder, trainer.normalizers, history)
        action = trainer.actor.deterministic_action(features)
        q1, q2 = trainer.critics.q1(trainer.normalizers.normalize_q(q_obs), action), trainer.critics.q2(trainer.normalizers.normalize_q(q_obs), action)
        t1 = trainer.targets.q1(trainer.normalizers.normalize_q(q_obs), action)
    stats = trainer.normalizers.export_statistics()
    return {"action": action, "q1": q1, "q2": q2, "target_q1": t1, "log_alpha": trainer.temperature.log_alpha.detach().clone(),
            "policy_mean": stats["policy"]["mean"], "q_var": stats["q"]["var"],
            "counts": (stats["policy"]["count"], stats["q"]["count"]), "bn_mean": trainer.encoder.backbone[1].running_mean.clone()}


def params(trainer: Trainer) -> list[torch.Tensor]:
    return [t.detach().clone() for m in (trainer.encoder, trainer.actor, trainer.critics, trainer.targets, trainer.temperature)
            for t in m.state_dict().values()]


class RestoreTest(unittest.TestCase):
    """保存恢复后输出、统计、计数与抽样一致。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ckpt.pt"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_outputs_statistics_and_counters_match(self) -> None:
        original = trained_trainer()
        save_checkpoint(original, self.path)
        restored = new_trainer()
        report = restore_checkpoint(restored, self.path)
        a, b = probe(original), probe(restored)
        for key in a:
            with self.subTest(key=key):
                self.assertTrue(a[key] == b[key] if key == "counts" else torch.equal(a[key], b[key]))
        self.assertEqual(asdict(original.counters), asdict(restored.counters))
        for name in original.optimizers:
            sa, sb = original.optimizers[name].state_dict()["state"], restored.optimizers[name].state_dict()["state"]
            self.assertTrue(all(torch.equal(sa[i]["exp_avg"], sb[i]["exp_avg"]) for i in sa), name)
        self.assertEqual((report.mode, report.replay_restored, report.environment_state_restored), ("exact_replay", True, False))
        self.assertEqual(report.config_differences, ())

    def test_exact_replay_reproduces_next_learning(self) -> None:
        original = trained_trainer()
        save_checkpoint(original, self.path)
        restored = new_trainer()
        restore_checkpoint(restored, self.path)
        for trainer in (original, restored):
            trainer.learn_from_replay(3)
            trainer.auxiliary_update()
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(params(original), params(restored))))
        self.assertEqual(asdict(original.counters), asdict(restored.counters))
        next_a = original.buffer.sample_starts(8, 5, generator=original.generators["replay"])
        next_b = restored.buffer.sample_starts(8, 5, generator=restored.generators["replay"])
        self.assertTrue(torch.equal(next_a[0], next_b[0]) and torch.equal(next_a[1], next_b[1]))

    def test_model_only_requires_new_experience(self) -> None:
        original = trained_trainer()
        save_checkpoint(original, self.path, include_replay=False)
        restored = new_trainer()
        report = restore_checkpoint(restored, self.path)
        self.assertEqual((report.mode, restored.buffer.size), ("model_only", 0))
        self.assertIn("不是精确接续", describe_resume_mode(report))
        self.assertTrue(torch.equal(probe(original)["action"], probe(restored)["action"]))
        self.assertEqual(restored.learn_from_replay(2), {"skipped": True})  # 经验池为空，先重新积累
        restored.train(2)
        self.assertGreater(restored.counters.critic_updates, original.counters.critic_updates)

    def test_environment_restarts_without_joining_old_experience(self) -> None:
        original = trained_trainer()
        save_checkpoint(original, self.path)
        last_episode = original.observation.episode_id.clone()
        restored = new_trainer()
        restore_checkpoint(restored, self.path)
        obs = restored.observation
        self.assertTrue(torch.equal(obs.episode_id, last_episode + 1))
        self.assertEqual((obs.step_index.tolist(), obs.history_length.tolist()), ([0, 0, 0], [1, 1, 1]))
        # 每个环境最后一条旧记录（中断时未结束的一轮）不会与恢复后写入的新记录连接。
        last_slots = (restored.buffer.state_dict()["writes"] - 1) % restored.buffer.capacity_per_env
        ended = restored.buffer.read_sequence(torch.arange(3), last_slots, 1).fields
        restored.collect_steps(2)
        joined = restored.buffer.read_sequence(torch.arange(3), last_slots, 2)
        for env in range(3):
            if not bool(ended["terminated"][env, 0] | ended["time_out"][env, 0]):
                self.assertFalse(bool(joined.valid[env, 1]), f"环境 {env} 的旧记录被接到新一轮")
        self.assertTrue(bool((joined.fields["episode_id"][:, 1] == last_episode + 1).all()))


class CompatibilityTest(unittest.TestCase):
    """不兼容的检查点被拒绝；原子保存。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.trainer = new_trainer()
        self.trainer.collect_steps(3)
        self.path = save_checkpoint(self.trainer, self.dir / "ckpt.pt")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rejects_mismatched_structure(self) -> None:
        base = cpu_debug_config(seed=3)
        other_bounds = replace(base, task=replace(base.task, wheel_speed_ranges={"joint_left_leg_4": (-12.0, 20.0),
                                                                                 "joint_right_leg_4": (-12.0, 20.0)}))
        other_network = replace(base, network=replace(base.network, actor_hidden_dims=(256, 128)))
        for config, pattern in ((other_bounds, "动作边界"), (other_network, "network.actor_hidden_dims")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(CheckpointError, pattern):
                restore_checkpoint(new_trainer(config), self.path)

    def test_rejects_other_formats_and_versions(self) -> None:
        ppo_like = self.dir / "ppo_model.pt"
        torch.save({"model_state_dict": {"critic.0.weight": torch.zeros(512, 54)}, "optimizer_state_dict": {}}, ppo_like)
        with self.assertRaisesRegex(CheckpointError, "原 PPO 的 V 参数不能当作 Q 加载"):
            restore_checkpoint(new_trainer(), ppo_like)
        state = torch.load(self.path, weights_only=True)
        state["version"] = 99
        torch.save(state, self.dir / "v99.pt")
        with self.assertRaisesRegex(CheckpointError, "版本"):
            restore_checkpoint(new_trainer(), self.dir / "v99.pt")
        self.assertEqual(load_checkpoint_config(self.path), self.trainer.config)

    def test_atomic_save_keeps_previous_file(self) -> None:
        before = self.path.read_bytes()
        self.trainer.collect_steps(2)

        def failing_save(obj, handle):
            handle.write(b"partial")
            raise OSError("模拟写入中断")

        with mock.patch("checkpoint.torch.save", failing_save), self.assertRaises(OSError):
            save_checkpoint(self.trainer, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["ckpt.pt"])  # 没有残留临时文件


class FullFlowTest(unittest.TestCase):
    """一次 CPU 流程：采集 → 学习 → 训练中按间隔保存 → 恢复 → 继续训练。"""

    def test_train_save_restore_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run" / "latest.pt"
            trainer = new_trainer()
            logs = trainer.train(4, checkpoint_path=path, checkpoint_every=2)
            self.assertTrue(path.exists())
            self.assertGreater(trainer.counters.critic_updates, 0)
            restored = new_trainer(load_checkpoint_config(path))
            report = restore_checkpoint(restored, path)
            self.assertEqual(report.counters["iterations"], 4)
            more = restored.train(2)
            self.assertEqual(restored.counters.iterations, 6)
            self.assertEqual(restored.counters.critic_updates, trainer.counters.critic_updates + 2 * 4)
            self.assertTrue(all("learn" in log for log in more) and len(logs) == 4)


if __name__ == "__main__":
    unittest.main()
