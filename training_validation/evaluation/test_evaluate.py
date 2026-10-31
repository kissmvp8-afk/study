"""T04 对照评价的验证程序（unittest + PyTorch CPU，无需仿真）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/evaluation/test_evaluate.py -v

数据说明：评价记录全部是人工构造的，期望值在注释中手算；run_evaluation 的检查使用 T02 假环境（非 Isaac 仿真）。
这里不包含任何真实 PPO / SAC 结果，也不得据此给出性能结论。
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
sys.path.insert(0, str(ROOT / "training_validation" / "training_loop"))

from evaluate import (  # noqa: E402
    EvaluationEpisode,
    EvaluationRecord,
    compare_runs,
    compute_metrics,
    run_evaluation,
    validate_record,
    write_report,
)
from fake_env import FakeLocomotionEnv  # noqa: E402

PROVENANCE = {"algorithm": "sac", "model_source": "人工记录", "config_name": "manual", "code_commit": "manual", "training_seed": 0}


def manual_record(provenance=PROVENANCE, torques=False, interaction_steps=1000, wall_time_s=12.5) -> EvaluationRecord:
    """三个人工 episode：真终止（2 步）、正常到时（1 步）、未完成（3 步）。"""
    ep1 = EvaluationEpisode(
        actions=torch.stack([torch.zeros(8), torch.ones(8)]), end_reason="terminated",
        commanded_velocity=torch.tensor([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]]),
        measured_velocity=torch.tensor([[0.7, 0.4, 0.25], [1.0, 0.0, 0.5]]),
        joint_torques=torch.full((2, 8), 2.0) if torques else None)
    ep2 = EvaluationEpisode(
        actions=torch.full((1, 8), 5.0), end_reason="time_out",
        commanded_velocity=torch.zeros(1, 3), measured_velocity=torch.tensor([[0.3, 0.4, -0.1]]),
        joint_torques=torch.full((1, 8), -1.0) if torques else None)
    ep3 = EvaluationEpisode(
        actions=torch.stack([torch.zeros(8), torch.full((8,), 0.5), torch.full((8,), 0.5)]), end_reason="unfinished",
        commanded_velocity=torch.zeros(3, 3), measured_velocity=torch.zeros(3, 3),
        joint_torques=torch.zeros(3, 8) if torques else None)
    return EvaluationRecord(provenance=dict(provenance), episodes=(ep1, ep2, ep3), interaction_steps=interaction_steps,
                            wall_time_s=wall_time_s)


class MetricTest(unittest.TestCase):
    """指标与独立手算一致，统计口径明确。"""

    def test_velocity_errors_step_weighted(self) -> None:
        # 人工记录是 float32 张量（0.7、0.4 等不能精确表示），因此按 6 位小数比较。
        m = compute_metrics(manual_record())
        # 线速度误差：ep1 两步 √(0.3²+0.4²)=0.5、0；ep2 一步 0.5；ep3 三步 0 → (0.5+0+0.5+0+0+0)/6
        self.assertAlmostEqual(m["linear_velocity_error"].value, 1.0 / 6, places=6)
        self.assertEqual((m["linear_velocity_error"].unit, m["linear_velocity_error"].samples), ("m/s", 6))
        # 偏航：|0.5−0.25| + 0 + |0−(−0.1)| + 0 + 0 + 0 = 0.35，/6
        self.assertAlmostEqual(m["yaw_rate_error"].value, 0.35 / 6, places=6)

    def test_end_reasons_fall_rate_and_durations(self) -> None:
        m = compute_metrics(manual_record())
        self.assertAlmostEqual(m["fall_rate"].value, 1 / 3)  # 只有真终止算摔倒；正常到时与未完成都不算
        self.assertEqual((m["time_out_count"].value, m["unfinished_count"].value), (1.0, 1.0))
        self.assertAlmostEqual(m["stable_duration"].value, (2 + 1 + 3) * 0.02 / 3)  # 按 episode 平均，s
        self.assertAlmostEqual(m["time_to_fall"].value, 2 * 0.02)

    def test_action_rate_within_episodes_only(self) -> None:
        # ep1：(1−0)² × 8 = 8；ep2 只有 1 步，无差分；ep3：(0.5−0)² × 8 = 2，(0.5−0.5)² × 8 = 0 → 共 3 个差分，(8 + 2 + 0) / 3
        # 若跨 episode 求差（ep1 末 1 → ep2 的 5），结果会不同。
        m = compute_metrics(manual_record())
        self.assertAlmostEqual(m["action_rate"].value, (8 + 2 + 0) / 3)
        self.assertEqual(m["action_rate"].samples, 3)

    def test_missing_signals_are_marked_not_zero(self) -> None:
        m = compute_metrics(manual_record(torques=False))
        for name in ("torque_abs_mean", "torque_squared_mean", "peak_memory"):
            self.assertEqual((m[name].status, m[name].value), ("missing", None), name)
        m = compute_metrics(manual_record(torques=True))
        self.assertAlmostEqual(m["torque_abs_mean"].value, (2 * 16 + 1 * 8 + 0) / 48)  # 48 = 6 步 × 8 关节
        # 只要有一个 episode 缺速度信号，速度指标就整体标为 missing，不对部分 episode 求平均。
        partial = manual_record()
        partial = EvaluationRecord(partial.provenance, partial.episodes[:2] + (EvaluationEpisode(torch.zeros(2, 8), "time_out"),))
        self.assertEqual(compute_metrics(partial)["linear_velocity_error"].status, "missing")

    def test_rejects_unknown_end_reason(self) -> None:
        with self.assertRaises(ValueError):
            validate_record(EvaluationRecord(PROVENANCE, (EvaluationEpisode(torch.zeros(1, 8), "fell"),)))


class ReportAndComparisonTest(unittest.TestCase):
    """报告与对照：证据不足时不发布性能结论。"""

    def test_report_blocks_conclusion_without_provenance_or_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            no_source = dict(PROVENANCE, model_source=None)
            report = write_report([manual_record(no_source)], path)
            self.assertTrue(report["performance_conclusion"].startswith("不发布"))
            self.assertIn("model_source", report["performance_conclusion"])
            self.assertIn("至少 3", report["performance_conclusion"])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(saved["runs"][0]["metrics"]["torque_abs_mean"]["value"])  # 缺失写成 null，不是 0
            self.assertIn("torque_abs_mean", saved["missing_metrics"])
            three = [manual_record(dict(PROVENANCE, training_seed=s)) for s in range(3)]
            self.assertFalse(write_report(three, path)["performance_conclusion"].startswith("不发布"))

    def test_compare_requires_fixed_thresholds_seeds_and_budget(self) -> None:
        ppo = [manual_record(dict(PROVENANCE, algorithm="ppo", training_seed=s)) for s in range(3)]
        sac = [manual_record(dict(PROVENANCE, training_seed=s)) for s in range(3)]
        open_result = compare_runs(ppo, sac, "interaction_steps")
        self.assertFalse(open_result["conclusion_allowed"])
        self.assertIsNone(open_result["metrics"]["fall_rate"]["verdict"])
        self.assertTrue(any("阈值尚未固定" in b for b in open_result["conclusion_blockers"]))
        fixed = compare_runs(ppo, sac, "interaction_steps", thresholds={"fall_rate": 0.0})
        self.assertTrue(fixed["conclusion_allowed"])
        self.assertEqual(fixed["metrics"]["fall_rate"]["verdict"], "通过")  # 差值 0 ≤ 0（人工记录相同）
        self.assertEqual(fixed["metrics"]["torque_abs_mean"], {"status": "missing"})
        no_budget = compare_runs(ppo[:2], sac, "wall_time", thresholds={})
        self.assertFalse(no_budget["conclusion_allowed"])
        self.assertTrue(any("2 个不同的有效训练随机起点" in b and "少于 3 个" in b for b in no_budget["conclusion_blockers"]))


class BudgetAndSeedRegressionTest(unittest.TestCase):
    """回归：预算必须可比；随机起点按不同的有效种子计数。"""

    @staticmethod
    def group(algorithm, seeds=(0, 1, 2), **budget):
        return [manual_record(dict(PROVENANCE, algorithm=algorithm, training_seed=s), **budget) for s in seeds]

    def assert_blocked(self, result, pattern):
        self.assertFalse(result["conclusion_allowed"])
        self.assertTrue(all(row.get("verdict") is None for row in result["metrics"].values()))
        self.assertTrue(any(pattern in b for b in result["conclusion_blockers"]), result["conclusion_blockers"])

    def test_interaction_steps_differ_by_1000x(self) -> None:
        result = compare_runs(self.group("ppo", interaction_steps=1_000), self.group("sac", interaction_steps=1_000_000),
                              "interaction_steps", thresholds={"fall_rate": 0.0})
        self.assert_blocked(result, "interaction_steps 预算不满足可比条件")
        self.assertIn("difference", result["metrics"]["fall_rate"])  # 差异照常展示，只是不判定
        self.assertFalse(result["budget_check"]["comparable"])

    def test_wall_time_outside_tolerance(self) -> None:
        result = compare_runs(self.group("ppo", wall_time_s=12.5), self.group("sac", wall_time_s=20.0),
                              "wall_time", thresholds={"fall_rate": 0.0})
        self.assert_blocked(result, "wall_time 预算不满足可比条件")
        self.assertIn("0.05", result["budget_check"]["rule"])  # 容差写入结果
        # 对照：相差 4%（12.5 → 13.0）在 5% 暂定容差内，可以判定。
        within = compare_runs(self.group("ppo", wall_time_s=12.5), self.group("sac", wall_time_s=13.0),
                              "wall_time", thresholds={"fall_rate": 0.0})
        self.assertTrue(within["conclusion_allowed"])
        self.assertEqual(within["metrics"]["fall_rate"]["verdict"], "通过")

    def test_three_records_with_same_seed_are_one_seed(self) -> None:
        same = self.group("sac", seeds=(7, 7, 7))
        with tempfile.TemporaryDirectory() as tmp:
            report = write_report(same, Path(tmp) / "report.json")
        self.assertTrue(report["performance_conclusion"].startswith("不发布"))
        self.assertIn("只有 1 个不同的有效训练随机起点（3 份记录）", report["performance_conclusion"])
        self.assertEqual(report["summary"]["training_seeds"], [7])
        self.assert_blocked(compare_runs(self.group("ppo"), same, "interaction_steps", thresholds={"fall_rate": 0.0}),
                            "candidate 只有 1 个不同的有效训练随机起点")

    def test_matching_budget_and_three_seeds_still_pass(self) -> None:
        result = compare_runs(self.group("ppo"), self.group("sac"), "interaction_steps", thresholds={"fall_rate": 0.0})
        self.assertTrue(result["conclusion_allowed"] and result["budget_check"]["comparable"])
        self.assertEqual(result["metrics"]["fall_rate"]["verdict"], "通过")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(write_report(self.group("sac"), Path(tmp) / "r.json")["performance_conclusion"].startswith("不发布"))


class RunEvaluationTest(unittest.TestCase):
    """在 T02 假环境上执行确定性评价：结束原因区分正确，缺失信号如实标记。"""

    def test_fake_environment_rollout(self) -> None:
        env = FakeLocomotionEnv(num_envs=2, q_dim=6, max_episode_steps=5, termination_steps=(3, None))
        policy = lambda obs: torch.zeros(obs.policy_history.shape[0], 8)  # noqa: E731
        record = run_evaluation(policy, env, num_steps=7, provenance=PROVENANCE)
        reasons = sorted(ep.end_reason for ep in record.episodes)
        # 环境 0：第 3 步真终止 ×2，剩 1 步未完成；环境 1：第 5 步到时，剩 2 步未完成。
        self.assertEqual(reasons, ["terminated", "terminated", "time_out", "unfinished", "unfinished"])
        self.assertEqual(sum(ep.steps for ep in record.episodes), 14)
        metrics = compute_metrics(record)
        self.assertAlmostEqual(metrics["fall_rate"].value, 2 / 5)
        self.assertEqual(metrics["linear_velocity_error"].status, "missing")  # 假环境不提供速度指令与真实速度
        signals = lambda step: {"commanded_velocity": torch.zeros(2, 3), "measured_velocity": step.next_q_observation[:, :3] / 2.0}  # noqa: E731
        with_signals = compute_metrics(run_evaluation(policy, env, 7, PROVENANCE, signals=signals))
        self.assertEqual(with_signals["linear_velocity_error"].status, "measured")
        self.assertEqual(with_signals["torque_abs_mean"].status, "missing")


if __name__ == "__main__":
    unittest.main()
