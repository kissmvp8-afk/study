"""T01 版本、依赖与训练配置的验证程序（unittest + PyTorch CPU，无需仿真）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/configuration/test_config.py -v

覆盖：配置命名与模式、来源齐全、未确定项与人工限位的拒绝、导出与加载往返、版本记录、经验池估算。
配置值在各模块中“实际生效”的检查在 T02 test_runner.py 中进行（需要构造 Trainer）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experience_returns" / "replay_buffer"))

from buffer import estimate_storage_bytes  # noqa: E402
from config import (  # noqa: E402
    AUTHOR_ALGORITHM,
    PAPER_ALGORITHM,
    ConfigurationError,
    check_memory_budget,
    cpu_debug_config,
    estimate_memory,
    export_run_configuration,
    isaac_config,
    load_run_configuration,
    undetermined_fields,
    validate_configuration,
)


class PresetTest(unittest.TestCase):
    """调试 / 仿真配置的命名、来源与未确定项。"""

    def test_names_modes_and_sources(self) -> None:
        debug, flat, rough = cpu_debug_config(), isaac_config("isaac_flat"), isaac_config("isaac_rough")
        self.assertEqual(len({debug.name, flat.name, rough.name}), 3)
        self.assertEqual((debug.mode, flat.mode), ("cpu_debug", "isaac_sim"))
        validate_configuration(debug)  # 调试配置可以运行
        for config in (flat, rough):
            validate_configuration(config, for_run=False)  # 来源齐全、组合合法
            self.assertIn("调试", debug.sources["algorithm.batch_size"])
            self.assertIn("论文候选", config.sources["algorithm.gamma"])
        self.assertIn("人工数据", debug.sources["task.joint_pos_limits"])

    def test_paper_and_author_values(self) -> None:
        self.assertEqual((PAPER_ALGORITHM.gamma, PAPER_ALGORITHM.n_steps, PAPER_ALGORITHM.tau, PAPER_ALGORITHM.initial_std),
                         (0.97, 5, 0.003, 0.15))
        self.assertEqual((PAPER_ALGORITHM.actor_lr, PAPER_ALGORITHM.critic_lr, PAPER_ALGORITHM.alpha_lr), (2e-4, 2e-4, 2e-5))
        self.assertEqual((AUTHOR_ALGORITHM.gamma, AUTHOR_ALGORITHM.tau, AUTHOR_ALGORITHM.actor_update_every,
                          AUTHOR_ALGORITHM.initial_alpha), (0.998, 0.005, 2, 0.05))

    def test_isaac_config_refuses_to_run_with_undetermined_fields(self) -> None:
        flat = isaac_config("isaac_flat")
        self.assertEqual(set(undetermined_fields(flat)), {
            "task.q_dim", "task.joint_pos_limits", "task.wheel_speed_ranges",
            "collection.steps_per_iteration", "collection.capacity_per_env",
        })
        self.assertEqual(flat.environment["robot_assets_commit"], "e7b32d9746df900680641ff2845ffb0b5ffe99ed")  # E01 锁定
        self.assertEqual(flat.device, "cpu")  # 学习器只实现了 CPU 版本
        with self.assertRaisesRegex(ConfigurationError, "task.q_dim 未确定"):
            validate_configuration(flat)

    def test_artificial_limits_rejected_for_simulation(self) -> None:
        debug = cpu_debug_config()
        leaked = replace(isaac_config("isaac_flat"), task=replace(debug.task, task="isaac_flat"))
        with self.assertRaisesRegex(ConfigurationError, "人工限位不能用于 Isaac"):
            validate_configuration(leaked, for_run=False)

    def test_invalid_combinations(self) -> None:
        debug = cpu_debug_config()
        cases = [
            (replace(debug, algorithm=replace(debug.algorithm, gamma=0.0)), "gamma"),
            (replace(debug, algorithm=replace(debug.algorithm, n_steps=100)), "超过 capacity_per_env"),
            (replace(debug, task=replace(debug.task, velocity_label_columns=(4, 7))), "超出 q_dim"),
            (replace(debug, sources={k: v for k, v in debug.sources.items() if k != "algorithm.tau"}), "algorithm.tau 缺少来源"),
            (replace(debug, mode="isaac_sim"), "isaac_flat 或 isaac_rough"),
            (replace(debug, device="meta"), "学习器只支持 cpu 或 cuda"),  # 拒绝未实现的后端
        ]
        for config, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ConfigurationError, pattern):
                validate_configuration(config)


class ExportTest(unittest.TestCase):
    """导出、加载往返与版本记录。"""

    def test_round_trip_and_versions(self) -> None:
        for config in (cpu_debug_config(seed=7), isaac_config("isaac_rough")):
            with self.subTest(name=config.name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run.json"
                record = export_run_configuration(config, path, ROOT)
                self.assertEqual(load_run_configuration(path), config)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)
                head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
                self.assertEqual(record["versions"]["git_commit"], head)
                self.assertEqual(record["versions"]["robot_assets_commit"], "e7b32d9746df900680641ff2845ffb0b5ffe99ed")
                self.assertEqual(record["undetermined_fields"], undetermined_fields(config))

    def test_rejects_other_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps({"format_version": 99, "config": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "格式版本"):
                load_run_configuration(path)


class MemoryTest(unittest.TestCase):
    """按 D01 / D02 的真实字段估算经验池。"""

    def test_estimate_matches_buffer_and_names_largest_part(self) -> None:
        debug = cpu_debug_config()
        estimate = estimate_memory(debug)
        self.assertEqual(estimate["total_bytes"], estimate_storage_bytes(3, 64, 6))
        largest = next(iter(estimate["by_field_bytes"]))
        self.assertIn(largest, ("policy_history", "next_policy_history"))  # 两段 [10,33] float32 历史占比最大
        with self.assertRaisesRegex(ConfigurationError, f"占用最多的是 {largest}"):
            check_memory_budget(estimate, available_bytes=1000)

    def test_isaac_scenario_requires_explicit_values(self) -> None:
        rough = isaac_config("isaac_rough")
        with self.assertRaisesRegex(ConfigurationError, "未确定"):
            estimate_memory(rough)
        scenario = estimate_memory(rough, q_dim=241, capacity_per_env=1221)  # E02 预期 C 与约 500 万条，仅作场景
        self.assertEqual(scenario["total_bytes"], 4096 * 1221 * 4654)
        self.assertAlmostEqual(scenario["total_bytes"] / 1e9, 23.28, delta=0.01)


if __name__ == "__main__":
    unittest.main()
