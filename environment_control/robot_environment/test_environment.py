"""E01 机器人与任务环境的 CPU 验证程序（unittest + PyTorch CPU，无需 Isaac Lab）。

运行：在仓库根目录执行（macOS / Linux 用 / 分隔路径，Windows PowerShell 用 \\ 也可以）
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B environment_control/robot_environment/test_environment.py -v
可选：设置环境变量 TITA_URDF 指向锁定提交 e7b32d9 的 robot.urdf，额外核对真实资产（不设置则跳过该项并在输出中显示 skipped）：
    PowerShell：$env:TITA_URDF = "<DDT_Lab 根目录>\\ddt_ros2_control\\urdfs\\tita_description\\urdf\\robot.urdf"

范围说明：
    - 纯数据部分（assets.py、metadata.py）用人工 URDF 与人工 RuntimeMetadata 检查规则与报错；
    - isaac_env.py / run_sac_isaac.py 需要 Isaac，这里只做语法与结构检查（钩子名称与 E05 调用一致、SAC 只改历史长度、
      注册 id 不与 PPO 重名、Isaac 模块不在启动前导入）。这些都**不是** Isaac 仿真验证。
    - 测试中的 TITA 限位数值抄自锁定 URDF（e7b32d9），只作为测试数据；运行时以 Isaac 读到的值为准。
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for path in ("environment_control/episode_lifecycle", "environment_control/observation_interface",
             "environment_control/action_contract", "training_validation/configuration", "training_validation/training_loop"):
    sys.path.insert(0, str(ROOT / path))

from assets import (ROBOT_ASSET, WHEEL_JOINTS, AssetError, parse_urdf_joints, sha256_of, soft_joint_limits,  # noqa: E402
                    urdf_path, verify_urdf, wheel_half_track, wheel_radius)
from baseline_source import load_baseline_action_contract  # noqa: E402
from config import TARGET_ENVIRONMENT, ConfigurationError, validate_configuration  # noqa: E402
from contract import ControlType  # noqa: E402
from metadata import (DEBUG_WINDOWS, PPO_TASK_IDS, SAC_TASK_IDS, MetadataError, RuntimeMetadata, baseline_command_ranges,  # noqa: E402
                      baseline_timing, compare_with_baseline, fill_isaac_run_config, required_wheel_speed, wheel_speed_ranges)
from observations import GROUP_FIELDS, expected_q_dim  # noqa: E402

# 锁定 URDF e7b32d9 中腿部关节的位置限位（rad），仅作测试数据。
TITA_URDF_LIMITS = {"leg_1": (-0.785398, 0.785398), "leg_2": (-1.919862, 3.490659), "leg_3": (-2.670354, -0.698132)}

SYNTHETIC_URDF = """<?xml version="1.0"?>
<robot name="two_wheel_test">
  <link name="base"/>
  <link name="left_hip"/><link name="left_wheel"><collision><geometry><cylinder length="0.03" radius="0.1"/></geometry></collision></link>
  <link name="right_hip"/><link name="right_wheel"><collision><geometry><cylinder length="0.03" radius="0.1"/></geometry></collision></link>
  <joint name="left_roll" type="revolute"><origin xyz="0 0.1 0" rpy="0 0 0"/><parent link="base"/><child link="left_hip"/>
    <axis xyz="1 0 0"/><limit lower="-0.5" upper="1.5" effort="10" velocity="5"/></joint>
  <joint name="joint_left_leg_4" type="continuous"><origin xyz="0 0.2 -0.3" rpy="0 0 0"/><parent link="left_hip"/><child link="left_wheel"/>
    <axis xyz="0 1 0"/></joint>
  <joint name="right_roll" type="revolute"><origin xyz="0 -0.1 0" rpy="0 0 0"/><parent link="base"/><child link="right_hip"/>
    <axis xyz="1 0 0"/><limit lower="-1.5" upper="0.5" effort="10" velocity="5"/></joint>
  <joint name="joint_right_leg_4" type="continuous"><origin xyz="0 -0.2 -0.3" rpy="0 0 0"/><parent link="right_hip"/><child link="right_wheel"/>
    <axis xyz="0 1 0"/></joint>
</robot>
"""


def write_fixture(directory: Path, with_meshes: bool = True) -> Path:
    """在 directory 下按 tita_description/urdf/robot.urdf 的布局写入人工 URDF（可选创建 meshes 目录）。"""
    path = directory / "tita_description" / "urdf" / "robot.urdf"
    path.parent.mkdir(parents=True)
    path.write_text(SYNTHETIC_URDF, encoding="utf-8")
    if with_meshes:
        (directory / "tita_description" / "meshes").mkdir()
    return path


def soft(limits, factor=0.9):
    mean, span = (limits[0] + limits[1]) / 2, limits[1] - limits[0]
    return mean - 0.5 * span * factor, mean + 0.5 * span * factor


def make_runtime(contract, task="isaac_flat", **changes) -> RuntimeMetadata:
    """按原版应有的样子构造运行时元数据（人工数据）；changes 用来制造单项错误。

    关节顺序故意与动作顺序不同（articulation 顺序由 PhysX 决定），用来确认核对不依赖两者相同。
    """
    names = tuple(sorted(contract.joint_names))
    limits = {n: soft(TITA_URDF_LIMITS[n.split("_", 2)[2]]) for n in names if not n.endswith("leg_4")}
    limits.update({n: soft((-62831.85, 62831.85)) for n in WHEEL_JOINTS})
    terms, seen = [], {}
    for e in contract.entries:
        kind = "position" if e.control_type is ControlType.POSITION else "velocity"
        if e.term_name not in seen:
            seen[e.term_name] = len(terms)
            terms.append([e.term_name, [], e.scale, kind])
        terms[seen[e.term_name]][1].append(e.joint_name)
    groups = ("policy", "critic", "priv") + (("scanner",) if task == "isaac_rough" else ())
    values = dict(
        task=task, num_envs=3, sim_device="cuda:0", joint_names=names,
        default_joint_pos={e.joint_name: e.default_target if e.control_type is ControlType.POSITION else 0.0 for e in contract.entries},
        soft_joint_pos_limits=limits, joint_vel_limits={n: 30.0 if n in WHEEL_JOINTS else 1e6 for n in names},
        action_terms=tuple((t[0], tuple(t[1]), t[2], t[3]) for t in terms),
        physics_dt=0.005, decimation=4, step_dt=0.02, max_episode_length=1000,
        obs_term_names={g: [f.name for f in GROUP_FIELDS[g]] for g in groups},
        obs_term_dims={g: [(f.dim,) for f in GROUP_FIELDS[g]] for g in groups},
        command_ranges={"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    )
    values.update(changes)
    return RuntimeMetadata(**values)


class AssetTest(unittest.TestCase):
    """URDF 解析、软限位公式、轮子几何与资产校验。"""

    def test_parse_soft_limits_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(Path(tmp))
            joints = parse_urdf_joints(path)
            self.assertEqual(list(joints), ["left_roll", "joint_left_leg_4", "right_roll", "joint_right_leg_4"])
            self.assertEqual(joints["left_roll"].axis, (1.0, 0.0, 0.0))
            limits = soft_joint_limits(joints, ["left_roll", "right_roll"], 0.9)
            self.assertAlmostEqual(limits["left_roll"][0], 0.5 - 0.9)  # 中点 0.5，半宽 1.0 × 0.9
            self.assertAlmostEqual(limits["left_roll"][1], 0.5 + 0.9)
            with self.assertRaisesRegex(AssetError, "没有位置限位"):
                soft_joint_limits(joints, ["joint_left_leg_4"], 0.9)  # continuous 关节
            self.assertEqual(wheel_radius(path, "joint_left_leg_4"), 0.1)
            # 零位：轮心 y = ±(0.1 + 0.2) ⇒ 半轮距 0.3。
            self.assertAlmostEqual(wheel_half_track(joints, {}), 0.3)
            # 绕 x 轴转 θ 后轮心 y = 0.1 + 0.2 cosθ + 0.3 sinθ（左），右侧同号转动：−0.1 − 0.2 cosθ + 0.3 sinθ。
            theta = 0.2
            expected = (0.1 + 0.2 * math.cos(theta) + 0.3 * math.sin(theta) - (-0.1 - 0.2 * math.cos(theta) + 0.3 * math.sin(theta))) / 2
            self.assertAlmostEqual(wheel_half_track(joints, {"left_roll": theta, "right_roll": theta}), expected)

    def test_verify_reports_path_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nowhere" / "robot.urdf"
            with self.assertRaisesRegex(AssetError, "找不到 TITA URDF"):
                verify_urdf(missing)
            path = write_fixture(Path(tmp))
            with self.assertRaisesRegex(AssetError, "校验值"):
                verify_urdf(path)  # 人工 URDF 与锁定版本不同
            with mock.patch.dict(ROBOT_ASSET, urdf_sha256=sha256_of(path)):
                self.assertEqual(verify_urdf(path), "exact")
                # Windows git autocrlf 把换行改成 CRLF：内容相同，记为 crlf_normalized；内容改动仍然报错。
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
                self.assertEqual(verify_urdf(path), "crlf_normalized")
                path.write_bytes(path.read_bytes().replace(b"0.1", b"0.2"))
                with self.assertRaisesRegex(AssetError, "校验值"):
                    verify_urdf(path)
            other = write_fixture(Path(tmp) / "no_meshes", with_meshes=False)
            with mock.patch.dict(ROBOT_ASSET, urdf_sha256=sha256_of(other)), self.assertRaisesRegex(AssetError, "网格目录"):
                verify_urdf(other)

    def test_paths_and_pinned_commit_consistent(self) -> None:
        # 用 pathlib 拼接（Windows 与 POSIX 都按各自分隔符）：根目录/ddt_ros2_control/urdfs/tita_description/urdf/robot.urdf
        self.assertEqual(urdf_path(Path("D:/work/DDT_Lab")).parts[-5:],
                         ("ddt_ros2_control", "urdfs", "tita_description", "urdf", "robot.urdf"))
        upstream = json.loads((ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))["robot_assets"]
        self.assertEqual(ROBOT_ASSET["commit"], upstream["commit"])
        self.assertEqual(ROBOT_ASSET["urdf_sha256"], upstream["urdf_sha256"])
        self.assertEqual(ROBOT_ASSET["commit"], TARGET_ENVIRONMENT["robot_assets_commit"])

    @unittest.skipUnless(os.environ.get("TITA_URDF"), "未设置 TITA_URDF，跳过真实资产核对")
    def test_real_urdf(self) -> None:
        path = Path(os.environ["TITA_URDF"])
        self.assertEqual(sha256_of(path), ROBOT_ASSET["urdf_sha256"])
        joints = parse_urdf_joints(path)
        contract = load_baseline_action_contract()
        self.assertTrue(set(contract.joint_names) <= set(joints))  # 8 个受控关节都在 URDF 中
        for name in contract.joint_names:
            if not name.endswith("leg_4"):
                self.assertEqual((joints[name].lower, joints[name].upper), TITA_URDF_LIMITS[name.split("_", 2)[2]])
        self.assertAlmostEqual(wheel_radius(path, WHEEL_JOINTS[0]), 0.0925)
        pose = {e.joint_name: e.default_target for e in contract.entries if e.control_type is ControlType.POSITION}
        half_track = wheel_half_track(joints, pose)
        self.assertAlmostEqual(half_track, 0.2825, places=3)
        self.assertAlmostEqual(required_wheel_speed(baseline_command_ranges(), half_track, 0.0925), 13.86, places=1)


class BaselineMetadataTest(unittest.TestCase):
    """从固定快照读取周期与指令；轮速依据与检查。"""

    def test_timing_and_commands(self) -> None:
        timing = baseline_timing()
        self.assertEqual((timing["physics_dt"], timing["decimation"], timing["episode_length_s"]), (0.005, 4, 20.0))
        self.assertAlmostEqual(timing["step_dt"], 0.02)
        self.assertEqual(timing["max_episode_length"], 1000)
        self.assertEqual(baseline_command_ranges(), {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)})

    def test_wheel_speed(self) -> None:
        commands = {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        required = required_wheel_speed(commands, 0.2825, 0.0925)
        self.assertAlmostEqual(required, (1.0 + 0.2825) / 0.0925)
        with self.assertRaisesRegex(MetadataError, "lin_vel_y"):
            required_wheel_speed(dict(commands, lin_vel_y=(-0.5, 0.5)), 0.2825, 0.0925)
        for bad in (None, float("nan"), 10.0, 31.0):
            with self.subTest(limit=bad), self.assertRaises(MetadataError):
                wheel_speed_ranges(bad, required)
        self.assertEqual(wheel_speed_ranges(20.0, required), {name: (-20.0, 20.0) for name in WHEEL_JOINTS})


class RuntimeComparisonTest(unittest.TestCase):
    """运行时元数据与原版 / E03 / URDF 的逐项核对（人工 RuntimeMetadata）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_baseline_action_contract()
        cls.timing, cls.commands = baseline_timing(), baseline_command_ranges()
        cls.urdf_soft = {n: soft(TITA_URDF_LIMITS[n.split("_", 2)[2]]) for n in cls.contract.joint_names if not n.endswith("leg_4")}

    def compare(self, runtime):
        return compare_with_baseline(runtime, self.contract, self.timing, self.commands, self.urdf_soft)

    def test_matching_runtime_returns_q_dim(self) -> None:
        for task in ("isaac_flat", "isaac_rough"):
            self.assertEqual(self.compare(make_runtime(self.contract, task)), expected_q_dim(task))

    def test_each_mismatch_is_reported(self) -> None:
        good = make_runtime(self.contract)
        terms = list(good.action_terms)
        cases = {
            "动作关节顺序": dict(action_terms=tuple(reversed(terms))),
            "scale": dict(action_terms=tuple((n, j, s * 2, k) for n, j, s, k in terms)),
            "默认角": dict(default_joint_pos=dict(good.default_joint_pos, joint_left_leg_2=0.0)),
            "step_dt": dict(step_dt=0.01),
            "max_episode_length": dict(max_episode_length=500),
            "指令 ang_vel_z": dict(command_ranges=dict(good.command_ranges, ang_vel_z=(-2.0, 2.0))),
            "软限位": dict(soft_joint_pos_limits=dict(good.soft_joint_pos_limits, joint_right_leg_3=(-2.0, -1.0))),
            "观测组 priv": dict(obs_term_dims=dict(good.obs_term_dims, priv=[(4,), (8,), (8,)])),
            "缺少受控关节": dict(joint_names=good.joint_names[1:]),
        }
        for pattern, change in cases.items():
            with self.subTest(pattern=pattern), self.assertRaisesRegex(MetadataError, pattern):
                self.compare(replace(good, **change))


class RunConfigTest(unittest.TestCase):
    """运行时元数据 + 用户轮速 → T01 配置；以及与现有 Trainer 的衔接。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_baseline_action_contract()

    def fill(self, **kwargs):
        runtime = make_runtime(self.contract)
        values = dict(seed=3, steps_per_iteration=8, capacity_per_env=64, debug_sizes=True)
        values.update(kwargs)
        return fill_isaac_run_config(runtime, self.contract, 54, wheel_speed_ranges(20.0, 13.86), 13.86, **values)

    def test_filled_config(self) -> None:
        config = self.fill()
        validate_configuration(config)
        t, c = config.task, config.collection
        self.assertEqual((config.device, config.mode, t.q_dim, c.num_envs), ("cpu", "isaac_sim", 54, 3))
        self.assertEqual(set(t.joint_pos_limits), {e.joint_name for e in self.contract.entries if e.control_type is ControlType.POSITION})
        self.assertEqual(t.wheel_speed_ranges, {name: (-20.0, 20.0) for name in WHEEL_JOINTS})
        self.assertFalse(t.action_limits_artificial)
        self.assertEqual((config.environment["sim_device"], config.environment["robot_assets_commit"]), ("cuda:0", ROBOT_ASSET["commit"]))
        self.assertEqual(config.sources["collection.capacity_per_env"], DEBUG_WINDOWS)
        self.assertIn("13.86", config.sources["task.wheel_speed_ranges"])
        self.assertEqual(self.fill(debug_sizes=False).sources["collection.capacity_per_env"], "用户设置")
        with self.assertRaisesRegex(ConfigurationError, "CPU"):
            validate_configuration(replace(config, device="cuda:0"))

    def test_trainer_runs_on_filled_config(self) -> None:
        """填好的 Isaac 配置能驱动现有 Trainer（钩子为 E05 测试中的人工模拟，不是 Isaac）。"""
        from lifecycle import EpisodeLifecycle
        from runner import Trainer
        from test_lifecycle import ScriptedSimulation

        config = self.fill()
        config = replace(config, algorithm=replace(config.algorithm, batch_size=8, auxiliary_batch_size=8))
        trainer = Trainer(config, EpisodeLifecycle(ScriptedSimulation(), "isaac_flat", 54), self.contract)
        logs = trainer.train(2)
        self.assertEqual(trainer.counters.env_steps, 16)
        self.assertIn("learn", logs[0])


class IsaacFileStructureTest(unittest.TestCase):
    """isaac_env.py / run_sac_isaac.py 的语法与结构检查（不导入 Isaac，不是仿真验证）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.isaac = ast.parse((HERE / "isaac_env.py").read_text(encoding="utf-8"))
        cls.launcher = ast.parse((HERE / "run_sac_isaac.py").read_text(encoding="utf-8"))

    def methods(self, class_name: str) -> set[str]:
        node = next(n for n in self.isaac.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}

    def test_hooks_match_lifecycle_calls(self) -> None:
        lifecycle = ast.parse((ROOT / "environment_control/episode_lifecycle/lifecycle.py").read_text(encoding="utf-8"))
        called = {n.attr for n in ast.walk(lifecycle)
                  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "hooks"}
        called |= {n.attr for n in ast.walk(lifecycle) if isinstance(n, ast.Attribute) and ast.unparse(n.value) == "self.hooks"}
        methods = self.methods("TitaSacHooks")
        attributes = {ast.unparse(t)[5:] for n in ast.walk(self.isaac) if isinstance(n, ast.Assign) for t in n.targets
                      if ast.unparse(t).startswith("self.")}
        self.assertTrue(called, "没有从 lifecycle.py 读到钩子调用")
        self.assertEqual(called - methods - attributes, set())

    def test_sac_cfg_changes_only_history(self) -> None:
        for name in ("TitaSacFlatEnvCfg", "TitaSacRoughEnvCfg"):
            node = next(n for n in self.isaac.body if isinstance(n, ast.ClassDef) and n.name == name)
            post_init = next(i for i in node.body if isinstance(i, ast.FunctionDef) and i.name == "__post_init__")
            assigns = [(ast.unparse(s.targets[0]), ast.literal_eval(s.value)) for s in post_init.body if isinstance(s, ast.Assign)]
            self.assertEqual(assigns, [("self.observations.policy.history_length", 0)])
        allowed = next(n for n in self.isaac.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "ALLOWED_SAC_DIFFERENCES")
        self.assertEqual(ast.literal_eval(allowed.value), {"observations.policy.history_length": (10, 0)})

    def test_task_ids_do_not_override_ppo(self) -> None:
        self.assertFalse(set(SAC_TASK_IDS.values()) & set(PPO_TASK_IDS.values()))
        self.assertEqual(PPO_TASK_IDS, {"isaac_flat": "DDT-Velocity-Flat-Tita-v0", "isaac_rough": "DDT-Velocity-Rough-Tita-v0"})

    def test_launcher_imports_isaac_modules_after_app_start(self) -> None:
        top_level = {alias.name for n in self.launcher.body if isinstance(n, (ast.Import, ast.ImportFrom))
                     for alias in n.names} | {n.module for n in self.launcher.body if isinstance(n, ast.ImportFrom)}
        self.assertEqual(top_level & {"isaac_env", "torch", "gymnasium", "runner", "lifecycle"}, set())
        self.assertIn("AppLauncher", top_level)  # 只有 AppLauncher 在模块顶层导入


if __name__ == "__main__":
    unittest.main()
