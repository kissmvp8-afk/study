"""E04 动作边界与初始中心的验证程序（标准库 unittest，无需 Isaac Lab 或仿真）。

运行：在仓库根目录执行
    python3 -B environment_control/action_bounds/test_bounds.py -v

数据说明：本文件中的软限位与工作轮速全部是**人工数据**，只用于检查公式，不是 TITA 实际限位，
也不代表已验证的工作轮速。默认目标和 scale 来自 E03 动作表（原版快照或同结构的人工动作表）。

预期值来源：按 README 公式手算（L=(下限-默认)/scale 等），在各案例注释中写出算式；
回到物理目标的检查使用规范公式 a=b+c*tanh(u) 与 E03 的 action_to_targets，不调用被测函数自证。
不覆盖：Actor 的真实输出、随机采样动作的分布，以及仿真中的开局姿态。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "action_contract"))

from baseline_source import load_baseline_action_contract, read_baseline_file, ENV_CFG_PATH  # noqa: E402
from bounds import ActionBoundsError, calibrate_initial_center, derive_bounds, validate_bounds  # noqa: E402
from contract import ActionTermSpec, ControlType, action_to_targets, build_action_contract  # noqa: E402

P, V = ControlType.POSITION, ControlType.VELOCITY


def single_position_contract(default: float = 0.8, scale: float = 0.25):
    """单个位置关节 "j" 的人工动作表，用于检查单维公式。"""
    return build_action_contract([ActionTermSpec("p", P, ("j",), scale)], {"j": default}, {"j": 0.0})


def deterministic_action(bounds, u):
    """规范公式 a = b + c * tanh(u)：确定性动作由内部中心 u 映射得到。"""
    return [b + c * math.tanh(x) for b, c, x in zip(bounds.center, bounds.half_width, u)]


def snapshot_available() -> bool:
    """固定快照可读取时返回 True（与 E03 测试相同的条件）。"""
    try:
        read_baseline_file(ENV_CFG_PATH)
    except (RuntimeError, FileNotFoundError):
        return False
    return True


# 人工数据（非 TITA 实际限位）：故意设成非对称，并给轮子也写了角度限位，用来确认速度项不会读取它。
ARTIFICIAL_JOINT_POS_LIMITS = {
    "joint_left_leg_1": (-0.3, 0.7),
    "joint_left_leg_2": (0.0, 1.6),
    "joint_left_leg_3": (-2.4, -0.6),
    "joint_left_leg_4": (-3.14, 3.14),
    "joint_right_leg_1": (-0.3, 0.7),
    "joint_right_leg_2": (0.0, 1.6),
    "joint_right_leg_3": (-2.4, -0.6),
    "joint_right_leg_4": (-3.14, 3.14),
}
ARTIFICIAL_WHEEL_SPEED_RANGES = {"joint_left_leg_4": (-10.0, 20.0), "joint_right_leg_4": (-10.0, 20.0)}


class SingleJointTest(unittest.TestCase):
    """单维公式与非对称校准。"""

    def test_readme_example_symmetric_bounds(self) -> None:
        # (0.3-0.8)/0.25 = -2，(1.3-0.8)/0.25 = 2。
        contract = single_position_contract()
        bounds = derive_bounds(contract, {"j": (0.3, 1.3)}, {})
        self.assertAlmostEqual(bounds.lower[0], -2.0, places=12)
        self.assertAlmostEqual(bounds.upper[0], 2.0, places=12)
        self.assertAlmostEqual(bounds.center[0], 0.0, places=12)
        self.assertAlmostEqual(bounds.half_width[0], 2.0, places=12)
        # 对称区间、目标为默认姿态时 u0=0，与作者的近零初始化一致。
        self.assertAlmostEqual(calibrate_initial_center(contract, bounds)[0], 0.0, places=12)

    def test_asymmetric_center_returns_to_desired_target(self) -> None:
        # 软限位 [0.5, 1.3]：L=(0.5-0.8)/0.25=-1.2，U=2，b=0.4，c=1.6。
        contract = single_position_contract()
        bounds = derive_bounds(contract, {"j": (0.5, 1.3)}, {})
        self.assertAlmostEqual(bounds.center[0], 0.4, places=12)
        self.assertAlmostEqual(bounds.half_width[0], 1.6, places=12)
        # 若像作者那样 u=0，确定性动作是 b=0.4，目标 0.8+0.25*0.4=0.9，偏离默认姿态。
        self.assertAlmostEqual(action_to_targets(contract, deterministic_action(bounds, [0.0]))[0], 0.9, places=12)
        for desired in (0.8, 1.0, 0.6):
            with self.subTest(desired=desired):
                u0 = calibrate_initial_center(contract, bounds, [desired])
                target = action_to_targets(contract, deterministic_action(bounds, u0))[0]
                self.assertAlmostEqual(target, desired, places=12)
        # 默认目标：a0=0，tanh(u0)=(0-0.4)/1.6=-0.25。
        self.assertAlmostEqual(calibrate_initial_center(contract, bounds)[0], math.atanh(-0.25), places=12)


@unittest.skipUnless(snapshot_available(), "本地缺少固定快照，无法生成 E03 原版动作表")
class EightDimMixedTest(unittest.TestCase):
    """用 E03 由原版快照生成的八维动作表，检查腿部与轮速分别使用正确的范围和 scale。"""

    def setUp(self) -> None:
        self.contract = load_baseline_action_contract()
        self.bounds = derive_bounds(self.contract, ARTIFICIAL_JOINT_POS_LIMITS, ARTIFICIAL_WHEEL_SPEED_RANGES)

    def test_bounds_per_control_type(self) -> None:
        expected = {
            0: (-1.6, 2.4),  # 左腿 1：((-0.3-0.1)/0.25, (0.7-0.1)/0.25)
            1: (-3.2, 3.2),  # 左腿 2：((0.0-0.8)/0.25, (1.6-0.8)/0.25)
            2: (-3.6, 3.6),  # 左腿 3：((-2.4+1.5)/0.25, (-0.6+1.5)/0.25)
            3: (-2.0, 4.0),  # 左轮：(-10/5, 20/5)，用轮速与 scale=5；不是角度限位 ±3.14 的换算
            4: (-1.6, 2.4),
            5: (-3.2, 3.2),
            6: (-3.6, 3.6),
            7: (-2.0, 4.0),
        }
        self.assertEqual(self.bounds.joint_names, self.contract.joint_names)
        for i, (low, high) in expected.items():
            with self.subTest(index=i):
                self.assertAlmostEqual(self.bounds.lower[i], low, places=12)
                self.assertAlmostEqual(self.bounds.upper[i], high, places=12)
        # 若误用第一个动作项的 0.25 作为全局 scale，轮子会得到 (-40, 80)。
        self.assertNotAlmostEqual(self.bounds.upper[3], 80.0)

    def test_default_targets_round_trip(self) -> None:
        u0 = calibrate_initial_center(self.contract, self.bounds)
        targets = action_to_targets(self.contract, deterministic_action(self.bounds, u0))
        for got, want in zip(targets, self.contract.default_targets):
            self.assertAlmostEqual(got, want, places=12)
        # 非对称的腿部 1 和轮子需要非零 u0；对称的腿部 2、3 为 0。
        for i in (0, 3, 4, 7):
            self.assertNotAlmostEqual(u0[i], 0.0)
        for i in (1, 2, 5, 6):
            self.assertAlmostEqual(u0[i], 0.0, places=12)

    def test_mixed_initial_targets_round_trip(self) -> None:
        desired = [0.2, 0.9, -1.4, 5.0, 0.0, 0.7, -1.6, -3.0]  # 6 个 rad，索引 3、7 为 rad/s
        u0 = calibrate_initial_center(self.contract, self.bounds, desired)
        targets = action_to_targets(self.contract, deterministic_action(self.bounds, u0))
        for got, want in zip(targets, desired):
            self.assertAlmostEqual(got, want, places=12)


class FailureTest(unittest.TestCase):
    """规范列出的必要失败案例。"""

    def test_default_pose_outside_soft_limits(self) -> None:
        # q0=0.8 在 [0.9, 1.3] 之外；若取绝对值会得到看似正常的 [0.4, 2]，这里必须报错。
        with self.assertRaisesRegex(ActionBoundsError, "不包含零动作"):
            derive_bounds(single_position_contract(), {"j": (0.9, 1.3)}, {})

    def test_default_pose_on_limit(self) -> None:
        with self.assertRaisesRegex(ActionBoundsError, "不包含零动作"):
            derive_bounds(single_position_contract(), {"j": (0.8, 1.3)}, {})

    def test_missing_wheel_speed_range(self) -> None:
        contract = build_action_contract([ActionTermSpec("w", V, ("wheel",), 5.0)], {"wheel": 0.0}, {"wheel": 0.0})
        # 只给了轮子的角度限位：速度项不能借用它，必须报缺少工作轮速。
        with self.assertRaisesRegex(ActionBoundsError, "wheel_speed_ranges"):
            derive_bounds(contract, {"wheel": (-3.14, 3.14)}, {})

    def test_infinite_limit(self) -> None:
        with self.assertRaisesRegex(ActionBoundsError, "有限"):
            derive_bounds(single_position_contract(), {"j": (0.3, math.inf)}, {})

    def test_empty_range(self) -> None:
        with self.assertRaisesRegex(ActionBoundsError, "半宽"):
            derive_bounds(single_position_contract(), {"j": (1.3, 0.3)}, {})

    def test_non_positive_scale(self) -> None:
        with self.assertRaisesRegex(ActionBoundsError, "必须为正"):
            derive_bounds(single_position_contract(scale=-0.25), {"j": (0.3, 1.3)}, {})

    def test_unreachable_initial_target(self) -> None:
        contract = single_position_contract()
        bounds = derive_bounds(contract, {"j": (0.3, 1.3)}, {})
        for target in (1.3, 1.5, 0.3):  # 恰在边界上或越界，都没有有限的 u0
            with self.subTest(target=target), self.assertRaisesRegex(ActionBoundsError, "无法精确到达"):
                calibrate_initial_center(contract, bounds, [target])

    def test_wrong_initial_target_length(self) -> None:
        contract = single_position_contract()
        bounds = derive_bounds(contract, {"j": (0.3, 1.3)}, {})
        with self.assertRaisesRegex(ActionBoundsError, "长度"):
            calibrate_initial_center(contract, bounds, [0.8, 0.8])

    def test_bounds_from_other_contract(self) -> None:
        bounds = derive_bounds(single_position_contract(), {"j": (0.3, 1.3)}, {})
        other = build_action_contract([ActionTermSpec("p", P, ("k",), 0.25)], {"k": 0.8}, {"k": 0.0})
        with self.assertRaisesRegex(ActionBoundsError, "顺序"):
            validate_bounds(other, bounds)


if __name__ == "__main__":
    unittest.main()
