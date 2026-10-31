"""E03 八维动作定义表的验证程序（标准库 unittest，无需 Isaac Lab 或仿真）。

运行：在仓库根目录执行
    python3 -B environment_control/action_contract/test_contract.py -v

覆盖范围：
    - 原版对照：从固定快照 8fe40aa 读取原配置生成动作表，与下方手写参考表逐项比对。
      参考表是独立依据（人工对照原文件填写，与 AI 规范 README 的“已核对的原配置动作表”一致），
      不由被测代码生成，因此能发现解析或构建逻辑的错误。
    - README 验收案例：零动作、第 1 维 0.4、第 4 维 0.4。
    - 规则案例：按动作项顺序而非字母序、轮速偏移取默认速度而非角度、拒绝遗漏/重名/类型错配。
不覆盖：仿真中 ActionManager 实际加载出的顺序与默认值、执行器力矩——这些需要 Isaac Lab 环境（E01）。
"""

from __future__ import annotations

import math
import shutil
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_source import (  # noqa: E402
    BASELINE_SNAPSHOT_COMMIT,
    ENV_CFG_PATH,
    assert_actions_not_modified,
    load_baseline_action_contract,
    parse_action_terms,
    read_baseline_file,
)
from contract import (  # noqa: E402
    TITA_ACTION_LAYOUT,
    ActionContract,
    ActionContractError,
    ActionTermSpec,
    ControlType,
    action_to_targets,
    build_action_contract,
    describe_action_contract,
    validate_action_contract,
)

P, V = ControlType.POSITION, ControlType.VELOCITY

# 手写参考表：(动作项名, 关节名, 类型, 默认目标, 缩放, 单位)。
# 依据 baseline/ppo 快照 rough_env_cfg.py ActionsCfg 与 ddt_robot.py DDT_TITA_CFG.init_state。
REFERENCE_TABLE = (
    ("joint_pos_0", "joint_left_leg_1", P, 0.1, 0.25, "rad"),
    ("joint_pos_1", "joint_left_leg_2", P, 0.8, 0.25, "rad"),
    ("joint_pos_2", "joint_left_leg_3", P, -1.5, 0.25, "rad"),
    ("joint_vel_3", "joint_left_leg_4", V, 0.0, 5.0, "rad/s"),
    ("joint_pos_4", "joint_right_leg_1", P, 0.1, 0.25, "rad"),
    ("joint_pos_5", "joint_right_leg_2", P, 0.8, 0.25, "rad"),
    ("joint_pos_6", "joint_right_leg_3", P, -1.5, 0.25, "rad"),
    ("joint_vel_7", "joint_right_leg_4", V, 0.0, 5.0, "rad/s"),
)
REFERENCE_DEFAULTS = tuple(row[3] for row in REFERENCE_TABLE)


def make_tita_like_contract(
    wheel_default_pos: float = 0.0, wheel_default_vel: float = 0.0, order: tuple[int, ...] | None = None
) -> ActionContract:
    """用人工元数据构造与 TITA 同结构的动作表，便于在不读 Git 的情况下检查规则。

    wheel_default_pos / wheel_default_vel 可设为不同值，用来区分“取角度”与“取速度”两种错误实现；
    order 给出动作项的排列（参考表行号），用于检查顺序是否被保留。
    """
    rows = REFERENCE_TABLE if order is None else tuple(REFERENCE_TABLE[i] for i in order)
    terms = [
        ActionTermSpec(term_name=t, control_type=c, joint_names=(j,), scale=s, clip={".*": (-100.0, 100.0)}, preserve_order=True)
        for t, j, c, _, s, _ in rows
    ]
    joint_pos = {j: (wheel_default_pos if c is V else d) for _, j, c, d, _, _ in REFERENCE_TABLE}
    joint_vel = {".*_leg_4": wheel_default_vel, "joint_(left|right)_leg_[123]": 0.0}
    return build_action_contract(terms, joint_pos, joint_vel)


def git_snapshot_available() -> bool:
    """本机有 git 且固定快照提交在本地对象库中时返回 True。"""
    if shutil.which("git") is None:
        return False
    try:
        read_baseline_file(ENV_CFG_PATH)
    except RuntimeError:
        return False
    return True


@unittest.skipUnless(git_snapshot_available(), f"本地缺少 git 或快照提交 {BASELINE_SNAPSHOT_COMMIT}；请先 git fetch origin baseline/ppo")
class BaselineSnapshotTest(unittest.TestCase):
    """用固定原版配置验证动作表；能捕捉解析顺序、类型识别和默认值来源的错误。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_baseline_action_contract()

    def test_matches_reference_table(self) -> None:
        validate_action_contract(self.contract)
        self.assertEqual(len(self.contract), 8)
        for entry, (term, joint, ctype, default, scale, unit) in zip(self.contract.entries, REFERENCE_TABLE):
            with self.subTest(index=entry.index):
                self.assertEqual(entry.term_name, term)
                self.assertEqual(entry.joint_name, joint)
                self.assertIs(entry.control_type, ctype)
                self.assertEqual(entry.unit, unit)
                self.assertAlmostEqual(entry.default_target, default, places=12)
                self.assertAlmostEqual(entry.scale, scale, places=12)
                # 原版对每项都配置了 ±100 的物理目标裁剪，正常动作范围内不起作用。
                self.assertEqual((entry.clip_low, entry.clip_high), (-100.0, 100.0))

    def test_layout_matches_root_readme_convention(self) -> None:
        # 根 README 约定：索引 0,1,2,4,5,6 是角度，3,7 是轮速。
        self.assertEqual(self.contract.indices_of(P), (0, 1, 2, 4, 5, 6))
        self.assertEqual(self.contract.indices_of(V), (3, 7))

    def test_baseline_does_not_modify_actions_at_runtime(self) -> None:
        # load_baseline_action_contract 内部已检查；这里显式确认守卫本身能识别改写。
        with self.assertRaises(ActionContractError):
            assert_actions_not_modified("class C:\n    def f(self):\n        self.actions.joint_pos_0 = None\n", "fake.py")

    def test_parser_keeps_declaration_order(self) -> None:
        names = [t.term_name for t in parse_action_terms(read_baseline_file(ENV_CFG_PATH))]
        self.assertEqual(names, [row[0] for row in REFERENCE_TABLE])


class AcceptanceCaseTest(unittest.TestCase):
    """AI 规范 README 列出的三个验收案例。"""

    def setUp(self) -> None:
        self.contract = make_tita_like_contract()
        validate_action_contract(self.contract)

    def test_zero_action_gives_default_posture_and_zero_wheel_speed(self) -> None:
        targets = action_to_targets(self.contract, [0.0] * 8)
        for got, want in zip(targets, REFERENCE_DEFAULTS):
            self.assertAlmostEqual(got, want, places=12)
        # 零动作 ≠ 零目标：腿部目标是默认姿态（非零角度），执行器会朝该姿态出力，故也不是零力矩。
        self.assertTrue(any(abs(targets[i]) > 0.0 for i in self.contract.indices_of(P)))
        self.assertEqual([targets[i] for i in self.contract.indices_of(V)], [0.0, 0.0])

    def test_first_action_moves_left_leg_1_from_0_1_to_0_2_rad(self) -> None:
        action = [0.0] * 8
        action[0] = 0.4
        targets = action_to_targets(self.contract, action)
        self.assertAlmostEqual(targets[0], 0.2, places=12)  # 0.1 + 0.25 × 0.4
        self.assertEqual(self.contract.entries[0].unit, "rad")
        for i in range(1, 8):
            self.assertAlmostEqual(targets[i], REFERENCE_DEFAULTS[i], places=12)

    def test_fourth_action_gives_left_wheel_2_rad_per_s(self) -> None:
        action = [0.0] * 8
        action[3] = 0.4
        targets = action_to_targets(self.contract, action)
        self.assertAlmostEqual(targets[3], 2.0, places=12)  # 0 + 5 × 0.4
        self.assertEqual(self.contract.entries[3].joint_name, "joint_left_leg_4")
        self.assertEqual(self.contract.entries[3].unit, "rad/s")  # 关节角速度，不是 m/s
        for i in (0, 1, 2, 4, 5, 6, 7):
            self.assertAlmostEqual(targets[i], REFERENCE_DEFAULTS[i], places=12)


class RuleTest(unittest.TestCase):
    """README“必须成立的规则”与 validate 的拒绝条件。"""

    def test_order_follows_terms_not_alphabetical(self) -> None:
        # 原版顺序恰好与字母序相同，无法区分两种实现，所以用打乱的顺序检查。
        order = (4, 0, 7, 1, 5, 3, 2, 6)
        contract = make_tita_like_contract(order=order)
        self.assertEqual(contract.joint_names, tuple(REFERENCE_TABLE[i][1] for i in order))
        validate_action_contract(contract, expected_layout=[(REFERENCE_TABLE[i][1], REFERENCE_TABLE[i][2]) for i in order])
        with self.assertRaises(ActionContractError):
            validate_action_contract(contract)  # 与固定 TITA 布局不一致

    def test_wheel_offset_uses_default_velocity_not_angle(self) -> None:
        # 原版轮子默认角与默认速度都是 0，用不同的值才能区分错误实现。
        contract = make_tita_like_contract(wheel_default_pos=0.7, wheel_default_vel=0.0)
        self.assertEqual([contract.default_targets[i] for i in (3, 7)], [0.0, 0.0])
        contract = make_tita_like_contract(wheel_default_pos=0.7, wheel_default_vel=0.25)
        self.assertEqual([contract.default_targets[i] for i in (3, 7)], [0.25, 0.25])

    def test_clip_applies_to_physical_target(self) -> None:
        # 与 Isaac Lab process_actions 一致：先 a*scale+offset 再裁剪；0.8 + 0.25×1 = 1.05 → 1.0。
        term = ActionTermSpec("p", P, ("j",), 0.25, clip={"j": (-1.0, 1.0)})
        contract = build_action_contract([term], {"j": 0.8}, {"j": 0.0})
        self.assertEqual(action_to_targets(contract, [1.0]), (1.0,))

    def test_rejects_missing_joint(self) -> None:
        contract = make_tita_like_contract()
        truncated = ActionContract(contract.entries[:7])
        with self.assertRaisesRegex(ActionContractError, "缺少关节"):
            validate_action_contract(truncated)

    def test_rejects_duplicate_joint(self) -> None:
        entries = list(make_tita_like_contract().entries)
        entries[4] = replace(entries[4], joint_name="joint_left_leg_1")
        with self.assertRaisesRegex(ActionContractError, "重复"):
            validate_action_contract(ActionContract(tuple(entries)))

    def test_rejects_wheel_declared_as_position(self) -> None:
        entries = list(make_tita_like_contract().entries)
        entries[3] = replace(entries[3], control_type=P, unit="rad")
        with self.assertRaisesRegex(ActionContractError, "VELOCITY"):
            validate_action_contract(ActionContract(tuple(entries)))

    def test_rejects_unit_mismatch(self) -> None:
        entries = list(make_tita_like_contract().entries)
        entries[3] = replace(entries[3], unit="m/s")
        with self.assertRaisesRegex(ActionContractError, "单位"):
            validate_action_contract(ActionContract(tuple(entries)))

    def test_rejects_zero_scale(self) -> None:
        entries = list(make_tita_like_contract().entries)
        entries[0] = replace(entries[0], scale=0.0)
        with self.assertRaisesRegex(ActionContractError, "scale"):
            validate_action_contract(ActionContract(tuple(entries)))

    def test_build_rejects_ambiguous_inputs(self) -> None:
        # 多关节项未保留顺序：真实顺序取决于模型关节表，不能凭配置猜。
        with self.assertRaisesRegex(ActionContractError, "preserve_order"):
            build_action_contract([ActionTermSpec("t", P, ("a", "b"), 1.0)], {"a": 0.0, "b": 0.0}, {".*": 0.0})
        # 关节没有默认值。
        with self.assertRaisesRegex(ActionContractError, "没有匹配"):
            build_action_contract([ActionTermSpec("t", P, ("a",), 1.0)], {"b": 0.0}, {".*": 0.0})
        # 默认值正则歧义。
        with self.assertRaisesRegex(ActionContractError, "多个配置项"):
            build_action_contract([ActionTermSpec("t", V, ("a",), 1.0)], {"a": 0.0}, {".*": 0.0, "a": 1.0})
        # 不支持的动作类型。
        with self.assertRaises(ActionContractError):
            ControlType.from_cfg_class_name("JointEffortActionCfg")

    def test_action_to_targets_rejects_bad_input(self) -> None:
        contract = make_tita_like_contract()
        with self.assertRaisesRegex(ActionContractError, "长度"):
            action_to_targets(contract, [0.0] * 7)
        with self.assertRaisesRegex(ActionContractError, "有限"):
            action_to_targets(contract, [math.nan] + [0.0] * 7)

    def test_describe_lists_every_index_with_unit(self) -> None:
        text = describe_action_contract(make_tita_like_contract())
        rows = [line for line in text.splitlines()[2:] if line.startswith("|")]
        self.assertEqual(len(rows), 8)
        self.assertIn("| 4 / 3 | `joint_vel_3` | `joint_left_leg_4` | 轮速 | 0 | 5 | rad/s |", text)
        self.assertIn("| 1 / 0 | `joint_pos_0` | `joint_left_leg_1` | 角度 | 0.1 | 0.25 | rad |", text)


if __name__ == "__main__":
    unittest.main()
