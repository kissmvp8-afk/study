"""E04 动作边界与初始中心：由各维物理范围反推 SAC 策略动作的边界，并校准初始确定性动作。

职责：
    - derive_bounds：按 E03 动作表的顺序，把物理范围换回策略动作坐标，得到每维下界 L、上界 U、
      区间中心 b=(U+L)/2、半宽 c=(U-L)/2。腿部用关节软限位（rad），轮子用调用方给定的工作轮速（rad/s）。
    - calibrate_initial_center：给定希望的初始物理目标，求内部初始中心 u0，使 a=b+c*tanh(u0) 正好是
      对应的初始确定性动作 a0。
    - validate_bounds：检查边界有限、半宽为正，且默认目标（零动作）严格位于区间内。

不负责：
    - 不实现 Actor，不采样随机动作，不计算 log 概率修正（属于 N03 动作分布）。
    - 不读取仿真或 URDF：软限位与工作轮速都由调用方显式传入。本模块不内置任何 TITA 实际限位，
      也不把模型的 velocity_limit 30 rad/s 当作工作轮速（那只是上限参考，不是已验证的工作规格）。

上游输入：E03 的 ActionContract（顺序、控制类型、默认目标 default_target、缩放 scale）。
下游输出：ActionBounds 与 u0，供 N03 设置 tanh 映射的 b、c 以及均值输出层的初始偏置。

与作者 SAC 的关系（rsl_rl_sac e0d243a，BSD-3-Clause）：
    作者 `SAC._compute_action_scaling` 用 |q_soft - q0| / scale 求两侧范围，并取第一个动作项的 scale
    作为全局缩放；Actor 输出 a = action_range*tanh(x) + action_bias，均值层初始化接近 0。
    本模块的 TITA 适配有三处不同：
    1. 不取绝对值：默认姿态在软限位之外时必须报错，绝对值会把这种错误掩盖成“看起来正常”的区间。
    2. 每维使用 E03 中各自的 scale 和控制类型；轮速项（scale=5）用工作轮速反推，不套用腿部的 0.25。
    3. 作者初始 u≈0 时，确定性动作落在区间中点 b；非对称区间下 b≠0，开局姿态就偏离默认姿态。
       这里显式求 u0，使初始确定性动作回到希望的目标。这是本项目的适配，不是作者已验证的 TITA 结果。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# E03 目录不是 Python 包，按与 E03 相同的方式把它加入搜索路径，直接复用其 ActionContract。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "action_contract"))

from contract import ActionContract, ControlType  # noqa: E402


class ActionBoundsError(ValueError):
    """边界推导、校验或初始中心校准失败；消息中写明索引与关节名。"""


@dataclass(frozen=True)
class ActionBounds:
    """策略动作坐标下的每维边界（无物理单位），顺序与 E03 动作表一致。

    字段均为长度等于动作维数的 tuple：
        joint_names: 各维关节名，用于确认边界与动作表是同一顺序。
        lower / upper: 下界 L、上界 U。
        center: 区间中心 b = (U + L) / 2。
        half_width: 半宽 c = (U - L) / 2，必须为正。
    Actor 输出 a = b + c * tanh(u)，因此任何有限的 u 都会落在开区间 (L, U) 内。
    """

    joint_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    center: tuple[float, ...]
    half_width: tuple[float, ...]


def derive_bounds(
    contract: ActionContract,
    joint_pos_limits: Mapping[str, tuple[float, float]],
    wheel_speed_ranges: Mapping[str, tuple[float, float]],
) -> ActionBounds:
    """由物理范围反推每维策略动作边界，并调用 validate_bounds 检查结果。

    换算是 E03 映射 target = default_target + scale * a 的逆：
        L = (物理下限 - default_target) / scale，U = (物理上限 - default_target) / scale。
    位置项的物理范围取 joint_pos_limits（软限位，rad），速度项取 wheel_speed_ranges（工作轮速，rad/s）。
    两类范围按 E03 的控制类型分别查找：即使 joint_pos_limits 里也写了轮子关节的角度限位，
    速度项也不会读取它，避免把轮速当成角度。

    参数：
        contract: E03 动作表。
        joint_pos_limits: 关节名 → (软下限, 软上限)，单位 rad；需覆盖所有位置项。
        wheel_speed_ranges: 关节名 → (最低工作轮速, 最高工作轮速)，单位 rad/s；需覆盖所有速度项。
    返回：ActionBounds。
    异常：ActionBoundsError：缺少某维物理范围、scale 不为正，或 validate_bounds 不通过。
    副作用：无。
    """
    lower: list[float] = []
    upper: list[float] = []
    for e in contract.entries:
        ranges = joint_pos_limits if e.control_type is ControlType.POSITION else wheel_speed_ranges
        if e.joint_name not in ranges:
            source = "joint_pos_limits（软限位）" if e.control_type is ControlType.POSITION else "wheel_speed_ranges（工作轮速）"
            raise ActionBoundsError(f"索引 {e.index} ({e.joint_name}) 缺少物理范围，需在 {source} 中给出")
        if not e.scale > 0.0:
            # scale 为负时上下限会对调，scale 为零时无法反推；E03 只要求非零，这里收紧为正。
            raise ActionBoundsError(f"索引 {e.index} ({e.joint_name}) 的 scale={e.scale} 必须为正")
        low, high = ranges[e.joint_name]
        lower.append((float(low) - e.default_target) / e.scale)
        upper.append((float(high) - e.default_target) / e.scale)
    bounds = ActionBounds(
        joint_names=contract.joint_names,
        lower=tuple(lower),
        upper=tuple(upper),
        center=tuple((u + l) / 2.0 for l, u in zip(lower, upper)),
        half_width=tuple((u - l) / 2.0 for l, u in zip(lower, upper)),
    )
    validate_bounds(contract, bounds)
    return bounds


def validate_bounds(contract: ActionContract, bounds: ActionBounds) -> None:
    """检查边界能用于 a = b + c * tanh(u)，并且零动作（默认目标）严格在区间内部。

    检查项：
        - 关节顺序与动作表一致。
        - L、U 有限：无限限位会让 b、c 变成 inf 或 nan，tanh 映射失去意义。
        - 半宽 c > 0：物理下限不小于上限时区间为空或退化。
        - L < 0 < U：零动作对应默认目标。默认姿态落在软限位外，或恰好在边界上时，
          tanh 映射只能无限逼近边界，无法精确到达，因此视为配置错误，不做夹紧。
    异常：ActionBoundsError，列出全部问题。
    副作用：无。
    """
    if bounds.joint_names != contract.joint_names:
        raise ActionBoundsError(f"边界顺序 {bounds.joint_names} 与动作表 {contract.joint_names} 不一致")
    problems: list[str] = []
    for i, (name, low, high, c) in enumerate(zip(bounds.joint_names, bounds.lower, bounds.upper, bounds.half_width)):
        if not (math.isfinite(low) and math.isfinite(high)):
            problems.append(f"索引 {i} ({name}) 的边界 [{low}, {high}] 不是有限数，检查物理范围是否含无限值")
        elif not c > 0.0:
            problems.append(f"索引 {i} ({name}) 的半宽 {c} 不为正，物理下限必须小于上限")
        elif not low < 0.0 < high:
            problems.append(f"索引 {i} ({name}) 的边界 [{low:g}, {high:g}] 不包含零动作，默认目标不在物理范围内部")
    if problems:
        raise ActionBoundsError("动作边界校验失败：\n- " + "\n- ".join(problems))


def calibrate_initial_center(
    contract: ActionContract,
    bounds: ActionBounds,
    initial_targets: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """求内部初始中心 u0，使初始确定性动作经 tanh 映射与 E03 换算后，正好到达希望的物理目标。

    步骤：
        a0 = (希望目标 - default_target) / scale     （物理目标 → 策略动作坐标）
        tanh(u0) = (a0 - b) / c  ⇒  u0 = atanh((a0 - b) / c)
    对称区间且 a0=0 时 u0=0，与作者的近零初始化一致；非对称区间下 b≠0，必须用 u0 抵消。

    只校准“确定性动作”（均值 u0 直接经 tanh 映射）。随机采样时 a=b+c*tanh(u0+σε)，tanh 是非线性的，
    非对称区间下采样动作的平均值一般不等于 a0，这里不声称两者相同。
    a0 越接近 L 或 U，|u0| 越大，tanh 越接近饱和、梯度越小；本函数只拒绝不可达的情况，不另设余量。

    参数：
        contract: E03 动作表。
        bounds: 同一动作表的 ActionBounds。
        initial_targets: 每维希望的初始物理目标（位置项 rad，速度项 rad/s）；None 表示使用默认目标，即 a0=0。
    返回：u0，长度等于动作维数的 tuple。
    异常：ActionBoundsError：长度不符，或某维 a0 不严格位于 (L, U) 内（在边界上或越界都无法得到有限 u0）。
    副作用：无。
    """
    targets = contract.default_targets if initial_targets is None else tuple(float(t) for t in initial_targets)
    if len(targets) != len(contract):
        raise ActionBoundsError(f"初始目标长度为 {len(targets)}，动作表维数为 {len(contract)}")
    u0: list[float] = []
    for e, target, low, high, b, c in zip(
        contract.entries, targets, bounds.lower, bounds.upper, bounds.center, bounds.half_width
    ):
        a0 = (target - e.default_target) / e.scale
        if not low < a0 < high:
            raise ActionBoundsError(
                f"索引 {e.index} ({e.joint_name}) 的初始目标 {target:g} {e.unit} 对应动作 {a0:g}，"
                f"不在开区间 ({low:g}, {high:g}) 内，tanh 映射无法精确到达"
            )
        u0.append(math.atanh((a0 - b) / c))
    return tuple(u0)
