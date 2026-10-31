"""E03 八维动作定义表：把策略输出的每一维对应到一个关节控制目标。

职责：
    - 依据环境动作项（action term）的声明顺序和模型默认状态，建立“索引 → 关节、控制类型、
      单位、默认目标、缩放、裁剪”的顺序表（ActionContract）。
    - 校验该表没有遗漏、重名、类型或单位错配，并能输出人可检查的表格。
    - 按原环境的同一公式把一条策略动作换算成物理控制目标，供验收与下游核对。

不负责：
    - 不生成随机动作、不计算 Q、不决定 SAC 的动作边界（边界属于 E04）。
    - 不计算力矩：本表给出的是 PD 控制器的“目标”，力矩由执行器根据目标与当前状态决定。

上游输入：
    动作项配置（ActionTermSpec 序列，顺序即环境拼接动作向量的顺序）与模型默认关节
    位置/速度（来自 ArticulationCfg.init_state）。固定原版的解析见同目录 baseline_source.py。

下游输出：
    ActionContract：E04 读取 scale / default_target / control_type 推导边界；经验记录、
    Q 网络与导出模块依赖同一索引顺序和“策略动作坐标”（无物理单位）。

与原 PPO / 作者 SAC 的关系：
    原 PPO（DDT_Lab 701a49c，本仓库快照 baseline/ppo 8fe40aa）由 Isaac Lab 的
    JointPositionAction / JointVelocityAction 执行 target = clip(a * scale + offset)。
    本模块复刻这一换算规则用于说明与核对，不改变其物理含义；SAC 只换动作采样方式，
    同一索引仍控制同一关节。作者 SAC 统一按关节位置缩放，未覆盖 TITA 的轮速动作，
    因此这里显式区分位置与速度两类控制。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence


class ActionContractError(ValueError):
    """动作表构建或校验失败。

    message 中逐条列出全部问题，方便一次看清配置哪里不对，而不是修一个报一个。
    """


class ControlType(Enum):
    """动作控制类型，值为 (Isaac Lab 动作配置类名, 物理目标单位)。

    POSITION：关节位置目标，单位 rad（弧度，角度）。
    VELOCITY：关节速度目标，单位 rad/s（每秒转过的弧度，转速）。轮子是转动关节，
    所以轮速目标是角速度 rad/s，不是车速 m/s；两者差一个轮半径因子。
    """

    POSITION = ("JointPositionActionCfg", "rad")
    VELOCITY = ("JointVelocityActionCfg", "rad/s")

    @property
    def cfg_class_name(self) -> str:
        """对应的 Isaac Lab 动作配置类名，用于从原配置识别控制类型。"""
        return self.value[0]

    @property
    def unit(self) -> str:
        """该类型物理目标的单位字符串。"""
        return self.value[1]

    @classmethod
    def from_cfg_class_name(cls, name: str) -> "ControlType":
        """由配置类名得到控制类型；不支持的动作类（如力矩、相对位置）抛 ActionContractError。"""
        for member in cls:
            if member.cfg_class_name == name:
                return member
        raise ActionContractError(f"不支持的动作配置类 {name!r}；E03 只定义关节位置与关节速度两类动作")


# 固定原版 TITA 的期望动作布局：顺序与类型。来源为 baseline/ppo 快照
# rough_env_cfg.py 的 ActionsCfg（joint_pos_0 … joint_vel_7 的声明顺序）。
# 索引 0,1,2,4,5,6 是腿部角度，3,7 是轮速（*_leg_4 为轮子关节）。
# 这里只固定“顺序与类型”，不固定 scale/默认值：它们由配置读取，并在测试中与原版逐项比对。
TITA_ACTION_LAYOUT: tuple[tuple[str, ControlType], ...] = (
    ("joint_left_leg_1", ControlType.POSITION),
    ("joint_left_leg_2", ControlType.POSITION),
    ("joint_left_leg_3", ControlType.POSITION),
    ("joint_left_leg_4", ControlType.VELOCITY),
    ("joint_right_leg_1", ControlType.POSITION),
    ("joint_right_leg_2", ControlType.POSITION),
    ("joint_right_leg_3", ControlType.POSITION),
    ("joint_right_leg_4", ControlType.VELOCITY),
)


@dataclass(frozen=True)
class ActionTermSpec:
    """一条环境动作项配置（对应 Isaac Lab ActionsCfg 中的一个字段）。

    参数：
        term_name: 动作项字段名，如 "joint_pos_0"；只用于报错与追溯。
        control_type: 位置或速度控制。
        joint_names: 该项控制的关节名（精确名称，不支持正则）；多关节时按此顺序展开。
        scale: 策略动作到物理目标的缩放，必须是标量（Isaac Lab 也允许按关节字典，TITA 未使用）。
        use_default_offset: True 时 offset 取模型默认值——位置项取默认关节角，速度项取默认关节速度。
        offset: use_default_offset=False 时使用的常数偏移（Isaac Lab 默认 0.0）。
        clip: 关节名正则 → (下限, 上限)，作用于换算后的物理目标；None 表示不裁剪。
        preserve_order: 对应 Isaac Lab 的 preserve_order。False 时 Isaac Lab 按模型关节顺序
            重排多关节项，仅凭配置无法确定顺序，因此多关节项必须为 True。
    """

    term_name: str
    control_type: ControlType
    joint_names: tuple[str, ...]
    scale: float
    use_default_offset: bool = True
    offset: float = 0.0
    clip: Mapping[str, tuple[float, float]] | None = None
    preserve_order: bool = False


@dataclass(frozen=True)
class ActionEntry:
    """动作表中的一行：策略动作第 index 维的完整物理含义。

    换算：target = clip(a[index] * scale + default_target, clip_low, clip_high)。
    a 是策略动作坐标，无物理单位；target 的单位是 unit。
    default_target 是“动作为零时的控制目标”：位置项为默认关节角 q0（rad），
    速度项为默认关节速度（TITA 为 0 rad/s）。
    """

    index: int
    term_name: str
    joint_name: str
    control_type: ControlType
    unit: str
    default_target: float
    scale: float
    clip_low: float = -math.inf
    clip_high: float = math.inf


@dataclass(frozen=True)
class ActionContract:
    """按策略输出索引排列的完整动作表（不可变）。

    entries[i] 描述动作向量第 i 维。提供按列取值的便捷属性，下游（如 E04）可直接
    转成张量使用，避免各模块各自再拼一次顺序。
    """

    entries: tuple[ActionEntry, ...]

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """各索引对应的关节名，顺序即动作向量顺序。"""
        return tuple(e.joint_name for e in self.entries)

    @property
    def scales(self) -> tuple[float, ...]:
        """各索引的缩放系数。"""
        return tuple(e.scale for e in self.entries)

    @property
    def default_targets(self) -> tuple[float, ...]:
        """各索引在零动作时的控制目标（位置项 rad，速度项 rad/s）。"""
        return tuple(e.default_target for e in self.entries)

    def indices_of(self, control_type: ControlType) -> tuple[int, ...]:
        """返回某一控制类型的全部索引，例如 TITA 速度项为 (3, 7)。"""
        return tuple(e.index for e in self.entries if e.control_type is control_type)


def resolve_joint_values(patterns: Mapping[str, float], joint_names: Iterable[str]) -> dict[str, float]:
    """把“关节名正则 → 数值”的配置展开成每个关节的具体数值。

    与 Isaac Lab 的 resolve_matching_names_values 一致：正则做整串匹配（fullmatch），
    一个关节匹配多个正则视为配置歧义。原版 init_state.joint_vel 写成 {".*": 0.0}，
    必须经过这一步才能得到每个轮子的默认速度。

    参数：
        patterns: 正则到数值的映射，如 {"joint_left_leg_1": 0.1} 或 {".*": 0.0}。
        joint_names: 需要取值的关节名。
    返回：
        关节名 → float。
    异常：
        ActionContractError：某关节无匹配或匹配多个正则（全部问题合并报告）。
    """
    resolved: dict[str, float] = {}
    problems: list[str] = []
    for joint in joint_names:
        matched = [p for p in patterns if re.fullmatch(p, joint)]
        if not matched:
            problems.append(f"关节 {joint!r} 没有匹配的默认值配置")
        elif len(matched) > 1:
            problems.append(f"关节 {joint!r} 同时匹配多个配置项 {matched}，无法确定默认值")
        else:
            resolved[joint] = float(patterns[matched[0]])
    if problems:
        raise ActionContractError("；".join(problems))
    return resolved


def _resolve_clip(clip: Mapping[str, tuple[float, float]] | None, joint: str) -> tuple[float, float]:
    """取某关节的物理目标裁剪区间；未配置或未匹配时为 (-inf, inf)，与 Isaac Lab 相同。"""
    if clip is None:
        return (-math.inf, math.inf)
    matched = [p for p in clip if re.fullmatch(p, joint)]
    if len(matched) > 1:
        raise ActionContractError(f"关节 {joint!r} 同时匹配多个 clip 配置项 {matched}")
    if not matched:
        return (-math.inf, math.inf)
    low, high = clip[matched[0]]
    return (float(low), float(high))


def build_action_contract(
    terms: Sequence[ActionTermSpec],
    default_joint_pos: Mapping[str, float],
    default_joint_vel: Mapping[str, float],
) -> ActionContract:
    """由动作项配置与模型默认状态建立动作表。

    参数：
        terms: 动作项，顺序必须是环境声明顺序（Isaac Lab ActionManager 按声明顺序拼接动作向量）。
        default_joint_pos: 关节名正则 → 默认关节角 (rad)，即 init_state.joint_pos。
        default_joint_vel: 关节名正则 → 默认关节速度 (rad/s)，即 init_state.joint_vel。
    返回：
        ActionContract，索引从 0 连续编号。不做 TITA 布局校验，需要时再调用 validate_action_contract。
    异常：
        ActionContractError：多关节项未 preserve_order、关节缺少默认值、clip 配置歧义等。
    副作用：无。
    """
    entries: list[ActionEntry] = []
    for term in terms:
        if len(term.joint_names) > 1 and not term.preserve_order:
            # preserve_order=False 时 Isaac Lab 按模型（URDF）关节顺序排列，配置里的书写顺序不算数；
            # 没有模型关节表就无法得知真实顺序，宁可报错也不按书写顺序或字母序猜。
            raise ActionContractError(
                f"动作项 {term.term_name!r} 含多个关节但 preserve_order=False，仅凭配置无法确定关节顺序"
            )
        for joint in term.joint_names:
            if not term.use_default_offset:
                offset = float(term.offset)
            elif term.control_type is ControlType.POSITION:
                offset = resolve_joint_values(default_joint_pos, [joint])[joint]
            else:
                # 速度项的默认偏移是“默认关节速度”，不是默认关节角。轮子关节的默认角恰好也是 0，
                # 若误用角度，在原版数值上看不出错，换一个轮子初始角就会让零动作变成持续转动。
                offset = resolve_joint_values(default_joint_vel, [joint])[joint]
            clip_low, clip_high = _resolve_clip(term.clip, joint)
            entries.append(
                ActionEntry(
                    index=len(entries),
                    term_name=term.term_name,
                    joint_name=joint,
                    control_type=term.control_type,
                    unit=term.control_type.unit,
                    default_target=offset,
                    scale=float(term.scale),
                    clip_low=clip_low,
                    clip_high=clip_high,
                )
            )
    return ActionContract(entries=tuple(entries))


def validate_action_contract(
    contract: ActionContract,
    expected_layout: Sequence[tuple[str, ControlType]] | None = TITA_ACTION_LAYOUT,
) -> None:
    """检查动作表的一致性，发现问题时一次性报告全部。

    检查项：
        - 索引从 0 连续；关节名不重复。
        - 单位与控制类型一致（位置 rad、速度 rad/s）。
        - scale 有限且非零，默认目标有限，clip 下限不大于上限。
        - 给定 expected_layout 时：维数、逐索引关节名与控制类型都必须一致。
          这里按索引逐个比较，而不是比较集合，所以“名字都在但顺序被重排/按字母排序”也会被拒绝。
    参数：
        contract: 待检查的动作表。
        expected_layout: 期望的 (关节名, 控制类型) 序列；默认是固定原版 TITA 布局，None 表示只做内部一致性检查。
    异常：
        ActionContractError：存在任何问题。
    副作用：无。
    """
    problems: list[str] = []
    entries = contract.entries
    if not entries:
        problems.append("动作表为空")

    seen: dict[str, int] = {}
    for position, e in enumerate(entries):
        if e.index != position:
            problems.append(f"第 {position} 行的 index 为 {e.index}，索引必须从 0 连续编号")
        if e.joint_name in seen:
            problems.append(f"关节 {e.joint_name!r} 重复出现在索引 {seen[e.joint_name]} 和 {position}")
        else:
            seen[e.joint_name] = position
        if e.unit != e.control_type.unit:
            problems.append(
                f"索引 {position} ({e.joint_name}) 单位为 {e.unit!r}，与 {e.control_type.name} 类型的 {e.control_type.unit!r} 不符"
            )
        if not math.isfinite(e.scale) or e.scale == 0.0:
            problems.append(f"索引 {position} ({e.joint_name}) 的 scale={e.scale} 必须为有限非零数")
        if not math.isfinite(e.default_target):
            problems.append(f"索引 {position} ({e.joint_name}) 的默认目标 {e.default_target} 不是有限数")
        if e.clip_low > e.clip_high:
            problems.append(f"索引 {position} ({e.joint_name}) 的 clip 下限 {e.clip_low} 大于上限 {e.clip_high}")

    if expected_layout is not None:
        if len(entries) != len(expected_layout):
            problems.append(f"动作维数为 {len(entries)}，期望 {len(expected_layout)}")
        expected_names = {name for name, _ in expected_layout}
        missing = [name for name in (n for n, _ in expected_layout) if name not in seen]
        if missing:
            problems.append(f"缺少关节 {missing}")
        unexpected = [e.joint_name for e in entries if e.joint_name not in expected_names]
        if unexpected:
            problems.append(f"出现期望布局之外的关节 {unexpected}")
        for position, (e, (name, ctype)) in enumerate(zip(entries, expected_layout)):
            if e.joint_name != name:
                problems.append(f"索引 {position} 应为 {name!r}，实际为 {e.joint_name!r}")
            if e.control_type is not ctype:
                problems.append(
                    f"索引 {position} ({e.joint_name}) 应为 {ctype.name} 控制，实际为 {e.control_type.name}"
                )

    if problems:
        raise ActionContractError("动作表校验失败：\n- " + "\n- ".join(problems))


def action_to_targets(contract: ActionContract, action: Sequence[float]) -> tuple[float, ...]:
    """把一条策略动作换算成各关节的物理控制目标（单条、纯 Python，供核对用）。

    公式与原环境 Isaac Lab JointAction.process_actions 相同：先 a*scale+offset，再按 clip 裁剪。
    clip 作用在物理目标上（单位 rad 或 rad/s），原版取 ±100，在正常范围内不起作用；
    它不是 SAC 的策略动作边界，后者由 E04 决定。

    注意：零动作得到的是默认目标（腿部默认姿态、轮速 0），不是零力矩。腿部 PD 执行器的力矩
    约为 kp*(q*-q) - kd*q̇，关节偏离默认角时仍会出力；轮子 kp=0、kd>0，转动时零动作相当于制动。

    参数：
        contract: 动作表。
        action: 长度等于动作维数的策略动作，无物理单位。
    返回：
        每维的物理目标（tuple[float, ...]），单位见 contract.entries[i].unit。
    异常：
        ActionContractError：长度不符或含 NaN/inf。
    副作用：无。批量张量计算由下游在环境或网络侧完成，本函数不处理批维。
    """
    if len(action) != len(contract.entries):
        raise ActionContractError(f"动作长度为 {len(action)}，动作表维数为 {len(contract.entries)}")
    targets: list[float] = []
    for e, a in zip(contract.entries, action):
        a = float(a)
        if not math.isfinite(a):
            raise ActionContractError(f"索引 {e.index} ({e.joint_name}) 的动作值 {a} 不是有限数")
        target = a * e.scale + e.default_target
        targets.append(min(max(target, e.clip_low), e.clip_high))
    return tuple(targets)


def describe_action_contract(contract: ActionContract) -> str:
    """生成 Markdown 表格，逐行列出索引、关节、类型、默认目标、缩放、单位与裁剪，供人工核对。"""
    type_label = {ControlType.POSITION: "角度", ControlType.VELOCITY: "轮速"}
    lines = [
        "| 序号 / 索引 | 动作项 | 关节名称 | 类型 | 默认目标 | 缩放 | 物理目标单位 | 物理目标裁剪 |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for e in contract.entries:
        clip = "无" if math.isinf(e.clip_low) and math.isinf(e.clip_high) else f"[{e.clip_low:g}, {e.clip_high:g}]"
        lines.append(
            f"| {e.index + 1} / {e.index} | `{e.term_name}` | `{e.joint_name}` | {type_label[e.control_type]} "
            f"| {e.default_target:g} | {e.scale:g} | {e.unit} | {clip} |"
        )
    return "\n".join(lines)
