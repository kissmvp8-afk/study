"""E01 机器人资产：锁定 ddt_ros2_control 中的 TITA URDF，并读取关节限位与轮子几何（纯 Python，不依赖 Isaac）。

锁定版本与依据：
    原版 DDT_Lab（701a49c，提交于 2026-06-26）的 README 只要求克隆 ddt_ros2_control 默认分支，没有固定版本。
    这里锁定该时间点之前 ddt_ros2_control main 的最新提交 e7b32d9（2026-05-29），使资产与原版 PPO 当时使用的一致。
    TITA URDF 在该仓库中只被 4f5d962（初始）、e62e1aa（2026-01-08）、beb05df（2026-10-30）修改过；
    e7b32d9 中的文件与 e62e1aa 相同（SHA-256 0941b831…），而 beb05df 已改动 URDF，因此不使用更新的版本。
    ddt_ros2_control 仓库未声明许可证：本仓库只记录路径、提交与校验值，不复制 URDF 或网格文件。

资源路径（与原版 ddt_robot.py 的 DDT_MODEL_DIR 一致）：<DDT_Lab 根目录>/ddt_ros2_control/urdfs/tita_description/urdf/robot.urdf；
网格位于同目录的 ../meshes。Isaac 启动时由原版 DDT_TITA_CFG 的 UrdfFileCfg 转换为 USD。

软限位（与 Isaac Lab 2.3.0 Articulation 相同）：中点 ± 0.5 × (上限 − 下限) × soft_joint_pos_limit_factor，
原版 DDT_TITA_CFG 中该系数为 0.9。轮子关节在 URDF 中写成 revolute、限位 ±62831.85 rad（相当于连续转动），
其位置限位对速度控制的轮子没有意义，E04 只使用腿部 6 个关节的软限位。
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROBOT_ASSET = {
    "repository": "DDTRobot/ddt_ros2_control",
    "url": "https://github.com/DDTRobot/ddt_ros2_control",
    "commit": "e7b32d9746df900680641ff2845ffb0b5ffe99ed",
    "urdf_path": "urdfs/tita_description/urdf/robot.urdf",
    "urdf_sha256": "0941b83136cd5c8c89bbb33165bff515a3a988d33319b67c8b06bfb9c71350ce",
    "license": "仓库未声明许可证；本仓库不复制其文件",
    "selection": "DDT_Lab 701a49c（2026-06-26）之前 ddt_ros2_control main 的最新提交；TITA URDF 与 e62e1aa 相同，晚于此的 beb05df 已修改 URDF",
}
WHEEL_JOINTS = ("joint_left_leg_4", "joint_right_leg_4")


class AssetError(RuntimeError):
    """资产缺失、校验值不符或 URDF 内容不符合预期。"""


@dataclass(frozen=True)
class UrdfJoint:
    """URDF 中的一个关节（位置单位 rad，origin 单位 m / rad）。"""

    name: str
    type: str
    parent: str
    child: str
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def urdf_path(ddt_lab_root: Path) -> Path:
    """原版 DDT_MODEL_DIR 约定下的 TITA URDF 路径。"""
    return Path(ddt_lab_root) / "ddt_ros2_control" / Path(ROBOT_ASSET["urdf_path"])


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_urdf(path: Path) -> str:
    """确认 URDF 存在且与锁定版本一致，网格目录存在；返回 "exact" 或 "crlf_normalized"。

    Windows 上 git 的 core.autocrlf=true 会在检出时把 LF 换成 CRLF，字节校验值随之改变而 XML 内容不变；
    这种情况把 CRLF 换回 LF 后再比较，一致则返回 "crlf_normalized"（调用方应记录下来），不一致仍然报错。
    异常：AssetError（文件或网格目录缺失时给出具体路径；校验值不符时给出实际值）。
    """
    path = Path(path)
    if not path.is_file():
        raise AssetError(f"找不到 TITA URDF：{path}（请在 DDT_Lab 根目录克隆 ddt_ros2_control 并检出 {ROBOT_ASSET['commit']}）")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == ROBOT_ASSET["urdf_sha256"]:
        status = "exact"
    elif hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == ROBOT_ASSET["urdf_sha256"]:
        status = "crlf_normalized"
    else:
        raise AssetError(f"URDF 校验值 {hashlib.sha256(data).hexdigest()} 与锁定版本 {ROBOT_ASSET['urdf_sha256']} 不符：{path}"
                         f"（确认 ddt_ros2_control 检出的是 {ROBOT_ASSET['commit']}）")
    meshes = path.parent.parent / "meshes"
    if not meshes.is_dir():
        raise AssetError(f"找不到网格目录：{meshes}")
    return status


def _triple(text: str | None, default: float = 0.0) -> tuple[float, float, float]:
    values = [float(v) for v in text.split()] if text else [default] * 3
    return values[0], values[1], values[2]


def parse_urdf_joints(path: Path) -> dict[str, UrdfJoint]:
    """解析 URDF 的全部关节（按文件顺序）。URDF 只作为数据读取。"""
    root = ET.parse(Path(path)).getroot()
    joints = {}
    for element in root.findall("joint"):
        limit, origin, axis = element.find("limit"), element.find("origin"), element.find("axis")
        number = lambda key: float(limit.get(key)) if limit is not None and limit.get(key) is not None else None  # noqa: E731
        joints[element.get("name")] = UrdfJoint(
            name=element.get("name"), type=element.get("type"),
            parent=element.find("parent").get("link"), child=element.find("child").get("link"),
            lower=number("lower"), upper=number("upper"), effort=number("effort"), velocity=number("velocity"),
            origin_xyz=_triple(origin.get("xyz") if origin is not None else None),
            origin_rpy=_triple(origin.get("rpy") if origin is not None else None),
            axis=_triple(axis.get("xyz") if axis is not None else None, default=0.0) if axis is not None else (1.0, 0.0, 0.0),
        )
    return joints


def wheel_radius(path: Path, wheel_joint: str) -> float:
    """轮子半径（m）：轮子关节子连杆碰撞体圆柱的 radius。"""
    root = ET.parse(Path(path)).getroot()
    child = parse_urdf_joints(path)[wheel_joint].child
    link = next(l for l in root.findall("link") if l.get("name") == child)
    cylinder = link.find("collision/geometry/cylinder")
    if cylinder is None:
        raise AssetError(f"{child} 的碰撞体不是圆柱，无法读取轮子半径")
    return float(cylinder.get("radius"))


def soft_joint_limits(joints: dict[str, UrdfJoint], names, factor: float) -> dict[str, tuple[float, float]]:
    """按 Isaac Lab 2.3.0 Articulation 的公式计算软限位（rad）。"""
    limits = {}
    for name in names:
        joint = joints[name]
        if joint.lower is None or joint.upper is None:
            raise AssetError(f"{name} 没有位置限位")
        mean, span = (joint.lower + joint.upper) / 2, joint.upper - joint.lower
        limits[name] = (mean - 0.5 * span * factor, mean + 0.5 * span * factor)
    return limits


def _transform(rpy, xyz):
    """URDF origin 的 4×4 齐次变换：R = Rz(yaw) · Ry(pitch) · Rx(roll)。"""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rotation = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr]]
    return [rotation[0] + [xyz[0]], rotation[1] + [xyz[1]], rotation[2] + [xyz[2]], [0.0, 0.0, 0.0, 1.0]]


def _axis_rotation(axis, angle):
    """绕单位轴 axis 转 angle 的齐次变换（Rodrigues 公式）。"""
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    c, s, v = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return [[c + x * x * v, x * y * v - z * s, x * z * v + y * s, 0.0],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s, 0.0],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def link_position(joints: dict[str, UrdfJoint], link: str, joint_positions: dict[str, float]) -> tuple[float, float, float]:
    """正运动学：给定关节角（rad，未给出的视为 0），返回连杆原点在根连杆坐标系中的位置（m）。"""
    by_child = {j.child: j for j in joints.values()}
    chain = []
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link].parent
    transform = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for joint in reversed(chain):
        transform = _matmul(transform, _transform(joint.origin_rpy, joint.origin_xyz))
        if joint.type in ("revolute", "continuous"):
            transform = _matmul(transform, _axis_rotation(joint.axis, joint_positions.get(joint.name, 0.0)))
    return transform[0][3], transform[1][3], transform[2][3]


def wheel_half_track(joints: dict[str, UrdfJoint], joint_positions: dict[str, float]) -> float:
    """两个轮心在根坐标系中的横向（y）距离的一半（m），在给定关节角（通常为默认姿态）下计算。"""
    left = link_position(joints, joints[WHEEL_JOINTS[0]].child, joint_positions)
    right = link_position(joints, joints[WHEEL_JOINTS[1]].child, joint_positions)
    return abs(left[1] - right[1]) / 2
