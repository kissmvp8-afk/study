"""E02 观测接口：定义 Actor 与 Q 各自看到的观测字段，维护 [B,10,33] 策略历史，并校验运行时观测。

字段来源：固定原版快照 rough_env_cfg.py 的 ObservationsCfg（PolicyCfg / CriticCfg / PrivCfg / ScannerCfg），
顺序、函数、缩放与噪声逐项与原配置一致（parse_observation_config 从快照静态解析并在测试中比对）。

策略观测（Actor，每帧 33 维，带观测噪声）：
    base_ang_vel 3（×0.25，身体系角速度 rad/s）| projected_gravity 3（×1，身体系重力方向）|
    velocity_commands 3（×(2, 2, 0.25)，指令 vx、vy m/s，wz rad/s）| joint_pos 8（×1，相对默认角 rad，轮子两项置 0）|
    joint_vel 8（×0.05，rad/s）| actions 8（×1，上一步策略动作，策略坐标）
    关节两项按 articulation 的关节顺序（joint_names=".*"、preserve_order=True），由运行时读取，不一定等于 E03 动作顺序。
    **不含身体线速度**：真实线速度只进入 Q 观测与速度监督标签，Actor 必须从历史中估计它（N01 速度头）。

Q 观测（训练侧，[B,C]，无噪声）= [critic 36 | priv 18 | scanner 187（仅复杂地形）]，拼接顺序同原 PPO wrapper：
    critic：base_lin_vel 3（×2.0）| base_ang_vel 3 | projected_gravity 3 | velocity_commands 3 | joint_pos 8 | joint_vel 8 | actions 8
    priv：contact_state 2（两个轮子是否接触，±0.5）| joint_kp_factor 8 | joint_kd_factor 8
    scanner：height_scan 187（17×11 网格）
平地 C=54、复杂地形 C=241 只是由原配置推出的预期值；validate_observations 用运行时 group_obs_term_dim 逐项核对，
不一致时报错，绝不截断或补齐。

速度监督标签：Q 观测前 3 列 = base_lin_vel × 2.0（与原 PPO critic_obs[:, :3] 相同）。标签与 N01 的速度估计都在这个
缩放下比较；除以 2.0 才是 m/s。

两层缩放不要混淆：上面的 ×0.25、×2.0 等是**环境字段缩放**（Isaac 观测项的 scale，已包含在观测值中）；
D04 归一化是之后在训练与推理中再做的统计标准化。本模块只读取环境输出，不再缩放；D04 也不撤销环境缩放。

策略历史：SAC 环境把 policy 组的 Isaac 历史长度设为 0（每次只得到当前帧），由本模块的 PolicyHistory 维护 10 帧历史。
原因：锁定的 Isaac Lab 2.3.0 在 step 中先重置结束的环境再计算观测，要在重置前取得含当前帧的历史，
只能在重置前调用一次 compute(update_history=True)，但这会让未结束环境的历史在重置后再次推进（重复当前帧）；
compute(update_history=False) 返回的是不含当前帧的旧历史。把历史放在这里，每个环境每步只推进一次。
推进规则逐条复刻 Isaac Lab 2.3.0 CircularBuffer：从旧到新排列（最新帧在最后）；重置后清零且推送计数归零；
重置后第一次推送时用该帧填满全部 10 个槽位；之后每次推送丢弃最旧帧。有效历史长度 = min(推送次数, 10)。
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "environment_control" / "action_contract"))

from baseline_source import ENV_CFG_PATH, read_baseline_file  # noqa: E402

HISTORY_LENGTH, POLICY_FRAME_DIM = 10, 33
VELOCITY_LABEL_COLUMNS = (0, 3)
VELOCITY_LABEL_SCALE = 2.0  # 原 CriticCfg.base_lin_vel 的 scale


class ObservationError(ValueError):
    """观测字段、形状或数值不符合约定。"""


@dataclass(frozen=True)
class ObservationField:
    """一个观测项：名称、Isaac 观测函数、预期维数、环境字段缩放、是否带噪声、含义与单位。"""

    name: str
    func: str
    dim: int
    scale: float | tuple[float, ...]
    noisy: bool
    meaning: str


POLICY_FIELDS = (
    ObservationField("base_ang_vel", "base_ang_vel", 3, 0.25, True, "身体系角速度 rad/s"),
    ObservationField("projected_gravity", "projected_gravity", 3, 1.0, True, "身体系下的单位重力方向"),
    ObservationField("velocity_commands", "generated_commands", 3, (2.0, 2.0, 0.25), False, "速度指令 vx、vy m/s，wz rad/s"),
    ObservationField("joint_pos", "joint_pos_rel_without_wheel", 8, 1.0, True, "相对默认角 rad，轮子两项置 0；articulation 关节顺序"),
    ObservationField("joint_vel", "joint_vel_rel", 8, 0.05, True, "关节速度 rad/s；articulation 关节顺序"),
    ObservationField("actions", "last_action", 8, 1.0, False, "上一步策略动作（策略坐标，E03 顺序）"),
)
CRITIC_FIELDS = (
    ObservationField("base_lin_vel", "base_lin_vel", 3, 2.0, False, "身体系线速度 m/s（真实值，只给 Q 与速度标签）"),
    *(ObservationField(f.name, f.func, f.dim, f.scale, False, f.meaning) for f in POLICY_FIELDS),
)
PRIV_FIELDS = (
    ObservationField("contact_state", "contact_state", 2, 1.0, False, "两个轮子接触状态（接触 +0.5，否则 −0.5）"),
    ObservationField("joint_kp_factor", "joint_kp_factor", 8, 1.0, False, "随机化后的刚度 / 默认刚度，截断到 [0, 2]"),
    ObservationField("joint_kd_factor", "joint_kd_factor", 8, 1.0, False, "随机化后的阻尼 / 默认阻尼，截断到 [0, 2]"),
)
SCANNER_FIELDS = (ObservationField("height_scan", "height_scan", 187, 1.0, False, "17×11 高度扫描（复杂地形）"),)
GROUP_FIELDS = {"policy": POLICY_FIELDS, "critic": CRITIC_FIELDS, "priv": PRIV_FIELDS, "scanner": SCANNER_FIELDS}
# 原配置的类名与观测组名的对应
GROUP_CLASSES = {"policy": "PolicyCfg", "critic": "CriticCfg", "priv": "PrivCfg", "scanner": "ScannerCfg"}


def q_groups(task: str) -> tuple[str, ...]:
    """Q 观测包含的观测组（按拼接顺序）。平地任务没有 scanner（原 flat_env_cfg 把 observations.scanner 设为 None）。"""
    if task == "isaac_flat":
        return ("critic", "priv")
    if task == "isaac_rough":
        return ("critic", "priv", "scanner")
    raise ObservationError(f"未知任务 {task!r}")


def expected_q_dim(task: str) -> int:
    """由原配置推出的预期 C（平地 54、复杂地形 241）；运行时仍须 validate_observations 核对。"""
    return sum(f.dim for group in q_groups(task) for f in GROUP_FIELDS[group])


def describe_observation_fields(task: str) -> list[dict]:
    """逐项列出策略帧与 Q 观测的字段：组、名称、列区间、缩放、噪声与含义。"""
    rows, offset = [], 0
    for f in POLICY_FIELDS:
        rows.append({"target": "policy_frame", "group": "policy", "field": f.name, "columns": (offset, offset + f.dim),
                     "scale": f.scale, "noisy": f.noisy, "meaning": f.meaning})
        offset += f.dim
    offset = 0
    for group in q_groups(task):
        for f in GROUP_FIELDS[group]:
            rows.append({"target": "q_observation", "group": group, "field": f.name, "columns": (offset, offset + f.dim),
                         "scale": f.scale, "noisy": f.noisy, "meaning": f.meaning})
            offset += f.dim
    return rows


def parse_observation_config(env_cfg_source: str) -> dict[str, list[dict]]:
    """从原版 rough_env_cfg.py 静态解析各观测组的项：{组名: [{name, func, scale, noisy}, ...]}（按声明顺序）。

    注释掉的项不在 AST 中，自然不会被读到（例如原配置中注释掉的 policy.height_scan）。
    """
    module = ast.parse(env_cfg_source)
    observations = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
    classes = {n.name: n for n in observations.body if isinstance(n, ast.ClassDef)}
    parsed = {}
    for group, class_name in GROUP_CLASSES.items():
        terms = []
        for stmt in classes[class_name].body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                keywords = {kw.arg: kw.value for kw in stmt.value.keywords}
                func = keywords["func"]
                scale = ast.literal_eval(keywords["scale"]) if "scale" in keywords else None
                terms.append({"name": stmt.targets[0].id, "func": func.attr if isinstance(func, ast.Attribute) else func.id,
                              "scale": tuple(scale) if isinstance(scale, list) else scale, "noisy": "noise" in keywords})
        parsed[group] = terms
    return parsed


def baseline_observation_config() -> dict[str, list[dict]]:
    """读取固定快照中的原配置并解析（需要本地有 baseline/ppo 快照）。"""
    return parse_observation_config(read_baseline_file(ENV_CFG_PATH))


def validate_observations(task: str, term_names: dict[str, list[str]], term_dims: dict[str, list[tuple[int, ...]]],
                          policy_frame_dim: int | None = None) -> int:
    """用 Isaac 运行时的 observation_manager.active_terms / group_obs_term_dim 核对字段名称、顺序与维数，返回实际 C。

    逐项比较名称与维数（不只比较总长度），因此调换两个同维字段也会被发现。
    异常：ObservationError，列出全部不一致。
    """
    problems = []
    for group in ("policy", *q_groups(task)):
        expected = [(f.name, f.dim) for f in GROUP_FIELDS[group]]
        names = term_names.get(group)
        dims = term_dims.get(group)
        if names is None or dims is None:
            problems.append(f"运行时缺少观测组 {group}")
            continue
        actual = [(n, int(d[-1])) for n, d in zip(names, dims)]
        if actual != expected:
            problems.append(f"观测组 {group} 字段不一致：预期 {expected}，实际 {actual}")
        if group == "policy" and any(len(d) != 1 for d in dims):
            problems.append(f"policy 组的每项应为单帧（Isaac 历史长度 0），实际形状 {dims}")
    if policy_frame_dim is not None and policy_frame_dim != POLICY_FRAME_DIM:
        problems.append(f"策略帧维数为 {policy_frame_dim}，应为 {POLICY_FRAME_DIM}")
    if problems:
        raise ObservationError("观测接口校验失败：\n- " + "\n- ".join(problems))
    return expected_q_dim(task)


def read_observation_groups(observations: dict[str, torch.Tensor], task: str, q_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把 Isaac observation_manager.compute() 的结果整理成 (策略帧 [B,33], Q 观测 [B,C])。

    Q 观测按 [critic | priv | scanner] 拼接（与原 PPO wrapper 相同）。形状或数值不符时报错，不截断或补齐。
    只拼接，不复制额外内存以外的处理；调用方负责在需要时克隆（E05 在重置前快照时克隆）。
    """
    frame = observations["policy"]
    if frame.dim() != 2 or frame.shape[1] != POLICY_FRAME_DIM:
        raise ObservationError(f"policy 观测应为 [B,{POLICY_FRAME_DIM}]（Isaac 历史长度 0），实际为 {list(frame.shape)}")
    q = torch.cat([observations[group] for group in q_groups(task)], dim=-1)
    if q.dim() != 2 or q.shape[1] != q_dim or q.shape[0] != frame.shape[0]:
        raise ObservationError(f"Q 观测应为 [{frame.shape[0]},{q_dim}]，实际为 {list(q.shape)}")
    if not (bool(torch.isfinite(frame).all()) and bool(torch.isfinite(q).all())):
        raise ObservationError("观测含 NaN 或 inf")
    return frame, q


def velocity_label(q_observation: torch.Tensor) -> torch.Tensor:
    """速度监督标签 [B,3]：Q 观测前 3 列，即 base_lin_vel × 2.0（不是 m/s）。"""
    start, end = VELOCITY_LABEL_COLUMNS
    return q_observation[:, start:end]


def velocity_label_to_mps(label: torch.Tensor) -> torch.Tensor:
    """把 2.0 缩放的速度标签或 N01 速度估计换算回 m/s。"""
    return label / VELOCITY_LABEL_SCALE


def push_frame(history: torch.Tensor, pushes: torch.Tensor, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Isaac Lab 2.3.0 CircularBuffer 规则推送一帧，返回 (新历史 [B,10,33], 新推送计数 [B])，不修改输入。

    pushes 为每个环境自上次重置以来的推送次数；pushes == 0 的环境用该帧填满全部槽位（重置后的第一帧），
    其余环境丢弃最旧帧、在末尾追加。返回新张量，因此可以安全地作为快照保存。
    """
    shifted = torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)
    first = (pushes == 0).reshape(-1, 1, 1)
    filled = frame.unsqueeze(1).expand_as(history)
    return torch.where(first, filled, shifted), pushes + 1


def valid_length(pushes: torch.Tensor) -> torch.Tensor:
    """有效历史长度 = min(推送次数, 10)，即 D01 的 history_length。"""
    return torch.clamp(pushes, max=HISTORY_LENGTH)


class PolicyHistory:
    """每个环境的 10 帧策略历史与推送计数（Isaac CircularBuffer 的等价实现，单独管理以避免重复推进）。"""

    def __init__(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        self.history = torch.zeros(num_envs, HISTORY_LENGTH, POLICY_FRAME_DIM, device=device)
        self.pushes = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """重置指定环境：历史清零、推送计数归零（与 CircularBuffer.reset 相同）。原地修改。"""
        self.history[env_ids] = 0.0
        self.pushes[env_ids] = 0

    def commit(self, env_ids: torch.Tensor, history: torch.Tensor, pushes: torch.Tensor) -> None:
        """把 push_frame 的结果写回指定环境（只写这些行，其余环境不变）。"""
        self.history[env_ids] = history[env_ids]
        self.pushes[env_ids] = pushes[env_ids]
