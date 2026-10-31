"""E01 任务元数据：原版配置的静态读取、运行时元数据的核对、工作轮速依据，以及把这些值填进 T01 运行配置（纯 Python）。

本文件不导入 Isaac。Isaac 相关部分（创建环境、从 manager 读取运行时元数据）在 isaac_env.py，
它把读到的值装进本文件的 RuntimeMetadata，再交给这里的纯函数核对、填配置；这样核对逻辑可以在 CPU 上测试。

时间关系（原版 TitaRoughEnvCfg.__post_init__，平地配置继承且不改写）：
    物理步长 sim.dt = 0.005 s，decimation = 4 ⇒ 策略周期 step_dt = 4 × 0.005 = 0.02 s（50 Hz）；
    episode_length_s = 20 s ⇒ 每轮最多 ceil(20 / 0.02) = 1000 次策略动作（Isaac max_episode_length）。
    每次 env.step 先执行 4 次物理步再计算终止、奖励与观测，D03 的折扣 γ 按“每次策略动作”计。

工作轮速（rad/s）：TITA 的轮子是速度控制，E04 需要一个工作轮速范围来确定动作边界。
    原版指令：lin_vel_x ∈ [−1, 1] m/s、lin_vel_y = 0、ang_vel_z ∈ [−1, 1] rad/s（heading 模式下也被夹在此范围内）。
    无滑动差速运动学：外侧轮轮速 = (|vx| + |wz| × 半轮距) / 轮半径。用默认姿态下的轮心距（URDF 正运动学，
    半轮距 ≈ 0.2825 m）与轮半径 0.0925 m，跟踪最大指令所需轮速 ≈ 13.86 rad/s。
    这是**下限依据**，不是工作范围：跟踪误差、腿部姿态变化（轮距随 leg_1 改变）、地形与打滑都会需要额外余量。
    模型 velocity_limit_sim = 30 rad/s 只是上限参考。余量没有已验证的依据，因此工作轮速由用户显式给出，
    本文件只检查它落在 [需求下限, 模型上限] 内，不提供默认值。
"""

from __future__ import annotations

import ast
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in ("environment_control/action_contract", "environment_control/observation_interface",
              "training_validation/configuration", "environment_control/robot_environment"):
    sys.path.insert(0, str(_ROOT / _path))

from assets import ROBOT_ASSET, WHEEL_JOINTS  # noqa: E402
from baseline_source import ENV_CFG_PATH, FLAT_ENV_CFG_PATH, read_baseline_file  # noqa: E402
from config import PPO, TARGET_ENVIRONMENT, RunConfig, isaac_config, validate_configuration  # noqa: E402
from contract import ActionContract, ControlType  # noqa: E402
from observations import validate_observations  # noqa: E402

MODEL_WHEEL_VELOCITY_LIMIT = 30.0  # 原版 DDT_TITA_CFG wheels.velocity_limit_sim（rad/s），只作上限参考
LEG_SOFT_LIMIT_FACTOR = 0.9  # 原版 DDT_TITA_CFG.soft_joint_pos_limit_factor
SAC_TASK_IDS = {"isaac_flat": "Tita-SAC-Velocity-Flat-v0", "isaac_rough": "Tita-SAC-Velocity-Rough-v0"}
PPO_TASK_IDS = {"isaac_flat": "DDT-Velocity-Flat-Tita-v0", "isaac_rough": "DDT-Velocity-Rough-Tita-v0"}
DEBUG_WINDOWS = "Windows 接入调试值（只验证能跑通，不是训练规模）"


class MetadataError(ValueError):
    """静态配置、运行时元数据或轮速设置不一致；消息列出全部问题。"""


@dataclass(frozen=True)
class RuntimeMetadata:
    """从 Isaac 环境读取的实际元数据（由 isaac_env.read_runtime_metadata 填写，取第 0 号环境的数值）。

    joint_names: articulation 的关节顺序（观测中 joint_pos / joint_vel 的顺序）。
    default_joint_pos: 关节名 → 默认角（rad）；soft_joint_pos_limits: 关节名 → (下, 上)（rad）；
    joint_vel_limits: 关节名 → PhysX 最大关节速度（rad/s；轮子应为执行器 velocity_limit_sim = 30）。
    action_terms: [(动作项名, 解析出的关节名, scale, 控制类型 "position"/"velocity")]，按 action_manager 顺序。
    obs_term_names / obs_term_dims: observation_manager.active_terms / group_obs_term_dim。
    command_ranges: 指令名 → (下, 上)。
    """

    task: str
    num_envs: int
    sim_device: str
    joint_names: tuple[str, ...]
    default_joint_pos: dict[str, float]
    soft_joint_pos_limits: dict[str, tuple[float, float]]
    joint_vel_limits: dict[str, float]
    action_terms: tuple[tuple[str, tuple[str, ...], float, str], ...]
    physics_dt: float
    decimation: int
    step_dt: float
    max_episode_length: int
    obs_term_names: dict[str, list[str]]
    obs_term_dims: dict[str, list[tuple[int, ...]]]
    command_ranges: dict[str, tuple[float, float]]


def _post_init_assignments(source: str, class_name: str) -> dict[str, object]:
    """读取类 __post_init__ 中 `self.xxx = 字面量` 形式的赋值，返回 {"xxx": 值}（非字面量的赋值跳过）。"""
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__post_init__":
                    found = {}
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                            target = ast.unparse(stmt.targets[0])
                            if target.startswith("self."):
                                try:
                                    found[target[5:]] = ast.literal_eval(stmt.value)
                                except ValueError:
                                    pass
                    return found
    raise MetadataError(f"源文件中没有 {class_name}.__post_init__")


def baseline_timing(repo_dir: Path = _ROOT) -> dict[str, float]:
    """从固定快照读取物理步长、decimation 与回合时长，并推出策略周期与每轮最大步数。

    平地配置继承复杂地形配置；这里同时确认平地 __post_init__ 没有改写这三项，否则两类任务的周期会不同。
    """
    rough = _post_init_assignments(read_baseline_file(ENV_CFG_PATH, repo_dir), "TitaRoughEnvCfg")
    flat = _post_init_assignments(read_baseline_file(FLAT_ENV_CFG_PATH, repo_dir), "TitaFlatEnvCfg")
    keys = ("sim.dt", "decimation", "episode_length_s")
    overridden = [k for k in keys if k in flat]
    if overridden or any(k not in rough for k in keys):
        raise MetadataError(f"原版周期设置缺失或被平地配置改写：{overridden}")
    step_dt = rough["decimation"] * rough["sim.dt"]
    return {"physics_dt": rough["sim.dt"], "decimation": rough["decimation"], "step_dt": step_dt,
            "episode_length_s": rough["episode_length_s"],
            # 与 Isaac Lab max_episode_length = ceil(episode_length_s / step_dt) 相同；先四舍五入去掉浮点误差。
            "max_episode_length": math.ceil(round(rough["episode_length_s"] / step_dt, 9))}


def baseline_command_ranges(repo_dir: Path = _ROOT) -> dict[str, tuple[float, float]]:
    """从固定快照的 CommandsCfg.base_velocity.ranges 读取 lin_vel_x / lin_vel_y / ang_vel_z（heading 含 math.pi，不读取）。"""
    module = ast.parse(read_baseline_file(ENV_CFG_PATH, repo_dir))
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("UniformVelocityCommandCfg.Ranges"):
            ranges = {}
            for kw in node.keywords:
                if kw.arg in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                    low, high = ast.literal_eval(kw.value)
                    ranges[kw.arg] = (float(low), float(high))
            return ranges
    raise MetadataError("没有找到 UniformVelocityCommandCfg.Ranges")


def required_wheel_speed(command_ranges: dict[str, tuple[float, float]], half_track: float, wheel_radius: float) -> float:
    """跟踪最大指令的无滑动外侧轮速（rad/s）= (max|vx| + max|wz| × 半轮距) / 轮半径。

    原版 lin_vel_y 恒为 0（轮足无法横移），若不为 0 则不能用差速模型，直接报错。
    """
    if any(abs(v) > 0.0 for v in command_ranges.get("lin_vel_y", (0.0, 0.0))):
        raise MetadataError("lin_vel_y 不为 0，差速轮速估算不适用")
    vx = max(abs(v) for v in command_ranges["lin_vel_x"])
    wz = max(abs(v) for v in command_ranges["ang_vel_z"])
    return (vx + wz * half_track) / wheel_radius


def wheel_speed_ranges(limit: float, required: float, model_limit: float = MODEL_WHEEL_VELOCITY_LIMIT) -> dict[str, tuple[float, float]]:
    """把用户给定的工作轮速上限变成 E04 的对称范围 (−limit, limit)。

    对称：前进/后退指令范围对称，左右轮关节轴方向也可能互为镜像，对称范围对两种符号都成立。
    异常：limit 不在 [required, model_limit] 内——低于需求下限会让最大指令无法跟踪；高于模型上限没有意义。
    """
    if limit is None or not math.isfinite(limit):
        raise MetadataError("必须显式给出工作轮速上限（rad/s）；没有默认值")
    if not required <= limit <= model_limit:
        raise MetadataError(f"工作轮速上限 {limit} rad/s 不在 [{required:.2f}（跟踪最大指令所需）, {model_limit}（模型上限）] 内")
    return {name: (-float(limit), float(limit)) for name in WHEEL_JOINTS}


def compare_with_baseline(runtime: RuntimeMetadata, contract: ActionContract, timing: dict, command_ranges: dict,
                          urdf_soft_limits: dict[str, tuple[float, float]] | None = None, tol: float = 1e-4) -> int:
    """逐项核对运行时元数据与原版配置 / E03 动作表 / URDF，返回运行时 Q 观测维数 C。

    核对：动作项展开后的关节顺序、控制类型与 scale 等于 E03；默认角等于 E03 默认目标；8 个受控关节都存在；
    周期与回合长度；指令范围；观测组字段与维数（E02 validate_observations）；给出 URDF 时腿部软限位与 URDF 推算一致。
    tol 用于 float32 运行时值与 Python 浮点的比较。
    异常：MetadataError，列出全部问题（E02 的观测错误原样放入列表）。
    """
    problems = []
    flat_actions = [(joint, scale, kind) for _, joints, scale, kind in runtime.action_terms for joint in joints]
    expected = [(e.joint_name, e.scale, "position" if e.control_type is ControlType.POSITION else "velocity")
                for e in contract.entries]
    if [a[0] for a in flat_actions] != [e[0] for e in expected]:
        problems.append(f"动作关节顺序 {[a[0] for a in flat_actions]} 与 E03 {[e[0] for e in expected]} 不一致")
    else:
        for (joint, scale, kind), (_, e_scale, e_kind) in zip(flat_actions, expected):
            if abs(scale - e_scale) > tol or kind != e_kind:
                problems.append(f"{joint}: 运行时 ({kind}, scale={scale}) 与 E03 ({e_kind}, scale={e_scale}) 不一致")
    missing = [e.joint_name for e in contract.entries if e.joint_name not in runtime.joint_names]
    if missing:
        problems.append(f"机器人缺少受控关节 {missing}")
    for e in contract.entries:
        if e.control_type is ControlType.POSITION and e.joint_name in runtime.default_joint_pos:
            if abs(runtime.default_joint_pos[e.joint_name] - e.default_target) > tol:
                problems.append(f"{e.joint_name} 默认角 {runtime.default_joint_pos[e.joint_name]} 与 E03 {e.default_target} 不一致")
    for key in ("physics_dt", "step_dt"):
        if abs(getattr(runtime, key) - timing[key]) > 1e-9:
            problems.append(f"{key}={getattr(runtime, key)} 与原配置 {timing[key]} 不一致")
    for key in ("decimation", "max_episode_length"):
        if getattr(runtime, key) != timing[key]:
            problems.append(f"{key}={getattr(runtime, key)} 与原配置 {timing[key]} 不一致")
    for name, (low, high) in command_ranges.items():
        actual = runtime.command_ranges.get(name)
        if actual is None or abs(actual[0] - low) > tol or abs(actual[1] - high) > tol:
            problems.append(f"指令 {name} 范围 {actual} 与原配置 {(low, high)} 不一致")
    if urdf_soft_limits is not None:
        for name, (low, high) in urdf_soft_limits.items():
            actual = runtime.soft_joint_pos_limits.get(name)
            if actual is None or abs(actual[0] - low) > tol or abs(actual[1] - high) > tol:
                problems.append(f"{name} 软限位 {actual} 与 URDF 推算 {(low, high)} 不一致")
    q_dim = None
    try:
        q_dim = validate_observations(runtime.task, runtime.obs_term_names, runtime.obs_term_dims)
    except ValueError as error:
        problems.append(str(error))
    if problems:
        raise MetadataError("运行时元数据与原版不一致：\n- " + "\n- ".join(problems))
    return q_dim


def leg_soft_limits(runtime: RuntimeMetadata, contract: ActionContract) -> dict[str, tuple[float, float]]:
    """取位置控制关节（6 个腿部关节）的运行时软限位；轮子是速度控制，其位置限位不进入 E04。"""
    return {e.joint_name: runtime.soft_joint_pos_limits[e.joint_name]
            for e in contract.entries if e.control_type is ControlType.POSITION}


def fill_isaac_run_config(runtime: RuntimeMetadata, contract: ActionContract, q_dim: int, wheel_ranges: dict,
                          required_speed: float, *, seed: int, steps_per_iteration: int, capacity_per_env: int,
                          debug_sizes: bool) -> RunConfig:
    """用运行时元数据与用户设置填写 T01 isaac_config，并调用 validate_configuration（for_run=True）。

    运行时读取的值（q_dim、软限位）保持运行时来源；工作轮速标为用户设置；采集规模在 debug_sizes=True 时标为
    Windows 接入调试值，否则标为用户设置。num_envs 取实际创建的环境数（与 Trainer 的一致性检查相同）。
    学习器设备固定为 CPU（T01 只实现了 CPU 学习器），仿真设备写入 environment["sim_device"]。
    """
    config = isaac_config(runtime.task, seed=seed)
    size_source = DEBUG_WINDOWS if debug_sizes else "用户设置"
    limit = wheel_ranges[WHEEL_JOINTS[0]][1]
    task = replace(config.task, q_dim=q_dim, joint_pos_limits=leg_soft_limits(runtime, contract),
                   wheel_speed_ranges=dict(wheel_ranges))
    collection = replace(config.collection, num_envs=runtime.num_envs, steps_per_iteration=steps_per_iteration,
                         capacity_per_env=capacity_per_env)
    sources = dict(config.sources)
    sources.update({
        "task.q_dim": "E02 运行时 group_obs_term_dim（validate_observations 核对）",
        "task.velocity_label_columns": f"{PPO}：critic 第一项 base_lin_vel × 2.0（E02 运行时核对）",
        "task.joint_pos_limits": f"E01 运行时 soft_joint_pos_limits（factor {LEG_SOFT_LIMIT_FACTOR}）",
        "task.wheel_speed_ranges": (f"用户设置 ±{limit} rad/s（跟踪最大指令所需 {required_speed:.2f}，"
                                    f"模型上限 {MODEL_WHEEL_VELOCITY_LIMIT}）"),
        "collection.num_envs": f"{size_source}（实际创建的环境数）",
        "collection.steps_per_iteration": size_source,
        "collection.capacity_per_env": size_source,
        "seed": "用户设置",
    })
    environment = dict(TARGET_ENVIRONMENT, robot_assets_commit=ROBOT_ASSET["commit"], sim_device=runtime.sim_device,
                       sac_task_id=SAC_TASK_IDS[runtime.task])
    filled = replace(config, task=task, collection=collection, environment=environment, sources=sources)
    validate_configuration(filled)
    return filled
