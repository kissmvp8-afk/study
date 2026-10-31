"""T01 版本、依赖与训练配置：集中定义一次训练实际使用的全部设置，并逐字段记录来源。

参数来源分三类（每个字段在 RunConfig.sources 中写明）：
    - 作者默认：leggedrobotics/rsl_rl_sac e0d243a 中 SAC.__init__ / OffPolicyRunner 的默认值。
    - 论文候选：根 README 与本模块 README 记录的论文候选值（gamma=0.97、n=5、tau=0.003、初始 std=0.15、
      actor/Q 学习率 2e-4、alpha 学习率 2e-5）。候选值只是实验起点，不是已验证的 TITA 最优值。
    - TITA 适配 / 原 PPO：本项目为 TITA 选定或沿用原 PPO 的设置（网络维数、辅助学习、归一化等）。
另有 CPU 调试值：只用于假环境下的流程检查，不能当作真实任务的默认值。

未确定项保持 None，并在 validate_configuration(for_run=True) 时拒绝运行：
    Isaac 配置中的 Q 观测维数（E02 运行时确定，平地预期 54、复杂地形预期 241）、关节软限位与工作轮速（E04 / E01）、
    每轮采集步数与经验池容量（按显存预算选择）、机器人资产版本（UPSTREAM.json 尚未锁定）。

配置生效：T02 Trainer 构造时把这里的每个值显式传给对应模块（见 T02 runner.py），不依赖各模块的默认常量。
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "experience_returns" / "transition_schema"))
sys.path.insert(0, str(_ROOT / "experience_returns" / "replay_buffer"))

# _expected_layout 是 D01 的字段布局表；与 D02 相同，经验池估算直接复用它，避免出现第二份字段表。
from buffer import WRITE_SEQ_DTYPE, bytes_per_record  # noqa: E402
from schema import _expected_layout  # noqa: E402

CONFIG_FORMAT_VERSION = 1
AUTHOR = "作者默认（rsl_rl_sac e0d243a）"
PAPER = "论文候选"
PPO = "原 PPO（DDT_Lab 701a49c / 快照 8fe40aa）"
TITA = "TITA 适配暂定值（未经训练验证）"
DEBUG = "CPU 调试值（仅用于假环境流程检查）"
UNDETERMINED = "未确定"


class ConfigurationError(ValueError):
    """配置不完整、组合不合法或资源不足；消息列出全部问题。"""


@dataclass(frozen=True)
class AlgorithmConfig:
    """SAC 及辅助学习的数值设置；单位见各字段，更新频率以“次 Critic 更新”或“轮迭代”计。"""

    gamma: float  # 折扣（每个策略周期 0.02 s）
    n_steps: int  # D03 多步回报的最大步数
    tau: float  # N08 目标 Q 跟随比例
    target_update_every: int  # 每多少次 Critic 更新执行一次慢更新
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    auxiliary_lr: float
    initial_alpha: float
    target_entropy_scale: float  # 目标熵 = −scale × 8（策略动作坐标，见 N07）
    initial_std: float  # N03 内部初始标准差
    max_grad_norm: float  # Critic、Actor、辅助学习共用的梯度裁剪上限
    batch_size: int  # 每次 Critic / Actor / alpha 更新的回放样本数
    critic_updates_per_iteration: int  # 每轮迭代的 Critic 更新次数（作者 num_learning_epochs × num_mini_batches）
    actor_update_every: int  # 每多少次 Critic 更新做一次 Actor 更新（作者 policy_frequency）
    alpha_update_every: int  # 每多少次 Critic 更新做一次 alpha 更新（作者每次都更新）
    auxiliary_updates_per_iteration: int  # 每轮迭代的辅助学习次数，与 Critic 更新次数无关
    auxiliary_batch_size: int
    normalizer_eps: float  # D04 加在标准差上的稳定常数


@dataclass(frozen=True)
class NetworkConfig:
    """网络维数（原 PPO TITA 配置）。"""

    encoder_backbone_dims: tuple[int, ...]
    encoder_latent_hidden_dim: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]


@dataclass(frozen=True)
class CollectionConfig:
    """采集规模与经验池容量。"""

    num_envs: int
    steps_per_iteration: int | None  # 每轮每个环境的采集步数（作者 num_steps_per_env，无默认值）
    warmup_iterations: int  # 前几轮只采集不学习（作者 start_training）；采集仍用有界 Actor
    capacity_per_env: int | None  # D02 每个环境保存的记录条数


@dataclass(frozen=True)
class TaskConfig:
    """任务与物理边界。人工限位只允许出现在 CPU 调试配置中。"""

    task: str  # "fake_cpu_debug"、"isaac_flat" 或 "isaac_rough"
    q_dim: int | None  # Q 观测维数 C，由 E02 运行时观测组确定
    velocity_label_columns: tuple[int, int]  # 速度监督标签在原始 Q 观测中的列 [start, end)
    joint_pos_limits: dict[str, tuple[float, float]] | None  # 关节软限位（rad）
    wheel_speed_ranges: dict[str, tuple[float, float]] | None  # 工作轮速（rad/s）
    action_limits_artificial: bool  # True 表示上面两项是人工数据
    initial_targets: tuple[float, ...] | None  # E04 希望的初始物理目标；None 表示 E03 默认目标


@dataclass(frozen=True)
class RunConfig:
    """一次运行的完整配置。sources 为“字段路径 → 来源说明”。"""

    name: str
    mode: str  # "cpu_debug" 或 "isaac_sim"
    seed: int
    device: str
    task: TaskConfig
    algorithm: AlgorithmConfig
    network: NetworkConfig
    collection: CollectionConfig
    environment: dict[str, str | None]  # 目标软件环境与资产版本
    sources: dict[str, str] = field(default_factory=dict)


AUTHOR_ALGORITHM = AlgorithmConfig(
    gamma=0.998, n_steps=1, tau=0.005, target_update_every=1, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3,
    auxiliary_lr=1e-3, initial_alpha=0.05, target_entropy_scale=1.0, initial_std=1.0, max_grad_norm=1.0,
    batch_size=256, critic_updates_per_iteration=1, actor_update_every=2, alpha_update_every=1,
    auxiliary_updates_per_iteration=1, auxiliary_batch_size=256, normalizer_eps=1e-2,
)
AUTHOR_SOURCES = {f"algorithm.{name}": AUTHOR for name in AlgorithmConfig.__dataclass_fields__}
AUTHOR_SOURCES.update({
    "algorithm.initial_std": "作者默认 init_noise_std=1.0（论文改为 0.15）",
    "algorithm.auxiliary_lr": f"{PPO} learning_rate 初值",
    "algorithm.auxiliary_updates_per_iteration": TITA,
    "algorithm.auxiliary_batch_size": TITA,
    "algorithm.normalizer_eps": f"{PPO} 与作者 EmpiricalNormalization 的 eps",
    "algorithm.alpha_update_every": "作者：每次 Critic 更新都更新 alpha",
    "algorithm.target_update_every": "作者：每次 Critic 更新都慢更新",
})

PAPER_ALGORITHM = replace(AUTHOR_ALGORITHM, gamma=0.97, n_steps=5, tau=0.003, initial_std=0.15,
                          actor_lr=2e-4, critic_lr=2e-4, alpha_lr=2e-5)
PAPER_SOURCES = dict(AUTHOR_SOURCES)
PAPER_SOURCES.update({f"algorithm.{name}": PAPER for name in
                      ("gamma", "n_steps", "tau", "initial_std", "actor_lr", "critic_lr", "alpha_lr")})

TITA_NETWORK = NetworkConfig(encoder_backbone_dims=(128, 64), encoder_latent_hidden_dim=32,
                             actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128))
NETWORK_SOURCES = {f"network.{name}": PPO for name in NetworkConfig.__dataclass_fields__}

TARGET_ENVIRONMENT = {"isaac_sim": "5.1", "isaac_lab": "2.3.0", "python": "3.11", "robot_assets_commit": None}

# CPU 调试用的人工限位：与 E04 测试中的人工数据一致，**不是 TITA 实际限位**。
_ARTIFICIAL_JOINT_POS_LIMITS = {
    "joint_left_leg_1": (-0.3, 0.7), "joint_left_leg_2": (0.0, 1.6), "joint_left_leg_3": (-2.4, -0.6),
    "joint_right_leg_1": (-0.3, 0.7), "joint_right_leg_2": (0.0, 1.6), "joint_right_leg_3": (-2.4, -0.6),
}
_ARTIFICIAL_WHEEL_SPEED_RANGES = {"joint_left_leg_4": (-10.0, 20.0), "joint_right_leg_4": (-10.0, 20.0)}


def cpu_debug_config(seed: int = 0) -> RunConfig:
    """CPU 假环境调试配置：论文候选的算法值 + 缩小的规模 + 人工限位。只用于流程检查，结果不是正式复现。"""
    algorithm = replace(PAPER_ALGORITHM, batch_size=16, critic_updates_per_iteration=4, auxiliary_batch_size=16)
    sources = dict(PAPER_SOURCES) | dict(NETWORK_SOURCES)
    sources.update({
        "algorithm.batch_size": DEBUG, "algorithm.critic_updates_per_iteration": DEBUG,
        "algorithm.auxiliary_batch_size": DEBUG,
        "collection.num_envs": DEBUG, "collection.steps_per_iteration": DEBUG,
        "collection.warmup_iterations": DEBUG, "collection.capacity_per_env": DEBUG,
        "task.task": "T02 假环境（非 Isaac 仿真）",
        "task.q_dim": "假环境的 Q 观测维数（非 E02 观测组）",
        "task.velocity_label_columns": f"{PPO}：base_lin_vel 为 critic 观测第一项，E02 待确认",
        "task.joint_pos_limits": "人工数据（同 E04 测试），非 TITA 实际软限位",
        "task.wheel_speed_ranges": "人工数据（同 E04 测试），非 TITA 工作轮速",
        "task.initial_targets": "E03 默认目标",
        "seed": DEBUG, "device": DEBUG,
    })
    return RunConfig(
        name="cpu-debug-fake-env", mode="cpu_debug", seed=seed, device="cpu",
        task=TaskConfig(task="fake_cpu_debug", q_dim=6, velocity_label_columns=(0, 3),
                        joint_pos_limits=dict(_ARTIFICIAL_JOINT_POS_LIMITS),
                        wheel_speed_ranges=dict(_ARTIFICIAL_WHEEL_SPEED_RANGES),
                        action_limits_artificial=True, initial_targets=None),
        algorithm=algorithm, network=TITA_NETWORK,
        collection=CollectionConfig(num_envs=3, steps_per_iteration=8, warmup_iterations=1, capacity_per_env=64),
        environment={"python": "3.11", "torch": "CPU", "isaac_sim": None, "isaac_lab": None, "robot_assets_commit": None},
        sources=sources,
    )


def isaac_config(task: str = "isaac_flat", seed: int = 0) -> RunConfig:
    """后续 Isaac 仿真配置（平地 isaac_flat / 复杂地形 isaac_rough）。未确定项保持 None，当前不能运行。"""
    if task not in ("isaac_flat", "isaac_rough"):
        raise ConfigurationError(f"未知任务 {task!r}")
    sources = dict(PAPER_SOURCES) | dict(NETWORK_SOURCES)
    sources.update({
        "collection.num_envs": f"{PPO} 场景 num_envs=4096",
        "collection.steps_per_iteration": f"{UNDETERMINED}：作者 num_steps_per_env 无默认值",
        "collection.warmup_iterations": "作者默认 start_training=0",
        "collection.capacity_per_env": f"{UNDETERMINED}：按显存预算选择（见 estimate_memory）",
        "task.task": "E01 平地 / 复杂地形任务（接入前不能运行）",
        "task.q_dim": f"{UNDETERMINED}：E02 运行时确定（{'平地预期 54' if task == 'isaac_flat' else '复杂地形预期 241'}）",
        "task.velocity_label_columns": f"{PPO}：base_lin_vel 为 critic 观测第一项，E02 待确认",
        "task.joint_pos_limits": f"{UNDETERMINED}：需 E01 从 soft_joint_pos_limits 读取",
        "task.wheel_speed_ranges": f"{UNDETERMINED}：需人工确定，不能直接用模型 30 rad/s",
        "task.initial_targets": "E03 默认目标",
        "seed": TITA, "device": TITA,
    })
    return RunConfig(
        name=f"{task}-sac", mode="isaac_sim", seed=seed, device="cuda",
        task=TaskConfig(task=task, q_dim=None, velocity_label_columns=(0, 3), joint_pos_limits=None,
                        wheel_speed_ranges=None, action_limits_artificial=False, initial_targets=None),
        algorithm=PAPER_ALGORITHM, network=TITA_NETWORK,
        collection=CollectionConfig(num_envs=4096, steps_per_iteration=None, warmup_iterations=0, capacity_per_env=None),
        environment=dict(TARGET_ENVIRONMENT), sources=sources,
    )


def _field_paths(config: RunConfig) -> list[str]:
    """需要来源说明的全部字段路径（不含 name、mode、environment、sources 本身）。"""
    paths = ["seed", "device"]
    for group in ("task", "algorithm", "network", "collection"):
        paths += [f"{group}.{name}" for name in getattr(config, group).__dataclass_fields__ if name != "action_limits_artificial"]
    return paths


def undetermined_fields(config: RunConfig) -> list[str]:
    """值为 None 的字段路径（initial_targets 的 None 表示使用 E03 默认目标，不算未确定）。"""
    found = []
    for path in _field_paths(config):
        group, _, name = path.partition(".")
        value = getattr(getattr(config, group), name) if name else getattr(config, group)
        if value is None and path != "task.initial_targets":
            found.append(path)
    return found


def validate_configuration(config: RunConfig, for_run: bool = True) -> None:
    """检查配置：来源齐全、数值范围、模式与任务一致、人工限位不进入仿真配置；for_run=True 时拒绝未确定项。

    异常：ConfigurationError，列出全部问题。
    """
    problems = [f"{p} 缺少来源说明" for p in _field_paths(config) if p not in config.sources]
    a, c, t = config.algorithm, config.collection, config.task
    if not 0.0 < a.gamma <= 1.0:
        problems.append(f"gamma={a.gamma} 必须在 (0, 1]")
    if not 0.0 <= a.tau <= 1.0:
        problems.append(f"tau={a.tau} 必须在 [0, 1]")
    for name in ("n_steps", "target_update_every", "batch_size", "actor_update_every", "alpha_update_every",
                 "auxiliary_batch_size", "critic_updates_per_iteration"):
        if getattr(a, name) < 1:
            problems.append(f"{name}={getattr(a, name)} 必须 ≥ 1")
    if a.auxiliary_updates_per_iteration < 0 or c.warmup_iterations < 0:
        problems.append("auxiliary_updates_per_iteration 与 warmup_iterations 不能为负")
    for name in ("actor_lr", "critic_lr", "alpha_lr", "auxiliary_lr", "initial_alpha", "initial_std", "max_grad_norm", "normalizer_eps"):
        if not getattr(a, name) > 0.0:
            problems.append(f"{name}={getattr(a, name)} 必须为正")
    if config.mode == "cpu_debug" and t.task != "fake_cpu_debug":
        problems.append("cpu_debug 模式只能使用假环境任务")
    if config.mode == "isaac_sim":
        if t.task not in ("isaac_flat", "isaac_rough"):
            problems.append("isaac_sim 模式必须使用 isaac_flat 或 isaac_rough")
        if t.action_limits_artificial:
            problems.append("人工限位不能用于 Isaac 仿真配置")
    if t.velocity_label_columns[1] - t.velocity_label_columns[0] != 3:
        problems.append("速度标签必须是 3 列")
    if t.q_dim is not None and t.velocity_label_columns[1] > t.q_dim:
        problems.append(f"速度标签列 {t.velocity_label_columns} 超出 q_dim={t.q_dim}")
    if c.capacity_per_env is not None and a.n_steps > c.capacity_per_env:
        problems.append(f"n_steps={a.n_steps} 超过 capacity_per_env={c.capacity_per_env}")
    if for_run:
        problems += [f"{p} 未确定，不能运行" for p in undetermined_fields(config)]
    if problems:
        raise ConfigurationError("配置检查失败：\n- " + "\n- ".join(problems))


def estimate_memory(config: RunConfig, q_dim: int | None = None, capacity_per_env: int | None = None) -> dict:
    """按 D01 实际字段形状与 dtype 估算 D02 经验池的张量存储，返回总量与按字段的分项（降序）。

    估算范围：只含经验池字段存储与每槽位的写入序号；不含网络参数、优化器状态、前向激活、
    PyTorch 缓存分配器余量以及 Isaac Sim 自身占用。q_dim / capacity_per_env 未确定时必须显式传入场景值。
    """
    q_dim = q_dim if q_dim is not None else config.task.q_dim
    capacity = capacity_per_env if capacity_per_env is not None else config.collection.capacity_per_env
    if q_dim is None or capacity is None:
        raise ConfigurationError("q_dim 或 capacity_per_env 未确定，请显式给出估算场景")
    records = config.collection.num_envs * capacity
    parts = {name: records * math.prod(shape) * torch.empty((), dtype=dtype).element_size()
             for name, (shape, dtype) in _expected_layout(q_dim).items()}
    parts["write_seq"] = records * torch.empty((), dtype=WRITE_SEQ_DTYPE).element_size()
    total = sum(parts.values())
    assert total == records * bytes_per_record(q_dim)  # 与 D02 的计算一致
    ordered = dict(sorted(parts.items(), key=lambda kv: kv[1], reverse=True))
    return {"records": records, "q_dim": q_dim, "capacity_per_env": capacity, "total_bytes": total,
            "by_field_bytes": ordered, "scope": "仅 D02 经验池张量存储"}


def check_memory_budget(estimate: dict, available_bytes: int) -> None:
    """预算不足时报错并指出占用最多的字段；不自动缩小实验规模。"""
    if estimate["total_bytes"] > available_bytes:
        name, size = next(iter(estimate["by_field_bytes"].items()))
        raise ConfigurationError(
            f"经验池需要 {estimate['total_bytes'] / 1e9:.2f} GB，超过可用 {available_bytes / 1e9:.2f} GB；"
            f"占用最多的是 {name}（{size / 1e9:.2f} GB）。请调整容量或 num_envs，本函数不会自动缩小规模"
        )


def version_report(repo_root: Path = _ROOT) -> dict:
    """记录代码版本、运行环境与固定来源：git 提交与是否有已跟踪文件的改动、Python / torch 版本、UPSTREAM.json。"""
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    upstream = json.loads((repo_root / "UPSTREAM.json").read_text(encoding="utf-8"))
    return {
        "git_commit": git("rev-parse", "HEAD") or None,
        "git_tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ppo_snapshot_commit": upstream["ppo"]["snapshot_commit"],
        "sac_reference_commit": upstream["sac_reference"]["commit"],
        "robot_assets_commit": upstream["robot_assets"]["commit"] or "未锁定",
    }


def config_to_dict(config: RunConfig) -> dict:
    """转换为只含 JSON 基本类型的字典（tuple 变为 list）。"""
    return json.loads(json.dumps(asdict(config)))


def config_from_dict(data: dict) -> RunConfig:
    """config_to_dict 的逆变换；把 list 还原为 tuple。"""
    def tuples(d: dict, keys: tuple[str, ...]) -> dict:
        return {k: (tuple(v) if k in keys and v is not None else v) for k, v in d.items()}

    task = dict(data["task"])
    for key in ("joint_pos_limits", "wheel_speed_ranges"):
        if task[key] is not None:
            task[key] = {k: tuple(v) for k, v in task[key].items()}
    return RunConfig(
        name=data["name"], mode=data["mode"], seed=data["seed"], device=data["device"],
        task=TaskConfig(**tuples(task, ("velocity_label_columns", "initial_targets"))),
        algorithm=AlgorithmConfig(**data["algorithm"]),
        network=NetworkConfig(**tuples(data["network"], ("encoder_backbone_dims", "actor_hidden_dims", "critic_hidden_dims"))),
        collection=CollectionConfig(**data["collection"]),
        environment=dict(data["environment"]), sources=dict(data["sources"]),
    )


def export_run_configuration(config: RunConfig, path: Path, repo_root: Path = _ROOT) -> dict:
    """把本次生效配置、来源、未确定项、版本信息与经验池估算写成 JSON，返回写入的内容。"""
    try:
        memory = estimate_memory(config)
    except ConfigurationError as exc:
        memory = {"undetermined": str(exc)}
    record = {
        "format_version": CONFIG_FORMAT_VERSION,
        "config": config_to_dict(config),
        "undetermined_fields": undetermined_fields(config),
        "versions": version_report(repo_root),
        "replay_memory_estimate": memory,
    }
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def load_run_configuration(path: Path) -> RunConfig:
    """读取 export_run_configuration 写出的文件并还原 RunConfig；格式版本不符时报错。"""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("format_version") != CONFIG_FORMAT_VERSION:
        raise ConfigurationError(f"配置格式版本 {record.get('format_version')} 与当前 {CONFIG_FORMAT_VERSION} 不符")
    return config_from_dict(record["config"])
