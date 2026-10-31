"""T03 训练进度保存与恢复：把 T02 Trainer 的全部训练状态写成带格式版本的检查点，并有明确语义地恢复。

保存内容（作者 SAC.save 只含 Actor、Critic、优化器与 alpha，这里补齐 TITA 训练需要的全部状态）：
    - 网络：N01 编码器（含 BN 运行统计）、N02 projector、N03 Actor（含 E04 的 b、c 缓冲）、
      N04 在线与目标双 Q、N07 log_alpha；
    - 四个优化器（critic / actor / alpha / auxiliary）的状态；
    - D04 两套统计（mean、var、count、eps）；
    - T02 分项计数（决定 Actor 每 2 次更新的相位、预热与保存时机）、三个随机数生成器状态；
    - 生效配置、版本信息、E04 动作边界与 E03 关节顺序；
    - 可选：D02 经验池完整状态（字段、写入序号、写入次数）。
不保存：环境（仿真）现场。恢复后环境重新 reset，中断时未结束的那一轮不会接着进行；
新一轮的 episode_id 从保存时的编号加一开始，step_index=0，D02 只按显式编号连接，因此新旧经历不会被拼接。

两种恢复语义（describe_resume_mode 给出说明）：
    - exact_replay（带经验池）：网络、优化器、统计、计数、随机数与经验池都恢复，下一批回放抽样与学习可复现；
      采集因环境重新开始而不同。
    - model_only（不带经验池）：经验池为空，需要重新积累经历后才能学习，**不是精确接续**。

格式与兼容性：文件必须带 format="tita-sac-checkpoint" 与版本号；Q 观测维数、任务、环境数、网络维数、
E04 动作边界或 E03 关节顺序不一致时拒绝加载，不静默截断或补齐。原 PPO 的 V 网络评价的是状态而非“状态 + 动作”，
输入维数与含义都不同，不能当作 SAC 的 Q 参数加载；没有上述格式标记的文件一律拒绝。
读取使用 torch.load(weights_only=True)，只接受张量与基本类型，不执行任意 pickle。

原子保存：先写同目录下的临时文件并 fsync，再用 os.replace 替换正式文件；中途失败时删除临时文件，原检查点保持不变。
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
for _path in ("training_validation/configuration", "training_validation/training_loop", "experience_returns/replay_buffer"):
    sys.path.insert(0, str(_ROOT / _path))

from buffer import ReplayBuffer  # noqa: E402
from config import config_from_dict, config_to_dict, version_report  # noqa: E402

CHECKPOINT_FORMAT = "tita-sac-checkpoint"
CHECKPOINT_VERSION = 1
MODULE_NAMES = ("encoder", "projector", "actor", "critics", "targets", "temperature")
# 不一致时拒绝加载的配置字段：决定张量形状或动作含义。
STRUCTURAL_FIELDS = ("task.task", "task.q_dim", "collection.num_envs", "collection.capacity_per_env",
                     "network.encoder_backbone_dims", "network.encoder_latent_hidden_dim",
                     "network.actor_hidden_dims", "network.critic_hidden_dims")


class CheckpointError(ValueError):
    """检查点格式、版本或结构与当前训练不兼容。"""


@dataclass(frozen=True)
class ResumeReport:
    """恢复结果。mode 为 "exact_replay" 或 "model_only"；config_differences 列出非结构性配置差异。"""

    mode: str
    replay_restored: bool
    environment_state_restored: bool
    config_differences: tuple[str, ...]
    counters: dict
    saved_versions: dict


def _bounds_record(trainer) -> dict:
    b = trainer.bounds
    return {"joint_names": list(b.joint_names), "lower": list(b.lower), "upper": list(b.upper),
            "center": list(b.center), "half_width": list(b.half_width)}


def collect_training_state(trainer, include_replay: bool) -> dict:
    """汇总 Trainer 的全部训练状态（只读，张量为 state_dict 引用或副本）。"""
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": config_to_dict(trainer.config),
        "versions": version_report(),
        "action_bounds": _bounds_record(trainer),
        "contract_joint_names": list(trainer.contract.joint_names),
        "modules": {name: getattr(trainer, name).state_dict() for name in MODULE_NAMES},
        "optimizers": {name: opt.state_dict() for name, opt in trainer.optimizers.items()},
        "normalizers": trainer.normalizers.export_statistics(),
        "counters": asdict(trainer.counters),
        "generators": {name: g.get_state() for name, g in trainer.generators.items()},
        "next_episode_ids": (trainer.observation.episode_id + 1).tolist(),
        "environment_state_saved": False,
        "replay": trainer.buffer.state_dict() if include_replay else None,
    }


def save_checkpoint(trainer, path: Path, include_replay: bool = True) -> Path:
    """原子保存检查点到 path，返回 path。副作用：写文件；失败时不留下临时文件、不改动已有检查点。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = collect_training_state(trainer, include_replay)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def _get(config: dict, dotted: str):
    group, _, name = dotted.partition(".")
    return config[group][name] if name else config[group]


def restore_checkpoint(trainer, path: Path) -> ResumeReport:
    """把检查点恢复进一个新构造的 Trainer（同配置结构、同环境类型），返回 ResumeReport。

    检查顺序：格式与版本 → 结构性配置 → 动作边界与关节顺序 → 依次恢复网络、优化器、统计、计数、随机数、经验池，
    最后让环境开始新一轮（episode_id 从保存时加一）。格式、版本与结构检查全部通过之前不修改 Trainer；
    不兼容时抛 CheckpointError。
    """
    try:
        state = torch.load(Path(path), weights_only=True)
    except Exception as exc:  # 非张量 / 基本类型内容（任意 pickle）也在这里被拒绝
        raise CheckpointError(f"无法以 weights_only 方式读取 {path}：{exc}") from exc
    if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("不是 TITA SAC 检查点（缺少格式标记）；原 PPO 的 V 参数不能当作 Q 加载")
    if state.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(f"检查点版本 {state.get('version')} 与当前 {CHECKPOINT_VERSION} 不符")

    current = config_to_dict(trainer.config)
    problems = [f"{f}：检查点 {_get(state['config'], f)!r}，当前 {_get(current, f)!r}"
                for f in STRUCTURAL_FIELDS if _get(state["config"], f) != _get(current, f)]
    if state["action_bounds"] != _bounds_record(trainer):
        problems.append("E04 动作边界与当前配置推导的不同")
    if state["contract_joint_names"] != list(trainer.contract.joint_names):
        problems.append("E03 关节顺序不同")
    if problems:
        raise CheckpointError("检查点与当前训练结构不兼容：\n- " + "\n- ".join(problems))
    differences = tuple(sorted(
        f"{group}.{name}" for group in ("algorithm", "collection", "task")
        for name in current[group] if f"{group}.{name}" not in STRUCTURAL_FIELDS
        and state["config"][group][name] != current[group][name]
    ))

    for name in MODULE_NAMES:
        getattr(trainer, name).load_state_dict(state["modules"][name], strict=True)
    for name, optimizer in trainer.optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    trainer.normalizers.load_statistics(state["normalizers"])
    for name, value in state["counters"].items():
        setattr(trainer.counters, name, value)
    for name, generator in trainer.generators.items():
        generator.set_state(state["generators"][name])
    if state["replay"] is not None:
        trainer.buffer.load_state_dict(state["replay"])
    else:
        c, t = trainer.config.collection, trainer.config.task
        trainer.buffer = ReplayBuffer(c.num_envs, c.capacity_per_env, t.q_dim)  # 明确为空经验池
    # 仿真现场没有保存：环境开始新一轮，编号接着保存时的 episode_id 往后排。
    trainer.observation = trainer.env.reset(first_episode_ids=torch.tensor(state["next_episode_ids"], dtype=torch.int64))
    return ResumeReport(
        mode="exact_replay" if state["replay"] is not None else "model_only",
        replay_restored=state["replay"] is not None, environment_state_restored=False,
        config_differences=differences, counters=dict(state["counters"]), saved_versions=dict(state["versions"]),
    )


def describe_resume_mode(report: ResumeReport) -> str:
    """给出恢复语义的中文说明，供日志记录。"""
    if report.mode == "exact_replay":
        text = ("带经验池恢复：网络、优化器、统计、计数、随机数与经验池均已恢复，下一批回放抽样与学习可复现；"
                "环境现场未保存，采集从新一轮开始，与中断前继续运行的结果不同。")
    else:
        text = ("仅模型恢复：经验池为空，需要重新积累经历后才能学习，不是精确接续；"
                "环境现场未保存，采集从新一轮开始。")
    if report.config_differences:
        text += f"与检查点不同的非结构性配置：{', '.join(report.config_differences)}。"
    return text


def load_checkpoint_config(path: Path):
    """只读取检查点中的生效配置（便于按原配置重建 Trainer 后再恢复）。"""
    state = torch.load(Path(path), weights_only=True)
    if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("不是 TITA SAC 检查点")
    return config_from_dict(state["config"])
