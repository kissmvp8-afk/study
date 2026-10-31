"""E01 启动入口：在 Isaac Lab 中创建 TITA SAC 环境，核对元数据，做少量交互或短训练并保存 / 恢复。

用法（Windows PowerShell，在本仓库根目录、已激活装有 Isaac Lab 2.3.0 的 Python 环境；完整步骤见本目录 README.md）：
    python environment_control\\robot_environment\\run_sac_isaac.py --task isaac_flat --mode inspect --headless
    python environment_control\\robot_environment\\run_sac_isaac.py --task isaac_flat --mode interact --steps 50 --headless
    python environment_control\\robot_environment\\run_sac_isaac.py --task isaac_flat --mode train --wheel_speed_limit <rad/s> --headless

三种模式：
    inspect：创建环境 → 核对资产校验值、动作顺序、默认角、软限位、周期、指令、观测组维数、SAC/PPO 配置差异 → 写 metadata.json。
    interact：inspect 之后经 E05 EpisodeLifecycle 执行 --steps 次零动作（默认姿态、轮速 0），检查形状、有限值与结束标记。
    train：inspect 之后用运行时元数据填 T01 配置 → T02 Trainer 采集 / 学习 --iterations 轮 → T03 保存 → 新 Trainer 恢复后再训练 1 轮。
默认的环境数、轮数、每轮步数与容量都是**接入调试值**，只证明流程能在 Isaac 中跑通，不是训练规模，也不能说明机器人学会了行走。
工作轮速 --wheel_speed_limit 没有默认值：必须由用户按依据给出（见 metadata.py 文件头），程序只检查它在 [需求下限, 模型上限] 内。

路径都用 pathlib 处理；输出目录名不含冒号（Windows 不允许）。Isaac 模块只能在 AppLauncher 启动之后导入，所以导入放在 main() 内。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
DEBUG_NUM_ENVS, DEBUG_ITERATIONS, DEBUG_STEPS_PER_ITERATION, DEBUG_CAPACITY_PER_ENV = 16, 3, 8, 200


def parse_args() -> argparse.Namespace:
    """命令行参数；AppLauncher 追加 --headless、--device（仿真设备，默认 cuda:0）等 Isaac 参数。"""
    parser = argparse.ArgumentParser(description="TITA SAC 的 Isaac Lab 接入检查与短训练")
    parser.add_argument("--task", choices=("isaac_flat", "isaac_rough"), required=True)
    parser.add_argument("--mode", choices=("inspect", "interact", "train"), required=True)
    parser.add_argument("--num_envs", type=int, default=DEBUG_NUM_ENVS, help=f"并行环境数（接入调试值 {DEBUG_NUM_ENVS}）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50, help="interact 模式的交互次数（每次 0.02 s）")
    parser.add_argument("--wheel_speed_limit", type=float, default=None, help="工作轮速上限 rad/s（train 必填，无默认值）")
    parser.add_argument("--iterations", type=int, default=DEBUG_ITERATIONS, help=f"train 轮数（接入调试值 {DEBUG_ITERATIONS}）")
    parser.add_argument("--steps_per_iteration", type=int, default=None, help=f"每轮每环境采集步数（缺省用调试值 {DEBUG_STEPS_PER_ITERATION}）")
    parser.add_argument("--capacity_per_env", type=int, default=None, help=f"每环境经验池容量（缺省用调试值 {DEBUG_CAPACITY_PER_ENV}）")
    parser.add_argument("--randomize_initial_episode_length", action="store_true",
                        help="开局把各环境回合计数随机打散（原 PPO 脚本的做法，需用户决定）")
    parser.add_argument("--output_dir", type=Path, default=None, help="输出目录（缺省为 logs\\sac_isaac\\<时间>）")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mode == "train" and args.wheel_speed_limit is None:
        # 在启动 Isaac 之前就拒绝：工作轮速没有已验证的默认值，不能悄悄用模型上限 30 rad/s 代替。
        parser.error("train 模式必须给出 --wheel_speed_limit（rad/s），依据见 environment_control/robot_environment/README.md")
    return args


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main(args: argparse.Namespace) -> dict:
    """按 --mode 执行，返回写入 summary.json 的摘要。任何核对失败都抛异常（不带病继续）。"""
    for path in ("environment_control/robot_environment", "environment_control/episode_lifecycle",
                 "environment_control/observation_interface", "environment_control/action_contract",
                 "training_validation/configuration", "training_validation/training_loop", "training_validation/checkpoints"):
        sys.path.insert(0, str(ROOT / path))
    import torch

    from assets import ROBOT_ASSET, WHEEL_JOINTS, parse_urdf_joints, soft_joint_limits, verify_urdf, wheel_half_track, wheel_radius
    from baseline_source import load_baseline_action_contract
    from contract import ControlType, validate_action_contract
    from isaac_env import TitaSacHooks, make_env, make_sac_cfg, read_runtime_metadata, sac_config_differences, urdf_path_from_cfg
    from lifecycle import EpisodeLifecycle
    from metadata import (LEG_SOFT_LIMIT_FACTOR, SAC_TASK_IDS, baseline_command_ranges, baseline_timing, compare_with_baseline,
                          fill_isaac_run_config, required_wheel_speed, wheel_speed_ranges)

    output = args.output_dir or Path("logs") / "sac_isaac" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sim_device = args.device or "cuda:0"

    # 1. 静态来源：E03 动作表（固定快照）、URDF（锁定提交 + 校验值）、原版周期与指令。
    contract = load_baseline_action_contract()
    validate_action_contract(contract)
    cfg = make_sac_cfg(args.task, args.num_envs, args.seed, sim_device)
    urdf = urdf_path_from_cfg(cfg)
    asset_check = verify_urdf(urdf)  # 资产缺失时这里报告具体路径；"crlf_normalized" 表示 Windows 换行被转换
    joints = parse_urdf_joints(urdf)
    legs = [e.joint_name for e in contract.entries if e.control_type is ControlType.POSITION]
    urdf_soft = soft_joint_limits(joints, legs, LEG_SOFT_LIMIT_FACTOR)
    default_pose = {e.joint_name: e.default_target for e in contract.entries if e.control_type is ControlType.POSITION}
    half_track, radius = wheel_half_track(joints, default_pose), wheel_radius(urdf, WHEEL_JOINTS[0])
    timing, commands = baseline_timing(), baseline_command_ranges()
    required = required_wheel_speed(commands, half_track, radius)
    differences = sac_config_differences(args.task)

    # 2. 创建环境并核对运行时元数据（运行时值保持运行时来源）。
    env = make_env(args.task, cfg)
    runtime = read_runtime_metadata(env, args.task)
    q_dim = compare_with_baseline(runtime, contract, timing, commands, urdf_soft)
    model_wheel_limit = min(runtime.joint_vel_limits[name] for name in WHEEL_JOINTS)
    summary = {
        "mode": args.mode, "task": args.task, "task_id": SAC_TASK_IDS[args.task], "sim_device": runtime.sim_device,
        "learner_device": "cpu", "num_envs": runtime.num_envs, "q_dim": q_dim,
        "robot_asset": {"commit": ROBOT_ASSET["commit"], "urdf": str(urdf), "sha256_check": asset_check},
        "joint_names_articulation_order": runtime.joint_names, "action_order": contract.joint_names,
        "default_joint_pos": runtime.default_joint_pos, "soft_joint_pos_limits": runtime.soft_joint_pos_limits,
        "joint_vel_limits": runtime.joint_vel_limits,
        "timing": {"physics_dt": runtime.physics_dt, "decimation": runtime.decimation, "step_dt": runtime.step_dt,
                   "max_episode_length": runtime.max_episode_length},
        "command_ranges": runtime.command_ranges, "obs_term_dims": runtime.obs_term_dims,
        "sac_vs_ppo_cfg_differences": differences,
        "wheel_speed_basis": {"half_track_m": half_track, "wheel_radius_m": radius, "required_rad_s": required,
                              "model_limit_rad_s": model_wheel_limit},
    }
    _write_json(output / "metadata.json", summary)

    if args.mode == "interact":
        lifecycle = EpisodeLifecycle(TitaSacHooks(env, args.randomize_initial_episode_length), args.task, q_dim)
        observation = lifecycle.reset()
        counts = {"terminated": 0, "time_out": 0, "reward_sum": 0.0}
        for _ in range(args.steps):
            step = lifecycle.step(torch.zeros(runtime.num_envs, len(contract)))  # 零动作 = 默认姿态、轮速 0
            for tensor in (step.next_policy_history, step.next_q_observation, step.reward):
                if not bool(torch.isfinite(tensor).all()):
                    raise FloatingPointError("交互中出现非有限数值")
            counts["terminated"] += int(step.terminated.sum())
            counts["time_out"] += int(step.time_out.sum())
            counts["reward_sum"] += float(step.reward.sum())
            observation = step.observation
        summary["interact"] = dict(counts, steps=args.steps, policy_history_shape=list(observation.policy_history.shape),
                                   q_observation_shape=list(observation.q_observation.shape),
                                   episode_id=observation.episode_id.tolist()[:8], step_index=observation.step_index.tolist()[:8])

    if args.mode == "train":
        from checkpoint import restore_checkpoint, save_checkpoint
        from config import export_run_configuration
        from runner import Trainer

        ranges = wheel_speed_ranges(args.wheel_speed_limit, required, model_limit=model_wheel_limit)
        debug_sizes = args.steps_per_iteration is None or args.capacity_per_env is None
        config = fill_isaac_run_config(
            runtime, contract, q_dim, ranges, required, seed=args.seed,
            steps_per_iteration=args.steps_per_iteration or DEBUG_STEPS_PER_ITERATION,
            capacity_per_env=args.capacity_per_env or DEBUG_CAPACITY_PER_ENV, debug_sizes=debug_sizes)
        export_run_configuration(config, output / "run_config.json")
        lifecycle = EpisodeLifecycle(TitaSacHooks(env, args.randomize_initial_episode_length), args.task, q_dim)
        trainer = Trainer(config, lifecycle, contract)
        logs = trainer.train(args.iterations)
        checkpoint = save_checkpoint(trainer, output / "checkpoint.pt")
        restored = Trainer(config, lifecycle, contract)  # 新 Trainer：重新初始化网络并重置环境
        report = restore_checkpoint(restored, checkpoint)
        resumed = restored.train(1)
        summary["train"] = {"debug_sizes": debug_sizes, "iterations": args.iterations, "logs": logs,
                            "checkpoint": str(checkpoint), "resume_mode": report.mode, "after_resume": resumed}

    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"[E01] 输出目录：{output.resolve()}")
    env.close()
    return summary


if __name__ == "__main__":
    cli = parse_args()
    simulation_app = AppLauncher(cli).app
    try:
        main(cli)
    finally:
        simulation_app.close()
