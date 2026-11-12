"""独立进程检查真实Isaac物理步与Tita资产；不是SAC学习或控制验收。"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scene", "tita"), required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("positive num-envs and steps required")

    # 沿用历史Kit启动顺序，避免在其初始化中再加载这两个二进制模块。
    import h5py  # noqa: F401
    import tensordict  # noqa: F401
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True, device="cuda:0", enable_cameras=False,
        kit_args="--allow-root --/app/extensions/fsWatcherEnabled=false",
    )
    app = launcher.app
    env = None
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.mode == "scene":
            import isaaclab.sim as sim_utils
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
            ground = sim_utils.GroundPlaneCfg()
            ground.func("/World/groundPlane", ground)
            sim.reset()
            for _ in range(args.steps):
                sim.step()
            sim.stop()
            result = {"mode": "scene", "physical_steps": args.steps, "dt_s": 0.005}
        else:
            import gymnasium as gym
            import ddt_lab.tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg
            task = "DDT-Velocity-Flat-Tita-v0"
            gym.spec(task)
            cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs)
            env = gym.make(task, cfg=cfg)
            env.reset(seed=0)
            raw = env.unwrapped
            assert raw.num_envs == args.num_envs
            actions = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=raw.device)
            for _ in range(args.steps):
                env.step(actions)
                assert torch.isfinite(raw.scene["robot"].data.root_state_w).all().item()
            result = {"mode": "tita", "task": task, "num_envs": raw.num_envs,
                      "vector_steps": args.steps, "transitions": args.steps * raw.num_envs,
                      "action_dim": raw.action_manager.total_action_dim,
                      "control_success_evaluated": False}
        torch.cuda.synchronize()
        result.update(status="passed", CUDA_device=torch.cuda.get_device_name(0), SAC_updates=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
