def make_trainer(args):
    """构造原 TRAIN 派生 SAC 任务，仅改变计划规定的站立比例与采集/学习规模。"""
    from assets import WHEEL_JOINTS, parse_urdf_joints, soft_joint_limits, verify_urdf, wheel_half_track, wheel_radius
    from baseline_source import load_baseline_action_contract
    from contract import ControlType, validate_action_contract
    from isaac_env import TitaSacHooks, make_env, make_sac_cfg, read_runtime_metadata, sac_config_differences, urdf_path_from_cfg
    from lifecycle import EpisodeLifecycle
    from metadata import LEG_SOFT_LIMIT_FACTOR, baseline_command_ranges, baseline_timing, compare_with_baseline, fill_isaac_run_config, required_wheel_speed, wheel_speed_ranges
    from config import export_run_configuration
    from runner import Trainer
    contract = load_baseline_action_contract()
    validate_action_contract(contract)
    cfg = make_sac_cfg('isaac_flat', 128, args.seed, args.device or 'cuda:0')
    cfg = configure(cfg, 'R20')
    urdf = urdf_path_from_cfg(cfg)
    asset_check = verify_urdf(urdf)
    joints = parse_urdf_joints(urdf)
    legs = [e.joint_name for e in contract.entries if e.control_type is ControlType.POSITION]
    urdf_soft = soft_joint_limits(joints, legs, LEG_SOFT_LIMIT_FACTOR)
    default_pose = {e.joint_name: e.default_target for e in contract.entries if e.control_type is ControlType.POSITION}
    half_track, radius = wheel_half_track(joints, default_pose), wheel_radius(urdf, WHEEL_JOINTS[0])
    timing, commands = baseline_timing(), baseline_command_ranges()
    required = required_wheel_speed(commands, half_track, radius)
    env = make_env('isaac_flat', cfg)
    args._env = env  # 之后任何异常都由 finally 关闭。
    runtime = read_runtime_metadata(env, 'isaac_flat')
    q_dim = compare_with_baseline(runtime, contract, timing, commands, urdf_soft)
    if q_dim != 54: raise ValueError(f'Flat Q dimension {q_dim}, expected 54')
    ranges = wheel_speed_ranges(20., required, model_limit=min(runtime.joint_vel_limits[n] for n in WHEEL_JOINTS))
    config = fill_isaac_run_config(runtime, contract, q_dim, ranges, required, seed=args.seed,
                                  steps_per_iteration=8, capacity_per_env=512, debug_sizes=False)
    config = replace(config, name=PLAN_ID, device=args.learner_device,
                     collection=replace(config.collection, warmup_iterations=16),
                     algorithm=replace(config.algorithm, batch_size=256, critic_updates_per_iteration=128,
                                       auxiliary_updates_per_iteration=128, auxiliary_batch_size=256),
                     sources=config.sources | {'device': PLAN_ID + ': explicitly selected, independently validated learner',
                                               'collection.warmup_iterations': PLAN_ID,
                                               'algorithm.critic_updates_per_iteration': PLAN_ID,
                                               'algorithm.auxiliary_updates_per_iteration': PLAN_ID})
    export_run_configuration(config, args.output_dir / 'run_config.json')
    # Isaac 的完整生效任务配置也保存；不是只保留训练器参数。
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(args.output_dir / 'environment_config.yaml'), cfg)
    write_json(args.output_dir / 'metadata.json', dict(plan_id=PLAN_ID, arm=args.arm,
        rel_standing_envs=cfg.commands.base_velocity.rel_standing_envs,
        runtime=asdict(runtime), asset_check=asset_check, sac_vs_ppo=sac_config_differences('isaac_flat'),
        learner_device=config.device, replay_device='cpu', seed=0, warmup_in_total=True,
        initial_episode_length_randomized=False, fixed_evaluation_commands_in_training=False))
    hooks = R7PHooks(env, False)
    trainer = Trainer(config, EpisodeLifecycle(hooks, 'isaac_flat', q_dim), contract)
    return trainer, hooks
