"""E01 Isaac 接入：SAC 任务配置与注册、按锁定 Isaac Lab 顺序实现 E05 钩子、读取运行时元数据。

依赖：Isaac Sim 5.1、Isaac Lab 2.3.0、已 `pip install -e source/ddt_lab` 的原版 DDT_Lab 701a49c。
**必须在 isaaclab.app.AppLauncher 启动之后再导入本文件**（Isaac 的模块需要已运行的 Kit），启动入口见 run_sac_isaac.py。
本文件没有在 Isaac 中执行过；CPU 测试只检查它的语法与结构（test_environment.py），真实行为待 Windows + Isaac 验证。

SAC 配置沿用原版的范围：
    TitaSacFlatEnvCfg / TitaSacRoughEnvCfg 直接继承原版 TitaFlatEnvCfg / TitaRoughEnvCfg：场景、机器人与执行器
    （DDT_TITA_CFG）、指令、奖励、终止、事件、课程、物理步长与 decimation 全部沿用。只改一处：
    observations.policy.history_length 10 → 0，由 E02 PolicyHistory 维护 10 帧历史（原因见 observations.py 文件头）。
    sac_config_differences 对比两份配置的 to_dict()，确认差异只有这一项。
环境注册：新增 Tita-SAC-Velocity-Flat-v0 / Tita-SAC-Velocity-Rough-v0；原版 DDT-Velocity-*-Tita-v0 不改动，
    注册前检查同名 id 不存在，注册后检查原版 id 仍指向原配置类。

TitaSacHooks 逐行复刻 Isaac Lab 2.3.0 ManagerBasedRLEnv.step（envs/manager_based_rl_env.py），只把它拆成 E05
run_isaac_step 需要的 6 个钩子，并由 E05 在重置前多调用一次 compute_observations。不调用 env.step()，
因为 env.step 在返回前就已经重置结束的环境，拿不到重置前的末状态。
直接使用了两个私有成员（_sim_step_counter、_reset_idx），它们只在锁定版本下核对过，升级 Isaac Lab 时必须重新核对。
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointPositionAction, JointVelocityAction
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass

from ddt_lab.tasks.manager_based.locomotion.robots.tita.flat_env_cfg import TitaFlatEnvCfg
from ddt_lab.tasks.manager_based.locomotion.robots.tita.rough_env_cfg import TitaRoughEnvCfg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata import PPO_TASK_IDS, SAC_TASK_IDS, MetadataError, RuntimeMetadata  # noqa: E402

ALLOWED_SAC_DIFFERENCES = {"observations.policy.history_length": (10, 0)}


@configclass
class TitaSacFlatEnvCfg(TitaFlatEnvCfg):
    """平地 SAC 任务：原版平地配置，只关闭 policy 组的 Isaac 历史。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 0  # 历史由 E02 维护；每次 compute 只得到当前帧 [B,33]


@configclass
class TitaSacRoughEnvCfg(TitaRoughEnvCfg):
    """复杂地形 SAC 任务：原版复杂地形配置，只关闭 policy 组的 Isaac 历史。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 0


SAC_CFG_CLASSES = {"isaac_flat": TitaSacFlatEnvCfg, "isaac_rough": TitaSacRoughEnvCfg}
PPO_CFG_CLASSES = {"isaac_flat": TitaFlatEnvCfg, "isaac_rough": TitaRoughEnvCfg}


def register_sac_tasks() -> None:
    """注册 SAC 任务 id（可重复调用）；不覆盖、不修改原版 PPO id。

    异常：MetadataError：SAC id 已被其他配置占用，或原版 PPO id 未注册 / 不再指向原配置类。
    """
    for task, task_id in SAC_TASK_IDS.items():
        if task_id in gym.registry:
            if gym.spec(task_id).kwargs.get("env_cfg_entry_point") is not SAC_CFG_CLASSES[task]:
                raise MetadataError(f"{task_id} 已被其他配置注册")
            continue
        gym.register(id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": SAC_CFG_CLASSES[task]})
    for task, task_id in PPO_TASK_IDS.items():
        if task_id not in gym.registry or gym.spec(task_id).kwargs.get("env_cfg_entry_point") is not PPO_CFG_CLASSES[task]:
            raise MetadataError(f"原版 PPO 任务 {task_id} 未注册或不再指向 {PPO_CFG_CLASSES[task].__name__}")


def _flatten(value, prefix: str = "") -> dict:
    """把嵌套字典展开成 {"a.b.c": 值}，用于逐项比较两份配置。"""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: value}


def sac_config_differences(task: str) -> dict[str, tuple]:
    """比较原版 PPO 配置与 SAC 配置（两者都是新建的默认实例），返回 {路径: (PPO 值, SAC 值)}。

    异常：MetadataError：出现 ALLOWED_SAC_DIFFERENCES 以外的差异（例如奖励、指令或周期被改动）。
    """
    ppo, sac = _flatten(PPO_CFG_CLASSES[task]().to_dict()), _flatten(SAC_CFG_CLASSES[task]().to_dict())
    diffs = {key: (ppo.get(key), sac.get(key)) for key in sorted(set(ppo) | set(sac)) if ppo.get(key) != sac.get(key)}
    if diffs != ALLOWED_SAC_DIFFERENCES:
        raise MetadataError(f"SAC 配置与原版的差异超出允许范围：{diffs}")
    return diffs


def assert_no_isaac_history(cfg) -> None:
    """确认所有观测组与观测项都没有 Isaac 历史（E05 每步调用两次 compute，有历史就会被重复推进）。"""
    problems = []
    for group_name, group in vars(cfg.observations).items():
        if group is None or group_name.startswith("_"):
            continue
        if getattr(group, "history_length", None) not in (None, 0):
            problems.append(f"观测组 {group_name}.history_length={group.history_length}")
        for term_name, term in vars(group).items():
            # 组的 history_length 不为 None 时会覆盖各项的设置（Isaac Lab 2.3.0 observation_manager），只有组未设置时才看各项。
            if isinstance(term, ObservationTermCfg) and term.history_length not in (0, None) and group.history_length is None:
                problems.append(f"观测项 {group_name}.{term_name}.history_length={term.history_length}")
    if problems:
        raise MetadataError("SAC 环境不能使用 Isaac 观测历史：" + "；".join(problems))


def make_sac_cfg(task: str, num_envs: int, seed: int, sim_device: str):
    """新建 SAC 环境配置，并设置并行环境数、随机种子与仿真设备（其余保持原版）。"""
    cfg = SAC_CFG_CLASSES[task]()
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    cfg.sim.device = sim_device
    assert_no_isaac_history(cfg)
    return cfg


def urdf_path_from_cfg(cfg) -> Path:
    """原版 DDT_TITA_CFG.spawn.asset_path（由 ddt_robot.py 的 DDT_MODEL_DIR 拼出）实际指向的 URDF。"""
    return Path(cfg.scene.robot.spawn.asset_path)


def make_env(task: str, cfg) -> ManagerBasedRLEnv:
    """通过 SAC 任务 id 创建环境，返回未包装的 ManagerBasedRLEnv（gym.make 的包装层不改变 Isaac 行为）。"""
    register_sac_tasks()
    env = gym.make(SAC_TASK_IDS[task], cfg=cfg, render_mode=None).unwrapped
    if not isinstance(env, ManagerBasedRLEnv):
        raise MetadataError(f"{SAC_TASK_IDS[task]} 创建出的不是 ManagerBasedRLEnv")
    return env


class TitaSacHooks:
    """E05 钩子的 Isaac 实现：每个方法对应锁定 ManagerBasedRLEnv.step 中的一段（注释标出原代码顺序）。

    参数：
        env: 由 make_env 创建的环境。
        randomize_initial_episode_length: True 时 reset_all 之后把 episode_length_buf 随机打散到 [0, max)，
            与原 PPO 训练脚本 init_at_random_ep_len=True 相同，避免所有环境同时到时；默认 False（需用户决定）。
    属性：num_envs、device（仿真设备）、last_episode_log（最近一次重置时 Isaac 记录的各奖励项回合统计）。
    """

    def __init__(self, env: ManagerBasedRLEnv, randomize_initial_episode_length: bool = False) -> None:
        if len(env.recorder_manager.active_terms) > 0:
            # 原 step 在有 recorder 时会在重置前额外计算并记录观测；本适配器没有复刻这一分支，直接拒绝。
            raise MetadataError("SAC 适配器不支持启用 recorder 的环境")
        self.env = env
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.randomize_initial_episode_length = randomize_initial_episode_length
        self.last_episode_log: dict = {}

    def reset_all(self) -> dict[str, torch.Tensor]:
        """重置全部环境（env.reset：_reset_idx → 写入仿真 → sim.forward → 计算观测），返回观测组字典。"""
        observations, _ = self.env.reset()
        if self.randomize_initial_episode_length:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        return observations

    def apply_action_and_simulate(self, action: torch.Tensor) -> None:
        """原 step 第 1 段：处理动作（E03 换算 target = default + scale × a 只在这里做一次）→ decimation 次物理步 → 计数加一。"""
        env = self.env
        env.action_manager.process_action(action.to(env.device))
        env.recorder_manager.record_pre_step()
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
        for _ in range(env.cfg.decimation):  # 4 × 0.005 s = 一个 0.02 s 策略周期
            env._sim_step_counter += 1
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.recorder_manager.record_post_physics_decimation_step()
            if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)
        env.episode_length_buf += 1
        env.common_step_counter += 1

    def compute_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """原 step 第 2 段：返回 (terminated, time_outs) 的副本（manager 的缓冲下一步会被原地改写）。"""
        env = self.env
        env.reset_buf = env.termination_manager.compute()
        env.reset_terminated = env.termination_manager.terminated
        env.reset_time_outs = env.termination_manager.time_outs
        return env.reset_terminated.clone(), env.reset_time_outs.clone()

    def compute_rewards(self) -> torch.Tensor:
        """原 step 第 3 段：reward_manager.compute(dt=step_dt)，即原版各奖励项 × 权重 × dt 之和。

        这是环境给出的真实奖励；原 PPO 在 rollout 中给到时步加 γ·V 的补偿，SAC 不做这一步（截断由 D03 自举处理）。
        """
        env = self.env
        env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
        return env.reward_buf.clone()

    def compute_observations(self) -> dict[str, torch.Tensor]:
        """计算全部观测组。SAC 配置没有 Isaac 历史（assert_no_isaac_history），update_history 不起作用，调用两次也不会重复推进。"""
        observations = self.env.observation_manager.compute(update_history=False)
        self.env.obs_buf = observations
        return observations

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """原 step 第 4 段：重置结束的环境（课程、场景、reset 事件、各 manager、episode_length_buf 归零）。"""
        env = self.env
        env.recorder_manager.record_pre_reset(env_ids)
        env._reset_idx(env_ids)
        if env.sim.has_rtx_sensors() and env.cfg.rerender_on_reset:
            env.sim.render()
        env.recorder_manager.record_post_reset(env_ids)
        self.last_episode_log = dict(env.extras.get("log", {}))

    def after_reset(self) -> None:
        """原 step 第 5 段：更新指令（到期重采样、heading 转为 wz），执行 interval 事件（如推机器人）。"""
        env = self.env
        env.command_manager.compute(dt=env.step_dt)
        if "interval" in env.event_manager.available_modes:
            env.event_manager.apply(mode="interval", dt=env.step_dt)


def read_runtime_metadata(env: ManagerBasedRLEnv, task: str) -> RuntimeMetadata:
    """从已创建的环境读取实际元数据（第 0 号环境；默认角与限位在各环境间相同，域随机化只改增益与质量）。

    动作项的关节按 JointAction 自身的解析方式（find_joints + preserve_order）重新解析，避免读取私有的 _joint_names；
    控制类型由动作项类判断，不是位置/速度动作项时报错。
    """
    robot = env.scene["robot"]
    names = tuple(robot.joint_names)
    default = robot.data.default_joint_pos[0].tolist()
    soft = robot.data.soft_joint_pos_limits[0].tolist()
    vel = robot.data.joint_vel_limits[0].tolist()
    terms = []
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        if isinstance(term, JointVelocityAction):
            kind = "velocity"
        elif isinstance(term, JointPositionAction):
            kind = "position"
        else:
            raise MetadataError(f"动作项 {term_name} 的类型 {type(term).__name__} 不是关节位置/速度动作")
        _, joints = robot.find_joints(term.cfg.joint_names, preserve_order=term.cfg.preserve_order)
        if len(joints) != term.action_dim:
            raise MetadataError(f"动作项 {term_name} 解析出 {len(joints)} 个关节，action_dim={term.action_dim}")
        terms.append((term_name, tuple(joints), float(term.cfg.scale), kind))
    ranges = env.cfg.commands.base_velocity.ranges
    return RuntimeMetadata(
        task=task, num_envs=env.num_envs, sim_device=str(env.device), joint_names=names,
        default_joint_pos=dict(zip(names, default)),
        soft_joint_pos_limits={n: (float(lo), float(hi)) for n, (lo, hi) in zip(names, soft)},
        joint_vel_limits=dict(zip(names, vel)), action_terms=tuple(terms),
        physics_dt=float(env.physics_dt), decimation=int(env.cfg.decimation), step_dt=float(env.step_dt),
        max_episode_length=int(env.max_episode_length),
        obs_term_names={g: list(t) for g, t in env.observation_manager.active_terms.items()},
        obs_term_dims={g: [tuple(d) for d in dims] for g, dims in env.observation_manager.group_obs_term_dim.items()},
        command_ranges={k: tuple(float(v) for v in getattr(ranges, k)) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
    )
