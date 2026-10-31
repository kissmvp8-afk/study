"""E05 终止、截断与重置前观测：把 Isaac Lab 的一步整理成 T02 需要的 reset / step 语义。

公共环境数据接口（T02 Trainer、T04 run_evaluation、T03 恢复都使用；原定义在 T02 fake_env.py，现移到这里，
字段与语义不变，fake_env 改为从这里导入）：
    EnvObservation：下一次交互所用的观测与编号；
    EnvStep：执行动作后**重置前**的真实下一观测（next_*）、真实奖励、结束原因，以及下一次交互的 observation。

一步的执行顺序（run_isaac_step，复刻锁定的 Isaac Lab 2.3.0 ManagerBasedRLEnv.step，只在重置前插入一次观测计算）：
    1. 处理动作 → decimation 次物理步 → episode_length_buf += 1          （hooks.apply_action_and_simulate）
    2. 计算终止（terminated）与到时（time_outs）                         （hooks.compute_terminations）
    3. 计算奖励（reward_manager.compute(dt)，真实奖励，不含 PPO 的超时价值补偿） （hooks.compute_rewards）
    4. **重置前观测**：此刻的状态就是动作真正到达的末状态，克隆保存       （hooks.compute_observations）
    5. 重置结束的环境（_reset_idx，会原地改写仿真状态与各 manager）        （hooks.reset_envs）
    6. 更新指令、执行 interval 事件                                       （hooks.after_reset）
    7. **重置后观测**：下一次交互使用                                     （hooks.compute_observations）
锁定版本中第 5 步发生在 step 返回之前，返回的观测对结束的环境已经是新一轮的，所以不能只包装 step 的返回值。

每个环境每步的数据来源：
    未结束：next_* 与 observation 都来自第 7 步（指令与 interval 事件更新之后，正是下一次决策看到的状态）；
            历史推进一次（推入第 7 步的帧）。第 4 步对这些环境的计算结果不使用，因此不会多用一次噪声样本。
    结束：next_* 来自第 4 步（重置前的末状态；历史 = 旧历史推入第 4 步的帧）；observation 来自第 7 步
          （历史清零后推入重置后第一帧并填满，有效长度 1）。两份数据是两个不同的物理状态，各采样一次噪声。
    按环境选择时，掩码按字段维度展开：历史用 [B,1,1]、Q 观测用 [B,1]（作者 SAC 的二维掩码不能直接用于 [B,10,33]）。
同一步既到达终止条件又到时：按真终止处理（classify_endings）。

编号：env_id 固定为 0..B-1；episode_id 在该环境结束后加一；step_index 是本轮内的动作序号，重置后归零。
reset(first_episode_ids) 让 T03 恢复训练后编号接着往后排；新一轮从 step 0 开始，D02 不会把它接到旧经历上。

设备：hooks 在仿真设备上（例如 cuda:0）；本类把动作搬到仿真设备，把观测、奖励与标记搬到学习器设备（默认 CPU），
每步一次 GPU→CPU 传输。学习器目前只在 CPU 上验证。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "environment_control" / "observation_interface"))

from observations import PolicyHistory, push_frame, read_observation_groups, valid_length  # noqa: E402


@dataclass(frozen=True)
class EnvObservation:
    """下一次交互所用的观测与编号（[E,...]）。"""

    policy_history: torch.Tensor  # [E,10,33] float32
    q_observation: torch.Tensor  # [E,C] float32
    history_length: torch.Tensor  # [E] int64
    env_id: torch.Tensor  # [E] int64
    episode_id: torch.Tensor  # [E] int64
    step_index: torch.Tensor  # [E] int64


@dataclass(frozen=True)
class EnvStep:
    """一步的结果：重置前的真实下一观测、奖励与结束原因，以及（重置后的）下一次交互观测。"""

    next_policy_history: torch.Tensor  # [E,10,33]
    next_q_observation: torch.Tensor  # [E,C]
    next_history_length: torch.Tensor  # [E]
    reward: torch.Tensor  # [E,1]
    terminated: torch.Tensor  # [E,1] bool
    time_out: torch.Tensor  # [E,1] bool，与 terminated 互斥
    observation: EnvObservation


class LifecycleError(RuntimeError):
    """一步的数据不完整或前后不一致（例如结束的环境缺少重置前观测）。"""


@dataclass(frozen=True)
class RawStep:
    """run_isaac_step 的原始结果（仍在仿真设备上）。"""

    pre_reset: dict[str, torch.Tensor] | None  # 第 4 步：重置前观测（已克隆）
    post_reset: dict[str, torch.Tensor]  # 第 7 步：重置后观测
    reward: torch.Tensor  # [B]
    terminated: torch.Tensor  # [B] bool，termination_manager.terminated
    truncated: torch.Tensor  # [B] bool，termination_manager.time_outs
    reset_env_ids: torch.Tensor  # [K]


def classify_endings(terminated: torch.Tensor, truncated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (terminated [B,1], time_out [B,1])：time_out 为纯时间截断 = 到时且未真终止（真终止优先）。"""
    terminated = terminated.reshape(-1, 1).bool()
    return terminated, truncated.reshape(-1, 1).bool() & ~terminated


def run_isaac_step(hooks, action: torch.Tensor) -> RawStep:
    """按文件头的 7 步顺序执行一步并返回原始结果。

    hooks 需提供：apply_action_and_simulate(action)、compute_terminations() -> (terminated, time_outs)、
    compute_rewards() -> [B]、compute_observations() -> {组名: 张量}、reset_envs(env_ids)、after_reset()。
    E01 的 TitaSacEnv 用 Isaac 的各个 manager 实现这些方法；CPU 测试用可控的模拟实现检查顺序与数据。
    """
    hooks.apply_action_and_simulate(action)
    terminated, truncated = hooks.compute_terminations()
    reward = hooks.compute_rewards()
    # 重置前观测：克隆，避免后续重置或观测计算原地复用同一块内存。
    pre_reset = {name: value.clone() for name, value in hooks.compute_observations().items()}
    reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
    if reset_env_ids.numel() > 0:
        hooks.reset_envs(reset_env_ids)
    hooks.after_reset()
    post_reset = hooks.compute_observations()
    return RawStep(pre_reset, post_reset, reward, terminated, truncated, reset_env_ids)


def capture_final_observations(history: PolicyHistory, frame_pre: torch.Tensor, q_pre: torch.Tensor):
    """结束环境的重置前末状态：(末历史 [B,10,33], 末推送计数 [B], 末 Q 观测 [B,C])，均为新张量（快照）。

    末历史 = 当前历史推入重置前的帧，不写回 history，因此之后的重置不会覆盖它。
    """
    final_history, final_pushes = push_frame(history.history, history.pushes, frame_pre)
    return final_history, final_pushes, q_pre.clone()


def build_step_record(history: PolicyHistory, ended: torch.Tensor, frame_pre: torch.Tensor | None, q_pre: torch.Tensor | None,
                      frame_post: torch.Tensor, q_post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成 next_*（经验记录用的真实下一状态），并按规则推进 history（每个环境一次）。

    返回 (next_policy_history, next_history_length, next_q_observation)。
    副作用：未结束的环境推入 frame_post；结束的环境先清零再推入 frame_post（重置后的第一帧）。
    异常：LifecycleError：有环境结束却没有重置前观测。
    """
    ended = ended.reshape(-1).bool()
    post_history, post_pushes = push_frame(history.history, history.pushes, frame_post)
    if bool(ended.any()):
        if frame_pre is None or q_pre is None:
            raise LifecycleError("有环境结束，但没有重置前观测，无法保存真实末状态")
        final_history, final_pushes, final_q = capture_final_observations(history, frame_pre, q_pre)
        next_history = torch.where(ended.reshape(-1, 1, 1), final_history, post_history)
        next_pushes = torch.where(ended, final_pushes, post_pushes)
        next_q = torch.where(ended.reshape(-1, 1), final_q, q_post)
    else:
        next_history, next_pushes, next_q = post_history, post_pushes, q_post
    continuing = (~ended).nonzero(as_tuple=False).squeeze(-1)
    history.commit(continuing, post_history, post_pushes)
    ended_ids = ended.nonzero(as_tuple=False).squeeze(-1)
    if ended_ids.numel() > 0:
        history.reset(ended_ids)
        fresh_history, fresh_pushes = push_frame(history.history, history.pushes, frame_post)
        history.commit(ended_ids, fresh_history, fresh_pushes)
    return next_history, valid_length(next_pushes), next_q


class EpisodeLifecycle:
    """T02 环境接口（reset / step）的实现：hooks 负责仿真操作，本类维护历史、编号并生成 EnvObservation / EnvStep。

    参数：
        hooks: 提供 run_isaac_step 所需方法，以及 num_envs、device、reset_all() -> 观测字典。
        task: "isaac_flat" 或 "isaac_rough"（决定 Q 观测拼接哪些组）。
        q_dim: 运行时核对过的 C（E02 validate_observations 的返回值）。
        output_device: 学习器设备。
    """

    def __init__(self, hooks, task: str, q_dim: int, output_device: str | torch.device = "cpu") -> None:
        self.hooks, self.task, self.q_dim = hooks, task, q_dim
        self.num_envs = int(hooks.num_envs)
        self.output_device = torch.device(output_device)
        self.history = PolicyHistory(self.num_envs, device=self.output_device)
        self.env_id = torch.arange(self.num_envs, dtype=torch.int64)
        self.episode_id = torch.zeros(self.num_envs, dtype=torch.int64)
        self.step_index = torch.zeros(self.num_envs, dtype=torch.int64)

    def _read(self, observations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        moved = {name: value.to(self.output_device) for name, value in observations.items()}
        return read_observation_groups(moved, self.task, self.q_dim)

    def _observation(self, q_post: torch.Tensor) -> EnvObservation:
        return EnvObservation(self.history.history.clone(), q_post.clone(), valid_length(self.history.pushes),
                              self.env_id.clone(), self.episode_id.clone(), self.step_index.clone())

    def reset(self, first_episode_ids: torch.Tensor | None = None) -> EnvObservation:
        """重置全部环境并返回初始观测；first_episode_ids 用于恢复训练后继续编号（不恢复中断前的仿真现场）。"""
        frame, q = self._read(self.hooks.reset_all())
        all_ids = torch.arange(self.num_envs)
        self.history.reset(all_ids)
        new_history, pushes = push_frame(self.history.history, self.history.pushes, frame)
        self.history.commit(all_ids, new_history, pushes)
        self.episode_id = torch.zeros(self.num_envs, dtype=torch.int64) if first_episode_ids is None else first_episode_ids.to(torch.int64).clone()
        self.step_index = torch.zeros(self.num_envs, dtype=torch.int64)
        return self._observation(q)

    def step(self, action: torch.Tensor) -> EnvStep:
        """执行一步策略动作（策略坐标，原样交给 Isaac 动作项做一次 E03 换算），返回 EnvStep。"""
        raw = run_isaac_step(self.hooks, action.detach().to(self.hooks.device))
        terminated, time_out = classify_endings(raw.terminated.to(self.output_device), raw.truncated.to(self.output_device))
        ended = (terminated | time_out).reshape(-1)
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool)
        reset_mask[raw.reset_env_ids.to("cpu")] = True
        if not torch.equal(reset_mask, ended.cpu()):
            raise LifecycleError("被重置的环境与终止 / 到时标记不一致")
        frame_post, q_post = self._read(raw.post_reset)
        frame_pre, q_pre = self._read(raw.pre_reset) if raw.pre_reset is not None else (None, None)
        next_history, next_length, next_q = build_step_record(self.history, ended, frame_pre, q_pre, frame_post, q_post)
        self.step_index += 1
        ended_cpu = ended.cpu()
        self.episode_id[ended_cpu] += 1
        self.step_index[ended_cpu] = 0
        return EnvStep(next_policy_history=next_history, next_q_observation=next_q, next_history_length=next_length,
                       reward=raw.reward.reshape(-1, 1).to(self.output_device, torch.float32), terminated=terminated,
                       time_out=time_out, observation=self._observation(q_post))
