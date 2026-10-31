"""T02 假环境：可控、确定性的人工环境，只用于在 CPU 上检查“采集 → 存储 → 抽样 → 更新”的数据流程。

它**不是** Isaac 仿真，也不模拟 TITA 动力学；结果不能称为仿真验证或机器人训练成功。

接口（T02 Trainer 与环境之间的约定，之后由 E01 / E05 在 Isaac Lab 上实现同一接口）：
    reset(first_episode_ids=None) -> EnvObservation
    step(action [E,8]) -> EnvStep
EnvStep 同时给出两种观测，这正是 E05 的职责：
    - next_*：执行动作后、任何重置之前的真实下一观测，写入 D01 记录；
    - observation：下一次交互使用的观测，结束的环境已经重置（新一轮 episode_id 加一、step_index=0）。
同一步既到达终止条件又到时，只报告真终止（E05 约定的真终止优先）。

人工动力学（每个环境独立，无随机性）：
    当前帧 f ∈ R^33；f' = 0.8 f + 0.2 · (W a)，W 为固定的 33×8 矩阵；奖励 r = 1 − 0.1 × 平均(a²)。
    Q 观测 = [2 × f[0:3] | f[3:C]]：前 3 列模仿原 PPO 中乘过 2.0 的 base_lin_vel（速度标签列），只是人工构造。
    重置帧 = −3 + 0.01 × 环境编号（与运动中的帧明显不同，便于检查末状态没有被重置状态替代）。
    历史：10 帧、从旧到新；新一轮用重置帧填满并令 history_length=1。真实 E05 的填充规则尚未确定，这里只是假环境的约定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

HISTORY_LENGTH, FRAME_DIM, ACTION_DIM = 10, 33, 8


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


class FakeLocomotionEnv:
    """E 个并行的人工环境。

    参数：
        num_envs: 环境数 E；q_dim: Q 观测维数 C（≥ 3）；max_episode_steps: 到时步数（纯截断）。
        termination_steps: 每个环境在本轮第几次动作后真终止（None 表示不终止）；与 max_episode_steps 相同时按真终止。
    received_actions 保存每次 step 实际收到的动作副本，供测试核对“记录中的动作 = 执行的动作”。
    """

    def __init__(self, num_envs: int, q_dim: int, max_episode_steps: int, termination_steps: Sequence[int | None]) -> None:
        if q_dim < 3 or len(termination_steps) != num_envs:
            raise ValueError("q_dim 必须 ≥ 3，termination_steps 长度必须等于 num_envs")
        self.num_envs, self.q_dim, self.max_episode_steps = num_envs, q_dim, max_episode_steps
        self.termination_steps = torch.tensor([s if s is not None else 0 for s in termination_steps], dtype=torch.int64)
        self.terminates = torch.tensor([s is not None for s in termination_steps])
        self.mixing = torch.linspace(-1.0, 1.0, FRAME_DIM * ACTION_DIM).reshape(FRAME_DIM, ACTION_DIM)
        self.env_id = torch.arange(num_envs, dtype=torch.int64)
        self.received_actions: list[torch.Tensor] = []
        self._history = torch.zeros(num_envs, HISTORY_LENGTH, FRAME_DIM)
        self._length = torch.ones(num_envs, dtype=torch.int64)
        self._episode = torch.zeros(num_envs, dtype=torch.int64)
        self._step = torch.zeros(num_envs, dtype=torch.int64)

    def _reset_frame(self) -> torch.Tensor:
        return (-3.0 + 0.01 * self.env_id.float()).reshape(-1, 1).expand(self.num_envs, FRAME_DIM)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        """把 mask 中的环境重置为新一轮：重置帧填满历史，长度 1，step 0。（原地修改内部缓冲。）"""
        frames = self._reset_frame()[mask]
        self._history[mask] = frames.unsqueeze(1).expand(-1, HISTORY_LENGTH, FRAME_DIM)
        self._length[mask] = 1
        self._step[mask] = 0

    def _q(self, history: torch.Tensor) -> torch.Tensor:
        frame = history[:, -1]
        return torch.cat([2.0 * frame[:, :3], frame[:, 3:self.q_dim]], dim=1)

    def _observation(self) -> EnvObservation:
        return EnvObservation(self._history.clone(), self._q(self._history), self._length.clone(), self.env_id.clone(),
                              self._episode.clone(), self._step.clone())

    def reset(self, first_episode_ids: torch.Tensor | None = None) -> EnvObservation:
        """重置全部环境；first_episode_ids 用于恢复训练后继续编号（不恢复中断前的仿真现场）。"""
        self._episode = torch.zeros(self.num_envs, dtype=torch.int64) if first_episode_ids is None else first_episode_ids.clone()
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool))
        return self._observation()

    def step(self, action: torch.Tensor) -> EnvStep:
        """执行动作 [E,8]（策略动作坐标），返回 EnvStep。"""
        if tuple(action.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(f"动作形状应为 [{self.num_envs},{ACTION_DIM}]，实际为 {list(action.shape)}")
        action = action.detach().to(torch.float32)
        self.received_actions.append(action.clone())
        frame = 0.8 * self._history[:, -1] + 0.2 * action @ self.mixing.T
        self._history = torch.cat([self._history[:, 1:], frame.unsqueeze(1)], dim=1)
        self._length = torch.clamp(self._length + 1, max=HISTORY_LENGTH)
        self._step += 1
        terminated = self.terminates & (self._step == self.termination_steps)
        time_out = (self._step == self.max_episode_steps) & ~terminated  # 真终止优先
        # 先复制重置前的真实下一观测，之后的重置会原地覆盖内部缓冲。
        next_history, next_length = self._history.clone(), self._length.clone()
        result = dict(next_policy_history=next_history, next_q_observation=self._q(next_history), next_history_length=next_length,
                      reward=(1.0 - 0.1 * action.pow(2).mean(dim=1, keepdim=True)),
                      terminated=terminated.unsqueeze(1), time_out=time_out.unsqueeze(1))
        ended = terminated | time_out
        self._episode[ended] += 1
        self._reset_envs(ended)
        return EnvStep(**result, observation=self._observation())
