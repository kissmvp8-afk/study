"""N04 双 Q 评分网络：两套独立参数，评价“Q 观测 + 策略坐标动作”的价值，并提供完整初始化的目标副本。

输入：
    q_observation [B,C]：D04 normalize_q 之后的 Q 观测（训练侧信息，可含 Actor 看不到的字段，如身体线速度）。
        C 由调用方按 E02 运行时观测组给出（平地预期 54、复杂地形预期 241），网络构造时固定，前向时校验。
    action [B,8]：策略动作坐标（与 E03 / D01 / N03 一致），不换成关节角或轮速。
拼接顺序：[q_observation | action]，经 512 → 256 → 128（ELU）→ 1，隐藏层沿用原 PPO critic_hidden_dims。

独立性：Q1、Q2 各自构造、各自随机初始化，不共享任何层；也不与 N01 编码器或 N03 Actor 共享可训练层。
目标副本：initialize_targets 用 deepcopy 完整复制在线双 Q（全部参数与缓冲），设为不求梯度、eval 模式；
之后只由 N08 慢更新修改。当前网络不含 BatchNorm 等缓冲，归一化由外部 D04 统计负责，在线与目标共用同一份统计。

第一版不加地形编码层：复杂地形的 187 维高度扫描直接作为 Q 观测的一部分输入 MLP。原 PPO 对高度扫描先过 128/64/32
的 scan_encoder；若之后加入，它必须是 QNetwork 的子模块，从而自动包含在目标副本、慢更新与保存恢复中。
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from torch import nn

ACTION_DIM = 8
CRITIC_HIDDEN_DIMS = (512, 256, 128)  # 原 PPO critic_hidden_dims


class QNetwork(nn.Module):
    """单个 Q：[q_observation | action] → 标量评分 [B,1]。"""

    def __init__(self, q_dim: int, action_dim: int = ACTION_DIM, hidden_dims: Sequence[int] = CRITIC_HIDDEN_DIMS) -> None:
        super().__init__()
        if q_dim <= 0:
            raise ValueError(f"q_dim={q_dim} 必须为正，由 E02 运行时观测组给出")
        self.q_dim, self.action_dim = q_dim, action_dim
        layers: list[nn.Module] = []
        previous = q_dim + action_dim
        for width in hidden_dims:
            layers += [nn.Linear(previous, width), nn.ELU()]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q_observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if q_observation.shape[-1] != self.q_dim or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Q 输入应为 [B,{self.q_dim}] 与 [B,{self.action_dim}]，实际为 {list(q_observation.shape)}、{list(action.shape)}"
            )
        return self.net(torch.cat([q_observation, action], dim=-1))


class TwinQ(nn.Module):
    """两个独立的 QNetwork（q1、q2）。"""

    def __init__(self, q_dim: int, action_dim: int = ACTION_DIM, hidden_dims: Sequence[int] = CRITIC_HIDDEN_DIMS) -> None:
        super().__init__()
        self.q1 = QNetwork(q_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(q_dim, action_dim, hidden_dims)


def evaluate_q_pair(critics: TwinQ, q_observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """用同一批状态和动作分别评价，返回 (q1 [B,1], q2 [B,1])。梯度是否保留由调用方的上下文决定。"""
    return critics.q1(q_observation, action), critics.q2(q_observation, action)


def initialize_targets(critics: TwinQ) -> TwinQ:
    """返回在线双 Q 的完整副本：数值相同、存储独立、参数 requires_grad=False、eval 模式。"""
    target = copy.deepcopy(critics)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target.eval()


def list_critic_parameters(critics: TwinQ) -> list[tuple[str, nn.Parameter]]:
    """在线双 Q 的全部可训练参数（名称, 参数），即 Critic 优化器应包含的全部参数。"""
    return [(name, p) for name, p in critics.named_parameters() if p.requires_grad]
