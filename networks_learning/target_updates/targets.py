"""N08 目标 Q 的慢更新：让目标双 Q 的参数缓慢跟随在线双 Q。

公式（逐个参数，Polyak 平均）：target ← (1 − tau) × target + tau × online。平均的是参数，不是两次评分。
tau 与更新频率共同决定跟随速度：每 update_every 次在线 Q 更新执行一次 soft_update。
作者 rsl_rl_sac 的默认值为 tau = 0.005、每次 Critic 更新都执行（update_every = 1），这里沿用作为起点。

覆盖范围：按名称逐一匹配在线与目标的全部参数（含将来加入的地形编码层等子模块），结构不一致时报错；
目标副本由 N04 initialize_targets 完整复制得到，参数 requires_grad=False，不在任何优化器中。

非参数缓冲状态的规则：
    - 当前 N04 双 Q 不含缓冲（无 BatchNorm），观测归一化由外部 D04 统计负责，在线与目标 Q 共用同一份统计、
      同一个 normalize_q，因此无需在目标网络中另存统计。
    - 若之后在 Q 中加入带缓冲的层，soft_update 对缓冲执行原样复制（不做平均），这是本项目的明确约定：
      浮点运行统计（如 running_mean、running_var）也可以另定平均规则，但整数计数（num_batches_tracked）不能插值，
      统一复制最简单且与在线网络一致。check_target_state 会列出这些缓冲，提示需要确认该规则。
"""

from __future__ import annotations

import torch
from torch import nn

TAU = 0.005  # 作者默认
UPDATE_EVERY = 1  # 作者：每次 Critic 更新后都慢更新一次


class TargetUpdateError(ValueError):
    """在线与目标网络结构不一致、共享存储或状态不符合约定。"""


def validate_target_structure(online: nn.Module, target: nn.Module) -> None:
    """核对在线 / 目标的参数与缓冲按名称、形状、dtype 一一对应，且没有任何一项共享存储。"""
    for kind, a, b in (("参数", dict(online.named_parameters()), dict(target.named_parameters())),
                       ("缓冲", dict(online.named_buffers()), dict(target.named_buffers()))):
        if a.keys() != b.keys():
            raise TargetUpdateError(f"{kind}名称不一致：只在在线中 {sorted(a.keys() - b.keys())}，只在目标中 {sorted(b.keys() - a.keys())}")
        for name in a:
            if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
                raise TargetUpdateError(f"{kind} {name} 形状或类型不一致")
            if a[name].untyped_storage().data_ptr() == b[name].untyped_storage().data_ptr():
                raise TargetUpdateError(f"{kind} {name} 与在线网络共享存储，慢更新会同时改动在线值")


def soft_update(online: nn.Module, target: nn.Module, tau: float = TAU) -> None:
    """target ← (1 − tau) × target + tau × online（参数）；缓冲原样复制。不修改在线网络。

    tau=0 时目标参数不变，tau=1 时完整复制在线参数；两种情况下缓冲都会被复制。
    """
    if not 0.0 <= tau <= 1.0:
        raise TargetUpdateError(f"tau={tau} 必须在 [0, 1]")
    validate_target_structure(online, target)
    online_params = dict(online.named_parameters())
    online_buffers = dict(online.named_buffers())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.lerp_(online_params[name], tau)
        for name, buffer in target.named_buffers():
            buffer.copy_(online_buffers[name])


def soft_update_due(critic_update_count: int, update_every: int = UPDATE_EVERY) -> bool:
    """第 critic_update_count 次（从 1 计）在线 Q 更新之后是否执行慢更新。"""
    if update_every < 1:
        raise TargetUpdateError(f"update_every={update_every} 必须 ≥ 1")
    return critic_update_count % update_every == 0


def check_target_state(online: nn.Module, target: nn.Module, optimizers: tuple[torch.optim.Optimizer, ...] = ()) -> dict:
    """检查目标网络：结构对应、无共享存储、参数不求梯度、不出现在给定优化器中；返回需要单独处理的缓冲名称。

    异常：TargetUpdateError：任一条件不满足。
    """
    validate_target_structure(online, target)
    trainable = [name for name, p in target.named_parameters() if p.requires_grad]
    if trainable:
        raise TargetUpdateError(f"目标参数 {trainable} 仍会求梯度")
    target_ids = {id(p) for p in target.parameters()}
    for optimizer in optimizers:
        if any(id(p) in target_ids for group in optimizer.param_groups for p in group["params"]):
            raise TargetUpdateError("目标参数出现在优化器中，会被梯度更新")
    return {"parameters": len(target_ids), "buffers_copied_verbatim": [name for name, _ in target.named_buffers()]}
