"""D05 历史窗口与有效性：把环境维护的 [B,10,33] 历史切成当前帧与两段相邻的 5 帧窗口。

输入约定（根 README / E02）：历史含当前帧，按从旧到新排列，索引 9 是时刻 t，索引 0 是 t-9。
输出：
    current  = 历史[:, 9]      时刻 t，            [B,33]
    recent   = 历史[:, 5:10]   时刻 t-4..t，       [B,5,33]  Actor / 编码器的决策窗口
    previous = 历史[:, 4:9]    时刻 t-5..t-1，     [B,5,33]  辅助学习（Barlow Twins）的另一窗口
两个窗口共用 t-4..t-1 四个时刻，合起来覆盖 t-5..t 六个时刻。

职责：
    - split_windows：只切片，不推进历史、不追加当前帧（历史由环境推进，再追加会让当前帧出现两次）。
      训练、评分与未来的导出 / 运行入口都调用这一个函数，保证读到同样的时间顺序。
    - valid_history_mask：按有效历史长度判断最近若干帧是否都是本轮真实帧；辅助学习要求 6 帧。
      不根据观测是否为零判断：真实观测可以为零，重置填充也未必是零。
    - validate_time_order：用 env_id / episode_id / step_index 检查一组记录的时间顺序，不从观测数值猜测时间。

不负责：归一化（D04，逐帧变换，与切片先后无关）、历史编码（N01）、辅助损失（N02）、模型导出（T05）。

与原 PPO 的关系：原 actor_critic._split_policy_obs 取 current = obs[:, -1]、hist = obs[:, -6:-1]（t-5..t-1），
决策窗口由 hist[:, 1:] 拼上 current 得到（t-4..t）；导出包装器 _InferenceWrapper 用另一种切法得到同样结果。
这里直接给出三段，结果与原 PPO 一致（测试中逐元素比对）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transition_schema"))

from schema import HISTORY_LENGTH, POLICY_OBS_DIM  # noqa: E402

WINDOW_FRAMES = 5  # 编码器窗口帧数（原 PPO bt_window=5）
AUXILIARY_REQUIRED_FRAMES = WINDOW_FRAMES + 1  # 两个窗口合起来覆盖的时刻数 t-5..t


class HistoryWindowError(ValueError):
    """历史形状、有效长度或时间编号不符合约定。"""


@dataclass(frozen=True)
class HistoryWindows:
    """split_windows 的结果。三个字段都是输入历史的切片视图（不复制），调用方不得原地修改。"""

    current: torch.Tensor
    recent: torch.Tensor
    previous: torch.Tensor


def split_windows(history: torch.Tensor) -> HistoryWindows:
    """切出当前帧、t-4..t 与 t-5..t-1 三段。

    参数：history [B,10,33]，含当前帧、从旧到新；可以是原始观测或 D04 归一化后的观测。
    返回：HistoryWindows(current [B,33], recent [B,5,33], previous [B,5,33])，均为视图。
    异常：HistoryWindowError：形状不是 [B,10,33]。
    """
    if history.dim() != 3 or tuple(history.shape[1:]) != (HISTORY_LENGTH, POLICY_OBS_DIM):
        raise HistoryWindowError(f"历史形状应为 [B,{HISTORY_LENGTH},{POLICY_OBS_DIM}]，实际为 {list(history.shape)}")
    end = HISTORY_LENGTH  # 索引 end-1 为时刻 t
    return HistoryWindows(
        current=history[:, end - 1],
        recent=history[:, end - WINDOW_FRAMES : end],
        previous=history[:, end - WINDOW_FRAMES - 1 : end - 1],
    )


def valid_history_mask(history_length: torch.Tensor, required_frames: int = AUXILIARY_REQUIRED_FRAMES) -> torch.Tensor:
    """返回 [B] bool：最近 required_frames 帧是否都是本轮的真实帧（history_length ≥ required_frames）。

    history_length 来自 D01 记录（当前历史用 history_length，下一历史用 next_history_length），表示历史中属于
    本轮真实观测的帧数，其余为重置填充。默认 6 对应辅助学习的两个窗口；不足 6 帧时 previous 窗口
    会包含填充帧或上一轮的内容，不能用于辅助损失。
    异常：HistoryWindowError：history_length 不在 1..10，或 required_frames 不在 1..10。
    """
    if not 1 <= required_frames <= HISTORY_LENGTH:
        raise HistoryWindowError(f"required_frames={required_frames} 必须在 1..{HISTORY_LENGTH}")
    if bool(((history_length < 1) | (history_length > HISTORY_LENGTH)).any()):
        raise HistoryWindowError(f"history_length 必须在 1..{HISTORY_LENGTH}，实际为 {history_length.tolist()}")
    return history_length >= required_frames


def validate_time_order(env_id: torch.Tensor, episode_id: torch.Tensor, step_index: torch.Tensor) -> None:
    """检查 [M, L] 的记录编号：每行同一环境、同一轮，step_index 逐条加一（无重复、无跳跃）。

    用于确认一组按时间排列的记录（例如 D02 read_sequence 的有效前缀）确实首尾相接；
    只看显式编号，不从观测数值推断时间。L=1 时总是通过。
    异常：HistoryWindowError：形状不一致，或列出跨环境、跨轮次、重复 / 跳跃的行号。
    """
    if not (env_id.shape == episode_id.shape == step_index.shape) or env_id.dim() != 2:
        raise HistoryWindowError(
            f"编号形状应为相同的 [M, L]，实际为 {list(env_id.shape)}、{list(episode_id.shape)}、{list(step_index.shape)}"
        )
    problems = []
    for name, ids in (("跨环境", env_id), ("跨轮次", episode_id)):
        bad = (ids != ids[:, :1]).any(dim=1)
        if bool(bad.any()):
            problems.append(f"{name}：行 {torch.nonzero(bad).reshape(-1).tolist()}")
    bad = (step_index[:, 1:] != step_index[:, :-1] + 1).any(dim=1)
    if bool(bad.any()):
        problems.append(f"step_index 不是逐条加一（重复或跳跃）：行 {torch.nonzero(bad).reshape(-1).tolist()}")
    if problems:
        raise HistoryWindowError("时间顺序检查失败：" + "；".join(problems))
