"""D04 观测归一化与统计管理：只用新采集的数据更新统计，回放、评分与运行时只使用统计。

变换（与原 PPO / 作者 SAC 的 EmpiricalNormalization 相同）：
    y = (x - mean) / (sqrt(var) + eps)，eps = 1e-2
eps 加在标准差上：常数通道 var=0 时 y = (x - mean)/eps，不会出现除零或 NaN；
统计以 float64 保存与合并，避免计数很大时 float32 累计误差，输出转换回输入的 dtype。

统计对象与字段（两套统计互相独立）：
    - Actor / 历史编码器：policy 统计，D=33，逐帧统计。[B,10,33] 的每一帧使用同一组 33 维统计，
      与原 PPO 对当前帧和历史帧共用 obs_normalizer 一致。
    - Q：q 统计，D=C（Q 观测全部字段）。online Q 评价 q_observation、target Q 评价尾部的
      next_q_observation，二者必须调用同一个 normalize_q 和同一份当前统计，保证输入变换一致。
    - 速度监督标签（N02）直接读取未归一化的原始 Q 观测字段（E02 约定的 2.0 缩放），本模块不处理。

哪些数据计入统计（每个真实物理状态只计一次）：
    - 每条新采集记录的真实下一观测最新一帧 next_policy_history[:, -1] 与 next_q_observation：
      覆盖每次动作后到达的状态，包括真终止 / 纯截断时重置前的末状态。
    - 一轮的第一帧（history_length == 1 的记录）的当前观测 policy_history[:, -1] 与 q_observation：
      它不是任何记录的“下一观测”，否则会漏计。重置后观测正是下一轮第一条记录的当前观测，在那里计一次，不另计。
    - 不计入：历史中较旧的帧（它们在之前的步骤已作为最新帧计入，重叠历史不重复计数）；
      重置填充帧；任何从经验池回放出来的记录（回放只调用 normalize）。
    调用方（采集循环）对每批新采集记录只调用一次 update_from_fresh_data。

与原 PPO / 作者 SAC 的关系：原 PPO 在 training 模式的 forward 中自动更新统计，SAC 反复回放旧数据时会重复计数，
因此这里把更新与使用拆开。作者 SAC 在 process_env_step 中用 true_next_obs 更新，其中真终止环境的
true_next_obs 是重置后观测、末状态不计入；这里末状态与重置后状态各计一次。

不负责：奖励归一化、尚未要求的其他变换；不修改 D01 / D02 中保存的原始经历。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transition_schema"))

from schema import POLICY_OBS_DIM, TransitionBatch  # noqa: E402

NORMALIZER_EPS = 1e-2  # 与原 PPO normalizer.py、作者 rsl_rl_sac EmpiricalNormalization 的默认值一致
STAT_DTYPE = torch.float64


class NormalizationError(ValueError):
    """统计更新、归一化或恢复的输入不符合约定。"""


class RunningNormalizer:
    """单个输入对象（维数 D）的均值、方差与计数。

    初始 mean=0、var=1、count=0（与原 PPO 相同），未更新时 normalize 近似为 x/(1+eps)。
    """

    def __init__(self, dim: int, eps: float = NORMALIZER_EPS, device: str | torch.device = "cpu") -> None:
        if dim <= 0 or not eps > 0.0:
            raise NormalizationError(f"dim={dim} 必须为正，eps={eps} 必须为正")
        self.dim = dim
        self.eps = float(eps)
        self.mean = torch.zeros(dim, dtype=STAT_DTYPE, device=device)
        self.var = torch.ones(dim, dtype=STAT_DTYPE, device=device)
        self.count = 0

    def update_from_fresh_data(self, x: torch.Tensor) -> None:
        """把一批新数据 [N, D] 合并进统计（总体方差，按批合并，等价于逐条更新）。

        合并公式（Chan 等的并行方差合并）：n = n_a + n_b，delta = mean_b - mean_a，
            mean = mean_a + delta * n_b / n
            var  = (var_a * n_a + var_b * n_b + delta^2 * n_a * n_b / n) / n
        n_a = 0 时结果就是这批数据本身的均值与方差，初始的 var=1 不参与。
        副作用：修改 mean、var、count。N=0 时不变。
        异常：NormalizationError：形状不符或含 NaN/inf（坏数据一旦进入统计就无法剔除）。
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise NormalizationError(f"更新数据形状应为 [N, {self.dim}]，实际为 {list(x.shape)}")
        if x.shape[0] == 0:
            return
        if not bool(torch.isfinite(x).all()):
            raise NormalizationError("更新数据含 NaN 或 inf")
        x = x.to(device=self.mean.device, dtype=STAT_DTYPE)
        n_b = x.shape[0]
        mean_b = x.mean(dim=0)
        var_b = x.var(dim=0, unbiased=False)
        n = self.count + n_b
        delta = mean_b - self.mean
        if self.count == 0:
            self.mean, self.var = mean_b, var_b
        else:
            self.mean = self.mean + delta * (n_b / n)
            self.var = (self.var * self.count + var_b * n_b + delta.square() * (self.count * n_b / n)) / n
        self.count = n

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (x - mean)/(sqrt(var) + eps)，最后一维为 D，其余维度任意。

        只读统计，不修改统计或 x；返回新张量，dtype 与 x 相同。
        """
        if x.shape[-1] != self.dim:
            raise NormalizationError(f"最后一维应为 {self.dim}，实际为 {list(x.shape)}")
        mean = self.mean.to(x.device)
        std = self.var.to(x.device).sqrt()
        return ((x.to(STAT_DTYPE) - mean) / (std + self.eps)).to(x.dtype)

    def export_statistics(self) -> dict:
        """导出完整统计状态（独立副本）：mean、var、count、eps、dim。"""
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count, "eps": self.eps, "dim": self.dim}

    def load_statistics(self, state: dict) -> None:
        """从 export_statistics 的结果恢复，包括 eps 与 count；复制输入，不共享存储。

        异常：NormalizationError：维数不符、count 为负或 var 含负数 / 非有限值。
        """
        if int(state["dim"]) != self.dim or tuple(state["mean"].shape) != (self.dim,) or tuple(state["var"].shape) != (self.dim,):
            raise NormalizationError(f"统计维数 {state['dim']} 与当前 {self.dim} 不符")
        var = state["var"].to(device=self.mean.device, dtype=STAT_DTYPE)
        if int(state["count"]) < 0 or not bool(torch.isfinite(var).all()) or bool((var < 0).any()):
            raise NormalizationError("统计状态无效：count 为负或 var 含负数 / 非有限值")
        self.mean = state["mean"].to(device=self.mean.device, dtype=STAT_DTYPE).clone()
        self.var = var.clone()
        self.count = int(state["count"])
        self.eps = float(state["eps"])


def select_fresh_samples(batch: TransitionBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """从一批新采集记录中选出计入统计的样本：(policy_frames [N_p, 33], q_rows [N_q, C])。

    规则见文件头：每条记录的真实下一观测各计一次；一轮第一条记录（history_length == 1）另计其当前观测。
    只读，不修改 batch。
    """
    first = batch.history_length == 1
    policy = torch.cat([batch.policy_history[first, -1], batch.next_policy_history[:, -1]], dim=0)
    q = torch.cat([batch.q_observation[first], batch.next_q_observation], dim=0)
    return policy, q


class ObservationNormalizers:
    """Actor（逐帧 33 维）与 Q（C 维）的两套统计及统一的使用入口。"""

    def __init__(self, q_dim: int, eps: float = NORMALIZER_EPS, device: str | torch.device = "cpu") -> None:
        self.policy = RunningNormalizer(POLICY_OBS_DIM, eps, device)
        self.q = RunningNormalizer(q_dim, eps, device)

    def update_from_fresh_data(self, batch: TransitionBatch) -> None:
        """用一批新采集记录更新两套统计；每批新数据只调用一次，回放数据不得调用。"""
        policy, q = select_fresh_samples(batch)
        self.policy.update_from_fresh_data(policy)
        self.q.update_from_fresh_data(q)

    def normalize_policy(self, history: torch.Tensor) -> torch.Tensor:
        """归一化策略观测，最后一维为 33（如 [B,10,33] 历史或 [B,33] 当前帧）；逐帧使用同一统计。"""
        return self.policy.normalize(history)

    def normalize_q(self, q_observation: torch.Tensor) -> torch.Tensor:
        """归一化 Q 观测 [B, C]；online Q 的当前输入与 target Q 的尾部输入都使用本函数。"""
        return self.q.normalize(q_observation)

    def export_statistics(self) -> dict:
        """导出两套完整统计：{"policy": ..., "q": ...}，供检查点（T03）保存。"""
        return {"policy": self.policy.export_statistics(), "q": self.q.export_statistics()}

    def load_statistics(self, state: dict) -> None:
        """恢复两套统计。"""
        self.policy.load_statistics(state["policy"])
        self.q.load_statistics(state["q"])
