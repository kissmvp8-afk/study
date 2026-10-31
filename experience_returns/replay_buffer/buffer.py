"""D02 经验池写入与抽样：按环境分行的循环缓冲区，保存 D01 的原始经历并读取同一轮内的连续片段。

职责：
    - ReplayBuffer.append：写入一批 D01 TransitionBatch（每个环境一条），复制进预分配存储并记录写入序号。
    - ReplayBuffer.valid_starts / sample_starts：列出或随机抽取 n 步片段的有效起点。
    - ReplayBuffer.read_sequence：读取从起点开始的 L 条记录，并标出哪些属于起点同一轮的连续记录。
    - estimate_storage_bytes：按 D01 实际字段形状与 dtype 计算存储字节数，用于选择容量。

不负责：不计算多步回报（D03）、不归一化（D04）、不切历史窗口（D05）、不编码历史；
第一版保存编码前的完整历史 [10,33]，由当前编码器在训练时重新解释旧经历。

存储布局：每个字段一块张量 [num_envs, capacity_per_env, *字段形状]，第 0 维即 env_id。
容量单位是“每个环境保存的记录条数”，总容量 = num_envs × capacity_per_env 条。
每个环境有独立的写指针；第 w 次写入（从 0 计）落在槽位 w % capacity_per_env，同时在
_write_seq 中记下 w。覆盖后槽位的写入序号会变，因此“槽位 s 的后继是槽位 s+1”只在
_write_seq[s+1] == _write_seq[s] + 1 时成立——这就是区分旧内容与新内容的覆盖代次。

连续性：只看显式编号，不看数组相邻。两条记录相邻当且仅当同一环境、后者是前者之后的下一次写入、
episode_id 相同且 step_index 恰好加一。真终止或纯截断之后的下一条属于新一轮，不与前一轮连接。

复制责任：append 通过索引赋值把来源张量复制进经验池，之后修改来源不影响池内记录；
read_sequence 使用高级索引读取，返回新张量，调用方修改读出结果也不影响池内记录。

与作者 SAC 的关系（rsl_rl_sac e0d243a，BSD-3-Clause）：作者同样按 [num_envs, 每环境容量] 存储，
但用全局写指针 step 判断片段是否跨越覆盖位置，并假定所有环境每步同时写入、数组相邻即时间相邻。
这里改为逐环境写指针 + 写入序号 + episode_id / step_index 检查，满足本项目“不依据数组相邻”的约定。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transition_schema"))

# _expected_layout 是 D01 内部用于校验的字段布局表（形状与 dtype）。这里直接复用它分配存储，
# 保证经验池与记录契约只有一个布局来源；D01 未公开此函数，改名公开属于 D01 的接口变更，本次不做。
from schema import FIELD_NAMES, TransitionBatch, _expected_layout, validate_transition  # noqa: E402

WRITE_SEQ_DTYPE = torch.int64  # 每个槽位的写入序号，-1 表示从未写入


class ReplayBufferError(ValueError):
    """经验池写入或读取不符合约定。"""


def bytes_per_record(q_dim: int) -> int:
    """一条记录在经验池中占用的字节数：D01 全部字段 + 1 个写入序号。

    q_dim=54（平地预期）时为 3158 字节，q_dim=241（复杂地形预期）时为 4654 字节；
    主要来自两段 [10,33] float32 历史（共 2640 字节）。
    """
    total = torch.empty((), dtype=WRITE_SEQ_DTYPE).element_size()
    for shape, dtype in _expected_layout(q_dim).values():
        total += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return total


def estimate_storage_bytes(num_envs: int, capacity_per_env: int, q_dim: int) -> int:
    """在分配之前估算经验池的张量存储字节数（不含每环境写指针等少量簿记）。

    例：4096 个环境 × 每环境 1221 条 ≈ 500 万条，q_dim=241 时约 23.3 GB；
    因此本模块不提供默认容量，由调用方按显存或内存预算显式选择。
    """
    return num_envs * capacity_per_env * bytes_per_record(q_dim)


@dataclass(frozen=True)
class SequenceBatch:
    """read_sequence 的结果：M 个起点各读 L 条记录。

    fields: D01 字段名 → [M, L, *字段形状] 的新张量（与经验池不共享存储）。
    valid: [M, L] bool。valid[m, j] 表示第 j 条与起点属于同一环境同一轮，且 0..j 之间连续未断；
        一旦断开（尚未写入、已被覆盖、遇到新一轮），之后全部为 False。
    env_ids: [M] int64；slots: [M, L] int64，读取的槽位，便于排查。
    """

    fields: dict[str, torch.Tensor]
    valid: torch.Tensor
    env_ids: torch.Tensor
    slots: torch.Tensor

    @property
    def length(self) -> int:
        """每个片段读取的记录数 L。"""
        return int(self.valid.shape[1])

    def at(self, offsets: torch.Tensor) -> TransitionBatch:
        """按行取第 offsets[m] 条记录，组成 [M, ...] 的 TransitionBatch。

        要求所取记录全部有效；无效位置的内容可能是空槽或其他轮次，不能当作经历使用。
        异常：ReplayBufferError：offsets 越界或指向无效记录。
        """
        rows = torch.arange(self.valid.shape[0], device=self.valid.device)
        offsets = offsets.to(device=self.valid.device, dtype=torch.int64)
        if bool(((offsets < 0) | (offsets >= self.length)).any()):
            raise ReplayBufferError(f"offsets 超出片段长度 {self.length}")
        if not bool(self.valid[rows, offsets].all()):
            raise ReplayBufferError(f"行 {torch.nonzero(~self.valid[rows, offsets]).reshape(-1).tolist()} 指向无效记录")
        return TransitionBatch(**{name: value[rows, offsets] for name, value in self.fields.items()})


class ReplayBuffer:
    """按环境分行的循环经验池，保存 D01 TransitionBatch 的全部字段。

    参数：
        num_envs: 并行环境数 E；写入记录的 env_id 必须在 [0, E)。
        capacity_per_env: 每个环境保存的记录条数（容量单位），必须为正；无默认值。
        q_dim: Q 观测维数 C，与 D01 make_transition 使用的值一致。
        device: 存储设备；写入记录必须位于同一设备。
    副作用：构造时一次性分配 estimate_storage_bytes(...) 字节的存储。
    """

    def __init__(self, num_envs: int, capacity_per_env: int, q_dim: int, device: str | torch.device = "cpu") -> None:
        if num_envs <= 0 or capacity_per_env <= 0:
            raise ReplayBufferError(f"num_envs={num_envs}、capacity_per_env={capacity_per_env} 必须为正")
        self.num_envs = num_envs
        self.capacity_per_env = capacity_per_env
        self.q_dim = q_dim
        self.device = torch.device(device)
        self._storage = {
            name: torch.zeros((num_envs, capacity_per_env, *shape), dtype=dtype, device=self.device)
            for name, (shape, dtype) in _expected_layout(q_dim).items()
        }
        self._write_seq = torch.full((num_envs, capacity_per_env), -1, dtype=WRITE_SEQ_DTYPE, device=self.device)
        self._writes = torch.zeros(num_envs, dtype=WRITE_SEQ_DTYPE, device=self.device)  # 每个环境已写入次数

    @property
    def size(self) -> int:
        """当前保存的记录条数（不超过 num_envs × capacity_per_env）。"""
        return int((self._write_seq >= 0).sum())

    @property
    def storage_bytes(self) -> int:
        """实际分配的张量存储字节数（字段存储 + 写入序号 + 写指针）。"""
        tensors = [*self._storage.values(), self._write_seq, self._writes]
        return sum(t.numel() * t.element_size() for t in tensors)

    def append(self, batch: TransitionBatch) -> torch.Tensor:
        """写入一批记录，每条写到其 env_id 那一行的下一个槽位，返回写入的槽位 [B]。

        一批内同一环境最多一条：同一时刻每个环境只产生一条经历，按批写入时也就不存在批内先后的歧义。
        满了之后覆盖该环境最旧的记录。只复制，不改变来源张量。
        异常：ReplayBufferError：设备不符、env_id 越界或重复；TransitionSchemaError：记录不符合 D01 契约。
        """
        validate_transition(batch, self.q_dim)
        if batch.env_id.device != self.device:
            raise ReplayBufferError(f"记录位于 {batch.env_id.device}，经验池位于 {self.device}")
        env = batch.env_id
        if bool(((env < 0) | (env >= self.num_envs)).any()):
            raise ReplayBufferError(f"env_id 必须在 [0, {self.num_envs})，实际为 {env.tolist()}")
        if env.unique().numel() != env.numel():
            raise ReplayBufferError(f"一批内 env_id 不能重复：{env.tolist()}")
        seq = self._writes[env]
        slot = seq % self.capacity_per_env
        for name in FIELD_NAMES:
            self._storage[name][env, slot] = getattr(batch, name)
        self._write_seq[env, slot] = seq
        self._writes[env] += 1
        return slot

    def _continuity(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算每个槽位的 (present, linked, boundary)，形状均为 [E, capacity]。

        present：槽位有记录。
        linked：下一槽位存放的正是该环境的下一次写入，且同一轮、step_index 加一。
        boundary：该记录是真终止或纯截断，后续不再属于同一轮。
        """
        present = self._write_seq >= 0
        nxt = (torch.arange(self.capacity_per_env, device=self.device) + 1) % self.capacity_per_env
        episode = self._storage["episode_id"]
        step = self._storage["step_index"]
        linked = (
            present
            & (self._write_seq[:, nxt] == self._write_seq + 1)
            & (episode[:, nxt] == episode)
            & (step[:, nxt] == step + 1)
        )
        boundary = present & (self._storage["terminated"] | self._storage["time_out"]).squeeze(-1)
        return present, linked, boundary

    def valid_starts(self, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """列出 n 步片段的全部有效起点，返回 (env_ids [K], slots [K])。

        有效起点：从起点开始，在同一轮内连续存在记录，直到出现结束边界（真终止或纯截断），或者凑满 n 条。
        若既没到 n 条、又没遇到边界就断了（后继尚未写入或已被覆盖），起点暂不可用：
        普通数据不足不能当作任务结束。n=1 时任何已写入的记录都是有效起点。
        D03 build_n_step_batch 使用同一规则检查读出的片段。
        异常：ReplayBufferError：n_steps 不在 1..capacity_per_env。
        """
        if not 1 <= n_steps <= self.capacity_per_env:
            raise ReplayBufferError(f"n_steps={n_steps} 必须在 1..capacity_per_env={self.capacity_per_env}")
        present, linked, boundary = self._continuity()
        ok = present.clone()
        open_ = present & ~boundary  # 尚未遇到边界、还需要更多后继的起点
        for j in range(n_steps - 1):
            # 第 j 条到第 j+1 条的连接，沿槽位维循环移位后与起点对齐。
            link_j = torch.roll(linked, shifts=-j, dims=1)
            ok &= ~open_ | link_j
            open_ &= link_j & ~torch.roll(boundary, shifts=-(j + 1), dims=1)
        env_ids, slots = torch.nonzero(ok, as_tuple=True)
        return env_ids, slots

    def sample_starts(
        self, batch_size: int, n_steps: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从有效起点中有放回地均匀抽取 batch_size 个，返回 (env_ids, slots)，各 [batch_size]。

        generator 为 CPU 上的 torch.Generator；相同种子与相同经验池内容会得到相同结果。
        异常：ReplayBufferError：没有有效起点或 batch_size 非正。
        """
        if batch_size <= 0:
            raise ReplayBufferError(f"batch_size={batch_size} 必须为正")
        env_ids, slots = self.valid_starts(n_steps)
        if env_ids.numel() == 0:
            raise ReplayBufferError(f"没有满足 n_steps={n_steps} 的有效起点")
        pick = torch.randint(env_ids.numel(), (batch_size,), generator=generator).to(self.device)
        return env_ids[pick], slots[pick]

    def read_sequence(self, env_ids: torch.Tensor, slots: torch.Tensor, length: int) -> SequenceBatch:
        """从每个起点读取 length 条记录（按写入顺序），并给出连续有效标记。

        参数：env_ids、slots 为 [M] int64；length 为 L ≥ 1 且不超过 capacity_per_env。
        返回：SequenceBatch，字段为新张量；valid[:, 0] 表示起点本身有记录。
        异常：ReplayBufferError：参数越界。
        """
        if not 1 <= length <= self.capacity_per_env:
            raise ReplayBufferError(f"length={length} 必须在 1..capacity_per_env={self.capacity_per_env}")
        env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        slots = slots.to(device=self.device, dtype=torch.int64)
        if bool(((env_ids < 0) | (env_ids >= self.num_envs) | (slots < 0) | (slots >= self.capacity_per_env)).any()):
            raise ReplayBufferError("env_ids 或 slots 越界")
        offsets = torch.arange(length, device=self.device)
        seq_slots = (slots.unsqueeze(1) + offsets) % self.capacity_per_env  # [M, L]
        rows = env_ids.unsqueeze(1).expand_as(seq_slots)
        present, linked, _ = self._continuity()
        valid = torch.empty_like(seq_slots, dtype=torch.bool)
        valid[:, 0] = present[env_ids, slots]
        for j in range(1, length):
            valid[:, j] = valid[:, j - 1] & linked[rows[:, j - 1], seq_slots[:, j - 1]]
        fields = {name: store[rows, seq_slots] for name, store in self._storage.items()}
        return SequenceBatch(fields=fields, valid=valid, env_ids=env_ids, slots=seq_slots)
