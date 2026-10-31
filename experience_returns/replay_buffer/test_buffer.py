"""D02 经验池的验证程序（unittest + PyTorch CPU，无需仿真或 GPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/replay_buffer/test_buffer.py -v

数据说明：全部为人工数据。每条记录的观测用 code = env*10000 + episode*100 + step 编码
（当前观测填 code，真实下一观测填 code+0.5），奖励等于 code，便于从读出内容反推它属于哪条经历。
测试中约定每轮 step_index 从 0 开始、history_length = min(step+1, 10)，这是人工数据的约定，
不代表 E01/E05 的实际编号来源。
不覆盖：E05 采集循环写入、GPU 设备、大容量性能。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transition_schema"))

from buffer import ReplayBuffer, ReplayBufferError, bytes_per_record, estimate_storage_bytes  # noqa: E402
from schema import FIELD_NAMES, make_transition  # noqa: E402

Q_DIM = 6  # 人工数据用的小 Q 维数；实际 C 由 E02 给出


def code_of(env: int, episode: int, step: int) -> float:
    """人工编码：能从数值反推 (env, episode, step)。"""
    return float(env * 10000 + episode * 100 + step)


def make_records(
    entries: list[tuple[int, int, int]],
    terminated: set[int] = frozenset(),
    time_out: set[int] = frozenset(),
    rewards: dict[int, float] | None = None,
    q_dim: int = Q_DIM,
):
    """构造一批 D01 记录。entries[i] = (env_id, episode_id, step_index)；terminated / time_out 为行号集合。

    rewards 可按行覆盖默认奖励（默认奖励 = code）。供 D02～D05 的测试共用。
    """
    b = len(entries)
    codes = torch.tensor([code_of(*e) for e in entries])
    steps = torch.tensor([e[2] for e in entries], dtype=torch.int64)
    history_length = torch.clamp(steps + 1, max=10)
    reward = codes.clone().reshape(b, 1)
    for row, value in (rewards or {}).items():
        reward[row, 0] = value
    term = torch.zeros(b, 1, dtype=torch.bool)
    tout = torch.zeros(b, 1, dtype=torch.bool)
    for row in terminated:
        term[row] = True
    for row in time_out:
        tout[row] = True
    data = {
        "policy_history": codes.reshape(b, 1, 1).expand(b, 10, 33).clone(),
        "next_policy_history": (codes + 0.5).reshape(b, 1, 1).expand(b, 10, 33).clone(),
        "q_observation": codes.reshape(b, 1).expand(b, q_dim).clone(),
        "next_q_observation": (codes + 0.5).reshape(b, 1).expand(b, q_dim).clone(),
        "action": (codes.reshape(b, 1) / 1e5).expand(b, 8).clone(),
        "reward": reward,
        "terminated": term,
        "time_out": tout,
        "env_id": torch.tensor([e[0] for e in entries], dtype=torch.int64),
        "episode_id": torch.tensor([e[1] for e in entries], dtype=torch.int64),
        "step_index": steps,
        "history_length": history_length,
        "next_history_length": torch.clamp(history_length + 1, max=10),
    }
    return make_transition(data, q_dim)


def write_episode(buffer: ReplayBuffer, env: int, episode: int, steps: range, terminal_at: int | None = None,
                  time_out_at: int | None = None) -> None:
    """把同一环境同一轮的若干步逐条写入经验池（每次 append 一条，模拟逐步采集）。"""
    for step in steps:
        buffer.append(
            make_records(
                [(env, episode, step)],
                terminated={0} if step == terminal_at else set(),
                time_out={0} if step == time_out_at else set(),
            )
        )


def start_steps(buffer: ReplayBuffer, n: int) -> list[tuple[int, int, int]]:
    """把有效起点换算成 (env, episode, step) 列表，便于按编号断言。"""
    env_ids, slots = buffer.valid_starts(n)
    read = buffer.read_sequence(env_ids, slots, 1)
    return sorted(
        zip(read.fields["env_id"][:, 0].tolist(), read.fields["episode_id"][:, 0].tolist(), read.fields["step_index"][:, 0].tolist())
    )


class WriteReadTest(unittest.TestCase):
    """D01 → D02：写入、读回、复制责任。"""

    def test_round_trip_all_fields(self) -> None:
        buffer = ReplayBuffer(num_envs=3, capacity_per_env=4, q_dim=Q_DIM)
        batch = make_records([(2, 7, 3), (0, 1, 0)], terminated={1})
        slots = buffer.append(batch)
        read = buffer.read_sequence(batch.env_id, slots, 1)
        self.assertTrue(bool(read.valid.all()))
        back = read.at(torch.zeros(2, dtype=torch.int64))
        for name in FIELD_NAMES:
            self.assertTrue(torch.equal(getattr(back, name), getattr(batch, name)), name)
        self.assertEqual(buffer.size, 2)

    def test_copies_on_write_and_read(self) -> None:
        buffer = ReplayBuffer(num_envs=1, capacity_per_env=2, q_dim=Q_DIM)
        batch = make_records([(0, 0, 0)])
        slots = buffer.append(batch)
        batch.reward.add_(1000.0)  # 来源被原地修改
        batch.next_policy_history.fill_(-1.0)
        read = buffer.read_sequence(batch.env_id, slots, 1)
        self.assertEqual(read.fields["reward"][0, 0, 0].item(), 0.0)
        self.assertEqual(read.fields["next_policy_history"][0, 0, 0, 0].item(), 0.5)
        read.fields["reward"].fill_(99.0)  # 读出结果被修改
        again = buffer.read_sequence(batch.env_id, slots, 1)
        self.assertEqual(again.fields["reward"][0, 0, 0].item(), 0.0)


class OverwriteTest(unittest.TestCase):
    """小容量循环覆盖：只返回有效且连续的片段。"""

    def test_wrapped_ring_keeps_only_valid_starts(self) -> None:
        buffer = ReplayBuffer(num_envs=1, capacity_per_env=4, q_dim=Q_DIM)
        write_episode(buffer, env=0, episode=0, steps=range(10))  # 写 10 次，池中只剩 step 6..9
        self.assertEqual(buffer.size, 4)
        self.assertEqual(start_steps(buffer, 1), [(0, 0, s) for s in (6, 7, 8, 9)])
        # n=3：step 8、9 的后继尚未写入、又未结束，暂不可用；step 9 的“下一槽位”存放的是更旧的 step 6。
        self.assertEqual(start_steps(buffer, 3), [(0, 0, 6), (0, 0, 7)])
        self.assertEqual(start_steps(buffer, 4), [(0, 0, 6)])

    def test_sampled_sequences_are_consecutive_after_many_overwrites(self) -> None:
        buffer = ReplayBuffer(num_envs=2, capacity_per_env=5, q_dim=Q_DIM)
        for step in range(23):
            buffer.append(make_records([(0, step // 7, step % 7), (1, 0, step)], terminated={0} if step % 7 == 6 else set()))
        env_ids, slots = buffer.sample_starts(64, 3, generator=torch.Generator().manual_seed(0))
        read = buffer.read_sequence(env_ids, slots, 3)
        for m in range(64):
            k = int(read.valid[m].sum())
            ended = bool((read.fields["terminated"][m, :k] | read.fields["time_out"][m, :k]).any())
            self.assertTrue(k == 3 or ended, "有效起点必须凑满 3 条或在边界处结束")
            steps = read.fields["step_index"][m, :k]
            self.assertTrue(bool((steps[1:] == steps[:-1] + 1).all()))
            self.assertEqual(read.fields["episode_id"][m, :k].unique().numel(), 1)
            self.assertEqual(read.fields["env_id"][m, :k].unique().tolist(), [int(env_ids[m])])


class MultiEnvAndBoundaryTest(unittest.TestCase):
    """多环境交错写入、结束边界与随机状态。"""

    def test_interleaved_envs_only_join_same_env(self) -> None:
        buffer = ReplayBuffer(num_envs=3, capacity_per_env=6, q_dim=Q_DIM)
        buffer.append(make_records([(1, 0, 0), (0, 0, 0)]))
        buffer.append(make_records([(1, 0, 1)]))
        buffer.append(make_records([(2, 0, 0), (1, 0, 2), (0, 0, 1)]))
        buffer.append(make_records([(0, 0, 2)]))
        self.assertEqual(start_steps(buffer, 3), [(0, 0, 0), (1, 0, 0)])
        env_ids, slots = buffer.valid_starts(3)
        read = buffer.read_sequence(env_ids, slots, 3)
        for m in range(env_ids.numel()):
            expected = [code_of(int(env_ids[m]), 0, s) for s in range(3)]
            self.assertEqual(read.fields["reward"][m, :, 0].tolist(), expected)

    def test_episode_boundaries(self) -> None:
        buffer = ReplayBuffer(num_envs=2, capacity_per_env=8, q_dim=Q_DIM)
        write_episode(buffer, 0, 0, range(3), terminal_at=2)  # 第 0 轮在 step 2 真终止
        write_episode(buffer, 0, 1, range(2))  # 第 1 轮 step 0、1，尚未结束
        write_episode(buffer, 1, 4, range(2), time_out_at=1)  # 环境 1 第 4 轮在 step 1 纯截断
        # n=3：第 0 轮的起点都能在边界处结束；第 1 轮缺后继，不可用；环境 1 的两步都在截断前或截断处结束。
        self.assertEqual(start_steps(buffer, 3), [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 4, 0), (1, 4, 1)])
        env_ids, slots = buffer.valid_starts(3)
        read = buffer.read_sequence(env_ids, slots, 3)
        row = [i for i in range(env_ids.numel()) if int(read.fields["step_index"][i, 0]) == 1 and int(env_ids[i]) == 0][0]
        self.assertEqual(read.valid[row].tolist(), [True, True, False])  # step 2 之后是新一轮

    def test_reproducible_sampling(self) -> None:
        buffer = ReplayBuffer(num_envs=2, capacity_per_env=10, q_dim=Q_DIM)
        write_episode(buffer, 0, 0, range(10))
        write_episode(buffer, 1, 0, range(10))
        first = buffer.sample_starts(16, 2, generator=torch.Generator().manual_seed(123))
        second = buffer.sample_starts(16, 2, generator=torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]))


class CapacityAndErrorTest(unittest.TestCase):
    """容量计算与错误输入。"""

    def test_bytes_follow_actual_fields(self) -> None:
        # 手算：两段历史 2*10*33*4 + 两段 Q 2*C*4 + 动作 8*4 + 奖励 4 + 两个 bool 标记 2 + 5 个 int64 编号 40 + 写入序号 8。
        for q_dim in (54, 241):
            expected = 2 * 10 * 33 * 4 + 2 * q_dim * 4 + 8 * 4 + 4 + 2 + 5 * 8 + 8
            self.assertEqual(bytes_per_record(q_dim), expected)
        buffer = ReplayBuffer(num_envs=3, capacity_per_env=5, q_dim=54)
        self.assertEqual(buffer.storage_bytes, estimate_storage_bytes(3, 5, 54) + 3 * 8)

    def test_rejects_bad_writes_and_reads(self) -> None:
        buffer = ReplayBuffer(num_envs=2, capacity_per_env=3, q_dim=Q_DIM)
        with self.assertRaisesRegex(ReplayBufferError, "不能重复"):
            buffer.append(make_records([(0, 0, 0), (0, 0, 1)]))
        with self.assertRaisesRegex(ReplayBufferError, "env_id 必须在"):
            buffer.append(make_records([(2, 0, 0)]))
        with self.assertRaisesRegex(ReplayBufferError, "没有满足"):
            buffer.sample_starts(4, 1)
        with self.assertRaisesRegex(ReplayBufferError, "n_steps=4"):
            buffer.valid_starts(4)
        with self.assertRaisesRegex(ReplayBufferError, "必须为正"):
            ReplayBuffer(num_envs=1, capacity_per_env=0, q_dim=Q_DIM)
        buffer.append(make_records([(0, 0, 0)]))
        read = buffer.read_sequence(torch.tensor([0]), torch.tensor([0]), 2)
        with self.assertRaisesRegex(ReplayBufferError, "无效记录"):
            read.at(torch.tensor([1]))


class StateTest(unittest.TestCase):
    """state_dict / load_state_dict：恢复后有效起点、读取与后续写入位置都与原经验池一致。"""

    def test_round_trip_and_continued_writes(self) -> None:
        original = ReplayBuffer(num_envs=2, capacity_per_env=4, q_dim=Q_DIM)
        for step in range(6):  # 环境 0 已覆盖一圈半，环境 1 未满
            original.append(make_records([(0, 0, step)] + ([(1, 2, step)] if step < 3 else [])))
        state = original.state_dict()
        restored = ReplayBuffer(num_envs=2, capacity_per_env=4, q_dim=Q_DIM)
        restored.load_state_dict(state)
        for n in (1, 2, 3):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(original.valid_starts(n), restored.valid_starts(n))))
        env_ids, slots = original.valid_starts(2)
        a, b = original.read_sequence(env_ids, slots, 2), restored.read_sequence(env_ids, slots, 2)
        self.assertTrue(all(torch.equal(a.fields[k], b.fields[k]) for k in a.fields) and torch.equal(a.valid, b.valid))
        nxt = make_records([(0, 0, 6), (1, 2, 3)])
        self.assertTrue(torch.equal(original.append(nxt), restored.append(nxt)))  # 下一次写入落在相同槽位
        state["storage"]["reward"].fill_(-1.0)  # 导出结果是副本
        self.assertFalse(bool((restored.read_sequence(env_ids, slots, 1).fields["reward"] == -1.0).any()))

    def test_rejects_mismatched_shape(self) -> None:
        state = ReplayBuffer(num_envs=2, capacity_per_env=4, q_dim=Q_DIM).state_dict()
        with self.assertRaisesRegex(ReplayBufferError, "不符"):
            ReplayBuffer(num_envs=2, capacity_per_env=5, q_dim=Q_DIM).load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
