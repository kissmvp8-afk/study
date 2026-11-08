"""T02 采集与学习调度：把 E03/E04、D01～D05、N01～N08 串成“采集 → 经验池 → 抽样学习”的训练流程。

本模块只负责调度：何时调用哪个已有模块、调用多少次、各项计数。不复制任何底层公式
（目标 Q、损失、密度、统计更新、慢更新都在各自模块中）。

一轮迭代（train）：
    1. collect_steps(steps_per_iteration)：当前 Actor 随机采样动作 → 环境执行 → D01 make_transition
       → D04 update_from_fresh_data（只用这批新数据）→ D02 append。预热期也使用这个有界、温和初始化的 Actor，
       不改成全范围均匀随机动作（规范要求；作者 start_training 期间同样用 Actor 采集）。
    2. 预热结束后 learn_from_replay(critic_updates_per_iteration)：每次 Critic 更新按作者 sac.py 的顺序：
         D02 抽样 → D03 n 步样本 → N05 目标与在线 Q 更新
         → 在当前状态上采样一次 (a, log π)
         → 到 alpha 更新时机：N07 用这次 log π 更新 log_alpha
         → 到 Actor 更新时机（默认每 2 次 Critic 更新一次，作者 policy_frequency）：N06 用**更新后的** alpha
           与同一次采样组装 Actor 损失并更新
         → 到慢更新时机：N08 soft_update。
       回放只调用 D04 normalize，不更新统计。
    3. auxiliary_update × auxiliary_updates_per_iteration：N02 辅助学习，频率按迭代计，与 Critic 更新次数无关，
       避免编码器随大量 Q 回放被重复训练。有效历史不足时按 N02 约定跳过并计数。
    4. 可选：按间隔调用 T03 save_checkpoint。

计数：环境步（env.step 调用次数）、transition（写入经验池的记录数 = 环境步 × 环境数）、Critic / Actor / alpha /
慢更新 / 辅助更新次数、辅助跳过次数、回放样本数（Critic 学习用）与辅助样本数，以及因经验不足跳过的学习轮数。

随机性：网络初始化使用 torch.manual_seed(config.seed)；之后的采集动作、回放抽样、学习中的采样各用一个
独立的 torch.Generator（collect / replay / learn），T03 保存并恢复它们的状态。环境本身的随机状态由环境负责。

环境接口：见 fake_env.py（reset / step 与 EnvObservation / EnvStep）。末状态、重置与编号由环境（将来是 E05）提供，
调度器不重新实现。
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    "environment_control/action_contract", "environment_control/action_bounds",
    "experience_returns/transition_schema", "experience_returns/replay_buffer", "experience_returns/n_step_returns",
    "experience_returns/normalization", "experience_returns/history_windows",
    "networks_learning/history_encoder", "networks_learning/auxiliary_learning", "networks_learning/actor_distribution",
    "networks_learning/q_networks", "networks_learning/critic_learning", "networks_learning/actor_learning",
    "networks_learning/entropy_temperature", "networks_learning/target_updates", "training_validation/configuration",
):
    sys.path.insert(0, str(_ROOT / _path))

from actor import SquashedGaussianActor  # noqa: E402
from actor_update import compute_actor_loss, make_actor_optimizer, update_actor  # noqa: E402
from auxiliary import BarlowProjector, make_auxiliary_optimizer, update_encoder  # noqa: E402
from baseline_source import load_baseline_action_contract  # noqa: E402
from bounds import calibrate_initial_center, derive_bounds  # noqa: E402
from buffer import ReplayBuffer  # noqa: E402
from config import RunConfig, validate_configuration  # noqa: E402
from critic_update import compute_target, make_critic_optimizer, update_critics  # noqa: E402
from critics import TwinQ, initialize_targets  # noqa: E402
from encoder import HistoryEncoder, features_from_history  # noqa: E402
from history import split_windows  # noqa: E402
from normalization import ObservationNormalizers  # noqa: E402
from returns import build_n_step_batch  # noqa: E402
from schema import make_transition  # noqa: E402
from targets import soft_update, soft_update_due  # noqa: E402
from temperature import Temperature, current_alpha, current_alpha_tensor, make_temperature_optimizer, target_entropy, update_temperature  # noqa: E402


def to_learning_device(value, device):
    """仅搬运批次张量，保持数据类、历史顺序、mask 与 dtype，不修改 CPU 经验池。"""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if is_dataclass(value):
        return type(value)(**{f.name: to_learning_device(getattr(value, f.name), device) for f in fields(value)})
    return value


@dataclass
class TrainingCounters:
    """训练进度的分项计数；全部保存在 T03 检查点中。"""

    iterations: int = 0
    env_steps: int = 0
    transitions: int = 0
    critic_updates: int = 0
    actor_updates: int = 0
    alpha_updates: int = 0
    target_updates: int = 0
    auxiliary_updates: int = 0
    auxiliary_skipped: int = 0
    replay_samples: int = 0
    auxiliary_samples: int = 0
    learning_skipped_iterations: int = 0


class Trainer:
    """持有全部模块、优化器、随机数生成器与计数；所有设置都来自 RunConfig。

    参数：
        config: T01 RunConfig，构造时用 validate_configuration 检查，未确定项会使构造失败。
        env: 实现 reset / step 接口的环境（目前只有 fake_env.FakeLocomotionEnv）。
        contract: E03 动作表；默认读取原版快照。
    副作用：构造时调用 torch.manual_seed(config.seed) 初始化网络，并调用 env.reset()。
    """

    def __init__(self, config: RunConfig, env, contract=None) -> None:
        validate_configuration(config)
        self.config, self.env = config, env
        self.device = torch.device(config.device)
        a, n, c, t = config.algorithm, config.network, config.collection, config.task
        if env.num_envs != c.num_envs or env.q_dim != t.q_dim:
            raise ValueError(f"环境 ({env.num_envs} 个, C={env.q_dim}) 与配置 ({c.num_envs} 个, C={t.q_dim}) 不符")
        torch.manual_seed(config.seed)
        self.contract = contract if contract is not None else load_baseline_action_contract()
        self.bounds = derive_bounds(self.contract, t.joint_pos_limits, t.wheel_speed_ranges)
        initial_center = calibrate_initial_center(self.contract, self.bounds, t.initial_targets)
        self.target_entropy = target_entropy(self.bounds.half_width, a.target_entropy_scale)

        self.normalizers = ObservationNormalizers(t.q_dim, eps=a.normalizer_eps)
        self.buffer = ReplayBuffer(c.num_envs, c.capacity_per_env, t.q_dim)
        self.encoder = HistoryEncoder(n.encoder_backbone_dims, n.encoder_latent_hidden_dim)
        self.projector = BarlowProjector()
        self.actor = SquashedGaussianActor(self.bounds, initial_center, hidden_dims=n.actor_hidden_dims, initial_std=a.initial_std)
        self.critics = TwinQ(t.q_dim, hidden_dims=n.critic_hidden_dims)
        self.targets = initialize_targets(self.critics)
        self.temperature = Temperature(a.initial_alpha)
        # 在构造优化器之前迁移参数和缓冲，优化器绑定正确设备的参数。
        for name in ("encoder", "projector", "actor", "critics", "targets", "temperature"):
            getattr(self, name).to(self.device)
        self.optimizers = {
            "critic": make_critic_optimizer(self.critics, lr=a.critic_lr),
            "actor": make_actor_optimizer(self.actor, lr=a.actor_lr),
            "alpha": make_temperature_optimizer(self.temperature, lr=a.alpha_lr),
            "auxiliary": make_auxiliary_optimizer(self.encoder, self.projector, lr=a.auxiliary_lr),
        }
        # 三个独立的随机数生成器：采集、回放抽样、学习中的采样（N05 尾部动作与 Actor/alpha 采样）。
        self.generators = {name: torch.Generator(device="cpu" if name == "replay" else self.device).manual_seed(config.seed + offset)
                           for offset, name in enumerate(("collect", "replay", "learn"), start=1)}
        self.counters = TrainingCounters()
        self.observation = env.reset()

    # ---------------- 采集 ----------------
    def collect_steps(self, num_steps: int) -> dict:
        """用当前 Actor（随机采样，有界、温和初始化）与环境交互 num_steps 步，写入经验池。

        每一步：动作 → env.step → D01 记录（当前观测 + 重置前真实下一观测）→ D04 统计（只用这批新数据）→ D02。
        返回本次采集的奖励均值、真终止数与纯截断数。
        """
        rewards, terminated, time_out = [], 0, 0
        for _ in range(num_steps):
            obs = self.observation
            with torch.no_grad():
                features = features_from_history(self.encoder, self.normalizers, obs.policy_history.to(self.device))
                action, _ = self.actor.sample_action_and_log_prob(features, generator=self.generators["collect"])
            if not bool(torch.isfinite(action).all()):
                raise FloatingPointError("采集动作出现非有限数值，未提交到仿真")
            action = action.cpu()  # E05 与 D01/D02 保持 CPU 契约；E05 再送至仿真设备。
            step = self.env.step(action)
            batch = make_transition({
                "policy_history": obs.policy_history, "next_policy_history": step.next_policy_history,
                "q_observation": obs.q_observation, "next_q_observation": step.next_q_observation,
                "action": action, "reward": step.reward, "terminated": step.terminated, "time_out": step.time_out,
                "env_id": obs.env_id, "episode_id": obs.episode_id, "step_index": obs.step_index,
                "history_length": obs.history_length, "next_history_length": step.next_history_length,
            }, self.config.task.q_dim)
            self.normalizers.update_from_fresh_data(batch)
            self.buffer.append(batch)
            self.observation = step.observation
            self.counters.env_steps += 1
            self.counters.transitions += batch.batch_size
            rewards.append(step.reward.mean().item())
            terminated += int(step.terminated.sum())
            time_out += int(step.time_out.sum())
        return {"reward_mean": sum(rewards) / max(len(rewards), 1), "terminated": terminated, "time_out": time_out}

    # ---------------- 回放学习 ----------------
    def learn_from_replay(self, num_critic_updates: int) -> dict:
        """执行 num_critic_updates 次 Critic 更新及其后的 alpha / Actor / 慢更新（顺序见文件头）。

        经验池中没有满足 n 步的有效起点时（D02 规则），跳过本次学习并计数。
        """
        a = self.config.algorithm
        if self.buffer.valid_starts(a.n_steps)[0].numel() == 0:
            self.counters.learning_skipped_iterations += 1
            return {"skipped": True}
        logs: dict[str, list[float]] = {}
        for _ in range(num_critic_updates):
            update_index = self.counters.critic_updates  # 本次更新之前已完成的 Critic 更新数（作者 update_step）
            env_ids, slots = self.buffer.sample_starts(a.batch_size, a.n_steps, generator=self.generators["replay"])
            batch = build_n_step_batch(self.buffer.read_sequence(env_ids, slots, a.n_steps), a.gamma, a.n_steps)
            batch = to_learning_device(batch, self.device)
            target = compute_target(batch, self.encoder, self.actor, self.targets, self.normalizers,
                                    current_alpha_tensor(self.temperature), generator=self.generators["learn"])
            critic_log = update_critics(self.critics, self.optimizers["critic"], batch, target, self.normalizers, a.max_grad_norm)
            if not all(math.isfinite(v) for v in critic_log.values()):
                raise FloatingPointError(f"Critic 更新出现非有限数值：{critic_log}")
            self.counters.critic_updates += 1
            self.counters.replay_samples += a.batch_size
            for key, value in critic_log.items():
                logs.setdefault(key, []).append(value)
            logs.setdefault("discount_max", []).append(batch.discount.max().item())

            # 在当前状态上只采样一次；alpha 与 Actor 都使用这次采样。
            features = features_from_history(self.encoder, self.normalizers, batch.first.policy_history)
            q_input = self.normalizers.normalize_q(batch.first.q_observation)
            action, log_prob = self.actor.sample_action_and_log_prob(features, generator=self.generators["learn"])
            if update_index % a.alpha_update_every == 0:
                alpha_log = update_temperature(self.temperature, self.optimizers["alpha"], log_prob,
                                               self.target_entropy.action_coordinate)
                if not all(math.isfinite(v) for v in alpha_log.values()):
                    raise FloatingPointError(f"alpha 更新出现非有限数值：{alpha_log}")
                self.counters.alpha_updates += 1
                logs.setdefault("alpha", []).append(alpha_log["alpha"])
            if update_index % a.actor_update_every == 0:
                # Actor 损失在 alpha 更新之后组装，因此使用的是新 alpha（作者顺序）。
                loss = compute_actor_loss(self.critics, q_input, action, log_prob, current_alpha_tensor(self.temperature))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Actor 损失出现非有限数值")
                update_actor(self.actor, self.optimizers["actor"], loss, a.max_grad_norm)
                self.counters.actor_updates += 1
                logs.setdefault("actor_loss", []).append(loss.item())
            if soft_update_due(self.counters.critic_updates, a.target_update_every):
                soft_update(self.critics, self.targets, a.tau)
                self.counters.target_updates += 1
        return {"skipped": False, **{k: sum(v) / len(v) for k, v in logs.items()}, "discount_max": max(logs["discount_max"])}

    # ---------------- 辅助学习 ----------------
    def auxiliary_update(self) -> dict:
        """一次 N02 辅助更新：从经验池抽取单步记录，按当前统计归一化后切窗口；有效历史不足时由 N02 跳过。"""
        a, t = self.config.algorithm, self.config.task
        if self.buffer.size == 0:
            self.counters.auxiliary_skipped += 1
            return {"skipped": True, "valid_samples": 0}
        env_ids, slots = self.buffer.sample_starts(a.auxiliary_batch_size, 1, generator=self.generators["replay"])
        first = self.buffer.read_sequence(env_ids, slots, 1).at(torch.zeros_like(env_ids))
        first = to_learning_device(first, self.device)
        windows = split_windows(self.normalizers.normalize_policy(first.policy_history))
        start, end = t.velocity_label_columns
        report = update_encoder(self.encoder, self.projector, self.optimizers["auxiliary"], windows,
                                first.history_length, first.q_observation[:, start:end], a.max_grad_norm)
        if not all(math.isfinite(v) for v in report.values() if isinstance(v, float)):
            raise FloatingPointError(f"辅助更新出现非有限数值：{report}")
        if report["skipped"]:
            self.counters.auxiliary_skipped += 1
        else:
            self.counters.auxiliary_updates += 1
            self.counters.auxiliary_samples += report["valid_samples"]
        return report

    # ---------------- 整体流程 ----------------
    def train(self, iterations: int, checkpoint_path: Path | None = None, checkpoint_every: int = 0,
              include_replay: bool = True) -> list[dict]:
        """运行 iterations 轮迭代（流程见文件头），返回每轮日志；日志中出现非有限数值时报错。"""
        if checkpoint_path is not None:
            sys.path.insert(0, str(_ROOT / "training_validation" / "checkpoints"))
            from checkpoint import save_checkpoint  # T03；仅在需要保存时导入
        c, a = self.config.collection, self.config.algorithm
        history = []
        for _ in range(iterations):
            log = {"iteration": self.counters.iterations, "collect": self.collect_steps(c.steps_per_iteration)}
            if self.counters.iterations >= c.warmup_iterations:
                log["learn"] = self.learn_from_replay(a.critic_updates_per_iteration)
                log["auxiliary"] = [self.auxiliary_update() for _ in range(a.auxiliary_updates_per_iteration)]
            self.counters.iterations += 1
            log["counters"] = asdict(self.counters)
            parts = [p for value in log.values() for p in (value if isinstance(value, list) else [value]) if isinstance(p, dict)]
            numbers = [v for part in parts for v in part.values() if isinstance(v, float)]
            if not all(math.isfinite(v) for v in numbers):
                raise FloatingPointError(f"第 {log['iteration']} 轮日志出现非有限数值：{log}")
            if checkpoint_path is not None and checkpoint_every > 0 and self.counters.iterations % checkpoint_every == 0:
                save_checkpoint(self, checkpoint_path, include_replay=include_replay)
            history.append(log)
        return history
