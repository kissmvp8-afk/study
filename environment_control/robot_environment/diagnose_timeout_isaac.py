"""用真实 Isaac 平地物理步补测 E05→D01/D03 的纯时间截断路径。

仅在正常 reset 后把 episode_length_buf 提前设为上限减二；正式回合
时长、物理参数、奖励、动作与真实终止条件均不改动。成功不表示机器人
稳定运行了完整 20 秒。观测通过原钩子的同一次 compute 捕获，不额外采样噪声。
脚本不训练、不保存模型；只写小型 JSON。Isaac 导入仍在 AppLauncher 之后。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    """限定单环境 headless，失败时最多短尝试三次。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_attempts", type=int, default=3, choices=range(1, 4))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.headless:
        parser.error("本诊断只允许 --headless")
    return args


def write_json(path, value):
    """禁止 NaN/Infinity 写入验收证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(args):
    """真实物理采集、原接口校验与证据输出；任一断言失败则非零退出。"""
    for relative in ("environment_control/episode_lifecycle", "environment_control/observation_interface",
                     "environment_control/action_contract", "training_validation/configuration",
                     "experience_returns/transition_schema", "experience_returns/replay_buffer",
                     "experience_returns/n_step_returns"):
        sys.path.insert(0, str(ROOT / relative))
    import carb
    import torch
    from assets import parse_urdf_joints, soft_joint_limits, verify_urdf
    from baseline_source import load_baseline_action_contract
    from buffer import ReplayBuffer
    from contract import ControlType
    from isaac_env import TitaSacHooks, make_env, make_sac_cfg, read_runtime_metadata, sac_config_differences, urdf_path_from_cfg
    from lifecycle import EpisodeLifecycle
    from metadata import LEG_SOFT_LIMIT_FACTOR, baseline_command_ranges, baseline_timing, compare_with_baseline
    from observations import read_observation_groups
    from returns import build_n_step_batch
    from schema import FIELD_NAMES, derive_bootstrap_mask, make_transition

    def serial(value):
        """把实际张量转为 JSON 数组，不生成或改写观测。"""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {key: serial(item) for key, item in value.items()}
        return value

    class CapturingHooks(TitaSacHooks):
        """只旁路复制真实钩子输出；终止、奖励、重置仍由父类执行。"""
        def compute_observations(self):
            """复制父类同一次计算结果，分别留存重置前/后的实际观测与根状态。"""
            groups = super().compute_observations()
            self.captures.append({"groups": {k: v.detach().cpu().clone() for k, v in groups.items()},
                                  "root_state_w": self.env.scene["robot"].data.root_state_w.detach().cpu().clone()})
            return groups

        def compute_terminations(self):
            """记录原管理器各项标记和计数器，不改写任何终止结果。"""
            flags = super().compute_terminations()
            self.actual_terms = {name: self.env.termination_manager.get_term(name).detach().cpu().clone()
                                 for name in self.env.termination_manager.active_terms}
            self.counter_at_termination = int(self.env.episode_length_buf[0])
            return flags

    # 与 inspect 相同的资产/运行时核对；不改生产配置来制造截断。
    cfg = make_sac_cfg("isaac_flat", 1, args.seed, args.device or "cuda:0")
    asset_check = verify_urdf(urdf_path_from_cfg(cfg))
    contract = load_baseline_action_contract()
    legs = [entry.joint_name for entry in contract.entries if entry.control_type is ControlType.POSITION]
    soft = soft_joint_limits(parse_urdf_joints(urdf_path_from_cfg(cfg)), legs, LEG_SOFT_LIMIT_FACTOR)
    differences = sac_config_differences("isaac_flat")
    env = make_env("isaac_flat", cfg)
    attempts = []
    try:
        runtime = read_runtime_metadata(env, "isaac_flat")
        q_dim = compare_with_baseline(runtime, contract, baseline_timing(), baseline_command_ranges(), soft)
        limit = int(env.max_episode_length)
        assert limit == 1000 and cfg.episode_length_s == 20
        assert set(env.termination_manager.active_terms) == {"time_out", "base_contact"}
        watcher = carb.settings.get_settings().get("/app/extensions/fsWatcherEnabled")
        assert watcher is False, "诊断必须禁用扩展文件热重载"
        write_json(args.output_dir / "metadata.json", {
            "task": "isaac_flat", "num_envs": 1, "sim_device": str(env.device), "q_dim": q_dim,
            "runtime": asdict(runtime), "asset_check": asset_check,
            "sac_vs_ppo_cfg_differences": differences, "fsWatcherEnabled": watcher,
            "max_episode_length": limit, "episode_length_s": cfg.episode_length_s,
            "termination_terms": list(env.termination_manager.active_terms),
        })
        hooks = CapturingHooks(env)
        lifecycle = EpisodeLifecycle(hooks, "isaac_flat", q_dim)
        action = torch.zeros(1, len(contract))

        def collect(observation):
            """原 E05 执行一步，再由原 D01 复制、检查真实采集记录。"""
            hooks.captures = []
            step = lifecycle.step(action)
            assert len(hooks.captures) == 2
            batch = make_transition({
                "policy_history": observation.policy_history, "next_policy_history": step.next_policy_history,
                "q_observation": observation.q_observation, "next_q_observation": step.next_q_observation,
                "action": action, "reward": step.reward, "terminated": step.terminated, "time_out": step.time_out,
                "env_id": observation.env_id, "episode_id": observation.episode_id, "step_index": observation.step_index,
                "history_length": observation.history_length, "next_history_length": step.next_history_length,
            }, q_dim)
            return step, batch

        for attempt in range(args.max_attempts):
            observation = lifecycle.reset()
            assert int(env.episode_length_buf[0]) == 0
            start_physics = int(env._sim_step_counter)
            # 唯一的诊断性状态注入：不修改配置上限，也不伪造结束标记。
            env.episode_length_buf.fill_(limit - 2)
            records = []
            attempt_report = {"attempt": attempt + 1, "counter_after_reset": 0, "injected_counter": limit - 2,
                              "steps": [], "pure_timeout": False}
            attempts.append(attempt_report)
            for _ in range(4):
                previous = observation
                step, batch = collect(previous)
                records.append(batch)
                attempt_report["steps"].append({"terminated": bool(step.terminated[0, 0]),
                                               "time_out": bool(step.time_out[0, 0]),
                                               "counter_at_termination": hooks.counter_at_termination,
                                               "manager_terms": serial(hooks.actual_terms)})
                observation = step.observation
                if bool(step.terminated[0, 0]):
                    # 真终止优先；只重试正常 reset，不关闭 base_contact。
                    break
                if not bool(step.time_out[0, 0]):
                    continue
                attempt_report["pure_timeout"] = True
                assert bool(hooks.actual_terms["time_out"][0])
                assert not bool(hooks.actual_terms["base_contact"][0])
                assert hooks.counter_at_termination == limit
                pre, post = hooks.captures
                frame_pre, q_pre = read_observation_groups(pre["groups"], "isaac_flat", q_dim)
                frame_post, q_post = read_observation_groups(post["groups"], "isaac_flat", q_dim)
                assert torch.equal(step.next_q_observation, q_pre)
                assert torch.equal(step.next_policy_history[:, -1], frame_pre)
                assert torch.equal(step.next_policy_history[:, :-1], previous.policy_history[:, 1:])
                assert torch.equal(step.next_history_length, previous.history_length + 1)
                assert torch.equal(observation.q_observation, q_post)
                assert torch.equal(observation.policy_history, frame_post[:, None].expand(1, 10, 33))
                assert observation.history_length.tolist() == [1] and observation.step_index.tolist() == [0]
                assert torch.equal(observation.episode_id, previous.episode_id + 1)
                assert not torch.equal(q_pre, q_post) and not torch.equal(pre["root_state_w"], post["root_state_w"])
                assert derive_bootstrap_mask(batch).tolist() == [[1.0]]
                final_step = step
                saved = {name: getattr(step, name).clone() for name in
                         ("next_policy_history", "next_q_observation", "next_history_length")}
                timeout_record_index = len(records) - 1
                actual_physics_to_timeout = int(env._sim_step_counter) - start_physics
                # 新回合继续真实交互三步；奖励保持原值，供 D02/D03 验证边界隔离。
                for _ in range(3):
                    following, following_batch = collect(observation)
                    records.append(following_batch)
                    observation = following.observation
                assert all(torch.equal(getattr(final_step, name), value) for name, value in saved.items())
                assert torch.equal(records[timeout_record_index].next_q_observation, q_pre)
                replay = ReplayBuffer(1, 16, q_dim)
                for record in records:
                    replay.append(record)
                sequence = replay.read_sequence(torch.tensor([0]), torch.tensor([0]), length=3)
                result = build_n_step_batch(sequence, gamma=0.97, n_steps=3)
                k = timeout_record_index + 1
                assert k <= 2 and result.steps.tolist() == [k]
                expected_reward = sum((0.97 ** j) * record.reward for j, record in enumerate(records[:k]))
                torch.testing.assert_close(result.reward_sum, expected_reward)
                torch.testing.assert_close(result.discount, torch.tensor([[0.97 ** k]]))
                assert result.bootstrap_mask.tolist() == [[1.0]]
                assert result.tail.episode_id.tolist() == records[0].episode_id.tolist()
                assert torch.equal(result.tail.next_q_observation, q_pre)
                assert torch.equal(result.tail.next_policy_history, saved["next_policy_history"])
                assert not bool(sequence.valid[0, k])
                assert int(sequence.fields["episode_id"][0, k]) != int(result.tail.episode_id[0])
                assert int(env.max_episode_length) == limit and cfg.episode_length_s == 20
                write_json(args.output_dir / "captured_observations.json", {
                    "pre_reset": serial(pre), "post_reset": serial(post),
                    "next_policy_history": serial(saved["next_policy_history"]),
                    "next_q_observation": serial(saved["next_q_observation"]),
                    "new_observation": serial(vars(final_step.observation)),
                    "transitions": [{name: serial(getattr(record, name)) for name in FIELD_NAMES} for record in records],
                })
                summary = {"status": "passed", "method": "counter_advanced_real_physics_timeout",
                           "limitation": "提前设置计数器触发真实截断，不证明稳定运行完整20秒",
                           "attempts": attempts, "terminated": False, "time_out": True,
                           "actual_physics_steps_to_timeout": actual_physics_to_timeout,
                           "actual_seconds_to_timeout": actual_physics_to_timeout * runtime.physics_dt,
                           "old_episode_id": serial(previous.episode_id),
                           "new_episode_id": serial(final_step.observation.episode_id),
                           "new_history_length": serial(final_step.observation.history_length),
                           "new_step_index": serial(final_step.observation.step_index),
                           "next_matches_captured_pre_reset": True, "observation_matches_captured_post_reset": True,
                           "saved_final_state_unchanged_after_following_steps": 3,
                           "d01_bootstrap_mask": serial(derive_bootstrap_mask(records[timeout_record_index])),
                           "d03": {"steps": serial(result.steps), "reward_sum": serial(result.reward_sum),
                                   "expected_old_episode_reward_sum": serial(expected_reward),
                                   "discount": serial(result.discount), "bootstrap_mask": serial(result.bootstrap_mask),
                                   "tail_episode_id": serial(result.tail.episode_id),
                                   "sequence_valid": serial(sequence.valid),
                                   "excluded_new_episode_reward": serial(records[k].reward)},
                           "fsWatcherEnabled": watcher, "formal_episode_length_s_unchanged": 20}
                write_json(args.output_dir / "summary.json", summary)
                print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
                return
        write_json(args.output_dir / "summary.json", {"status": "not_covered", "attempts": attempts})
        raise RuntimeError("有限尝试内未捕获纯时间截断；不得记为通过")
    finally:
        env.close()


if __name__ == "__main__":
    cli = parse_args()
    simulation_app = AppLauncher(cli).app
    try:
        main(cli)
    finally:
        simulation_app.close()
