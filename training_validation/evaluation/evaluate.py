"""T04 PPO 与 SAC 对照评价：定义评价记录，计算可复查的指标，并在证据不足时拒绝给出性能结论。

本模块只处理“评价记录 → 指标 → 报告 / 对照”。真实的 PPO / SAC 对照需要 Isaac 仿真中的评价运行，目前尚未进行，
本模块不生成任何虚构的性能结论。

评价记录（EvaluationRecord）：一次评价运行的来源信息 + 若干 episode；每个 episode 按控制周期逐步记录：
    commanded_velocity / measured_velocity：[T,3]，(vx, vy, wz)，单位 m/s、m/s、rad/s，身体（偏航对齐）坐标系；
    actions：[T,8]，策略动作坐标（E03 顺序，无单位）；
    joint_torques：[T,8]，N·m；
    end_reason："terminated"（真终止，计为摔倒）、"time_out"（正常到时）、"unfinished"（评价时长结束时仍在运行）。
缺失的信号为 None，对应指标标记为 missing，不用 0 冒充已测结果。

指标与统计口径（compute_metrics）：
    - 线速度跟踪误差：‖(vx, vy)_cmd − (vx, vy)_meas‖₂ 对全部 episode 的全部步取平均（按步加权，长 episode 权重大），m/s；
      偏航角速度误差：|wz_cmd − wz_meas| 同口径，rad/s。
    - 摔倒率：真终止 episode 数 / 全部 episode 数。正常到时与 unfinished 都不算摔倒；分母包含 unfinished，
      并单独报告 unfinished 数，便于判断评价时长是否足够。
    - 稳定时长：每个 episode 的步数 × 控制周期（s），对 episode 取平均（按 episode 加权）；另报告摔倒 episode 的平均倒地时间。
    - 动作平滑性：同一 episode 内相邻两步策略动作差的平方和 Σ(a_t − a_{t−1})² 的平均（与原 PPO action_rate 奖励项同形式），
      无单位；不跨 episode 求差。
    - 力矩：|τ| 的平均与 τ² 的平均（N·m、N²·m²），按步与关节平均。
    - 耗时与内存：记录中的墙钟时间（s）与峰值内存（字节），缺失则 missing。
对照（compare_runs）：两组运行（各至少 3 个不同的有效训练随机起点）按交互步数或实际耗时两种预算之一比较。
所选预算必须满足可比条件，否则只展示差异、不判定：
    - interaction_steps：两组全部记录的交互步数必须相等（固定配置下可精确控制）；
    - wall_time：两组全部记录的墙钟时间满足 最大值 ≤ 最小值 × (1 + WALL_TIME_RELATIVE_TOLERANCE)，
      容差 5% 是本项目暂定的约定（未经验证），写入结果的 budget_check 中。
验收阈值必须在得到 PPO 波动范围后、正式 SAC 比较前固定，未给出阈值时只报告差异，不判定优劣。
随机起点按有效（非空）且互不相同的 training_seed 计数（distinct_training_seeds），不按记录条数；
compare_runs 与 write_report 使用同一口径。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

import torch

END_REASONS = ("terminated", "time_out", "unfinished")
REQUIRED_PROVENANCE = ("algorithm", "model_source", "config_name", "code_commit", "training_seed")
MIN_RECOMMENDED_SEEDS = 3
WALL_TIME_RELATIVE_TOLERANCE = 0.05  # 耗时预算的相对容差：本项目暂定约定，未经验证
CONTROL_DT = 0.02  # 策略周期 0.02 s（50 Hz），根 README 统一约定


@dataclass(frozen=True)
class EvaluationEpisode:
    """一个 episode 的逐步记录；信号缺失时为 None。"""

    actions: torch.Tensor  # [T,8]
    end_reason: str
    commanded_velocity: torch.Tensor | None = None  # [T,3]
    measured_velocity: torch.Tensor | None = None  # [T,3]
    joint_torques: torch.Tensor | None = None  # [T,8]

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class EvaluationRecord:
    """一次评价运行：来源（provenance）、控制周期、episode 列表与资源记录。"""

    provenance: dict
    episodes: tuple[EvaluationEpisode, ...]
    control_dt: float = CONTROL_DT
    interaction_steps: int | None = None  # 训练使用的环境交互步数（预算口径一）
    wall_time_s: float | None = None  # 训练墙钟时间（预算口径二）
    peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class MetricValue:
    """一个指标：value 为 None 时 status 必为 "missing"。"""

    value: float | None
    unit: str
    scope: str
    status: str = "measured"
    samples: int = 0

    @staticmethod
    def missing(unit: str, scope: str) -> "MetricValue":
        return MetricValue(None, unit, scope, "missing", 0)


def validate_record(record: EvaluationRecord) -> None:
    """检查 episode 的结束原因、形状一致与信号形状。异常：ValueError。"""
    for i, ep in enumerate(record.episodes):
        if ep.end_reason not in END_REASONS:
            raise ValueError(f"episode {i} 的结束原因 {ep.end_reason!r} 不在 {END_REASONS}")
        if ep.actions.dim() != 2 or ep.actions.shape[1] != 8 or ep.steps < 1:
            raise ValueError(f"episode {i} 的动作形状应为 [T,8]（T ≥ 1），实际为 {list(ep.actions.shape)}")
        for name, width in (("commanded_velocity", 3), ("measured_velocity", 3), ("joint_torques", 8)):
            value = getattr(ep, name)
            if value is not None and tuple(value.shape) != (ep.steps, width):
                raise ValueError(f"episode {i} 的 {name} 形状应为 [{ep.steps},{width}]，实际为 {list(value.shape)}")


def _all_present(episodes, name: str) -> bool:
    return bool(episodes) and all(getattr(ep, name) is not None for ep in episodes)


def compute_metrics(record: EvaluationRecord) -> dict[str, MetricValue]:
    """按文件头的口径计算全部指标；缺少任一 episode 的所需信号时该指标为 missing（不对部分 episode 求平均）。"""
    validate_record(record)
    eps, dt = record.episodes, record.control_dt
    total_steps = sum(ep.steps for ep in eps)
    metrics: dict[str, MetricValue] = {}
    if _all_present(eps, "commanded_velocity") and _all_present(eps, "measured_velocity"):
        cmd = torch.cat([ep.commanded_velocity for ep in eps]).double()
        meas = torch.cat([ep.measured_velocity for ep in eps]).double()
        metrics["linear_velocity_error"] = MetricValue((cmd[:, :2] - meas[:, :2]).norm(dim=1).mean().item(), "m/s", "按步平均", samples=total_steps)
        metrics["yaw_rate_error"] = MetricValue((cmd[:, 2] - meas[:, 2]).abs().mean().item(), "rad/s", "按步平均", samples=total_steps)
    else:
        metrics["linear_velocity_error"] = MetricValue.missing("m/s", "按步平均")
        metrics["yaw_rate_error"] = MetricValue.missing("rad/s", "按步平均")
    counts = {reason: sum(ep.end_reason == reason for ep in eps) for reason in END_REASONS}
    metrics["fall_rate"] = MetricValue(counts["terminated"] / len(eps) if eps else None, "比例",
                                       "真终止 episode / 全部 episode（到时与 unfinished 不算摔倒）",
                                       "measured" if eps else "missing", len(eps))
    metrics["time_out_count"] = MetricValue(float(counts["time_out"]), "个", "正常到时的 episode", samples=len(eps))
    metrics["unfinished_count"] = MetricValue(float(counts["unfinished"]), "个", "评价结束时仍在运行的 episode", samples=len(eps))
    metrics["stable_duration"] = MetricValue(mean(ep.steps * dt for ep in eps) if eps else None, "s", "按 episode 平均",
                                             "measured" if eps else "missing", len(eps))
    fallen = [ep.steps * dt for ep in eps if ep.end_reason == "terminated"]
    metrics["time_to_fall"] = (MetricValue(mean(fallen), "s", "摔倒 episode 的平均倒地时间", samples=len(fallen))
                               if fallen else MetricValue.missing("s", "摔倒 episode 的平均倒地时间（无摔倒）"))
    diffs = [(ep.actions[1:] - ep.actions[:-1]).double().pow(2).sum(dim=1) for ep in eps if ep.steps > 1]
    metrics["action_rate"] = (MetricValue(torch.cat(diffs).mean().item(), "策略动作单位²", "episode 内相邻步 Σ(Δa)² 的平均",
                                          samples=sum(d.numel() for d in diffs))
                              if diffs else MetricValue.missing("策略动作单位²", "episode 内相邻步 Σ(Δa)² 的平均"))
    if _all_present(eps, "joint_torques"):
        tau = torch.cat([ep.joint_torques for ep in eps]).double()
        metrics["torque_abs_mean"] = MetricValue(tau.abs().mean().item(), "N·m", "按步与关节平均", samples=tau.numel())
        metrics["torque_squared_mean"] = MetricValue(tau.pow(2).mean().item(), "N²·m²", "按步与关节平均", samples=tau.numel())
    else:
        metrics["torque_abs_mean"] = MetricValue.missing("N·m", "按步与关节平均")
        metrics["torque_squared_mean"] = MetricValue.missing("N²·m²", "按步与关节平均")
    for name, value, unit in (("wall_time", record.wall_time_s, "s"), ("peak_memory", record.peak_memory_bytes, "字节"),
                              ("interaction_steps", record.interaction_steps, "步")):
        metrics[name] = MetricValue(float(value), unit, "记录值", samples=1) if value is not None else MetricValue.missing(unit, "记录值")
    return metrics


def distinct_training_seeds(records: list[EvaluationRecord]) -> list:
    """有效（非 None、非空字符串）且互不相同的训练随机起点，按字符串排序。重复种子的多份记录只算一个起点。"""
    seeds = {r.provenance.get("training_seed") for r in records}
    return sorted((s for s in seeds if s not in (None, "")), key=str)


def check_budget(baseline: list[EvaluationRecord], candidate: list[EvaluationRecord], budget: str) -> dict:
    """检查所选预算在两组全部记录之间是否可比，返回 {rule, values, comparable, reason}。

    interaction_steps 必须全部相等；wall_time 要求 最大值 ≤ 最小值 × (1 + WALL_TIME_RELATIVE_TOLERANCE)。
    任一记录缺少该预算时不可比。
    """
    field = "interaction_steps" if budget == "interaction_steps" else "wall_time_s"
    values = {"baseline": [getattr(r, field) for r in baseline], "candidate": [getattr(r, field) for r in candidate]}
    flat = values["baseline"] + values["candidate"]
    if budget == "interaction_steps":
        rule = "两组全部记录的交互步数相等"
    else:
        rule = f"最大墙钟时间 ≤ 最小墙钟时间 × (1 + {WALL_TIME_RELATIVE_TOLERANCE})（暂定容差）"
    if not flat or any(v is None for v in flat):
        return {"rule": rule, "values": values, "comparable": False, "reason": f"有记录缺少 {budget} 预算"}
    if budget == "interaction_steps":
        comparable = len(set(flat)) == 1
    else:
        comparable = min(flat) > 0 and max(flat) <= min(flat) * (1 + WALL_TIME_RELATIVE_TOLERANCE)
    reason = None if comparable else f"{budget} 预算不满足可比条件：{rule}；实际 {values}"
    return {"rule": rule, "values": values, "comparable": comparable, "reason": reason}


def missing_provenance(record: EvaluationRecord) -> list[str]:
    """缺少的来源字段（模型来源、配置、代码版本、训练随机起点等）。"""
    return [key for key in REQUIRED_PROVENANCE if record.provenance.get(key) in (None, "")]


def run_evaluation(policy, env, num_steps: int, provenance: dict, signals=None, control_dt: float = CONTROL_DT) -> EvaluationRecord:
    """在实现 T02 reset / step 接口的环境上执行确定性评价，返回 EvaluationRecord。

    policy(observation) -> [E,8] 策略动作（应为确定性动作，例如 T05 导出管线）；
    signals(step) -> {"commanded_velocity": [E,3], "measured_velocity": [E,3], "joint_torques": [E,8]}，
    可只提供部分键；为 None 时这些信号全部缺失（T02 假环境不提供速度指令、真实速度与力矩）。
    评价结束时仍在运行的 episode 记为 unfinished。副作用：调用 env.reset() 与 env.step()。
    """
    observation = env.reset()
    buffers = [{"actions": [], "commanded_velocity": [], "measured_velocity": [], "joint_torques": []} for _ in range(env.num_envs)]
    episodes: list[EvaluationEpisode] = []

    def close(env_index: int, reason: str) -> None:
        b = buffers[env_index]
        if not b["actions"]:
            return
        stacked = {k: (torch.stack(v) if v and len(v) == len(b["actions"]) else None) for k, v in b.items()}
        episodes.append(EvaluationEpisode(stacked["actions"], reason, stacked["commanded_velocity"],
                                          stacked["measured_velocity"], stacked["joint_torques"]))
        buffers[env_index] = {k: [] for k in b}

    with torch.no_grad():
        for _ in range(num_steps):
            action = policy(observation)
            step = env.step(action)
            extra = signals(step) if signals is not None else {}
            for e in range(env.num_envs):
                buffers[e]["actions"].append(action[e].detach().clone())
                for key, value in extra.items():
                    buffers[e][key].append(value[e].detach().clone())
                if bool(step.terminated[e, 0]):
                    close(e, "terminated")
                elif bool(step.time_out[e, 0]):
                    close(e, "time_out")
            observation = step.observation
    for e in range(env.num_envs):
        close(e, "unfinished")
    return EvaluationRecord(provenance=dict(provenance), episodes=tuple(episodes), control_dt=control_dt)


def _metric_to_dict(metric: MetricValue) -> dict:
    return {"value": metric.value, "unit": metric.unit, "scope": metric.scope, "status": metric.status, "samples": metric.samples}


def summarize_runs(records: list[EvaluationRecord]) -> dict:
    """同一方法多次运行（不同训练随机起点）的指标：每项给出各次的值、均值与总体标准差；任一次缺失则该项为 missing。"""
    per_run = [compute_metrics(r) for r in records]
    summary = {}
    for name in per_run[0] if per_run else ():
        values = [m[name].value for m in per_run]
        if any(v is None for v in values):
            summary[name] = {"status": "missing", "values": values, "unit": per_run[0][name].unit}
        else:
            summary[name] = {"status": "measured", "values": values, "mean": mean(values),
                             "std": pstdev(values) if len(values) > 1 else None, "unit": per_run[0][name].unit}
    return {"runs": len(records), "training_seeds": distinct_training_seeds(records), "metrics": summary}


def compare_runs(baseline: list[EvaluationRecord], candidate: list[EvaluationRecord], budget: str,
                 thresholds: dict[str, float] | None = None) -> dict:
    """比较两组运行（例如 PPO 与 SAC）。budget 为 "interaction_steps" 或 "wall_time"，须满足 check_budget 的可比条件。

    thresholds：{指标名: 允许的 candidate − baseline 上限}，必须在正式比较前固定；None 时只报告差异、不判定。
    任一阻断条件成立（随机起点不足、缺来源、预算不可比、阈值未固定）时仍返回各指标差异，
    但 conclusion_allowed=False、全部 verdict 为 None，并在 conclusion_blockers 中写明原因。
    """
    if budget not in ("interaction_steps", "wall_time"):
        raise ValueError("budget 必须是 interaction_steps 或 wall_time")
    blockers = []
    for label, group in (("baseline", baseline), ("candidate", candidate)):
        seeds = distinct_training_seeds(group)
        if len(seeds) < MIN_RECOMMENDED_SEEDS:
            blockers.append(f"{label} 只有 {len(seeds)} 个不同的有效训练随机起点（{len(group)} 份记录），少于 {MIN_RECOMMENDED_SEEDS} 个")
        if any(missing_provenance(r) for r in group):
            blockers.append(f"{label} 有记录缺少来源信息")
    budget_check = check_budget(baseline, candidate, budget)
    if not budget_check["comparable"]:
        blockers.append(budget_check["reason"])
    if thresholds is None:
        blockers.append("验收阈值尚未固定（应在获得 PPO 波动范围后、正式 SAC 比较前固定）")
    a, b = summarize_runs(baseline), summarize_runs(candidate)
    rows = {}
    for name in a["metrics"]:
        ma, mb = a["metrics"][name], b["metrics"][name]
        if ma["status"] == "missing" or mb["status"] == "missing":
            rows[name] = {"status": "missing"}
            continue
        diff = mb["mean"] - ma["mean"]
        verdict = None
        if thresholds is not None and name in thresholds and not blockers:
            verdict = "通过" if diff <= thresholds[name] else "未通过"
        rows[name] = {"baseline_mean": ma["mean"], "candidate_mean": mb["mean"], "difference": diff, "verdict": verdict}
    return {"budget": budget, "budget_check": budget_check, "metrics": rows, "conclusion_blockers": blockers,
            "conclusion_allowed": not blockers}


def write_report(records: list[EvaluationRecord], path: Path) -> dict:
    """写出评价报告（JSON）：样本量、各指标（含缺失标记）、多次运行的波动、来源与未验证部分。

    缺少来源信息，或不同的有效训练随机起点少于 3 个（与 compare_runs 同一口径，不按记录条数）时，
    report["performance_conclusion"] 标为不发布，并写明原因。
    """
    summary = summarize_runs(records)
    blockers = [f"运行 {i} 缺少来源字段 {missing}" for i, r in enumerate(records) if (missing := missing_provenance(r))]
    seeds = distinct_training_seeds(records)
    if len(seeds) < MIN_RECOMMENDED_SEEDS:
        blockers.append(f"只有 {len(seeds)} 个不同的有效训练随机起点（{len(records)} 份记录），建议至少 {MIN_RECOMMENDED_SEEDS} 个")
    report = {
        "runs": [{"provenance": r.provenance, "episodes": len(r.episodes), "steps": sum(ep.steps for ep in r.episodes),
                  "metrics": {k: _metric_to_dict(v) for k, v in compute_metrics(r).items()}} for r in records],
        "summary": summary,
        "missing_metrics": sorted({k for r in records for k, v in compute_metrics(r).items() if v.status == "missing"}),
        "performance_conclusion": "不发布：" + "；".join(blockers) if blockers else "可在固定阈值下用 compare_runs 判定",
        "unverified": ["真实 PPO / SAC 对照尚未在 Isaac 仿真中运行", "导出模型与原生推理的差异由 T05 单独检查"],
    }
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: None if o is None else str(o)) + "\n",
                          encoding="utf-8")
    return report
