"""独立复核R1的JSON/CSV证据；只读取本地文件，不训练、不加载模型、不启动仿真。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-root', type=Path, default=ROOT / 'merged')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'review')
args = parser.parse_args()
DATA = args.evidence_root.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(exist_ok=True)


def read(path):
    """读取单份JSON证据。"""
    return json.loads(path.read_text())


def save(name, value):
    """把复算结果写入指定输出目录，拒绝非有限JSON数值。"""
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_save(name, rows):
    """保存逐阶段或逐运行的独立计算表。"""
    with (OUT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected(i, protocol):
    """按控制步和协议返回机身速度指令，单位为m/s和rad/s。"""
    if protocol == 'P':
        return [0, 0, 0]
    if protocol == 'turn':
        return [0.2, 0, 0.5] if 100 <= i < 300 else ([0.2, 0, -0.5] if 400 <= i < 600 else [0, 0, 0])
    return [0.3, 0, 0] if 100 <= i < 300 else ([-0.3, 0, 0] if i >= 400 else [0, 0, 0])


stages = []
runs = []
discrepancies = []
for file in sorted(DATA.glob('results/**/steps.jsonl')):
    rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
    directory = file.parent
    name = directory.name
    protocol = 'P' if name.startswith('ppo_P') else ('turn' if name.startswith('ppo_turn') else 'standard')
    summary = read(directory / 'summary.json')
    assert summary['steps'] == len(rows), name
    for i, row in enumerate(rows):
        assert row['step'] == i, name
        assert abs(row['t_start_s'] - i * .02) < 1e-7, name
        assert abs(row['t_end_s'] - (i + 1) * .02) < 1e-7, name
        assert np.allclose(row['effective_command'], expected(i, protocol), atol=1e-7, rtol=0), name
        for key in ('measured_body_vx_vy_wz', 'root_pos_w', 'gravity_b', 'actor_action'):
            assert np.isfinite(row[key]).all(), (name, key)
        assert not (row['terminated'] or row['time_out']) or i == len(rows) - 1, name
    windows = [('settle', 0, 2), ('stable_2_10', 2, 10), ('stable_10_18', 10, 18), ('last_1s', 17, 18)] if protocol == 'P' else [
        ('idle_before', 0, 2), ('turn_left' if protocol == 'turn' else 'forward', 2, 6),
        ('idle_between', 6, 8), ('turn_right' if protocol == 'turn' else 'reverse', 8, 12),
        ('idle_before_tail', 1, 2), ('idle_between_tail', 7, 8), ('stable_all_after_2', 2, 18 if protocol == 'turn' else 12)]
    if protocol == 'turn':
        windows += [('idle_after', 12, 18), ('last_1s', 17, 18)]
    supplied = {r['stage']: r for r in summary.get('stage_metrics', [])}
    own = {}
    for stage, start, end in windows:
        rr = rows[round(start * 50):round(end * 50)]
        complete = len(rr) == round((end - start) * 50)
        if not rr:
            continue
        v = np.array([r['measured_body_vx_vy_wz'] for r in rr])
        cmd = np.array([r['effective_command'] for r in rr])
        pos = np.array([r['root_pos_w'] for r in rr])
        tilt = np.arccos(np.clip([-r['gravity_b'][2] for r in rr], -1, 1))
        m = dict(run=name, source=str(file.relative_to(DATA)), stage=stage, samples=len(rr), complete=complete,
                 mean_vx=float(v[:, 0].mean()), mean_vy=float(v[:, 1].mean()), mean_wz=float(v[:, 2].mean()),
                 vx_mae=float(abs(v[:, 0] - cmd[:, 0]).mean()), wz_mae=float(abs(v[:, 2] - cmd[:, 2]).mean()),
                 mean_planar_speed=float(np.linalg.norm(v[:, :2], axis=1).mean()), mean_abs_wz=float(abs(v[:, 2]).mean()),
                 mean_height_m=float(pos[:, 2].mean()), minimum_height_m=float(pos[:, 2].min()),
                 maximum_tilt_rad=float(tilt.max()))
        if all('root_com_pos_w_before' in r for r in rr):
            before = np.array([r['root_com_pos_w_before'] for r in rr])
            after = np.array([r['root_com_pos_w'] for r in rr])
            m.update(net_drift_m=float(np.linalg.norm(after[-1, :2] - before[0, :2])),
                     path_length_m=float(np.linalg.norm(after[:, :2] - before[:, :2], axis=1).sum()))
        else:
            m.update(net_drift_m=None, path_length_m=None)
        for key, val in m.items():
            if key in supplied.get(stage, {}) and isinstance(val, (float, int)) and not isinstance(val, bool):
                other = supplied[stage][key]
                if other is not None and abs(other - val) > 1e-6:
                    discrepancies.append(dict(run=name, stage=stage, key=key, computed=val, supplied=other))
        own[stage] = m
        stages.append(m)
    base = [r.get('contact_named_max_N', {}).get('base', r.get('contact_force_history_max_N', [float('inf')])[0]) for r in rows]
    gate = None
    if protocol == 'standard':
        gate = len(rows) == 600 and not any(r['terminated'] or r['time_out'] for r in rows)
        gate = gate and all(own.get(s, {}).get('complete', False) for s in ('idle_before', 'forward', 'idle_between', 'reverse'))
        gate = gate and all(own[s]['vx_mae'] <= .12 and own[s]['mean_abs_wz'] <= .2 for s in ('forward', 'reverse'))
        gate = gate and own['idle_between_tail']['mean_planar_speed'] <= .05 and own['stable_all_after_2']['mean_height_m'] >= .25 and max(base) <= 1
    runs.append(dict(run=name, source=str(directory.relative_to(DATA)), steps=len(rows), seconds=len(rows)/50,
                     terminations=sum(r['terminated'] for r in rows), timeouts=sum(r['time_out'] for r in rows),
                     max_base_contact_N=max(base), gate_pass=gate, raw_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    if name == 'S1_1048576_S_101':
        timeline = []
        for r in rows:
            timeline.append(dict(step=r['step'], t=r['t_end_s'], height=r['root_pos_w'][2],
                                 tilt=math.acos(max(-1, min(1, -r['gravity_b'][2]))),
                                 vx=r['measured_body_vx_vy_wz'][0], vy=r['measured_body_vx_vy_wz'][1], wz=r['measured_body_vx_vy_wz'][2],
                                 base_contact_N=r['contact_named_max_N']['base'],
                                 left_wheel=r['wheel_target_rad_s'][0], right_wheel=r['wheel_target_rad_s'][1],
                                 mean_latent_std=sum(r['actor_std'])/8,
                                 max_leg_contact_N=max(r['contact_named_max_N']['left_leg_3'], r['contact_named_max_N']['right_leg_3']),
                                 terminated=r['terminated']))
        csv_save('s1-early-termination-timeline.csv', timeline)

csv_save('independent-stage-metrics.csv', stages)
csv_save('independent-runs.csv', runs)
save('metric-discrepancies.json', discrepancies)
training = []
for arm in ('S0', 'S1'):
    rows = list(csv.DictReader((DATA / f'results/training/{arm}/iterations.csv').open()))
    assert len(rows) == 8192
    for i, row in enumerate(rows, 1):
        assert int(row['iteration']) == i and int(row['transitions']) == 128*i and int(row['env_steps']) == 8*i
        for key in ('critic_updates', 'alpha_updates', 'target_updates', 'auxiliary_updates'):
            assert int(row[key]) == max(0, i-128)*16, (arm, i, key)
        assert int(row['actor_updates']) == max(0, i-128)*8
        assert int(row['auxiliary_skipped']) == 0
    for start, end in ((129, 228), (1949, 2048), (3997, 4096), (8093, 8192)):
        selected = rows[start-1:end]
        m = dict(arm=arm, start_iteration=start, end_iteration=end)
        for key in ('reward_mean', 'alpha', 'critic_loss', 'q1_mean', 'q2_mean', 'target_mean', 'terminated', 'time_out', 'iteration_seconds'):
            m[key] = float(np.mean([float(r[key]) for r in selected]))
        training.append(m)
csv_save('training-windows.csv', training)
physical = ('root_pos_w', 'root_com_pos_w', 'root_quat_w', 'root_lin_vel_w', 'root_ang_vel_w', 'joint_pos', 'joint_vel', 'material_properties', 'masses', 'inertias', 'coms', 'explicit_actuators')
pairs = {}
for seed in (101, 102, 103):
    base = read(DATA / f'results/diagnostics/ppo_P00_{seed}/configuration.json')
    pairs[str(seed)] = {}
    for condition in ('P10', 'P01', 'P11'):
        other = read(DATA / f'results/diagnostics/ppo_{condition}_{seed}/configuration.json')
        pairs[str(seed)][condition] = {k: base['initial_state'][k] == other['initial_state'][k] for k in physical}
        pairs[str(seed)][condition]['reset_wrench_pre_intervention_match'] = base['reset_wrench_before_intervention'] == other['reset_wrench_before_intervention']
save('paired-physical-start-check.json', pairs)
save('audit-summary.json', dict(runs=len(runs), total_steps=sum(r['steps'] for r in runs),
                              metric_discrepancies=len(discrepancies), training_rows_verified=16384,
                              all_ppo_physical_pairs_match=all(all(v.values()) for p in pairs.values() for v in p.values()),
                              models_deserialized=False, simulator_executed=False))
print((OUT / 'audit-summary.json').read_text())
