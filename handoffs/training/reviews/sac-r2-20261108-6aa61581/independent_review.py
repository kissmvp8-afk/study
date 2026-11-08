"""只读复算SAC-R2原始评价和训练日志，不执行附件脚本、反序列化权重或启动仿真。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-root', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
DATA = args.evidence_root
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def table(name, rows):
    with (OUT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


stages, runs, differences, failures = [], [], [], []
windows = [('idle_before', 0, 2), ('forward', 2, 6), ('idle_between', 6, 8), ('reverse', 8, 12),
           ('idle_before_tail', 1, 2), ('idle_between_tail', 7, 8), ('stable_all_after_2', 2, 12)]
for file in sorted(DATA.glob('runs/**/steps.jsonl')):
    name = file.parent.name
    phase = file.parent.parent.name
    rows = [json.loads(line) for line in file.open()]
    summary = read(file.parent / 'summary.json')
    cfg = read(file.parent / 'configuration.json')
    assert len(rows) == summary['steps']
    for i, r in enumerate(rows):
        command = [.3, 0, 0] if 100 <= i < 300 else ([-.3, 0, 0] if i >= 400 else [0, 0, 0])
        assert r['step'] == i and abs(r['t_start_s'] - i*.02) < 1e-8 and abs(r['t_end_s'] - (i+1)*.02) < 1e-8
        assert np.allclose(r['effective_command'], command, atol=1e-7, rtol=0)
        for key in ('measured_body_vx_vy_wz', 'gravity_b', 'root_pos_w', 'actor_action', 'actor_std'):
            assert np.isfinite(r[key]).all(), (name, key)
        assert not (r['terminated'] or r['time_out']) or i == len(rows)-1
        if cfg.get('zero_reset_wrench'):
            assert np.count_nonzero(r['external_force_b_command']) == np.count_nonzero(r['external_torque_b_command']) == 0
    supplied = {r['stage']: r for r in summary['stage_metrics']}
    own = {}
    for stage, start, end in windows:
        rr = rows[start*50:end*50]
        if not rr:
            continue
        v = np.array([r['measured_body_vx_vy_wz'] for r in rr])
        c = np.array([r['effective_command'] for r in rr])
        before = np.array([r['root_com_pos_w_before'] for r in rr])
        after = np.array([r['root_com_pos_w'] for r in rr])
        pos = np.array([r['root_pos_w'] for r in rr])
        tilt = np.arccos(np.clip([-r['gravity_b'][2] for r in rr], -1, 1))
        m = dict(phase=phase, run=name, stage=stage, samples=len(rr), complete=len(rr)==(end-start)*50,
                 mean_vx=float(v[:, 0].mean()), mean_vy=float(v[:, 1].mean()), mean_wz=float(v[:, 2].mean()),
                 vx_mae=float(abs(v[:, 0]-c[:, 0]).mean()), wz_mae=float(abs(v[:, 2]-c[:, 2]).mean()),
                 mean_abs_wz=float(abs(v[:, 2]).mean()), mean_planar_speed=float(np.linalg.norm(v[:, :2], axis=1).mean()),
                 mean_height_m=float(pos[:, 2].mean()), minimum_height_m=float(pos[:, 2].min()), maximum_tilt_rad=float(tilt.max()),
                 net_drift_m=float(np.linalg.norm(after[-1, :2]-before[0, :2])),
                 path_length_m=float(np.linalg.norm(after[:, :2]-before[:, :2], axis=1).sum()))
        for key in ('actor_action', 'leg_target_rad', 'wheel_target_rad_s'):
            a = np.array([r[key] for r in rr])
            target_key = dict(actor_action='actor_action_rate_per_s', leg_target_rad='leg_target_rate_rad_s', wheel_target_rad_s='wheel_target_rate_rad_s2')[key]
            m[target_key] = float(abs(np.diff(a, axis=0)).mean()/.02) if len(rr)>1 else None
        for key, val in m.items():
            other = supplied.get(stage, {}).get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and isinstance(other, (int, float)):
                if abs(val-other) > 1e-6:
                    differences.append(dict(run=name, stage=stage, key=key, actual=val, supplied=other))
        own[stage] = m
        stages.append(m)
    base_max = max(r['contact_named_max_N']['base'] for r in rows)
    gate = len(rows)==600 and not any(r['terminated'] or r['time_out'] for r in rows)
    gate = gate and all(own[k]['complete'] and own[k]['vx_mae']<=.12 and own[k]['mean_abs_wz']<=.2 for k in ('forward', 'reverse'))
    gate = gate and own['idle_between_tail']['mean_planar_speed']<=.05 and own['stable_all_after_2']['mean_height_m']>=.25 and base_max<=1
    assert gate == summary['development_gate_pass']
    frozen = read(file.parent/'frozen-before.json') == read(file.parent/'frozen-after.json')
    assert frozen
    runs.append(dict(phase=phase, run=name, execution=cfg['execution'], seed=cfg['evaluation_seed'], steps=len(rows), seconds=len(rows)/50,
                     true_terminations=sum(r['terminated'] for r in rows), timeouts=sum(r['time_out'] for r in rows),
                     base_contact_max_N=base_max, gate_pass=gate, frozen_state_equal=frozen,
                     model_sha256=cfg['sha256'], steps_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    if any(r['terminated'] for r in rows):
        r=rows[-1]
        failures.append(dict(phase=phase, run=name, seconds=len(rows)/50, base_contact_N=base_max,
                             last_height_m=r['root_pos_w'][2], last_tilt_rad=math.acos(max(-1, min(1, -r['gravity_b'][2]))),
                             maximum_planar_speed=max(math.hypot(*r['measured_body_vx_vy_wz'][:2]) for r in rows),
                             last_termination_terms=r['termination_terms']))
table('independent-stage-metrics.csv', stages)
table('independent-runs.csv', runs)
save('metric-discrepancies.json', differences)
save('failure-events.json', failures)

training, diagnostics = [], []
for arm in ('R20', 'R21'):
    rows=list(csv.DictReader((DATA/f'runs/phase-b-001/{arm}/iterations.csv').open()))
    assert len(rows)==8192
    for i,r in enumerate(rows, 1):
        assert int(r['iteration'])==i and int(r['transitions'])==i*128 and int(r['env_steps'])==i*8
        for key in ('critic_updates', 'alpha_updates', 'target_updates', 'auxiliary_updates'):
            assert int(r[key])==max(0, i-128)*16
        assert int(r['actor_updates'])==max(0, i-128)*8 and int(r['auxiliary_skipped'])==0
    compact=[]
    term_disagreements=0
    for i,line in enumerate((DATA/f'runs/phase-b-001/{arm}/iterations.jsonl').open()):
        d=json.loads(line)
        rd=d['r2_diagnostics']
        assert rd['samples']==128 and rd['zero_wrench_checked_each_control_step_and_reset']
        expected_weight=0 if arm=='R20' else -2
        assert rd['parking_weight']==expected_weight
        assert abs(rd['parking_weighted_dt_mean']-rd['parking_unweighted_mean']*expected_weight*.02)<1e-7
        commands=np.array(rd['command_distribution']['all_samples'])
        mask=(np.linalg.norm(commands[:, :2],axis=1)<.05)&(abs(commands[:, 2])<.05)
        assert abs(mask.mean()-rd['parking_gate_ratio'])<1e-7
        assert abs((commands==0).all(axis=1).mean()-rd['exact_zero_command_ratio'])<1e-7
        assert rd['physical_wheel_target_rad_s']['abs_max']<=20.0001
        mismatch=d['collect']['time_out']!=rd['termination_counts']['time_out'] or d['collect']['terminated']!=rd['termination_counts']['base_contact']
        term_disagreements+=mismatch
        entropy=d['action_entropy'].get('mean')
        c=rows[i]
        compact.append(dict(arm=arm, iteration=i+1, reward_mean=float(c['reward_mean']), alpha=float(c['alpha']) if c['alpha'] else None,
                            entropy=entropy, internal_std_mean=d['exploration']['internal_std_mean'],
                            parking_gate_ratio=rd['parking_gate_ratio'], exact_zero_command_ratio=rd['exact_zero_command_ratio'],
                            parking_weighted_dt_mean=rd['parking_weighted_dt_mean'],
                            wheel_near_bound_ratio=rd['physical_wheel_target_rad_s']['near_bound_95pct_ratio'],
                            terminations=d['collect']['terminated'], timeouts=d['collect']['time_out']))
    table(f'{arm}-compact-training.csv', compact)
    for start,end in ((129,228),(1949,2048),(3997,4096),(8093,8192)):
        selected=compact[start-1:end]
        m=dict(arm=arm, start_iteration=start, end_iteration=end)
        for key in selected[0]:
            if key not in ('arm','iteration'):
                m[key]=float(np.mean([x[key] for x in selected]))
        training.append(m)
    diagnostics.append(dict(arm=arm, iterations_checked=len(rows), current_termination_count_disagreement_iterations=term_disagreements,
                            reward_weight_dt_identity_pass=True, parking_command_mask_pass=True))
table('training-windows.csv',training)
save('training-diagnostic-audit.json',diagnostics)
physical=('root_pos_w','root_com_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel','material_properties','masses','inertias','coms','explicit_actuators')
pairs=[]
for arm,seed,execution in [(a,s,'D') for a in ('S0','S1') for s in (101,102,103)]+[('S1',101,'S')]:
    base=read(DATA/f'runs/phase-a-001/{arm}_F00_{seed}_{execution}/configuration.json')
    for condition in ('F10','F01','F11'):
        other=read(DATA/f'runs/phase-a-001/{arm}_{condition}_{seed}_{execution}/configuration.json')
        matches={key:base['initial_state'][key]==other['initial_state'][key] for key in physical}
        matches['reset_wrench_pre_intervention']=base['reset_wrench_before_intervention']==other['reset_wrench_before_intervention']
        pairs.append(dict(arm=arm,seed=seed,execution=execution,condition=condition,matches=matches))
save('paired-physical-starts.json',pairs)
assert all(all(p['matches'].values()) for p in pairs)
summary=dict(raw_runs=len(runs), raw_steps=sum(r['steps'] for r in runs), metric_discrepancies=len(differences),
             training_csv_rows=16384, training_jsonl_rows=16384, physical_pair_groups=7, all_physical_pairs_match=True,
             models_deserialized=False, simulation_run=False,
             diagnostic_termination_counts_reliable=all(d['current_termination_count_disagreement_iterations']==0 for d in diagnostics))
save('audit-summary.json', summary)
print(json.dumps(summary,indent=2))
