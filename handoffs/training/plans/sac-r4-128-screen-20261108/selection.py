"""SAC-R4只读评价/晋级：从逐步物理数据计算，训练奖励不参与排名。仅标准库。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

DT = .02
DIRECTIONS = tuple('D'+str(i) for i in range(6))
FORKS = (1001, 1002)
SEEDS = (101, 102, 103)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def mean(values):
    return sum(values)/len(values)


def command(protocol, i):
    if protocol == 'pulse' or i < 100:
        return [0., 0., 0.]
    if protocol == 'turn':
        if i < 300: return [.2, 0., .5]
        if i < 400: return [0., 0., 0.]
        if i < 600: return [.2, 0., -.5]
        return [0., 0., 0.]
    if i < 300: return [.3, 0., 0.]
    if i < 400: return [0., 0., 0.]
    return [-.3, 0., 0.]


def close(a, b):
    return len(a) == len(b) and all(abs(x-y) < 1e-7 for x, y in zip(a, b))


def window(rows, start, end):
    part = rows[round(start/DT):round(end/DT)]
    if len(part) != round((end-start)/DT):
        return None
    return dict(vx=mean([r['measured_body_vx_vy_wz'][0] for r in part]),
                vx_mae=mean([abs(r['measured_body_vx_vy_wz'][0]-r['effective_command'][0]) for r in part]),
                wz_mae=mean([abs(r['measured_body_vx_vy_wz'][2]-r['effective_command'][2]) for r in part]),
                wz=mean([r['measured_body_vx_vy_wz'][2] for r in part]),
                abs_wz=mean([abs(r['measured_body_vx_vy_wz'][2]) for r in part]),
                speed=mean([math.hypot(*r['measured_body_vx_vy_wz'][:2]) for r in part]),
                height=mean([r['root_pos_w'][2] for r in part]))


def analyze_rows(rows, protocol='standard', pulse_direction=None):
    require(protocol in ('standard', 'turn', 'pulse'), 'Unknown protocol')
    limit = 600 if protocol == 'standard' else 900
    require(0 < len(rows) <= limit and finite(rows), 'Empty, oversized or nonfinite evidence')
    for i, r in enumerate(rows):
        require(r['step'] == i and abs(r['t_start_s']-i*DT)<1e-8 and
                abs(r['t_end_s']-(i+1)*DT)<1e-8, 'Noncontiguous physical samples')
        require(close(r['effective_command'], command(protocol, i)), 'Wrong command schedule')
        require(len(r['measured_body_vx_vy_wz']) == 3 and len(r['wheel_target_rad_s']) == 2,
                'Wrong velocity/action shape')
        require(isinstance(r['terminated'], bool) and isinstance(r['time_out'], bool), 'Invalid done flags')
        require(r['termination_terms'] == {'base_contact': r['terminated'], 'time_out': r['time_out']},
                'Termination terms must be current returned events, not retained get_term values')
        if i < len(rows)-1:
            require(not r['terminated'] and not r['time_out'], 'Samples continued after first done')
        require(all(v == 0 for body in r['external_torque_b_command'] for v in body), 'Unexpected torque')
        if protocol != 'pulse':
            require(all(v == 0 for body in r['external_force_b_command'] for v in body), 'Unexpected force')
    # A shortened file is a valid failed evaluation only when the actual final transition ends it.
    require(len(rows) == limit or rows[-1]['terminated'] or rows[-1]['time_out'], 'Truncated upload/evaluation')
    intact = len(rows) == limit and not any(r['terminated'] or r['time_out'] for r in rows)
    base_max = max(r['contact_named_max_N']['base'] for r in rows)
    survive = intact and base_max <= 1.
    duration = limit*DT
    metrics = {name: window(rows, a, b) for name, a, b in (
        ('forward', 2, 6), ('reverse', 8, 12), ('stop1', 1, 2),
        ('stop2', 7, 8), ('height', 2, duration))}
    if protocol == 'turn': metrics['stop3'] = window(rows, 17, 18)
    changes = [mean([abs(x-y)/DT for x, y in zip(b['wheel_target_rad_s'], a['wheel_target_rad_s'])])
               for a, b in zip(rows, rows[1:])]
    wheel_rate = mean(changes) if changes else None
    common = survive and metrics['height'] is not None and metrics['height']['height'] >= .25
    passed = competent = False
    loss = 5.  # A decision penalty; missing physical measurements above remain null.
    if intact and protocol in ('standard', 'turn'):
        f, r, s1, s2 = (metrics[k] for k in ('forward', 'reverse', 'stop1', 'stop2'))
        stops = [s1, s2] if protocol == 'standard' else [s2, metrics['stop3']]
        parked = all(s['speed'] <= .05 and s['abs_wz'] <= .10 for s in stops)
        if protocol == 'standard':
            passed = common and f['vx_mae'] <= .12 and r['vx_mae'] <= .12 and parked and max(f['abs_wz'], r['abs_wz']) <= .20
            competent = survive and f['vx'] >= .10 and r['vx'] <= -.10 and max(f['vx_mae'], r['vx_mae']) <= .20
            clip = lambda x: min(5., max(0., x))
            loss = (.45*clip(mean([f['vx_mae'], r['vx_mae']])/.12) +
                    .20*clip(max(s['speed'] for s in stops)/.05) +
                    .15*clip(max(s['abs_wz'] for s in stops)/.10) +
                    .15*clip(mean([f['abs_wz'], r['abs_wz']])/.20) +
                    .05*clip(wheel_rate/200.))
        else:
            passed = common and parked and max(f['vx_mae'], r['vx_mae']) <= .12 and max(f['wz_mae'], r['wz_mae']) <= .15 and f['wz'] > 0 and r['wz'] < 0
    recovery_delay = None
    if protocol == 'pulse':
        vectors = {'x+': [20.,0.,0.], 'x-': [-20.,0.,0.], 'y+': [0.,20.,0.], 'y-': [0.,-20.,0.]}
        require(pulse_direction in vectors, 'Missing pulse direction')
        substeps = [s for row in rows for s in row['pulse_substeps']]
        require(len(substeps) == 4*len(rows), 'Missing pulse physics samples')
        for i, s in enumerate(substeps):
            active = 800 <= i < 840
            require(s['substep'] == i and abs(s['t_start_s']-.005*i)<1e-8 and
                    close(s['base_force_b_N'], vectors[pulse_direction] if active else [0.,0.,0.]) and
                    close(s['base_torque_b_Nm'], [0.,0.,0.]) and s['other_body_force_abs_max'] == 0 and
                    s['all_torque_abs_max'] == 0 and s['is_global'] is False and
                    s['submission_enabled'] == active, 'Wrong pulse submission')
        eligible = [r for r in rows if r['t_end_s'] >= 4.2-1e-8]
        for i in range(max(0, len(eligible)-50)):
            part = eligible[i:i+51]
            if all(math.hypot(*r['measured_body_vx_vy_wz'][:2]) <= .05 and abs(r['measured_body_vx_vy_wz'][2]) <= .10 for r in part):
                recovery_delay = part[0]['t_end_s']-4.2
                break
        passed = common and recovery_delay is not None and recovery_delay <= 2.+1e-8
    return dict(steps=len(rows), survived=survive, task_pass=bool(passed), motion_competent=bool(competent),
                loss=loss, loss_is_failure_penalty=not intact, base_contact_max_N=base_max,
                metrics=metrics, wheel_target_change_rate_rad_s2=wheel_rate, recovery_delay_s=recovery_delay)


def expected_counters(learning):
    return dict(iterations=16+learning, env_steps=8*(16+learning), transitions=1024*(16+learning),
                critic_updates=128*learning, actor_updates=64*learning, alpha_updates=128*learning,
                target_updates=128*learning, learning_skipped_iterations=0)


def rank_direction(direction, cases):
    groups = [[c for c in cases if c['fork_seed'] == seed] for seed in FORKS]
    require(all(len(g) == 6 for g in groups), 'Each direction requires both forks and six cases per fork')
    fraction = lambda group, key: mean([float(c[key]) for c in group])
    worst_survival = min(fraction(g, 'survived') for g in groups)
    worst_pass = min(fraction(g, 'task_pass') for g in groups)
    all_pass = fraction(cases, 'task_pass')
    worst_motion = min(fraction(g, 'motion_competent') for g in groups)
    all_motion = fraction(cases, 'motion_competent')
    losses = sorted(c['loss'] for c in cases)
    p90 = losses[math.ceil(.90*len(losses))-1]
    avg = mean(losses)
    key = [-worst_survival, -worst_pass, -all_pass, -worst_motion, -all_motion, p90, avg, DIRECTIONS.index(direction)]
    return dict(direction=direction, sort_key=key, worst_fork_survival=worst_survival,
                worst_fork_pass=worst_pass, overall_pass=all_pass, worst_fork_motion=worst_motion,
                overall_motion=all_motion, p90_loss=p90, mean_loss=avg)


def select(ranked, round_number):
    ordered = sorted(ranked, key=lambda r: r['sort_key'])
    require(round_number in (1, 2, 3, 4), 'Bad round')
    if round_number == 1:
        top = [r['direction'] for r in ordered if r['direction'] != 'D0'][:2]
        if any(r['direction'] == 'D0' for r in ordered): top.append('D0')
        else: top = [r['direction'] for r in ordered[:3]]
    else:
        top = [r['direction'] for r in ordered[:{2:2, 3:1, 4:1}[round_number]]]
    return ordered, top


def assess_manifest(manifest_path):
    path = Path(manifest_path).resolve(); root = path.parent
    data = json.loads(path.read_text()); q = data['q']; stage = data['round']
    require(isinstance(q, int) and q in range(64, 2049, 16), 'Invalid calibrated q')
    require(stage in (1,2,3,4), 'Invalid round')
    learning = [q, 5*q//2, 4*q, 6*q][stage-1]
    extra = data.get('extra_champion_chunks', 0)
    require(isinstance(extra, int) and 0 <= extra <= 4 and (extra == 0 or stage == 4), 'Invalid extra chunk')
    learning += extra*q//2
    require(data['plan_id'] == 'SAC-R4-128-SCREEN-20261108', 'Plan mismatch')
    # Controller places each round manifest in the task root; all inputs stay below this root.
    def local(relative):
        p = (root/relative).resolve()
        require(p.is_relative_to(root), 'Evidence path escapes task root')
        return p
    def read(relative): return json.loads(local(relative).read_text())
    parameters = read('parameters.json')
    common_keys = ('num_envs','steps_per_macro','warmup_macros','capacity_per_env',
                   'critic_updates_per_learning_macro','actor_updates_per_learning_macro',
                   'alpha_updates_per_learning_macro','target_updates_per_learning_macro',
                   'auxiliary_attempts_per_learning_macro','microcycles','critic_updates_per_microcycle',
                   'auxiliary_attempts_per_microcycle','batch_size','auxiliary_batch_size',
                   'gamma','n_steps','tau','critic_lr','actor_lr','alpha_lr','auxiliary_lr',
                   'standing_command_fraction','policy_observation_noise','reset_force_N','reset_torque_Nm',
                   'interval_push','physics_dt_s','decimation','control_dt_s','tf32_matmul','tf32_cudnn')
    grouped = {}; seen = set(); initial = {}; all_cases = []
    for arm in data['arms']:
        d = arm['direction']; seed = arm['fork_seed']
        require(d in DIRECTIONS and seed in FORKS and (d,seed) not in seen, 'Duplicate/unknown arm')
        seen.add((d,seed)); require(arm['learning_macros'] == learning, 'Unequal learning budget')
        for k, v in expected_counters(learning).items(): require(arm['counters'][k] == v, 'Counter mismatch: '+k)
        require(arm['counters']['auxiliary_updates']+arm['counters']['auxiliary_skipped'] == 128*learning, 'Auxiliary attempts mismatch')
        actual = hashlib.sha256(local(arm['model_path']).read_bytes()).hexdigest()
        require(actual == arm['model_sha256'], 'Checkpoint hash mismatch')
        require(arm['settings_verified'] is True, 'Missing server-side effective setting checks')
        effective = read(arm['effective_settings_path'])
        for key in common_keys:
            require(effective[key] == parameters['common'][key], 'Effective setting mismatch: '+key)
        declared = next(r for r in parameters['directions'] if r['id'] == d)
        for key in ('alpha_entry_multiplier','track_ang_vel_z_exp_weight','parking_body_velocity_weight'):
            require(effective[key] == declared[key], 'Direction setting mismatch: '+key)
        cases = arm['cases']; require(len(cases) == 6, 'Missing development cases')
        require({(c['execution'],c['eval_seed']) for c in cases} == {(e,s) for e in ('D','S') for s in SEEDS}, 'Wrong/duplicate development seeds')
        for case in cases:
            cfg = read(case['configuration_path'])
            for k, v in dict(sha256=actual, execution=case['execution'], evaluation_seed=case['eval_seed'],
                             condition='standard', policy_noise='on', zero_reset_wrench=True,
                             interval_push_disabled=True, tf32_matmul=False, tf32_cudnn=False,
                             termination_event_source='current_compute_return').items():
                require(cfg[k] == v, 'Evaluation protocol mismatch: '+k)
            require(read(case['frozen_before_path']) == read(case['frozen_after_path']), 'Evaluation changed frozen state')
            fingerprint = cfg['physical_initial_state_sha256']; require(len(fingerprint)==64, 'Missing initial physics fingerprint')
            pair = (case['execution'],case['eval_seed'])
            if pair in initial: require(initial[pair] == fingerprint, 'Physical evaluation starts not paired')
            initial[pair] = fingerprint
            rows = [json.loads(line) for line in local(case['steps_path']).read_text().splitlines() if line.strip()]
            result = analyze_rows(rows) | dict(direction=d, fork_seed=seed, execution=case['execution'],eval_seed=case['eval_seed'])
            grouped.setdefault(d, []).append(result); all_cases.append(result)
    require(len(seen) == 2*len(grouped), 'A direction has only one continuation fork')
    require(set(data['expected_directions']) == set(grouped), 'Missing scheduled direction; incomplete round cannot promote')
    ranked, promoted = select([rank_direction(d,c) for d,c in grouped.items()], stage)
    return dict(plan_id=data['plan_id'], round=stage, q=q, common_learning_macros=learning,
                ranking=ranked, promoted=promoted, baseline_reserved=stage==1 and 'D0' in promoted,
                cases=all_cases, input_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                caveat='Development selection only; no independent training-seed or real-robot claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = assess_manifest(args.manifest)
    output = Path(args.output); output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('round','q','promoted')}, ensure_ascii=False))
