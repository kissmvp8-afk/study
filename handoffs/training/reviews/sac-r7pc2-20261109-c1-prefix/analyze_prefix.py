"""Read verified C1 logs only. No simulator, training, or parameter mutation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODELS = {
    'parent': 'cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3',
    'P0_1401_L48': 'f918d54b8ddeaab05d3b4db0dc012db96e09fd4e223e39c664a1422642c89783',
    'P0_1402_L48': '18e70c5e63a329743794ea6da3ddf59b25ae3e111c0d6e1d0acc04c50413303a',
    'P1_1401_L48': 'c0f1befd1985b99776647850001b3be841173a9fe406b7d6b59630722086eba1',
    'P1_1402_L48': '6b71a242bb92188f835be8ba70b459a7ee271792ebf0127e0a9b7050c2bc27c4',
}


def tilt(row):
    return math.degrees(math.acos(max(-1, min(1, -row['gravity_b'][2]))))


def mean(values):
    return sum(values) / len(values)


def event(rows, predicate):
    for row in rows:
        if predicate(row):
            return {'step_zero_based': row['step'], 't_start_s': row['t_start_s'],
                    't_end_s': row['t_end_s']}
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, inputs, summaries, trajectories, physical, rng = [], [], [], {}, [], []
    for model, sha in MODELS.items():
        for execution in ('D', 'S'):
            folder = args.evidence / 'evaluations' / sha / 'development' / f'standard-{execution}-231' / 'attempt-1'
            for name in ('steps.jsonl', 'configuration.json', 'frozen-before.json', 'frozen-after.json'):
                path = folder / name
                raw = path.read_bytes()
                inputs.append({'path': str(path.relative_to(args.evidence)), 'bytes': len(raw),
                               'sha256': hashlib.sha256(raw).hexdigest()})
            config = json.loads((folder / 'configuration.json').read_text())
            physical.append(config['physical_initial_state_sha256'])
            rng.append(config['rng_identity'])
            assert (folder / 'frozen-before.json').read_bytes() == (folder / 'frozen-after.json').read_bytes()
            assert config['sha256'] == sha and config['evaluation_seed'] == 231
            rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
            expected = 29 if model == 'P1_1401_L48' else 600
            assert len(rows) == expected, (model, execution, len(rows))
            prefix = rows[:29]
            assert all(row['step'] == i and abs(row['t_end_s']-(i+1)*.02) < 1e-9 for i, row in enumerate(prefix))
            assert all(row['effective_command'] == [0., 0., 0.] for row in prefix)
            for row in prefix:
                assert not any(v for group in ('external_force_b_command', 'external_torque_b_command') for vector in row[group] for v in vector)
                record = dict(model=model, execution=execution, seed=231, step=row['step'],
                              t_start_s=row['t_start_s'], t_end_s=row['t_end_s'],
                              tilt_deg=tilt(row), root_z_m=row['root_pos_w'][2],
                              vx_before_mps=row['velocity_true_mps_before'][0],
                              vx_pred_before_mps=row['encoder_velocity_mps'][0],
                              vx_error_before_mps=row['encoder_velocity_mps'][0]-row['velocity_true_mps_before'][0],
                              vx_after_mps=row['measured_body_vx_vy_wz'][0],
                              vy_after_mps=row['measured_body_vx_vy_wz'][1],
                              wz_after_radps=row['measured_body_vx_vy_wz'][2],
                              base_contact_N=row['contact_named_max_N']['base'],
                              left_wheel_contact_N=row['contact_named_max_N']['left_wheel'],
                              right_wheel_contact_N=row['contact_named_max_N']['right_wheel'],
                              left_leg3_contact_N=row['contact_named_max_N']['left_leg_3'],
                              right_leg3_contact_N=row['contact_named_max_N']['right_leg_3'],
                              terminated=row['terminated'])
                for side, i in [('left', 0), ('right', 1)]:
                    record[f'{side}_wheel_target_radps'] = row['wheel_target_rad_s'][i]
                    record[f'{side}_wheel_actual_radps'] = row['wheel_actual_rad_s'][i]
                for j, joint in enumerate(('left1', 'left2', 'left3', 'right1', 'right2', 'right3')):
                    record[f'{joint}_target_rad'] = row['leg_target_rad'][j]
                    record[f'{joint}_actual_rad'] = row['leg_actual_rad'][j]
                records.append(record)
            contact_events = {name: event(prefix, lambda r, n=name: r['contact_named_max_N'][n] > 1)
                              for name in prefix[0]['contact_named_max_N']}
            summary = dict(model=model, execution=execution, seed=231, original_steps=len(rows),
                           prefix_steps=29, prefix_end_s=.58,
                           first_tilt_events={str(d): event(prefix, lambda r, d=d: tilt(r) >= d) for d in (10, 20, 30, 40)},
                           first_contact_events=contact_events,
                           final_prefix_tilt_deg=tilt(prefix[-1]), max_prefix_tilt_deg=max(map(tilt, prefix)),
                           final_prefix_root_z_m=prefix[-1]['root_pos_w'][2],
                           first_targets=dict(leg=prefix[0]['leg_target_rad'], wheel=prefix[0]['wheel_target_rad_s']),
                           last_targets=dict(leg=prefix[-1]['leg_target_rad'], wheel=prefix[-1]['wheel_target_rad_s']),
                           near_bound_steps_19radps=[sum(abs(r['wheel_target_rad_s'][i]) >= 19 for r in prefix) for i in (0, 1)],
                           vx_mae_matching_before_by_prefix={str(n): mean([abs(r['encoder_velocity_mps'][0]-r['velocity_true_mps_before'][0]) for r in prefix[:n]]) for n in (1, 5, 10, 29)},
                           vy_mae_matching_before_by_prefix={str(n): mean([abs(r['encoder_velocity_mps'][1]-r['velocity_true_mps_before'][1]) for r in prefix[:n]]) for n in (1, 5, 10, 29)},
                           vz_mae_matching_before_by_prefix={str(n): mean([abs(r['encoder_velocity_mps'][2]-r['velocity_true_mps_before'][2]) for r in prefix[:n]]) for n in (1, 5, 10, 29)},
                           right_leg2_target_total_variation_rad=sum(abs(a['leg_target_rad'][4]-b['leg_target_rad'][4]) for a, b in zip(prefix, prefix[1:])),
                           right_leg2_actual_at_prefix_end_rad=prefix[-1]['leg_actual_rad'][4],
                           base_contact_peak_N=max(r['contact_named_max_N']['base'] for r in prefix),
                           terminal_at_prefix_end=prefix[-1]['termination_terms'])
            summaries.append(summary)
            trajectories[(model, execution)] = prefix
    assert len(set(physical)) == 1
    assert all(x == rng[0] for x in rng)
    comparisons = []
    for model in list(MODELS)[1:]:
        for execution in ('D', 'S'):
            a, b = trajectories[('parent', execution)][0], trajectories[(model, execution)][0]
            assert a['velocity_true_mps_before'] == b['velocity_true_mps_before']
            comparisons.append(dict(model=model, execution=execution,
                                    leg_target_difference_at_first_action_rad=[y-x for x, y in zip(a['leg_target_rad'], b['leg_target_rad'])],
                                    wheel_target_difference_at_first_action_radps=[y-x for x, y in zip(a['wheel_target_rad_s'], b['wheel_target_rad_s'])]))
    with (args.out / 'prefix-steps.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        w.writeheader(); w.writerows(records)
    output = dict(input='C1 fixed 94ec37af230d8ec0364dece1ee77fa15eacc9eab; C2 raw not yet available',
                  physical_initial_sha256=physical[0], physical_initial_equal=True, rng_initial_equal=True,
                  definition='post-step tilt acos(-g_z); contacts max across last 3 physical samples (10ms span); threshold events are descriptive, not acceptance criteria',
                  prefix_rows=len(records), cases=summaries, first_action_parent_differences=comparisons,
                  limitations=['C2 equality not independently audited from raw files yet', 'Raw noisy policy histories are absent',
                               'Closed-loop trajectories diverge after action0; later comparisons are not common-state interventions',
                               'Control-step contact maxima do not resolve 5ms substep contact ordering',
                               'Near-bound actions and estimation error are correlates, not proved causes'])
    (args.out / 'prefix-summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    (args.out / 'input-files.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    colors = {'parent':'#2166ac', 'P0_1401_L48':'#737373', 'P1_1401_L48':'#d6604d', 'P1_1402_L48':'#1b9e77'}
    for col, execution in enumerate(('D', 'S')):
        for model, color in colors.items():
            rows = trajectories[(model, execution)]
            t = [r['t_end_s'] for r in rows]
            axes[0, col].plot(t, [tilt(r) for r in rows], label=model, color=color)
            axes[1, col].plot(t, [r['leg_target_rad'][4] for r in rows], label=model+' target', color=color)
            if model in ('parent', 'P1_1401_L48'):
                axes[1, col].plot(t, [r['leg_actual_rad'][4] for r in rows], color=color, linestyle='--', alpha=.65)
            axes[2, col].plot([r['t_start_s'] for r in rows], [r['encoder_velocity_mps'][0]-r['velocity_true_mps_before'][0] for r in rows], color=color)
        axes[0, col].set_title(f'seed231 {execution}, first29 control steps')
        axes[0, col].set_ylabel('Body tilt after physics (deg)')
        axes[1, col].set_ylabel('Right leg2 (rad); actual dashed')
        axes[2, col].set_ylabel('vx prediction minus truth\nBEFORE action (m/s)')
        axes[2, col].set_xlabel('Physical time (s)')
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            ax.axvline(.58, color='#d6604d', linestyle=':', alpha=.7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('C1 verified trajectories: serial parent vs candidates; chronology does not prove cause')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(args.out/'prefix-diagnostic.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'cases':len(summaries), 'prefix_rows':len(records), 'physical_initial_equal':True, 'initial_rng_equal':True}))


if __name__ == '__main__':
    main()
