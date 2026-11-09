"""只读复算 L24 单 seed D 探索；与 C1 的固定来源逐项核对，不修改原始成绩。"""
import hashlib
import json
import math
import sys
from pathlib import Path

BASE = Path('/workspace/sac-r7pc2-review')
C1 = Path('/workspace/sac-r7pc1-review/evidence')
RAW = BASE / 'L24-evidence'
sys.path.insert(0, str(C1 / 'evaluation'))
from r4_protocol import analyze_rows


def read(p):
    return json.loads(p.read_text())


def tilt(r):
    return math.degrees(math.acos(max(-1., min(1., -r['gravity_b'][2]))))


def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            compare(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    elif isinstance(a, float):
        assert abs(a-b) < 1e-12, (a, b)
    else:
        assert a == b, (a, b)


parent = C1 / 'evaluations/cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3/development/standard-D-231/attempt-1'
parent_config = read(parent/'configuration.json')
remote = read(RAW/'exploratory-results.json')
native = {k:v for k,v in parent_config['source_identity'].items() if '/study/' in k}
models = {x['sha256']:x for x in read(BASE/'L24-verification.json')}
recomputed = []
traces = {}
for folder in sorted((RAW/'cases').iterdir()):
    cfg = read(folder/'configuration.json')
    rows = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    result = analyze_rows(rows, 'standard')
    original = next(x for x in remote['cases'] if x['model']['sha256'] == cfg['sha256'])
    compare(result, original['metrics'])
    assert len(rows) == 600 and cfg['execution'] == 'D' and cfg['evaluation_seed'] == 231
    assert read(folder/'frozen-before.json') == read(folder/'frozen-after.json')
    assert cfg['initial_state'] == parent_config['initial_state']
    assert cfg['rng_identity'] == parent_config['rng_identity']
    assert {k:v for k,v in cfg['source_identity'].items() if '/study/' in k} == native
    assert cfg['policy_noise'] == 'on' and cfg['zero_reset_wrench'] and cfg['interval_push_disabled']
    assert not cfg['tf32_matmul'] and not cfg['tf32_cudnn']
    assert cfg['sha256'] in models and cfg['checkpoint_counters'] == models[cfg['sha256']]['counters']
    assert all(r['actor_std'] for r in rows)
    prefix = rows[:29]
    result.update(model=folder.name, sha256=cfg['sha256'],
                  prefix29_max_tilt_deg=max(map(tilt,prefix)),
                  prefix29_end_tilt_deg=tilt(prefix[-1]),
                  prefix29_right_leg2_target_TV_rad=sum(abs(a['leg_target_rad'][4]-b['leg_target_rad'][4]) for a,b in zip(prefix,prefix[1:])),
                  startup_initial_state_equals_C1=True, initial_RNG_equals_C1=True,
                  checkpoint_counters_match_actual_download=True,
                  archive_steps_sha256=hashlib.sha256((folder/'steps.jsonl').read_bytes()).hexdigest())
    recomputed.append(result)
    traces[folder.name] = rows

source_checks = {}
for name in ('evaluate_sac_r7p.py','evaluation_r7p_helpers.py','evaluate_with_kit_repair.py',
             'kit_command_repair.py','r4_protocol.py','r7p_evaluation.py'):
    old = C1/'evaluation'/name
    new = next((RAW/'archive-source-evidence').glob('*-'+name))
    assert old.read_bytes() == new.read_bytes(), name
    source_checks[name] = hashlib.sha256(old.read_bytes()).hexdigest()

output = dict(fixed_commit='32f519378e1d55173ceca46984d26ee19d0eb0de',
              result_path='results/tita-sac/sac-r7p-l24-explore-20261109',
              original_C1_commit='94ec37af230d8ec0364dece1ee77fa15eacc9eab',
              independently_recomputed_cases=4, physical_steps=2400,
              mode='four_lane', executions=['D'], evaluation_seeds=[231],
              metrics_match_frozen_report=True, same_key_source_SHA256=source_checks,
              source_and_initial_conditions_match=True, cases=recomputed,
              no_new_training_or_simulation_by_reviewer=True,
              interpretation='P1/1401 L24 D survives600 while L48 D fails29 at same seed/initial/RNG/source. This brackets this deterministic case regression between savedL24 andL48; stochastic L24 and root cause remain untested.',
              does_not_establish=['unseen-seed acceptance','precision reward causality','safe deployable L24 policy','serial/four-lane equivalence for L24'])
(BASE/'audit/L24-independent-review.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'cases':4,'steps':2400,'all_reports_equal':True,'survived':sum(x['survived'] for x in recomputed),'passed':sum(x['task_pass'] for x in recomputed)},ensure_ascii=False))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
bad=C1/'evaluations/c0f1befd1985b99776647850001b3be841173a9fe406b7d6b59630722086eba1/development/standard-D-231/attempt-1/steps.jsonl'
plot_traces={'parent':([json.loads(s) for s in (parent/'steps.jsonl').read_text().splitlines()][:29],'#2166ac'),
             'P1/1401 L24':(traces['P1-1401-L24-231-D'][:29],'#1b9e77'),
             'P1/1401 L48':([json.loads(s) for s in bad.read_text().splitlines()],'#d6604d')}
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for label,(rows,color) in plot_traces.items():
    t=[r['t_end_s'] for r in rows]
    axes[0,0].plot(t,list(map(tilt,rows)),color=color,label=label)
    axes[0,1].plot(t,[r['leg_target_rad'][4] for r in rows],color=color,label=label)
    axes[1,0].plot(t,[r['leg_actual_rad'][4] for r in rows],color=color,label=label)
    axes[1,1].plot([r['t_start_s'] for r in rows],[r['encoder_velocity_mps'][1]-r['velocity_true_mps_before'][1] for r in rows],color=color,label=label)
for ax,title in zip(axes.flat,['Body tilt AFTER physics (deg)','Right leg2 target (rad)','Right leg2 actual (rad)','Lateral velocity error BEFORE action (m/s)']):
    ax.set_title(title);ax.grid(alpha=.2);ax.axvline(.58,color='gray',linestyle=':')
for ax in axes[1,:]:ax.set_xlabel('Physical time (s)')
axes[0,0].legend()
fig.suptitle('seed231, deterministic only: verified L24 vs L48; first29 steps, same initial state and RNG')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(BASE/'audit/L24-vs-L48.png',dpi=150);plt.close(fig)
