"""只读复算R8两段训练日志及40例开发评价，保留恢复的物理重置边界。"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

B=Path('/workspace/sac-r8-review');OLD=B/'L2-evidence';NEW=B/'L48-evidence';OUT=B/'audit';OUT.mkdir(exist_ok=True)
C1=Path('/workspace/sac-r7pc1-review/evidence')
sys.path.insert(0,str(C1/'evaluation'))
from r4_protocol import analyze_rows
from evaluation_r7p_helpers import physical_fingerprint

def read(p):return json.loads(p.read_text())
def write(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def close(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):assert abs(a-b)<1e-11,(a,b)
    else:assert a==b,(a,b)

training=[];reset_pairs={}
for arm in ('A0-1501','A0-1502','A1-1501','A1-1502'):
    total={k:0 for k in ('critic','actor','alpha','auxiliary','target')};seen=[];success_aux=skipped_aux=0;normalizer_final=None
    for root,expected in ((OLD,list(range(1,19))),(NEW,list(range(19,65)))):
        file=root/'arms'/arm/'macros.jsonl';seq=[]
        with file.open() as f:
            for line in f:
                row=json.loads(line);k=row['macro_iteration'];seq.append(k);L=max(0,k-16)
                assert row['counters']['iterations']==k and row['counters']['transitions']==1024*k
                for key,mult in [('critic_updates',128),('actor_updates',64),('alpha_updates',128),('target_updates',128),('auxiliary_updates',128)]:assert row['counters'][key]==mult*L
                updates=row['optimizer_steps_actual'];learning=k>16
                assert updates==dict(critic=128 if learning else 0,actor=64 if learning else 0,alpha=128 if learning else 0,auxiliary=128 if learning else 0,target=128 if learning else 0)
                assert row['target_updates_actual']==updates['target']
                for n,count in updates.items():total[n]+=count
                assert row['auxiliary_attempts_actual']==(128 if learning else 0)
                assert row['auxiliary_success_actual']==(128 if learning else 0)
                assert row['auxiliary_skipped_actual']==0
                success_aux+=row['auxiliary_success_actual'];skipped_aux+=row['auxiliary_skipped_actual']
                assert len(row['microcycles'])==(8 if learning else 0)
                for cycle in row['microcycles']:
                    assert not cycle['learn']['skipped'] and len(cycle['auxiliary'])==16
                    assert all(not a['skipped'] for a in cycle['auxiliary'])
                    assert all(math.isfinite(x) for x in cycle['learn'].values() if isinstance(x,float))
                diag=row['diagnostics'];assert diag['samples']==1024 and diag['parking_weight']==-8
                assert abs(diag['parking_weighted_dt_mean']-diag['parking_unweighted_mean']*(-8)*.02)<1e-7
                normalizer_final=row['normalizer_counts']
        assert seq==expected;(seen.extend(seq))
    assert seen==list(range(1,65)) and total==dict(critic=6144,actor=3072,alpha=6144,auxiliary=6144,target=6144)
    ef=read(NEW/'arms'/arm/'effective-settings.json');expected_lr=5e-5 if arm.startswith('A0') else 1e-5
    assert ef['actor_lr']==expected_lr and ef['parking_body_velocity_weight']==-8 and ef['command_resampling_time_range']==[2,4]
    assert ef['new_parking_precision_weight']==0 and ef['num_envs']==128
    restored=read(NEW/'arms'/arm/'resume-restoration-check.json');assert restored['status']=='passed_runtime' and restored['environment_state_restored'] is False
    checkpoint=read(OLD/'arms'/arm/'STATUS.json')['checkpoint']['sha256'];assert restored['checkpoint_sha256']==checkpoint
    assert restored['counters']['iterations']==18 and restored['counters']['critic_updates']==256
    fp=read(next((NEW/'arms'/arm).glob('initial-physical-*.json')))['sha256']
    fork=arm.split('-')[1]
    if fork in reset_pairs:assert reset_pairs[fork]==fp
    reset_pairs[fork]=fp
    training.append(dict(arm=arm,actor_lr=expected_lr,macro_rows=64,prefill_macros=16,learning_macros=48,actual_successful_updates=total,
        auxiliary_success=success_aux,auxiliary_skipped=skipped_aux,transitions=65536,normalizer_final=normalizer_final,
        restored_from_L2_sha256=checkpoint,physics_restored=False,same_fork_reset_physical_sha256=fp))
write('training-audit.json',dict(trajectories=training,total_macro_rows=256,total_transitions=262144,physical_reset_after_L2=True,
    same_fork_new_physical_initial_state_equal=True,not_original_continuous_training=True))

evaldir=NEW/'backup-evaluation'
def file(case,name):
    matches=list(evaldir.glob(case+'--*--'+name));assert len(matches)==1,(case,name,len(matches));return matches[0]
controller=read(next(evaldir.glob('*--controller-state.json')))
assert controller['status']=='completed' and controller['launched']==40 and len(controller['cases'])==40 and controller['training_updates']==0
cases=[];models={};reference={};frozen={};prefixes={}
verified_models=read(B/'model-and-resume-verification.json')
for task in controller['planned_tasks']:
    arm=task['model']['direction']+'-'+str(task['model']['fork_seed']);seed=task['seed'];ex=task['execution'];case=f'{arm}-L48-{seed}-{ex}'
    key=(arm,seed,ex);assert key not in prefixes
    cfg=read(file(case,'configuration.json'));before=read(file(case,'frozen-before.json'));after=read(file(case,'frozen-after.json'))
    assert before==after
    if arm in frozen:assert frozen[arm]==before
    frozen[arm]=before
    assert cfg['phase_id']=='SAC-R8-L48-FINAL40-20261109' and cfg['seed_split']=='development'
    assert cfg['evaluation_seed']==seed and cfg['execution']==ex and seed in (231,232,233,234,235)
    assert cfg['sha256']==task['model']['model_sha256'] and cfg['condition']=='standard'
    model=next(x for x in verified_models if arm in x['name'] and 'learning48' in x['name']);assert cfg['sha256']==model['sha256'] and cfg['checkpoint_counters']==model['counters']
    assert cfg['policy_noise']=='on' and cfg['factor']=='F01' and cfg['zero_reset_wrench'] and cfg['interval_push_disabled']
    assert not cfg['tf32_matmul'] and not cfg['tf32_cudnn']
    assert cfg['evaluation_noise_seed']==100000+seed and cfg['termination_event_source']=='current_compute_return'
    fp=physical_fingerprint(cfg['initial_state']);assert fp==cfg['physical_initial_state_sha256']
    parent=read(C1/'evaluations/cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3/development'/f'standard-{ex}-{seed}'/'attempt-1/configuration.json')
    assert cfg['initial_state']==parent['initial_state'] and cfg['rng_identity']==parent['rng_identity']
    # 物理/网络模块源码保持；评价器仅ID、phase和seed白名单变更另审。
    s={Path(p).name:h for p,h in cfg['source_identity'].items()};ps={Path(p).name:h for p,h in parent['source_identity'].items()}
    for n in set(s)&set(ps)-{'evaluate_sac_r7p.py','evaluation_r7p_helpers.py'}:assert s[n]==ps[n],n
    assert read(file(case,'process-result.json'))['returncode']==0 and read(file(case,'status.json'))['status']=='passed'
    rows=[json.loads(line) for line in file(case,'steps.jsonl').read_text().splitlines()];assert len(rows)==600
    result=analyze_rows(rows,'standard');close(result,read(file(case,'protocol-result.json')))
    m=result['metrics'];P=max(m[k]['speed'] for k in ('stop1','stop2'));W=max(m[k]['abs_wz'] for k in ('stop1','stop2'));Mavg=mean([m[k]['vx_mae'] for k in ('forward','reverse')]);Mmax=max(m[k]['vx_mae'] for k in ('forward','reverse'))
    # 独立窗口和门槛复算。
    def window(a,b):
        r=rows[a:b]
        return dict(speed=mean(math.hypot(*x['measured_body_vx_vy_wz'][:2]) for x in r),yaw=mean(abs(x['measured_body_vx_vy_wz'][2]) for x in r),mae=mean(abs(x['measured_body_vx_vy_wz'][0]-x['effective_command'][0]) for x in r))
    p1,p2,f,r=window(50,100),window(350,400),window(100,300),window(400,600)
    assert abs(P-max(p1['speed'],p2['speed']))<1e-12 and abs(W-max(p1['yaw'],p2['yaw']))<1e-12
    flags=dict(parking_speed=P>.05,parking_yaw=W>.10,moving_mae=Mmax>.12,moving_yaw=max(f['yaw'],r['yaw'])>.20,height=m['height']['height']<.25,survival=not result['survived'])
    assert result['task_pass']==(not any(flags.values()))
    c=dict(arm=arm,seed=seed,execution=ex,steps=len(rows),survived=result['survived'],task_pass=result['task_pass'],P=P,W=W,Mavg=Mavg,Mmax=Mmax,
        forward_mae=m['forward']['vx_mae'],reverse_mae=m['reverse']['vx_mae'],stop1_speed=m['stop1']['speed'],stop2_speed=m['stop2']['speed'],
        stop1_yaw=m['stop1']['abs_wz'],stop2_yaw=m['stop2']['abs_wz'],move_yaw=max(f['yaw'],r['yaw']),height=m['height']['height'],
        fail_reasons=[k for k,v in flags.items() if v],sha256=cfg['sha256'],steps_sha256=hashlib.sha256(file(case,'steps.jsonl').read_bytes()).hexdigest())
    cases.append(c);prefixes[key]=rows[0];models[arm]=cfg['sha256']
assert len(cases)==40 and len({(c['arm'],c['seed'],c['execution']) for c in cases})==40
for arm in models:
    for seed in (231,232,233,234,235):
        a,b=prefixes[arm,seed,'D'],prefixes[arm,seed,'S']
        for n in ('actor_mu','actor_std','encoder_velocity_mps','velocity_true_mps_before'):assert a[n]==b[n]
summaries=[]
for arm in sorted(models):
    part=[c for c in cases if c['arm']==arm]
    summaries.append(dict(arm=arm,n=10,survival_count=sum(x['survived'] for x in part),pass_count=sum(x['task_pass'] for x in part),
        **{k:mean(x[k] for x in part) for k in ('P','W','Mavg','Mmax')},
        Pmax=max(x['P'] for x in part),Wmax=max(x['W'] for x in part),Mmax_worst=max(x['Mmax'] for x in part),
        failure_counts={n:sum(n in c['fail_reasons'] for c in part) for n in ('parking_speed','parking_yaw','moving_mae','moving_yaw','height','survival')}))
reported=read(NEW/'final40-summary.json')
for s in summaries:
    original=next(x['summary'] for x in reported['models'] if x['direction']+'-'+str(x['fork_seed'])==s['arm'])
    for k,n in [('survival_count','survival_count'),('pass_count','pass_count'),('parking_P','P'),('parking_yaw','W'),('moving_mae','Mavg')]:assert abs(original[k]-s[n])<1e-12
parent=read(Path('/workspace/sac-r8-plan/handoff/parameters.json'))['parent_reference_metrics'];decisions=[]
for seed in (1501,1502):
    control=next(s for s in summaries if s['arm']==f'A0-{seed}');child=next(s for s in summaries if s['arm']==f'A1-{seed}')
    checks=dict(survival10=child['survival_count']==10,passes_vs_control=child['pass_count']>=control['pass_count'],passes_vs_parent=child['pass_count']>=parent['passed'],
        P_vs_control=child['P']<=control['P']-.005,P_vs_parent=child['P']<=parent['P']-.005,
        Mavg_vs_control=child['Mavg']<=control['Mavg']+.01,Mavg_vs_parent=child['Mavg']<=parent['Mavg']+.01,W=child['W']<=.1)
    decisions.append(dict(fork_seed=seed,checks=checks,direction_signal=all(checks.values())))
close(decisions,reported['pairs']);assert reported['direction_reproduced']==all(d['direction_signal'] for d in decisions)
write('case-results.json',cases);write('group-results.json',summaries);write('decision.json',dict(pairs=decisions,direction_reproduced=all(d['direction_signal'] for d in decisions),parent_replaced=False,heldout_run=False,development_seed_set=[231,232,233,234,235]))
with (OUT/'cases.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cases[0]),lineterminator='\n');w.writeheader()
    for c in cases:w.writerow(dict(c,fail_reasons=';'.join(c['fail_reasons'])))
write('audit-receipt.json',dict(fixed_final_commit='1d4bcb41aa8171d36008425c7554ffb1a2747dca',new_macro_rows_recomputed=256,successful_updates_per_trajectory=dict(critic=6144,actor=3072,alpha=6144,target=6144,auxiliary=6144),raw_development_cases=40,raw_development_steps=24000,all_survived=True,source_physical_initial_RNG_and_frozen_checks=True,continuous_physics=False,new_training_or_simulation_started_by_reviewer=False))
print(json.dumps(dict(macro_rows=256,eval_cases=40,eval_steps=24000,pass_counts={s['arm']:s['pass_count'] for s in summaries},direction_reproduced=False)))
