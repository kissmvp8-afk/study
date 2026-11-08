"""独立复算20组温度实现诊断：只读配置、训练/计时/评价记录，不载入模型。"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--evidence-root',type=Path,required=True)
ap.add_argument('--output-dir',type=Path,required=True)
args=ap.parse_args();DATA=args.evidence_root;OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def table(name,rows):
    with (OUT/name).open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


manifest=read(DATA/'supplement/MEMBER_SHA256.json')
assert all((DATA/n).is_file() and digest(DATA/n)==h for n,h in manifest.items())
original=read(DATA/'metadata/EVIDENCE_MANIFEST.json')
assert all((DATA/'run-001'/n).is_file() and digest(DATA/'run-001'/n)==h for n,h in original.items())
provided=list(csv.DictReader((DATA/'supplement/GROUP_UPDATES_TIMINGS.csv').open()))
assert len(provided)==20
groups=[];pairs=[];stages=[];evaluations=[];training_pairs=[];metric_discrepancies=[]
window_defs=[('forward',2,6),('reverse',8,12),('idle_before_tail',1,2),('idle_between_tail',7,8),('stable_all_after_2',2,12)]
for seed in range(10):
    pair=DATA/f'run-001/pair-{seed:02d}'
    ps=read(pair/'pair_summary.json')
    expected_order=['baseline','tensor'] if seed%2==0 else ['tensor','baseline']
    assert ps['order']==expected_order and ps['complete']
    records={}
    for variant in ('baseline','tensor'):
        root=pair/variant;s=read(root/'summary.json');cfg=read(root/'run_config.json')['config']
        log=[json.loads(l) for l in (root/'iterations.jsonl').open()]
        tm=[json.loads(l) for l in (root/'timings.jsonl').open()]
        assert cfg['collection']['warmup_iterations']==0 and cfg['collection']['num_envs']==128 and cfg['seed']==seed
        assert len(log)==len(tm)==64 and s['status']=='completed' and s['completed_target']
        for i,(d,t) in enumerate(zip(log,tm),1):
            c=d['counters']
            assert c['iterations']==t['iteration']==i and c['transitions']==i*1024 and c['env_steps']==i*8
            for k in ('critic_updates','alpha_updates','target_updates','auxiliary_updates'):
                assert c[k]==i*16
            assert c['actor_updates']==i*8 and c['auxiliary_skipped']==c['learning_skipped_iterations']==0
            assert t['phase']=='measurement' and t['critic_updates']==i*16
        assert s['counters']==log[-1]['counters']
        assert s['actual_optimizer_steps']=={'critic':1024,'actor':512,'alpha':1024,'auxiliary':1024}
        assert s['final_fingerprints']['networks']!=s['initial_fingerprints']['networks']
        sums={k:sum(float(r[k]) for r in tm) for k in tm[0] if k.endswith('_s')}
        arm=ps['arms'][variant]
        for key in ('collect_s','learn_s','auxiliary_s','iteration_wall_s','source_hash_s','diagnostics_s'):
            assert abs(sums[key]-arm['timing'][key]['sum'])<1e-8
        row=next(r for r in provided if int(r['seed'])==seed and r['variant']==variant)
        for key in ('iterations','transitions','critic_updates','actor_updates','alpha_updates','auxiliary_updates'):
            assert int(row['summary.counters.'+key])==s['counters'][key]
        assert abs(float(row['timing.learn_s.sum'])-sums['learn_s'])<1e-8
        d=dict(seed=seed,variant=variant,position=expected_order.index(variant)+1,num_envs=128,warmup_iterations=0,
               iterations=64,transitions=65536,critic_updates=1024,actor_updates=512,alpha_updates=1024,
               target_updates=1024,auxiliary_updates=1024,auxiliary_skips=0,
               replay_sample_ratio=s['counters']['replay_samples']/s['counters']['transitions'],
               learn_s=sums['learn_s'],collect_s=sums['collect_s'],auxiliary_s=sums['auxiliary_s'],
               iteration_wall_s=sums['iteration_wall_s'],process_wall_s=arm['process_wall_s'],
               residual_cuda_wait_s=sums['residual_cuda_wait_s'],
               model_hash_record=s['final_model']['sha256'],last_alpha=log[-1]['learn']['alpha'],
               last_internal_std=log[-1]['exploration']['internal_std_mean'])
        groups.append(d)
        records[variant]=(s,cfg,log,d)

        ev=DATA/f'run-001/evaluation/pair-{seed:02d}/{variant}'
        rows=[json.loads(l) for l in (ev/'steps.jsonl').open()]
        ecfg=read(ev/'configuration.json');es=read(ev/'summary.json')
        assert ecfg['training_seed']==seed and ecfg['evaluation_seed']==101+seed%3
        assert ecfg['execution']=='D' and ecfg['factor']=='F01' and ecfg['policy_noise']=='on' and ecfg['zero_reset_wrench'] and ecfg['interval_push_disabled']
        assert len(rows)==es['steps']
        assert ecfg['sha256']==s['final_model']['sha256']
        assert es['normalizers_unchanged'] and es['frozen_state_unchanged'] and es['checkpoint_unchanged']
        assert read(ev/'frozen-before.json')==read(ev/'frozen-after.json')
        for i,r in enumerate(rows):
            target=[.3,0,0] if 100<=i<300 else ([-.3,0,0] if i>=400 else [0,0,0])
            assert r['step']==i and abs(r['t_start_s']-.02*i)<1e-8 and abs(r['t_end_s']-.02*(i+1))<1e-8
            assert np.allclose(r['effective_command'],target,atol=1e-7,rtol=0)
            assert np.count_nonzero(r['external_force_b_command'])==np.count_nonzero(r['external_torque_b_command'])==0
            assert not (r['terminated'] or r['time_out']) or i==len(rows)-1
        sm={m['stage']:m for m in es['stage_metrics']};own={}
        for stage,start,end in window_defs:
            rr=rows[start*50:end*50]
            if not rr:continue
            v=np.array([r['measured_body_vx_vy_wz'] for r in rr]);c=np.array([r['effective_command'] for r in rr])
            pos=np.array([r['root_pos_w'] for r in rr])
            m=dict(seed=seed,variant=variant,stage=stage,samples=len(rr),complete=len(rr)==(end-start)*50,
                   vx_mae=float(abs(v[:,0]-c[:,0]).mean()),mean_vx=float(v[:,0].mean()),
                   mean_planar_speed=float(np.linalg.norm(v[:,:2],axis=1).mean()),mean_abs_wz=float(abs(v[:,2]).mean()),
                   mean_height_m=float(pos[:,2].mean()))
            for k,value in m.items():
                if k in sm.get(stage,{}) and isinstance(value,(float,int)) and not isinstance(value,bool):
                    if abs(value-sm[stage][k])>1e-6:metric_discrepancies.append(dict(seed=seed,variant=variant,stage=stage,key=k))
            stages.append(m);own[stage]=m
        gate=len(rows)==600 and not any(r['terminated'] or r['time_out'] for r in rows)
        gate=gate and all(own[k]['vx_mae']<=.12 and own[k]['mean_abs_wz']<=.2 for k in ('forward','reverse'))
        gate=gate and own['idle_between_tail']['mean_planar_speed']<=.05 and own['stable_all_after_2']['mean_height_m']>=.25 and max(r['contact_named_max_N']['base'] for r in rows)<=1
        assert gate==es['development_gate_pass']
        evaluations.append(dict(seed=seed,variant=variant,eval_seed=ecfg['evaluation_seed'],steps=len(rows),seconds=len(rows)/50,
                                true_terminations=sum(r['terminated'] for r in rows),timeouts=sum(r['time_out'] for r in rows),
                                gate_pass=gate,steps_sha256=digest(ev/'steps.jsonl')))
    bs,bc,bl,b=records['baseline'];ts,tc,tl,t=records['tensor']
    pair_eval=DATA/f'run-001/evaluation/pair-{seed:02d}'
    eq=dict(seed=seed,initial_fingerprints_match=bs['initial_fingerprints']==ts['initial_fingerprints'],
            effective_learning_configs_equal=bc==tc,training_jsonl_byte_equal=digest(pair/'baseline/iterations.jsonl')==digest(pair/'tensor/iterations.jsonl'),
            final_fingerprints_records_equal=bs['final_fingerprints']==ts['final_fingerprints'],
            final_model_hash_records_equal=bs['final_model']['sha256']==ts['final_model']['sha256'],
            evaluation_initial_physics_equal=read(pair_eval/'baseline/configuration.json')['initial_state']==read(pair_eval/'tensor/configuration.json')['initial_state'],
            evaluation_steps_byte_equal=digest(pair_eval/'baseline/steps.jsonl')==digest(pair_eval/'tensor/steps.jsonl'))
    assert all(v for k,v in eq.items() if k!='seed')
    training_pairs.append(eq)
    pairs.append(dict(seed=seed,first=expected_order[0],baseline_learn_s=b['learn_s'],tensor_learn_s=t['learn_s'],
                      delta_learn_s=t['learn_s']-b['learn_s'],delta_iteration_wall_s=t['iteration_wall_s']-b['iteration_wall_s'],
                      delta_process_wall_s=t['process_wall_s']-b['process_wall_s'],
                      baseline_residual_wait_s=b['residual_cuda_wait_s'],tensor_residual_wait_s=t['residual_cuda_wait_s']))

stats={}
for metric in ('learn_s','iteration_wall_s','process_wall_s'):
    d=[p['delta_'+metric] for p in pairs];mean=statistics.mean(d);sd=statistics.stdev(d)
    base_mean=statistics.mean(g[metric] for g in groups if g['variant']=='baseline')
    tensor_mean=statistics.mean(g[metric] for g in groups if g['variant']=='tensor')
    half=2.2621571627409915*sd/math.sqrt(10)
    stats[metric]=dict(n_pairs=10,baseline_mean_s=base_mean,tensor_mean_s=tensor_mean,mean_delta_s=mean,
                       median_delta_s=statistics.median(d),sd_delta_s=sd,
                       ci95_exact_t9_s=[mean-half,mean+half],relative_mean_delta=mean/base_mean,
                       ci95_relative_delta=[(mean-half)/base_mean,(mean+half)/base_mean],
                       delta_when_baseline_first=statistics.mean(p['delta_'+metric] for p in pairs if p['first']=='baseline'),
                       delta_when_tensor_first=statistics.mean(p['delta_'+metric] for p in pairs if p['first']=='tensor'))
table('groups.csv',groups);table('paired-timings.csv',pairs);table('paired-compatibility.csv',training_pairs)
table('evaluation-runs.csv',evaluations);table('evaluation-stages.csv',stages)
save('timing-statistics.json',stats);save('metric-discrepancies.json',metric_discrepancies)
summary=dict(package_members_hashed=len(manifest),original_execution_files_hashed=len(original),training_groups=20,
             training_iterations_checked=1280,timing_rows_checked=1280,actual_update_records_match=True,warmup_iterations=0,
             critic_alpha_target_aux_updates_per_group=1024,actor_updates_per_group=512,
             compatible_pairs=10,training_log_pairs_byte_equal=10,evaluation_pairs_byte_equal=10,
             model_hash_records_pairs_equal=10,model_weight_bytes_verified=False,
             evaluation_runs=20,evaluation_steps=sum(r['steps'] for r in evaluations),
             complete_600_step_runs=sum(r['steps']==600 for r in evaluations),gate_passes=sum(r['gate_pass'] for r in evaluations),
             metric_discrepancies=len(metric_discrepancies),simulator_started=False,models_deserialized=False)
save('audit-summary.json',summary)
print(json.dumps(summary,indent=2))
