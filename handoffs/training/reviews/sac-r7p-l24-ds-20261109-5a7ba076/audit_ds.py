"""只读复算L24原始D/S日志，按同seed与C1 L48/父模型配对；不启动仿真。"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

SHA48 = {
    'parent': 'cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3',
    'P0-1401': 'f918d54b8ddeaab05d3b4db0dc012db96e09fd4e223e39c664a1422642c89783',
    'P0-1402': '18e70c5e63a329743794ea6da3ddf59b25ae3e111c0d6e1d0acc04c50413303a',
    'P1-1401': 'c0f1befd1985b99776647850001b3be841173a9fe406b7d6b59630722086eba1',
    'P1-1402': '6b71a242bb92188f835be8ba70b459a7ee271792ebf0127e0a9b7050c2bc27c4',
}


def read(p):
    return json.loads(p.read_text())


def mean(x):
    return sum(x)/len(x)


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: close(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): close(x, y)
    elif isinstance(a, float):
        assert abs(a-b) < 1e-11, (a, b)
    else:
        assert a == b, (a, b)


def source_id(cfg):
    value = {Path(p).name: h for p, h in cfg['source_identity'].items()}
    assert len(value) == len(cfg['source_identity'])
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=Path('/workspace/sac-r7p-ds-review'))
    ap.add_argument('--c1', type=Path, default=Path('/workspace/sac-r7pc1-review/evidence'))
    ap.add_argument('--old-l24', type=Path, default=Path('/workspace/sac-r7pc2-review/L24-evidence'))
    args = ap.parse_args()
    out = args.work/'audit';out.mkdir(exist_ok=True)
    sys.path.insert(0, str(args.c1/'evaluation'))
    from r4_protocol import analyze_rows
    from evaluation_r7p_helpers import physical_fingerprint
    bounds = next(iter(read(args.work/'model-inference-reference.json').values()))['action_bounds']
    centers, widths = bounds['center'], bounds['half_width']
    l24models = {x['sha256']: x for x in read(Path('/workspace/sac-r7pc2-review/L24-verification.json'))}
    cases, inputs, cache, frozen_states = [], [], {}, {}
    folders = []
    for root in (args.old_l24, args.work/'231s', args.work/'232233d', args.work/'232233s'):
        report = read(root/'exploratory-results.json')
        assert report['status'] == 'completed' and report['training_updates'] == 0
        for folder in sorted((root/'cases').iterdir()):
            arm = '-'.join(folder.name.split('-')[:2])
            folders.append((arm, 24, folder))
    for arm, sha in SHA48.items():
        for seed in (231,232,233):
            for ex in ('D','S'):
                folders.append((arm, 0 if arm == 'parent' else 48,
                    args.c1/'evaluations'/sha/'development'/f'standard-{ex}-{seed}'/'attempt-1'))
    reference = {}
    sources = None
    for arm, learning, folder in folders:
        cfg = read(folder/'configuration.json');seed, ex = cfg['evaluation_seed'], cfg['execution']
        key = (arm, learning, seed, ex);assert key not in cache
        if learning == 24:
            assert cfg['sha256'] in l24models
            assert cfg['checkpoint_counters'] == l24models[cfg['sha256']]['counters']
        else:
            assert cfg['sha256'] == SHA48[arm]
        assert (seed in (231,232,233) and ex in ('D','S') and cfg['condition']=='standard')
        assert cfg['policy_noise']=='on' and cfg['zero_reset_wrench'] and cfg['interval_push_disabled']
        assert not cfg['tf32_matmul'] and not cfg['tf32_cudnn']
        assert cfg['termination_event_source']=='current_compute_return'
        assert cfg['evaluation_noise_seed']==100000+seed
        assert cfg['task']=='DDT-Velocity-Flat-Tita-v0' and not cfg['Play_task_used']
        assert read(folder/'frozen-before.json') == read(folder/'frozen-after.json')
        frozen_states[key] = read(folder/'frozen-before.json')
        assert read(folder/'process-result.json')['returncode'] == 0
        assert read(folder/'status.json')['status'] == 'passed'
        current_source = source_id(cfg)
        if sources is None: sources = current_source
        assert current_source == sources, ('Source mismatch', folder)
        fp = physical_fingerprint(cfg['initial_state'])
        assert fp == cfg['physical_initial_state_sha256']
        identity = {'physical':cfg['initial_state'], 'rng':cfg['rng_identity']}
        if seed not in reference: reference[seed] = identity
        assert identity == reference[seed], ('Unpaired physical initial state or RNG', folder)
        rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        result = analyze_rows(rows, 'standard')
        close(result, read(folder/'protocol-result.json'))
        for name in ('configuration.json','steps.jsonl','frozen-before.json','frozen-after.json','protocol-result.json'):
            p=folder/name;raw=p.read_bytes()
            inputs.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        m=result['metrics'];complete = len(rows)==600 and not result['loss_is_failure_penalty']
        summary=dict(arm=arm,learning=learning,seed=seed,execution=ex,sha256=cfg['sha256'],
            steps=len(rows),survived=result['survived'],task_pass=result['task_pass'],
            physical_initial_sha256=fp,source_equal=True,initial_physical_and_RNG_equal=True,
            P=max(m[k]['speed'] for k in ('stop1','stop2')) if complete else None,
            W=max(m[k]['abs_wz'] for k in ('stop1','stop2')) if complete else None,
            Mavg=mean([m[k]['vx_mae'] for k in ('forward','reverse')]) if complete else None,
            Mmax=max(m[k]['vx_mae'] for k in ('forward','reverse')) if complete else None,
            forward_mae=m['forward']['vx_mae'] if complete else None,
            reverse_mae=m['reverse']['vx_mae'] if complete else None,
            stop1_speed=m['stop1']['speed'] if m['stop1'] else None,
            stop2_speed=m['stop2']['speed'] if m['stop2'] else None,
            move_yaw_max=max(m[k]['abs_wz'] for k in ('forward','reverse')) if complete else None,
            height_mean=m['height']['height'] if complete else None,
            base_contact_max_N=result['base_contact_max_N'],
            wheel_target_change_rate_rad_s2=result['wheel_target_change_rate_rad_s2'])
        if complete:
            # 独立窗口复算：不使用原评分器的窗口函数或聚合值。
            def window(a,b):
                part=rows[a:b]
                return {'speed':mean([math.hypot(*r['measured_body_vx_vy_wz'][:2]) for r in part]),
                    'yaw':mean([abs(r['measured_body_vx_vy_wz'][2]) for r in part]),
                    'mae':mean([abs(r['measured_body_vx_vy_wz'][0]-r['effective_command'][0]) for r in part])}
            s1,s2,f,r=window(50,100),window(350,400),window(100,300),window(400,600)
            for left,right in [(summary['P'],max(s1['speed'],s2['speed'])),(summary['W'],max(s1['yaw'],s2['yaw'])),
                               (summary['Mmax'],max(f['mae'],r['mae']))]: assert abs(left-right)<1e-12
            flags=dict(parking_speed=summary['P']>.05,parking_yaw=summary['W']>.10,
                       moving_mae=summary['Mmax']>.12,moving_yaw=summary['move_yaw_max']>.20,
                       height=summary['height_mean']<.25, survival=not summary['survived'])
            assert result['task_pass'] == (not any(flags.values()))
            summary['fail_reasons']=[k for k,v in flags.items() if v]
        else:
            summary['fail_reasons']=['true_early_termination']
        # 同一个S访问状态上的瞬时采样偏差，不能当成实际D/S闭环轨迹差值。
        noise=[]
        for r in rows:
            deterministic=[c+w*math.tanh(mu) for c,w,mu in zip(centers,widths,r['actor_mu'])]
            delta=[a-d for a,d in zip(r['actor_action'],deterministic)]
            if ex=='D': assert max(map(abs,delta))<2e-5
            if 50<=r['step']<100 or 350<=r['step']<400:
                noise.append((mean([abs(delta[i])*5 for i in (3,7)]),
                              mean([abs(delta[i])*.25 for i in (0,1,2,4,5,6)])))
        summary['same_state_sampling_wheel_abs_radps']=mean([n[0] for n in noise]) if noise else None
        summary['same_state_sampling_leg_abs_rad']=mean([n[1] for n in noise]) if noise else None
        cases.append(summary);cache[key]=(rows,cfg,summary)
    assert len(cases)==54 and sum(c['steps'] for c in cases)==31258
    pairs=[]
    for arm in list(SHA48)[1:]:
        for seed in (231,232,233):
            d,dc,ds=cache[arm,24,seed,'D'];s,sc,ss=cache[arm,24,seed,'S']
            assert dc['sha256']==sc['sha256']
            assert frozen_states[arm,24,seed,'D'] == frozen_states[arm,24,seed,'S']
            for field in ('actor_mu','actor_std','encoder_velocity_mps','velocity_true_mps_before'):
                assert d[0][field]==s[0][field], (arm,seed,field)
            pairs.append(dict(arm=arm,seed=seed,**{k+'_S_minus_D':ss[k]-ds[k] for k in ('P','W','Mavg','Mmax')},
                D_pass=ds['task_pass'],S_pass=ss['task_pass']))
    comparisons=[]
    for arm in list(SHA48)[1:]:
        for seed in (231,232,233):
            for ex in ('D','S'):
                a=cache[arm,24,seed,ex][2];b=cache[arm,48,seed,ex][2]
                comparisons.append(dict(arm=arm,seed=seed,execution=ex,L24_pass=a['task_pass'],L48_pass=b['task_pass'],
                    L24_survive=a['survived'],L48_survive=b['survived'],
                    **{k+'_L48_minus_L24':b[k]-a[k] if b[k] is not None else None for k in ('P','W','Mavg','Mmax')}))
    groups=[]
    for arm in SHA48:
        for learning in ([0] if arm=='parent' else [24,48]):
            for ex in ('D','S'):
                selected=[c for c in cases if c['arm']==arm and c['learning']==learning and c['execution']==ex and c['seed'] in (232,233)]
                groups.append(dict(arm=arm,learning=learning,execution=ex,seeds=[232,233],n=len(selected),
                    survive=sum(c['survived'] for c in selected),passed=sum(c['task_pass'] for c in selected),
                    **{k:mean([c[k] for c in selected]) if all(c[k] is not None for c in selected) else None for k in ('P','W','Mavg','Mmax')}))
    for name,data in [('cases.json',cases),('groups-232233.json',groups),('paired-DS.json',pairs),
                      ('paired-L24-L48.json',comparisons),('input-files.json',inputs),('source-fingerprints.json',sources)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    fields=[k for k in cases[0] if k!='fail_reasons']+['fail_reasons']
    with (out/'cases.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');w.writeheader()
        for c in cases:w.writerow(dict(c,fail_reasons=';'.join(c['fail_reasons'])))
    brief=dict(cases_recomputed=len(cases),physical_steps=sum(c['steps'] for c in cases),
        L24_total=24,C1_recomputed_cases=30,physical_initial_RNG_source_matched=True,
        DS_first_inference_equal=True,raw_noisy_history_present=False,
        primary_evaluation_seeds=[232,233],additional_temporal_diagnosis_seed=231)
    (out/'audit-receipt.json').write_text(json.dumps(brief,indent=2)+'\n');print(json.dumps(brief))


if __name__=='__main__':main()
