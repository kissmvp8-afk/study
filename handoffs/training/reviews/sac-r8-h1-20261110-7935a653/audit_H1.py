"""只读独立复算H1二十例及两项兼容探针，不运行归档控制器。"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from statistics import mean
import math

B=Path('/workspace/sac-r9-overnight-plan');RAW=B/'H1-evidence';OUT=B/'H1-audit';OUT.mkdir(exist_ok=True)
C1=Path('/workspace/sac-r7pc1-review/evidence')
sys.path.insert(0,str(C1/'evaluation'))
from r4_protocol import analyze_rows
from evaluation_r7p_helpers import physical_fingerprint
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def norm(cfg):
    cfg=json.loads(json.dumps(cfg))
    cfg['original_push']=re.sub(r'(<function push_by_setting_velocity at )0x[0-9a-fA-F]+(>)',r'\1ADDRESS\2',cfg['original_push'])
    for v in cfg['reward_terms'].values():v['function']=re.sub(r' at 0x[0-9a-fA-F]+>', '>',v['function'])
    return cfg
def eq(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:eq(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):eq(x,y)
    elif isinstance(a,float):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b

probes=[]
old_parent=C1/'evaluations/cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3/development/standard-D-231/attempt-1'
old_candidate=Path('/workspace/sac-r8-review/L48-evidence/backup-evaluation')
for role in ('parent','candidate'):
    new=RAW/'probes'/f'{role}-231-D'
    def oldfile(name):
        if role=='parent':return old_parent/name
        m=list(old_candidate.glob('A1-1502-L48-231-D--*--'+name));assert len(m)==1;return m[0]
    a,b=rows(oldfile('steps.jsonl')),rows(new/'steps.jsonl');assert len(a)==len(b)==600 and a==b
    oldcfg,newcfg=read(oldfile('configuration.json')),read(new/'configuration.json')
    for name,ignore in [('configuration.json',{'argv','checkpoint','plan_id','phase_id','rollout_ready_unix','source_identity'}),('summary.json',{'elapsed_s','phase_id','plan_id','rollout_finished_unix'}),('frozen-before.json',set()),('frozen-after.json',set()),('protocol-result.json',set()),('stage-metrics.json',set())]:
        x,y=read(oldfile(name)),read(new/name)
        if name=='configuration.json':x,y=norm(x),norm(y)
        if isinstance(x,dict):
            x={k:v for k,v in x.items() if k not in ignore};y={k:v for k,v in y.items() if k not in ignore}
        assert x==y,(role,name)
    adapter={'evaluate_sac_r7p.py','evaluation_r7p_helpers.py','c1_metrics.py','h1_controller.py'}
    s={Path(k).name:v for k,v in oldcfg['source_identity'].items() if Path(k).name not in adapter}
    t={Path(k).name:v for k,v in newcfg['source_identity'].items() if Path(k).name not in adapter}
    if set(s)!=set(t):
        assert set(s)^set(t)=={'evaluate_with_kit_repair.py'}
        x,y=read(oldfile('kit-command-repair.json')),read(new/'kit-command-repair.json')
        actual=hashlib.sha256((RAW/'source/code/evaluate_with_kit_repair.py').read_bytes()).hexdigest()
        assert x['wrapper_sha256']==y['wrapper_sha256']==actual and x['provider_manifest_sha256']==y['provider_manifest_sha256']
        assert {k:v for k,v in x.items() if k!='original_evaluator_sha256'}=={k:v for k,v in y.items() if k!='original_evaluator_sha256'}
        s.pop('evaluate_with_kit_repair.py',None);t.pop('evaluate_with_kit_repair.py',None)
    assert s==t
    probes.append(dict(role=role,steps=600,raw_trajectory_exact=True,max_numeric_difference=0,trajectory_fields_ignored=[],restricted_pointer_normalization=True,nonadapter_sources_equal=True,old_parent_probe_not_rerun=True if role=='parent' else None))

cases=[];refs={};frozen={};initial={}
for folder in sorted((RAW/'cases').iterdir()):
    role,seed,ex=folder.name.split('-');seed=int(seed)
    assert role in ('parent','candidate') and seed in range(261,266) and ex in ('D','S')
    cfg=read(folder/'configuration.json');assert cfg['seed_split']=='heldout' and cfg['evaluation_seed']==seed and cfg['execution']==ex
    assert cfg['policy_noise']=='on' and cfg['zero_reset_wrench'] and cfg['interval_push_disabled'] and not cfg['tf32_matmul'] and not cfg['tf32_cudnn']
    assert cfg['sha256']==('7da5bcb7929d162da32735cdf9ec9c37a85a71bd7d2615c4b2b8f674a1ca78d1' if role=='candidate' else 'cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3')
    fingerprint=physical_fingerprint(cfg['initial_state']);assert fingerprint==cfg['physical_initial_state_sha256']
    identity=(cfg['initial_state'],cfg['rng_identity'])
    if seed in refs:assert refs[seed]==identity
    refs[seed]=identity
    before=read(folder/'frozen-before.json');assert before==read(folder/'frozen-after.json')
    if role in frozen:assert frozen[role]==before
    frozen[role]=before
    r=rows(folder/'steps.jsonl');assert len(r)==600
    assert read(folder/'process-result.json')['returncode']==0 and read(folder/'status.json')['status']=='passed'
    result=analyze_rows(r,'standard');eq(result,read(folder/'protocol-result.json'))
    m=result['metrics'];P=max(m[k]['speed'] for k in ('stop1','stop2'));W=max(m[k]['abs_wz'] for k in ('stop1','stop2'))
    independent=mean([math.hypot(*x['measured_body_vx_vy_wz'][:2]) for x in r[50:100]]);assert abs(independent-m['stop1']['speed'])<1e-12
    Mavg=mean(m[k]['vx_mae'] for k in ('forward','reverse'));Mmax=max(m[k]['vx_mae'] for k in ('forward','reverse'))
    reasons=[]
    for k in ('stop1','stop2'):
        if m[k]['speed']>.05:reasons.append(k+'.speed')
        if m[k]['abs_wz']>.10:reasons.append(k+'.yaw')
    if Mmax>.12:reasons.append('moving_MAE')
    if max(m[k]['abs_wz'] for k in ('forward','reverse'))>.20:reasons.append('moving_yaw')
    if m['height']['height']<.25:reasons.append('height')
    if not result['survived']:reasons.append('survival')
    assert result['task_pass']==(not bool(reasons))
    cases.append(dict(role=role,seed=seed,execution=ex,steps=600,survived=result['survived'],passed=result['task_pass'],P=P,W=W,Mavg=Mavg,Mmax=Mmax,stop1_speed=m['stop1']['speed'],stop2_speed=m['stop2']['speed'],failed_gates=';'.join(reasons)))
    initial[role,seed,ex]=r[0]
assert len(cases)==20 and len({(c['role'],c['seed'],c['execution']) for c in cases})==20
for role in ('parent','candidate'):
    for seed in range(261,266):
        a,b=initial[role,seed,'D'],initial[role,seed,'S']
        for k in ('actor_mu','actor_std','encoder_velocity_mps','velocity_true_mps_before'):assert a[k]==b[k]
groups=[]
for role in ('parent','candidate'):
    p=[x for x in cases if x['role']==role]
    groups.append(dict(role=role,n=10,survived=sum(c['survived'] for c in p),passed=sum(c['passed'] for c in p),**{k:mean(c[k] for c in p) for k in ('P','W','Mavg','Mmax')},maxP=max(c['P'] for c in p)))
summary=read(RAW/'h1-summary.json')
for g in groups:
    s=summary[g['role']];assert g['passed']==s['pass_count'] and g['survived']==s['survival_count']
    for k,n in [('P','parking_P'),('W','parking_yaw'),('Mavg','moving_mae'),('Mmax','moving_Mmax')]:assert abs(g[k]-s[n])<1e-12
candidate=next(x for x in groups if x['role']=='candidate');assert candidate['passed']==6 and candidate['survived']==10
fails=[c for c in cases if c['role']=='candidate' and not c['passed']]
assert {(c['seed'],c['execution']) for c in fails}=={(s,e) for s in (261,262) for e in ('D','S')}
assert all(c['failed_gates']=='stop1.speed' for c in fails)
out=dict(fixed_commit='7935a653b8f801fa946cbc6676e061a4438de39e',cases=20,physical_heldout_steps=12000,compatibility_steps=1200,groups=groups,probes=probes,candidate_heldout_passed=False,candidate_failed_cases=fails,independently_repeated_server_registry_scan=False,no_new_simulation_or_training=True)
(OUT/'H1-independent-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
with (OUT/'H1-cases.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cases[0]),lineterminator='\n');w.writeheader();w.writerows(cases)
print(json.dumps(dict(groups=groups,probes_exact=True,candidate_failures=fails)))
