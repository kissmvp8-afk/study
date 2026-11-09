"""CPU-only, read-only evidence comparison. Consumes saved scores; never re-scores C1."""
import argparse,hashlib,json,math
from pathlib import Path
ARMS=[(d,f) for d in ('P0','P1') for f in (1401,1402)]
SEEDS=(231,232,233,234,235)
SCOPES={'231D':((231,),'D'),'231S':((231,),'S'),'232233D':((232,233),'D'),'232233S':((232,233),'S'),'234235D':((234,235),'D'),'234235S':((234,235),'S')}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def require(test,message):
 if not test:raise ValueError(message)
def finite(v):
 if isinstance(v,float):require(math.isfinite(v),'Nonfinite metric')
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
def full(score):return score['steps']==600 and not score['loss_is_failure_penalty']
def values(score):
 finite(score)
 if not full(score):return dict(parking_P=None,parking_yaw=None,moving_mae=None,stop1_speed=None,stop2_speed=None,stop1_abswz=None,stop2_abswz=None,forward_mae=None,reverse_mae=None)
 m=score['metrics']
 return dict(parking_P=max(m[w]['speed'] for w in ('stop1','stop2')),parking_yaw=max(m[w]['abs_wz'] for w in ('stop1','stop2')),moving_mae=(m['forward']['vx_mae']+m['reverse']['vx_mae'])/2,
  stop1_speed=m['stop1']['speed'],stop2_speed=m['stop2']['speed'],stop1_abswz=m['stop1']['abs_wz'],stop2_abswz=m['stop2']['abs_wz'],forward_mae=m['forward']['vx_mae'],reverse_mae=m['reverse']['vx_mae'])
def aggregate(cases):
 complete=len(cases)==10;all_full=complete and all(full(c['score']) for c in cases);v=[values(c['score']) for c in cases]
 result=dict(observed_cases=len(cases),required_cases=10,complete_case_set=complete,all_trajectories_full=all_full,pass_count=sum(bool(c['score']['task_pass']) for c in cases),survival_count=sum(bool(c['score']['survived']) for c in cases),counts_are_partial=not complete,
  continuous_mean_rule='All ten cases must have600 steps with no failure penalty; otherwise all continuous aggregate means are NULL. Never average survivors only.')
 keys=('parking_P','parking_yaw','moving_mae','stop1_speed','stop2_speed','stop1_abswz','stop2_abswz','forward_mae','reverse_mae')
 result.update({k:sum(x[k] for x in v)/10 if all_full else None for k in keys});return result

def validated(directory,score,model_sha,seed,execution):
 directory=Path(directory);cfg=read(directory/'configuration.json')
 require(read(directory/'process-result.json').get('returncode')==0 and read(directory/'status.json').get('status')=='passed','Case process/status not confirmed')
 for k,v in {'sha256':model_sha,'evaluation_seed':seed,'execution':execution,'condition':'standard','seed_split':'development','policy_noise':'on','zero_reset_wrench':True,'interval_push_disabled':True,'tf32_matmul':False,'tf32_cudnn':False}.items():require(cfg[k]==v,'Case contract mismatch '+k)
 require(read(directory/'frozen-before.json')==read(directory/'frozen-after.json'),'Evaluation mutated frozen state')
 steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines() if x.strip()]
 require(len(steps)==score['steps'],'Saved score step count differs from raw evidence')
 require(all(not x.get('terminated') and not x.get('time_out') for x in steps[:-1]),'Trajectory continued after true termination')
 if score['survived']:require(len(steps)==600 and not any(x.get('terminated') or x.get('time_out') for x in steps),'Saved survival contradicts raw evidence')
 initial=hashlib.sha256(json.dumps(cfg['initial_state'],sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest();require(initial==cfg['physical_initial_state_sha256'],'Physical fingerprint corrupted')
 finite(score)
 return dict(directory=str(directory),score=score,config=cfg,steps_sha256=sha(directory/'steps.jsonl'),configuration_sha256=sha(directory/'configuration.json'),frozen_before_sha256=sha(directory/'frozen-before.json'),frozen_after_sha256=sha(directory/'frozen-after.json'),metadata=read(directory/'metadata.json'),repair=read(directory/'kit-command-repair.json'))
def paired(a,b):
 ca,cb=a['config'],b['config']
 keys=('evaluation_seed','execution','initial_state','physical_initial_state_sha256','rng_identity','source_identity','env','factor','policy_noise','zero_reset_wrench','interval_push_disabled','tf32_matmul','tf32_cudnn','dimensions','termination_event_source','contact_body_order','contact_criterion')
 for key in keys:require(ca[key]==cb[key],'L24/L48 pairing mismatch '+key)
 require(a['metadata']['script_sha256']==b['metadata']['script_sha256'],'Evaluator source differs')
 for key in ('original_evaluator_sha256','provider_manifest_sha256','wrapper_sha256'):require(a['repair'][key]==b['repair'][key],'Kit source differs '+key)
 va,vb=values(a['score']),values(b['score'])
 return dict(seed=ca['evaluation_seed'],execution=ca['execution'],L24=a['score'],L48=b['score'],L48_minus_L24={key:(vb[key]-va[key]) if va[key] is not None and vb[key] is not None else None for key in va},pass_change=int(b['score']['task_pass'])-int(a['score']['task_pass']),survival_change=int(b['score']['survived'])-int(a['score']['survived']),L24_steps_sha256=a['steps_sha256'],L48_steps_sha256=b['steps_sha256'],protocol_pairing_passed=True)

def summarize(cfg):
 c1=Path(cfg['c1_root']).resolve();r7p=Path(cfg['r7p_root']).resolve();oldstate=read(c1/'parallel-state.json');require(oldstate['status']=='completed' and len(oldstate['cases'])==50,'Require original complete C1 state, never substitute reruns')
 originals={};inputs=[dict(path=str(c1/'parallel-state.json'),sha256=sha(c1/'parallel-state.json'))]
 for row in oldstate['cases']:
  model=row['model'];identity=(model.get('direction'),model.get('fork_seed'))
  if identity not in ARMS:continue
  case=row['case'];key=(*identity,case['eval_seed'],case['execution']);require(key not in originals,'Duplicate C1 identity');originals[key]=(model,case)
 require(len(originals)==40,'Require all four C1 candidate models x10')
 l24={};missing=[];invalid=[]
 require(set(cfg['l24_roots'])==set(SCOPES),'Exactly six named L24 nodes required')
 for scope,(seeds,execution) in SCOPES.items():
  root=Path(cfg['l24_roots'][scope]).resolve();p=root/'controller-state.json'
  if not p.exists():missing.append(dict(scope=scope,reason='node state missing'));continue
  state=read(p);inputs.append(dict(path=str(p),sha256=sha(p)))
  for row in state.get('cases',[]):
   model=row.get('model') or row['task']['model'];seed=row.get('task',{}).get('seed',seeds[0]);d,f=model['direction'],model['fork_seed']
   require(seed in seeds and (d,f) in ARMS,'Node case outside fixed scope')
   key=(d,f,seed,execution);require(key not in l24,'Duplicate L24 identity')
   if not row.get('technical_pass',False):missing.append(dict(key=key,reason='technical failure or incomplete case'));continue
   l24[key]=(root/'cases'/f'{d}-{f}-L24-{seed}-{execution}',row['metrics'],model)
 report=dict(status='incomplete',scope='Four fixed L24 candidates versus original C1 L48, same five seeds x D/S; D/S are not independent training seeds.',input_evidence=inputs,arms=[],missing=missing,invalid=invalid,original_C1_scores_not_recomputed_or_replaced=True,training_updates=0)
 for d,f in ARMS:
  node=read(r7p/'arms'/f'{d}-{f}'/'node-learning-24.json')['model'];model24=r7p/'arms'/f'{d}-{f}'/'model-learning-24.pt';require(sha(model24)==node['sha256'],'L24 bytes differ from node')
  cases24=[];cases48=[];pairs=[]
  for seed in SEEDS:
   for execution in ('D','S'):
    key=(d,f,seed,execution);model48,case48=originals[key]
    modelpath=Path(model48['model_path']);modelpath=modelpath if modelpath.is_absolute() else c1/modelpath;require(sha(modelpath)==model48['model_sha256'],'C1 model bytes changed')
    old=validated((c1/case48['configuration_path']).parent,case48['result'],model48['model_sha256'],seed,execution);cases48.append(old)
    if key not in l24:report['missing'].append(dict(key=key,reason='no authoritative L24 score'));continue
    directory,score,model=l24[key]
    try:
     require(model['sha256']==node['sha256'],'L24 model identity mismatch')
     new=validated(directory,score,node['sha256'],seed,execution);pair=paired(new,old);cases24.append(new);pairs.append(pair)
    except (ValueError,KeyError,FileNotFoundError) as exc:invalid.append(dict(key=key,error=str(exc)))
  s24,s48=aggregate(cases24),aggregate(cases48)
  delta={k:(s48[k]-s24[k]) if s24[k] is not None and s48[k] is not None else None for k in ('parking_P','parking_yaw','moving_mae')}
  report['arms'].append(dict(direction=d,fork_seed=f,L24=s24,L48=s48,paired_cases=pairs,L48_minus_L24=delta,P1_degradation_descriptive=(dict(survival_lost=sum(p['survival_change']<0 for p in pairs),passes_lost=sum(p['pass_change']<0 for p in pairs),passes_gained=sum(p['pass_change']>0 for p in pairs),complete=len(pairs)==10,no_causal_claim=True) if d=='P1' else None)))
 report['complete']=all(len(x['paired_cases'])==10 for x in report['arms']) and not invalid
 report['status']='invalid_pairing' if invalid else 'complete_fair_comparison' if report['complete'] else 'incomplete_no_full_conclusion'
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Do not overwrite prior summary')
 result=summarize(read(a.config));out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False));print(result['status'])
