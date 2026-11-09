"""Independent C4 read-only supplement. Never replace saved scores or original summary."""
import argparse, hashlib, json
from pathlib import Path
import summarize_l24_l48 as base
KEYS=('P','W','Mavg','Mmax','moving_wz_max','height','forward_mae','reverse_mae','forward_abswz','reverse_abswz','stop1_speed','stop2_speed','stop1_abswz','stop2_abswz')
def details(score):
 m=score['metrics'];v={k:None for k in KEYS};failed=[];missing=[]
 for stage,field,key,threshold,minimum in [('forward','vx_mae','forward_mae',.12,False),('reverse','vx_mae','reverse_mae',.12,False),('forward','abs_wz','forward_abswz',.20,False),('reverse','abs_wz','reverse_abswz',.20,False),('stop1','speed','stop1_speed',.05,False),('stop2','speed','stop2_speed',.05,False),('stop1','abs_wz','stop1_abswz',.10,False),('stop2','abs_wz','stop2_abswz',.10,False),('height','height','height',.25,True)]:
  x=m.get(stage);value=x.get(field) if x else None;v[key]=value
  if value is None:missing.append(key)
  elif (value<threshold if minimum else value>threshold):failed.append(dict(gate=key,value=value,threshold=threshold,required_comparison='>=' if minimum else '<='))
 for key,parts,op in [('P',('stop1_speed','stop2_speed'),max),('W',('stop1_abswz','stop2_abswz'),max),('Mavg',('forward_mae','reverse_mae'),lambda a:sum(a)/2),('Mmax',('forward_mae','reverse_mae'),max),('moving_wz_max',('forward_abswz','reverse_abswz'),max)]:
  if all(v[k] is not None for k in parts):v[key]=op([v[k] for k in parts])
 if not score['survived']:failed.append(dict(gate='survival',steps=score['steps'],base_contact_max_N=score.get('base_contact_max_N')))
 if score['steps']!=600:failed.append(dict(gate='full_600_steps',steps=score['steps']))
 expected=not failed and not missing
 if expected!=score['task_pass']:raise ValueError('Saved score disagrees with complete unchanged gate explanation')
 return dict(values=v,failed_gates=failed,unavailable_gates=missing,survived=score['survived'],task_pass=score['task_pass'],steps=score['steps'],original_score=score)
def aggregate(cases):
 full=len(cases)==10 and all(c['steps']==600 and not c['original_score']['loss_is_failure_penalty'] for c in cases)
 return dict(n=len(cases),survived=sum(c['survived'] for c in cases),passed=sum(c['task_pass'] for c in cases),full_stage_means_available=full,means={k:sum(c['values'][k] for c in cases)/10 if full and all(c['values'][k] is not None for c in cases) else None for k in KEYS})
def subtract(a,b):
 return {k:a['means'][k]-b['means'][k] if a['means'][k] is not None and b['means'][k] is not None else None for k in KEYS}
def raw_context(directory):
 rows=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines() if x.strip()];meta=base.read(directory/'metadata.json');cfg=base.read(directory/'configuration.json')
 return dict(directory=str(directory),steps_sha256=base.sha(directory/'steps.jsonl'),terminal_events=[dict(step_index=r['step_index'],t_end_s=r['t_end_s'],terminated=r['terminated'],time_out=r['time_out'],base_contact_N=r['contact_named_max_N']['base']) for r in rows if r['terminated'] or r['time_out']],base_contact_peak_N=max(r['contact_named_max_N']['base'] for r in rows),rollout_ready_unix=meta.get('rollout_ready_unix',cfg.get('rollout_ready_unix')),rollout_finished_unix=meta.get('rollout_finished_unix'),argv=meta.get('argv'),physical_initial_state_sha256=cfg['physical_initial_state_sha256'])
def run(config):
 comparison=base.summarize(config)
 if not comparison['complete']:raise ValueError('Requires all40 validated pairs; original comparison stays unchanged')
 c1=Path(config['c1_root']);state=base.read(c1/'parallel-state.json');old={};groups={};identities={}
 for row in state['cases']:
  m,c=row['model'],row['case'];key=(m['direction'],m['fork_seed'],c['eval_seed'],c['execution']);old[key]=(m,c,(c1/c['configuration_path']).parent)
 for arm in comparison['arms']:
  d,f=arm['direction'],arm['fork_seed']
  for pair in arm['paired_cases']:
   s,e=pair['seed'],pair['execution'];m,c,directory=old[d,f,s,e]
   for learning in (24,48):
    if learning==24:
     scope=next(k for k,(seeds,execution) in base.SCOPES.items() if s in seeds and execution==e);directory24=Path(config['l24_roots'][scope])/'cases'/f'{d}-{f}-L24-{s}-{e}'
    path=directory24 if learning==24 else directory
    entry=dict(direction=d,fork_seed=f,learning=learning,seed=s,execution=e,**details(pair['L'+str(learning)]),**raw_context(path))
    groups.setdefault(f'{d}/{f}/L{learning}',[]).append(entry)
 for (d,f,s,e),(m,c,path) in old.items():
  if (d,f) in base.ARMS:continue
  valid=base.validated(path,c['result'],m['model_sha256'],s,e);mp=Path(m['model_path']);mp=mp if mp.is_absolute() else c1/mp
  if base.sha(mp)!=m['model_sha256']:raise ValueError('Parent model SHA changed')
  groups.setdefault('parent',[]).append(dict(direction=d,fork_seed=f,learning=48,seed=s,execution=e,**details(c['result']),**raw_context(path)))
 if len(groups.get('parent',[]))!=10:raise ValueError('Exactly10 parent cases required, no replicated parent')
 expected={(s,e) for s in range(231,236) for e in ('D','S')}
 for key,cases in groups.items():
  if len(cases)!=10 or {(c['seed'],c['execution']) for c in cases}!=expected:raise ValueError('Incomplete or duplicate case set '+key)
 summaries={k:aggregate(v) for k,v in groups.items()};contrasts=[]
 for f in (1401,1402):
  for learning in (24,48):
   a,b=f'P1/{f}/L{learning}',f'P0/{f}/L{learning}';contrasts.append(dict(kind='same_fork_P1_minus_P0',fork=f,learning=learning,mean_deltas=subtract(summaries[a],summaries[b]),survival_delta=summaries[a]['survived']-summaries[b]['survived'],pass_delta=summaries[a]['passed']-summaries[b]['passed']))
  for d in ('P0','P1'):
   a,b=f'{d}/{f}/L48',f'{d}/{f}/L24';contrasts.append(dict(kind='same_candidate_L48_minus_L24',direction=d,fork=f,mean_deltas=subtract(summaries[a],summaries[b]),by_seed_execution=[dict(seed=x['seed'],execution=x['execution'],deltas={k:x['values'][k]-y['values'][k] if x['values'][k] is not None and y['values'][k] is not None else None for k in KEYS}) for x in groups[a] for y in groups[b] if (x['seed'],x['execution'])==(y['seed'],y['execution'])]))
 correction=next(c for c in groups['P0/1401/L24'] if c['seed']==232 and c['execution']=='S')
 return dict(status='complete_C4_metric_supplement_not_new_acceptance',previous_summary_commit='d504e397ca0b07200abe0f8eb5195773c0ae799d',pair_validation=comparison['status'],input_evidence=comparison['input_evidence'],groups={k:dict(summary=summaries[k],cases=v) for k,v in groups.items()},contrasts=contrasts,explicit_omission_correction=correction,original_scores_unchanged=True,training_updates=0,limitations=['Exploratory shared seeds, not independent validation or causal reward evidence.','Scheduling timestamps are copied when present; missing remains null and does not prove concurrency equivalence.','C2 archive/retry evidence and full archive audit are separate, not certified by this supplement.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Never overwrite existing summary')
 result=run(base.read(a.config));out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({k:v['summary'] for k,v in result['groups'].items()},indent=2))
