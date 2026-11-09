"""R7P固定30例开发诊断；保留原物理评分，无晋级/留出/追加训练。"""
from pathlib import Path
from evaluation_r7p_runner import atomic,local,read,sha
from evaluation_r7p_helpers import physical_fingerprint
from r4_protocol import analyze_rows
import json,math
PLAN='SAC-R7P-PRECISION-PILOT-20261109'
PARENT_SHA='cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3'
SEEDS=(141,142,143)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def mean(v):return sum(v)/len(v)
def development_cases(model_path,model_sha256,evaluate_case):
 return [evaluate_case(model_path,model_sha256,'development','standard',e,s) for e in ('D','S') for s in SEEDS]
def validate_cases(root,model,cases,paired=None):
 require(len(cases)==6 and {(c['execution'],c['eval_seed']) for c in cases}=={(e,s) for e in ('D','S') for s in SEEDS},'Exactly six fixed cases required')
 require(sha(local(root,model['model_path']))==model['model_sha256'],'Model bytes changed')
 paired={} if paired is None else paired;results=[]
 for case in cases:
  cfg=read(local(root,case['configuration_path']));directory=local(root,case['configuration_path']).parent
  require(read(directory/'process-result.json').get('returncode')==0 and read(directory/'status.json')['status']=='passed','Case exit not confirmed')
  for k,v in dict(plan_id=PLAN,sha256=model['model_sha256'],condition='standard',seed_split='development',execution=case['execution'],evaluation_seed=case['eval_seed'],policy_noise='on',zero_reset_wrench=True,interval_push_disabled=True,tf32_matmul=False,tf32_cudnn=False,termination_event_source='current_compute_return').items():require(cfg[k]==v,'Evaluation contract '+k)
  require(read(local(root,case['frozen_before_path']))==read(local(root,case['frozen_after_path'])),'Evaluation changed frozen state')
  fp=physical_fingerprint(cfg['initial_state']);require(fp==cfg['physical_initial_state_sha256'],'Physical fingerprint corruption')
  seed=case['eval_seed']
  if seed in paired:require(paired[seed]==fp,'Unpaired actual initial physics')
  paired[seed]=fp
  require(set(cfg['rng_identity'])=={'python','numpy','torch','cuda','sample_generator'},'Missing independent RNG identity')
  rows=[json.loads(line) for line in local(root,case['steps_path']).read_text().splitlines() if line.strip()]
  results.append(dict(analyze_rows(rows,'standard'),execution=case['execution'],eval_seed=seed))
 return results

def fork_summary(cases):
 require(len(cases)==6,'Six cases per fork')
 complete=all(c['steps']==600 and not c['loss_is_failure_penalty'] for c in cases)
 p=yaw=mae=None
 if complete:
  p=mean([max(c['metrics'][w]['speed'] for w in ('stop1','stop2')) for c in cases])
  yaw=mean([max(c['metrics'][w]['abs_wz'] for w in ('stop1','stop2')) for c in cases])
  mae=mean([mean([c['metrics'][w]['vx_mae'] for w in ('forward','reverse')]) for c in cases])
 return dict(complete=complete,survival_count=sum(bool(c['survived']) for c in cases),pass_count=sum(bool(c['task_pass']) for c in cases),motion_count=sum(bool(c['motion_competent']) for c in cases),parking_P=p,parking_yaw=yaw,moving_mae=mae)
def summarize_direction(direction,arms):
 group=[a for a in arms if a['direction']==direction]
 require(len(group)==2 and {a['fork_seed'] for a in group}=={1401,1402},'Incomplete paired direction')
 forks={str(a['fork_seed']):fork_summary(a['results']) for a in group}
 values=list(forks.values());worst=lambda k:min(x[k] for x in values)
 maximum=lambda k:max(x[k] for x in values) if all(x[k] is not None for x in values) else None
 return dict(direction=direction,forks=forks,worst_fork_survival_count=worst('survival_count'),worst_fork_pass_count=worst('pass_count'),overall_pass_count=sum(x['pass_count'] for x in values),worst_fork_motion_count=worst('motion_count'),**{k:maximum(k) for k in ('parking_P','parking_yaw','moving_mae')})
def diagnostic_decision(parent,arms):
 p0=summarize_direction('P0',arms);p1=summarize_direction('P1',arms)
 finite=lambda v:v is not None and math.isfinite(v)
 checks={'both_P1_survival6':p1['worst_fork_survival_count']==6,'both_P1_motion6':p1['worst_fork_motion_count']==6,'worst_pass_not_lower':p1['worst_fork_pass_count']>=p0['worst_fork_pass_count'],'overall_pass_not_lower':p1['overall_pass_count']>=p0['overall_pass_count'],'moving_mae':finite(p1['moving_mae']) and finite(p0['moving_mae']) and p1['moving_mae']<=.12 and p1['moving_mae']<=p0['moving_mae']+.01,'parking_yaw':finite(p1['parking_yaw']) and finite(p0['parking_yaw']) and p1['parking_yaw']<=max(.10,p0['parking_yaw'])}
 eligible=all(finite(v) for v in (p1['parking_P'],p0['parking_P'],parent['parking_P'])) and p0['parking_P']>0
 benefit={'one_more_worst_fork_case':p1['worst_fork_pass_count']>=p0['worst_fork_pass_count']+1,'parking20pct_and_parent':bool(eligible and p1['parking_P']<=.8*p0['parking_P'] and p1['parking_P']<=parent['parking_P']),'parking_percentage_eligible':eligible}
 promising=all(checks.values()) and (benefit['one_more_worst_fork_case'] or benefit['parking20pct_and_parent'])
 # 分指标报告两fork信号方向相反，不将其混成一个有因果含义的分数。
 inconsistent=[]
 for metric in ('pass_count','parking_P','parking_yaw','moving_mae'):
  deltas=[]
  for fork in ('1401','1402'):
   a,b=p1['forks'][fork][metric],p0['forks'][fork][metric]
   if a is not None and b is not None:deltas.append(a-b)
  if len(deltas)==2 and deltas[0]*deltas[1]<0:inconsistent.append(metric)
 return dict(P0=p0,P1=p1,required_all=checks,benefit=benefit,promising_short_budget=promising,status='promising_short_budget' if promising else 'no_progress_short_budget',inconsistent=bool(inconsistent),inconsistent_metrics=inconsistent,promotion=False,independent_acceptance=False,heldout='not_run_by_scope',turn='not_run_by_scope',pulse='not_run_by_scope',future_action='Report hypothesis only; no automatic continuation or substitution')

def run_endpoint(root,parent,arms,evaluate_case):
 root=Path(root).resolve();require(parent['model_sha256']==PARENT_SHA,'Wrong frozen R6 parent')
 require(local(root,parent['model_path']).stat().st_size==9167667,'Wrong frozen parent size')
 require(len(arms)==4 and {(a['direction'],a['fork_seed'],a['learning_macros']) for a in arms}=={(d,f,48) for d in ('P0','P1') for f in (1401,1402)},'Require complete same-L48 four-arm endpoint')
 paired={};models=[]
 for item in [dict(parent,reference=True),*sorted(arms,key=lambda a:(a['direction'],a['fork_seed']))]:
  cases=development_cases(item['model_path'],item['model_sha256'],evaluate_case)
  model=dict(item,cases=cases,results=validate_cases(root,item,cases,paired));models.append(model)
  atomic(root/'endpoint-development.json',dict(plan_id=PLAN,scope='development_diagnostic_only',complete=len(models)==5,models=models,total_cases=sum(len(m['cases']) for m in models),independent_acceptance=False))
 decision=diagnostic_decision(fork_summary(models[0]['results']),models[1:])
 # 每个同seed/执行的原始结果都保留，避免只展示合并平均。
 matched=[{'execution':e,'seed':s,'models':[dict(model_sha256=m['model_sha256'],direction=m.get('direction','parent'),fork_seed=m.get('fork_seed'),result=next(r for r in m['results'] if r['execution']==e and r['eval_seed']==s)) for m in models]} for e in ('D','S') for s in SEEDS]
 result=dict(plan_id=PLAN,scope='development_diagnostic_only',complete=True,total_cases=30,models=models,decision=decision,matched_cases=matched,independent_acceptance=False)
 atomic(root/'endpoint-development.json',result);return result
