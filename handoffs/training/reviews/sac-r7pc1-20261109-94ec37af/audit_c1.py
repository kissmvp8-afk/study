"""独立读取固定C1证据；不执行归档代码，不启动仿真/训练。"""
import csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
R=Path(__file__).resolve().parent;E=R/'evidence';O=R/'audit';O.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return True
handoff=read(E/'c1-final-handoff.json');pstate=read(E/'parallel-state.json');serial=read(E/'controller-state-before-parallel.json');cfg=read(E/'confirm-config.json')
assert handoff['complete'] and len(pstate['cases'])==50 and pstate['status']=='completed'
assert (E/'controller-state.json').read_bytes()==(E/'controller-state-before-parallel.json').read_bytes()
assert sha(E/'controller-state-before-parallel.json')==read(E/'parallel-takeover.json')['original_state_sha256']
serial_paths={row['case']['steps_path'] for row in serial['cases']}
for child in read(E/'parallel-takeover.json')['child_exits']:
 assert child['returncode']==0
 absolute=child['identity']['output'];relative=absolute.split('/20261109T113422Z/',1)[1]
 serial_paths.add(relative+'/steps.jsonl')
assert len(serial_paths)==pstate['cached_cases']==14
models={m['model_sha256']:m for m in cfg['models']}
index={row['case']['steps_path']:row for row in pstate['cases']}
assert len(index)==50
identities={(row['model']['model_sha256'],row['case']['execution'],row['case']['eval_seed']) for row in pstate['cases']}
assert identities=={(sha,e,s) for sha in models for e in ('D','S') for s in range(231,236)}
paired={};rng_pairs={};cases=[];failures=[];intervals=[];source_ids=[]
for path,desc in sorted(index.items()):
 folder=E/Path(path).parent;cc=read(folder/'configuration.json');ss=read(folder/'summary.json');rows=[json.loads(x) for x in (E/path).read_text().splitlines() if x.strip()]
 model=desc['model'];case=desc['case'];n=len(rows);assert finite(rows) and n in (29,600)
 for k,v in {'sha256':model['model_sha256'],'phase_id':'SAC-R7P-C1-FROZEN-CONFIRM-20261109','condition':'standard','seed_split':'development','execution':case['execution'],'evaluation_seed':case['eval_seed'],'policy_noise':'on','zero_reset_wrench':True,'interval_push_disabled':True,'tf32_matmul':False,'tf32_cudnn':False,'termination_event_source':'current_compute_return'}.items():assert cc[k]==v,(path,k)
 assert read(folder/'process-result.json')['returncode']==0 and read(folder/'status.json')['status']=='passed'
 assert read(folder/'frozen-before.json')==read(folder/'frozen-after.json')
 assert ss['checkpoint_sha256']==model['model_sha256'] and ss['checkpoint_unchanged'] and ss['normalizers_unchanged'] and ss['frozen_state_unchanged']
 assert ss['steps']==n and abs(ss['simulated_seconds']-.02*n)<1e-8
 fp=hashlib.sha256(json.dumps(cc['initial_state'],sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
 assert fp==cc['physical_initial_state_sha256']==case['physical_initial_state_sha256']
 seed=case['eval_seed']
 if seed in paired:assert paired[seed]==fp
 paired[seed]=fp
 identity=(case['execution'],seed)
 if identity in rng_pairs:assert cc['rng_identity']==rng_pairs[identity],(path,'RNG pairing')
 rng_pairs[identity]=cc['rng_identity']
 for i,x in enumerate(rows):
  assert x['step']==i and abs(x['t_start_s']-.02*i)<1e-8 and abs(x['t_end_s']-.02*(i+1))<1e-8
  command=[0,0,0] if i<100 or 300<=i<400 else ([.3,0,0] if i<300 else [-.3,0,0])
  assert max(abs(a-b) for a,b in zip(command,x['effective_command']))<1e-7
  assert x['termination_terms']=={'base_contact':x['terminated'],'time_out':x['time_out']}
  assert all(v==0 for b in x['external_force_b_command'] for v in b) and all(v==0 for b in x['external_torque_b_command'] for v in b)
  if i<n-1:assert not x['terminated'] and not x['time_out']
 assert n==600 or rows[-1]['terminated']
 metrics={}
 for stage,a,b in [('stop1',1,2),('stop2',7,8),('forward',2,6),('reverse',8,12),('height',2,12)]:
  window=rows[round(a/.02):round(b/.02)]
  if len(window)!=round((b-a)/.02):metrics[stage]=None;continue
  metrics[stage]={'speed':mean(math.hypot(*x['measured_body_vx_vy_wz'][:2]) for x in window),'vx':mean(x['measured_body_vx_vy_wz'][0] for x in window),'vx_mae':mean(abs(x['measured_body_vx_vy_wz'][0]-x['effective_command'][0]) for x in window),'abs_wz':mean(abs(x['measured_body_vx_vy_wz'][2]) for x in window),'height':mean(x['root_pos_w'][2] for x in window)}
 survived=n==600 and not any(x['terminated'] or x['time_out'] for x in rows) and max(x['contact_named_max_N']['base'] for x in rows)<=1
 bad=[]
 for s in ['stop1','stop2']:
  if metrics[s] is None:bad.append(s+'_unobserved')
  else:
   if metrics[s]['speed']>.05:bad.append(s+'_speed')
   if metrics[s]['abs_wz']>.1:bad.append(s+'_yaw')
 for s in ['forward','reverse']:
  if metrics[s] is None:bad.append(s+'_unobserved')
  else:
   if metrics[s]['vx_mae']>.12:bad.append(s+'_mae')
   if metrics[s]['abs_wz']>.2:bad.append(s+'_yaw')
 if not survived:bad.append('survival')
 if metrics['height'] is None:bad.append('height_unobserved')
 elif metrics['height']['height']<.25:bad.append('height')
 competent=survived and metrics['forward'] is not None and metrics['reverse'] is not None and metrics['forward']['vx']>=.1 and metrics['reverse']['vx']<=-.1 and max(metrics[s]['vx_mae'] for s in ['forward','reverse'])<=.2
 reported=case['result'];assert reported['steps']==n and reported['survived']==survived and reported['task_pass']==(not bad) and reported['motion_competent']==competent
 for stage,m in metrics.items():
  if m is None:assert reported['metrics'][stage] is None
  else:
   for k,v in m.items():assert abs(v-reported['metrics'][stage][k])<1e-10,(path,stage,k)
 mode='serial' if path in serial_paths else 'four_lane'
 record={'direction':model['direction'],'fork':model['fork_seed'],'seed':seed,'execution':case['execution'],'model_sha256':model['model_sha256'],'path':path,'mode':mode,'steps':n,'survived':survived,'passed':not bad,'motion':competent,'failures':bad,'metrics':metrics}
 cases.append(record)
 intervals.append({'path':path,'mode':mode,'start':cc['rollout_ready_unix'],'end':ss['rollout_finished_unix']})
 source_ids.append(cc['source_identity'])
 if n<600:
  last=rows[-1];first=next(i for i,x in enumerate(rows) if x['contact_named_max_N']['base']>1)
  assert model['direction']=='P1' and model['fork_seed']==1401 and seed==231 and n==29
  assert last['termination_terms']=={'base_contact':True,'time_out':False} and first==28
  failures.append({'path':path,'mode':mode,'seed':seed,'execution':case['execution'],'steps':n,'elapsed_sim_s':n*.02,'terminal_step':last['step'],'terminal_terms':last['termination_terms'],'base_contact_history_peak_N':last['contact_named_max_N']['base'],'height_m':last['root_pos_w'][2],'body_tilt_deg':math.degrees(math.acos(max(-1,min(1,-last['gravity_b'][2])))),'wheel_targets_rad_s':last['wheel_target_rad_s'],'external_command_force_zero':True,'first_base_contact_step':first})
assert len(cases)==50 and sum(c['steps'] for c in cases)==28858 and len(failures)==2
assert Counter(c['mode'] for c in cases)=={'serial':14,'four_lane':36}
groups={}
for direction,fork in sorted({(c['direction'],c['fork']) for c in cases}):
 cc=[c for c in cases if (c['direction'],c['fork'])==(direction,fork)];assert len(cc)==10
 complete=all(c['steps']==600 for c in cc)
 groups[direction+'-'+str(fork)]={'cases':10,'survived':sum(c['survived'] for c in cc),'passed':sum(c['passed'] for c in cc),'motion':sum(c['motion'] for c in cc),'P':mean(max(c['metrics'][s]['speed'] for s in ('stop1','stop2')) for c in cc) if complete else None,'yaw':mean(max(c['metrics'][s]['abs_wz'] for s in ('stop1','stop2')) for c in cc) if complete else None,'moving_mae':mean(mean(c['metrics'][s]['vx_mae'] for s in ('forward','reverse')) for c in cc) if complete else None,'modes':dict(Counter(c['mode'] for c in cc)),'failures':dict(Counter(b for c in cc for b in c['failures']))}
for reported in handoff['summary']['models']:
 model=reported['model'];g=groups[model['direction']+'-'+str(model['fork_seed'])]
 assert g['passed']==reported['pass_count']
 for k,j in [('P','parking_P'),('yaw','parking_yaw'),('moving_mae','moving_mae')]:
  assert (g[k] is None and reported[j] is None) or (g[k] is not None and abs(g[k]-reported[j])<1e-12)
effects=[]
for f in (1401,1402):
 a,b=groups['P0-'+str(f)],groups['P1-'+str(f)]
 available=all(x[k] is not None for x in (a,b) for k in ['P','yaw','moving_mae'])
 checks={'complete_metrics':available,'parking_improves_005':available and b['P']<=a['P']-.005,'passes_not_lower':b['passed']>=a['passed'],'moving_gate':available and b['moving_mae']<=.12 and b['moving_mae']<=a['moving_mae']+.01,'yaw_gate':available and b['yaw']<=max(.1,a['yaw'])}
 effects.append({'fork':f,'checks':checks,'effect_reproduced':all(checks.values())})
assert [x['effect_reproduced'] for x in effects]==[False,True]
old=Path('/workspace/sac-r7p-review/evidence/evaluations/cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3/development/standard-D-141/attempt-1')
new=E/'evaluations/cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3/development/standard-D-141/attempt-1'
assert (old/'steps.jsonl').read_bytes()==(new/'steps.jsonl').read_bytes()
assert sum(1 for _ in (new/'steps.jsonl').open())==600
source_equal=all(x==source_ids[0] for x in source_ids)
for f in failures:
 interval=next(x for x in intervals if x['path']==f['path'])
 f['other_rollout_intervals_overlapping']=[x['path'] for x in intervals if x['path']!=f['path'] and max(x['start'],interval['start'])<min(x['end'],interval['end'])]
frozen=read(E/'frozen-inputs.json');source_records=[]
root=cfg['workroot'].rstrip('/')+'/'
for f in frozen:
 if f['path'].startswith(root):
  rel=f['path'][len(root):];p=E/rel
  if p.exists():assert sha(p)==f['sha256'];source_records.append({'path':rel,'status':'sha_match','sha256':f['sha256']})
  else:source_records.append({'path':rel,'status':'not_in_archive'})
 else:source_records.append({'path':f['path'],'status':'external_dependency_not_in_archive'})
save('cases.json',cases);save('groups.json',groups);save('terminal-failures.json',failures);save('paired-effects.json',effects);save('mode-case-map.json',intervals);save('source-inputs.json',source_records)
summary={'input_commit':read(R/'input.json')['commit'],'archive':read(R/'archive-verification.json'),'standard_cases':50,'standard_physical_steps':28858,'legacy_parity_steps':600,'total_physical_steps_read':29458,'all_frozen_checks_pass':True,'same_source_identity_all50':source_equal,'unique_case_count':len(identities),'mode_counts':dict(Counter(c['mode'] for c in cases)),'groups':groups,'effect_reproduced':False,'standard_replication':False,'legacy_step_bytes_equal':True,'cross_mode_same_model_seed_pair_in_original50':False,'cross_mode_numerical_equivalence_proven':False,'new_training_or_simulation_started_by_reviewer':False,'zero_update_evidence':'Evaluation-only sources, unchanged model hashes and all frozen-before/after fingerprints; optimizer updates not independently instrumented'}
save('summary.json',summary)
with (O/'case-table.csv').open('w',newline='') as f:
 basic=['direction','fork','seed','execution','mode','steps','survived','passed']
 derived={'stop1_speed':('stop1','speed'),'stop2_speed':('stop2','speed'),'stop1_yaw':('stop1','abs_wz'),'stop2_yaw':('stop2','abs_wz'),'forward_mae':('forward','vx_mae'),'reverse_mae':('reverse','vx_mae')}
 fields=basic+['failures']+list(derived);w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for c in cases:w.writerow({k:c[k] for k in basic}|{'failures':';'.join(c['failures'])}|{k:(c['metrics'][s][m] if c['metrics'][s] is not None else None) for k,(s,m) in derived.items()})
print(json.dumps({'groups':groups,'effects':effects,'failures':failures,'source_identity_all_equal':source_equal,'case_counts':summary['mode_counts']},ensure_ascii=False,indent=2))
