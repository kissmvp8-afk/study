"""独立标准库复算；不导入或运行证据包的训练/评价代码。"""
import csv,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence';O=R/'audit';O.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def avg(v):return sum(v)/len(v)
def save(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def finite(d):
 if isinstance(d,float):return math.isfinite(d)
 if isinstance(d,dict):return all(finite(v) for v in d.values())
 if isinstance(d,list):return all(finite(v) for v in d)
 return True
endpoint=read(E/'endpoint-development.json');bysha={m['model_sha256']:m for m in endpoint['models']}
cases=[];paired={};evaldur=[]
for path in sorted((E/'evaluations').rglob('steps.jsonl')):
 cfg=read(path.parent/'configuration.json');summ=read(path.parent/'summary.json')
 rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
 assert finite(rows) and len(rows)==600
 assert cfg['evaluation_seed'] in (141,142,143) and cfg['execution'] in ('D','S')
 for k,v in {'condition':'standard','seed_split':'development','policy_noise':'on','zero_reset_wrench':True,'interval_push_disabled':True,'tf32_matmul':False,'tf32_cudnn':False,'termination_event_source':'current_compute_return'}.items():assert cfg[k]==v
 assert read(path.parent/'frozen-before.json')==read(path.parent/'frozen-after.json')
 assert read(path.parent/'process-result.json')['returncode']==0
 fp=hashlib.sha256(json.dumps(cfg['initial_state'],sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
 assert fp==cfg['physical_initial_state_sha256']
 seed=cfg['evaluation_seed']
 if seed in paired:assert fp==paired[seed]
 paired[seed]=fp
 for i,x in enumerate(rows):
  assert x['step']==i and abs(x['t_start_s']-i*.02)<1e-8 and abs(x['t_end_s']-(i+1)*.02)<1e-8
  cmd=[0,0,0] if i<100 or 300<=i<400 else ([.3,0,0] if i<300 else [-.3,0,0])
  assert all(abs(a-b)<1e-7 for a,b in zip(cmd,x['effective_command']))
  assert x['termination_terms']=={'base_contact':x['terminated'],'time_out':x['time_out']}
  assert all(v==0 for body in x['external_force_b_command'] for v in body)
  assert all(v==0 for body in x['external_torque_b_command'] for v in body)
 metrics={};est={}
 for name,a,b in [('stop1',1,2),('stop2',7,8),('forward',2,6),('reverse',8,12),('height',2,12)]:
  ss=rows[round(a/.02):round(b/.02)];vv=[x['measured_body_vx_vy_wz'] for x in ss]
  metrics[name]={'vx':avg([v[0] for v in vv]),'vx_mae':avg([abs(v[0]-x['effective_command'][0]) for x,v in zip(ss,vv)]),'speed':avg([math.hypot(v[0],v[1]) for v in vv]),'abs_wz':avg([abs(v[2]) for v in vv]),'height':avg([x['root_pos_w'][2] for x in ss])}
  err=[x['encoder_velocity_mps'][0]-x['velocity_true_mps_before'][0] for x in ss]
  est[name]={'vx_mae':avg([abs(v) for v in err]),'vx_bias':avg(err)}
 survived=not any(x['terminated'] or x['time_out'] for x in rows) and max(x['contact_named_max_N']['base'] for x in rows)<=1
 failures=[]
 for stage in ('stop1','stop2'):
  if metrics[stage]['speed']>.05:failures.append(stage+'_speed')
  if metrics[stage]['abs_wz']>.10:failures.append(stage+'_yaw')
 for stage in ('forward','reverse'):
  if metrics[stage]['vx_mae']>.12:failures.append(stage+'_mae')
  if metrics[stage]['abs_wz']>.20:failures.append(stage+'_yaw')
 if metrics['height']['height']<.25:failures.append('height')
 if not survived:failures.append('survival')
 competent=survived and metrics['forward']['vx']>=.1 and metrics['reverse']['vx']<=-.1 and max(metrics[s]['vx_mae'] for s in ('forward','reverse'))<=.20
 model=bysha[cfg['sha256']];ref=next(c for c in model['cases'] if c['execution']==cfg['execution'] and c['eval_seed']==seed)['result']
 for s,mm in metrics.items():
  for k,v in mm.items():assert abs(v-ref['metrics'][s][k])<1e-10,(path,s,k)
 assert (not failures)==ref['task_pass'] and survived==ref['survived'] and competent==ref['motion_competent']
 record=dict(direction='parent' if model.get('reference') else model['direction'],fork=model['fork_seed'],seed=seed,execution=cfg['execution'],sha256=cfg['sha256'],survived=survived,passed=not failures,motion=competent,failures=failures,metrics=metrics,estimator=est,rows=600)
 cases.append(record)
 assert len({x['episode_id'] for x in rows})==1
 summarytime={k:v for k,v in summ.items() if isinstance(v,(int,float)) and ('time' in k or 'elapsed' in k or 'epoch' in k)}
 evaldur.append({'folder':str(path.parent.relative_to(E)),'summary_timing':summarytime})
assert len(cases)==30 and len(paired)==3
def aggregate(cc):
 return {'cases':len(cc),'passed':sum(c['passed'] for c in cc),'survived':sum(c['survived'] for c in cc),'motion':sum(c['motion'] for c in cc),'P':avg([max(c['metrics'][s]['speed'] for s in ('stop1','stop2')) for c in cc]),'yaw':avg([max(c['metrics'][s]['abs_wz'] for s in ('stop1','stop2')) for c in cc]),'moving_mae':avg([avg([c['metrics'][s]['vx_mae'] for s in ('forward','reverse')]) for c in cc]),'failures':dict(Counter(f for c in cc for f in c['failures']))}
groups={}
for direction,fork in sorted({(c['direction'],c['fork']) for c in cases}):groups[direction+'-'+str(fork)]=aggregate([c for c in cases if c['direction']==direction and c['fork']==fork])
deltas=[]
for fork in (1401,1402):
 a=groups['P0-'+str(fork)];b=groups['P1-'+str(fork)]
 deltas.append({'fork':fork,'P0':a,'P1':b,'P1_minus_P0_P':b['P']-a['P'],'P_relative_change':b['P']/a['P']-1,'pass_gain':b['passed']-a['passed']})
paired_pass=[]
for fork in (1401,1402):
 for seed in (141,142,143):
  for ex in ('D','S'):
   a=next(c for c in cases if (c['direction'],c['fork'],c['seed'],c['execution'])==('P0',fork,seed,ex))
   b=next(c for c in cases if (c['direction'],c['fork'],c['seed'],c['execution'])==('P1',fork,seed,ex))
   paired_pass.append({'fork':fork,'seed':seed,'execution':ex,'P0_pass':a['passed'],'P1_pass':b['passed'],'P0_failures':a['failures'],'P1_failures':b['failures']})
save('cases.json',cases);save('groups.json',groups);save('fork-differences.json',deltas);save('paired-pass.json',paired_pass);save('evaluation-time-fields.json',evaldur)
print('evaluation30 cases18000 steps matched',groups,flush=True)

training=[];allentries={};initialphys={};reward_counts=Counter();totals=Counter()
for arm in sorted((E/'arms').iterdir()):
 if not arm.is_dir():continue
 settings=read(arm/'effective-settings.json');status=read(arm/'STATUS.json');direction=arm.name.split('-')[0];fork=int(arm.name.split('-')[1]);w=0 if direction=='P0' else .5
 assert settings['parking_body_velocity_weight']==-8 and settings['command_resampling_time_range']==[2,4]
 assert settings['new_parking_precision_weight']==w and settings['replay_capacity_total']==65536 and settings['num_envs']==128 and settings['actor_lr']==5e-5
 assert settings['freeze_representation'] is False and settings['tf32_matmul'] is False and settings['tf32_cudnn'] is False
 assert status['status']=='completed'
 entry=read(next(arm.glob('entry-fingerprints-*.json')));allentries[arm.name]=entry
 initialphys[arm.name]=read(next(arm.glob('initial-physical-*.json')))
 assert entry['entry']['parent_sha256']=='cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3'
 assert abs(entry['entry']['effective_entry_alpha']-7.544342952314764e-5)<1e-12
 count=0;masked=0;extra_nonzero=0;firstcounter=None
 with (arm/'macros.jsonl').open() as f:
  for i,line in enumerate(f,1):
   m=json.loads(line);assert m['macro_iteration']==i and finite(m)
   L=max(0,i-16);c=m['counters']
   assert c['iterations']==i and c['transitions']==i*1024 and c['env_steps']==i*8
   for k,n in [('critic_updates',128),('actor_updates',64),('alpha_updates',128),('target_updates',128),('auxiliary_updates',128)]:assert c[k]==L*n,(arm,i,k)
   assert c['auxiliary_skipped']==c['learning_skipped_iterations']==0
   assert len(m['microcycles'])==(8 if L else 0)
   assert m['auxiliary_attempts_actual']==m['auxiliary_success_actual']==(128 if L else 0) and m['auxiliary_skipped_actual']==0
   control=m['diagnostics']['per_environment_control_steps'];assert len(control)==8
   for row in control:
    pr=row['precision'];assert pr['weight']==w and pr['dt']==.02
    assert row['command']==row['pre_reset_command']
    assert len(row['command'])==len(row['vx_vy_wz'])==128
    for j,(cmd,v) in enumerate(zip(row['command'],row['vx_vy_wz'])):
     mask=math.hypot(*cmd[:2])<.001 and abs(cmd[2])<.001
     cost=1-math.exp(-(v[0]**2+v[1]**2)/.1**2)+.25*(1-math.exp(-v[2]**2/.2**2))
     assert bool(pr['mask'][j])==mask
     assert abs(pr['cost'][j]-cost)<3e-6
     extra=-w*mask*cost*.02
     assert abs(pr['extra'][j]-extra)<4e-8
     assert abs(pr['total_reward'][j]-pr['original_reward'][j]-pr['extra'][j])<3e-6
     oldmask=math.hypot(*cmd[:2])<.05 and abs(cmd[2])<.05
     raw=oldmask*(v[0]**2+v[1]**2+.25*v[2]**2)
     assert abs(row['parking_raw'][j]-raw)<max(2e-6,abs(raw)*3e-6)
     assert abs(row['parking_weighted_dt'][j]+8*raw*.02)<max(2e-6,abs(raw)*5e-7)
     count+=1;masked+=int(mask);extra_nonzero+=int(abs(pr['extra'][j])>1e-10)
   assert abs(m['diagnostics']['parking_weight']+8)<1e-10
 assert i==64 and count==65536 and c==status['counters']
 assert masked>0 and (extra_nonzero>0 if w else extra_nonzero==0)
 timing=[json.loads(x) for x in (arm/'timings.jsonl').read_text().splitlines()]
 learning=[t['wall_s'] for t in timing if t['learning']]
 training.append({'arm':arm.name,'counters':c,'reward_transitions_checked':count,'precision_mask_count':masked,'precision_nonzero_count':extra_nonzero,'learning_wall_sum':sum(learning),'learning_macro_mean':avg(learning),'learning_macro_p95':statistics.quantiles(learning,n=20)[18],'settings':settings})
 for k in ['transitions','critic_updates','actor_updates','alpha_updates','target_updates','auxiliary_updates','auxiliary_skipped']:totals[k]+=c[k]
 print('training arm checked',arm.name,count,'rewards',flush=True)
assert len(training)==4
for fork in (1401,1402):
 a=allentries['P0-'+str(fork)];b=allentries['P1-'+str(fork)]
 for k in ['modules','optimizers','normalizers']:assert a[k]==b[k],(fork,k)
 assert initialphys['P0-'+str(fork)]['sha256']==initialphys['P1-'+str(fork)]['sha256']
controller=read(E/'controller-state.json');assert controller['status']=='training_completed'
assert abs(controller['elapsed_seconds']-423.53247809410095)<1e-9
assert abs(controller['completed_epoch']-controller['started_epoch']-controller['elapsed_seconds'])<1e-5
save('training.json',{'arms':training,'totals':dict(totals),'parent_pair_entry_fingerprints_match':True,'pair_initial_physics_fingerprints_match':True,'controller':controller,'source':'full raw archive, not earlier partial Git snapshot'})
print('training counters and262144 reward contributions independently verified',flush=True)
gpu=read(E/'resources/learning-gpu-summary.json')
with (E/'resources/gpu-samples.csv').open() as f:rr=list(csv.DictReader(f))
save('summary.json',{'input_commit':read(R/'input.json')['input_commit'],'raw_archive_sha256':read(R/'download-verification.json')['archive_sha256'],'manifest_files_verified':575,'evaluation_cases':30,'evaluation_steps':18000,'training_macros':256,'training_learning_macros':192,'training_transitions':262144,'reward_contributions_recomputed':262144,'groups':groups,'fork_differences':deltas,'training_totals':dict(totals),'gpu_report':gpu,'gpu_csv_header':list(rr[0]),'stale_git_controller_index':True,'no_gpu_or_simulation_run_by_reviewer':True})
