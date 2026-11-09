"""R7P最小四路：错峰启动、独立worker、16prefill屏障、24/48节点与半小时记录。"""
import argparse,fcntl,os,signal,subprocess,time,shutil
from pathlib import Path
from r7p_ops import *

def release(root,token,runtime):
 directory=root/'barriers'/token;paths={d+'-'+str(f):directory/(d+'-'+str(f)+'.ready.json') for d,f in ARMS}
 if not all(p.is_file() for p in paths.values()):return False
 identities=[]
 for name,p in paths.items():
  x=read(p);identity=dict(x['identity']);d,f=name.split('-')
  if identity.pop('direction')!=d or identity.pop('fork_seed')!=int(f) or identity['plan_id']!=PLAN or x['barrier_id']!=token or x['counters']['iterations']!=16:raise ValueError('Wrong ready identity/prefill count')
  for k in ('critic_updates','actor_updates','alpha_updates','target_updates','auxiliary_updates','auxiliary_skipped'):
   if x['counters'].get(k)!=0:raise ValueError('Learning occurred before common prefill release')
  identities.append(identity)
 if any(x!=identities[0] for x in identities):raise ValueError('Worker source/parent/config identities differ')
 obj=dict(barrier_id=token,ready_sha256={n:sha(p) for n,p in paths.items()})
 target=directory/'release.json'
 if target.exists():
  if read(target)!=obj:raise ValueError('Barrier release changed')
 else:atomic(target,obj)
 return True

def progress(root,runtime,started,status):
 arms={}
 for d,f in ARMS:
  base=root/'arms'/('%s-%d'%(d,f));one={}
  for name in ('STATUS.json','latest-fullresume.json'):
   p=base/name
   if p.is_file():one[name]=read(p)
  one['model_nodes_present']=[s for s in (24,48) if (base/('model-learning-%d.pt'%s)).is_file()]
  arms[d+'-'+str(f)]=one
 record=dict(plan_id=PLAN,status=status,epoch=time.time(),elapsed_seconds=time.time()-started,
             soft_target_utc=runtime['soft_target_utc'],lease_expires_at_utc=runtime['lease_expires_at_utc'],
             runtime_sha256=sha(root/'runtime.json'),parameters_sha256=sha(root/'parameters.json'),arms=arms,
             monitoring_not_acceptance=True)
 atomic(root/'progress'/('node-%d.json'%time.time_ns()),record);atomic(root/'latest-progress.json',record)
 return record

def check_exited_workers(jobs):
 """Kit可能错误后exit0：每个已退出worker仍须有完整trained回执，不自动重启。"""
 for process,name,response in jobs:
  code=process.poll()
  if code is None:continue
  item=read(response) if response.is_file() else None
  if code!=0 or not isinstance(item,dict) or item.get('status')!='trained':
   raise RuntimeError('Exited worker has no valid trained response: '+str(dict(worker=name,returncode=code,response_exists=response.is_file(),response_status=item.get('status') if isinstance(item,dict) else None,process_log_retained=True)))

def stop(jobs):
 for p,_,_ in jobs:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 end=time.time()+60
 for p,_,_ in jobs:
  try:p.wait(timeout=max(.1,end-time.time()))
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def run(runtime_path):
 runtime=read(runtime_path);root=Path(runtime['workroot']).resolve();root.mkdir(parents=True,exist_ok=True)
 if runtime['plan_id']!=PLAN or runtime.get('direct_formal_authorized') is not True:raise ValueError('Missing direct R7P authorization')
 lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (root/'controller-state.json').exists():raise ValueError('Existing run; never duplicate four workers')
 verify(runtime['frozen_inputs']);lease=epoch(runtime['lease_expires_at_utc']);deadline=lease-60
 if time.time()>=deadline:raise TimeoutError('Actual lease endpoint reached; no renewal')
 if shutil.disk_usage(root).free<12*2**30:raise RuntimeError('Disk below12GiB new-job floor')
 started=time.time();jobs=[];backup=None;logs=[];last_record=started;released=False;token=runtime['barrier_id']
 state=dict(plan_id=PLAN,status='starting',started_epoch=started,no_preflight_or_old_budget_gate=True)
 atomic(root/'controller-state.json',state)
 def backup_once():
  nonlocal backup
  if runtime.get('backup_command') and (backup is None or backup.poll() is not None):
   log=(root/'backup-process.log').open('a');logs.append(log)
   backup=subprocess.Popen(runtime['backup_command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 def on_signal(signum,frame):raise InterruptedError('Controller received signal '+str(signum))
 signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
 try:
  for direction,fork in ARMS:
   check_exited_workers(jobs)
   verify(runtime['frozen_inputs'])
   if time.time()>=deadline or shutil.disk_usage(root).free<12*2**30:raise RuntimeError('Lease/disk prevents next worker')
   name=direction+'-'+str(fork);req=root/'requests'/(name+'.json');resp=root/'requests'/(name+'.response.json')
   if req.exists() or resp.exists():raise ValueError('Existing worker request; no automatic duplicate')
   request=dict(plan_id=PLAN,workroot=str(root),parameters_path='parameters.json',parent_selection_path='parent-selection.json',direction=direction,fork_seed=fork,
                target_learning_macros=48,action='train_evaluate',execution_mode='four_lane',barrier_id=token,barrier_timeout_s=1200,deadline_utc=iso(deadline),
                latest_user_override_path='user-execution-override.json')
   atomic(req,request);log=(root/'requests'/(name+'.log')).open('w');logs.append(log)
   command=runtime['worker_command']+['--request',str(req),'--response',str(resp)]
   env=dict(os.environ,**runtime['child_env'])
   for key in runtime.get('child_env_remove',[]):env.pop(key,None)
   p=subprocess.Popen(command,cwd=runtime['child_cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   jobs.append((p,name,resp));state.update(status='running',pids={n:p.pid for p,n,_ in jobs});atomic(root/'controller-state.json',state)
   # Stagger process creation; worker flock serializes expensive initialization/prefill.
   if len(jobs)<4:time.sleep(runtime.get('launch_stagger_seconds',2))
  progress(root,runtime,started,'running')
  while any(p.poll() is None for p,_,_ in jobs):
   verify(runtime['frozen_inputs'])
   check_exited_workers(jobs)
   if time.time()>=deadline or shutil.disk_usage(root).free<10*2**30:raise RuntimeError('Lease save margin/disk10GiB stop')
   if not released:released=release(root,token,runtime)
   if time.time()-last_record>=1800:
    progress(root,runtime,started,'running');backup_once();last_record=time.time()
   time.sleep(2)
  check_exited_workers(jobs)
  responses=[]
  for _,name,path in jobs:
   response=read(path)
   if response['status']!='trained':raise ValueError('Uncompleted training response '+name)
   a=response['training_arm'];c=a['counters']
   if a['learning_macros']!=48 or c['iterations']!=64 or c['transitions']!=65536 or c['critic_updates']!=6144 or c['actor_updates']!=3072:raise ValueError('Final fixed budget count mismatch')
   responses.append(response)
  state.update(status='training_completed',completed_epoch=time.time(),elapsed_seconds=time.time()-started,arms=responses,endpoint_diagnostics='pending_30_cases',acceptance_claim=False)
  atomic(root/'controller-state.json',state);progress(root,runtime,started,'training_completed');backup_once()
  return state
 except BaseException as exc:
  stop(jobs);state.update(status='stopped',error=str(exc),completed_epoch=time.time(),elapsed_seconds=time.time()-started)
  atomic(root/'controller-state.json',state);progress(root,runtime,started,'stopped');backup_once();raise
 finally:
  # Backup is independent and may finish after controller returns; never kill unrelated training.
  for log in logs:log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);run(p.parse_args().runtime)
